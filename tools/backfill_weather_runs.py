"""Backfill every archived Open-Meteo run of Belgium's weather groups, for intraday backtests.

Usage: uv run python tools/backfill_weather_runs.py [--start 2026-04-07] [--end 2026-11-22]
       uv run python tools/backfill_weather_runs.py --status

One Single Runs API request per run (11 points = 11 calls), newest run first, all groups
interleaved. Paced to --daily-calls per 24 h (any 24 h window stays under it; the free tier
allows 10000/day and live runs need the rest). Resumable: each run is cached as
data/archive_runs/<group>/<run>.parquet (empty if not archived); rerun to resume.
Uploads benchmark/archive_runs/<group>.parquet daily and at the end: index (run, time), hourly
openmeteo.VARIABLES (area-weighted means, like openmeteo._run), `available` (UTC).

`available`: when the run's data_run/.../meta.json landed in Open-Meteo's S3 bucket
(created_at; a few minutes after the API serves it, so conservative). The bucket keeps 3 months,
so older runs get run + the group's median latency for that run hour (ECMWF 00/12 UTC runs
land ~30 min later than 06/18), flagged `available_estimated`.
The bucket's .om files can't replace the API: they hold raw fields (no diffuse radiation or
100 m wind for some models, AIFS 6-hourly) that the API derives and interpolates.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from openpredico.data import hf, openmeteo

CACHE = Path(__file__).resolve().parents[1] / "data" / "archive_runs"
GROUPS = {  # group -> (openmeteo area, hours between archived runs, S3 model name)
    "weather_be": ("be", 6, "ecmwf_ifs"),
    "weather_be_aifs": ("be_aifs", 6, "ecmwf_aifs025_single"),
    "weather_be_d2": ("be_d2", 3, "dwd_icon_d2"),
    "weather_be_knmi": ("be_knmi", 3, "knmi_harmonie_arome_netherlands"),
}
S3 = "https://openmeteo.s3.amazonaws.com/data_run"


def runs(start, end):
    """(run, group) for every scheduled run in [start, end), newest first (usable soonest)."""
    out = [
        (r, g)
        for g, (_, every, _) in GROUPS.items()
        for r in pd.date_range(start, end, freq=f"{every}h", inclusive="left", tz="UTC")
    ]
    return sorted(out, reverse=True)


def path(group, run):
    return CACHE / group / f"{run:%Y%m%dT%H}.parquet"


def s3_available(todo):
    """S3 created_at per (group, run), fetched once each (None = not in the bucket)."""
    store = CACHE / "available.json"
    known = json.loads(store.read_text()) if store.exists() else {}
    cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=95)
    for run, group in todo:
        key = f"{group} {run:%Y-%m-%dT%H}"
        if key in known or run < cutoff:
            continue
        resp = requests.get(f"{S3}/{GROUPS[group][2]}/{run:%Y/%m/%d/%H%M}Z/meta.json", timeout=30)
        known[key] = resp.json()["created_at"] if resp.ok else None
        time.sleep(0.2)  # AWS, not the Open-Meteo API budget
    store.write_text(json.dumps(known, indent=0))
    return known


def assemble(group, known):
    parts = {}
    for f in sorted((CACHE / group).glob("*.parquet")):
        df = pd.read_parquet(f)
        if len(df):
            parts[pd.Timestamp(f.stem, tz="UTC")] = df
    df = pd.concat(parts, names=["run", "time"])
    seen = {
        pd.Timestamp(k.split()[1], tz="UTC"): pd.Timestamp(v)
        for k, v in known.items()
        if v and k.split()[0] == group
    }
    seen = pd.Series(seen, dtype="datetime64[ns, UTC]")
    lag = (seen - seen.index).groupby(seen.index.hour).median()  # per run hour
    runs = pd.DatetimeIndex(list(parts))
    avail = seen.reindex(runs)
    estimated = avail.isna()
    avail = avail.fillna(pd.Series(runs + lag.reindex(runs.hour).to_numpy(), runs))
    run = df.index.get_level_values("run")
    df["available"] = avail.reindex(run).set_axis(df.index)
    df["available_estimated"] = estimated.reindex(run).set_axis(df.index)
    return df


def upload(known):
    for group in GROUPS:
        if (CACHE / group).exists() and any((CACHE / group).glob("*.parquet")):
            df = assemble(group, known)
            path_in_repo = f"archive_runs/{group}.parquet"
            hf.upload(df, "benchmark", path_in_repo, f"Open-Meteo runs: {path_in_repo}")
            print(f"uploaded {group}: {df.index.get_level_values('run').nunique()} runs")


def main():
    global CACHE
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-04-07")
    p.add_argument("--end", default=f"{pd.Timestamp.now('UTC'):%Y-%m-%d}")
    p.add_argument("--daily-calls", type=int, default=8000)
    p.add_argument("--cache", type=Path, help="default data/archive_runs; outlive a worktree")
    p.add_argument("--status", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--hours", type=float, help="stop (and upload) after this many hours")
    a = p.parse_args()
    CACHE = a.cache or CACHE
    todo = runs(a.start, a.end)
    left = [(r, g) for r, g in todo if not path(g, r).exists()]
    print(f"{len(todo) - len(left)}/{len(todo)} runs cached, {len(left)} left")
    if a.status:
        return
    known = s3_available(todo)
    pause = 86400 / a.daily_calls  # seconds per call
    last_upload = time.time()
    began = time.time()
    for i, (run, group) in enumerate(left):
        if a.hours and time.time() - began > a.hours * 3600:
            break  # a scheduled rerun resumes from the cache
        area = GROUPS[group][0]
        started = time.time()
        while True:
            try:
                df = openmeteo.fetch_run(run, area)
                break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    df = pd.DataFrame(columns=openmeteo.VARIABLES)  # not archived
                    break
                print(f"{group} {run}: {e}, retrying in 30 min", flush=True)
            except requests.RequestException as e:
                print(f"{group} {run}: {e}, retrying in 30 min", flush=True)
            time.sleep(1800)
        path(group, run).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path(group, run))
        eta = pd.Timedelta(seconds=(len(left) - i - 1) * pause * len(openmeteo.POINTS[area]))
        print(f"{group} {run:%Y-%m-%d %H}Z: {len(df)} h, {len(left) - i - 1} left, ETA {eta}")
        if not a.no_upload and time.time() - last_upload > 86400:
            upload(known)
            last_upload = time.time()
        time.sleep(max(0, pause * len(openmeteo.POINTS[area]) - (time.time() - started)))
    if not a.no_upload:
        upload(known)


if __name__ == "__main__":
    main()
