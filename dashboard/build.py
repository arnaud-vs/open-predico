"""Build the dashboard's JSON from the public HF datasets (`benchmark`, `live`), then optionally
push the static Space. The page only draws: every score here uses evaluation/metrics.py.

    python dashboard/build.py                 # writes dashboard/data/
    python dashboard/build.py --upload        # and pushes index.html, app.js, data/ to the Space
    python dashboard/build.py --upload --data-only  # only data/ (Modal: its page copy is stale)
    python dashboard/build.py --demo-live     # local preview only: backtest days in the live format
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils.sha import git_hash

from openpredico.config import load_targets
from openpredico.data import openmeteo
from openpredico.data.catalog import CALENDAR
from openpredico.data.groups import load_groups
from openpredico.data.hf import ORG, download
from openpredico.evaluation.experiment import TARGET_COLUMNS, load_benchmark
from openpredico.evaluation.metrics import daily_scores

HERE = Path(__file__).parent
PREFIX = "predictions/backtest/"
BASELINE = "slot_climatology_28d"
# Dashboard markets (app.js MARKETS) and the short name of each target inside the page.
MARKETS = {
    "es-mfrr": {"spain_mfrr_energy": "energy", "spain_mfrr_price": "price"},
    "de-wind": {"germany_wind_50hertz_ca": "wind"},
    "de-solar": {"germany_solar_50hertz_ca": "solar"},
    "be-solar": {"belgium_solar_opendata_elia": "solar_be"},
}
CONFIG = "t0gamma_cov_ctx90"  # = live.submit.CONFIG (not imported: it loads the t0 model code)
LIVE_CONFIG = {  # market -> backtest config of its live model (targets.toml live_config)
    m: load_targets()[next(iter(shorts))].get("live_config", CONFIG)
    for m, shorts in MARKETS.items()
}
SHORT = {target: short for m in MARKETS.values() for target, short in m.items()}
UNITS = {SHORT[n]: t["unit"] for n, t in load_targets().items() if n in SHORT}  # shown
# Input features tab outside Spain (Spain shows the ESIOS catalog): entsoe source descriptions.
ENTSOE = {"A75": "actual generation", "A69": "day-ahead forecast", "A65": "day-ahead load forecast"}
PSR = {"B16": "solar", "B18": "wind offshore", "B19": "wind onshore"}
TYPES = {  # config name prefix -> model type; anything else is ML
    "slot_climatology": "baseline",
    "chronos": "foundation model",
    "timesfm": "foundation model",
    "timestral": "foundation model",
    "t0": "foundation model",
    "ciproto": "foundation model",
    "timeh2o": "foundation model",
    "elia": "TSO forecast",
    "entsoe": "TSO forecast",
}
STEP = pd.Timedelta("15min")


def live_config(target: str) -> str:
    return load_targets()[target].get("live_config", CONFIG)


LIVE_COLUMNS = (
    *("q10", "q50", "q90", "s10", "s50", "s90", "obs", "b10", "b50", "b90"),
    *("t10", "t50", "t90"),  # the TSO's own forecast, where the target has one
)


def arr(values) -> list:
    """Floats rounded to 0.1 with NaN as null: the JSON stays small."""
    a = np.round(np.asarray(values, dtype=float), 1)
    return [None if np.isnan(v) else v for v in a.tolist()]


def secs(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def write(out: Path, name: str, obj) -> None:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def backtest(out: Path) -> dict:
    files = [
        f
        for f in HfApi().list_repo_files(f"{ORG}/benchmark", repo_type="dataset")
        if f.startswith(PREFIX) and f.endswith(".parquet")
    ]
    days, profiles, grid = {}, {}, None
    for f in sorted(files):
        config, target = f.removeprefix(PREFIX).removesuffix(".parquet").split("/")
        if target not in SHORT:
            continue  # no dashboard market yet (Belgium solar)
        t, tz = SHORT[target], load_targets()[target]["timezone"]
        df = download("benchmark", f)
        grid = df.index if grid is None else grid
        df = df.reindex(grid)
        write(out, f"steps/{t}/{config}.json", {c: arr(df[c]) for c in ("q10", "q50", "q90")})
        d = daily_scores(df[["q10", "q50", "q90"]], df.obs, tz=tz)
        days.setdefault(t, {})[config] = d[["rmse", "winkler", "coverage"]]
        local = df.index.tz_convert(tz)
        ae = (df.obs - df.q50).abs()
        profiles.setdefault(t, {})[config] = {
            "hour": arr(ae.groupby(local.hour).mean().reindex(range(24))),
            "weekday": arr(ae.groupby(local.dayofweek).mean().reindex(range(7))),
        }
        print(f"backtest {config}/{t}", flush=True)
    out_days = {}
    for t, by_config in days.items():
        index = sorted(set().union(*(d.index for d in by_config.values())))
        out_days[t] = {
            "days": [d.strftime("%Y-%m-%d") for d in index],
            "configs": {
                c: {m: arr(d[m].reindex(index)) for m in ("rmse", "winkler", "coverage")}
                for c, d in by_config.items()
            },
        }
    write(out, "days.json", out_days)
    write(out, "profiles.json", profiles)
    configs = sorted({c for d in days.values() for c in d})
    return {
        "start": secs(grid[0]),
        "n": len(grid),
        "configs": [
            {
                "name": c,
                "type": next((v for k, v in TYPES.items() if c.startswith(k)), "ML"),
                # context days from the config name (`ctx90`, `climatology_28d`), else None
                "context": int(m[1])
                if (m := re.search(r"ctx(\d+)|_(\d+)d\b", c)) and m[1]
                else int(m[2])
                if m
                else None,
            }
            for c in configs
        ],
    }


def target_series(out: Path) -> None:
    """Per market, its targets' truth from the benchmark: target/<market>.json."""
    groups, targets = load_groups(), load_targets()
    for market, shorts in MARKETS.items():
        cols = {TARGET_COLUMNS[t]: s for t, s in shorts.items()}
        need = {g for t in shorts for g in targets[t]["groups"] if TARGET_COLUMNS[t] in groups[g]}
        df = pd.concat(load_benchmark(need).values(), axis=1)[list(cols)].asfreq(STEP)
        series = {s: arr(df[c]) for c, s in cols.items()}
        write(out, f"target/{market}.json", {"start": secs(df.index[0]), **series})


def describe(source: str, names: dict) -> str:
    prefix, key = source.split(":")
    if prefix == "entsoe":
        document, _, psrs = key.partition("_")
        what = " + ".join(PSR[p] for p in psrs.split("+")) if psrs else "total"
        return f"ENTSO-E {document} {ENTSOE[document]} ({what}), 50Hertz control area"
    if prefix == "openmeteo":
        area, variable = key.split(".")
        how = "capacity-weighted mean" if area.startswith("be") else "mean"
        return f"Open-Meteo {variable}, {run_label(area)} run of D-1, {how} over the {area} points"
    if prefix == "elia":
        return f"Elia Open Data {key.split('.')[1]}, Belgium total (ods032, then ods209/ods210)"
    if prefix == "calendar":
        return CALENDAR[key]
    return names.get((prefix, key), "")


def catalog(out: Path) -> None:
    """ESIOS catalog (Spain's tab) and, per market, its feature columns from groups.toml."""
    cat = download("benchmark", "catalog/inputs.parquet")
    write(out, "catalog.json", json.loads(cat.to_json(orient="records")))
    names = {(r.source, str(r.id)): r.name for r in cat.itertuples()}
    groups, targets = load_groups(), load_targets()
    features = {
        market: [
            {"group": g, "column": c, **d, "description": describe(d["source"], names)}
            for g in dict.fromkeys(g for t in shorts for g in targets[t]["groups"])
            for c, d in groups[g].items()
        ]
        for market, shorts in MARKETS.items()
    }
    write(out, "features.json", features)


# Region and country outlines for the weather-points map (geoBoundaries, pinned release).
GB = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen"
REGIONS = {
    "de": {
        "url": f"{GB}/DEU/ADM1/geoBoundaries-DEU-ADM1_simplified.geojson",
        "country": f"{GB}/DEU/ADM0/geoBoundaries-DEU-ADM0_simplified.geojson",
        "keep": {"Berlin", "Brandenburg", "Hamburg", "Mecklenburg-Vorpommern", "Sachsen"}
        | {"Sachsen-Anhalt", "Thüringen"},  # the 50Hertz control area
        "label": "50Hertz control area (states)",
        "credit": "geoBoundaries (BKG, dl-de/by-2-0)",
    },
    "be": {
        "url": f"{GB}/BEL/ADM2/geoBoundaries-BEL-ADM2_simplified.geojson",
        "country": f"{GB}/BEL/ADM0/geoBoundaries-BEL-ADM0_simplified.geojson",
        "keep": None,  # every province
        "label": "Belgian provinces",
        "credit": "geoBoundaries (ODbL 1.0)",
    },
}
RUNS = {"ecmwf_ifs": "ECMWF IFS"} | {
    "icon_d2": "ICON-D2",
    "knmi_harmonie_arome_netherlands": "KNMI HARMONIE-AROME",
    "ecmwf_aifs025_single": "ECMWF AIFS",
}


def run_hour(area: str) -> int:
    """UTC hour of the run of D-1 (or D-2 for a positive shift) that an area reads."""
    shift = openmeteo.RUN_SHIFT.get(area, pd.Timedelta(0))
    return (pd.Timedelta(days=1) - shift).seconds // 3600 % 24


def run_label(area: str) -> str:
    """e.g. "ECMWF IFS 00 UTC": the model and run hour an Open-Meteo area reads."""
    return f"{RUNS[openmeteo.AREA_MODEL.get(area, openmeteo.MODEL)]} {run_hour(area):02d} UTC"


def weather_areas(shorts: dict) -> set[str]:
    """The Open-Meteo areas the targets' feature groups read."""
    groups, targets = load_groups(), load_targets()
    return {
        c["source"].split(":")[1].split(".")[0]
        for t in shorts
        for g in targets[t]["groups"]
        for c in groups[g].values()
        if c["source"].startswith("openmeteo:")
    }


def _round(coords):
    return (
        [_round(c) for c in coords]
        if isinstance(coords[0], list)
        else [round(x, 3) for x in coords]
    )


def _outlines(url: str, keep: set | None = None) -> dict:
    """A geoBoundaries layer as a small FeatureCollection: name and rounded coordinates."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": f["properties"]["shapeName"]},
                "geometry": f["geometry"] | {"coordinates": _round(f["geometry"]["coordinates"])},
            }
            for f in requests.get(url, timeout=120).json()["features"]
            if keep is None or f["properties"]["shapeName"] in keep
        ],
    }


def geo(out: Path) -> None:
    """Per market with Open-Meteo inputs: region and country outlines, the weather points (weight
    share)."""
    data = {}
    for market, shorts in MARKETS.items():
        areas = weather_areas(shorts)
        if not areas:
            continue
        region = REGIONS["be" if market.startswith("be") else "de"]
        sets = {}  # same points -> one set, listing every run read on them
        for a in sorted(areas):
            points, w = openmeteo.POINTS[a], openmeteo.WEIGHTS.get(a)
            w = w or [1] * len(points)
            run = run_label(a)
            shares = [[p[0], p[1], round(x / sum(w), 4)] for p, x in zip(points, w, strict=True)]
            entry = sets.setdefault(
                tuple(points),
                {"areas": [], "runs": [], "weighted": a in openmeteo.WEIGHTS, "points": shares},
            )
            entry["areas"].append(a)
            entry["runs"].append(run)
        data[market] = {
            "regions": _outlines(region["url"], region["keep"]),
            "country": _outlines(region["country"]),
            "label": region["label"],
            "credit": region["credit"],
            "sets": list(sets.values()),
        }
    write(out, "geo.json", data)


# The live submit cron runs at :15 in the hour before the gate (live/modal_app.py).
SUBMIT_BEFORE_GATE = pd.Timedelta("45min")


def weather(out: Path, now: pd.Timestamp) -> None:
    """Per market with Open-Meteo inputs, for today's D+1 session (D-1 = today, local): each
    weather model's archived runs (cycle, latency), the run we read, and origin, submit and gate
    as UTC seconds. Today's local day bounds too, for the hourly intraday gates."""
    data = {}
    for market, shorts in MARKETS.items():
        areas = weather_areas(shorts)
        if not areas:
            continue
        t = load_targets()[next(iter(shorts))]
        today = now.tz_convert(t["timezone"]).normalize()
        origin, gate = (today + pd.Timedelta(f"{t[k]}:00") for k in ("origin", "gate"))
        models = {}
        for a in sorted(areas):
            model = openmeteo.AREA_MODEL.get(a, openmeteo.MODEL)
            used = openmeteo.run_time(today + pd.DateOffset(days=1), a)  # for target day D+1
            models.setdefault(
                model,
                {
                    "name": RUNS[model],
                    "cycle": openmeteo.CYCLE[model] * 3600,
                    "latency": int(openmeteo.LATENCY[model].total_seconds()),
                    "used": secs(used),
                },
            )
        data[market] = {
            "day": f"{today:%Y-%m-%d}",
            "origin": secs(origin),
            "submit": secs(gate - SUBMIT_BEFORE_GATE),
            "gate": secs(gate),
            "day_start": secs(today),
            "day_end": secs(today + pd.DateOffset(days=1)),
            "models": list(models.values()),
        }
    write(out, "weather.json", data)


def live_frames(demo: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Published live forecasts per short target name, and the runs table (None if absent)."""
    if not demo:
        try:
            runs = download("live", "runs.parquet")
        except Exception:  # dataset not published yet
            return {}, None
        frames = {
            SHORT[t]: download("live", f"forecasts/{t}.parquet") for t in runs.target.unique()
        }
        return frames, runs
    # Demo: the live config's last 21 backtest days, two of them without a Predico submission.
    frames, rows = {}, []
    for target, t in SHORT.items():
        fc = download("benchmark", f"{PREFIX}{live_config(target)}/{target}.parquet").iloc[
            -21 * 96 :
        ]
        base = download("benchmark", f"{PREFIX}{BASELINE}/{target}.parquet").reindex(fc.index)
        fc[["b10", "b50", "b90"]] = base[["q10", "q50", "q90"]].to_numpy()
        day = fc.index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")
        fc = fc[day != sorted(set(day))[6]]  # one missed day
        day = fc.index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")
        registered = ~np.isin(day, sorted(set(day))[-3:-1])
        for q in ("10", "50", "90"):
            fc[f"s{q}"] = fc[f"q{q}"].where(registered)
        fc["model"], fc["env"] = live_config(target), "demo"
        fc.loc[day == sorted(set(day))[2], ["q10", "q50", "q90"]] = np.nan  # registered, no run
        frames[t] = fc
        cols = [c for g in load_targets()[target]["groups"] for c in load_groups()[g]]
        rows.append(
            {
                "target": target,
                "run": fc.index[-1],
                "model": live_config(target),
                "features": ",".join(c for c in cols if c != TARGET_COLUMNS[target]),
            }
        )
    return frames, pd.DataFrame(rows)


def official(target: str, demo: bool, days: list[str]) -> list[dict]:
    """Predico's scores of our submissions (our rank only), as records; [] before the first."""
    if demo:  # local preview only: made-up ranks on the demo days, never uploaded
        rng = np.random.default_rng(0)
        return [
            {"day": d, "variable": v, "metric": m, "value": float(rng.uniform(50, 300)),
             "rank": int(rng.integers(1, 9)), "participants": 8}
            for d in days for v, m in (("q50", "rmse"), ("q10", "pinball"), ("q90", "pinball"))
        ]  # fmt: skip
    try:
        df = download("live", f"scores/{target}.parquet")
    except Exception:  # not published yet
        return []
    return json.loads(df.round(3).to_json(orient="records"))


def env_days(g: pd.DataFrame) -> dict[str, str]:
    """Environment per Madrid target day: days before the first production run's day were
    playground (qualifying; hindcast runs only fill in the forecast for those days)."""
    day = pd.Series(g.index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d"), index=g.index)
    first = min(day[g.env == "production"], default="9999")
    return {d: "playground" if d < first else "production" for d in day.unique()}


def live(out: Path, demo: bool) -> None:
    frames, runs = live_frames(demo)
    if runs is None:
        write(out, "live.json", {"available": False})
        return
    data = {"available": True, "demo": demo, "targets": {}}
    for t, df in frames.items():
        grid = pd.date_range(df.index.min(), df.index.max(), freq=STEP)
        g = df.reindex(grid).reindex(columns=[*LIVE_COLUMNS, "env"])
        target_runs = runs[runs.target.map(SHORT) == t].sort_values("run")
        last = target_runs.iloc[-1]
        data["targets"][t] = {
            "start": secs(grid[0]),
            # q*: continuous forecast at the gate; s*: what Predico registered; b*: climatology
            **{c: arr(g[c]) for c in LIVE_COLUMNS},
            "model": last.model,
            "env": str(df.env.dropna().iloc[-1]) if "env" in df else "playground",
            # per target day: playground days are marked on the page (never scored by Predico)
            "env_days": env_days(g),
            "runs": len(target_runs),
            "last_run": str(last.run),
            "features": [f for f in str(last.features).split(",") if f],
            "official": official(
                last.target,
                demo,
                sorted(set(df.index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")))[:-1],
            ),
        }
    write(out, "live.json", data)


SPACE = f"{ORG}/dashboard"
UPLOAD = ["index.html", "app.js", "README.md", "data/**"]


def unchanged(meta: dict, data_only: bool = False) -> bool:
    """True if the Space already serves every file we would upload, and the same meta.json
    apart from `generated`: then an upload would only bump the build time."""
    api = HfApi()
    try:
        old = json.loads(
            Path(api.hf_hub_download(SPACE, "data/meta.json", repo_type="space")).read_text()
        )
    except Exception:  # first upload
        return False
    if {**old, "generated": None} != {**json.loads(json.dumps(meta)), "generated": None}:
        return False
    remote = {
        f.path: f
        for f in api.list_repo_tree(SPACE, repo_type="space", recursive=True)
        if isinstance(f, RepoFile)
    }
    page = [] if data_only else [HERE / n for n in UPLOAD[:3]]
    local = [*page, *(HERE / "data").rglob("*.json")]
    for path in local:
        rel, data = path.relative_to(HERE).as_posix(), path.read_bytes()
        if rel == "data/meta.json":
            continue
        f = remote.get(rel)
        if f is None or (
            hashlib.sha256(data).hexdigest() != f.lfs.sha256
            if f.lfs
            else git_hash(data) != f.blob_id
        ):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "data")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--data-only", action="store_true")  # leave the Space's page files as they are
    p.add_argument("--demo-live", action="store_true")
    a = p.parse_args()
    if a.upload and a.demo_live:
        raise SystemExit("--demo-live is for local previews; never upload it")
    meta = backtest(a.out)
    target_series(a.out)
    catalog(a.out)
    geo(a.out)
    weather(a.out, pd.Timestamp.now(tz="UTC"))
    live(a.out, a.demo_live)
    meta |= {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
        "baseline": BASELINE,
        "live_config": LIVE_CONFIG,
        "units": UNITS,
        "short": SHORT,
    }
    if a.upload and unchanged(meta, a.data_only):
        print("dashboard unchanged apart from the build time: no Space commit")
        return
    write(a.out, "meta.json", meta)
    if a.upload:
        HfApi().upload_folder(
            repo_id=SPACE,
            repo_type="space",
            folder_path=HERE,
            allow_patterns=["data/**"] if a.data_only else UPLOAD,
            delete_patterns=["app.py", "requirements.txt"],  # the old stlite app
            commit_message=f"Dashboard data {meta['generated']}",
        )


if __name__ == "__main__":
    main()
