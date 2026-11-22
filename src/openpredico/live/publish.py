"""After gate closure: publish what Predico scored, with actuals, to the public HF `live` dataset.

For each target day whose gate has passed, Predico sliced the latest run submitted before that
gate, so that run's rows are what we publish. Days with no run before the gate stay absent (the
dashboard shows them as missed).

Incremental: what is already published is the cursor. Only the days from the first unpublished
one, and at least the last WINDOW_DAYS (truth revisions, late Predico scores), are recomputed;
earlier days are kept as published. Only the runs Predico sliced for those days, plus runs not
yet in `runs.parquet`, are downloaded from `live-private`. `--full` rebuilds everything.

    forecasts/<target>.parquet  q10/q50/q90, model, env, run: our continuous forecast as sliced
                                at the gate; s10/s50/s90: what Predico registered as our session
                                submissions, production or else playground (also days from
                                before the HF record); obs; b10/b50/b90
                                (slot climatology from the same run's inputs, the baseline);
                                t10/t50/t90: the TSO's own forecast (targets.toml `tso`)
    scores/<target>.parquet     Predico's official scores of our submissions per day, variable
                                and metric, with our rank and the number of participants
    runs.parquet                one row per run and target: model and input features

Usage: python -m openpredico.live.publish [--full]
"""

import sys
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi

from openpredico.config import live_targets
from openpredico.data import hf
from openpredico.data.groups import build, fetch_source, load_groups
from openpredico.evaluation.backtest import PUBLICATION_LAG
from openpredico.evaluation.experiment import TARGET_COLUMNS, tso_forecast
from openpredico.live.predico import Predico
from openpredico.live.submit import CONFIG
from openpredico.models.baselines import slot_climatology

# ponytail: ENTSO-E revises truth for up to 35 days (PLAN.md); a week catches most of it
WINDOW_DAYS = 7


def gate_closes(index: pd.DatetimeIndex, tz: str, gate: str) -> pd.DatetimeIndex:
    """Gate closure for each timestamp's target day: D-1 at `gate`, local time."""
    d1 = (index.tz_convert(tz).normalize() - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    return pd.DatetimeIndex(pd.to_datetime(d1 + " " + gate)).tz_localize(tz)


def scored(runs: dict[pd.Timestamp, pd.DataFrame], tz: str, gate: str, now: pd.Timestamp):
    """Per quarter-hour, the row of the latest run before its day's gate, if that gate passed."""
    df = pd.concat([fc.assign(run=run) for run, fc in runs.items()])
    closes = gate_closes(df.index, tz, gate)
    df = df[(df.run.to_numpy() < closes) & (closes <= now)].sort_values("run", kind="stable")
    return df[~df.index.duplicated(keep="last")].sort_index()


def window_start(published: pd.DatetimeIndex | None, tz: str, now: pd.Timestamp):
    """Local midnight of the first day to recompute: the day after the last published one, or
    WINDOW_DAYS ago if earlier. None (nothing published yet) recomputes every day."""
    if published is None or published.empty:
        return None
    after_last = published.max().tz_convert(tz).normalize() + pd.DateOffset(days=1)
    return min(after_last, now.tz_convert(tz).normalize() - pd.DateOffset(days=WINDOW_DAYS))


def runs_to_read(stamps: list[pd.Timestamp], start, tz: str, gate: str, now: pd.Timestamp):
    """Runs Predico sliced for the gated days from `start` on: per day, the latest run before
    its gate. A run covers D+1 to D+10, so this also finds a days-old fallback run."""
    first = start if start is not None else min(stamps).tz_convert(tz).normalize()
    last = now.tz_convert(tz).normalize() + pd.DateOffset(days=1)
    days = pd.date_range(first, last, freq=pd.DateOffset(days=1))
    closes = gate_closes(days.tz_convert("UTC"), tz, gate)
    picked = {max((s for s in stamps if s < c), default=None) for c in closes[closes <= now]}
    # ponytail: assumes that run has every row of the day; a partial run would need the one before
    return picked - {None}


def _published(path: str) -> pd.DataFrame | None:
    try:
        return hf.download("live", path)
    except Exception:  # not published yet
        return None


def main(full: bool = False) -> None:
    now = pd.Timestamp.now(tz="UTC")
    files = HfApi().list_repo_files(f"{hf.ORG}/live-private", repo_type="dataset")
    client, playground = Predico(), Predico("PREDICO_PLAYGROUND_URL")
    HfApi().create_repo(f"{hf.ORG}/live", repo_type="dataset", private=False, exist_ok=True)
    old_runs = None if full else _published("runs.parquet")
    run_rows = []
    for name, t in live_targets().items():
        tz = t["timezone"]
        prev = None if full else _published(f"forecasts/{name}.parquet")
        if prev is not None:
            prev.index = prev.index.as_unit("us")
        start = window_start(None if prev is None else prev.index, tz, now)
        paths = {
            pd.Timestamp(
                datetime.strptime(f.rsplit("/", 1)[1].removesuffix(".parquet"), "%Y-%m-%dT%H%MZ"),
                tz="UTC",
            ): f
            for f in files
            if f.startswith(f"forecasts/{name}/")
        }
        known = set() if old_runs is None else set(old_runs.run[old_runs.target == name])
        need = runs_to_read(list(paths), start, tz, t["gate"], now) if paths else set()
        runs, rows = {}, []
        for run in sorted(need | (set(paths) - known)):
            f = paths[run]
            fc = hf.download("live-private", f)
            fc.index = fc.index.as_unit("us")  # wind-pair runs are ns; mixed units join to object
            fc = fc.assign(model=fc.get("model", CONFIG))  # pre-tag runs: CONFIG
            inputs = hf.download("live-private", f.replace("forecasts/", "inputs/", 1))
            if run not in known:
                features = [c for c in inputs.columns if c != TARGET_COLUMNS[name]]
                rows.append(
                    {
                        "target": name,
                        "run": run,
                        "model": fc.model.iloc[0],
                        "features": ",".join(features),
                    }
                )
            if run in need:
                y = inputs[TARGET_COLUMNS[name]]
                days = pd.DatetimeIndex(fc.index.tz_convert(tz).normalize().unique())
                base = slot_climatology(y[y.index < run - PUBLICATION_LAG], days, tz=tz)
                fc[["b10", "b50", "b90"]] = base.reindex(fc.index).to_numpy()
                runs[run] = fc
        print(f"{name}: read {len(runs)} sliced and {len(rows)} new of {len(paths)} runs")
        out = scored(runs, tz, t["gate"], now) if runs else pd.DataFrame()
        if start is not None and not out.empty:
            out = out[out.index >= start]
        since = None if start is None else f"{start:%Y-%m-%d}"
        challenges = client.our_challenges(t["predico_resource"], since)
        # production first; playground (qualifying, never scored) fills the days before it
        sub = client.submitted_forecasts(challenges).combine_first(
            playground.submitted_forecasts(playground.our_challenges(t["predico_resource"], since))
        )
        sub = sub[gate_closes(sub.index, tz, t["gate"]) <= now]  # never an open day
        sub = sub.rename(columns=lambda c: "s" + c[1:])  # q50 -> s50
        out = sub if out.empty else out.join(sub, how="outer")
        run_rows += rows
        if out.empty:
            continue  # nothing newly gated: what is published stays
        begin = (out.index.min() if start is None else start).tz_convert("UTC")
        truth = fetch_source(t["truth"], *(x.tz_localize(None).isoformat() for x in (begin, now)))
        measured = out.index < now - PUBLICATION_LAG
        obs = truth.reindex(out.index)
        if t["unit"] == "MWh":
            obs[measured] = obs[measured].fillna(0)  # missing = no activation (PLAN.md)
        out["obs"] = obs
        if "tso" in t:  # the TSO's own forecast (published after the gate): t10/t50/t90
            cols = {c for v in t["tso"].values() for c in v.split("+")}
            # through the last gated day, which can still be ahead of now
            stop = (out.index.max() + pd.Timedelta("15min")).tz_localize(None).isoformat()
            frames = {
                g: build(g, begin.tz_localize(None).isoformat(), stop, tz, lenient=True)
                for g, columns in load_groups().items()
                if cols & columns.keys()
            }
            try:
                tso = tso_forecast(t, frames).reindex(out.index)
                out[["t10", "t50", "t90"]] = tso.to_numpy()
            except KeyError:  # a TSO column failed to build: publish without it
                print(f"{name}: TSO forecast left out")
        if prev is not None:  # days before the window stay as published
            out = pd.concat([prev[prev.index < start], out]).sort_index()
        hf.upload(out, "live", f"forecasts/{name}.parquet", f"Publish {name} {now:%Y-%m-%d}")
        # Our own official scores and rank only; never other forecasters' results.
        scores = client.scores(challenges)
        old = None if prev is None else _published(f"scores/{name}.parquet")
        if old is not None:
            scores = pd.concat([old[old.day < since], scores], ignore_index=True)
        hf.upload(scores, "live", f"scores/{name}.parquet", f"Scores {name}")
        days = out.index.tz_convert(tz).normalize()
        print(f"{name}: {days.nunique()} days, {days[out.s50.notna()].nunique()} registered")
    if run_rows:  # new runs since the last publish
        runs_df = pd.concat([old_runs, pd.DataFrame(run_rows)], ignore_index=True)
        hf.upload(runs_df, "live", "runs.parquet", f"Runs {now:%Y-%m-%d}")


if __name__ == "__main__":
    main(full="--full" in sys.argv[1:])
