"""After gate closure: publish what Predico scored, with actuals, to the public HF `live` dataset.

For each target day whose gate has passed, Predico sliced the latest run submitted before that
gate, so that run's rows are what we publish. Days with no run before the gate stay absent (the
dashboard shows them as missed). Rebuilt from `live-private` on every run, so it is idempotent.

    forecasts/<target>.parquet  q10/q50/q90, model, env, run: our continuous forecast as sliced
                                at the gate; s10/s50/s90: what Predico registered as our session
                                submissions (also days from before the HF record); obs; b10/b50/b90
                                (slot climatology from the same run's inputs, the baseline)
    scores/<target>.parquet     Predico's official scores of our submissions per day, variable
                                and metric, with our rank and the number of participants
    runs.parquet                one row per run and target: model and input features

Usage: python -m openpredico.live.publish
"""

from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi

from openpredico.config import live_targets
from openpredico.data import hf
from openpredico.data.groups import fetch_source
from openpredico.evaluation.backtest import PUBLICATION_LAG
from openpredico.evaluation.experiment import TARGET_COLUMNS
from openpredico.live.predico import Predico
from openpredico.live.submit import CONFIG
from openpredico.models.baselines import slot_climatology


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


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    files = HfApi().list_repo_files(f"{hf.ORG}/live-private", repo_type="dataset")
    client, run_rows = Predico(), []
    HfApi().create_repo(f"{hf.ORG}/live", repo_type="dataset", private=False, exist_ok=True)
    for name, t in live_targets().items():
        runs, rows = {}, []
        # ponytail: re-downloads every run daily (~2 files/run); keep a cursor if it gets slow
        for f in sorted(f for f in files if f.startswith(f"forecasts/{name}/")):
            stamp = f.rsplit("/", 1)[1].removesuffix(".parquet")
            run = pd.Timestamp(datetime.strptime(stamp, "%Y-%m-%dT%H%MZ"), tz="UTC")
            fc = hf.download("live-private", f)
            fc.index = fc.index.as_unit("us")  # wind-pair runs are ns; mixed units join to object
            inputs = hf.download("live-private", f.replace("forecasts/", "inputs/", 1))
            y = inputs[TARGET_COLUMNS[name]]
            days = pd.DatetimeIndex(fc.index.tz_convert(t["timezone"]).normalize().unique())
            base = slot_climatology(y[y.index < run - PUBLICATION_LAG], days, tz=t["timezone"])
            fc[["b10", "b50", "b90"]] = base.reindex(fc.index).to_numpy()
            runs[run] = fc = fc.assign(model=fc.get("model", CONFIG))  # pre-tag runs: CONFIG
            features = [c for c in inputs.columns if c != TARGET_COLUMNS[name]]
            rows.append(
                {
                    "target": name,
                    "run": run,
                    "model": fc.model.iloc[0],
                    "features": ",".join(features),
                }
            )
        out = scored(runs, t["timezone"], t["gate"], now) if runs else pd.DataFrame()
        challenges = client.our_challenges(t["predico_resource"])
        sub = client.submitted_forecasts(challenges)
        sub = sub[gate_closes(sub.index, t["timezone"], t["gate"]) <= now]  # never an open day
        sub = sub.rename(columns=lambda c: "s" + c[1:])  # q50 -> s50
        out = sub if out.empty else out.join(sub, how="outer")
        if out.empty:
            continue  # nothing gated yet
        start, end = (x.tz_localize(None).isoformat() for x in (out.index.min(), now))
        truth = fetch_source(t["truth"], start, end)
        known = out.index < now - PUBLICATION_LAG
        obs = truth.reindex(out.index)
        if t["unit"] == "MWh":
            obs[known] = obs[known].fillna(0)  # missing quarter-hour = no activation (PLAN.md)
        out["obs"] = obs
        hf.upload(out, "live", f"forecasts/{name}.parquet", f"Publish {name} {now:%Y-%m-%d}")
        # Our own official scores and rank only; never other forecasters' results.
        hf.upload(client.scores(challenges), "live", f"scores/{name}.parquet", f"Scores {name}")
        run_rows += rows
        days = out.index.tz_convert(t["timezone"]).normalize()
        print(f"{name}: {days.nunique()} days, {days[out.s50.notna()].nunique()} registered")
    if run_rows:
        hf.upload(pd.DataFrame(run_rows), "live", "runs.parquet", f"Runs {now:%Y-%m-%d}")


if __name__ == "__main__":
    main()
