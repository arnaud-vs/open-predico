"""After gate closure: publish what Predico scored, with actuals, to the public HF `live` dataset.

For each target day whose gate has passed, Predico sliced the latest run submitted before that
gate, so that run's rows are what we publish. Days with no run before the gate stay absent (the
dashboard shows them as missed). Rebuilt from `live-private` on every run, so it is idempotent.

    forecasts/<target>.parquet  q10/q50/q90, model, env, run, obs, b10/b50/b90 (slot climatology
                                from the same run's inputs, the baseline for normalised scores),
                                submitted (Predico registered a session submission that day)
    runs.parquet                one row per run and target: model and input features

Usage: python -m openpredico.live.publish
"""

from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi

from openpredico.config import load_targets
from openpredico.data import esios, hf
from openpredico.evaluation.backtest import PUBLICATION_LAG
from openpredico.evaluation.experiment import TARGET_COLUMNS
from openpredico.live.predico import Predico
from openpredico.live.submit import CONFIG
from openpredico.models.baselines import slot_climatology


def scored(runs: dict[pd.Timestamp, pd.DataFrame], tz: str, gate: str, now: pd.Timestamp):
    """Per quarter-hour, the row of the latest run before its day's gate, if that gate passed."""
    df = pd.concat([fc.assign(run=run) for run, fc in runs.items()])
    d1 = (df.index.tz_convert(tz).normalize() - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    closes = pd.DatetimeIndex(pd.to_datetime(d1 + " " + gate)).tz_localize(tz)
    df = df[(df.run.to_numpy() < closes) & (closes <= now)].sort_values("run", kind="stable")
    return df[~df.index.duplicated(keep="last")].sort_index()


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    files = HfApi().list_repo_files(f"{hf.ORG}/live-private", repo_type="dataset")
    client, run_rows = Predico(), []
    HfApi().create_repo(f"{hf.ORG}/live", repo_type="dataset", private=False, exist_ok=True)
    for name, t in load_targets().items():
        runs, rows = {}, []
        # ponytail: re-downloads every run daily (~2 files/run); keep a cursor if it gets slow
        for f in sorted(f for f in files if f.startswith(f"forecasts/{name}/")):
            stamp = f.rsplit("/", 1)[1].removesuffix(".parquet")
            run = pd.Timestamp(datetime.strptime(stamp, "%Y-%m-%dT%H%MZ"), tz="UTC")
            fc = hf.download("live-private", f)
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
        if out.empty:
            continue  # no gate has passed since our first run
        indicator = int(t["truth"].removeprefix("esios:"))
        start, end = (x.tz_localize(None).isoformat() for x in (out.index.min(), now))
        truth = esios.fetch(indicator, start, end)
        known = out.index < now - PUBLICATION_LAG
        obs = truth.reindex(out.index)
        if t["unit"] == "MWh":
            obs[known] = obs[known].fillna(0)  # missing quarter-hour = no activation (PLAN.md)
        out["obs"] = obs
        confirmed = client.submitted_days(t["predico_resource"])
        out["submitted"] = out.index.tz_convert(t["timezone"]).strftime("%Y-%m-%d").isin(confirmed)
        hf.upload(out, "live", f"forecasts/{name}.parquet", f"Publish {name} {now:%Y-%m-%d}")
        run_rows += rows
        print(f"{name}: {out.index.normalize().nunique()} days from {len(runs)} runs")
    if run_rows:
        hf.upload(pd.DataFrame(run_rows), "live", "runs.parquet", f"Runs {now:%Y-%m-%d}")


if __name__ == "__main__":
    main()
