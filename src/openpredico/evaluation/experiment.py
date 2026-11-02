"""Backtest a TSFM with covariates on the published features (`open-predico/benchmark`, v1).

Every v1 column is used: `availability` in groups.toml splits known-future from past columns,
the target is the first column and the other target a past covariate. Predictions (with the
truth as `obs`) go to `<out>/<config>/<target>.parquet`, ready for `evaluation.results`.
`drop` leaves out comma-separated columns (ablation); `drop=all` keeps only the target history
(univariate: no past, known-future or calendar covariates).

    python -m openpredico.evaluation.experiment <config> <repo> <context days> <out dir> \
        [<device>] [<drop>] [<origin HH:MM, local; default from targets.toml>]
"""

import importlib
import sys
import time
from pathlib import Path

import pandas as pd

from openpredico.config import load_targets
from openpredico.data.groups import load_groups
from openpredico.data.hf import download
from openpredico.evaluation.backtest import backtest

# ponytail: target name -> its column; move to targets.toml when a third target arrives.
TARGET_COLUMNS = {"spain_mfrr_energy": "mfrr_energy", "spain_mfrr_price": "mfrr_price"}
CATEGORICAL = ["qh_of_day", "weekday"]
# Hugging Face org of the weights -> adapter module in models/tsfm/.
ADAPTERS = {
    "mistralai": "timestral",
    "amazon": "chronos2",
    "google": "timesfm",
    "theforecastingcompany": "t0",
}
MAX_NAN = 0.9  # columns missing more than this share of the history are dropped


# ponytail: groups added after v1 (not on HF yet); drop once v2 is published and chosen.
NOT_IN_V1 = {"mfrr_bids"}


def load_v1(version: str = "v1") -> dict[str, pd.DataFrame]:
    groups = [g for g in load_groups() if g not in NOT_IN_V1]
    return {g: download("benchmark", f"{version}/{g}.parquet") for g in groups}


def features(
    frames: dict[str, pd.DataFrame], target: str, drop: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    """History frame (target first, then past columns, then future ones) and the future columns.

    `drop`: comma-separated columns to leave out, or "all" for the target alone.
    """
    groups = load_groups()
    y = TARGET_COLUMNS[target]
    if drop == "all":
        return frames["target"][[y]].astype(float), []
    df = pd.concat(frames.values(), axis=1).drop(columns=[c for c in drop.split(",") if c])
    nan = df.isna().mean()
    drop = [c for c in nan.index[nan > MAX_NAN] if c not in TARGET_COLUMNS.values()]
    if drop:
        print(f"dropped (>{MAX_NAN:.0%} missing): {drop}")
    df = df.drop(columns=drop)
    # ESIOS omits zero quarter-hours in the sparse mFRR series: missing = no activation.
    # ponytail: hard-coded until groups.toml carries `fill = 0` for these columns.
    mfrr = [c for c in groups["mfrr"] if c in df]
    df[mfrr] = df[mfrr].fillna(0)
    future = [c for g in groups.values() for c, d in g.items() if d["availability"] == "future"]
    future = [c for c in future if c in df]
    past = [c for c in df if c not in future and c != y]
    return df[[y, *past, *future]].astype(float), future


def run(
    config: str,
    repo: str,
    context_days: int,
    out: Path,
    device: str = "cpu",
    drop: str = "",
    origin: str = "",
) -> None:
    frames = load_v1()
    adapter = importlib.import_module(f"openpredico.models.tsfm.{ADAPTERS[repo.split('/')[0]]}")
    model = adapter.load(repo, device)
    for target, t in load_targets().items():
        history, future = features(frames, target, drop)
        days = pd.date_range(*t["test_period"], tz=t["timezone"])
        predict = adapter.make_predict(model, context_days, CATEGORICAL)
        start = time.time()
        preds = backtest(predict, history, days, origin or t["origin"], future)
        print(
            f"{target}: {len(future)} future, {history.shape[1] - 1 - len(future)} past "
            f"columns, {len(days)} days in {time.time() - start:.0f} s"
        )
        preds["obs"] = frames["target"][TARGET_COLUMNS[target]]  # raw truth, NaN kept
        preds.index.name = "time"
        (out / config).mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out / config / f"{target}.parquet")


if __name__ == "__main__":
    run(*sys.argv[1:3], int(sys.argv[3]), Path(sys.argv[4]), *sys.argv[5:])
