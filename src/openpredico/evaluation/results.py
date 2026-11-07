"""Score backtest prediction files into `results/<config>.csv` (one row per target and day).

Prediction files hold q10/q50/q90 and the truth `obs` on a UTC index, one per config and target,
at `predictions/backtest/<config>/<target>.parquet` in the `open-predico/benchmark` dataset.

    uv run python -m openpredico.evaluation.results <config> [<prediction dir>]
"""

import sys
from pathlib import Path

import pandas as pd

from openpredico.config import load_targets
from openpredico.data.hf import download
from openpredico.evaluation.experiment import NOT_IN_V1
from openpredico.evaluation.metrics import daily_scores

RESULTS = Path(__file__).resolve().parents[3] / "results"


def score(preds: pd.DataFrame, tz: str = "Europe/Madrid") -> pd.DataFrame:
    """Daily scores of one prediction file (days in `tz`); the `obs` column is the truth."""
    return daily_scores(preds[["q10", "q50", "q90"]], preds.obs, tz)


def results(config: str, local_dir: Path | None = None) -> pd.DataFrame:
    """Daily scores for every target of `config`, from `local_dir` if given, else from HF."""
    out = []
    for target, t in load_targets().items():
        if NOT_IN_V1 & set(t["groups"]):
            continue  # no backtest yet
        path = f"predictions/backtest/{config}/{target}.parquet"
        if local_dir:
            preds = pd.read_parquet(local_dir / config / f"{target}.parquet")
        else:
            preds = download("benchmark", path)
        s = score(preds, t["timezone"]).reset_index()
        s.insert(0, "target", target)
        s["day"] = s.day.dt.strftime("%Y-%m-%d")
        out.append(s)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    config = sys.argv[1]
    local = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    RESULTS.mkdir(exist_ok=True)
    results(config, local).to_csv(RESULTS / f"{config}.csv", index=False, float_format="%.4f")
