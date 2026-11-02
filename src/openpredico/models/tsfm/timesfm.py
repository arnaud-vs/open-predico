"""TimesFM 3.0 (Google), zero-shot with covariates. `pip install openpredico[timesfm]`.

Code: Apache-2.0. Weights (google/timesfm-3.0-pytorch): TimesFM Non-Commercial License v1.0,
non-commercial and non-production use only: benchmark only, not for live submission.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)


def load(repo: str = "google/timesfm-3.0-pytorch", device: str = "cpu"):
    from timesfm3 import TimesFM3Forecaster  # optional dependency

    return TimesFM3Forecaster.from_pretrained(repo, device=device)


def make_predict(
    model, context_days: int = 14, categorical: Sequence[str] = ()
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]:
    """`predict(past, future, days)` for the backtest; load `model` once, outside.

    `past`: target (first column) and past-only covariates, up to the cutoff.
    `future`: known-future covariates through the end of the last target day.
    NaN is passed through: TimesFM 3 interpolates gaps in its inputs itself.
    """

    def predict(past: pd.DataFrame, future: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        end = past.index[-1]
        ctx = past[past.index > end - pd.Timedelta(days=context_days)].asfreq(STEP)
        stop = (days.max() + pd.DateOffset(days=1)).tz_convert("UTC")  # calendar day: 92/96/100
        grid = pd.date_range(end + STEP, stop, freq=STEP, inclusive="left")
        fut = future.reindex(ctx.index.append(grid))
        # ponytail: TimesFM 3 has no categorical input; `categorical` columns go in as numbers.
        fc = model.predict(
            ctx.iloc[:, 0].to_numpy(float),
            horizon=len(grid),
            past_only_covariates=ctx.iloc[:, 1:].to_numpy(float).T if ctx.shape[1] > 1 else None,
            past_future_covariates=fut.to_numpy(float).T if fut.shape[1] else None,
            return_quantiles=True,
        )
        q = np.asarray(fc.quantiles, dtype=float).reshape(len(grid), -1)[:, [0, 4, 8]]  # deciles
        out = pd.DataFrame(np.sort(q, axis=1), index=grid, columns=["q10", "q50", "q90"])
        return out[out.index >= days.min().tz_convert("UTC")]  # drop the roll-through to day D

    return predict
