"""t0-gamma (The Forecasting Company), zero-shot with covariates. `pip install openpredico[t0]`.

Code (tfc-t0) and weights: Apache-2.0.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)


def load(repo: str = "theforecastingcompany/t0-gamma", device: str = "cpu"):
    from t0 import T0Forecaster  # optional dependency

    return T0Forecaster.from_pretrained(repo).to(device).eval()


def make_predict(
    model, context_days: int = 14, categorical: Sequence[str] = ()
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]:
    """`predict(past, future, days)` for the backtest; load `model` once, outside.

    `past`: target (first column) and past-only covariates, up to the cutoff. t0 takes past
    covariates as extra context variates, forecast jointly with the target (only it is kept).
    `future`: known-future covariates through the end of the last target day.
    NaN is passed through: t0 masks missing values itself.
    """

    def predict(past: pd.DataFrame, future: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        end = past.index[-1]
        ctx = past[past.index > end - pd.Timedelta(days=context_days)].asfreq(STEP)
        stop = (days.max() + pd.DateOffset(days=1)).tz_convert("UTC")  # calendar day: 92/96/100
        grid = pd.date_range(end + STEP, stop, freq=STEP, inclusive="left")
        fut = future.reindex(ctx.index.append(grid))
        # ponytail: t0 has no categorical input; `categorical` columns go in as numbers.
        fc = model.predict(
            ctx.to_numpy("float32").T[None],  # (1, V, T)
            horizon=len(grid),
            quantile_levels=[0.1, 0.5, 0.9],
            future_covariates=fut.to_numpy("float32").T[None] if fut.shape[1] else None,
        )
        q = np.array(fc.quantiles[0, 0].tolist())  # (1, V, H, 3) -> the target's (H, 3)
        out = pd.DataFrame(np.sort(q, axis=1), index=grid, columns=["q10", "q50", "q90"])
        return out[out.index >= days.min().tz_convert("UTC")]  # drop the roll-through to day D

    return predict
