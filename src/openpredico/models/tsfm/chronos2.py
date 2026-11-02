"""Chronos-2 (Amazon), zero-shot with covariates. `pip install openpredico[chronos2]`.

Code and weights: Apache-2.0. Context capped at 8192 steps (~85 days of quarter-hours).
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)


def load(repo: str = "amazon/chronos-2", device: str = "cpu"):
    from chronos import Chronos2Pipeline  # optional dependency

    return Chronos2Pipeline.from_pretrained(repo, device_map=device)


def make_predict(
    model, context_days: int = 14, categorical: Sequence[str] = ()
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]:
    """`predict(past, future, days)` for the backtest; load `model` once, outside.

    `past`: target (first column) and past-only covariates, up to the cutoff.
    `future`: known-future covariates through the end of the last target day; the
    `categorical` ones are passed as strings (Chronos-2 encodes them), the rest are numeric.
    NaN is passed through: Chronos-2 masks missing values itself.
    """

    def predict(past: pd.DataFrame, future: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        end = past.index[-1]
        ctx = past[past.index > end - pd.Timedelta(days=context_days)].asfreq(STEP)
        stop = (days.max() + pd.DateOffset(days=1)).tz_convert("UTC")  # calendar day: 92/96/100
        grid = pd.date_range(end + STEP, stop, freq=STEP, inclusive="left")

        def col(s: pd.Series) -> np.ndarray:
            return s.to_numpy(str) if s.name in categorical else s.to_numpy(float)

        fut = future.reindex(ctx.index.append(grid))
        hist, ahead = fut.iloc[: len(ctx)], fut.iloc[len(ctx) :]
        inputs = {
            "target": ctx.iloc[:, 0].to_numpy(float),
            "past_covariates": {c: col(s) for df in (ctx.iloc[:, 1:], hist) for c, s in df.items()},
            "future_covariates": {c: col(s) for c, s in ahead.items()},
        }
        q, _ = model.predict_quantiles(
            [inputs], prediction_length=len(grid), quantile_levels=[0.1, 0.5, 0.9]
        )
        q = np.asarray(q[0], dtype=float).reshape(len(grid), 3)  # (1, steps, 3), on CPU
        out = pd.DataFrame(np.sort(q, axis=1), index=grid, columns=["q10", "q50", "q90"])
        return out[out.index >= days.min().tz_convert("UTC")]  # drop the roll-through to day D

    return predict
