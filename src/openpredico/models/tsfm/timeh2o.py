"""TimeH2O (H2O.ai), zero-shot with known-future covariates.
Run with `uv run --with h2ofm==0.43.1 python -m ...` (not an extra, as ci-proto).

Code (h2ofm): Apache-2.0. Weights: H2O Model Non-Commercial License, gated on Hugging Face
(accept the terms on the model page first): backtest only, never live. timeh2o-v1 ignores
past-only covariates, so only the target and the known-future columns reach the model.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
QUANTILES = (0.1, 0.5, 0.9)


def load(repo: str = "h2oai/timeh2o-v1", device: str = "cpu"):
    from h2ofm import TimeH2O  # optional dependency

    return TimeH2O.from_pretrained(repo, device=device)


def make_predict(
    model, context_days: int = 14, categorical: Sequence[str] = ()
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]:
    """`predict(past, future, days)` for the backtest; load `model` once, outside.

    `past`: target (first column) and past-only covariates (ignored by timeh2o-v1).
    `future`: known-future covariates through the end of the last target day; the
    `categorical` ones are passed as strings (labels). NaN is passed through.
    """

    def predict(past: pd.DataFrame, future: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        end = past.index[-1]
        ctx = past[past.index > end - pd.Timedelta(days=context_days)].asfreq(STEP)
        stop = (days.max() + pd.DateOffset(days=1)).tz_convert("UTC")  # calendar day: 92/96/100
        grid = pd.date_range(end + STEP, stop, freq=STEP, inclusive="left")
        fut = future.reindex(ctx.index.append(grid))
        rows = [s.to_numpy(str) if c in categorical else s.to_numpy(float) for c, s in fut.items()]
        fc = model.forecast(
            ctx.iloc[:, 0].to_numpy(float),
            horizon=len(grid),
            future_covariates=rows or None,
            quantile_levels=QUANTILES,
        )
        q = np.asarray(fc.quantiles, dtype=float).reshape(len(grid), 3)
        out = pd.DataFrame(np.sort(q, axis=1), index=grid, columns=["q10", "q50", "q90"])
        return out[out.index >= days.min().tz_convert("UTC")]  # drop the roll-through to day D

    return predict
