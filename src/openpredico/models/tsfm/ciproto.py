"""ci-proto (Continuous Intelligence), zero-shot with covariates. Not an extra (it pins torch
below the t0 extra): `uv run --with continuousintelligence==0.1.1 python -m ...`.

Code and weights: ci-proto Code and Model Licence v1.0, commercial use allowed. Publicly
distributed use must show "Uses ci-proto, released by Continuous Intelligence." and notify
contact@continuousintelligence.co.uk once. Histories over 16384 steps (~170 days of
quarter-hours) switch to the long checkpoint automatically.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
QUANTILES = [0.1, 0.5, 0.9]


def load(repo: str = "ContinuousIntelligence/ci-proto", device: str = "cpu"):
    import continuousintelligence as ci  # optional dependency

    return ci.load_model(repo, device=device)


def make_predict(
    model, context_days: int = 14, categorical: Sequence[str] = ()
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]:
    """`predict(past, future, days)` for the backtest; load `model` once, outside.

    `past`: target (first column) and past-only covariates, up to the cutoff.
    `future`: known-future covariates through the end of the last target day; the
    `categorical` ones are passed as category codes. Timestamps give ci-proto its own
    calendar features. NaN is passed through: ci-proto masks missing values itself.
    """
    import torch

    levels = [round(float(q), 2) for q in model.quantiles]
    rows = [levels.index(q) for q in QUANTILES]

    def predict(past: pd.DataFrame, future: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        end = past.index[-1]
        ctx = past[past.index > end - pd.Timedelta(days=context_days)].asfreq(STEP)
        stop = (days.max() + pd.DateOffset(days=1)).tz_convert("UTC")  # calendar day: 92/96/100
        grid = pd.date_range(end + STEP, stop, freq=STEP, inclusive="left")
        fut = future.reindex(ctx.index.append(grid))

        def tensor(df: pd.DataFrame):
            return torch.tensor(df.to_numpy("float32").T) if df.shape[1] else None

        q = model.forecast(
            tensor(ctx.iloc[:, :1]),
            horizon=len(grid),
            past_covariates=tensor(ctx.iloc[:, 1:]),
            future_covariates=tensor(fut),
            categorical_future=tuple(i for i, c in enumerate(fut) if c in categorical),
            timestamps=torch.tensor(ctx.index.append(grid).as_unit("ns").asi8),
            freq="15min",
        )
        q = q[0, rows].numpy().T  # (V, Q, H) -> the target's (H, 3)
        out = pd.DataFrame(np.sort(q, axis=1), index=grid, columns=["q10", "q50", "q90"])
        return out[out.index >= days.min().tz_convert("UTC")]  # drop the roll-through to day D

    return predict
