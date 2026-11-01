"""Timestral (Mistral AI), zero-shot with covariates. `pip install openpredico[timestral]`.

Weights: Modified MIT (attribution; no use if your company's monthly revenue exceeds $20M).
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)


def load(repo: str = "mistralai/Timestral-5M", device: str = "cpu"):
    from timestral import TimestralForecaster  # optional dependency

    return TimestralForecaster.from_pretrained(repo, device=device)


def make_predict(
    model, context_days: int = 14, categorical: Sequence[str] = ()
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]:
    """`predict(past, future, days)` for the backtest; load `model` once, outside.

    `past`: target (first column) and past-only covariates, up to the cutoff.
    `future`: known-future covariates through the end of the last target day; the
    `categorical` ones are target-encoded by Timestral, the rest are numeric.
    NaN is passed through: Timestral masks missing values itself.
    """

    def predict(past: pd.DataFrame, future: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        end = past.index[-1]
        ctx = past[past.index > end - pd.Timedelta(days=context_days)].asfreq(STEP)
        stop = (days.max() + pd.DateOffset(days=1)).tz_convert("UTC")  # calendar day: 92/96/100
        grid = pd.date_range(end + STEP, stop, freq=STEP, inclusive="left")
        fut = future.reindex(ctx.index.append(grid))
        cat = [c for c in fut.columns if c in categorical]
        num = fut.drop(columns=cat)

        def rows(df: pd.DataFrame) -> np.ndarray | None:  # (1, C, L), or None without columns
            return df.to_numpy(float).T[None] if df.shape[1] else None

        fc = model.forecast(
            ctx.iloc[:, 0].to_numpy(float),
            prediction_length=len(grid),
            past_covariates=rows(ctx.iloc[:, 1:]),
            known_covariates=rows(num),
            categorical_known_covariates=rows(fut[cat]),
        )
        q = np.stack([np.asarray(fc.quantile(p)).reshape(-1) for p in (0.1, 0.5, 0.9)], axis=1)
        out = pd.DataFrame(np.sort(q, axis=1), index=grid, columns=["q10", "q50", "q90"])
        return out[out.index >= days.min().tz_convert("UTC")]  # drop the roll-through to day D

    return predict
