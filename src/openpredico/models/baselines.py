"""Baselines. Each returns q10/q50/q90 for the 15-min slots of the target days."""

import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


def slot_climatology(
    history: pd.Series, target_days: pd.DatetimeIndex, days: int = 28, tz: str = "Europe/Madrid"
) -> pd.DataFrame:
    """Quantiles per local quarter-hour of day, pooled over the last `days` days of `history`.

    Weekdays are pooled on purpose: per-weekday quantiles scored worse on Spain mFRR
    (2026 backtest), because the weekly pattern is weak and samples per slot get too few.
    `history` must end where the forecaster's knowledge ends; NaN values are ignored.
    """
    local = history.tz_convert(tz)
    recent = local[local.index >= local.index.max() - pd.Timedelta(days=days)].dropna()
    by_slot = recent.groupby(_slot(recent.index)).quantile(list(QUANTILES)).unstack()
    by_slot.columns = [f"q{round(q * 100)}" for q in QUANTILES]

    grid = pd.date_range(
        target_days.min(), target_days.max() + pd.Timedelta(days=1), freq="15min", inclusive="left"
    )  # target_days are local midnights, so DST days get 92 or 100 slots
    out = by_slot.reindex(_slot(grid)).set_axis(grid)
    return out.tz_convert("UTC")


def _slot(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.hour * 4 + index.minute // 15)
