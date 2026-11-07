"""Rolling-origin backtest: for each target day, forecast from what was known at the origin."""

from collections.abc import Callable, Sequence

import pandas as pd

# ponytail: one fixed publication lag for all past data; ESIOS balancing data shows up ~1-1.5 h
# late. Make it per column (from groups.toml) when features with other lags arrive.
PUBLICATION_LAG = pd.Timedelta(hours=2)


def backtest(
    predict: Callable[..., pd.DataFrame],
    history: pd.Series | pd.DataFrame,
    days: pd.DatetimeIndex,
    origin: str = "14:00",
    future_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Concatenated D+1 forecasts for `days` (local midnights).

    Series `history`: `predict(past, [day])` sees only values published before origin on D-1.
    DataFrame `history` (target first): `predict(past, future, [day])`; `past` holds the other
    columns up to that same cutoff, `future` the `future_cols` through the end of the target day.
    """
    cols = list(future_cols)
    out = []
    for day in days:
        d1 = (day - pd.DateOffset(days=1)).date()  # calendar day: safe across DST
        cutoff = pd.Timestamp(f"{d1} {origin}").tz_localize(day.tz).tz_convert("UTC")
        past = history[history.index < cutoff - PUBLICATION_LAG]
        target = pd.DatetimeIndex([day])
        if isinstance(history, pd.Series):
            out.append(predict(past, target))
            continue
        stop = (day + pd.DateOffset(days=1)).tz_convert("UTC")
        future = history.loc[history.index < stop, cols]
        out.append(predict(past.drop(columns=cols), future, target))
    return pd.concat(out)


def clip(fc: pd.DataFrame, target: dict) -> pd.DataFrame:
    """Generation targets: q10/q50/q90 clipped to [0, capacity] (keeps them sorted).

    Market targets (may be negative) are returned unchanged.
    """
    if target["kind"] != "generation":
        return fc
    q = ["q10", "q50", "q90"]
    return fc.assign(**{c: fc[c].clip(0, target["capacity"]) for c in q})
