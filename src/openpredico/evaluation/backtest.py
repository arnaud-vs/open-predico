"""Rolling-origin backtest: for each target day, forecast from what was known at the origin."""

from collections.abc import Callable

import pandas as pd

# ponytail: one fixed publication lag for all past data; ESIOS balancing data shows up ~1-1.5 h
# late. Make it per column (from groups.toml) when features with other lags arrive.
PUBLICATION_LAG = pd.Timedelta(hours=2)


def backtest(
    predict: Callable[[pd.Series, pd.DatetimeIndex], pd.DataFrame],
    history: pd.Series,
    days: pd.DatetimeIndex,
    origin: str = "13:00",
) -> pd.DataFrame:
    """Concatenated D+1 forecasts for `days` (local midnights).

    `predict(past, [day])` sees only values published before origin on D-1.
    """
    out = []
    for day in days:
        d1 = (day - pd.DateOffset(days=1)).date()  # calendar day: safe across DST
        cutoff = pd.Timestamp(f"{d1} {origin}").tz_localize(day.tz).tz_convert("UTC")
        past = history[history.index < cutoff - PUBLICATION_LAG]
        out.append(predict(past, pd.DatetimeIndex([day])))
    return pd.concat(out)
