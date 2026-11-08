"""Open-Meteo weather forecasts as issued three days ahead, averaged over a few points per area.

Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api), `<variable>_previous_day3`:
"the value that was predicted" 72 h before its valid time. The latest value we use (D+2 00:00
UTC, the hour after a target day for radiation) was predicted at D-1 00:00 UTC, hours before a
14:00 D-1 origin, in the backtest and the live run alike (same vintage, history from 2024).
Not `_previous_day2`: 48 h before D+1's afternoon is D-1's afternoon, after the origin.
Model: Open-Meteo's best match (ICON in Germany). Hourly values are repeated per quarter-hour.

Keys (`openmeteo:<area>.<variable>` in groups.toml): an area of POINTS and an hourly variable.
No key needed; the free tier is non-commercial (DATA_SOURCES.md).
"""

import time

import pandas as pd
import requests

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VINTAGE = "previous_day3"  # 72 h before valid time: issued before every origin (docstring)
# ponytail: a handful of hand-picked points across the 50Hertz area, plain mean per area.
POINTS = {  # area -> [(lat, lon)]
    "land": [
        (53.1, 13.9),  # Brandenburg (Uckermark)
        (53.8, 12.5),  # Mecklenburg-Vorpommern
        (51.9, 11.7),  # Saxony-Anhalt
        (51.2, 13.3),  # Saxony
        (51.0, 11.1),  # Thuringia
    ],
    "offshore": [(54.8, 14.1), (54.6, 12.65)],  # Baltic: Arkona / Baltic 2, Baltic 1
}
VARIABLES = [
    "wind_speed_100m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
]
PRECEDING_HOUR = {"shortwave_radiation", "direct_radiation", "diffuse_radiation"}  # hour means
CHUNK = pd.DateOffset(years=1)
STEP = pd.Timedelta("15min")


def fetch_frame(area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """VARIABLES averaged over the area's points, on the 15-min UTC grid [start, end)."""
    grid = pd.date_range(start, end, freq=STEP, inclusive="left", name="time")
    parts, t = [], start
    while t < end:
        parts.append(_get(POINTS[area], t, min(t + CHUNK, end)))
        t += CHUNK
        time.sleep(2)  # few, large requests; well under Open-Meteo's per-minute limit
    hourly = pd.concat(parts).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="last")]  # chunks overlap by a day
    out = pd.DataFrame(index=grid)
    for v in VARIABLES:
        s = hourly[v]
        if v in PRECEDING_HOUR:  # the 10:00 value is the 09:00-10:00 mean
            s = s.set_axis(s.index - pd.Timedelta("1h"))
        out[v] = s.reindex(grid, method="ffill", limit=3)  # hourly -> its four quarter-hours
    return out


def _get(points: list, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly `_{VINTAGE}` values in [start, end), mean over `points`."""
    last = end - pd.Timedelta("1ns")
    resp = requests.get(
        BASE_URL,
        params={
            "latitude": ",".join(str(p[0]) for p in points),
            "longitude": ",".join(str(p[1]) for p in points),
            "hourly": ",".join(f"{v}_{VINTAGE}" for v in VARIABLES),
            "start_date": f"{start - pd.Timedelta('1h'):%Y-%m-%d}",  # radiation: next hour's mean
            "end_date": f"{last + pd.Timedelta('1h'):%Y-%m-%d}",
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        },
        timeout=300,
    )
    resp.raise_for_status()
    frames = []
    locations = resp.json()
    for loc in locations if isinstance(locations, list) else [locations]:
        df = pd.DataFrame(loc["hourly"])
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        frames.append(df.rename(columns=lambda c: c.removesuffix(f"_{VINTAGE}")))
    return pd.concat(frames).groupby(level=0).mean()
