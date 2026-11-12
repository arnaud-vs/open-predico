"""Open-Meteo weather forecasts from ECMWF IFS's 00 UTC run of D-1, averaged over points per area.

Single Runs API (https://open-meteo.com/en/docs/single-runs-api), model `ecmwf_ifs` (9 km, runs
archived since March 2024; other models only since April 2026). Each local target day takes the
00 UTC run of the day before it: 23 to 47 h ahead, published around 08:00 UTC, hours before a
14:00 D-1 origin. The backtest and the live run read the same run, so both see the same vintage.
The 06 UTC run would be 6 h fresher but lands around 13:00 UTC, too close to the origin.

Keys (`openmeteo:<area>.<variable>` in groups.toml): an area of POINTS and an hourly variable.
One request per run covers every point and variable. No key needed; the free tier is
non-commercial (DATA_SOURCES.md).
"""

import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
TZ = "Europe/Berlin"  # target days are local days
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
STEP = pd.Timedelta("15min")
# A request costs one call per point (7). Free tier: 5000 calls/h. The live run (~92 days) can
# use 1 s; a backfill of hundreds of days sets PAUSE = 6 (~4200 calls/h).
PAUSE = 1
_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        # waits 10..160 s (~5 min): ConnectionErrors outlasted 3 short retries on 2026-11-12
        max_retries=Retry(total=5, backoff_factor=10, status_forcelist=[429, 500, 502, 503])
    ),
)


def fetch(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Per area, VARIABLES averaged over its points on the 15-min UTC grid [start, end)."""
    grid = pd.date_range(start, end, freq=STEP, inclusive="left", name="time")
    days = pd.date_range(grid[0].tz_convert(TZ).normalize(), grid[-1].tz_convert(TZ), freq="D")
    parts = {area: [] for area in POINTS}
    for day in days:
        for area, hourly in _run(day).items():
            parts[area].append(hourly)
        time.sleep(PAUSE)
    out = {}
    for area, frames in parts.items():
        hourly = pd.concat(frames)
        out[area] = hourly.reindex(grid, method="ffill", limit=3)  # hour -> its quarter-hours
    return out


def run_time(day: pd.Timestamp) -> pd.Timestamp:
    """The run a local target day reads: 00 UTC on the local day before."""
    return pd.Timestamp(f"{day - pd.Timedelta(days=1):%Y-%m-%d}", tz="UTC")


def _run(day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Hourly forecasts of `run_time(day)` for the hours of local `day`, mean per area."""
    points = [p for ps in POINTS.values() for p in ps]
    resp = _session.get(
        BASE_URL,
        params={
            "latitude": ",".join(str(p[0]) for p in points),
            "longitude": ",".join(str(p[1]) for p in points),
            "hourly": ",".join(VARIABLES),
            "models": MODEL,
            "run": f"{run_time(day):%Y-%m-%dT%H:%M}",
            "forecast_days": 3,  # covers the local day after the run day, plus an hour
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        },
        timeout=60,
    )
    resp.raise_for_status()
    locations = resp.json()
    locations = locations if isinstance(locations, list) else [locations]
    out, i = {}, 0
    lo, hi = day.tz_convert("UTC"), (day + pd.DateOffset(days=1)).tz_convert("UTC")
    for area, ps in POINTS.items():
        frames = []
        for loc in locations[i : i + len(ps)]:
            df = pd.DataFrame(loc["hourly"])
            df.index = pd.to_datetime(df.pop("time"), utc=True)
            for v in PRECEDING_HOUR & set(df):  # the 10:00 value is the 09:00-10:00 mean
                df[v] = df[v].shift(-1)
            frames.append(df[(df.index >= lo) & (df.index < hi)])
        out[area] = pd.concat(frames).groupby(level=0).mean()
        i += len(ps)
    return out
