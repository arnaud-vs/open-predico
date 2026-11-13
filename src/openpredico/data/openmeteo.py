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
    "be": [  # one point per Belgian province (Elia ods032 regions)
        (51.22, 4.62),  # Antwerp
        (50.99, 5.45),  # Limburg
        (51.03, 3.80),  # East Flanders
        (51.05, 3.05),  # West Flanders
        (50.88, 4.55),  # Flemish Brabant
        (50.85, 4.35),  # Brussels
        (50.67, 4.58),  # Walloon Brabant
        (50.47, 3.95),  # Hainaut
        (50.33, 4.85),  # Namur
        (50.53, 5.70),  # Liège
        (49.95, 5.50),  # Luxembourg
    ],
}
# Weighted means (else plain): Elia's monitored PV capacity per province, MW, on 2026-10-25.
# ponytail: fixed weights; capacity grows a few % a year, mostly in Flanders.
WEIGHTS = {"be": [2029, 1565, 2040, 1912, 1168, 331, 358, 1058, 445, 854, 368]}
# Belgium's 09:00 origin (10:00 gate) comes before the 00 UTC run lands (~08:00 UTC): its areas
# read the 18 UTC run of D-2 (29 to 53 h ahead) instead.
EARLIER = {"be": pd.Timedelta(hours=6)}
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
# A request costs one call per point (7, or 11 for "be"). Free tier: 5000 calls/h. The live run
# (~92 days) can use 1 s; a backfill of hundreds of days sets PAUSE = 6 (~4200 calls/h; "be": 9).
PAUSE = 1
_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        # waits 10..160 s (~5 min): ConnectionErrors outlasted 3 short retries on 2026-11-12
        max_retries=Retry(total=5, backoff_factor=10, status_forcelist=[429, 500, 502, 503])
    ),
)


def fetch(start: pd.Timestamp, end: pd.Timestamp, areas=None) -> dict[str, pd.DataFrame]:
    """Per area, VARIABLES averaged over its points on the 15-min UTC grid [start, end).

    `areas` (default: those reading the 00 UTC run) must share one run (EARLIER).
    """
    areas = areas or [a for a in POINTS if a not in EARLIER]
    grid = pd.date_range(start, end, freq=STEP, inclusive="left", name="time")
    days = pd.date_range(grid[0].tz_convert(TZ).normalize(), grid[-1].tz_convert(TZ), freq="D")
    parts = {area: [] for area in areas}
    for day in days:
        for area, hourly in _run(day, areas).items():
            parts[area].append(hourly)
        time.sleep(PAUSE)
    out = {}
    for area, frames in parts.items():
        hourly = pd.concat(frames)
        out[area] = hourly.reindex(grid, method="ffill", limit=3)  # hour -> its quarter-hours
    return out


def run_time(day: pd.Timestamp, area: str = "land") -> pd.Timestamp:
    """The run a local target day reads: 00 UTC on the local day before, or EARLIER."""
    run = pd.Timestamp(f"{day - pd.Timedelta(days=1):%Y-%m-%d}", tz="UTC")
    return run - EARLIER.get(area, pd.Timedelta(0))


def _run(day: pd.Timestamp, areas: list[str]) -> dict[str, pd.DataFrame]:
    """Hourly forecasts of the areas' run for the hours of local `day`, mean per area."""
    points = [p for a in areas for p in POINTS[a]]
    resp = _session.get(
        BASE_URL,
        params={
            "latitude": ",".join(str(p[0]) for p in points),
            "longitude": ",".join(str(p[1]) for p in points),
            "hourly": ",".join(VARIABLES),
            "models": MODEL,
            "run": f"{run_time(day, areas[0]):%Y-%m-%dT%H:%M}",
            "forecast_days": 3,  # covers the local target day, plus an hour
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
    for area in areas:
        ps, frames = POINTS[area], []
        for loc in locations[i : i + len(ps)]:
            df = pd.DataFrame(loc["hourly"])
            df.index = pd.to_datetime(df.pop("time"), utc=True)
            for v in PRECEDING_HOUR & set(df):  # the 10:00 value is the 09:00-10:00 mean
                df[v] = df[v].shift(-1)
            frames.append(df[(df.index >= lo) & (df.index < hi)])
        if w := WEIGHTS.get(area):
            out[area] = sum(f * k for f, k in zip(frames, w, strict=True)) / sum(w)
        else:
            out[area] = pd.concat(frames).groupby(level=0).mean()
        i += len(ps)
    return out
