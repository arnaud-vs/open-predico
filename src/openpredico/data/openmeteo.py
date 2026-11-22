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

import logging
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# ponytail: the Standard plan key (OPENMETEO_API_KEY) doesn't cover the Single Runs API
# (Professional and up: HTTP 403), so runs use the free tier (non-commercial) until upgraded.
# Set API_KEY = os.environ.get("OPENMETEO_API_KEY") and the "customer-" host after an upgrade.
API_KEY = None
BASE_URL = f"https://{'customer-' if API_KEY else ''}single-runs-api.open-meteo.com/v1/forecast"
logging.getLogger("urllib3").setLevel(logging.ERROR)  # retry warnings print the URL (key)
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
POINTS["be_d2"] = POINTS["be_knmi"] = POINTS["be_aifs"] = POINTS["be"]  # other models (AREA_MODEL)
# Weighted means (else plain): Elia's monitored PV capacity per province, MW, on 2026-10-25.
# ponytail: fixed weights; capacity grows a few % a year, mostly in Flanders.
WEIGHTS = {"be": [2029, 1565, 2040, 1912, 1168, 331, 358, 1058, 445, 854, 368]}
WEIGHTS["be_d2"] = WEIGHTS["be_knmi"] = WEIGHTS["be_aifs"] = WEIGHTS["be"]
# Areas fetched in their own request, with their run's shift back from 00 UTC of D-1. ECMWF runs
# land ~6h40 after init (06 UTC run: 12:40 UTC on 2026-11-14), so the 00 UTC run is out ~06:40
# UTC, before Belgium's 09:00 origin (07:00 UTC in summer); the 06 UTC run never is.
# Backtest 2026-11-13 used the 18 UTC run of D-2 (shift 6 h) for Belgium (benchmark v3).
# ICON-D2 (DWD, 2 km, 48 h, every 3 h; 03 UTC runs archived since 2026-04-02) lands ~1h55
# after init (15 UTC run: 16:54 UTC on 2026-11-14): the 03 UTC run of D-1 is out ~05:00 UTC.
# KNMI HARMONIE-AROME (2 km, Benelux, 60 h; hourly runs, archived every 3 h since ~2026-04-05)
# lands ~2h36 after init: 03 UTC run of D-1 (~05:40 UTC). ECMWF AIFS (AI model, 0.25 deg;
# archived since 2026-04-02) lands ~5h38 after init: 00 UTC run of D-1 (~05:40 UTC).
RUN_SHIFT = {
    "be": pd.Timedelta(0),
    "be_d2": pd.Timedelta(hours=-3),
    "be_knmi": pd.Timedelta(hours=-3),
    "be_aifs": pd.Timedelta(0),
}
AREA_MODEL = {  # else MODEL
    "be_d2": "icon_d2",
    "be_knmi": "knmi_harmonie_arome_netherlands",
    "be_aifs": "ecmwf_aifs025_single",
}
# Per model: hours between the runs the Single Runs API archives, and how long after init a run
# is there (the figures above; IFS's own 00 UTC run landed ~06:40 UTC). Shown on the dashboard.
CYCLE = {"ecmwf_ifs": 6, "ecmwf_aifs025_single": 6, "icon_d2": 3}
CYCLE["knmi_harmonie_arome_netherlands"] = 3  # hourly runs, archived every 3 h
LATENCY = {
    "ecmwf_ifs": pd.Timedelta("6h40min"),
    "ecmwf_aifs025_single": pd.Timedelta("5h38min"),
    "icon_d2": pd.Timedelta("1h55min"),
    "knmi_harmonie_arome_netherlands": pd.Timedelta("2h36min"),
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
# A request costs one call per point (7, or 11 for Belgium). Free tier: 600 calls/min, 5000/h.
# Seconds to wait per point after each request: live (~92 days) 0.15 (~400 calls/min); a backfill
# of hundreds of days sets PAUSE = 0.8 (~4400 calls/h).
PAUSE = 0.15
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

    `areas` (default: those not in RUN_SHIFT) must share one run (RUN_SHIFT).
    """
    areas = areas or [a for a in POINTS if a not in RUN_SHIFT]
    grid = pd.date_range(start, end, freq=STEP, inclusive="left", name="time")
    days = pd.date_range(grid[0].tz_convert(TZ).normalize(), grid[-1].tz_convert(TZ), freq="D")
    parts = {area: [] for area in areas}
    for day in days:
        try:
            run = _run(day, areas)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 400:
                raise
            print(f"openmeteo {areas[0]}: no run for {day:%Y-%m-%d}, left empty")  # archive gap
            continue
        for area, hourly in run.items():
            parts[area].append(hourly)
        time.sleep(PAUSE * sum(len(POINTS[a]) for a in areas))
    out = {}
    for area, frames in parts.items():
        if not frames:
            raise ValueError(f"openmeteo {area}: no run for any day")
        hourly = pd.concat(frames)
        out[area] = hourly.reindex(grid, method="ffill", limit=3)  # hour -> its quarter-hours
    return out


def run_time(day: pd.Timestamp, area: str = "land") -> pd.Timestamp:
    """The run a local target day reads: 00 UTC on the local day before, minus RUN_SHIFT."""
    run = pd.Timestamp(f"{day - pd.Timedelta(days=1):%Y-%m-%d}", tz="UTC")
    return run - RUN_SHIFT.get(area, pd.Timedelta(0))


def fetch_run(run: pd.Timestamp, area: str, forecast_days: int = 3) -> pd.DataFrame:
    """Hourly VARIABLES of one run of the area's model, every hour it covers, mean per area.

    Raises HTTPError 400 when the run isn't archived.
    """
    frames = _get(POINTS[area], AREA_MODEL.get(area, MODEL), run, forecast_days, area)
    return _mean(area, frames).dropna(how="all")


def _run(day: pd.Timestamp, areas: list[str]) -> dict[str, pd.DataFrame]:
    """Hourly forecasts of the areas' run for the hours of local `day`, mean per area."""
    points = [p for a in areas for p in POINTS[a]]
    run = run_time(day, areas[0])
    frames = _get(points, AREA_MODEL.get(areas[0], MODEL), run, 3, f"{areas[0]} {day:%Y-%m-%d}")
    out, i = {}, 0
    lo, hi = day.tz_convert("UTC"), (day + pd.DateOffset(days=1)).tz_convert("UTC")
    for area in areas:
        n = len(POINTS[area])
        out[area] = _mean(area, [f[(f.index >= lo) & (f.index < hi)] for f in frames[i : i + n]])
        i += n
    return out


def _get(points, model: str, run: pd.Timestamp, forecast_days: int, label: str):
    """Hourly VARIABLES per point of one run (UTC; radiation shifted to the hour it starts)."""
    started = time.time()
    try:
        resp = _session.get(
            BASE_URL,
            params={
                "latitude": ",".join(str(p[0]) for p in points),
                "longitude": ",".join(str(p[1]) for p in points),
                "hourly": ",".join(VARIABLES),
                "models": model,
                "run": f"{run:%Y-%m-%dT%H:%M}",
                "forecast_days": forecast_days,  # from 00 UTC of the run's day
                "timezone": "GMT",
                "wind_speed_unit": "ms",
            }
            | ({"apikey": API_KEY} if API_KEY else {}),
            timeout=60,
        )
        if (took := time.time() - started) > 20:  # retries are silent otherwise
            print(f"openmeteo {label}: {took:.0f} s, HTTP {resp.status_code}")
        resp.raise_for_status()
    except requests.RequestException as e:  # its message quotes the URL, key included
        raise redact(e) from None
    return _frames(resp)


def redact(e: requests.RequestException) -> requests.RequestException:
    """The error with the API key cut from its message (it quotes the URL)."""
    if API_KEY:
        e.args = tuple(str(a).replace(API_KEY, "***") for a in e.args)
    return e


def _frames(resp: requests.Response) -> list[pd.DataFrame]:
    locations = resp.json()
    frames = []
    for loc in locations if isinstance(locations, list) else [locations]:
        df = pd.DataFrame(loc["hourly"])
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        for v in PRECEDING_HOUR & set(df):  # the 10:00 value is the 09:00-10:00 mean
            df[v] = df[v].shift(-1)
        frames.append(df)
    return frames


def _mean(area: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The area's points averaged: weighted by WEIGHTS if any, else plain."""
    if w := WEIGHTS.get(area):
        return sum(f * k for f, k in zip(frames, w, strict=True)) / sum(w)
    return pd.concat(frames).groupby(level=0).mean()
