"""Google DeepMind WeatherNext 3 forecasts from BigQuery, per area on the 15-min UTC grid.

Table `open-predico.weathernext_3.weathernext_3_0_0_0p1deg` (0.1 deg, ensemble statistics; we
read the `_mean`s), a linked dataset that Application Default Credentials can query
(`gcloud auth application-default login`; extra `weathernext`). Hourly runs, 48 h ahead (00/06/12/18
UTC: 360 h). Runs land on GCS ~6h15-7h00 after init (Aug-Nov 2026); we assume LATENCY = 7h45.
Runs before 2026-08-28 were written on 2026-08-27: reforecasts, not what was out at the time.

Licence: WeatherNext data never goes to a public dataset (DATA_SOURCES.md); groups with a
`weathernext:` column are stored in the private HF dataset `hf.PRIVATE`.

Cost (BigQuery bills per byte): a query must filter one `init_time` and the points' cells
(`ST_DWITHIN`), and take the lead from the array offset (selecting `hours` or `time` adds
~125 GB to the estimate). The dry-run estimate is ~130 GB per selected column although a run
bills ~20 MB; BigQuery refuses a `maximum_bytes_billed` below the estimate, so we allow the
estimate + 10% and stop when a query bills over MAX_BILLED.

Keys (`weathernext:<area>.<variable>` in groups.toml): an area of AREAS and a variable of
openmeteo.VARIABLES, same units as Open-Meteo.
"""

import itertools
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from openpredico.data import openmeteo

PROJECT = "open-predico"
TABLE = f"{PROJECT}.weathernext_3.weathernext_3_0_0_0p1deg"
MAX_BILLED = 10**9  # bytes per query; a run bills ~20 MB, more means the filters didn't prune
# Our area -> its openmeteo.POINTS / WEIGHTS (the nearest 0.1 deg cell per point).
AREAS = {"be_wn3": "be", "land_wn3": "land", "offshore_wn3": "offshore"}
LATENCY = pd.Timedelta("7h45min")
WORKERS = 8  # queries in flight
# Run per area: 00 UTC of D-1 minus the shift (openmeteo.run_time). Belgium (origin 09:00
# Brussels, 07:00 UTC in summer): 23 UTC of D-2, out ~06:45 UTC, covers through 23 UTC of D, all
# of the local day in both seasons. Germany (14:00 Berlin, 12:00 UTC in summer): 04 UTC of D-1,
# out ~11:45 UTC, covers through 04 UTC of D+1.
RUN_SHIFT = {"be_wn3": pd.Timedelta(hours=1)}
RUN_SHIFT["land_wn3"] = RUN_SHIFT["offshore_wn3"] = pd.Timedelta(hours=-4)
# openmeteo variable -> (WN3 column, scale, offset); diffuse = shortwave - direct below.
COLUMNS = {
    "wind_speed_100m": ("wind_speed_100m_mean", 1, 0),
    "wind_speed_10m": ("wind_speed_10m_mean", 1, 0),
    "shortwave_radiation": ("surface_solar_radiation_downwards_1hr_mean", 1 / 3600, 0),  # J/m2
    "direct_radiation": ("total_sky_direct_solar_radiation_at_surface_1hr_mean", 1 / 3600, 0),
    "cloud_cover": ("total_cloud_cover_mean", 100, 0),  # 0-1 -> %
    "temperature_2m": ("temperature_2m_mean", 1, -273.15),  # K -> degC
}


def run_time(day: pd.Timestamp, area: str) -> pd.Timestamp:
    """The run a local target day reads (RUN_SHIFT)."""
    return openmeteo.run_time(day) - RUN_SHIFT[area]


def fetch(start, end, areas=None, client=None) -> dict[str, pd.DataFrame]:
    """Per area, openmeteo.VARIABLES on the 15-min UTC grid [start, end), one query per run."""
    areas = areas or list(AREAS)
    grid = pd.date_range(start, end, freq=openmeteo.STEP, inclusive="left", name="time")
    tz = openmeteo.TZ
    days = pd.date_range(grid[0].tz_convert(tz).normalize(), grid[-1].tz_convert(tz), freq="D")
    days = [(d, d.tz_convert("UTC"), (d + pd.DateOffset(days=1)).tz_convert("UTC")) for d in days]
    need = {}  # run -> the areas reading it
    for (day, _, _), area in itertools.product(days, areas):
        need.setdefault(run_time(day, area), []).append(area)
    # ponytail: a query takes ~20 s (mostly BigQuery's own latency): a few at a time
    pool = ThreadPoolExecutor(WORKERS)
    try:
        # a client per query: a shared one hung after ~120 threaded queries (2026-11-22)
        got = pool.map(lambda r: _run(client or _client(), r, list(dict.fromkeys(need[r]))), need)
        runs = dict(zip(need, got, strict=True))
    finally:  # an error (e.g. a query billing over MAX_BILLED) cancels the queued queries
        pool.shutdown(cancel_futures=True)
    parts = {a: [] for a in areas}
    for (day, lo, hi), area in itertools.product(days, areas):
        hourly = runs[run_time(day, area)].get(area)
        if hourly is None:
            print(f"weathernext {area}: no run for {day:%Y-%m-%d}, left empty")
            continue
        parts[area].append(hourly[(hourly.index >= lo) & (hourly.index < hi)])
    out = {}
    for area, frames in parts.items():
        if not frames:
            raise ValueError(f"weathernext {area}: no run for any day")
        out[area] = pd.concat(frames).reindex(grid, method="ffill", limit=3)  # hour -> quarters
    return out


def _run(client, run: pd.Timestamp, areas: list[str]) -> dict[str, pd.DataFrame]:
    """Hourly variables of one run, mean per area (weighted as openmeteo._mean); {} if absent."""
    points = [p for a in areas for p in openmeteo.POINTS[AREAS[a]]]
    sql = query(run, points)
    cap = int(client.query(sql, job_config=_config(dry_run=True)).total_bytes_processed * 1.1)
    job = client.query(sql, job_config=_config(maximum_bytes_billed=cap))
    rows = pd.DataFrame([dict(r.items()) for r in job.result(timeout=600)])
    if job.total_bytes_billed >= MAX_BILLED:  # stop everything: the filters didn't prune
        raise RuntimeError(f"weathernext {run}: billed {job.total_bytes_billed / 1e9:.1f} GB")
    print(f"weathernext {run:%Y-%m-%d %H} UTC: {job.total_bytes_billed / 1e6:.0f} MB billed")
    if rows.empty:
        return {}
    rows.index = run + pd.to_timedelta(rows.pop("i") + 1, unit="h")  # offset 0 = 1 h ahead
    df = pd.DataFrame({v: rows[c] * k + b for v, (c, k, b) in COLUMNS.items()}, index=rows.index)
    df["diffuse_radiation"] = (df.shortwave_radiation - df.direct_radiation).clip(lower=0)
    rad = ["shortwave_radiation", "direct_radiation", "diffuse_radiation"]  # preceding hour
    frames = []
    for _, f in df.groupby(rows.k.to_numpy()):  # point order (groupby sorts k)
        f = f.sort_index()
        f[rad] = f[rad].shift(-1)  # the 10:00 value is the 09:00-10:00 mean, as openmeteo
        frames.append(f[openmeteo.VARIABLES])
    out, i = {}, 0
    for a in areas:
        n = len(openmeteo.POINTS[AREAS[a]])
        out[a] = openmeteo._mean(AREAS[a], frames[i : i + n]).dropna(how="all")
        i += n
    return out


def query(run: pd.Timestamp, points: list[tuple[float, float]]) -> str:
    """SQL: the nearest cell's forecast per point `k`, lead `i` (hours - 1), COLUMNS."""
    pts = ", ".join(f"STRUCT({k} AS k, {x} AS lon, {y} AS lat)" for k, (y, x) in enumerate(points))
    multi = "MULTIPOINT(" + ", ".join(f"{x} {y}" for y, x in points) + ")"
    cols = ", ".join(f"f.{c}" for c, _, _ in COLUMNS.values())
    return f"""
WITH pts AS (SELECT k, ST_GEOGPOINT(lon, lat) AS p FROM UNNEST([{pts}])),
cells AS (
  SELECT geography, forecast FROM `{TABLE}`
  WHERE init_time = TIMESTAMP '{run:%Y-%m-%d %H:%M:%S}'
    AND ST_DWITHIN(geography, ST_GEOGFROMTEXT('{multi}'), 8000)
),
nearest AS (
  SELECT k, forecast FROM cells JOIN pts ON ST_DWITHIN(cells.geography, pts.p, 8000)
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY k ORDER BY ST_DISTANCE(cells.geography, pts.p), ST_ASTEXT(cells.geography)
  ) = 1
)
SELECT k, i, {cols} FROM nearest, nearest.forecast f WITH OFFSET i"""


def _client():
    from google.cloud import bigquery  # optional extra `weathernext`

    return bigquery.Client(project=PROJECT)


def _config(**kwargs):
    from google.cloud import bigquery  # optional extra `weathernext`

    return bigquery.QueryJobConfig(**kwargs)
