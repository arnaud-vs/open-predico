"""ESIOS (Red Eléctrica) API client. Docs: https://api.esios.ree.es/

Needs a free personal token in the ESIOS_TOKEN environment variable.
"""

import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://api.esios.ree.es"
CHUNK = pd.DateOffset(months=1)  # ponytail: monthly requests; widen if ESIOS allows longer ranges

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    ),
)


def fetch(indicator: int, start: str, end: str, geo_id: int | None = None) -> pd.Series:
    """Values of one indicator in [start, end), indexed by UTC timestamp, at native resolution.

    No time_trunc is sent on purpose: ESIOS would aggregate with `sum` by default, which is
    wrong for prices. Resample downstream with the right aggregation.
    """
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return to_series(fetch_values(indicator, start, end), indicator, geo_id, start, end)


def fetch_values(indicator: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Raw ESIOS `values` entries in [start, end), all geos; pass them to `to_series`."""
    values = []
    t = start
    while t < end:
        values += _get(indicator, t, min(t + CHUNK, end))
        t += CHUNK
        time.sleep(1)  # ponytail: crude rate limit; ESIOS's WAF blocks bursts of requests
    return values


def _get(indicator: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    resp = _session.get(
        f"{BASE_URL}/indicators/{indicator}",
        params={"start_date": start.isoformat(), "end_date": end.isoformat()},
        headers={
            "Accept": "application/json; application/vnd.esios-api-v1+json",
            "Content-Type": "application/json",
            "x-api-key": os.environ["ESIOS_TOKEN"],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["indicator"]["values"]


def to_series(
    values: list[dict], indicator: int, geo_id: int | None, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Turn ESIOS `values` entries into a UTC-indexed series for a single geo."""
    df = pd.DataFrame(values, columns=["value", "datetime_utc", "geo_id"])
    geos = df["geo_id"].unique()
    if geo_id is None and len(geos) > 1:
        raise ValueError(f"indicator {indicator} has several geos {sorted(geos)}; pass geo_id")
    if geo_id is not None:
        df = df[df["geo_id"] == geo_id]
    s = pd.Series(
        df["value"].to_numpy(dtype=float),
        index=pd.to_datetime(df["datetime_utc"], utc=True),
        name=indicator,
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()  # chunk edges may overlap
    return s[(s.index >= start) & (s.index < end)]
