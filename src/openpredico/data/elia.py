"""Elia Open Data (opendata.elia.be): Belgian solar, Belgium total, MW per quarter-hour.

Keys (`elia:<dataset>.<field>` in groups.toml), e.g. `ods032.measured`: Predico's truth for
belgium_solar_opendata_elia. One bulk export request per series, no key needed.
"""

import pandas as pd
import requests

BASE_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
# ponytail: ods032 (historical) stops at 2026-10-25; Elia moved solar to ods209 (historical,
# `measured_upscaled`) and ods210 (near-real-time, `measured`), split by gridconnectiontype
# (Total/TSO/DSO). Add those before the live run.
WHERE = {"ods032": "region = 'Belgium'"}


def fetch(key: str, start, end) -> pd.Series:
    """One field of a dataset on the 15-min UTC grid [start, end); missing quarter-hours NaN."""
    dataset, field = key.split(".")
    start, end = (pd.Timestamp(t) for t in (start, end))
    start, end = (t.tz_localize("UTC") if t.tz is None else t for t in (start, end))
    where = f"{WHERE[dataset]} and datetime >= date'{start:%Y-%m-%dT%H:%M:%SZ}'"
    where += f" and datetime < date'{end:%Y-%m-%dT%H:%M:%SZ}'"
    resp = requests.get(
        f"{BASE_URL}/{dataset}/exports/json",
        params={"select": f"datetime, {field}", "where": where},
        timeout=300,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json(), columns=["datetime", field])
    s = df.set_index(pd.to_datetime(df.datetime, utc=True))[field].astype(float)
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="time")
    return s[~s.index.duplicated()].reindex(grid)
