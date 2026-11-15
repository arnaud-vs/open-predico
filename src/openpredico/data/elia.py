"""Elia Open Data (opendata.elia.be): Belgian solar, Belgium total, MW per quarter-hour.

Keys (`elia:solar.<field>` in groups.toml): one field stitched across Elia's solar datasets, as
Predico's truth for belgium_solar_opendata_elia is (raw measurements matched exactly, 2026-11-15):
ods032 `measured` (region Belgium) up to 2026-10-25, then ods209 `measured_upscaled` (historical,
daily, gridconnectiontype Total = TSO + DSO), with ods210 (near-real-time, Total) for the hours
ods209 does not have yet. Forecast fields (`dayahead11hforecast`, ...) carry the same names in
all three. One bulk export request per dataset, no key needed.
"""

import pandas as pd
import requests

BASE_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
TOTAL = "region = 'Belgium' and gridconnectiontype = 'Total'"
# (dataset, row filter, field renames), highest priority first
DATASETS = [
    ("ods209", TOTAL, {"measured": "measured_upscaled"}),
    ("ods210", TOTAL, {}),
    ("ods032", "region = 'Belgium'", {}),
]
SWITCH = pd.Timestamp("2026-10-25 23:00", tz="UTC")  # ods032 ends, ods209/ods210 begin


def fetch(key: str, start, end) -> pd.Series:
    """One field on the 15-min UTC grid [start, end); missing quarter-hours NaN."""
    family, field = key.split(".")
    if family != "solar":
        raise ValueError(f"unknown Elia key {key!r}")
    start, end = (pd.Timestamp(t) for t in (start, end))
    start, end = (t.tz_localize("UTC") if t.tz is None else t for t in (start, end))
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="time")
    out = pd.Series(float("nan"), index=grid)
    for dataset, where, rename in DATASETS:
        old = dataset == "ods032"
        lo, hi = (start, min(end, SWITCH)) if old else (max(start, SWITCH), end)
        if lo < hi:
            out = out.combine_first(_get(dataset, rename.get(field, field), where, lo, hi))
    return out.reindex(grid).rename(None)


def _get(dataset: str, field: str, where: str, start, end) -> pd.Series:
    where += f" and datetime >= date'{start:%Y-%m-%dT%H:%M:%SZ}'"
    where += f" and datetime < date'{end:%Y-%m-%dT%H:%M:%SZ}'"
    resp = requests.get(
        f"{BASE_URL}/{dataset}/exports/json",
        params={"select": f"datetime, {field}", "where": where},
        timeout=300,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json(), columns=["datetime", field])
    s = pd.Series(df[field].astype(float).values, index=pd.to_datetime(df.datetime, utc=True))
    return s[~s.index.duplicated()].dropna()
