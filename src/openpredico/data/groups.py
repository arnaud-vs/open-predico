"""Build feature groups (configs/groups.toml) on the 15-min UTC grid.

Benchmark build: python -m openpredico.data.groups --start 2024-12-10 --end 2026-11-01 --version v1
"""

import argparse
import tomllib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dateutil.easter import easter

from openpredico.config import CONFIGS
from openpredico.data import entsoe_bids, esios, hf

TZ = "Europe/Madrid"
COLUMN_KEYS = {"source", "geo", "fill", "availability"}
AVAILABILITY = {"future", "past"}
PREFIXES = {"esios", "calendar", "entsoe_bids"}
# ponytail: bid ladder built locally by `entsoe_bids ladder` (File Library login needed), not
# fetched here. Put it at this path before building the `mfrr_bids` group.
BIDS_LADDER = CONFIGS.parent / "data" / "bids_ladder.parquet"


def load_groups(path: Path = CONFIGS / "groups.toml") -> dict[str, dict]:
    """Return {group: {column: definition}}, raising ValueError on any invalid entry."""
    groups = tomllib.loads(path.read_text())
    seen_columns, seen_sources = set(), set()
    for group, columns in groups.items():
        for name, c in columns.items():
            where = f"{group}.{name}"
            if c.keys() - COLUMN_KEYS or not {"source", "availability"} <= c.keys():
                raise ValueError(f"{where}: keys must be source, availability, [geo], [fill]")
            if c["availability"] not in AVAILABILITY:
                raise ValueError(f"{where}: availability must be one of {sorted(AVAILABILITY)}")
            if c["source"].split(":")[0] not in PREFIXES:
                raise ValueError(f"{where}: source prefix must be one of {sorted(PREFIXES)}")
            if name in seen_columns:
                raise ValueError(f"{where}: duplicate column name")
            source = (c["source"], c.get("geo"))
            if source in seen_sources:
                raise ValueError(f"{where}: duplicate source {source}")
            seen_columns.add(name)
            seen_sources.add(source)
    return groups


def build(group: str, start: str, end: str) -> pd.DataFrame:
    """One group's columns on the 15-min UTC grid [start, end), with `fill` applied."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="time")
    columns = load_groups()[group]
    raw = {}  # one fetch per ESIOS id, shared by its geos
    df = pd.DataFrame(index=grid)
    for name, c in columns.items():
        prefix, key = c["source"].split(":")
        if prefix == "calendar":
            df[name] = CALENDAR[key](grid)
            continue
        if prefix == "entsoe_bids":
            df[name] = entsoe_bids.to_series(key, BIDS_LADDER).reindex(grid)
            continue
        if key not in raw:
            raw[key] = esios.fetch_values(int(key), start, end)
        s = esios.to_series(raw[key], int(key), c.get("geo"), start, end)
        df[name] = to_grid(s, grid)
    fills = {name: c["fill"] for name, c in columns.items() if "fill" in c}
    return df.fillna(fills)


def to_grid(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Put a native-resolution series on the 15-min grid, repeating hourly values per quarter.

    Points before the local day of the first non-:00 timestamp are hourly (ESIOS series only
    switch from hourly to 15-min, at local midnight); a repeat never overwrites a real value.
    """
    # ponytail: in a sparse 15-min series, :00 points on days before its first non-:00 point
    # are spread over their hour. Declare a switch date per column if that ever matters.
    quarter = s.index[s.index.minute != 0]
    hourly = s[s.index < quarter.min().tz_convert(TZ).normalize()] if len(quarter) else s
    shifted = [hourly.set_axis(hourly.index + pd.Timedelta(minutes=m)) for m in (15, 30, 45)]
    out = pd.concat([s, *shifted])
    return out[~out.index.duplicated()].reindex(grid)


def _local(grid: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return grid.tz_convert(TZ)


def qh_of_day(grid: pd.DatetimeIndex) -> pd.Index:
    local = _local(grid)
    return local.hour * 4 + local.minute // 15


def weekday(grid: pd.DatetimeIndex) -> pd.Index:
    return _local(grid).weekday  # Monday = 0


def holiday_es(grid: pd.DatetimeIndex) -> pd.Index:
    """1 on Spanish national holidays (local date), else 0."""
    # ponytail: national holidays only; regional ones (and moved Sunday holidays) left out.
    local = _local(grid)
    fixed = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]
    days = set()
    for y in range(local.year.min(), local.year.max() + 1):
        days |= {date(y, m, d) for m, d in fixed} | {easter(y) - timedelta(days=2)}
    return pd.Index(local.date).isin(days).astype(int)


def dst_switch_day(grid: pd.DatetimeIndex) -> pd.Index:
    """1 on local days that are 23 h or 25 h long, else 0."""
    day = _local(grid).normalize()
    next_day = (day + pd.Timedelta(hours=26)).normalize()
    return ((next_day - day) != pd.Timedelta(days=1)).astype(int)


CALENDAR = {f.__name__: f for f in (qh_of_day, weekday, holiday_es, dst_switch_day)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build every group and upload it to HF.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--version", required=True)
    args = p.parse_args()
    hf.HfApi().create_repo(f"{hf.ORG}/benchmark", repo_type="dataset", private=False, exist_ok=True)
    for group in load_groups():
        df = build(group, args.start, args.end)
        print(group, df.shape, df.isna().mean().round(3).to_dict(), flush=True)
        hf.upload(
            df,
            "benchmark",
            f"{args.version}/{group}.parquet",
            f"{args.version}: {group} {args.start} to {args.end}",
        )


if __name__ == "__main__":
    main()
