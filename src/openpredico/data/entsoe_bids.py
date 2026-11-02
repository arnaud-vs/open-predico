"""Spain's mFRR balancing energy bids (ENTSO-E 12.3.B/C) from the ENTSO-E File Library.

The REST API keeps ~100 days and returns 100 bids per request, so history comes from the
File Library (Transparency Platform login in ENTSOE_USERNAME / ENTSOE_PWD).
API: https://documenter.getpostman.com/view/28274243/2sB2qgfz3W

- Archive (`BalancingEnergyBidsArchives_12.3.B_C_r3/<YYYY_MM>/ES/`): one zip per quarter-hour
  holding IEC 62325 ReserveBid_MarketDocument XML files named
  `<Direction>_<Product>_<?>_<Status>_<n>.xml`; fetched in batches of 100 (`archive/*.zip`).
- Current (`BalancingEnergyBids_12.3.B_C_r3.1/`): one all-Europe CSV per day, kept filtered to
  Spain's mFRR rows (`daily/*.csv.gz`).

    python -m openpredico.data.entsoe_bids download <raw dir> 2024-12 2026-05 2026_11_01
    python -m openpredico.data.entsoe_bids ladder <raw dir> <out.parquet>
"""

import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_URL = "https://keycloak.tp.entsoe.eu/realms/tp/protocol/openid-connect/token"
FMS = "https://fms.tp.entsoe.eu"
ARCHIVE = "/TP_export/BalancingEnergyBidsArchives_12.3.B_C_r3/"
CURRENT = "/TP_export/BalancingEnergyBids_12.3.B_C_r3.1/"
AREA = "10YES-REE------0"
DEPTHS = (0, 50, 100, 200, 300, 500, 750, 1000, 1500)  # MW; 0 = first bid in merit order
MIN_INTERVAL = 2.0  # seconds between requests (ENTSO-E rate limit)
START = "2024-12-10"  # Spain joins MARI

_last = 0.0
_token_cache: list[str] = []
calls = 0


def _request(url: str, data: bytes, headers: dict) -> bytes:
    global _last, calls
    time.sleep(max(0.0, _last + MIN_INTERVAL - time.time()))
    _last, calls = time.time(), calls + 1
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:  # errors carry no credentials
        return r.read()


def _token() -> str:
    form = {
        "client_id": "tp-fms-public",
        "grant_type": "password",
        "username": os.environ["ENTSOE_USERNAME"],
        "password": os.environ["ENTSOE_PWD"],
    }
    body = _request(TOKEN_URL, urllib.parse.urlencode(form).encode(), {})
    return json.loads(body)["access_token"]


def _fms(endpoint: str, body: dict) -> bytes:
    for attempt in range(2):  # the token expires after minutes: refresh once
        if attempt or not _token_cache:
            _token_cache[:] = [_token()]
        headers = {"Authorization": f"Bearer {_token_cache[0]}", "Content-Type": "application/json"}
        try:
            return _request(f"{FMS}/{endpoint}", json.dumps(body).encode(), headers)
        except urllib.error.HTTPError as e:
            if attempt or e.code != 401:
                raise


def _list(path: str) -> list[dict]:
    out, page = [], 0
    while True:
        body = {"path": path, "pageInfo": {"pageIndex": page, "pageSize": 5000}}
        items = json.loads(_fms("listFolder", body))["contentItemList"]
        out, page = out + items, page + 1
        if len(items) < 5000:
            return sorted(out, key=lambda i: i["name"])


def _get(file_ids: list[str], as_zip: bool) -> bytes:
    body = {"fileIdList": file_ids, "topLevelFolder": "TP_export", "downloadAsZip": as_zip}
    return _fms("downloadFileContent", body)


def download(raw: Path, first: str, last: str, last_day: str) -> None:
    """Archive months `first`..`last` (YYYY-MM), then daily files up to `last_day` (YYYY_MM_DD).

    Existing files are skipped, so a rerun resumes.
    """
    (raw / "archive").mkdir(parents=True, exist_ok=True)
    (raw / "daily").mkdir(exist_ok=True)
    for m in pd.period_range(first, last, freq="M").strftime("%Y_%m"):
        files = [i for i in _list(f"{ARCHIVE}{m}/ES/") if i["name"].startswith("mFRR")]
        for k in range(0, len(files), 100):  # downloadFileContent takes at most 100 ids
            dest = raw / "archive" / f"{m}_{k // 100:02d}.zip"
            if not dest.exists():
                dest.write_bytes(_get([i["fileId"] for i in files[k : k + 100]], True))
        print(m, len(files), "files", calls, "requests", flush=True)
    for item in _list(CURRENT):
        day = item["name"][:10]
        dest = raw / "daily" / f"{day}.csv.gz"
        if day > last_day or dest.exists():
            continue
        content = _get([item["fileId"]], False)
        if content[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(content))
            content = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(content), sep="\t", dtype=str)
        df[(df.AreaCode == AREA) & (df.ReserveType == "mFRR")].to_csv(dest, index=False)
        print(day, calls, "requests", flush=True)
    print("done", calls, "requests")


DIRECTION = {"A01": "up", "A02": "down"}


def read_xml(xml: bytes) -> pd.DataFrame:
    """Available bids of one ReserveBid_MarketDocument: time (UTC), direction, price, mw.

    Namespace-agnostic: the archive holds both schema versions 7:1 and 7:2.
    """
    rows = []
    for ts in ET.fromstring(xml).iterfind("{*}Bid_TimeSeries"):
        if ts.findtext("{*}status/{*}value") != "A06":  # A06 = available
            continue
        direction = DIRECTION[ts.findtext("{*}flowDirection.direction")]
        for period in ts.iterfind("{*}Period"):
            start = pd.Timestamp(period.findtext("{*}timeInterval/{*}start"))
            step = pd.Timedelta(period.findtext("{*}resolution"))
            for p in period.iterfind("{*}Point"):
                rows.append(
                    (
                        start + (int(p.findtext("{*}position")) - 1) * step,
                        direction,
                        float(p.findtext("{*}energy_Price.amount")),
                        float(p.findtext("{*}quantity.quantity")),
                    )
                )
    return pd.DataFrame(rows, columns=["time", "direction", "price", "mw"])


def read_archive(path: Path) -> pd.DataFrame:
    """Available bids in one archive batch (zip of per-quarter-hour zips of XML files)."""
    parts = []
    with zipfile.ZipFile(path) as outer:
        for name in outer.namelist():
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for xml in inner.namelist():
                    # Skips Unavailable files, and the pre-MARI `Specific_None_None` files
                    # (no status) that run until Spain's go-live on 2024-12-10.
                    if "_Available_" in xml:
                        parts.append(read_xml(inner.read(xml)))
    return pd.concat(parts, ignore_index=True) if parts else read_xml(b"<x/>")


def read_daily(path: Path) -> pd.DataFrame:
    """Available bids in one daily CSV (already Spain + mFRR): time (UTC), direction, price, mw."""
    cols = ["ISP(UTC)", "Direction", "Status", "Price[Currency/MWh]", "Volume[MW]"]
    df = pd.read_csv(path, usecols=cols)
    df = df[df.Status == "Available"]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(df["ISP(UTC)"], utc=True),
            "direction": df.Direction.str.lower(),
            "price": df["Price[Currency/MWh]"],
            "mw": df["Volume[MW]"],
        }
    )


def ladder(bids: pd.DataFrame) -> pd.DataFrame:
    """Marginal price at each cumulative depth (DEPTHS, MW) per quarter-hour and direction.

    Up bids are walked cheapest first, down bids highest price first. A depth that the offered
    volume does not reach is NaN. Also returns the total offered MW per direction.
    """
    df = bids.assign(key=np.where(bids.direction == "up", bids.price, -bids.price))
    df = df.sort_values(["time", "direction", "key"], kind="stable")
    g = ["time", "direction"]
    df["cum"] = df.groupby(g).mw.cumsum()
    cols = {"mw": df.groupby(g).mw.sum()}
    for d in DEPTHS:
        reached = df[df.cum >= d] if d else df
        cols[f"p{d}"] = reached.groupby(g).price.first()
    wide = pd.DataFrame(cols).unstack("direction")
    wide.columns = [f"bid_{direction}_{c}" for c, direction in wide.columns]
    wide["bid_mid_p100"] = (wide.get("bid_up_p100") + wide.get("bid_down_p100")) / 2
    return wide


def build_ladder(raw: Path, out: Path, end: str = "2026-11-02") -> pd.DataFrame:
    """Ladder from every raw file, one file at a time, on the 15-min UTC grid (index `time`).

    Both sources hold Standard mFRR SA and SA & DA bids (both are offered for scheduled
    activation, whose price is the target); linked/complex bids are kept as single bids.
    """
    parts, n = [], 0
    files = sorted((raw / "archive").glob("*.zip")) + sorted((raw / "daily").glob("*.csv.gz"))
    for path in files:
        bids = read_archive(path) if path.suffix == ".zip" else read_daily(path)
        n += len(bids)
        if len(bids):
            parts.append(ladder(bids))
        print(f"{path.name}: {len(bids)} bids", flush=True)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]  # archive and daily overlap on 2026-04-30
    grid = pd.date_range(START, end, freq="15min", inclusive="left", tz="UTC", name="time")
    df = df.reindex(grid)
    df.to_parquet(out)
    print(f"{n} bids, {df.bid_up_p0.notna().sum()} quarter-hours with up bids")
    return df


def to_series(column: str, path: Path) -> pd.Series:
    """One ladder column, for `groups.build` (source = "entsoe_bids:<column>")."""
    return pd.read_parquet(path, columns=[column])[column]


if __name__ == "__main__":
    cmd, raw, *rest = sys.argv[1:]
    if cmd == "download":
        download(Path(raw), *rest)
    else:
        build_ladder(Path(raw), Path(rest[0]))
