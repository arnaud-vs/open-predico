"""Spain's mFRR balancing energy bids (ENTSO-E 12.3.B/C) from the ENTSO-E File Library.

The REST API keeps ~100 days and returns 100 bids per request, so history comes from the
File Library (monthly CSV files, Transparency Platform login in ENTSOE_USERNAME / ENTSOE_PWD).
API: https://documenter.getpostman.com/view/28274243/2sB2qgfz3W

    python -m openpredico.data.entsoe_bids download <raw dir> 2024-12 2026-10
    python -m openpredico.data.entsoe_bids ladder <raw dir> <out.parquet>
"""

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_URL = "https://keycloak.tp.entsoe.eu/realms/tp/protocol/openid-connect/token"
FMS = "https://fms.tp.entsoe.eu"
FOLDER_HINT = "12.3.B"  # folder name contains the article number
AREA = "10YES-REE------0"
DEPTHS = (0, 50, 100, 200, 300, 500, 750, 1000, 1500)  # MW; 0 = first bid in merit order
MIN_INTERVAL = 2.0  # seconds between requests (ENTSO-E rate limit)

_last = 0.0
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


def _fms(endpoint: str, body: dict, token: str) -> bytes:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return _request(f"{FMS}/{endpoint}", json.dumps(body).encode(), headers)


def _list(path: str, token: str) -> list[dict]:
    body = {"path": path, "pageInfo": {"pageIndex": 0, "pageSize": 5000}}
    return json.loads(_fms("listFolder", body, token))["contentItemList"]


def download(raw: Path, first: str, last: str) -> None:
    """Download the monthly files from `first` to `last` (YYYY-MM) into `raw`, skipping existing."""
    # ponytail: a token lasts minutes; fine for ~25 monthly files, refresh it if downloads fail.
    token = _token()
    folders = [i["name"] for i in _list("/TP_export/", token) if FOLDER_HINT in i["name"]]
    if len(folders) != 1:
        raise SystemExit(f"expected one folder matching {FOLDER_HINT}, got {folders}")
    folder = f"/TP_export/{folders[0].strip('/')}/"
    months = {p.strftime("%Y_%m") for p in pd.period_range(first, last, freq="M")}
    raw.mkdir(parents=True, exist_ok=True)
    for item in sorted(_list(folder, token), key=lambda i: i["name"]):
        name = item["name"]
        if name[:7] not in months or (raw / name).exists():
            continue
        body = {"fileIdList": [item["fileId"]], "topLevelFolder": "TP_export"}
        content = _fms("downloadFileContent", body, token)
        if content[:2] == b"PK":  # zipped
            content = zipfile.ZipFile(io.BytesIO(content)).read(name)
        (raw / name).write_bytes(content)
        print(f"{name}: {len(content) / 2**20:.0f} MB", flush=True)
    print(f"{calls} requests to ENTSO-E")


# File Library CSV columns and status filter.
# ponytail: UNVERIFIED guesses (no file downloaded yet). Check against the first monthly file,
# incl. product type (standard vs specific) and direct vs scheduled activation, before use.
COLS = {
    "time": "DateTime(UTC)",
    "area": "AreaCode",
    "direction": "Direction",
    "price": "Price",
    "mw": "OfferedQuantity",
    "status": "Status",
}


def read_month(path: Path) -> pd.DataFrame:
    """Spain's available bids in one monthly file: time (UTC), direction (up/down), price, mw."""
    usecols = list(COLS.values())
    chunks = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=1_000_000)
    df = pd.concat(c[c[COLS["area"]] == AREA] for c in chunks)
    df = df.rename(columns={v: k for k, v in COLS.items()})
    df = df[df.status.str.lower().eq("available")]
    df["time"] = pd.to_datetime(df.time, utc=True)
    df["direction"] = df.direction.str.lower().map({"up": "up", "down": "down"})
    return df[["time", "direction", "price", "mw"]].dropna()


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


def build_ladder(raw: Path, out: Path) -> pd.DataFrame:
    """Ladder for every monthly file in `raw` on the 15-min UTC grid (index `time`)."""
    parts = []
    for path in sorted(raw.glob("*.csv")):
        bids = read_month(path)
        # Bids for 15-min products; hourly/other resolutions would be repeated per quarter.
        parts.append(ladder(bids))
        print(f"{path.name}: {len(bids)} bids", flush=True)
    df = pd.concat(parts).sort_index()
    grid = pd.date_range(df.index.min(), df.index.max(), freq="15min", name="time")
    df = df[~df.index.duplicated()].reindex(grid)
    df.to_parquet(out)
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
