"""ENTSO-E Transparency Platform REST API: 50Hertz control-area series on the 15-min UTC grid.

Needs a free API key in the ENTSOE_API_KEY environment variable.
API: https://documenter.getpostman.com/view/7009892/2s93JtP3F6

Keys (`entsoe:<key>` in groups.toml): `<document>_<psr>` with psr codes joined by `+` to sum,
or `<document>` alone for load. A75 = actual generation per production type, A69 = day-ahead
wind and solar forecast, A65 = day-ahead total load forecast; B16 solar, B18 wind offshore,
B19 wind onshore. `A75_B18+B19` is Predico's 50Hertz wind truth.

Predico's rule (50Hertz program page): curveType A03 omits positions that repeat the previous
value, so each series is carried forward on the 15-min grid, then summed. Carrying forward stops
at the last published period: the unpublished tail stays NaN.
"""

import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

BASE_URL = "https://web-api.tp.entsoe.eu/api"
# ponytail: one area, 50Hertz. Make it a per-column setting when another area arrives.
AREA = "10YDE-VE-------2"
PROCESS = {"A75": "A16", "A69": "A01", "A65": "A01"}  # realised, day-ahead, day-ahead
DOMAIN = {"A75": "in_Domain", "A69": "in_Domain", "A65": "outBiddingZone_Domain"}
CHUNK = pd.DateOffset(years=1)  # the API's longest range per request
MIN_INTERVAL = 2.0  # seconds between requests (ENTSO-E rate limit)
STEP = pd.Timedelta("15min")
RETRY_WAITS = (10, 30, 60, 0)  # seconds after each failed try; four tries, the last not waited

_last = 0.0


def fetch(key: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """One `entsoe:` source over UTC [start, end), MW on the 15-min grid; NaN where unpublished."""
    document, _, psrs = key.partition("_")
    start, end = (pd.to_datetime(t, utc=True) for t in (start, end))  # naive = UTC
    grid = pd.date_range(start, end, freq=STEP, inclusive="left", name="time")
    parts = [_fetch_psr(document, psr or None, start, end) for psr in psrs.split("+")]
    if any(p.empty for p in parts):
        return pd.Series(float("nan"), index=grid, name=key)
    # each series is carried forward to its own last period, so the sum stops at the earliest
    return sum(p.reindex(grid) for p in parts).rename(key)


def _fetch_psr(document: str, psr: str | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    parts, t = [], start
    while t < end:
        parts.append(parse(_get(document, psr, t, min(t + CHUNK, end))))
        t += CHUNK
    if not parts:  # empty range, e.g. truth for days that haven't started
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]  # chunk edges may overlap
    if s.empty:
        return s
    # Predico: gaps between periods also repeat the last value, up to the last published one
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq=STEP, name="time")).ffill()


def _get(document: str, psr: str | None, start: pd.Timestamp, end: pd.Timestamp) -> bytes:
    global _last
    time.sleep(max(0.0, _last + MIN_INTERVAL - time.time()))
    _last = time.time()
    params = {
        "securityToken": os.environ["ENTSOE_API_KEY"],
        "documentType": document,
        "processType": PROCESS[document],
        DOMAIN[document]: AREA,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd": end.strftime("%Y%m%d%H%M"),
    }
    if psr:
        params["psrType"] = psr
    for wait in RETRY_WAITS:  # 5xx (599 on 2026-11-09) and 429 are transient on ENTSO-E's side
        resp = requests.get(BASE_URL, params=params, timeout=300)
        if resp.status_code < 500 and resp.status_code != 429:
            break
        time.sleep(wait)
        _last = time.time()
    if resp.status_code == 400 and b"No matching data" in resp.content:
        return b"<x/>"  # nothing published in this range
    if not resp.ok:  # not raise_for_status: its message carries the URL, so the key
        raise RuntimeError(f"ENTSO-E {document} {psr}: HTTP {resp.status_code}")
    return resp.content


def parse(xml: bytes) -> pd.Series:
    """MW per quarter-hour from one market document, every period expanded to the 15-min grid.

    A03 omits positions that repeat the previous value; hourly periods repeat per quarter-hour.
    """
    parts = []
    # ponytail: every series counts, so A75 consumption series (outBiddingZone) would be added
    # in; B16/B18/B19 have none.
    for p in ET.fromstring(xml).iterfind("{*}TimeSeries/{*}Period"):
        start = pd.Timestamp(p.findtext("{*}timeInterval/{*}start"))
        end = pd.Timestamp(p.findtext("{*}timeInterval/{*}end"))
        res = pd.Timedelta(p.findtext("{*}resolution"))
        points = pd.Series(
            {
                int(x.findtext("{*}position")): float(x.findtext("{*}quantity"))
                for x in p.iterfind("{*}Point")
            }
        )
        grid = pd.date_range(start, end, freq=STEP, inclusive="left")
        position = (grid - start) // res + 1  # A03: omitted positions repeat the previous one
        values = points.reindex(range(1, position.max() + 1)).ffill()
        parts.append(pd.Series(values.loc[position].to_numpy(), index=grid))
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    return pd.concat([empty, *parts])
