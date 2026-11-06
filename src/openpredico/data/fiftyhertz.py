"""Public 50Hertz wind actuals/forecasts, in MW on a native quarter-hour UTC index.

No API key. Source: https://www.50hertz.com/en/Transparency/GridData/Productiongridfeed-in/Windpower
The website publishes naive German local timestamps in chronological record order.
Keep that order until DST localisation: the repeated autumn hour is not a duplicate.
The documented day-ahead forecast is published at 18:00 D-1: NOT a known-future
input for a 15:00 D-1 gate without evidence of an earlier publication.
"""

import json
import re
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://ds.50hertz.com/api"
TZ = "Europe/Berlin"
KINDS = {"actual": "WindPowerActual", "forecast": "WindPowerForecast"}
COLUMNS = {"Value": "total", "Value2": "onshore", "Value3": "offshore"}
COMPONENTS = {"total", "onshore", "offshore", "sum"}
_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    ),
)


def source_key(key: str) -> tuple[str, str]:
    """Keys: actual_total/onshore/offshore/sum, or forecast_<component>."""
    kind, _, component = key.partition("_")
    if kind not in KINDS or component not in COMPONENTS:
        raise ValueError("50Hertz source must be actual_<component> or forecast_<component>")
    return kind, component


def parse_records(text: str) -> pd.DataFrame:
    """Parse JSON or website JSONP. Missing/sentinel values stay NaN; no truth filling."""
    text = text.strip()
    wrapped = re.fullmatch(r"(?:[\w.]+)?\s*\((.*)\)\s*;?", text, flags=re.DOTALL)
    records = json.loads(wrapped.group(1) if wrapped else text)
    if not isinstance(records, list):
        raise ValueError("Expected a 50Hertz record list")
    if not records:
        return pd.DataFrame(
            columns=[*COLUMNS.values(), "sum"],
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
            dtype=float,
        )
    frame = pd.DataFrame(records)
    required = {"FromTime", *COLUMNS}
    if not required <= set(frame):
        raise ValueError("Missing 50Hertz record fields")
    local = pd.DatetimeIndex(pd.to_datetime(frame.FromTime, errors="raise"))
    # Ambiguous incomplete autumn responses fail rather than guess an hour or fill gaps.
    index = local.tz_localize(TZ, ambiguous="infer", nonexistent="raise").tz_convert("UTC")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Invalid or unordered 50Hertz timestamps")
    if ((index.minute % 15 != 0) | (index.second != 0) | (index.microsecond != 0)).any():
        raise ValueError("50Hertz timestamps must be on the quarter-hour grid")
    values = frame[list(COLUMNS)].apply(pd.to_numeric, errors="coerce").rename(columns=COLUMNS)
    values = values.where(values >= 0)  # source missing markers: -1 and n.a.; zero is valid
    values.index = index.rename("time")
    values["sum"] = values[["onshore", "offshore"]].sum(axis=1, min_count=2)
    return values


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def fetch_frame(kind: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Fetch local days intersecting UTC [start, end). No imputation or persistence.

    ponytail: daily requests suit recent updates/smoke tests; use annual CSV archives
    for multi-year backfills. Retrieval time is NOT the provider's publication time.
    """
    if kind not in KINDS:
        raise ValueError("50Hertz kind must be actual or forecast")
    start, end = _utc(start), _utc(end)
    if start >= end:
        raise ValueError("start must precede end")
    days = pd.date_range(
        start.tz_convert(TZ).normalize(),
        (end - pd.Timedelta("1ns")).tz_convert(TZ).normalize(),
        freq="D",
    )
    frames = []
    for i, day in enumerate(days):
        if i:
            time.sleep(2)
        response = _session.get(
            f"{BASE_URL}/{KINDS[kind]}/ListRecordsJsonp",
            params={"filterDateTime": day.strftime("%Y-%m-%d")},
            timeout=45,
        )
        response.raise_for_status()
        frame = parse_records(response.text)
        if not frame.empty and not (frame.index.tz_convert(TZ).date == day.date()).all():
            raise ValueError("50Hertz response contains another local day")
        frames.append(frame)
    frame = pd.concat(frames)
    if frame.index.has_duplicates:
        raise ValueError("Overlapping 50Hertz day responses")
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    frame.attrs.update(
        source="50Hertz", unit="MW", retrieved_at_utc=pd.Timestamp.now(tz="UTC").isoformat()
    )
    return frame


def fetch(key: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """Return one source column, e.g. actual_sum; preserve the published total separately."""
    kind, component = source_key(key)
    return fetch_frame(kind, start, end)[component].rename(key)
