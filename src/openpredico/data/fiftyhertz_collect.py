"""Collect revision-aware input snapshots; independent of forecasts and submission."""

import json
from pathlib import Path

import pandas as pd

from openpredico.data.fiftyhertz import TZ, fetch_frame


def collect(output: Path, now: pd.Timestamp | None = None) -> dict:
    """Fetch today plus seven prior local days and persist one timestamped snapshot.

    Keep today's incomplete observations; report freshness/gaps instead of filling.
    These small snapshots preserve revisions and can later feed a forecast pipeline.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now.tz_convert("UTC")
    start = now.tz_convert(TZ).normalize() - pd.DateOffset(days=7)
    frame = fetch_frame("actual", start, now)
    if frame.empty:
        raise RuntimeError("50Hertz returned no actuals; snapshot not saved")
    frame = frame.loc[frame.index < now]
    latest = frame[["onshore", "offshore"]].dropna().index.max()
    if pd.isna(latest):
        raise RuntimeError("50Hertz returned no complete wind components")
    stamp = now.strftime("%Y-%m-%dT%H%M%S%fZ")
    folder = output / "snapshots" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": "50Hertz",
        "collected_at_utc": now.isoformat(),
        "request_start_utc": start.tz_convert("UTC").isoformat(),
        "request_end_utc": now.isoformat(),
        "rows": len(frame),
        "latest_complete_interval_start_utc": latest.isoformat(),
        "latest_complete_interval_end_utc": (latest + pd.Timedelta("15min")).isoformat(),
        "publication_lag_minutes": max(
            0, (now - latest - pd.Timedelta("15min")).total_seconds() / 60
        ),
        "missing_by_column": {c: int(n) for c, n in frame.isna().sum().items()},
        "missing_timestamp_slots_between_endpoints": len(
            pd.date_range(frame.index.min(), frame.index.max(), freq="15min").difference(
                frame.index
            )
        ),
    }
    frame.to_parquet(folder / "actuals.parquet")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Snapshot history is authoritative; latest.json is only a convenience pointer.
    (output / "latest.json").write_text(json.dumps({"snapshot": stamp, **manifest}, indent=2))
    return manifest
