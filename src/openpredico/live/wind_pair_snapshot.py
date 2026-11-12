"""Public, past-only bootstrap snapshots. Never serialize arbitrary runtime directories."""

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from openpredico.live.wind_pair_state import digest, parquet, require_production_assets
from openpredico.models.conformal import TZ, day_grid

ARCHIVE = "wind-pair-clean-seed.zip"
MAX_BYTES = 100 * 1024 * 1024
BASE_FILES = {
    "actuals.parquet",
    "members.parquet",
    "totals.parquet",
    "weather/onshore.parquet",
    "weather/offshore.parquet",
}
MEMBERS = [f"{model}_{part}" for model in ("t0", "timestral") for part in ("onshore", "offshore")]
TOTALS = ["onshore_point_mw", "offshore_point_mw", "point_mw"]
KINDS = {"live", "hindcast", "hindcast_seed"}


class SnapshotError(ValueError):
    """Fixed error codes are safe for status logs; no raw data or credentials."""


def _frame(frame, cutoff):
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise SnapshotError("invalid_utc_index")
    frame = frame.copy()
    frame.index = frame.index.as_unit("ns")
    frame.index.name = "time"
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise SnapshotError("invalid_time_order")
    return frame.loc[frame.index < cutoff]


def _numeric(frame, columns):
    out = frame[columns].astype(float)
    out.attrs = {}
    out.columns.name = None
    if np.isinf(out.to_numpy()).any():
        raise SnapshotError("nonfinite_values")
    return out


def _member_frame(frame):
    out = _numeric(frame, MEMBERS)
    out["issue_time_utc"] = pd.to_datetime(frame.issue_time_utc, utc=True).map(
        lambda value: value.isoformat()
    )
    out["delivery_day"] = frame.index.tz_convert(TZ).strftime("%Y-%m-%d")
    if not frame.kind.isin(KINDS).all():
        raise SnapshotError("unsupported_forecast_kind")
    out["kind"] = frame.kind
    return out


def inspect(data, *, config_hash, now):
    """Independent publication check, also run after GitHub Actions downloads the ZIP."""
    if len(data) > MAX_BYTES:
        raise SnapshotError("archive_too_large")
    today = now.tz_convert(TZ).normalize()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 1000 or sum(x.file_size for x in entries) > 400 * 1024 * 1024:
            raise SnapshotError("archive_expansion_limit")
        if len(set(archive.namelist())) != len(entries):
            raise SnapshotError("duplicate_archive_entry")
        manifest = json.loads(archive.read("manifest.json"))
        allowed_keys = {
            "config_sha256",
            "pipeline",
            "models",
            "truth_source",
            "production_approval",
            "historical_vintages_verified",
            "seed_member_kind",
            "last_member_day",
            "files",
            "snapshot_schema",
            "snapshot_date",
            "cutoff_utc",
            "created_at_utc",
            "attribution",
            "last_complete_truth_day",
            "last_weather_bank_day",
        }
        if set(manifest) != allowed_keys or manifest["snapshot_schema"] != 1:
            raise SnapshotError("unsupported_manifest")
        require_production_assets(manifest)
        if (
            manifest["pipeline"] != "wind_pair_t0_timestral_six_v1"
            or manifest["truth_source"] != "entsoe:A75_B18+B19@10YDE-VE-------2"
            or set(manifest["models"]) != {"t0_revision", "timestral_revision"}
            or not all(re.fullmatch(r"[0-9a-f]{40}", v) for v in manifest["models"].values())
            or not re.fullmatch(
                r"clean-refit-sha256:[0-9a-f]{64}", manifest["production_approval"]["reference"]
            )
            or manifest["historical_vintages_verified"] is not False
        ):
            raise SnapshotError("invalid_public_metadata")
        if manifest["config_sha256"] != config_hash:
            raise SnapshotError("config_mismatch")
        cutoff = pd.Timestamp(manifest["cutoff_utc"])
        date = pd.Timestamp(manifest["snapshot_date"], tz=TZ)
        if cutoff != date.tz_convert("UTC") or date > today or date < today - pd.DateOffset(days=1):
            raise SnapshotError("invalid_snapshot_cutoff")
        if pd.Timestamp(manifest["created_at_utc"]) > now:
            raise SnapshotError("future_creation_time")
        names = set(manifest["files"])
        if not BASE_FILES <= names or set(archive.namelist()) != names | {"manifest.json"}:
            raise SnapshotError("unexpected_archive_files")
        bank_days = set()
        for name in names - BASE_FILES:
            match = re.fullmatch(r"banks/(\d{4}-\d{2}-\d{2})/(onshore|offshore)\.npz", name)
            if not match:
                raise SnapshotError("unexpected_bank_file")
            pd.Timestamp(match[1], tz=TZ)
            bank_days.add(match[1])
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as bank:
                w = bank["speed"]
                if set(bank.files) != {"speed"} or w.shape != (6, 3, 4):
                    raise SnapshotError("invalid_bank_schema")
                if not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(-1), 1):
                    raise SnapshotError("invalid_bank_weights")
        if not bank_days or any(
            f"banks/{day}/{component}.npz" not in names
            for day in bank_days
            for component in ("onshore", "offshore")
        ):
            raise SnapshotError("missing_weather_banks")
        if manifest["last_weather_bank_day"] != max(bank_days):
            raise SnapshotError("bank_day_mismatch")
        for name in names:
            if digest(archive.read(name)) != manifest["files"][name]:
                raise SnapshotError("checksum_mismatch")
        frames = {name: pd.read_parquet(io.BytesIO(archive.read(name))) for name in BASE_FILES}
    for name, frame in frames.items():
        if frame.attrs or frame.columns.name is not None or frame.index.name not in {None, "time"}:
            raise SnapshotError("unexpected_frame_metadata")
        filtered = _frame(frame, cutoff)
        if filtered.empty or len(filtered) != len(frame):
            raise SnapshotError("future_or_empty_frame")
        if name != "members.parquet" and name != "totals.parquet":
            if frame.index.min() > cutoff - pd.Timedelta(days=93):
                raise SnapshotError("insufficient_context_history")
        if name.startswith("weather/"):
            if list(frame.columns) != list(range(96)):
                raise SnapshotError("invalid_weather_schema")
            _numeric(frame, list(range(96)))
    actuals = frames["actuals.parquet"]
    if set(actuals) != {"onshore", "offshore", "total", "received_at"}:
        raise SnapshotError("invalid_actual_schema")
    _numeric(actuals, ["onshore", "offshore", "total"])
    received = pd.to_datetime(actuals.received_at, utc=True, errors="raise")
    if received.isna().any() or not (received > actuals.index).all() or not (received <= now).all():
        raise SnapshotError("invalid_actual_acquisition_time")
    if not np.allclose(actuals.total, actuals.onshore + actuals.offshore, equal_nan=True):
        raise SnapshotError("actual_total_mismatch")
    members, totals = frames["members.parquet"], frames["totals.parquet"]
    if set(members) != set(MEMBERS + ["issue_time_utc", "delivery_day", "kind"]):
        raise SnapshotError("invalid_member_schema")
    if set(totals) != set(TOTALS + ["issue_time_utc", "delivery_day"]):
        raise SnapshotError("invalid_total_schema")
    if not members.index.equals(totals.index) or not members.kind.isin(KINDS).all():
        raise SnapshotError("member_total_mismatch")
    for frame, columns in [(members, MEMBERS), (totals, TOTALS)]:
        values = _numeric(frame, columns).to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise SnapshotError("invalid_forecast_values")
        days = frame.index.tz_convert(TZ).strftime("%Y-%m-%d")
        if not np.array_equal(days, frame.delivery_day.to_numpy()):
            raise SnapshotError("delivery_day_mismatch")
        for day, group in frame.groupby(days):
            if not group.index.equals(day_grid(day)):
                raise SnapshotError("incomplete_forecast_day")
            if not (pd.to_datetime(group.issue_time_utc, utc=True) < group.index.min()).all():
                raise SnapshotError("invalid_forecast_origin")
    yesterday = str((date - pd.DateOffset(days=1)).date())
    if members.delivery_day.max() != yesterday or manifest["last_member_day"] != yesterday:
        raise SnapshotError("stale_forecast_history")
    grid = day_grid(yesterday)
    for name, columns in [
        ("weather/onshore.parquet", list(range(96))),
        ("weather/offshore.parquet", list(range(96))),
    ]:
        if not np.isfinite(frames[name].reindex(grid)[columns].to_numpy(float)).all():
            raise SnapshotError("incomplete_last_day")
    complete_days = [
        day
        for day, group in members.groupby("delivery_day")
        if actuals.total.reindex(group.index).notna().all()
    ]
    if len(complete_days) < 30 or members.delivery_day.nunique() > 60:
        raise SnapshotError("insufficient_calibration_history")
    if max(complete_days) < str((date - pd.DateOffset(days=2)).date()):
        raise SnapshotError("stale_actual_history")
    if manifest["last_complete_truth_day"] != max(complete_days):
        raise SnapshotError("truth_day_mismatch")
    return manifest


def export(root: Path, output: Path, *, cfg: dict, config_hash: str, now: pd.Timestamp):
    """Called only through the serialized Modal worker; never fetches or refits models."""
    date = now.tz_convert(TZ).normalize()
    cutoff = date.tz_convert("UTC")
    day = str(date.date())
    folder = output / day
    path = folder / ARCHIVE
    if path.exists():
        data = path.read_bytes()
        manifest = inspect(data, config_hash=config_hash, now=now)
    else:
        source = json.loads((root / "seed.json").read_text())
        require_production_assets(source)
        if source["config_sha256"] != config_hash:
            raise SnapshotError("installed_config_mismatch")
        reference = source["production_approval"]["reference"]
        if not re.fullmatch(r"clean-refit-sha256:[0-9a-f]{64}", reference):
            raise SnapshotError("unrecognized_clean_provenance")
        files = {}
        start = cutoff - pd.Timedelta(days=120)
        actual_history = None
        for name in ["actuals", "weather/onshore", "weather/offshore"]:
            frame = _frame(pd.read_parquet(root / f"{name}.parquet"), cutoff)
            frame = frame.loc[frame.index >= start]
            if name == "actuals":
                out = _numeric(frame, ["onshore", "offshore", "total"])
                out["received_at"] = pd.to_datetime(frame.received_at, utc=True).map(
                    lambda value: value.isoformat()
                )
            else:
                out = _numeric(frame, list(range(96)))
            if name == "actuals":
                actual_history = out
            files[f"{name}.parquet"] = parquet(out)
        members = _frame(pd.read_parquet(root / "members.parquet"), cutoff)
        dates = sorted(set(members.index.tz_convert(TZ).strftime("%Y-%m-%d")))[-60:]
        members = members.loc[members.index.tz_convert(TZ).strftime("%Y-%m-%d").isin(dates)]
        members = _member_frame(members)
        totals = _frame(pd.read_parquet(root / "totals.parquet"), cutoff).reindex(members.index)
        out = _numeric(totals, TOTALS)
        out["issue_time_utc"] = pd.to_datetime(totals.issue_time_utc, utc=True).map(
            lambda value: value.isoformat()
        )
        out["delivery_day"] = totals.index.tz_convert(TZ).strftime("%Y-%m-%d")
        files["members.parquet"], files["totals.parquet"] = parquet(members), parquet(out)
        bank_paths = sorted((root / "banks").glob("*/*.npz"))
        prior_days = [b.parent.name for b in bank_paths if b.parent.name <= dates[0]]
        anchor = max(prior_days) if prior_days else dates[0]
        for bank in bank_paths:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", bank.parent.name):
                raise SnapshotError("invalid_bank_date")
            if bank.parent.name < anchor:
                continue
            if bank.name not in {"onshore.npz", "offshore.npz"}:
                raise SnapshotError("invalid_bank_name")
            with np.load(bank, allow_pickle=False) as values:
                buf = io.BytesIO()
                np.savez_compressed(buf, speed=values["speed"])
            files[f"banks/{bank.parent.name}/{bank.name}"] = buf.getvalue()
        manifest = {
            "snapshot_schema": 1,
            "snapshot_date": day,
            "cutoff_utc": cutoff.isoformat(),
            "created_at_utc": now.isoformat(),
            "config_sha256": config_hash,
            "pipeline": cfg["pipeline"],
            "models": cfg["models"],
            "truth_source": cfg["truth_source"],
            "production_approval": {"approved": True, "reference": reference},
            "historical_vintages_verified": False,
            "seed_member_kind": "historical_hindcasts_and_past_live_forecasts",
            "last_member_day": members.delivery_day.max(),
            "last_weather_bank_day": max(
                name.split("/")[1] for name in files if name.startswith("banks/")
            ),
            "last_complete_truth_day": max(
                (d for d in dates if actual_history.total.reindex(day_grid(d)).notna().all()),
                default="",
            ),
            "attribution": [
                "Weather data by Open-Meteo https://open-meteo.com/; DWD ICON and NOAA GFS; "
                "CC BY 4.0 https://creativecommons.org/licenses/by/4.0/; modified height, "
                "direction and quarter-hour weather channels, including estimated T100",
                "Generation: ENTSO-E Transparency Platform https://transparency.entsoe.eu/; "
                "50Hertz control area, total is onshore plus offshore",
            ],
            "files": {name: digest(data) for name, data in files.items()},
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in files.items():
                archive.writestr(name, value)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        data = buf.getvalue()
        inspect(data, config_hash=config_hash, now=now)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    public = {
        "schema_version": 1,
        "snapshot_date": day,
        "historical_through": manifest["last_member_day"],
        "last_complete_truth_day": manifest["last_complete_truth_day"],
        "last_weather_bank_day": manifest["last_weather_bank_day"],
        "cutoff_utc": manifest["cutoff_utc"],
        "created_at_utc": manifest["created_at_utc"],
        "config_sha256": config_hash,
        "pipeline": manifest["pipeline"],
        "sha256": digest(data),
        "bytes": len(data),
    }
    (folder / "manifest.json").write_text(json.dumps(public, indent=2) + "\n")
    (folder / f"{ARCHIVE}.sha256").write_text(f"{public['sha256']}  {ARCHIVE}\n")
    return {"status": "exported", **public, "volume_prefix": f"exports/wind-pair/{day}"}
