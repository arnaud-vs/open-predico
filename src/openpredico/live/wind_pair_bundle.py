"""Atomic private candidate bundle, independently checked before incumbent replacement."""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

from openpredico.live.wind_pair_state import digest, parquet
from openpredico.models.conformal import TZ, day_grid

TARGET = "germany_wind_50hertz_ca"
REPO = "open-predico/live-private"
PREFIX = "wind-pair/v1"


def encode(forecast, inputs, calibration, audit, raw_weather, fit_members) -> bytes:
    files = {
        "forecast.parquet": parquet(forecast),
        "inputs.parquet": parquet(inputs),
        "calibration.parquet": parquet(calibration),
        "fit_members.parquet": parquet(fit_members),
        "raw_weather.json": json.dumps(raw_weather).encode(),
    }
    audit = dict(audit, files={name: digest(data) for name, data in files.items()})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("audit.json", json.dumps(audit))
    return buf.getvalue()


def decode(data, *, now, config_hash, capacity):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        audit = json.loads(archive.read("audit.json"))
        if audit["target"] != TARGET or audit["config_sha256"] != config_hash:
            raise ValueError("candidate identity/config mismatch")
        day = str((now.tz_convert(TZ).normalize() + pd.DateOffset(days=1)).date())
        origin = pd.Timestamp(audit["issue_time_utc"])
        previous = (pd.Timestamp(day) - pd.DateOffset(days=1)).date()
        expected = pd.Timestamp(f"{previous} 13:30", tz=TZ).tz_convert("UTC")
        if audit["day"] != day or origin != expected or not origin <= now:
            raise ValueError("candidate date/issue mismatch")
        created = pd.Timestamp(audit["created_at_utc"])
        latest = pd.Timestamp(audit["latest_input_received_at_utc"])
        cutoff = pd.Timestamp(f"{previous} 14:00", tz=TZ).tz_convert("UTC")
        if created > cutoff or latest >= origin or created > now:
            raise ValueError("candidate late or inputs after issue")
        if audit["calibration_days"] < 30 or audit["pipeline"] != "wind_pair_t0_timestral_six_v1":
            raise ValueError("invalid candidate calibration/pipeline")
        required = {
            "forecast.parquet",
            "inputs.parquet",
            "calibration.parquet",
            "fit_members.parquet",
            "raw_weather.json",
        }
        if set(audit["files"]) != required:
            raise ValueError("candidate manifest incomplete")
        if pd.Timestamp(audit["latest_actual_available_at"]) >= origin:
            raise ValueError("calibration actuals after issue")
        gate = pd.Timestamp(f"{previous} 15:00", tz=TZ).tz_convert("UTC")
        if now >= gate:
            raise ValueError("candidate gate already closed")
        for name, expected_hash in audit["files"].items():
            if digest(archive.read(name)) != expected_hash:
                raise ValueError("candidate checksum mismatch")
        fc = pd.read_parquet(io.BytesIO(archive.read("forecast.parquet")))
        inputs = pd.read_parquet(io.BytesIO(archive.read("inputs.parquet")))
    actual_columns = [c for c in inputs if c.startswith("wind_pair.") and c.endswith(".actual")]
    if (
        not actual_columns
        or inputs.loc[inputs.index >= origin - pd.Timedelta(hours=2), actual_columns]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("stored model inputs contain future actuals")
    if not fc.index.equals(day_grid(day)):
        raise ValueError("candidate grid incomplete")
    values = fc[["q10", "q50", "q90"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > capacity).any():
        raise ValueError("candidate bounds invalid")
    if (np.diff(values, axis=1) < 0).any() or not fc.model.eq(audit["pipeline"]).all():
        raise ValueError("candidate quantiles/model invalid")
    return fc, inputs, audit


def upload(data: bytes, day: str) -> None:
    api = HfApi()
    if not api.repo_info(REPO, repo_type="dataset").private:
        raise ValueError("candidate storage must be private")
    try:
        api.upload_file(
            path_or_fileobj=data,
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=f"{PREFIX}/candidates/{day}.zip",
            commit_message=f"Ready private wind pair {day}",
        )
    except Exception:
        raise RuntimeError("private wind-pair candidate upload failed") from None


def load(now, config_hash, capacity):
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", timeout=5)
    if not info.private:
        raise ValueError("candidate storage must be private")
    day = str((now.tz_convert(TZ).normalize() + pd.DateOffset(days=1)).date())
    token = get_token()
    url = hf_hub_url(REPO, f"{PREFIX}/candidates/{day}.zip", repo_type="dataset", revision=info.sha)
    try:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=(5, 8)
        )
    except requests.RequestException:
        raise RuntimeError("private wind candidate download failed") from None
    if response.status_code != 200:
        raise RuntimeError(f"private wind candidate HTTP {response.status_code}")
    return decode(response.content, now=now, config_hash=config_hash, capacity=capacity)


def backup_seed(seed: bytes):
    api = HfApi()
    if not api.repo_info(REPO, repo_type="dataset", timeout=5).private:
        raise ValueError("seed backup must be private")
    try:
        api.upload_file(
            path_or_fileobj=seed,
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=f"{PREFIX}/bootstrap/{digest(seed)}.zip",
            commit_message="Private wind-pair bootstrap",
        )
    except Exception:
        raise RuntimeError("private seed backup failed") from None
