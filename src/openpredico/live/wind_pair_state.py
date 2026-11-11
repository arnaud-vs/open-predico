"""Private wind-pair state and immutable seed install; no credentials in stored records."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parquet(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf)
    return buf.getvalue()


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)


def require_production_assets(manifest: dict) -> None:
    """Maintainer-reviewed rights cover banks AND forecasts/calibration derived from them."""
    approval = manifest.get("production_approval", {})
    if approval.get("approved") is not True or not approval.get("reference", "").strip():
        raise ValueError("research seed is not approved for production; review asset rights")


def install(seed: bytes, root: Path, config_hash: str) -> None:
    """Seed data stays out of git. Refuse traversal, checksum errors and state overwrite."""
    if (root / "seed.json").exists():
        require_production_assets(json.loads((root / "seed.json").read_text()))
        if json.loads((root / "seed.json").read_text())["seed_sha256"] != digest(seed):
            raise ValueError("different seed already installed; use a new state volume")
        return
    with zipfile.ZipFile(io.BytesIO(seed)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        require_production_assets(manifest)
        if manifest["config_sha256"] != config_hash:
            raise ValueError("seed/config mismatch")
        for name, expected in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe seed path")
            data = archive.read(name)
            if digest(data) != expected:
                raise ValueError("seed checksum mismatch")
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    root.mkdir(parents=True, exist_ok=True)
    manifest["seed_sha256"] = digest(seed)
    tmp = root / "seed.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(root / "seed.json")


def append(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([old, new]).sort_index(kind="stable")
    return out[~out.index.duplicated(keep="last")]


def raw(frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    return frame.reindex(index).to_numpy(float).reshape(len(index), 6, 4, 4)


def bank(root: Path, day: str, component: str):
    candidates = sorted(p for p in (root / "banks").glob("*") if p.name <= day)
    if not candidates:
        raise ValueError("no approved pre-issue weather bank")
    path = candidates[-1] / f"{component}.npz"
    with np.load(path, allow_pickle=False) as data:
        weights = data["speed"].copy()
    return weights, {"bank_day": candidates[-1].name, "bank_sha256": digest(path.read_bytes())}


def compact(root: Path):
    """Keep sufficient state for a 92-day context and seven-day automatic catchup."""
    cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=120)
    paths = [root / f"{name}.parquet" for name in ("actuals", "members", "totals")]
    paths += [root / "weather" / f"{component}.parquet" for component in ("onshore", "offshore")]
    for path in paths:
        frame = pd.read_parquet(path)
        write_frame(path, frame[frame.index >= cutoff])
    for folder, pattern in [("runs", "*.json"), ("pending", "*.zip"), ("ready", "*")]:
        for path in (root / folder).glob(pattern):
            if path.is_file() and path.stem < cutoff.tz_convert("Europe/Berlin").strftime(
                "%Y-%m-%d"
            ):
                path.unlink()
