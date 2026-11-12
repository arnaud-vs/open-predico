"""Build the private bootstrap artifact from reviewed research caches; never write data in git.

Usage: python tools/build_wind_pair_seed.py --research-root /path/to/outputs
       --output /secure/seed.zip
The operator may instead use the already prepared immutable artifact and its checksum.
"""

import argparse
import hashlib
import io
import json
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def parquet(frame):
    buf = io.BytesIO()
    frame.to_parquet(buf)
    return buf.getvalue()


def verified_clean_run(path, cfg):
    manifest_data = (path / "asset_manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    provenance = manifest["provenance"]
    if (
        provenance.get("timesfm_outputs_used_for_fit_or_forecast") is not False
        or provenance.get("models")
        != {
            "t0": cfg["models"]["t0_revision"],
            "timestral": cfg["models"]["timestral_revision"],
        }
        or provenance.get("weather_order") != cfg["weather_models"]
        or manifest.get("members_days") != 104
    ):
        raise ValueError("unverified replacement weather assets")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe clean asset path")
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError("clean asset checksum mismatch")
    return {
        "approved": True,
        "reference": "clean-refit-sha256:" + hashlib.sha256(manifest_data).hexdigest(),
        "scope": "Reviewed clean weather-bank and dependent forecast provenance; "
        "operator must still verify model/data terms and cloud execution.",
    }, manifest_data


def build(research, output, clean_run=None):
    if output.resolve().is_relative_to(REPO.resolve()):
        raise ValueError("keep bootstrap data outside the git checkout")
    cfg_data = (REPO / "configs/live/wind_pair.toml").read_bytes()
    cfg = tomllib.loads(cfg_data.decode())
    files = {}
    approval = {
        "approved": False,
        "reference": "",
        "reason": "Research banks fitted using TimesFM3 outputs; production rights unresolved. "
        "Rebuild banks and all dependent member/total/calibration forecasts without those "
        "outputs, or obtain permission covering their use before maintainer approval.",
    }
    if clean_run:
        approval, provenance = verified_clean_run(clean_run, cfg)
        files["clean-refit-manifest.json"] = provenance
    on = research / "predico_mock_50hertz_wind_regimes_2026"
    off = research / "predico_mock_50hertz_offshore_blend_t100_2026"
    for component, source in [("onshore", on), ("offshore", off)]:
        with np.load(source / "history_weather.npz", allow_pickle=False) as data:
            raw = data["raw"].copy()
            index = pd.to_datetime(data["times"], utc=True)
        if component == "onshore":
            temp = pd.read_parquet(
                research
                / "predico_mock_50hertz_temperature100_2026"
                / "temperature_history.parquet"
            ).reindex(index)
            raw[:, :, :, 3] = temp.to_numpy()[:, :, None]
        files[f"weather/{component}.parquet"] = parquet(
            pd.DataFrame(raw.reshape(len(index), 96), index=index)
        )
        bank_paths = (
            sorted((clean_run / "banks").glob(f"*/{component}.npz"))
            if clean_run
            else sorted((source / "banks").glob("*.npz"))
        )
        for bank in bank_paths:
            with np.load(bank, allow_pickle=False) as values:
                speed = values["speed"]
                if speed.shape != (6, 3, 4) or not np.isfinite(speed).all():
                    raise ValueError("invalid research bank")
                buf = io.BytesIO()
                np.savez_compressed(buf, speed=speed)
                day = bank.parent.name if clean_run else bank.stem
                files[f"banks/{day}/{component}.npz"] = buf.getvalue()
    on_points = json.loads(
        (research / "predico_mock_50hertz_onshore6_2026/config.json").read_text()
    )["onshore_points"]
    off_points = [
        [p["latitude"], p["longitude"]] for p in json.loads((off / "points.json").read_text())
    ]
    if cfg["onshore_points"] != on_points or cfg["offshore_points"] != off_points:
        raise ValueError("research/config points differ")
    if clean_run:
        members = pd.read_parquet(clean_run / "members.parquet")
        totals = pd.read_parquet(clean_run / "totals.parquet")
        totals = totals[
            ["onshore_point_mw", "offshore_point_mw", "point_mw", "issue_time_utc", "delivery_day"]
        ]
        files["members.parquet"] = parquet(members)
        files["totals.parquet"] = parquet(totals)
    else:
        members = None
        for model in ("t0", "timestral"):
            for component in ("onshore", "offshore"):
                source = (
                    research
                    / "predico_mock_50hertz_ensemble_blend_t100_2026/components/t0_gamma"
                    / component
                    if model == "t0"
                    else research
                    / "predico_mock_50hertz_timestral_blend_t100_2026/predictions"
                    / component
                )
                frame = pd.concat(
                    [pd.read_parquet(p) for p in sorted(source.glob("*.parquet"))]
                ).sort_index()
                if not frame.checkpoint_revision.eq(cfg["models"][f"{model}_revision"]).all():
                    raise ValueError("research checkpoint mismatch")
                if members is None:
                    members = frame[["issue_time_utc", "delivery_day"]].copy()
                else:
                    pd.testing.assert_index_equal(members.index, frame.index)
                if not np.isfinite(frame.q50).all():
                    raise ValueError("nonfinite research member")
                members[f"{model}_{component}"] = frame.q50
        members["kind"] = "hindcast_seed"
        files["members.parquet"] = parquet(members)
        totals = pd.read_parquet(
            research
            / "predico_mock_50hertz_ensemble_blend_t100_2026"
            / "leave_one_out/without_chronos2.parquet"
        )
        totals = totals[
            ["onshore_point_mw", "offshore_point_mw", "q50", "issue_time_utc", "delivery_day"]
        ]
        totals = totals.rename(columns={"q50": "point_mw"})
        totals["point_mw"] = totals.point_mw.clip(0, cfg["capacity_mw"])
        files["totals.parquet"] = parquet(totals)
    official = pd.read_parquet(
        research
        / "predico_mock_50hertz_timestral_blend_t100_2026"
        / "entsoe_benchmark_truth.parquet"
    )
    actuals = official[["wind_50hertz_onshore", "wind_50hertz_offshore", "wind_50hertz"]].rename(
        columns={
            "wind_50hertz_onshore": "onshore",
            "wind_50hertz_offshore": "offshore",
            "wind_50hertz": "total",
        }
    )
    actuals["received_at"] = datetime.now(UTC).isoformat()
    files["actuals.parquet"] = parquet(actuals)
    manifest = {
        "config_sha256": hashlib.sha256(cfg_data).hexdigest(),
        "pipeline": cfg["pipeline"],
        "models": cfg["models"],
        "truth_source": cfg["truth_source"],
        "production_approval": approval,
        "historical_vintages_verified": False,
        "seed_member_kind": "hindcast_seed",
        "last_member_day": members.delivery_day.max(),
        "attribution": ["ENTSO-E Transparency Platform", "Open-Meteo, DWD ICON and NOAA GFS"],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"Private seed prepared: {len(files)} files; SHA256 {checksum}")  # no forecast values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clean-run", type=Path, help="Checksummed clean refit with regenerated forecasts"
    )
    args = parser.parse_args()
    build(args.research_root, args.output, args.clean_run)
