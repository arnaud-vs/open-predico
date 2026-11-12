import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from openpredico.live import wind_pair as producer
from openpredico.live import wind_pair_bundle as bundle
from openpredico.live import wind_pair_consumer as consumer
from openpredico.live import wind_pair_state as state
from openpredico.models import wind_pair as model
from openpredico.models.conformal import day_grid


@pytest.fixture
def case():
    day = "2026-10-25"
    origin = producer.origin_for(day)
    cfg, sha = producer.config()
    idx = pd.date_range(origin - pd.Timedelta(days=92), day_grid(day)[-1], freq="15min")
    inputs = pd.DataFrame(
        {
            "wind_pair.onshore.actual": 60.0,
            "wind_pair.offshore.actual": 40.0,
            "wind_pair.onshore.w0": 5.0,
        },
        index=idx,
    )
    inputs.loc[
        inputs.index >= origin - pd.Timedelta(hours=2),
        ["wind_pair.onshore.actual", "wind_pair.offshore.actual"],
    ] = np.nan
    totals = pd.DataFrame(
        {"onshore_point_mw": 60.0, "offshore_point_mw": 40.0, "point_mw": 100.0}, index=idx
    )
    dates = idx.tz_convert("Europe/Berlin").strftime("%Y-%m-%d")
    totals["delivery_day"] = dates
    totals["issue_time_utc"] = [producer.origin_for(d).isoformat() for d in dates]
    members = pd.DataFrame(
        {
            f"{m}_{c}": 60.0 if c == "onshore" else 40.0
            for m in model.MEMBERS
            for c in ("onshore", "offshore")
        },
        index=idx,
    )
    members["delivery_day"] = dates
    members["issue_time_utc"] = totals.issue_time_utc
    members["kind"] = "hindcast_seed"
    actuals = pd.DataFrame(
        {
            "onshore": 60.0,
            "offshore": 40.0,
            "total": 100.0,
            "received_at": (origin - pd.Timedelta(minutes=20)).isoformat(),
        },
        index=idx,
    )
    actuals = actuals[actuals.index < origin - pd.Timedelta(hours=1)]
    forecast = pd.DataFrame(
        {"q10": 80.0, "q50": 100.0, "q90": 120.0, "model": producer.PIPELINE}, index=day_grid(day)
    )
    audit = dict(
        target=bundle.TARGET,
        pipeline=producer.PIPELINE,
        day=day,
        config_sha256=sha,
        issue_time_utc=origin.isoformat(),
        created_at_utc=(origin + pd.Timedelta(minutes=5)).isoformat(),
        latest_input_received_at_utc=(origin - pd.Timedelta(minutes=20)).isoformat(),
        latest_actual_available_at=(origin - pd.Timedelta(minutes=20)).isoformat(),
        calibration_days=60,
    )
    data = bundle.encode(forecast, inputs, totals, audit, [], members)
    return dict(
        day=day,
        origin=origin,
        now=origin + pd.Timedelta(minutes=45),
        cfg=cfg,
        sha=sha,
        inputs=inputs,
        totals=totals,
        members=members,
        actuals=actuals,
        forecast=forecast,
        audit=audit,
        data=data,
    )


def test_valid_dst_bundle_and_consumer_tail(case, monkeypatch):
    fc, inputs, audit = bundle.decode(
        case["data"], now=case["now"], config_hash=case["sha"], capacity=24500
    )
    assert len(fc) == 100
    monkeypatch.setattr(bundle, "load", lambda *args: (fc, inputs, audit))
    out, source = consumer.ready(bundle.TARGET, case["now"])
    assert (
        out.iloc[:100].q50.eq(100).all() and out.iloc[100:].model.eq("slot_climatology_28d").all()
    )
    assert "wind_50hertz" in source
    from openpredico.live.submit import validate

    validate(out, case["now"])


@pytest.mark.parametrize(
    "problem", ["stale", "after_issue", "late", "calibration", "config", "crossed", "future_truth"]
)
def test_candidate_rejects_unsafe_handoffs(case, problem):
    a, f, inputs = case["audit"].copy(), case["forecast"].copy(), case["inputs"].copy()
    sha = case["sha"]
    if problem == "stale":
        a["day"] = "2026-10-24"
    if problem == "after_issue":
        a["latest_input_received_at_utc"] = case["origin"].isoformat()
    if problem == "late":
        a["created_at_utc"] = (case["origin"] + pd.Timedelta(minutes=31)).isoformat()
    if problem == "calibration":
        a["calibration_days"] = 29
    if problem == "config":
        sha = "another-config"
    if problem == "crossed":
        f["q10"] = 101.0
    if problem == "future_truth":
        inputs["wind_pair.onshore.actual"] = 1.0
    data = bundle.encode(f, inputs, case["totals"], a, [], case["members"])
    with pytest.raises(ValueError):
        bundle.decode(data, now=case["now"], config_hash=sha, capacity=24500)


def test_checksum_and_closed_gate(case):
    with zipfile.ZipFile(io.BytesIO(case["data"])) as source:
        files = {n: source.read(n) for n in source.namelist()}
    files["raw_weather.json"] = b"changed"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for n, content in files.items():
            archive.writestr(n, content)
    with pytest.raises(ValueError, match="checksum"):
        bundle.decode(buf.getvalue(), now=case["now"], config_hash=case["sha"], capacity=24500)
    with pytest.raises(ValueError, match="closed"):
        bundle.decode(
            case["data"],
            now=case["origin"] + pd.Timedelta(minutes=90),
            config_hash=case["sha"],
            capacity=24500,
        )


def test_other_targets_and_unavailable_candidate_keep_incumbent(case, monkeypatch):
    def fail(*args):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(bundle, "load", fail)
    assert consumer.ready("germany_solar_50hertz_ca", case["now"]) is None
    assert consumer.ready("spain_mfrr_energy", case["now"]) is None
    assert consumer.ready(bundle.TARGET, case["now"]) is None


def test_no_source_fetch_after_issue_and_atomic_upload_retry(case, monkeypatch, tmp_path):
    (tmp_path / "seed.json").write_text(
        json.dumps(
            {
                "config_sha256": case["sha"],
                "production_approval": {"approved": True, "reference": "synthetic test assets"},
            }
        )
    )
    state.write_frame(
        tmp_path / "members.parquet",
        case["members"][case["members"].index < day_grid(case["day"])[0]],
    )
    counters = {"collect": 0, "upload": 0}

    def collect(*args):
        counters["collect"] += 1
        return [{"retrieved_at_utc": (case["origin"] - pd.Timedelta(minutes=20)).isoformat()}]

    monkeypatch.setattr(producer, "_collect", collect)
    monkeypatch.setattr(producer, "_actuals", lambda *a: case["actuals"])
    monkeypatch.setattr(producer, "_new_points", lambda *a, **kw: case["members"])
    monkeypatch.setattr(producer, "_new_totals", lambda *a, **kw: case["totals"])
    wx = case["inputs"][["wind_pair.onshore.w0"]]
    monkeypatch.setattr(
        producer,
        "_features",
        lambda *a: (wx.rename(columns={wx.columns[0]: f"wind_pair.{a[2]}.w0"}), {}),
    )

    def upload(*args):
        counters["upload"] += 1
        if counters["upload"] == 1:
            raise RuntimeError("transient upload failure")

    monkeypatch.setattr(bundle, "upload", upload)
    clock = iter(
        [case["origin"] - pd.Timedelta(minutes=30), case["origin"] - pd.Timedelta(minutes=10)]
    )
    with pytest.raises(RuntimeError, match="transient"):
        producer.produce(tmp_path, clock=lambda: next(clock))
    before = (tmp_path / "pending" / f"{case['day']}.zip").read_bytes()
    result = producer.produce(tmp_path, clock=lambda: case["origin"] + pd.Timedelta(minutes=5))
    assert result["upload_retry"] and counters == {"collect": 1, "upload": 2}
    assert before == (tmp_path / "pending" / f"{case['day']}.zip").read_bytes()
    assert producer.produce(tmp_path, clock=lambda: case["now"])["status"] == "already_ready"


def test_late_collection_fails_before_any_source_call(case, tmp_path):
    (tmp_path / "seed.json").write_text(
        json.dumps(
            {
                "config_sha256": case["sha"],
                "production_approval": {"approved": True, "reference": "synthetic test assets"},
            }
        )
    )
    with pytest.raises(ValueError, match="after frozen"):
        producer.produce(tmp_path, clock=lambda: case["origin"])


def test_upgrade_retry_only_for_fallback_wind(case, monkeypatch):
    from openpredico.data import hf

    files = [f"forecasts/{bundle.TARGET}/{case['now']:%Y-%m-%d}T1200Z.parquet"]
    monkeypatch.setattr(hf, "download", lambda *a: pd.DataFrame({"model": ["t0gamma_cov_ctx90"]}))
    monkeypatch.setattr(consumer, "ready", lambda *a: (case["forecast"], case["inputs"]))
    assert consumer.needs_retry(case["now"], files)
    monkeypatch.setattr(hf, "download", lambda *a: pd.DataFrame({"model": [producer.PIPELINE]}))
    assert not consumer.needs_retry(case["now"], files)
    assert not consumer.needs_retry(case["origin"] + pd.Timedelta(hours=3), files)


def test_main_prefers_pair_without_loading_incumbent_and_falls_back(case, monkeypatch):
    from openpredico.live import submit

    class Clock:
        @staticmethod
        def now(**kwargs):
            return case["now"]

    monkeypatch.setattr(submit.pd, "Timestamp", Clock)
    monkeypatch.setattr(submit, "ready_wind_pair", lambda *a: (case["forecast"], case["inputs"]))
    calls = []
    monkeypatch.setattr(submit.t0, "load", lambda *a: calls.append("load"))
    submit.main(dry_run=True, only=[bundle.TARGET])
    assert calls == []
    monkeypatch.setattr(submit, "ready_wind_pair", lambda *a: None)
    monkeypatch.setattr(submit.t0, "make_predict", lambda *a: object())
    monkeypatch.setattr(submit, "build_frames", lambda *a: {})
    monkeypatch.setattr(submit, "forecast", lambda *a: (case["forecast"], case["inputs"]))
    submit.main(dry_run=True, only=[bundle.TARGET])
    assert calls == ["load"]


def test_seed_checksums_traversal_and_state_protection(tmp_path):
    def seed(name="data.parquet", value=b"data", expected=None):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr(name, value)
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "config_sha256": "cfg",
                        "production_approval": {
                            "approved": True,
                            "reference": "synthetic test assets",
                        },
                        "files": {name: expected or state.digest(value)},
                    }
                ),
            )
        return buf.getvalue()

    with pytest.raises(ValueError, match="unsafe"):
        state.install(seed("../outside"), tmp_path, "cfg")
    with pytest.raises(ValueError, match="checksum"):
        state.install(seed(expected="wrong"), tmp_path, "cfg")
    assert not (tmp_path / "seed.json").exists()
    original = seed()
    state.install(original, tmp_path, "cfg")
    state.install(original, tmp_path, "cfg")
    with pytest.raises(ValueError, match="different seed"):
        state.install(seed(value=b"new"), tmp_path, "cfg")
    assert (tmp_path / "data.parquet").read_bytes() == b"data"


def test_pair_weights_do_not_use_future_truth(case):
    before, _ = model.ensemble(
        case["members"], case["actuals"].onshore, case["day"], case["origin"]
    )
    truth = case["actuals"].onshore.copy()
    truth.loc[truth.index >= case["origin"] - pd.Timedelta(hours=2)] = 1e12
    after, audit = model.ensemble(case["members"], truth, case["day"], case["origin"])
    pd.testing.assert_frame_equal(before, after)
    w = np.asarray(audit["onshore_weights"])
    assert (w >= -1e-8).all() and (w <= 0.700001).all() and np.allclose(w.sum(axis=1), 1)


def test_late_source_arrival_stops_before_model(case, monkeypatch, tmp_path):
    (tmp_path / "seed.json").write_text(
        json.dumps(
            {
                "config_sha256": case["sha"],
                "production_approval": {"approved": True, "reference": "synthetic test assets"},
            }
        )
    )
    state.write_frame(tmp_path / "members.parquet", case["members"])
    monkeypatch.setattr(
        producer,
        "_collect",
        lambda *a: [{"retrieved_at_utc": (case["origin"] + pd.Timedelta(minutes=1)).isoformat()}],
    )
    monkeypatch.setattr(producer, "_actuals", lambda *a: case["actuals"])
    with pytest.raises(ValueError, match="arrived after"):
        producer.produce(tmp_path, clock=lambda: case["origin"] - pd.Timedelta(minutes=1))


def test_validation_forecasts_do_not_overwrite_saved_members(case, monkeypatch, tmp_path):
    state.write_frame(tmp_path / "members.parquet", case["members"])
    state.write_frame(tmp_path / "totals.parquet", case["totals"])
    old_members = (tmp_path / "members.parquet").read_bytes()
    old_totals = (tmp_path / "totals.parquet").read_bytes()
    columns = [f"{m}_{c}" for m in model.MEMBERS for c in ("onshore", "offshore")]
    monkeypatch.setattr(
        model,
        "predict",
        lambda *a, **kw: {case["day"]: case["members"].reindex(day_grid(case["day"]))[columns]},
    )
    out = producer._new_points(
        tmp_path,
        case["cfg"],
        [case["day"]],
        case["actuals"],
        "cpu",
        force=case["day"],
        persist=False,
    )
    producer._new_totals(
        tmp_path, case["cfg"], [case["day"]], case["actuals"], out, force=case["day"], persist=False
    )
    assert (tmp_path / "members.parquet").read_bytes() == old_members
    assert (tmp_path / "totals.parquet").read_bytes() == old_totals


def test_private_storage_guard_stops_download(case, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        bundle,
        "HfApi",
        lambda: SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(private=False)),
    )
    with pytest.raises(ValueError, match="private"):
        bundle.load(case["now"], case["sha"], 24500)


def test_research_seed_cannot_be_used_in_production(case, tmp_path):
    with pytest.raises(ValueError, match="not approved"):
        state.require_production_assets({})
    with pytest.raises(ValueError, match="not approved"):
        state.require_production_assets({"production_approval": {"approved": True}})
    (tmp_path / "seed.json").write_text(json.dumps({"config_sha256": case["sha"]}))
    with pytest.raises(ValueError, match="not approved"):
        producer.produce(tmp_path, clock=lambda: case["origin"] - pd.Timedelta(minutes=30))


def test_clean_asset_approval_requires_matching_provenance_and_checksums(case, tmp_path):
    import runpy
    from pathlib import Path

    tool = Path(producer.__file__).resolve().parents[3] / "tools/build_wind_pair_seed.py"
    verify = runpy.run_path(str(tool))["verified_clean_run"]
    (tmp_path / "asset.txt").write_bytes(b"synthetic")
    cfg, _ = producer.config()
    manifest = {
        "members_days": 104,
        "provenance": {
            "timesfm_outputs_used_for_fit_or_forecast": False,
            "models": {
                "t0": cfg["models"]["t0_revision"],
                "timestral": cfg["models"]["timestral_revision"],
            },
            "weather_order": cfg["weather_models"],
        },
        "files": {"asset.txt": state.digest(b"synthetic")},
    }
    path = tmp_path / "asset_manifest.json"
    path.write_text(json.dumps(manifest))
    approval, _ = verify(tmp_path, cfg)
    assert approval["approved"] is True
    assert approval["reference"].startswith("clean-refit-sha256:")
    (tmp_path / "asset.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path, cfg)
    manifest["provenance"]["timesfm_outputs_used_for_fit_or_forecast"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unverified"):
        verify(tmp_path, cfg)
