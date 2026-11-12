import base64
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openpredico.live import wind_pair as pipeline
from openpredico.live import wind_pair_snapshot as snapshot
from openpredico.live.wind_pair_state import digest
from openpredico.models.conformal import TZ, day_grid


def source(root, date="2026-11-12"):
    root.mkdir()
    now = pd.Timestamp(f"{date} 18:30", tz="UTC")
    end = pd.Timestamp(date, tz=TZ).tz_convert("UTC")
    index = pd.date_range(end - pd.Timedelta(days=120), end + pd.Timedelta(days=2), freq="15min")
    cfg, sha = pipeline.config()
    (root / "seed.json").write_text(
        json.dumps(
            {
                "config_sha256": sha,
                "production_approval": {
                    "approved": True,
                    "reference": "clean-refit-sha256:" + "a" * 64,
                },
            }
        )
    )
    actuals = pd.DataFrame(
        {
            "onshore": 2.0,
            "offshore": 1.0,
            "total": 3.0,
            "received_at": (now - pd.Timedelta(hours=1)).isoformat(),
        },
        index=index,
    )
    frames = {"actuals": actuals}
    forecast_index = index[index >= end - pd.Timedelta(days=75)]
    local_days = forecast_index.tz_convert(TZ).strftime("%Y-%m-%d")
    origins = {d: pipeline.origin_for(d).isoformat() for d in set(local_days)}
    common = {"issue_time_utc": [origins[d] for d in local_days], "delivery_day": local_days}
    frames["members"] = pd.DataFrame(
        {**{c: 1.0 for c in snapshot.MEMBERS}, **common, "kind": "live"}, index=forecast_index
    )
    frames["totals"] = pd.DataFrame(
        {"onshore_point_mw": 1.0, "offshore_point_mw": 1.0, "point_mw": 2.0, **common},
        index=forecast_index,
    )
    for name, frame in frames.items():
        frame["private_token"] = "MUST_NOT_ESCAPE"
        frame.attrs = {"secret": "MUST_NOT_ESCAPE"}
        frame.to_parquet(root / f"{name}.parquet")
    (root / "weather").mkdir()
    for component in ("onshore", "offshore"):
        weather = pd.DataFrame(np.ones((len(index), 96), dtype=np.float32), index=index)
        weather["private_token"] = "MUST_NOT_ESCAPE"
        weather.attrs = {"secret": "MUST_NOT_ESCAPE"}
        # Parquet requires uniform string column names when mixed; use a numeric extra instead.
        weather = weather.drop(columns="private_token")
        weather[999] = 9876.0
        weather.to_parquet(root / "weather" / f"{component}.parquet")
    for d in [
        str((end.tz_convert(TZ) - pd.DateOffset(days=100)).date()),
        str((end.tz_convert(TZ) + pd.DateOffset(days=1)).date()),
    ]:
        folder = root / "banks" / d
        folder.mkdir(parents=True)
        for component in ("onshore", "offshore"):
            np.savez_compressed(
                folder / f"{component}.npz",
                speed=np.ones((6, 3, 4)) / 4,
                private_token="MUST_NOT_ESCAPE",
            )
    (root / "pending").mkdir()
    (root / "pending/future.zip").write_bytes(b"MUST_NOT_ESCAPE")
    return cfg, sha, now


def changed(data, change):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = {name: z.read(name) for name in z.namelist()}
    manifest = json.loads(files.pop("manifest.json"))
    change(files)
    manifest["files"] = {name: digest(value) for name, value in files.items()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, value in files.items():
            z.writestr(name, value)
        z.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def test_export_strips_future_rows_private_columns_and_metadata(tmp_path):
    root, output = tmp_path / "state", tmp_path / "exports"
    cfg, sha, now = source(root)
    result = snapshot.export(root, output, cfg=cfg, config_hash=sha, now=now)
    path = output / result["snapshot_date"] / snapshot.ARCHIVE
    data = path.read_bytes()
    manifest = snapshot.inspect(data, config_hash=sha, now=now)
    assert manifest["last_member_day"] == "2026-11-11"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "pending/future.zip" not in archive.namelist()
        assert any("banks/2026-11-13/" in n for n in archive.namelist())
        assert manifest["last_weather_bank_day"] == "2026-11-13"
        for name in snapshot.BASE_FILES:
            frame = pd.read_parquet(io.BytesIO(archive.read(name)))
            assert frame.index.max() < pd.Timestamp(manifest["cutoff_utc"])
            assert not frame.attrs
            assert "private_token" not in frame
            assert 999 not in frame
        members = pd.read_parquet(io.BytesIO(archive.read("members.parquet")))
        assert members.delivery_day.nunique() == 60
    again = snapshot.export(
        root, output, cfg=cfg, config_hash=sha, now=now + pd.Timedelta(minutes=10)
    )
    assert again == result
    assert path.read_bytes() == data


def test_stale_truth_prevents_publication(tmp_path):
    root = tmp_path / "state"
    cfg, sha, now = source(root)
    p = root / "actuals.parquet"
    actuals = pd.read_parquet(p)
    actuals.loc[
        day_grid("2026-11-10").append(day_grid("2026-11-11")), ["onshore", "offshore", "total"]
    ] = np.nan
    actuals.to_parquet(p)
    with pytest.raises(snapshot.SnapshotError, match="stale_actual_history"):
        snapshot.export(root, tmp_path / "exports", cfg=cfg, config_hash=sha, now=now)
    assert not list((tmp_path / "exports").rglob("*.zip"))


def test_independent_check_rejects_rehashed_future_rows_and_extra_files(tmp_path):
    root = tmp_path / "state"
    cfg, sha, now = source(root)
    output = tmp_path / "exports"
    result = snapshot.export(root, output, cfg=cfg, config_hash=sha, now=now)
    data = (output / result["snapshot_date"] / snapshot.ARCHIVE).read_bytes()

    def future(files):
        f = pd.read_parquet(io.BytesIO(files["members.parquet"]))
        row = f.iloc[-1:].copy()
        row.index = day_grid("2026-11-12")[:1]
        buf = io.BytesIO()
        pd.concat([f, row]).to_parquet(buf)
        files["members.parquet"] = buf.getvalue()

    with pytest.raises(snapshot.SnapshotError, match="future_or_empty_frame"):
        snapshot.inspect(changed(data, future), config_hash=sha, now=now)
    with pytest.raises(snapshot.SnapshotError, match="unexpected"):
        snapshot.inspect(
            changed(data, lambda f: f.update({"raw_weather.json": b"secret"})),
            config_hash=sha,
            now=now,
        )
    with pytest.raises(snapshot.SnapshotError, match="config_mismatch"):
        snapshot.inspect(data, config_hash="changed", now=now)


def test_snapshot_cutoff_and_full_last_day_respect_dst(tmp_path):
    root = tmp_path / "state"
    cfg, sha, now = source(root, "2026-10-26")
    output = tmp_path / "exports"
    result = snapshot.export(root, output, cfg=cfg, config_hash=sha, now=now)
    assert result["cutoff_utc"] == "2026-10-25T23:00:00+00:00"
    with zipfile.ZipFile(output / result["snapshot_date"] / snapshot.ARCHIVE) as z:
        f = pd.read_parquet(io.BytesIO(z.read("members.parquet")))
        assert len(f[f.delivery_day.eq("2026-10-25")]) == 100


@pytest.fixture
def publisher():
    path = Path(__file__).resolve().parents[1] / "tools/publish_wind_snapshot.py"
    spec = importlib.util.spec_from_file_location("snapshot_publisher_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pointer_never_rewinds_or_writes_main(publisher, monkeypatch):
    writes = []
    previous = {"snapshot_date": "2026-11-12", "sha256": "abc"}

    def fake(*args, data=None, **kwargs):
        if data:
            writes.append((args, data))
            return {}
        if "/git/ref/" in args[1]:
            return {"ref": "exists"}
        return {"sha": "old", "content": base64.b64encode(json.dumps(previous).encode()).decode()}

    monkeypatch.setattr(publisher, "gh", fake)
    publisher.pointer_update("owner/repo", previous)
    assert writes == []
    with pytest.raises(RuntimeError, match="backwards"):
        publisher.pointer_update("owner/repo", {"snapshot_date": "2026-11-11", "sha256": "older"})
    with pytest.raises(RuntimeError, match="immutable"):
        publisher.pointer_update("owner/repo", {**previous, "sha256": "different"})
    publisher.pointer_update("owner/repo", {"snapshot_date": "2026-11-13", "sha256": "new"})
    assert len(writes) == 1
    assert writes[0][1]["branch"] == publisher.BRANCH
    assert writes[0][0][-2:] == ("--method", "PUT")


@pytest.mark.parametrize("draft", [True, False])
def test_bad_release_checksum_never_publishes_or_updates_pointer(
    publisher, monkeypatch, tmp_path, draft
):
    calls = []
    monkeypatch.setattr(publisher, "release_metadata", lambda *a: {"draft": draft})
    monkeypatch.setattr(publisher, "published_archive_matches", lambda *a: False)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        publisher, "pointer_update", lambda *a: pytest.fail("pointer must stay unchanged")
    )
    public = {"snapshot_date": "2026-11-12", "sha256": "abc"}
    with pytest.raises(RuntimeError, match="checksum"):
        publisher.publish(tmp_path, public, "owner/repo", "a" * 40)
    assert not any("--draft=false" in command for command in calls)
    assert not any("--clobber" in command for command in calls)


def test_recent_unpublished_actuals_stay_missing_without_blocking_snapshot(tmp_path):
    root = tmp_path / "state"
    cfg, sha, now = source(root)
    path = root / "actuals.parquet"
    actuals = pd.read_parquet(path)
    actuals.loc[day_grid("2026-11-11"), ["onshore", "offshore", "total"]] = np.nan
    actuals.to_parquet(path)
    output = tmp_path / "exports"
    result = snapshot.export(root, output, cfg=cfg, config_hash=sha, now=now)
    assert result["last_complete_truth_day"] == "2026-11-10"
    with zipfile.ZipFile(output / result["snapshot_date"] / snapshot.ARCHIVE) as z:
        saved = pd.read_parquet(io.BytesIO(z.read("actuals.parquet")))
        assert saved.reindex(day_grid("2026-11-11")).total.isna().all()


def test_resuming_draft_without_published_tag(publisher, monkeypatch):
    def fake(*args, **kwargs):
        if "/releases/tags/" in args[1]:
            return None
        return [{"tag_name": "wind-pair-bootstrap-2026-11-12", "draft": True, "assets": []}]

    monkeypatch.setattr(publisher, "gh", fake)
    assert (
        publisher.release_metadata("owner/repo", "wind-pair-bootstrap-2026-11-12")["draft"] is True
    )


def test_first_pointer_creates_metadata_only_branch(publisher, monkeypatch):
    writes = []

    def fake(*args, data=None, **kwargs):
        if data is None:
            return None
        writes.append((args[1], data))
        return {"sha": "a" * 40}

    monkeypatch.setattr(publisher, "gh", fake)
    publisher.pointer_update("owner/repo", {"snapshot_date": "2026-11-12", "sha256": "abc"})
    assert len(writes) == 3
    assert [item["path"] for item in writes[0][1]["tree"]] == [publisher.POINTER]
    assert writes[1][1]["parents"] == []
    assert writes[1][1]["author"] == publisher.BOT
    assert writes[2][1]["ref"] == f"refs/heads/{publisher.BRANCH}"
