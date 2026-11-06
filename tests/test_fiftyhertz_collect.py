import json

import pandas as pd

from openpredico.data import fiftyhertz_collect


def test_collection_saves_revisions_and_reports_freshness(tmp_path, monkeypatch):
    index = pd.date_range("2026-11-06T11:00Z", periods=4, freq="15min", name="time")
    frame = pd.DataFrame({"total": 3.0, "onshore": 2.0, "offshore": 1.0, "sum": 3.0}, index=index)
    calls = []

    def fetch(kind, start, end):
        calls.append((kind, start, end))
        return frame

    monkeypatch.setattr(fiftyhertz_collect, "fetch_frame", fetch)
    for minute in [30, 31]:
        now = pd.Timestamp(f"2026-11-06T12:{minute}Z")
        result = fiftyhertz_collect.collect(tmp_path, now)
        assert result["publication_lag_minutes"] == minute
    assert calls[0][1] == pd.Timestamp("2026-10-29T23:00Z")
    assert len(list((tmp_path / "snapshots").glob("*/actuals.parquet"))) == 2
    assert json.loads((tmp_path / "latest.json").read_text())["rows"] == 4
