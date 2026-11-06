import json

import pandas as pd
import pytest

from openpredico.data import fiftyhertz, groups


def records(day, periods=None):
    local = pd.date_range(
        day,
        pd.Timestamp(day) + pd.DateOffset(days=1),
        freq="15min",
        tz="Europe/Berlin",
        inclusive="left",
    )
    if periods:
        local = local[:periods]
    return [
        {"FromTime": t.strftime("%Y-%m-%dT%H:%M:%S"), "Value": 10.26, "Value2": 7, "Value3": 3}
        for t in local
    ]


@pytest.mark.parametrize("day,n", [("2026-11-04", 96), ("2026-03-29", 92), ("2026-10-25", 100)])
def test_dst_days_and_distinct_total(day, n):
    frame = fiftyhertz.parse_records(f"({json.dumps(records(day))});")
    assert len(frame) == frame.index.nunique() == n
    assert frame.index.to_series().diff().dropna().eq(pd.Timedelta("15min")).all()
    assert frame.total.iloc[0] == 10.26 and frame["sum"].iloc[0] == 10


def test_missing_values_do_not_become_zero_or_partial_sums():
    data = records("2026-11-04", 3)
    data[0].update(Value=-1, Value2="n.a.")
    data[1].update(Value3=None)
    data[2].update(Value=0, Value2=0, Value3=0)
    frame = fiftyhertz.parse_records(json.dumps(data))
    assert pd.isna(frame.total.iloc[0]) and pd.isna(frame["sum"].iloc[0])
    assert pd.isna(frame["sum"].iloc[1]) and frame["sum"].iloc[2] == 0


def test_empty_and_invalid_responses():
    assert fiftyhertz.parse_records("([]);").empty
    with pytest.raises(ValueError):
        fiftyhertz.parse_records('{"error": "unavailable"}')
    data = records("2026-11-04", 2)
    with pytest.raises(ValueError, match="unordered"):
        fiftyhertz.parse_records(json.dumps(data[::-1]))
    data[0]["FromTime"] = "2026-11-04T00:01:00"
    with pytest.raises(ValueError, match="quarter-hour"):
        fiftyhertz.parse_records(json.dumps(data))


def test_fetch_clips_range_and_passes_query(monkeypatch):
    calls = []

    class Response:
        text = json.dumps(records("2026-11-04"))

        def raise_for_status(self):
            pass

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(fiftyhertz._session, "get", get)
    out = fiftyhertz.fetch("actual_sum", "2026-11-04T00:15Z", "2026-11-04T00:45Z")
    assert len(out) == 2 and out.eq(10).all()
    assert calls[0][1]["params"] == {"filterDateTime": "2026-11-04"}
    assert out.index[0] == pd.Timestamp("2026-11-04T00:15Z")


def test_builder_caches_components_and_preserves_gaps(monkeypatch):
    calls = []
    frame = fiftyhertz.parse_records(json.dumps(records("2026-11-04", 2)))
    monkeypatch.setattr(
        groups,
        "load_groups",
        lambda: {
            "wind": {
                "wind_total": {"source": "fiftyhertz:actual_total"},
                "wind_sum": {"source": "fiftyhertz:actual_sum"},
            }
        },
    )

    def fetch(*args):
        calls.append(args)
        return frame

    monkeypatch.setattr(fiftyhertz, "fetch_frame", fetch)
    out = groups.build("wind", "2026-11-03T23:00", "2026-11-04T00:00")
    assert len(calls) == 1
    assert out.wind_sum.iloc[0] == 10 and out.iloc[2:].isna().all().all()


@pytest.mark.parametrize("extra", ["fill = 0,", "geo = 1,"])
def test_group_disallows_filling_and_other_area(tmp_path, extra):
    p = tmp_path / "groups.toml"
    p.write_text(
        '[wind]\nx = { source = "fiftyhertz:actual_sum", ' + extra + ' availability = "past" }'
    )
    with pytest.raises(ValueError, match="filling"):
        groups.load_groups(p)


def test_forecast_cannot_be_declared_known_before_gate(tmp_path):
    p = tmp_path / "groups.toml"
    p.write_text('[wind]\nx = { source = "fiftyhertz:forecast_total", availability = "future" }')
    with pytest.raises(ValueError, match="18:00"):
        groups.load_groups(p)
