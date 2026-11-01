import pandas as pd
import pytest

from openpredico.data.esios import to_series

START, END = pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-10-02", tz="UTC")


def v(ts, value, geo=8741):
    return {"value": value, "datetime_utc": ts, "geo_id": geo}


def test_to_series_sorts_dedups_and_clips_range():
    values = [
        v("2026-10-01T00:15:00Z", 2.0),
        v("2026-10-01T00:00:00Z", 1.0),
        v("2026-10-01T00:15:00Z", 2.5),  # overlap at a chunk edge: last one wins
        v("2026-10-02T00:00:00Z", 9.0),  # end is exclusive
    ]
    s = to_series(values, 2197, None, START, END)
    assert list(s.index) == [START, START + pd.Timedelta("15min")]
    assert list(s) == [1.0, 2.5]


def test_to_series_requires_geo_when_ambiguous():
    values = [v("2026-10-01T00:00:00Z", 1.0, geo=1), v("2026-10-01T00:00:00Z", 2.0, geo=2)]
    with pytest.raises(ValueError, match="geo_id"):
        to_series(values, 1, None, START, END)
    assert list(to_series(values, 1, 2, START, END)) == [2.0]
