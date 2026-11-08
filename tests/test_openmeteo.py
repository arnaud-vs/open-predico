import pandas as pd

from openpredico.data import openmeteo


def test_hourly_to_quarter_hours_radiation_is_preceding_hour(monkeypatch):
    hours = pd.date_range("2026-06-01 09:00", periods=3, freq="h", tz="UTC")
    hourly = pd.DataFrame({v: [1.0, 2.0, 3.0] for v in openmeteo.VARIABLES}, index=hours)
    monkeypatch.setattr(openmeteo, "_get", lambda points, start, end: hourly)
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: None)
    out = openmeteo.fetch_frame(
        "land",
        pd.Timestamp("2026-06-01 09:00", tz="UTC"),
        pd.Timestamp("2026-06-01 11:00", tz="UTC"),
    )
    assert list(out.wind_speed_100m) == [1.0] * 4 + [2.0] * 4  # instant value, repeated
    assert list(out.shortwave_radiation) == [2.0] * 4 + [3.0] * 4  # 10:00 = 09:00-10:00 mean
