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


def test_vintage_is_issued_before_the_origin():
    # Latest value used: D+2 00:00 UTC (radiation of D+1's last hour). Origin: 14:00 D-1 local.
    lead = pd.Timedelta(days=int(openmeteo.VINTAGE.removeprefix("previous_day")))
    latest = pd.Timestamp("2026-11-11 00:00", tz="UTC")  # D+1 = 10 Nov
    origin = pd.Timestamp("2026-11-09 14:00", tz="Europe/Berlin")
    assert latest - lead <= origin - pd.Timedelta("6h")  # with slack for run publication
