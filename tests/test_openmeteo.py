import pandas as pd

from openpredico.data import openmeteo


def test_each_local_day_reads_the_00z_run_of_the_day_before():
    day = pd.Timestamp("2026-03-29", tz="Europe/Berlin")  # DST day
    assert openmeteo.run_time(day) == pd.Timestamp("2026-03-28 00:00", tz="UTC")
    origin = (day - pd.Timedelta(days=1)).replace(hour=14).tz_convert("UTC")
    assert openmeteo.run_time(day) + pd.Timedelta("8h") < origin  # published ~08:00 UTC
    be = openmeteo.run_time(day, "be")  # Belgium: 18 UTC run of D-2, out before the 09:00 origin
    assert be == pd.Timestamp("2026-03-27 18:00", tz="UTC")
    assert be + pd.Timedelta("8h") < (day - pd.Timedelta(days=1)).replace(hour=9).tz_convert("UTC")


def test_run_keeps_its_local_day_and_shifts_radiation_to_the_hour_it_ends(monkeypatch):
    hours = pd.date_range("2026-06-01 00:00", periods=72, freq="h", tz="UTC")
    times = list(hours.strftime("%Y-%m-%dT%H:%M"))
    hourly = {"time": times, **{v: list(range(72)) for v in openmeteo.VARIABLES}}
    areas = ["land", "offshore"]
    n = sum(len(openmeteo.POINTS[a]) for a in areas)

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"hourly": hourly}] * n

    monkeypatch.setattr(openmeteo._session, "get", lambda *a, **k: Resp())
    out = openmeteo._run(pd.Timestamp("2026-06-02", tz="Europe/Berlin"), areas)["land"]
    assert out.index[0] == pd.Timestamp("2026-06-01 22:00", tz="UTC")  # local midnight
    assert len(out) == 24
    assert out.wind_speed_100m.iloc[0] == 22  # instant value at 22:00
    assert out.shortwave_radiation.iloc[0] == 23  # 22:00-23:00 mean, labelled 23:00 by the API
