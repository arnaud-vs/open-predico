import pandas as pd

from openpredico.live.submit import forecast, validate


def test_forecast_t0_d1_then_climatology_no_lookahead():
    now = pd.Timestamp("2026-10-24 13:00", tz="UTC")  # D+1 = 25 Oct, 100 quarter-hours
    idx = pd.date_range("2026-09-01", "2026-10-26", freq="15min", tz="UTC", inclusive="left")
    target = pd.DataFrame({"mfrr_energy": 1.0, "mfrr_price": 2.0}, index=idx)
    calendar = pd.DataFrame({"qh_of_day": idx.hour * 4.0}, index=idx)
    seen = {}

    def predict(past, future, days):
        seen.update(past_end=past.index.max(), cols=list(future.columns))
        grid = pd.date_range(days[0], periods=100, freq="15min").tz_convert("UTC")
        return pd.DataFrame({"q10": 7.0, "q50": 8.0, "q90": 9.0}, index=grid)

    frames = {"target": target, "calendar": calendar}
    fc, inputs = forecast("spain_mfrr_energy", frames, predict, now)
    validate(fc, now)
    assert seen["past_end"] < now - pd.Timedelta(hours=2)  # publication lag respected
    assert seen["cols"] == ["qh_of_day"]
    assert (fc.q50.iloc[:100] == 8).all() and (fc.q50.iloc[100:] == 1).all()
    assert inputs.loc[inputs.index >= now, "mfrr_energy"].isna().all()  # no future truth stored


def test_forecast_generation_sees_own_groups_and_is_clipped():
    now = pd.Timestamp("2026-11-07 12:30", tz="UTC")
    idx = pd.date_range("2026-09-01", "2026-11-09", freq="15min", tz="UTC", inclusive="left")
    frames = {
        "wind_50hertz": pd.DataFrame({"wind_50hertz": 5.0, "wind_50hertz_onshore": 4.0}, index=idx),
        "calendar_de": pd.DataFrame({"holiday_de": 0.0}, index=idx),
        "target": pd.DataFrame({"mfrr_energy": 1.0}, index=idx),  # Spain's: must be ignored
    }
    seen = {}

    def predict(past, future, days):
        seen.update(past=list(past.columns), future=list(future.columns))
        grid = pd.date_range(days[0], periods=96, freq="15min").tz_convert("UTC")
        return pd.DataFrame({"q10": -3.0, "q50": 8.0, "q90": 1e6}, index=grid)

    fc, _ = forecast("germany_wind_50hertz_ca", frames, predict, now)
    validate(fc, now)
    assert seen == {"past": ["wind_50hertz", "wind_50hertz_onshore"], "future": ["holiday_de"]}
    assert fc.q10.min() == 0 and fc.q90.max() == 26000  # clipped to [0, capacity]
