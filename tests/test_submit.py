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
        "target_50hertz": pd.DataFrame({"wind_50hertz": 5.0, "solar_50hertz": 4.0}, index=idx),
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
    assert seen == {"past": ["wind_50hertz", "solar_50hertz"], "future": ["holiday_de"]}
    assert fc.q10.min() == 0 and fc.q90.max() == 24500  # clipped to [0, capacity]


def test_forecast_drops_a_weather_model_missing_on_d1_or_absent():
    now = pd.Timestamp("2026-11-15 08:15", tz="UTC")  # Belgium, D+1 = 16 Nov
    idx = pd.date_range("2026-08-01", "2026-11-18", freq="15min", tz="UTC", inclusive="left")
    knmi = pd.DataFrame({"ghi_be_knmi": 1.0}, index=idx)
    knmi.loc[knmi.index >= "2026-11-15 23:00", "ghi_be_knmi"] = None  # KNMI run missing for D+1
    frames = {  # weather_be_aifs absent: its source failed (lenient build)
        "target_elia": pd.DataFrame({"solar_be": 5.0}, index=idx),
        "weather_be": pd.DataFrame({"ghi_be": 1.0}, index=idx),
        "weather_be_knmi": knmi,
    }
    seen = {}

    def predict(past, future, days):
        seen.update(past=list(past.columns), future=list(future.columns))
        grid = pd.date_range(days[0], periods=96, freq="15min").tz_convert("UTC")
        return pd.DataFrame({"q10": 1.0, "q50": 2.0, "q90": 3.0}, index=grid)

    fc, _ = forecast("belgium_solar_opendata_elia", frames, predict, now)
    validate(fc, now)
    assert seen == {"past": ["solar_be"], "future": ["ghi_be"]}


def test_archived_fetches_only_new_days_and_keeps_old_ones(monkeypatch):
    from openpredico.live import submit

    idx = pd.date_range("2026-01-01", "2026-11-20", freq="15min", tz="UTC", inclusive="left")
    old = pd.DataFrame({"ghi_be": 1.0}, index=idx)
    calls, saved = [], {}

    def build(g, start, end, tz, lenient):
        calls.append(start)
        grid = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
        return pd.DataFrame({"ghi_be": 2.0}, index=grid)

    monkeypatch.setattr(submit.hf, "download", lambda repo, path: old)
    monkeypatch.setattr(submit.hf, "upload", lambda df, *a: saved.update(df=df))
    monkeypatch.setattr(submit, "build", build)
    start, end = pd.Timestamp("2026-08-20", tz="UTC"), pd.Timestamp("2026-11-22", tz="UTC")
    out = submit.archived("weather_be", start, end, "Europe/Brussels", save=True)
    assert calls == ["2026-11-18T00:00:00"]  # from the day before the last complete day
    assert out.index[0] == start and out.index[-1] < end and out.notna().all().all()
    assert (out.loc[:"2026-11-17", "ghi_be"] == 1).all() and (
        out.loc["2026-11-18":, "ghi_be"] == 2
    ).all()
    assert saved["df"].index[0] == idx[0]  # old days kept


def test_archived_raises_on_hf_errors_other_than_a_missing_file(monkeypatch):
    import pytest
    from huggingface_hub.errors import LocalEntryNotFoundError

    from openpredico.live import submit

    def down(repo, path):
        raise LocalEntryNotFoundError("HF unreachable")  # outage, not a missing archive

    monkeypatch.setattr(submit.hf, "download", down)
    monkeypatch.setattr(submit.hf, "upload", lambda *a: pytest.fail("archive overwritten"))
    start, end = pd.Timestamp("2026-08-20", tz="UTC"), pd.Timestamp("2026-11-22", tz="UTC")
    with pytest.raises(LocalEntryNotFoundError):
        submit.archived("weather_be", start, end, "Europe/Brussels", save=True)
