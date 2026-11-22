import pandas as pd
import requests

from openpredico.live import submit
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
    fc, _ = forecast("belgium_solar_opendata_elia", frames, _intraday_predict(seen), now)
    validate(fc, now)
    assert seen["past"] == ["solar_be"] and seen["future"] == ["ghi_be"]  # D+1 session hour


def _intraday_predict(seen):
    def predict(past, future, days, start):  # t0: from the end of past through the end of D+1
        seen.update(past=list(past.columns), future=list(future.columns), past_end=past.index[-1])
        stop = (days[0] + pd.Timedelta("1D")).tz_convert("UTC")
        grid = pd.date_range(past.index[-1], stop, freq="15min")[1:-1]
        out = pd.DataFrame({"q10": 1.0, "q50": 2.0, "q90": 3.0}, index=grid)
        return out[out.index >= start]

    return predict


def test_intraday_forecast_starts_now_and_drops_weather_missing_on_d1():
    now = pd.Timestamp("2026-11-15 00:20", tz="UTC")  # Belgium 01:20: D+1 runs not out yet
    idx = pd.date_range("2026-08-01", "2026-11-18", freq="15min", tz="UTC", inclusive="left")
    weather = pd.DataFrame({"ghi_be": 1.0, "ghi_be_knmi": 1.0}, index=idx)
    weather.loc[weather.index >= "2026-11-15 23:00", "ghi_be_knmi"] = None  # KNMI misses D+1
    weather.loc["2026-11-15 12:00", "ghi_be"] = None  # IFS misses a step today
    frames = {
        "target_elia": pd.DataFrame({"solar_be": 5.0}, index=idx),
        "weather_be": weather[["ghi_be"]],
        "weather_be_knmi": weather[["ghi_be_knmi"]],
    }
    seen = {}
    fc, _ = forecast("belgium_solar_opendata_elia", frames, _intraday_predict(seen), now)
    validate(fc, now)
    assert seen["future"] == []  # both left out: no NaN known-future inputs reach t0
    assert seen["past_end"] == pd.Timestamp("2026-11-14 23:45", tz="UTC")  # 30 min Elia lag
    assert fc.index[0] == pd.Timestamp("2026-11-15 00:30", tz="UTC")  # from now on
    live = "t0gamma_cov_wn3_ctx90"  # targets.toml live_config
    assert (fc.loc[:"2026-11-16 22:45", "model"] == live).all()


def test_archived_fetches_only_new_days_and_keeps_old_ones(monkeypatch):
    from openpredico.live import submit

    tz = "Europe/Brussels"
    idx = pd.date_range("2026-01-01", "2026-11-20", freq="15min", tz=tz, inclusive="left")
    old = pd.DataFrame({"ghi_be": 1.0}, index=idx.tz_convert("UTC"))
    calls, saved = [], {}

    def build(g, start, end, tz, lenient):
        calls.append(start)
        grid = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
        return pd.DataFrame({"ghi_be": 2.0}, index=grid)

    monkeypatch.setattr(submit.hf, "download", lambda repo, path: old)
    monkeypatch.setattr(submit.hf, "upload", lambda df, *a: saved.update(df=df))
    monkeypatch.setattr(submit, "build", build)
    start, end = pd.Timestamp("2026-08-20", tz="UTC"), pd.Timestamp("2026-11-22", tz="UTC")
    out = submit.archived("weather_be", start, end, tz, save=True)
    assert calls == ["2026-11-19T23:00:00"]  # first local day not archived; a run never changes
    assert out.index[0] == start and out.index[-1] < end and out.notna().all().all()
    assert (out.loc[:"2026-11-19 22:45", "ghi_be"] == 1).all()
    assert (out.loc["2026-11-19 23:00":, "ghi_be"] == 2).all()
    assert saved["df"].index[0] == idx[0]  # old days kept


def test_archived_refetches_the_revision_window_and_saves_once_a_day(monkeypatch):
    from openpredico.live import submit

    tz = "Europe/Brussels"
    idx = pd.date_range("2026-08-01", "2026-11-21 10:00", freq="15min", tz="UTC", inclusive="left")
    archive = {"df": pd.DataFrame({"solar_be": 1.0}, index=idx)}
    calls, saves = [], []

    def build(g, start, end, tz, lenient):
        calls.append(start)
        grid = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
        new = pd.DataFrame({"solar_be": 2.0}, index=grid)  # revised and new values
        new.loc[new.index >= pd.Timestamp("2026-11-22 10:00", tz="UTC")] = None
        return new

    def upload(df, *a):
        saves.append(df.index.max())
        archive["df"] = df

    monkeypatch.setattr(submit.hf, "download", lambda repo, path: archive["df"])
    monkeypatch.setattr(submit.hf, "upload", upload)
    monkeypatch.setattr(submit, "build", build)
    start, end = pd.Timestamp("2026-08-20", tz="UTC"), pd.Timestamp("2026-11-23 23:00", tz="UTC")
    out = submit.archived("target_elia", start, end, tz, save=True)
    assert calls == ["2026-11-13T23:00:00"]  # REVISE["elia"] = 7 days before 21 Nov, local
    assert (out.loc[:"2026-11-13 22:45", "solar_be"] == 1).all()
    assert (out.loc["2026-11-13 23:00":"2026-11-22 09:45", "solar_be"] == 2).all()  # new wins
    assert out.loc["2026-11-22 10:00":, "solar_be"].isna().all()  # unpublished: no fill
    assert len(saves) == 1  # the frontier moved from 21 to 22 Nov
    submit.archived("target_elia", start, end, tz, save=True)  # an hour later, same day
    assert len(saves) == 1 and calls[-1] == "2026-11-14T23:00:00"


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


def test_older_run_fills_d1_where_the_day_older_run_reaches(monkeypatch):
    grid = pd.date_range(
        "2026-11-22 23:00", "2026-11-24 23:00", freq="15min", tz="UTC", inclusive="left"
    )
    asked = []

    def fetch_run(run, area):
        asked.append((run, area))
        if area == "be_d2":
            raise requests.HTTPError("no run")
        hours = pd.date_range(run, run + pd.Timedelta(hours=60), freq="h")
        return pd.DataFrame({"shortwave_radiation": 1.0}, index=hours)

    monkeypatch.setattr(submit.openmeteo, "fetch_run", fetch_run)
    sources = {
        "ghi_be": "openmeteo:be.shortwave_radiation",
        "ghi_be_d2": "openmeteo:be_d2.shortwave_radiation",
    }
    out = submit.older_run(sources, grid, "Europe/Brussels")
    # D+1 = 24 Nov: IFS run of 23 Nov 00 UTC minus a day = 22 Nov 00 UTC, reaching 24 Nov 12 UTC
    assert asked[0] == (pd.Timestamp("2026-11-22", tz="UTC"), "be")
    assert out.loc["2026-11-24 12:00", "ghi_be"] == 1.0
    assert out.loc["2026-11-24 13:15":, "ghi_be"].isna().all()
    assert out["ghi_be_d2"].isna().all()


def test_wn3_archive_is_private_and_a_bigquery_failure_drops_it(monkeypatch):
    from openpredico.data import groups, weathernext
    from openpredico.evaluation import experiment

    tz, name = "Europe/Brussels", "belgium_solar_opendata_elia"
    targets = experiment.load_targets()
    assert "weather_be_wn3" in targets[name]["groups"]  # live since 2026-11-22 (PLAN.md)
    monkeypatch.setattr(experiment, "load_targets", lambda: targets)
    monkeypatch.setattr(submit, "live_targets", lambda: targets)
    idx = pd.date_range("2026-08-01", "2026-11-15 23:00", freq="15min", tz="UTC", inclusive="left")
    repos = []

    def down(repo, path):
        repos.append(repo)
        return pd.DataFrame({c: 1.0 for c in groups.load_groups()["weather_be_wn3"]}, index=idx)

    def fail(*a):
        raise RuntimeError("BigQuery down")

    monkeypatch.setattr(submit.hf, "download", down)
    monkeypatch.setattr(submit.hf, "upload", lambda df, repo, *a: repos.append(repo))
    monkeypatch.setattr(weathernext, "fetch", fail)
    assert groups.repo("weather_be_wn3") == "weather-private"
    assert groups.repo("weather_be") == "benchmark"
    start, end = pd.Timestamp("2026-08-01", tz="UTC"), pd.Timestamp("2026-11-16 23:00", tz="UTC")
    wn3 = submit.archived("weather_be_wn3", start, end, tz, save=True)
    assert repos == ["weather-private"]  # read there; nothing new, so nothing saved
    now = pd.Timestamp("2026-11-15 08:15", tz="UTC")  # D+1 session hour
    frames = {"target_elia": pd.DataFrame({"solar_be": 5.0}, index=wn3.index)}
    frames["weather_be_wn3"] = wn3
    seen = {}
    fc, _ = forecast(name, frames, _intraday_predict(seen), now)
    validate(fc, now)
    assert seen["future"] == []  # WN3 missing on D+1: left out, the run still submits
