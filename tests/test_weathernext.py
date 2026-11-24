import pandas as pd

from openpredico.data import openmeteo, weathernext


def test_runs_cover_the_local_day_and_land_before_the_origin():
    for day in ("2026-06-02", "2026-12-02"):  # summer and winter
        d = pd.Timestamp(day, tz="Europe/Brussels")
        for area, hour in (("be_wn3", 9), ("land_wn3", 14)):
            run = weathernext.run_time(d, area)
            origin = (d - pd.Timedelta(days=1)).replace(hour=hour).tz_convert("UTC")
            assert run + weathernext.LATENCY <= origin - pd.Timedelta("15min")
            end = (d + pd.Timedelta(days=1)).tz_convert("UTC")  # 48 h runs
            assert run + pd.Timedelta("48h") >= end - pd.Timedelta("1h")


def test_fetch_assembles_one_query_per_run(monkeypatch):
    n = len(openmeteo.POINTS["be"])
    cols = {c for c, _, _ in weathernext.COLUMNS.values()}
    rows = [
        {"k": k, "i": i, **{c: 1.0 for c in cols}, "temperature_2m_mean": 273.15 + i + 100 * k}
        | {"surface_solar_radiation_downwards_1hr_mean": 3600.0 * i}
        for k in range(n)
        for i in range(48)
    ]
    sqls = []

    class Job:
        total_bytes_processed = total_bytes_billed = 10**7

        def result(self, timeout):
            return rows

    class Client:
        def query(self, sql, job_config):
            sqls.append(sql)
            return Job()

    monkeypatch.setattr(weathernext, "_config", dict)
    start, end = (pd.Timestamp(t, tz="UTC") for t in ("2026-06-01 22:00", "2026-06-02 22:00"))
    out = weathernext.fetch(start, end, ["be_wn3"], Client())
    assert len(sqls) == 2 and all("init_time = TIMESTAMP '2026-05-31 23:00:00'" in s for s in sqls)
    df = out["be_wn3"]  # local 2 June: UTC 22:00 on 1 June onwards, leads 23..46 h
    w = openmeteo.WEIGHTS["be"]
    mean_k = sum(k * x for k, x in enumerate(w)) / sum(w)
    assert df.temperature_2m.iloc[0] == 22 + 100 * mean_k  # offset 22 = 23 h ahead: 22:00
    assert df.shortwave_radiation.iloc[0] == 23  # 22:00-23:00, labelled 23:00 (offset 23)
    assert df.cloud_cover.iloc[0] == 100 and len(df) == 96 and not df.isna().any().any()


def test_fetch_leaves_out_an_area_whose_run_is_not_out_yet(monkeypatch):
    cols = {c for c, _, _ in weathernext.COLUMNS.values()}
    n = len(openmeteo.POINTS["be"])
    rows = [{"k": k, "i": i, **{c: 1.0 for c in cols}} for k in range(n) for i in range(48)]

    class Job:
        total_bytes_processed = total_bytes_billed = 10**7

        def __init__(self, sql):
            self.sql = sql

        def result(self, timeout):  # Germany's 04 UTC run is not out yet
            return [] if "04:00:00" in self.sql else rows

    class Client:
        def query(self, sql, job_config):
            return Job(sql)

    monkeypatch.setattr(weathernext, "_config", dict)
    start, end = (pd.Timestamp(t, tz="UTC") for t in ("2026-06-01 22:00", "2026-06-02 22:00"))
    out = weathernext.fetch(start, end, ["be_wn3", "land_wn3"], Client())
    assert list(out) == ["be_wn3"] and out["be_wn3"].notna().all().all()


def test_credentials_use_modal_identity_token_else_adc(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("google.auth")
    for var in ("MODAL_IDENTITY_TOKEN_FILE", "MODAL_IDENTITY_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    weathernext.credentials.cache_clear()
    assert weathernext.credentials() is None  # laptop: Application Default Credentials
    weathernext.credentials.cache_clear()
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN", "fake.jwt.token")
    creds = weathernext.credentials()
    assert creds.service_account_email == weathernext.SERVICE_ACCOUNT
    assert creds.retrieve_subject_token(None) == "fake.jwt.token"  # via the temp file, no network
    weathernext.credentials.cache_clear()
    (tmp_path / "token").write_text("refreshed.jwt")
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN_FILE", str(tmp_path / "token"))
    assert weathernext.credentials().retrieve_subject_token(None) == "refreshed.jwt"
    weathernext.credentials.cache_clear()


def test_backfill_runs_resumes_and_keeps_each_run_with_its_leads(monkeypatch):
    stored, asked = {}, []
    t = pd.date_range("2026-11-20 01:00", periods=3, freq="h", tz="UTC")

    def run(client, r, areas):
        asked.append(r)
        return {} if r.hour == 2 else {"be_wn3": pd.DataFrame({"ghi": 1.0}, index=t)}

    monkeypatch.setattr(weathernext, "_client", lambda: None)
    monkeypatch.setattr(weathernext, "_run", run)
    monkeypatch.setattr(weathernext.hf, "upload", lambda df, *a: stored.update(df=df))

    def download(*a):
        if "df" not in stored:
            raise FileNotFoundError
        return stored["df"]

    monkeypatch.setattr(weathernext.hf, "download", download)
    weathernext.backfill_runs("2026-11-20 00:00Z", "2026-11-20 03:00Z", every=2)
    df = stored["df"]
    assert sorted(set(df.index.get_level_values("run").hour)) == [0, 1]  # 02 UTC run missing
    assert (
        len(df) == 6
        and (df.available - df.index.get_level_values("run")).eq(weathernext.LATENCY).all()
    )
    asked.clear()
    weathernext.backfill_runs("2026-11-20 00:00Z", "2026-11-20 03:00Z")
    assert [r.hour for r in asked] == [2]  # only the missing run is asked again
