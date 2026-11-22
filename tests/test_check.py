import pandas as pd

from openpredico.live.submit import not_ready

BE = "belgium_solar_opendata_elia"


class _Client:
    def coverage(self):
        rows = [{"challenge_id": "e", "ready": True}, {"challenge_id": "p", "ready": False}]
        rows += [{"challenge_id": "i", "ready": False, "start": "2026-11-15T14:00:00+00:00"}]
        return {"data": {"coverage": rows + [{"challenge_id": "x", "ready": False}]}}

    def open_challenges(self, name):
        return {
            "spain_mfrr_energy": [{"id": "e", "resource_name": name}],
            "spain_mfrr_price": [{"id": "p", "resource_name": name}],
            BE: [{"id": "i", "resource_name": name}],  # intraday: the hourly run covers it
        }.get(name, [])  # no open German session


def test_not_ready_only_lists_our_uncovered_challenges():
    now = pd.Timestamp("2026-11-15 13:30", tz="UTC")
    assert not_ready(_Client(), now) == ["spain_mfrr_price", BE]  # "x": someone else's target


def test_in_session_is_the_hour_before_each_gate_and_always_for_intraday():
    from openpredico.live.submit import in_session

    def at(hm):
        return in_session(pd.Timestamp(f"2026-11-09 {hm}", tz="Europe/Madrid"))

    # gate 15:00 Berlin = 15:00 Madrid
    assert at("14:15") == ["germany_wind_50hertz_ca", "germany_solar_50hertz_ca", BE]
    assert set(at("16:15")) == {"spain_mfrr_energy", "spain_mfrr_price", BE}
    assert at("17:00") == [BE] and at("13:59") == [BE]


def test_not_run_lists_session_targets_without_a_run_today_or_this_hour():
    from openpredico.live.submit import not_run

    now = pd.Timestamp("2026-11-09 16:30", tz="Europe/Madrid").tz_convert("UTC")
    files = [
        "forecasts/spain_mfrr_energy/2026-11-09T1515Z.parquet",
        "forecasts/spain_mfrr_price/2026-11-08T1515Z.parquet",  # yesterday's
        f"intraday/forecasts/{BE}/2026-11-09T1415Z.parquet",  # last hour's
    ]
    assert not_run(now, files) == ["spain_mfrr_price", BE]
    files.append(f"intraday/forecasts/{BE}/2026-11-09T1515Z.parquet")
    assert not_run(now, files) == ["spain_mfrr_price"]


def test_missing_never_forecasts_a_challenge_before_its_session(monkeypatch):
    from openpredico.live import submit

    class Api:
        def list_repo_files(self, *a, **k):
            return []

    monkeypatch.setattr(submit, "HfApi", Api)
    monkeypatch.setattr(submit, "pair_needs_retry", lambda *a: False)
    now = pd.Timestamp("2026-11-15 12:30", tz="Europe/Madrid").tz_convert("UTC")
    assert "spain_mfrr_price" not in submit.missing(_Client(), now)  # open, not ready, gate 17:00
    now = pd.Timestamp("2026-11-15 16:30", tz="Europe/Madrid").tz_convert("UTC")
    assert {"spain_mfrr_energy", "spain_mfrr_price"} <= set(submit.missing(_Client(), now))
