import pandas as pd

from openpredico.live.submit import not_ready


class _Client:
    def coverage(self):
        rows = [{"challenge_id": "e", "ready": True}, {"challenge_id": "p", "ready": False}]
        return {"data": {"coverage": rows + [{"challenge_id": "x", "ready": False}]}}

    def open_challenges(self, name):
        return {
            "spain_mfrr_energy": [{"id": "e", "resource_name": name}],
            "spain_mfrr_price": [{"id": "p", "resource_name": name}],
        }.get(name, [])  # no open German session


def test_not_ready_only_lists_our_uncovered_challenges():
    assert not_ready(_Client()) == ["spain_mfrr_price"]  # "x" belongs to someone else's target


def test_in_session_is_the_hour_before_each_gate():
    from openpredico.live.submit import in_session

    def at(hm):
        return in_session(pd.Timestamp(f"2026-11-09 {hm}", tz="Europe/Madrid"))

    # gate 15:00 Berlin = 15:00 Madrid
    assert at("14:15") == ["germany_wind_50hertz_ca", "germany_solar_50hertz_ca"]
    assert set(at("16:15")) == {"spain_mfrr_energy", "spain_mfrr_price"}
    assert at("17:00") == [] and at("13:59") == []
