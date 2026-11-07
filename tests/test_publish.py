import pandas as pd

from openpredico.live.publish import scored

TZ = "Europe/Madrid"


def _run(day: str, value: float) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(day, tz=TZ), periods=2 * 96, freq="15min").tz_convert("UTC")
    return pd.DataFrame({"q50": value}, index=idx)  # covers `day` and the day after


def test_scored_takes_latest_run_before_each_gate_and_hides_open_days():
    runs = {
        pd.Timestamp("2026-11-03 15:00", tz="UTC"): _run("2026-11-04", 1.0),  # before Nov 4 gate
        pd.Timestamp("2026-11-03 16:30", tz="UTC"): _run("2026-11-04", 2.0),  # after it: too late
        pd.Timestamp("2026-11-04 15:00", tz="UTC"): _run("2026-11-05", 3.0),
    }
    now = pd.Timestamp("2026-11-04 17:00", tz="UTC")  # Nov 5 gate (16:00 UTC) passed, Nov 6 not
    out = scored(runs, TZ, "17:00", now)
    day = out.index.tz_convert(TZ).normalize().strftime("%m-%d")
    assert (out.q50[day == "11-04"] == 1).all()  # the 16:30 run missed the gate
    assert (out.q50[day == "11-05"] == 3).all()  # latest run before the Nov 5 gate
    assert set(day) == {"11-04", "11-05"}  # Nov 6 stays private until its gate


def test_gate_closes_is_d_minus_1_at_gate_local():
    from openpredico.live.publish import gate_closes

    idx = pd.DatetimeIndex(["2026-11-06 23:00", "2026-11-07 22:45"], tz="UTC")  # Nov 7 Madrid
    closes = gate_closes(idx, TZ, "17:00")
    assert (closes == pd.Timestamp("2026-11-06 17:00", tz=TZ)).all()
