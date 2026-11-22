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


def test_incremental_window_reads_only_the_runs_predico_sliced():
    from openpredico.live.publish import WINDOW_DAYS, runs_to_read, window_start

    now = pd.Timestamp("2026-11-20 17:00", tz="UTC")  # Nov 21 gate (16:00 UTC) passed
    published = pd.date_range("2026-11-01", "2026-11-20 23:45", freq="15min", tz=TZ)
    start = window_start(published.tz_convert("UTC"), TZ, now)
    assert start == pd.Timestamp("2026-11-20", tz=TZ) - pd.DateOffset(days=WINDOW_DAYS)
    assert window_start(None, TZ, now) is None  # nothing published: rebuild all
    stalled = window_start(published[:96].tz_convert("UTC"), TZ, now)  # last publish: Nov 1
    assert stalled == pd.Timestamp("2026-11-02", tz=TZ)

    stamps = [pd.Timestamp(f"2026-11-{d:02d} 15:15", tz="UTC") for d in range(1, 21)]
    stamps.remove(pd.Timestamp("2026-11-17 15:15", tz="UTC"))  # failed: Nov 18 takes Nov 16's
    stamps.append(pd.Timestamp("2026-11-20 16:30", tz="UTC"))  # after the gate: never sliced
    got = runs_to_read(stamps, start, TZ, "17:00", now)
    lo, hi = pd.Timestamp("2026-11-12", tz="UTC"), pd.Timestamp("2026-11-20 16:00", tz="UTC")
    assert got == {s for s in stamps if lo <= s < hi}  # Nov 13..21 from the Nov 12..20 runs
