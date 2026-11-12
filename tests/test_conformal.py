import pandas as pd
import pytest

from openpredico.models.conformal import calibrate, day_grid, total_point


def history(first="2026-09-14"):
    parts = []
    for day in pd.date_range(first, periods=40).strftime("%Y-%m-%d"):
        grid = day_grid(day)
        parts.append(
            pd.DataFrame(
                {
                    "point_mw": 100.0,
                    "actual_mw": 80.0,
                    "issue_time_utc": grid.min() - pd.Timedelta(hours=12),
                    "actual_available_at": grid.max() + pd.Timedelta(hours=2),
                    "pipeline": "v1",
                    "truth_source": "test",
                },
                index=grid,
            )
        )
    return pd.concat(parts)


def run(h):
    return calibrate(
        pd.Series(100.0, index=day_grid("2026-10-25")),
        h,
        origin=pd.Timestamp("2026-10-24T11:30Z"),
        pipeline="v1",
        truth_source="test",
    )


def test_known_signed_errors_and_dst():
    fc, audit = run(history())
    assert len(fc) == 100 and fc.q10.eq(80).all() and fc.q50.eq(100).all()
    assert fc.q90.eq(100).all() and audit["calibration_days"] == 40


def test_unavailable_or_other_pipeline_actuals_do_not_change_output():
    h = history()
    first, _ = run(h)
    extra = h.copy()
    extra["actual_mw"] = 1000000.0
    extra["actual_available_at"] = pd.Timestamp("2026-10-26T00:00Z")
    # Replace five days by unavailable revisions: fewer days but same eligible scores.
    h.loc[h.index < day_grid("2026-09-19")[0]] = extra
    changed, audit = run(h)
    pd.testing.assert_frame_equal(first, changed)
    assert audit["calibration_days"] == 35
    h["pipeline"] = "other"
    with pytest.raises(ValueError, match="insufficient"):
        run(h)


def test_missing_truth_is_not_filled_and_no_warmup_guess():
    h = history().iloc[: 29 * 96].copy()
    with pytest.raises(ValueError, match="insufficient"):
        run(h)
    h = history()
    h.iloc[0, h.columns.get_loc("actual_mw")] = float("nan")
    _, audit = run(h)
    assert audit["calibration_days"] == 39


def test_component_grid_and_quantile_sum_safety():
    grid = day_grid("2026-10-25")
    summed = total_point(pd.Series(60.0, index=grid), pd.Series(40.0, index=grid))
    assert summed.point_mw.eq(100).all()
    with pytest.raises(ValueError, match="grids differ"):
        total_point(pd.Series(60.0, index=grid), pd.Series(40.0, index=grid[:-1]))


def test_finite_sample_rank_not_interpolated_percentile():
    h = history().iloc[-30 * 96 :].copy()
    days = h.index.tz_convert("Europe/Berlin").strftime("%Y-%m-%d")
    for n, day in enumerate(sorted(set(days)), 1):
        h.loc[days == day, "actual_mw"] = 100 - n
    fc, _ = run(h)
    # ceil((30+1)*0.9) = 28th of the signed scores 1..30.
    assert fc.q10.eq(72).all()
    h["truth_source"] = "another_truth"
    with pytest.raises(ValueError, match="insufficient"):
        run(h)


def test_spring_dst_day_has_92_ordered_quantiles():
    point = pd.Series(100.0, index=day_grid("2027-03-28"))
    fc, _ = calibrate(
        point,
        history("2027-02-15"),
        origin=pd.Timestamp("2027-03-27T12:30Z"),
        pipeline="v1",
        truth_source="test",
    )
    assert len(fc) == 92
    assert (fc.q10 <= fc.q50).all() and (fc.q50 <= fc.q90).all()


def test_stale_history_fails_instead_of_reusing_old_intervals():
    h = history()
    h.loc[h.index >= day_grid("2026-10-20")[0], "actual_available_at"] = pd.Timestamp(
        "2026-10-26T00:00Z"
    )
    with pytest.raises(ValueError, match="stale"):
        run(h)
