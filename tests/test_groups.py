import pandas as pd
import pytest

from openpredico.data.groups import (
    dst_switch_day,
    holiday_de,
    holiday_es,
    load_groups,
    to_grid,
)


def test_repo_groups_are_valid():
    assert {"target", "da_market", "calendar"} <= load_groups().keys()


@pytest.mark.parametrize(
    "toml, error",
    [
        ('[a]\nx = { source = "esios:1", availability = "later" }\n', "availability"),
        ('[a]\nx = { source = "entsoe:1", availability = "past" }\n', "prefix"),
        (
            '[a]\nx = { source = "esios:1", availability = "past" }\n'
            '[b]\nx = { source = "esios:2", availability = "past" }\n',
            "duplicate column",
        ),
        (
            '[a]\nx = { source = "esios:1", geo = 3, availability = "past" }\n'
            '[b]\ny = { source = "esios:1", geo = 3, availability = "past" }\n',
            "duplicate source",
        ),
    ],
)
def test_invalid_groups_are_rejected(tmp_path, toml, error):
    path = tmp_path / "groups.toml"
    path.write_text(toml)
    with pytest.raises(ValueError, match=error):
        load_groups(path)


def test_to_grid_repeats_hourly_until_switch_and_keeps_gaps():
    grid = pd.date_range("2026-01-01 21:00", periods=12, freq="15min", tz="UTC")
    s = pd.Series(
        [1.0, 2.0, 3.0, 4.0],  # hourly, then 15-min from local midnight (23:00 UTC), with gaps
        index=pd.to_datetime(
            ["2026-01-01 21:00", "2026-01-01 22:00", "2026-01-01 23:00", "2026-01-01 23:30"],
            utc=True,
        ),
    )
    out = to_grid(s, grid)
    assert list(out.iloc[:8]) == [1.0] * 4 + [2.0] * 4
    assert out.iloc[8] == 3.0 and pd.isna(out.iloc[9]) and out.iloc[10] == 4.0
    assert list(out.to_frame("x").fillna({"x": 0}).x.iloc[9:]) == [0.0, 4.0, 0.0]


def test_calendar_holiday_and_dst():
    days = pd.to_datetime(
        [
            "2026-04-03 10:00",
            "2026-04-02 10:00",
            "2026-03-29 10:00",
            "2026-10-25 10:00",
            "2026-10-26 10:00",
            "2026-12-08 10:00",
        ],
        utc=True,
    )
    assert list(holiday_es(days)) == [1, 0, 0, 0, 0, 1]  # Good Friday, Immaculate Conception
    assert list(dst_switch_day(days)) == [0, 0, 1, 1, 0, 0]


def test_holiday_de_uses_berlin_dates():
    days = pd.to_datetime(
        ["2026-05-14 10:00", "2026-05-25 10:00", "2026-10-02 22:30", "2026-10-12 10:00"],
        utc=True,
    )  # Ascension, Whit Monday, 3 Oct 00:30 Berlin (German Unity), 12 Oct (Spanish only)
    assert list(holiday_de(days)) == [1, 1, 1, 0]
