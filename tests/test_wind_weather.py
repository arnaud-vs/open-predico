from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openpredico.data import wind_weather as weather


def record():
    return {
        "model": "icon_eu",
        "responses": [
            {
                "hourly": {
                    "time": ["2026-11-11T06:00", "2026-11-11T07:00"],
                    "wind_speed_80m": [10.0, 10.0],
                    "wind_speed_120m": [10.0, 10.0],
                    "wind_direction_120m": [350.0, 10.0],
                    "temperature_80m": [8.0, 10.0],
                    "temperature_120m": [6.0, 8.0],
                },
                "hourly_units": {
                    "wind_speed_80m": "m/s",
                    "temperature_80m": "°C",
                    "temperature_120m": "°C",
                },
            }
        ],
    }


def test_decode_preserves_point_channels_and_circular_direction():
    out = weather.decode_run(record())
    assert out.shape == (5, 4)
    np.testing.assert_allclose(out.speed_p1, 10.0)
    assert out.t100_p1.iloc[0] == 7.0 and out.t100_p1.iloc[-1] == 9.0
    assert abs(out.sin_p1.iloc[2]) < 1e-12 and out.cos_p1.iloc[2] > 0.98
    assert str(out.index.tz) == "UTC"


def test_runs_use_prior_local_calendar_date_across_dst():
    assert weather.run_time("2026-03-30", "icon_eu") == pd.Timestamp("2026-03-29T06:00Z")
    assert weather.run_time("2026-10-26", "gfs_global") == pd.Timestamp("2026-10-25T00:00Z")


def test_commercial_guard_and_error_never_echo_key(monkeypatch):
    with pytest.raises(ValueError, match="licensed"):
        weather.fetch_run("2026-11-12", "icon_eu", [(54.0, 13.0)], commercial=True)
    monkeypatch.setattr(
        weather.requests, "get", lambda *a, **kw: SimpleNamespace(ok=False, status_code=503)
    )
    with pytest.raises(RuntimeError) as error:
        weather.fetch_run("2026-11-12", "icon_eu", [(54.0, 13.0)], api_key="private-test-key")
    assert str(error.value) == "point-weather HTTP 503"


def test_blend_orders_and_missing_sources_keep_eu_fallback():
    raw = np.ones((2, 6, 4, 4))
    raw[:, :, 0, 0] = np.arange(6)[None, :] + 5
    raw[:, :, 0, 3] = 12
    bank = np.zeros((6, 3, 4))
    bank[:, :, 0] = 1
    onshore = weather.blend(raw, bank, component="onshore")
    offshore = weather.blend(raw, bank, component="offshore")
    np.testing.assert_array_equal(onshore[:, 0::4], raw[:, :, 0, 0])
    np.testing.assert_array_equal(offshore[:, :6], raw[:, :, 0, 0])
    assert (onshore[:, 3::4] == 12).all() and (offshore[:, 18:] == 12).all()
    raw[0, 0, 0, :3] = np.nan
    assert np.isnan(weather.blend(raw, bank, component="offshore")[0, 0])
