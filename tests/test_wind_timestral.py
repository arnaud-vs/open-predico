from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openpredico.models.conformal import day_grid
from openpredico.models.wind_timestral import MODEL, REVISION, load, point


def test_load_pins_tp_checkpoint(monkeypatch):
    import sys

    calls = []
    fake = SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setitem(sys.modules, "timestral", SimpleNamespace(TimestralForecaster=fake))
    load(local_files_only=True)
    args, kw = calls[0]
    assert args == (MODEL,) and kw["revision"] == REVISION
    assert kw["local_files_only"] is True


@pytest.mark.parametrize("day", ["2026-10-25", "2026-03-29", "2026-03-30"])
def test_actual_cutoff_dst_and_future_weather(day, monkeypatch):
    import sys
    from contextlib import nullcontext

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    previous = (pd.Timestamp(day) - pd.DateOffset(days=1)).date()
    origin = pd.Timestamp(f"{previous} 13:30", tz="Europe/Berlin").tz_convert("UTC")
    cut = origin - pd.Timedelta(hours=2)
    grid = pd.date_range(cut - pd.Timedelta(days=92), day_grid(day)[-1], freq="15min")
    actual = pd.Series(100.0, index=grid)
    weather = pd.DataFrame(1.0, index=grid, columns=[f"w{i}" for i in range(24)])
    contexts = []

    class Fake:
        def forecast(self, y, prediction_length, known_covariates, **kwargs):
            contexts.append(y.copy())
            assert y.shape == (1, 1, 8832)
            assert known_covariates.shape == (1, 24, 8832 + prediction_length)
            return SimpleNamespace(median=np.full((1, 1, prediction_length), 123.0))

    first = point(Fake(), actual, weather, day=day, origin=origin)
    actual.loc[actual.index >= cut] = 1e9
    second = point(Fake(), actual, weather, day=day, origin=origin)
    pd.testing.assert_series_equal(first, second)
    np.testing.assert_array_equal(contexts[0], contexts[1])
    assert first.index.equals(day_grid(day))
    weather.loc[weather.index >= cut, "w0"] = np.nan
    with pytest.raises(ValueError, match="future weather"):
        point(Fake(), actual, weather, day=day, origin=origin)
