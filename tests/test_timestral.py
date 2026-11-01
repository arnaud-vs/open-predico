import numpy as np
import pandas as pd
import pytest

from openpredico.evaluation.backtest import backtest
from openpredico.models.tsfm.timestral import make_predict


class _Host(np.ndarray):  # a tensor already on the host
    def cpu(self):
        return self


class _Fake:  # stands in for TimestralForecaster; returns crossing quantiles on purpose
    def forecast(self, context, prediction_length, **cov):
        t = len(context)
        assert cov["past_covariates"].shape == (1, 1, t)
        assert cov["known_covariates"].shape == (1, 1, t + prediction_length)
        assert cov["categorical_known_covariates"].shape == (1, 1, t + prediction_length)
        assert not np.isnan(cov["known_covariates"]).any()  # known through the end of day D
        levels = {0.1: 3.0, 0.5: 2.0, 0.9: 1.0}
        q = lambda p: np.full((1, 1, prediction_length), levels[p]).view(_Host)  # noqa: E731
        return type("F", (), {"quantile": staticmethod(q)})()


def test_backtest_with_covariates_on_dst_day():
    idx = pd.date_range("2026-03-01", "2026-03-31", freq="15min", tz="UTC")
    hist = pd.DataFrame({"y": 1.0, "past": 2.0, "known": 3.0, "qh": idx.hour * 4}, index=idx)
    day = pd.date_range("2026-03-29", periods=1, tz="Europe/Madrid")  # 23 h day

    predict = make_predict(_Fake(), categorical=["qh"])
    out = backtest(predict, hist, day, future_cols=["known", "qh"])
    assert len(out) == 92
    assert out.index[0] == day[0].tz_convert("UTC")
    assert (out.q10 <= out.q50).all() and (out.q50 <= out.q90).all()


def test_mps_matches_cpu():  # downloads Timestral-5M; runs only on a Mac with timestral installed
    torch = pytest.importorskip("torch")
    pytest.importorskip("timestral")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    from openpredico.models.tsfm.timestral import load

    rng = np.random.default_rng(0)
    t, h = 96 * 30, 136
    kw = {
        "past_covariates": rng.random((1, 2, t)),
        "known_covariates": rng.random((1, 1, t + h)),
        "categorical_known_covariates": (np.arange(t + h) % 96)[None, None].astype(float),
    }
    y = rng.random(t) * 100
    cpu, mps = (load(device=d).forecast(y, h, **kw).quantiles.cpu().numpy() for d in ("cpu", "mps"))
    np.testing.assert_allclose(mps, cpu, atol=1e-2)
