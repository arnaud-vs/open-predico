import numpy as np
import pandas as pd

from openpredico.evaluation.backtest import backtest
from openpredico.models.tsfm.timesfm import make_predict


class _Fake:  # stands in for TimesFM3Forecaster; returns crossing deciles on purpose
    def predict(self, context, horizon, past_only_covariates, past_future_covariates, **kw):
        t = len(context)
        assert past_only_covariates.shape == (1, t)
        assert past_future_covariates.shape == (2, t + horizon)
        assert not np.isnan(past_future_covariates).any()  # known through the end of day D
        return type("F", (), {"quantiles": np.tile(np.arange(9.0)[::-1], (horizon, 1))})()


def test_backtest_with_covariates_on_dst_day():
    idx = pd.date_range("2026-03-01", "2026-03-31", freq="15min", tz="UTC")
    hist = pd.DataFrame({"y": 1.0, "past": 2.0, "known": 3.0, "qh": idx.hour * 4}, index=idx)
    day = pd.date_range("2026-03-29", periods=1, tz="Europe/Madrid")  # 23 h day

    out = backtest(make_predict(_Fake()), hist, day, future_cols=["known", "qh"])
    assert len(out) == 92
    assert out.index[0] == day[0].tz_convert("UTC")
    assert (out.q10 == 0).all() and (out.q50 == 4).all() and (out.q90 == 8).all()
