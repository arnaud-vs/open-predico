import numpy as np
import pandas as pd

from openpredico.evaluation.backtest import backtest
from openpredico.models.tsfm.t0 import make_predict


class _Fake:  # stands in for T0Forecaster; returns crossing quantiles on purpose
    def predict(self, context, horizon, quantile_levels, future_covariates):
        _, v, t = context.shape
        assert v == 2  # target + past covariate
        assert future_covariates.shape == (1, 2, t + horizon)
        assert not np.isnan(future_covariates).any()  # known through the end of day D
        q = np.tile([3.0, 2.0, 1.0], (1, v, horizon, 1))
        return type("F", (), {"quantiles": q})()


def test_backtest_with_covariates_on_dst_day():
    idx = pd.date_range("2026-03-01", "2026-03-31", freq="15min", tz="UTC")
    hist = pd.DataFrame({"y": 1.0, "past": 2.0, "known": 3.0, "qh": idx.hour * 4}, index=idx)
    day = pd.date_range("2026-03-29", periods=1, tz="Europe/Madrid")  # 23 h day

    out = backtest(make_predict(_Fake()), hist, day, future_cols=["known", "qh"])
    assert len(out) == 92
    assert out.index[0] == day[0].tz_convert("UTC")
    assert (out.q10 <= out.q50).all() and (out.q50 <= out.q90).all()


def test_cut_past_applies_each_columns_own_lag():
    from openpredico.evaluation.backtest import cut_past

    idx = pd.date_range("2026-11-01", "2026-11-02", freq="15min", tz="UTC", inclusive="left")
    hist = pd.DataFrame({"solar_be": 1.0, "mfrr_energy": 2.0, "qh_of_day": 3.0}, index=idx)
    cutoff = pd.Timestamp("2026-11-01 12:00", tz="UTC")
    past = cut_past(hist, cutoff)
    assert past.index[-1] == pd.Timestamp("2026-11-01 11:15", tz="UTC")  # Elia: 30 min
    two_hours = pd.Timestamp("2026-11-01 09:45", tz="UTC")
    assert past.mfrr_energy.last_valid_index() == two_hours  # default lag
    assert past.qh_of_day.notna().all()  # future columns follow the rows
    assert cut_past(hist[["mfrr_energy"]], cutoff).index[-1] == two_hours
