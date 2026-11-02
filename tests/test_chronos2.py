import numpy as np
import pandas as pd

from openpredico.evaluation.backtest import backtest
from openpredico.models.tsfm.chronos2 import make_predict


class _Fake:  # stands in for Chronos2Pipeline; returns crossing quantiles on purpose
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        (x,) = inputs
        t = len(x["target"])
        assert {k: len(v) for k, v in x["past_covariates"].items()} == dict.fromkeys(
            ["past", "known", "qh"], t
        )
        assert set(x["future_covariates"]) == {"known", "qh"}
        assert x["future_covariates"]["qh"].dtype.kind == "U"  # categorical as strings
        assert not np.isnan(x["future_covariates"]["known"]).any()
        assert len(x["future_covariates"]["known"]) == prediction_length
        return [np.tile([3.0, 2.0, 1.0], (1, prediction_length, 1))], None


def test_backtest_with_covariates_on_dst_day():
    idx = pd.date_range("2026-03-01", "2026-03-31", freq="15min", tz="UTC")
    hist = pd.DataFrame({"y": 1.0, "past": 2.0, "known": 3.0, "qh": idx.hour * 4}, index=idx)
    day = pd.date_range("2026-03-29", periods=1, tz="Europe/Madrid")  # 23 h day

    predict = make_predict(_Fake(), categorical=["qh"])
    out = backtest(predict, hist, day, future_cols=["known", "qh"])
    assert len(out) == 92
    assert out.index[0] == day[0].tz_convert("UTC")
    assert (out.q10 <= out.q50).all() and (out.q50 <= out.q90).all()
