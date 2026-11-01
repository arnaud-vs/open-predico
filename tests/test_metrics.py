import numpy as np
import pandas as pd

from openpredico.evaluation.metrics import daily_scores


def test_daily_scores_match_predico_formulas_and_skip_incomplete_days():
    idx = pd.date_range("2026-10-30 23:00", periods=2 * 96, freq="15min", tz="UTC")  # 2 local days
    fc = pd.DataFrame({"q10": 0.0, "q50": 1.0, "q90": 2.0}, index=idx)
    truth = pd.Series(1.0, index=idx)
    truth.iloc[0] = 3.0  # day 1: one step above q90 by 1 -> Winkler +10 at that step
    truth.iloc[100] = np.nan  # day 2 incomplete -> skipped

    s = daily_scores(fc, truth)
    assert len(s) == 1
    d = s.iloc[0]
    assert np.isclose(d.rmse, np.sqrt(4 / 96))  # one error of 2, squared, mean, then sqrt
    assert np.isclose(d.winkler, 2 + 10 / 96)  # width 2 everywhere + 10 * 1 once
    assert np.isclose(d.coverage, 95 / 96)
