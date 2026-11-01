import numpy as np
import pandas as pd

from openpredico.evaluation.results import score


def test_score_uses_obs_column_as_truth():
    idx = pd.date_range("2026-10-30 23:00", periods=96, freq="15min", tz="UTC")  # one local day
    preds = pd.DataFrame({"q10": 0.0, "q50": 1.0, "q90": 2.0, "obs": 3.0}, index=idx)
    s = score(preds)
    assert len(s) == 1
    assert np.isclose(s.iloc[0].rmse, 2.0)
    assert np.isclose(s.iloc[0].winkler, 2 + 10)
