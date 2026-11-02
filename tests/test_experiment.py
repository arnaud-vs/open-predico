import numpy as np
import pandas as pd

from openpredico.evaluation.experiment import features


def test_features_split_fill_and_drop():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    frames = {
        "target": pd.DataFrame({"mfrr_energy": 1.0, "mfrr_price": [np.nan, 2, 3, 4]}, index=idx),
        "mfrr": pd.DataFrame(
            {"mfrr_up": [np.nan, 1, 2, 3], "mfrr_direct_requested_up": np.nan}, index=idx
        ),
        "da_market": pd.DataFrame({"da_price_es": 5.0}, index=idx),
    }
    df, future = features(frames, "spain_mfrr_price")
    assert list(df) == ["mfrr_price", "mfrr_energy", "mfrr_up", "da_price_es"]
    assert future == ["da_price_es"]
    assert df.mfrr_up.iloc[0] == 0  # sparse mFRR: missing = no activation
    assert np.isnan(df.mfrr_price.iloc[0])  # target never filled


def test_features_univariate_and_ablation():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    frames = {
        "target": pd.DataFrame({"mfrr_energy": 1.0, "mfrr_price": 2.0}, index=idx),
        "da_market": pd.DataFrame({"da_price_es": 5.0, "x": 6.0}, index=idx),
    }
    df, future = features(frames, "spain_mfrr_energy", "all")
    assert list(df) == ["mfrr_energy"] and future == []
    df, future = features(frames, "spain_mfrr_energy", "x")
    assert "x" not in df and "da_price_es" in df
