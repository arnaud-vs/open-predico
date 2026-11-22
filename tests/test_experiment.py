import numpy as np
import pandas as pd

from openpredico.evaluation import experiment
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


def test_load_benchmark_takes_each_group_from_its_latest_version(monkeypatch):
    files = ["v1/target.parquet", "v1/calendar.parquet", "v2/calendar_de.parquet",
             "v10/calendar.parquet", "v2/notes.md", "predictions/x/target.parquet"]  # fmt: skip
    private = {"weather-private": ["v6/weather_be_wn3.parquet"]}

    class Denied(experiment.HfHubHTTPError):
        def __init__(self):
            Exception.__init__(self, "401")

    class Api:
        readable = True

        def list_repo_files(self, repo_id, **kwargs):
            if repo_id.endswith("private") and not Api.readable:
                raise Denied()
            return private.get(repo_id.split("/")[1], files)

    monkeypatch.setattr(experiment, "HfApi", Api)
    monkeypatch.setattr(experiment, "download", lambda repo, path: f"{repo}/{path}")
    assert experiment.load_benchmark() == {
        "target": "benchmark/v1/target.parquet",
        "calendar": "benchmark/v10/calendar.parquet",
        "calendar_de": "benchmark/v2/calendar_de.parquet",
        "weather_be_wn3": "weather-private/v6/weather_be_wn3.parquet",
    }
    Api.readable = False  # contributors: the licensed groups are left out
    assert "weather_be_wn3" not in experiment.load_benchmark()
