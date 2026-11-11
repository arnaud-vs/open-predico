"""Explicit TP++ Timestral-209M point adapter for the optional six-point wind candidate.

No change to the generic Timestral adapter's 5M default or to the active submission model.
The caller supplies vintage-checked 24-channel point weather and acquired actual history.
"""

import numpy as np
import pandas as pd

from openpredico.models.conformal import TZ, day_grid, total_point

MODEL = "mistralai/Timestral-209M"
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"


def load(device: str = "cpu", cache_dir: str | None = None, local_files_only: bool = False):
    from timestral import TimestralForecaster

    if device == "mps":
        from openpredico.models.tsfm.timestral import _float64_on_cpu

        _float64_on_cpu()
    return TimestralForecaster.from_pretrained(
        MODEL,
        revision=REVISION,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def point(
    model,
    actual: pd.Series,
    weather: pd.DataFrame,
    *,
    day: str,
    origin: pd.Timestamp,
) -> pd.Series:
    """Native quarter-hour Q50, 92 days before origin−2h, full next local day.

    Future actuals are sliced off before model invocation. Historical missing values are
    passed through to the native model; future weather gaps fail instead of changing inputs.
    """
    import torch

    local = origin.tz_convert(TZ)
    delivery = day_grid(day)
    previous = (pd.Timestamp(day) - pd.DateOffset(days=1)).date()
    expected = pd.Timestamp(f"{previous} 13:30", tz=TZ)
    if local != expected:
        raise ValueError("the wind candidate requires its tested 13:30 D-1 origin")
    cut = origin - pd.Timedelta(hours=2)
    context = pd.date_range(cut - pd.Timedelta(days=92), cut, freq="15min", inclusive="left")
    bridge = pd.date_range(
        cut, delivery[-1] + pd.Timedelta(minutes=15), freq="15min", inclusive="left"
    )
    if actual.index.has_duplicates or weather.index.has_duplicates:
        raise ValueError("duplicate input timestamps")
    if str(actual.index.tz) != "UTC" or str(weather.index.tz) != "UTC":
        raise ValueError("inputs must use UTC")
    if weather.shape[1] != 24:
        raise ValueError("six points require exactly 24 weather channels")
    cov = weather.reindex(context.append(bridge)).to_numpy(dtype=np.float32)
    y = actual.reindex(context).to_numpy(dtype=np.float32)
    if np.isnan(y).mean() >= 0.01 or np.isnan(cov[: len(context)]).mean() >= 0.1:
        raise ValueError("insufficient historical inputs")
    if not np.isfinite(cov[len(context) :]).all():
        raise ValueError("future weather coverage is incomplete")
    if np.isinf(y).any() or np.isinf(cov).any():
        raise ValueError("infinite model input")
    with torch.inference_mode():
        fc = model.forecast(
            y[None, None, :],
            prediction_length=len(bridge),
            known_covariates=cov.T[None],
            symmetric_averaging=True,
            clip_nonnegative=True,
        )
    values = fc.median
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values).reshape(-1)
    if len(values) != len(bridge) or not np.isfinite(values).all():
        raise ValueError("invalid Timestral output")
    return pd.Series(np.maximum(values[-len(delivery) :], 0), index=delivery, name="point_mw")


def total(
    model,
    actuals: dict[str, pd.Series],
    weather: dict[str, pd.DataFrame],
    *,
    day: str,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Two independent component forecasts; total quantiles belong to conformal.py."""
    components = {
        key: point(model, actuals[key], weather[key], day=day, origin=origin)
        for key in ("onshore", "offshore")
    }
    return total_point(components["onshore"], components["offshore"])
