"""Two-model TP++ recipe: learned onshore, mean offshore; no production side effects."""

import gc
from types import SimpleNamespace

import numpy as np
import pandas as pd

from openpredico.models import wind_timestral

T0_MODEL = "theforecastingcompany/t0-gamma"
T0_REVISION = "e08aee083778f1f32da4974c0a34276776a35c52"
MEMBERS = ("t0", "timestral")


def weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The tested TP++ constrained squared-error objective and settings."""
    from scipy.optimize import minimize

    equal = np.full(2, 0.5)
    scale = float(np.std(y))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.mean(np.abs(y))), 1.0)
    x, y = x / scale, y / scale

    def objective(w):
        return float(np.mean((y - x @ w) ** 2) + 0.1 * np.sum((w - equal) ** 2))

    fit = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(0, 0.7)] * 2,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return fit.x if fit.success and np.isfinite(fit.x).all() else equal


def ensemble(members: pd.DataFrame, truth: pd.Series, day: str, origin: pd.Timestamp):
    """Only complete training days ending before issue−2h; equal-weight cold start."""
    from openpredico.models.conformal import TZ, day_grid

    grid = day_grid(day)
    current = members.reindex(grid)
    columns = [f"{m}_{t}" for m in MEMBERS for t in ("onshore", "offshore")]
    if not np.isfinite(current[columns]).all().all():
        raise ValueError("incomplete current pair members")
    days = members.index.tz_convert(TZ).strftime("%Y-%m-%d")
    eligible = []
    for date, g in members.groupby(days, sort=True):
        idx = day_grid(date)
        if idx[-1] + pd.Timedelta(minutes=15) > origin - pd.Timedelta(hours=2):
            continue
        if not g.index.equals(idx) or not truth.reindex(idx).notna().all():
            continue
        if not np.isfinite(g[columns]).all().all():
            continue
        if not (pd.to_datetime(g.issue_time_utc, utc=True) < g.index.min()).all():
            continue
        eligible.append(g)
    eligible = eligible[-60:]
    w = np.full((4, 2), 0.5)
    if len(eligible) >= 21:
        train = pd.concat(eligible)
        bands = train.index.tz_convert(TZ).hour // 6
        for band in range(4):
            g = train[bands == band]
            w[band] = weights(
                g[["t0_onshore", "timestral_onshore"]].to_numpy(float),
                truth.reindex(g.index).to_numpy(float),
            )
    bands = grid.tz_convert(TZ).hour // 6
    on = np.einsum("nm,nm->n", current[["t0_onshore", "timestral_onshore"]], w[bands])
    off = current[["t0_offshore", "timestral_offshore"]].mean(axis=1)
    out = pd.DataFrame({"onshore_point_mw": on, "offshore_point_mw": off}, index=grid)
    out["point_mw"] = out.sum(axis=1)
    return out, {
        "training_days": len(eligible),
        "onshore_weights": w.tolist(),
        "last_training_time": eligible[-1].index.max().isoformat() if eligible else None,
    }


class _T0:
    def __init__(self, model):
        self.model = model

    def forecast(self, y, prediction_length, known_covariates, **kwargs):
        fc = self.model.predict(
            y,
            horizon=prediction_length,
            future_covariates=known_covariates,
            quantile_levels=[0.1, 0.5, 0.9],
            trim_leading_missing=True,
            symmetric_averaging=True,
            clip_nonnegative=True,
            snap_integers=True,
        )
        return SimpleNamespace(median=fc.median)


def predict(days, inputs, actuals, device="cpu", cache_dir=None):
    """Load one engine at a time; return all four member Q50s on aligned delivery grids."""
    import torch

    outputs = {}
    for member in MEMBERS:
        if member == "t0":
            from t0 import T0Forecaster

            engine = (
                T0Forecaster.from_pretrained(
                    T0_MODEL, revision=T0_REVISION, cache_dir=cache_dir, local_files_only=True
                )
                .eval()
                .to(device)
            )
            model = _T0(engine)
        else:
            engine = wind_timestral.load(device=device, cache_dir=cache_dir, local_files_only=True)
            model = engine
        for day, origin in days:
            for component in ("onshore", "offshore"):
                wx = inputs(day, component)
                point = wind_timestral.point(model, actuals[component], wx, day=day, origin=origin)
                outputs.setdefault(day, {})[f"{member}_{component}"] = point
        del model, engine
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    return {day: pd.DataFrame(columns) for day, columns in outputs.items()}
