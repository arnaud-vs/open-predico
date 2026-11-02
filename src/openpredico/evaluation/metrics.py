"""Daily scores, matching Predico's evaluation (https://predico.inesctec.pt/docs/evaluation/).

Official: RMSE on q50 (sqrt of the day's mean squared error) and Winkler on q10/q90 (alpha=0.2,
mean over the day, not normalised). Quarter-hours without truth (e.g. no mFRR activation, so no
price) are left out of the day's means; a day with no truth at all is skipped.
ponytail: Predico's handling of missing truth is unconfirmed; ask them and match it.
Diagnostics: MAE, mean pinball over the three quantiles, q10-q90 coverage.
"""

import numpy as np
import pandas as pd

ALPHA = 0.2
QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def daily_scores(fc: pd.DataFrame, truth: pd.Series, tz: str = "Europe/Madrid") -> pd.DataFrame:
    """One row per local target day: rmse, winkler, mae, pinball, coverage.

    `fc` has q10/q50/q90 columns on a UTC index; `truth` is aligned by timestamp.
    """
    df = fc.join(truth.rename("obs"), how="left")
    day = df.index.tz_convert(tz).normalize().rename("day")
    o, q10, q50, q90 = df.obs, df.q10, df.q50, df.q90
    per_step = pd.DataFrame(
        {
            "se": (o - q50) ** 2,
            "winkler": (q90 - q10)
            + (2 / ALPHA) * np.maximum(0, q10 - o)
            + (2 / ALPHA) * np.maximum(0, o - q90),
            "ae": (o - q50).abs(),
            "pinball": sum(_pinball(o, df[c], q) for c, q in QUANTILES.items()) / len(QUANTILES),
            "covered": ((o >= q10) & (o <= q90)).astype(float).where(o.notna()),
        },
        index=df.index,
    )
    g = per_step.groupby(day)
    out = pd.DataFrame(
        {
            "rmse": np.sqrt(g.se.mean()),
            "winkler": g.winkler.mean(),
            "mae": g.ae.mean(),
            "pinball": g.pinball.mean(),
            "coverage": g.covered.mean(),
        }
    )
    return out[g.se.count() > 0]  # NaN steps drop out of the means


def _pinball(obs: pd.Series, pred: pd.Series, q: float) -> pd.Series:
    e = obs - pred
    return np.maximum(q * e, (q - 1) * e)
