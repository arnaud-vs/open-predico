"""Rolling one-sided conformal tails around an unchanged total-wind point forecast.

Calibration uses only complete, available historical days of the same pipeline and truth.
Temporal dependence means the nominal coverage is a target, not a distribution-free promise.
"""

import math

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"


def day_grid(day: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz=TZ)
    return (
        pd.date_range(start, start + pd.DateOffset(days=1), freq="15min", inclusive="left")
        .tz_convert("UTC")
        .as_unit("ns")
    )


def _grid(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise ValueError("a UTC DatetimeIndex is required")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("duplicate or unsorted timestamps")


def rank(n: int, tail_probability: float) -> int:
    """Finite-sample conformal rank: the k-th smallest of n scores bounds a new one with
    probability >= 1 - tail_probability (k > n: too few scores)."""
    return math.ceil((n + 1) * (1 - tail_probability))


def total_point(onshore: pd.Series, offshore: pd.Series) -> pd.DataFrame:
    """Add aligned point forecasts, never component quantiles."""
    if not onshore.index.equals(offshore.index):
        raise ValueError("component grids differ")
    out = pd.DataFrame({"onshore_point_mw": onshore, "offshore_point_mw": offshore})
    _grid(out)
    if out.isna().any().any() or not out.ge(0).all().all():
        raise ValueError("components must be finite, nonnegative point forecasts")
    if out.isin([float("inf"), -float("inf")]).any().any():
        raise ValueError("nonfinite components")
    out["point_mw"] = out.astype("float64").sum(axis=1)
    return out


def calibrate(
    point: pd.Series,
    history: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    pipeline: str,
    truth_source: str,
    window_days: int = 60,
    min_days: int = 30,
    tail_probability: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """Q10/Q90 from signed total errors; preserve the supplied Q50.

    history columns: point_mw, actual_mw, issue_time_utc, actual_available_at,
    pipeline, truth_source. `actual_available_at` records acquisition, not delivery time.
    For offline experiments an assumed availability must be explicitly labelled externally.
    Missing truth rejects the whole calibration day; no filling or partial-day inclusion.
    """
    _grid(point.to_frame())
    _grid(history)
    if origin.tzinfo is None or not 0 < tail_probability < 0.5:
        raise ValueError("timezone-aware origin and a valid tail probability are required")
    if not 1 <= min_days <= window_days:
        raise ValueError("invalid calibration window")
    if point.empty or point.isna().any() or not point.ge(0).all():
        raise ValueError("invalid point forecast")
    if point.isin([float("inf"), -float("inf")]).any():
        raise ValueError("invalid point forecast")
    target_day = str(point.index[0].tz_convert(TZ).date())
    if not point.index.equals(day_grid(target_day)) or point.index.min() <= origin:
        raise ValueError("forecast must cover one complete future local delivery day")
    cols = [
        "point_mw",
        "actual_mw",
        "issue_time_utc",
        "actual_available_at",
        "pipeline",
        "truth_source",
    ]
    h = history[cols].copy()
    h = h[h.pipeline.eq(pipeline) & h.truth_source.eq(truth_source)]
    for col in ["issue_time_utc", "actual_available_at"]:
        h[col] = pd.to_datetime(h[col], utc=True, errors="raise")
    h["day"] = h.index.tz_convert(TZ).strftime("%Y-%m-%d")
    eligible = []
    for day, g in h.groupby("day", sort=True):
        if (
            day >= target_day
            or pd.Timestamp(day) < pd.Timestamp(target_day) - pd.Timedelta(days=window_days + 7)
            or not g.index.equals(day_grid(day))
        ):
            continue
        if not g[["point_mw", "actual_mw"]].notna().all().all():
            continue
        if g[["point_mw", "actual_mw"]].isin([float("inf"), -float("inf")]).any().any():
            continue
        if not (g.actual_available_at < origin).all():
            continue
        if not (g.issue_time_utc < g.index.min()).all():
            continue
        if not (g.actual_available_at > g.index).all():
            continue
        eligible.append(g)
    if len(eligible) < min_days:
        raise ValueError("insufficient complete available calibration days")
    if pd.Timestamp(eligible[-1].day.iloc[0]) < pd.Timestamp(target_day) - pd.Timedelta(days=3):
        raise ValueError("stale calibration history")
    h = pd.concat(eligible[-window_days:])
    local = h.index.tz_convert(TZ)
    h["slot"] = local.hour * 4 + local.minute // 15
    h["lower_score"] = h.point_mw - h.actual_mw
    h["upper_score"] = -h.lower_score
    # Autumn DST duplicates a local slot: maximum per day gives each day one conservative score.
    scores = h.groupby(["day", "slot"])[["lower_score", "upper_score"]].max()
    bounds = {}
    for slot, group in scores.groupby("slot"):
        if len(group) < min_days or len(group) < rank(len(group), tail_probability):
            raise ValueError("insufficient per-slot conformal calibration")
        k = rank(len(group), tail_probability)
        bounds[slot] = [
            max(0.0, group[c].sort_values().iloc[k - 1]) for c in ["lower_score", "upper_score"]
        ]
    local = point.index.tz_convert(TZ)
    slots = local.hour * 4 + local.minute // 15
    if any(slot not in bounds for slot in slots):
        raise ValueError("unseen forecast slot")
    out = pd.DataFrame({"q50": point})
    out["q10"] = (point - [bounds[s][0] for s in slots]).clip(lower=0)
    out["q90"] = point + [bounds[s][1] for s in slots]
    return out[["q10", "q50", "q90"]], {
        "method": "rolling_asymmetric_one_sided_conformal",
        "tail_probability": tail_probability,
        "calibration_days": h.day.nunique(),
        "first_calibration_day": h.day.min(),
        "last_calibration_day": h.day.max(),
        "latest_actual_available_at": h.actual_available_at.max().isoformat(),
        "pipeline": pipeline,
        "truth_source": truth_source,
    }


def recalibrate(
    preds: pd.DataFrame,
    tz: str,
    *,
    band: bool = True,
    window_days: int = 60,
    min_days: int = 30,
    tail_probability: float = 0.1,
) -> pd.DataFrame:
    """Backtest predictions (q10, q50, q90, obs on a UTC index) with rolling conformal Q10/Q90.

    Same rule as `calibrate` (per local slot, one-sided signed scores, the `rank`-th of the last
    `window_days` complete days), applied to a whole backtest at once. `band=True` conformalizes
    the model's own Q10/Q90 (CQR: each bound moves by its score, so too-wide bands narrow);
    `band=False` rebuilds them around Q50 as the wind pair does. Q50 is unchanged. Slots with
    fewer than `min_days` scores keep the model's own Q10/Q90. Not clipped to capacity.
    """
    # ponytail: day D calibrates on days up to D-2 (D-1 is incomplete at a D-1 origin); a
    # same-day (intraday) origin would need the truth lag instead.
    lo, hi = (preds.q10, preds.q90) if band else (preds.q50, preds.q50)
    local = preds.index.tz_convert(tz)
    day = pd.Series(local.tz_localize(None).normalize(), index=preds.index)
    frame = pd.DataFrame(
        {"day": day, "slot": local.hour * 4 + local.minute // 15, "lower": lo - preds.obs}
    )
    frame["upper"] = preds.obs - hi
    complete = preds[["q10", "q50", "q90", "obs"]].notna().all(axis=1).groupby(day).all()
    # Autumn DST repeats a local slot: the larger score, one per day and slot (as `calibrate`).
    scores = frame[frame.day.map(complete)].groupby(["day", "slot"])[["lower", "upper"]].max()
    bounds = []
    for side in ("lower", "upper"):
        table = scores[side].unstack("slot")
        rows = {}
        for d in day.unique():
            past = np.sort(table.loc[: d - pd.Timedelta(days=2)].tail(window_days), axis=0)
            n = (~np.isnan(past)).sum(axis=0)  # np.sort puts NaN last
            k = np.array([rank(m, tail_probability) for m in n], dtype=int)
            ok = (n >= min_days) & (k <= n)
            if not len(past):
                continue
            pick = past[np.clip(k - 1, 0, len(past) - 1), np.arange(len(n))]
            rows[d] = np.where(ok, pick, np.nan)
        bounds.append(pd.DataFrame(rows, index=table.columns).T.stack().rename(side))
    b = pd.concat(bounds, axis=1).reindex(pd.MultiIndex.from_arrays([day, frame.slot]))
    out = preds.copy()
    out["q10"] = (lo - b.lower.to_numpy()).fillna(preds.q10).clip(lower=0)
    out["q90"] = (hi + b.upper.to_numpy()).fillna(preds.q90)
    out["q10"] = out[["q10", "q50"]].min(axis=1)
    out["q90"] = out[["q90", "q50"]].max(axis=1)
    return out


if __name__ == "__main__":
    # python -m openpredico.models.conformal <config> <prediction dir> [band|point]: writes
    # <config>_cqr (band) or <config>_cp (point) next to it, for evaluation.results.
    import sys
    from pathlib import Path

    from openpredico.config import load_targets
    from openpredico.evaluation.backtest import clip

    config, root, mode = sys.argv[1], Path(sys.argv[2]), (sys.argv[3:] or ["band"])[0]
    out = root / f"{config}_{'cqr' if mode == 'band' else 'cp'}"
    out.mkdir(exist_ok=True)
    for f in (root / config).glob("*.parquet"):
        t = load_targets()[f.stem]
        p = pd.read_parquet(f)
        r = recalibrate(p, t["timezone"], band=mode == "band")
        clip(r[["q10", "q50", "q90"]], t).assign(obs=r.obs).to_parquet(out / f.name)
