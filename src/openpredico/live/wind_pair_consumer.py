"""Optional ready-pair handoff; any failure preserves the existing wind forecast path."""

import pandas as pd

from openpredico.config import load_targets
from openpredico.evaluation.backtest import clip
from openpredico.evaluation.experiment import TARGET_COLUMNS
from openpredico.live import wind_pair_bundle as bundle
from openpredico.models.baselines import slot_climatology
from openpredico.models.conformal import TZ


def ready(target, now):
    if target != bundle.TARGET:
        return None
    try:
        from openpredico.live.wind_pair import config

        cfg, sha = config()
        if not cfg["enabled"]:
            return None
        capacity = load_targets()[target]["capacity"]
        if capacity != cfg["capacity_mw"]:
            raise ValueError("candidate/target capacity mismatch")
        d1, inputs, audit = bundle.load(now, sha, capacity)
        origin = pd.Timestamp(audit["issue_time_utc"])
        y = inputs["wind_pair.onshore.actual"] + inputs["wind_pair.offshore.actual"]
        inputs[TARGET_COLUMNS[target]] = y
        start = pd.Timestamp(audit["day"], tz=TZ) + pd.DateOffset(days=1)
        days = pd.date_range(start, periods=9, freq="D")
        rest = slot_climatology(y[y.index < origin - pd.Timedelta(hours=2)], days, tz=TZ)
        fc = clip(
            pd.concat([d1, rest.assign(model="slot_climatology_28d")]), load_targets()[target]
        )
        fc = fc[fc.index <= now + pd.Timedelta(hours=240)]
        if (
            not fc.notna().all().all()
            or not (fc.q10 <= fc.q50).all()
            or not (fc.q50 <= fc.q90).all()
        ):
            raise ValueError("incomplete pair/tail forecast")
        if not (fc.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
            raise ValueError("pair/tail grid has gaps")
        return fc, inputs
    except Exception as error:
        # Never print token-bearing request URLs, response bodies or forecast values.
        print(f"{target}: wind-pair unavailable ({type(error).__name__}); retaining incumbent")
        return None


def needs_retry(now, files):
    """Upgrade a fallback run when a private pair becomes readable before wind's gate."""
    from openpredico.data import hf

    local = now.tz_convert(TZ)
    gate = pd.Timestamp(f"{local:%Y-%m-%d} {load_targets()[bundle.TARGET]['gate']}", tz=TZ)
    if not gate - pd.Timedelta(hours=1) <= local < gate:
        return False
    candidates = sorted(
        f for f in files if f.startswith(f"forecasts/{bundle.TARGET}/{now:%Y-%m-%d}T")
    )
    if not candidates:
        return False  # existing not_run already handles missing archives
    try:
        latest = hf.download("live-private", candidates[-1])
        if latest.model.iloc[0] == "wind_pair_t0_timestral_six_v1":
            return False
        return ready(bundle.TARGET, now) is not None
    except Exception as error:
        print(f"wind-pair retry check unavailable ({type(error).__name__})")
        return False
