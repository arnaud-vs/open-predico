"""Wind-pair producer: incremental sources -> member points -> ensemble -> conformal -> HF.

No Predico client: the existing submit job consumes only an independently verified bundle.
A failed producer never removes the incumbent or writes a ready partial forecast.
"""

import gc
import hashlib
import json
import os
import time
import tomllib

import numpy as np
import pandas as pd

from openpredico.config import CONFIGS, load_targets
from openpredico.data import entsoe, wind_weather
from openpredico.live import wind_pair_bundle as bundle
from openpredico.live import wind_pair_state as state
from openpredico.models import wind_pair as pair
from openpredico.models.conformal import TZ, calibrate, day_grid

CONFIG_PATH = CONFIGS / "live/wind_pair.toml"
PIPELINE = "wind_pair_t0_timestral_six_v1"


def config():
    data = CONFIG_PATH.read_bytes()
    cfg = tomllib.loads(data.decode())
    required = {
        "pipeline": PIPELINE,
        "origin": "13:30",
        "context_days": 92,
        "actual_lag_hours": 2,
        "weather_models": list(wind_weather.MODELS),
    }
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError("unsupported wind-pair configuration")
    if cfg["ensemble"] != {
        "onshore": "constrained_learned",
        "offshore": "simple_mean",
        "window_days": 60,
        "minimum_days": 21,
        "regularization": 0.1,
        "max_weight": 0.7,
    }:
        raise ValueError("unsupported ensemble settings")
    if cfg["models"] != {
        "t0_revision": pair.T0_REVISION,
        "timestral_revision": pair.wind_timestral.REVISION,
    }:
        raise ValueError("unsupported model revisions")
    if any(len(cfg[f"{c}_points"]) != 6 for c in ("onshore", "offshore")):
        raise ValueError("six weather points required for each component")
    calibration = cfg["calibration"]
    if (
        calibration["window_days"],
        calibration["minimum_days"],
        calibration["tail_probability"],
        calibration["quantiles"],
    ) != (60, 30, 0.1, [0.1, 0.5, 0.9]):
        raise ValueError("unsupported conformal settings")
    target = load_targets()[bundle.TARGET]
    if (
        target["gate"] != "15:00"
        or target["timezone"] != TZ
        or target["capacity"] != cfg["capacity_mw"]
    ):
        raise ValueError("target gate/timezone/capacity changed; review wind-pair deployment")
    if cfg["storage"] != {
        "dataset": bundle.REPO,
        "prefix": bundle.PREFIX,
        "candidate_before": "14:00",
    }:
        raise ValueError("unsupported storage/deadline settings")
    if cfg["tso_proxy_inputs"] or cfg["wind_speed_knots_ms"] != [2.5, 7.5, 12.5]:
        raise ValueError("unsupported features")
    return cfg, hashlib.sha256(data).hexdigest()


def origin_for(day):
    previous = (pd.Timestamp(day) - pd.DateOffset(days=1)).date()
    return pd.Timestamp(f"{previous} 13:30", tz=TZ).tz_convert("UTC")


def _load(root, name):
    return pd.read_parquet(root / f"{name}.parquet")


def _collect(root, cfg, days):
    """Save daily raw runs and append target-day weather, never weather observations."""
    # ponytail: free endpoint unless a paid key is set, as the incumbent's ECMWF weather does;
    # set OPEN_METEO_BASE_URL/OPEN_METEO_API_KEY (open-predico-wind-weather) for commercial use
    licensed = os.getenv("WIND_PAIR_WEATHER_LICENSED") == "1"
    url = os.getenv("OPEN_METEO_BASE_URL") or wind_weather.BASE_URL
    key = os.getenv("OPEN_METEO_API_KEY")
    points = cfg["onshore_points"] + cfg["offshore_points"]
    all_records = []
    for day in days:
        record_path = root / "runs" / f"{day}.json"
        if record_path.exists():
            records = json.loads(record_path.read_text())
        else:
            records = []
            for model in wind_weather.MODELS:
                records.append(
                    wind_weather.fetch_run(
                        day, model, points, base_url=url, api_key=key, commercial=licensed
                    )
                )
                time.sleep(1)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = record_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(records))
            tmp.replace(record_path)
        decoded = [wind_weather.decode_run(record) for record in records]
        idx = decoded[0].index
        raw = np.stack(
            [f.reindex(idx).to_numpy().reshape(len(idx), 12, 4) for f in decoded], axis=2
        )
        raw[:, :, :, 3] = raw[:, :, 0, 3][:, :, None]
        grid = day_grid(day)
        for component, lo in [("onshore", 0), ("offshore", 6)]:
            frame = pd.DataFrame(raw[:, lo : lo + 6].reshape(len(idx), 96), index=idx)
            today = frame.reindex(grid)
            if not np.isfinite(today).all().all():
                raise ValueError("incomplete new weather day")
            path = root / "weather" / f"{component}.parquet"
            state.write_frame(path, state.append(pd.read_parquet(path), today))
        all_records.extend(records)
    return all_records


def _actuals(root):
    previous = _load(root, "actuals")
    now = pd.Timestamp.now(tz="UTC")
    # ENTSO-E revises recent days; retain actual acquisition time for calibration as-of filtering.
    start = max(
        previous.index.min(),
        min(
            now.normalize() - pd.Timedelta(days=40),
            previous.index.max().normalize() - pd.Timedelta(days=1),
        ),
    )
    end = now.ceil("15min")
    try:
        on, off = [entsoe.fetch(f"A75_{code}", start, end) for code in ("B19", "B18")]
    except Exception as error:
        raise RuntimeError(f"ENTSOE wind acquisition failed ({type(error).__name__})") from None
    received = pd.Timestamp.now(tz="UTC")
    fresh = pd.DataFrame({"onshore": on, "offshore": off})
    fresh["total"] = fresh.onshore + fresh.offshore
    fresh["received_at"] = received.isoformat()
    # Never replace a known value with an unpublished tail, and never fill missing truth.
    fresh = fresh[fresh[["onshore", "offshore"]].notna().all(axis=1)]
    out = state.append(previous, fresh)
    state.write_frame(root / "actuals.parquet", out)
    return out


def _features(root, day, component, cfg):
    origin = origin_for(day)
    cut = origin - pd.Timedelta(hours=2)
    context = pd.date_range(cut - pd.Timedelta(days=92), cut, freq="15min", inclusive="left")
    delivery = day_grid(day)
    bridge = pd.date_range(
        cut, delivery[-1] + pd.Timedelta(minutes=15), freq="15min", inclusive="left"
    )
    records = json.loads((root / "runs" / f"{day}.json").read_text())
    decoded = [wind_weather.decode_run(r) for r in records]
    raw = np.stack(
        [f.reindex(bridge).to_numpy().reshape(len(bridge), 12, 4) for f in decoded], axis=2
    )
    raw[:, :, :, 3] = raw[:, :, 0, 3][:, :, None]
    lo = 0 if component == "onshore" else 6
    raw_future = raw[:, lo : lo + 6]
    historical = pd.read_parquet(root / "weather" / f"{component}.parquet")
    bank, meta = state.bank(root, day, component)
    past = wind_weather.blend(state.raw(historical, context), bank, component=component)
    future = wind_weather.blend(raw_future, bank, component=component)
    frame = pd.DataFrame(
        np.concatenate([past, future]),
        index=context.append(bridge),
        columns=[f"wind_pair.{component}.w{i}" for i in range(24)],
    )
    return frame, meta


def _new_points(root, cfg, days, actuals, device, force=None, persist=True, live_day=None):
    members = _load(root, "members")
    if force is not None:
        members = members[~members.index.isin(day_grid(force))]
    columns = [f"{m}_{c}" for m in pair.MEMBERS for c in ("onshore", "offshore")]
    pending = [
        day for day in days if not np.isfinite(members.reindex(day_grid(day))[columns]).all().all()
    ]
    if not pending:
        return members

    def inputs(day, component):
        return _features(root, day, component, cfg)[0]

    result = pair.predict(
        [(day, origin_for(day)) for day in pending],
        inputs,
        {c: actuals[c] for c in ("onshore", "offshore")},
        device=device,
        cache_dir=os.getenv("HF_HUB_CACHE"),
    )
    for day, frame in result.items():
        frame["issue_time_utc"] = origin_for(day).isoformat()
        frame["delivery_day"] = day
        frame["kind"] = "live" if day == live_day else "hindcast"
        members = state.append(members, frame)
    if persist:
        state.write_frame(root / "members.parquet", members)
    gc.collect()
    return members


def _new_totals(root, cfg, days, actuals, members, force=None, persist=True):
    totals = _load(root, "totals")
    if force is not None:
        totals = totals[~totals.index.isin(day_grid(force))]
    for day in days:
        if day_grid(day).isin(totals.index).all():
            continue
        point, _ = pair.ensemble(members, actuals.onshore, day, origin_for(day))
        point["point_mw"] = point.point_mw.clip(0, cfg["capacity_mw"])
        point["issue_time_utc"] = origin_for(day).isoformat()
        point["delivery_day"] = day
        totals = state.append(totals, point)
    if persist:
        state.write_frame(root / "totals.parquet", totals)
    return totals


def initialize(root, *, device="cpu"):
    """One-off catch-up after seed install, no ready candidate or Predico submission."""
    cfg, sha = config()
    seed = json.loads((root / "seed.json").read_text())
    state.require_production_assets(seed)
    if seed["config_sha256"] != sha:
        raise ValueError("installed seed/config mismatch")
    actuals = _actuals(root)
    members = _load(root, "members")
    end = str((pd.Timestamp.now(tz=TZ).normalize() + pd.DateOffset(days=1)).date())
    start = members.index.max().tz_convert(TZ).normalize() + pd.DateOffset(days=1)
    days = pd.date_range(start, pd.Timestamp(end, tz=TZ), freq="D").strftime("%Y-%m-%d").tolist()
    if days:
        _collect(root, cfg, days)
        members = _new_points(root, cfg, days, actuals, device)
        _new_totals(root, cfg, days, actuals, members)
    return {"status": "initialized", "catchup_days": len(days)}


def produce(root, *, device="cpu", clock=None):
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    cfg, sha = config()
    if not cfg["enabled"]:
        return {"status": "disabled"}
    now = clock()
    day = str((now.tz_convert(TZ).normalize() + pd.DateOffset(days=1)).date())
    origin = origin_for(day)
    seed = json.loads((root / "seed.json").read_text())
    state.require_production_assets(seed)
    if seed["config_sha256"] != sha:
        raise ValueError("seed/config mismatch")
    pending = root / "pending" / f"{day}.zip"
    if (root / "ready" / day).exists():
        return {"status": "already_ready", "day": day}
    if pending.exists():
        # Upload retry preserves the exact pre-issue snapshot, even after the source cutoff.
        deadline = pd.Timestamp(f"{now.tz_convert(TZ):%Y-%m-%d} 14:00", tz=TZ).tz_convert("UTC")
        if now >= deadline:
            raise ValueError("candidate upload retry deadline passed")
        bundle.decode(
            pending.read_bytes(), now=max(now, origin), config_hash=sha, capacity=cfg["capacity_mw"]
        )
        # No partial recomputation.
        bundle.upload(pending.read_bytes(), day)
        (root / "ready").mkdir(exist_ok=True)
        (root / "ready" / day).touch()
        return {"status": "ready", "day": day, "upload_retry": True}
    if now >= origin:
        raise ValueError("source collection started after frozen issue")
    members = _load(root, "members")
    first_missing = members.index.max().tz_convert(TZ).normalize() + pd.DateOffset(days=1)
    gap = (pd.Timestamp(day) - first_missing.tz_localize(None)).days
    if gap > 7:
        raise ValueError("state needs one-off catchup before daily production")
    collect_days = (
        pd.date_range(
            min(first_missing, pd.Timestamp(day, tz=TZ)), pd.Timestamp(day, tz=TZ), freq="D"
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )
    records = _collect(root, cfg, collect_days)
    actuals = _actuals(root)
    received = max(
        [pd.Timestamp(r["retrieved_at_utc"]) for r in records]
        + [pd.to_datetime(actuals.received_at, utc=True).max()]
    )
    if received >= origin:
        raise ValueError("inputs arrived after frozen issue")
    members = _new_points(root, cfg, collect_days, actuals, device, force=day, live_day=day)
    totals = _new_totals(root, cfg, collect_days, actuals, members, force=day)
    current = totals.reindex(day_grid(day))
    history = totals[["point_mw", "issue_time_utc"]].copy()
    history["point_mw"] = history.point_mw.clip(0, cfg["capacity_mw"])
    history["actual_mw"] = actuals.total.reindex(history.index)
    history["actual_available_at"] = actuals.received_at.reindex(history.index)
    history["pipeline"] = PIPELINE
    history["truth_source"] = cfg["truth_source"]
    fc, audit = calibrate(
        current.point_mw,
        history,
        origin=origin,
        pipeline=PIPELINE,
        truth_source=cfg["truth_source"],
    )
    fc = fc.clip(0, cfg["capacity_mw"])
    fc["model"] = PIPELINE
    inputs, banks = [], {}
    for component in ("onshore", "offshore"):
        wx, banks[component] = _features(root, day, component, cfg)
        y = actuals[component].reindex(wx.index).where(wx.index < origin - pd.Timedelta(hours=2))
        inputs.append(wx.assign(**{f"wind_pair.{component}.actual": y}))
    inputs = pd.concat(inputs, axis=1)
    created = clock()
    deadline = pd.Timestamp(f"{now.tz_convert(TZ):%Y-%m-%d} 14:00", tz=TZ).tz_convert("UTC")
    if created >= deadline:
        raise ValueError("candidate missed pre-submission deadline")
    _, weight_audit = pair.ensemble(members, actuals.onshore, day, origin)
    audit.update(
        target=bundle.TARGET,
        day=day,
        issue_time_utc=origin.isoformat(),
        created_at_utc=created.isoformat(),
        latest_input_received_at_utc=received.isoformat(),
        config_sha256=sha,
        weather_banks=banks,
        ensemble=weight_audit,
        historical_seed_vintages_verified=False,
        model_revisions=cfg["models"],
        newly_backfilled_days=max(0, len(collect_days) - 1),
    )
    fit_members = members.copy()
    fit_members["actual_onshore"] = actuals.onshore.reindex(members.index)
    fit_members["actual_received_at"] = actuals.received_at.reindex(members.index)
    data = bundle.encode(fc, inputs, history[history.index < origin], audit, records, fit_members)
    pending.parent.mkdir(parents=True, exist_ok=True)
    tmp = pending.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(pending)
    bundle.upload(data, day)
    (root / "ready").mkdir(exist_ok=True)
    (root / "ready" / day).touch()
    return {"status": "ready", "day": day, "intervals": len(fc)}


def smoke(root, *, device="cpu"):
    """Real-model and calibration check at invocation time, never a ready production run."""
    started = time.monotonic()
    result = initialize(root, device=device)
    cfg, _ = config()
    day = str((pd.Timestamp.now(tz=TZ).normalize() + pd.DateOffset(days=1)).date())
    actuals = _load(root, "actuals")
    # Re-run current members through the real engines to validate hardware even if cache is full.
    members = _new_points(root, cfg, [day], actuals, device, force=day, persist=False)
    totals = _new_totals(root, cfg, [day], actuals, members, force=day, persist=False)
    current = totals.reindex(day_grid(day))
    h = totals[["point_mw", "issue_time_utc"]].copy()
    h["actual_mw"] = actuals.total.reindex(h.index)
    h["actual_available_at"] = actuals.received_at.reindex(h.index)
    h["pipeline"], h["truth_source"] = PIPELINE, cfg["truth_source"]
    asof = max(origin_for(day), pd.Timestamp.now(tz="UTC"))
    fc, audit = calibrate(
        current.point_mw, h, origin=asof, pipeline=PIPELINE, truth_source=cfg["truth_source"]
    )
    return dict(
        result,
        validation_only=True,
        model_intervals=len(fc),
        calibration_days=audit["calibration_days"],
        seconds=time.monotonic() - started,
    )
