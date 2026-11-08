"""Daily live run: t0-gamma D+1 (the target's groups as covariates, 90-day context) plus
slot-climatology D+2..D+10 for every target -> HF live-private -> Predico.

Usage: python -m openpredico.live.submit [--dry-run] [--target a,b] [--check]
       --dry-run skips HF and Predico; --target limits the run to these targets.
Needs the t0 extra: `uv sync --extra t0`.
"""

import sys
from collections.abc import Callable

import pandas as pd

from openpredico.config import load_targets
from openpredico.data import hf
from openpredico.data.groups import build
from openpredico.evaluation.backtest import PUBLICATION_LAG, clip
from openpredico.evaluation.experiment import CATEGORICAL, features
from openpredico.live.predico import Predico
from openpredico.models.baselines import slot_climatology
from openpredico.models.tsfm import t0

MODEL = "theforecastingcompany/t0-gamma"  # best Apache-2.0 model in the backtest
CONFIG, FALLBACK = "t0gamma_cov_ctx90", "slot_climatology_28d"  # backtest config names
CONTEXT_DAYS = 90
HORIZON_DAYS = 10


def build_frames(now: pd.Timestamp, groups: list[str], tz: str) -> dict[str, pd.DataFrame]:
    """`groups` from CONTEXT_DAYS before now through the end of D+1 (local time `tz`)."""
    local = now.tz_convert(tz)
    start = (local - pd.Timedelta(days=CONTEXT_DAYS + 1)).normalize().tz_convert("UTC")
    end = (local.normalize() + pd.DateOffset(days=2)).tz_convert("UTC")
    start, end = start.tz_localize(None).isoformat(), end.tz_localize(None).isoformat()
    # ponytail: ~30 ESIOS ids x 4 monthly chunks at 1 req/s, refetched daily; cache on HF if slow
    # ponytail: 50Hertz: one request per local day, 2 s apart (~3 min for 92 days)
    return {g: build(g, start, end, tz) for g in groups}


def forecast(
    target: str, frames: dict[str, pd.DataFrame], predict: Callable, now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecast D+1..D+10 within 240 h, exact model inputs). Same cut as the backtest."""
    t = load_targets()[target]
    tz = t["timezone"]
    # ponytail: the >90%-missing column drop is decided on the live window, not the backtest's
    history, future_cols = features(frames, target)
    days = pd.date_range(
        now.tz_convert(tz).normalize() + pd.Timedelta(days=1), periods=HORIZON_DAYS, tz=tz
    )
    stop = (days[0] + pd.DateOffset(days=1)).tz_convert("UTC")
    past = history[history.index < now - PUBLICATION_LAG]
    future = history.loc[history.index < stop, future_cols]
    late = [c for c in future_cols if future.loc[future.index >= days[0], c].isna().any()]
    if late:
        print(f"{target}: known-future columns with gaps on D+1: {late}")  # names only
    past = past.drop(columns=future_cols)
    d1 = predict(past, future, days[:1])
    rest = slot_climatology(past.iloc[:, 0], days[1:], tz=tz)  # D+2..D+10: unpaid fallback
    fc = clip(pd.concat([d1.assign(model=CONFIG), rest.assign(model=FALLBACK)]), t)
    # Predico accepts at most 240 h ahead, so the tail of D+10 is cut
    return fc[fc.index <= now + pd.Timedelta(hours=240)], pd.concat([past, future], axis=1)


def validate(fc: pd.DataFrame, now: pd.Timestamp) -> None:
    assert fc.notna().all().all(), "NaN in forecast"
    assert (fc.q10 <= fc.q50).all() and (fc.q50 <= fc.q90).all(), "quantiles cross"
    assert fc.index.min() > now and fc.index.max() - now <= pd.Timedelta(hours=240), (
        "outside horizon"
    )
    assert (fc.index.to_series().diff().dropna() == pd.Timedelta("15min")).all(), "grid has gaps"


def main(dry_run: bool, only: list[str] | None = None) -> None:
    """Submit every target, or only the targets named in `only`."""
    now = pd.Timestamp.now(tz="UTC")
    targets = {n: t for n, t in load_targets().items() if not only or n in only}
    if unknown := set(only or ()) - targets.keys():
        sys.exit(f"unknown targets: {sorted(unknown)}")
    client = None if dry_run else Predico()  # log in first: fail fast on bad credentials
    predict = t0.make_predict(t0.load(MODEL), CONTEXT_DAYS, CATEGORICAL)
    built = {}  # (groups, tz) -> frames: the Spain targets share one build
    for name, target in targets.items():
        key = (tuple(target["groups"]), target["timezone"])
        if key not in built:
            built[key] = build_frames(now, *key)
        fc, inputs = forecast(name, built[key], predict, now)
        validate(fc, now)
        print(f"{name}: {len(fc)} steps {fc.index.min()} .. {fc.index.max()}")  # no values
        if dry_run:
            continue
        env = "playground" if "playground" in client.api else "production"
        run = f"{name}/{now:%Y-%m-%dT%H%MZ}"
        hf.upload(fc.assign(env=env), "live-private", f"forecasts/{run}.parquet", f"Forecast {run}")
        hf.upload(inputs, "live-private", f"inputs/{run}.parquet", f"Inputs {run}")
        client.submit_continuous(client.resource_id(target["predico_resource"]), fc)
        print(f"{name}: submitted")
    if client:
        print("coverage:", client.coverage())


def in_session(now: pd.Timestamp) -> list[str]:
    """Targets whose D+1 session is open now: the hour before their gate, local time."""
    out = []
    for name, t in load_targets().items():
        local = now.tz_convert(t["timezone"])
        gate = pd.Timestamp(f"{local:%Y-%m-%d} {t['gate']}").tz_localize(t["timezone"])
        if gate - pd.Timedelta("1h") <= local < gate:
            out.append(name)
    return out


def not_ready(client: Predico) -> list[str]:
    """Our targets with an open challenge that our forecasts don't fully cover yet."""
    ready = {c["challenge_id"]: c["ready"] for c in client.coverage()["data"]["coverage"]}
    names = [t["predico_resource"] for t in load_targets().values()]
    return [
        c["resource_name"]
        for n in names
        for c in client.open_challenges(n)
        if not ready.get(c["id"])
    ]


def check_and_retry() -> None:
    """Run in each open session before the gate: resubmit if an open challenge isn't covered."""
    missing = not_ready(Predico())
    print(f"not ready: {missing}")
    if not missing:
        return  # covered, or no session of ours open right now
    main(
        dry_run=False,
        only=[n for n, t in load_targets().items() if t["predico_resource"] in missing],
    )
    if missing := not_ready(Predico()):
        sys.exit(f"still not ready after resubmit: {missing}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_and_retry()
    else:
        only = (
            sys.argv[sys.argv.index("--target") + 1].split(",") if "--target" in sys.argv else None
        )
        main(dry_run="--dry-run" in sys.argv, only=only)
