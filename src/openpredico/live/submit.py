"""Daily live run: t0-gamma D+1 (the target's groups as covariates, 90-day context) plus
slot-climatology D+2..D+10 for every target -> HF live-private -> Predico.

Usage: python -m openpredico.live.submit [--dry-run] [--target a,b] [--check]
       --dry-run skips HF and Predico; --target limits the run to these targets.
Needs the t0 extra: `uv sync --extra t0`.
"""

import sys
import time
from collections.abc import Callable

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError

from openpredico.config import live_targets
from openpredico.data import hf
from openpredico.data.groups import build, load_groups
from openpredico.evaluation.backtest import PUBLICATION_LAG, clip
from openpredico.evaluation.experiment import CATEGORICAL, features, load_benchmark
from openpredico.live.predico import Predico
from openpredico.live.wind_pair_consumer import needs_retry as pair_needs_retry
from openpredico.live.wind_pair_consumer import ready as ready_wind_pair
from openpredico.models.baselines import slot_climatology
from openpredico.models.tsfm import t0

MODEL = "theforecastingcompany/t0-gamma"  # best Apache-2.0 model in the backtest
CONFIG, FALLBACK = "t0gamma_cov_ctx90", "slot_climatology_28d"  # backtest config names
CONTEXT_DAYS = 90
HORIZON_DAYS = 10
ARCHIVE = "archive/{}.parquet"  # per non-calendar group, in the (public) benchmark repo
# Days refetched before an archive's frontier each run, per source; refetched values win.
# ESIOS overwrites some forecasts after publication (PLAN: Benchmark), later revisions unchecked;
# ENTSO-E revises A75 over 35 days (PLAN: 50Hertz); Elia's daily ods209 replaces the
# near-real-time ods210 a day later; an Open-Meteo run never changes. The window costs one
# request per series whatever its length (ESIOS chunks a month, ENTSO-E a year, Elia one export).
REVISE = {"openmeteo": 0, "esios": 7, "elia": 7, "entsoe": 35}
STEP = pd.Timedelta("15min")


def build_frames(
    now: pd.Timestamp,
    groups: list[str],
    tz: str,
    first: pd.Timestamp | None = None,
    save: bool = False,
) -> dict[str, pd.DataFrame]:
    """`groups` from CONTEXT_DAYS before `first` (default now) through the end of now's D+1.

    Every group but calendar ones (computed) comes from its archive (`archived`); `save` writes
    the extended archives.
    """
    local = now.tz_convert(tz)
    since = (first if first is not None else now).tz_convert(tz)
    start = (since - pd.Timedelta(days=CONTEXT_DAYS + 1)).normalize().tz_convert("UTC")
    end = (local.normalize() + pd.DateOffset(days=2)).tz_convert("UTC")
    out, groups_toml = {}, load_groups()
    for g in groups:
        t0_ = time.time()
        if all(c["source"].startswith("calendar:") for c in groups_toml[g].values()):
            out[g] = build(g, *(t.tz_localize(None).isoformat() for t in (start, end)), tz, True)
        else:
            out[g] = archived(g, start, end, tz, save)
        print(f"built {g} in {time.time() - t0_:.0f} s")  # timing only, no values
    return out


def archived(g: str, start: pd.Timestamp, end: pd.Timestamp, tz: str, save: bool) -> pd.DataFrame:
    """Group `g` on [start, end) from its archive on HF (`benchmark/archive/<g>.parquet`),
    extended each run: only the days from its `frontier` on, minus its sources' REVISE window,
    are fetched, and fetched values win. Old days are kept (the growing backtest set). Seeded
    from the backtest backfill (`load_benchmark`) if there is no archive yet. `save` writes the
    archive back only when its frontier reached a later day: about one commit a day per group,
    even for hourly runs."""
    columns = load_groups()[g]
    try:
        old = hf.download("benchmark", ARCHIVE.format(g))
    except RemoteEntryNotFoundError:  # no archive yet: seed from the backfill, else fetch it all
        # Only a missing file: any other error (HF down) raises, so a reseed never overwrites
        # the archive. Not LocalEntryNotFoundError either: that one means HF was unreachable.
        old = load_benchmark({g}).get(g, pd.DataFrame())
    old = old[[c for c in columns if c in old]]  # columns dropped from groups.toml stay out
    was = frontier(old, columns, tz)
    revise = pd.Timedelta(days=max(REVISE[c["source"].split(":")[0]] for c in columns.values()))
    since = start if was is None or old.index.min() > start else max(start, was - revise)
    new = pd.DataFrame()
    if since < end:
        new = build(g, *(t.tz_localize(None).isoformat() for t in (since, end)), tz, lenient=True)
    merged = new if old.empty else old if new.empty else new.combine_first(old).sort_index()
    merged = merged[[c for c in columns if c in merged]]
    reached = frontier(merged, columns, tz)
    if save and reached is not None and (was is None or reached > was):
        msg = f"Archive {g}: through {reached:%Y-%m-%d}"
        hf.upload(merged, "benchmark", ARCHIVE.format(g), msg)
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="time")
    return merged.reindex(grid)


def frontier(df: pd.DataFrame, columns: dict, tz: str) -> pd.Timestamp | None:
    """Start (UTC) of the first local day `df` doesn't fully hold; None if it holds nothing.

    The earlier of its last row with any past value and its last row with any future value, so
    sparse columns (mFRR direct activations) don't hold it back. Columns with a `fill` are left
    out: the fill runs past the published data.
    """
    ends = []
    for availability in ("past", "future"):
        cols = [c for c, d in columns.items() if d["availability"] == availability]
        if cols := [c for c in cols if c in df and "fill" not in columns[c]]:
            ends.append(df[cols].dropna(how="all").index.max())
    if not ends or any(pd.isna(e) for e in ends):
        return None
    return (min(ends) + STEP).tz_convert(tz).normalize().tz_convert("UTC")


def forecast(
    target: str, frames: dict[str, pd.DataFrame], predict: Callable, now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecast D+1..D+10 within 240 h, exact model inputs). Same cut as the backtest."""
    t = live_targets()[target]
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
    if late:  # left out rather than filled: TSFMs take any covariate set
        print(f"{target}: left out, known-future columns with gaps on D+1: {late}")  # names
        future_cols = [c for c in future_cols if c not in late]
        future, past = future[future_cols], past.drop(columns=late)
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
    targets = {n: t for n, t in live_targets().items() if not only or n in only}
    if unknown := set(only or ()) - targets.keys():
        sys.exit(f"unknown targets: {sorted(unknown)}")
    client = None if dry_run else Predico()  # log in first: fail fast on bad credentials
    predict = None  # load lazily: a ready wind pair does not need incumbent D+1 inference
    built = {}  # (groups, tz) -> frames: the Spain targets share one build
    for name, target in targets.items():
        candidate = ready_wind_pair(name, now)
        if candidate is not None:
            fc, inputs = candidate
        else:
            if predict is None:
                predict = t0.make_predict(t0.load(MODEL), CONTEXT_DAYS, CATEGORICAL)
            key = (tuple(target["groups"]), target["timezone"])
            if key not in built:
                built[key] = build_frames(now, *key, save=not dry_run)
            fc, inputs = forecast(name, built[key], predict, now)
        validate(fc, now)
        print(f"{name}: {len(fc)} steps {fc.index.min()} .. {fc.index.max()}")  # no values
        if dry_run:
            continue
        # Predico first: an HF outage (429 on 2026-11-08) must not block the submission.
        client.submit_continuous(client.resource_id(target["predico_resource"]), fc)
        print(f"{name}: submitted")
        env = "playground" if "playground" in client.api else "production"
        run = f"{name}/{now:%Y-%m-%dT%H%MZ}"
        hf.upload(fc.assign(env=env), "live-private", f"forecasts/{run}.parquet", f"Forecast {run}")
        hf.upload(inputs, "live-private", f"inputs/{run}.parquet", f"Inputs {run}")
    if client:
        print("coverage:", client.coverage())


def hindcast(first: str, last: str, only: list[str]) -> None:
    """Rerun the live model for past target days [first, last] we never submitted.

    Each run is cut exactly as a live run 45 min before that day's gate (`forecast`), from one
    data build for the whole span. Saved to `live-private` with env "hindcast", so the dashboard
    shows it as a model forecast without a submission. Never sent to Predico.
    ponytail: ESIOS values revised after the original run time leak in, as in the backtest.
    """
    predict = t0.make_predict(t0.load(MODEL), CONTEXT_DAYS, CATEGORICAL)
    built = {}  # (groups, tz) -> frames: the Spain targets share one build
    for name in only:
        t = live_targets()[name]
        tz = t["timezone"]
        days = pd.date_range(first, last, tz=tz)
        nows = [
            (pd.Timestamp(f"{d - pd.Timedelta(days=1):%Y-%m-%d} {t['gate']}", tz=tz))
            - pd.Timedelta("45min")
            for d in days
        ]
        key = (tuple(t["groups"]), tz)
        if key not in built:
            built[key] = build_frames(nows[-1].tz_convert("UTC"), *key, first=nows[0])
        for now in nows:
            now = now.tz_convert("UTC")
            fc, inputs = forecast(name, built[key], predict, now)
            validate(fc, now)
            run = f"{name}/{now:%Y-%m-%dT%H%MZ}"
            hf.upload(fc.assign(env="hindcast"), "live-private", f"forecasts/{run}.parquet", run)
            hf.upload(inputs, "live-private", f"inputs/{run}.parquet", f"Inputs {run}")
            print(f"{run}: hindcast saved")  # no values


def in_session(now: pd.Timestamp) -> list[str]:
    """Targets whose D+1 session is open now: the hour before their gate, local time."""
    out = []
    for name, t in live_targets().items():
        local = now.tz_convert(t["timezone"])
        gate = pd.Timestamp(f"{local:%Y-%m-%d} {t['gate']}").tz_localize(t["timezone"])
        if gate - pd.Timedelta("1h") <= local < gate:
            out.append(name)
    return out


def not_ready(client: Predico, now: pd.Timestamp) -> list[str]:
    """Our targets with an open D+1 challenge that our forecasts don't fully cover yet.

    Challenges starting before the target's next local midnight are intraday (Belgium, hourly):
    our forecast starts at D+1, so they are never covered and not counted.
    """
    cov = {c["challenge_id"]: c for c in client.coverage()["data"]["coverage"]}
    out = []
    for t in live_targets().values():
        tomorrow = now.tz_convert(t["timezone"]).normalize() + pd.DateOffset(days=1)
        for c in client.open_challenges(t["predico_resource"]):
            row = cov.get(c["id"], {})
            if row.get("start") and pd.Timestamp(row["start"]) < tomorrow:
                continue  # ponytail: intraday not forecast; add a from-now run to cover it
            if not row.get("ready"):
                out.append(c["resource_name"])
    return out


def not_run(now: pd.Timestamp, files: list[str]) -> list[str]:
    """Targets in session without a live run archived today (`files`: live-private paths).

    Predico's coverage can't tell: yesterday's climatology D+2 already covers today's challenge.
    """
    return [
        n
        for n in in_session(now)
        if not any(f.startswith(f"forecasts/{n}/{now:%Y-%m-%d}T") for f in files)
    ]


def missing(client: Predico, now: pd.Timestamp) -> list[str]:
    """Targets in session to (re)submit: not covered on Predico, or no t0 run of ours today.

    Only targets in session: a challenge open hours before its gate is never forecast early.
    """
    by_resource = {t["predico_resource"]: n for n, t in live_targets().items()}
    files = HfApi().list_repo_files(f"{hf.ORG}/live-private", repo_type="dataset")
    uncovered = {by_resource[r] for r in not_ready(client, now)} & set(in_session(now))
    pending = uncovered | set(not_run(now, files))
    if pair_needs_retry(now, files):
        pending.add("germany_wind_50hertz_ca")
    return sorted(pending)


def check_and_retry() -> None:
    """One scheduler tick (live/modal_app.py, every 15 min): submit the targets that `missing`
    lists, so a session's first tick submits and later ones retry what failed."""
    now = pd.Timestamp.now(tz="UTC")
    if not in_session(now):
        print("no session of ours open")  # the cheap path: no Predico login, no HF call
        return
    todo = missing(Predico(), now)
    print(f"missing: {todo}")
    if not todo:
        return  # every target in session has its run and is covered
    main(dry_run=False, only=todo)
    if todo := missing(Predico(), pd.Timestamp.now(tz="UTC")):
        sys.exit(f"still missing after resubmit: {todo}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_and_retry()
    elif "--hindcast" in sys.argv:  # --hindcast FIRST LAST --target a,b (local target days)
        i = sys.argv.index("--hindcast")
        hindcast(*sys.argv[i + 1 : i + 3], sys.argv[sys.argv.index("--target") + 1].split(","))
    else:
        only = (
            sys.argv[sys.argv.index("--target") + 1].split(",") if "--target" in sys.argv else None
        )
        main(dry_run="--dry-run" in sys.argv, only=only)
