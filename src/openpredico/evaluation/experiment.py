"""Backtest a TSFM with covariates on the published features (`open-predico/benchmark`).

Every column of the target's `groups` (targets.toml) is used: `availability` in groups.toml
splits known-future from past columns, the target is the first column and the other columns of
its group (e.g. the other Spain target) are past covariates. Predictions (with the
truth as `obs`) go to `<out>/<config>/<target>.parquet`, ready for `evaluation.results`.
`drop` leaves out comma-separated columns (ablation); `drop=all` keeps only the target history
(univariate: no past, known-future or calendar covariates). `targets` limits the run to
comma-separated targets. Repo `slot_climatology` runs that baseline over `context days`.

    python -m openpredico.evaluation.experiment <config> <repo> <context days> <out dir> \
        [<device>] [<drop>] [<origin HH:MM, local; default from targets.toml>] [<targets>]
"""

import importlib
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from openpredico.config import load_targets
from openpredico.data.groups import load_groups
from openpredico.data.hf import ORG, download
from openpredico.evaluation.backtest import backtest, clip
from openpredico.models.baselines import slot_climatology

TARGET_COLUMNS = {name: t["column"] for name, t in load_targets().items()}
CATEGORICAL = ["qh_of_day", "weekday"]
# Hugging Face org of the weights -> adapter module in models/tsfm/.
ADAPTERS = {
    "mistralai": "timestral",
    "amazon": "chronos2",
    "google": "timesfm",
    "theforecastingcompany": "t0",
}
MAX_NAN = 0.9  # columns missing more than this share of the history are dropped


def load_benchmark(groups: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """`groups` (default all) from the latest version folder (`v1/`, `v2/`, ...) holding each.

    Versions only add groups (v2: the 50Hertz ones); unpublished groups (`mfrr_bids`) are absent.
    """
    groups, latest = groups or load_groups().keys(), {}
    files = HfApi().list_repo_files(f"{ORG}/benchmark", repo_type="dataset")
    for f in files:
        version, _, name = f.partition("/")
        group = name.removesuffix(".parquet")
        if version[:1] == "v" and version[1:].isdigit() and group in groups:
            latest[group] = max(latest.get(group, version), version, key=lambda v: int(v[1:]))
    return {g: download("benchmark", f"{v}/{g}.parquet") for g, v in latest.items()}


def features(
    frames: dict[str, pd.DataFrame], target: str, drop: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    """History frame (target first, then past columns, then future ones) and the future columns.

    `drop`: comma-separated columns to leave out, or "all" for the target alone.
    """
    groups = load_groups()
    t = load_targets()[target]
    y = t["column"]
    df = pd.concat([frames[g] for g in t["groups"] if g in frames], axis=1)
    if drop == "all":
        return df[[y]].astype(float), []
    df = df.drop(columns=[c for c in drop.split(",") if c])
    nan = df.isna().mean()
    drop = [c for c in nan.index[nan > MAX_NAN] if c not in TARGET_COLUMNS.values()]
    if drop:
        print(f"dropped (>{MAX_NAN:.0%} missing): {drop}")
    df = df.drop(columns=drop)
    # ESIOS omits zero quarter-hours in the sparse mFRR series: missing = no activation.
    # ponytail: hard-coded until groups.toml carries `fill = 0` for these columns.
    mfrr = [c for c in groups["mfrr"] if c in df]
    df[mfrr] = df[mfrr].fillna(0)
    future = [c for g in groups.values() for c, d in g.items() if d["availability"] == "future"]
    future = [c for c in future if c in df]
    past = [c for c in df if c not in future and c != y]
    return df[[y, *past, *future]].astype(float), future


def run(
    config: str,
    repo: str,
    context_days: int,
    out: Path,
    device: str = "cpu",
    drop: str = "",
    origin: str = "",
    targets: str = "",
) -> None:
    frames = load_benchmark()
    if repo != "slot_climatology":
        adapter = importlib.import_module(f"openpredico.models.tsfm.{ADAPTERS[repo.split('/')[0]]}")
        model = adapter.load(repo, device)
    for target, t in load_targets().items():
        if targets and target not in targets.split(","):
            continue
        if missing := set(t["groups"]) - frames.keys():
            print(f"{target}: skipped, groups not on HF: {sorted(missing)}")
            continue
        history, future = features(frames, target, drop)
        days = pd.date_range(*t["test_period"], tz=t["timezone"])
        if repo == "slot_climatology":

            def predict(past, future, days, tz=t["timezone"]):
                return slot_climatology(past.iloc[:, 0], days, context_days, tz)
        else:
            predict = adapter.make_predict(model, context_days, CATEGORICAL)
        start = time.time()
        preds = clip(backtest(predict, history, days, origin or t["origin"], future), t)
        print(
            f"{target}: {len(future)} future, {history.shape[1] - 1 - len(future)} past "
            f"columns, {len(days)} days in {time.time() - start:.0f} s"
        )
        preds["obs"] = history[t["column"]]  # raw truth, NaN kept
        preds.index.name = "time"
        (out / config).mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out / config / f"{target}.parquet")


if __name__ == "__main__":
    run(*sys.argv[1:3], int(sys.argv[3]), Path(sys.argv[4]), *sys.argv[5:])
