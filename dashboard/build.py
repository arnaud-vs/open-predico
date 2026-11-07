"""Build the dashboard's JSON from the public HF datasets (`benchmark`, `live`), then optionally
push the static Space. The page only draws: every score here uses evaluation/metrics.py.

    python dashboard/build.py                 # writes dashboard/data/
    python dashboard/build.py --upload        # and pushes index.html, app.js, data/ to the Space
    python dashboard/build.py --demo-live     # local preview only: backtest days in the live format
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from openpredico.config import load_targets
from openpredico.data.groups import load_groups
from openpredico.data.hf import ORG, download
from openpredico.evaluation.experiment import NOT_IN_V1, TARGET_COLUMNS
from openpredico.evaluation.metrics import daily_scores

HERE = Path(__file__).parent
PREFIX = "predictions/backtest/"
BASELINE = "slot_climatology_28d"
LIVE_CONFIG = "t0gamma_cov_ctx90"
SHORT = {"spain_mfrr_energy": "energy", "spain_mfrr_price": "price"}
UNITS = {"energy": "MWh", "price": "EUR/MWh"}
TYPES = {  # config name prefix -> model type; anything else is ML
    "slot_climatology": "baseline",
    "chronos": "foundation model",
    "timesfm": "foundation model",
    "timestral": "foundation model",
    "t0": "foundation model",
}
STEP = pd.Timedelta("15min")
LIVE_COLUMNS = ("q10", "q50", "q90", "s10", "s50", "s90", "obs", "b10", "b50", "b90")


def arr(values) -> list:
    """Floats rounded to 0.1 with NaN as null: the JSON stays small."""
    a = np.round(np.asarray(values, dtype=float), 1)
    return [None if np.isnan(v) else v for v in a.tolist()]


def secs(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def write(out: Path, name: str, obj) -> None:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def backtest(out: Path, tz: str) -> dict:
    files = [
        f
        for f in HfApi().list_repo_files(f"{ORG}/benchmark", repo_type="dataset")
        if f.startswith(PREFIX) and f.endswith(".parquet")
    ]
    days, profiles, grid = {}, {}, None
    for f in sorted(files):
        config, target = f.removeprefix(PREFIX).removesuffix(".parquet").split("/")
        t = SHORT[target]
        df = download("benchmark", f)
        grid = df.index if grid is None else grid
        df = df.reindex(grid)
        write(out, f"steps/{t}/{config}.json", {c: arr(df[c]) for c in ("q10", "q50", "q90")})
        d = daily_scores(df[["q10", "q50", "q90"]], df.obs, tz=tz)
        days.setdefault(t, {})[config] = d[["rmse", "winkler", "coverage"]]
        local = df.index.tz_convert(tz)
        ae = (df.obs - df.q50).abs()
        profiles.setdefault(t, {})[config] = {
            "hour": arr(ae.groupby(local.hour).mean().reindex(range(24))),
            "weekday": arr(ae.groupby(local.dayofweek).mean().reindex(range(7))),
        }
        print(f"backtest {config}/{t}", flush=True)
    out_days = {}
    for t, by_config in days.items():
        index = sorted(set().union(*(d.index for d in by_config.values())))
        out_days[t] = {
            "days": [d.strftime("%Y-%m-%d") for d in index],
            "configs": {
                c: {m: arr(d[m].reindex(index)) for m in ("rmse", "winkler", "coverage")}
                for c, d in by_config.items()
            },
        }
    write(out, "days.json", out_days)
    write(out, "profiles.json", profiles)
    configs = sorted({c for d in days.values() for c in d})
    return {
        "start": secs(grid[0]),
        "n": len(grid),
        "configs": [
            {"name": c, "type": next((v for k, v in TYPES.items() if c.startswith(k)), "ML")}
            for c in configs
        ],
    }


def target_series(out: Path) -> None:
    df = download("benchmark", "v1/target.parquet").asfreq(STEP)
    write(
        out,
        "target.json",
        {"start": secs(df.index[0]), "energy": arr(df.mfrr_energy), "price": arr(df.mfrr_price)},
    )


def catalog(out: Path) -> None:
    cat = download("benchmark", "catalog/inputs.parquet")
    write(out, "catalog.json", json.loads(cat.to_json(orient="records")))


def live_frames(demo: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Published live forecasts per short target name, and the runs table (None if absent)."""
    if not demo:
        try:
            runs = download("live", "runs.parquet")
        except Exception:  # dataset not published yet
            return {}, None
        frames = {
            SHORT[t]: download("live", f"forecasts/{t}.parquet") for t in runs.target.unique()
        }
        return frames, runs
    # Demo: the live config's last 21 backtest days, two of them without a Predico submission.
    frames, rows = {}, []
    for target, t in SHORT.items():
        fc = download("benchmark", f"{PREFIX}{LIVE_CONFIG}/{target}.parquet").iloc[-21 * 96 :]
        base = download("benchmark", f"{PREFIX}{BASELINE}/{target}.parquet").reindex(fc.index)
        fc[["b10", "b50", "b90"]] = base[["q10", "q50", "q90"]].to_numpy()
        day = fc.index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")
        fc = fc[day != sorted(set(day))[6]]  # one missed day
        day = fc.index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")
        registered = ~np.isin(day, sorted(set(day))[-3:-1])
        for q in ("10", "50", "90"):
            fc[f"s{q}"] = fc[f"q{q}"].where(registered)
        fc["model"], fc["env"] = LIVE_CONFIG, "demo"
        fc.loc[day == sorted(set(day))[2], ["q10", "q50", "q90"]] = np.nan  # registered, no run
        frames[t] = fc
        cols = [c for g, cs in load_groups().items() if g not in NOT_IN_V1 for c in cs]
        rows.append(
            {
                "target": target,
                "run": fc.index[-1],
                "model": LIVE_CONFIG,
                "features": ",".join(c for c in cols if c != TARGET_COLUMNS[target]),
            }
        )
    return frames, pd.DataFrame(rows)


def live(out: Path, demo: bool) -> None:
    frames, runs = live_frames(demo)
    if runs is None:
        write(out, "live.json", {"available": False})
        return
    data = {"available": True, "demo": demo, "targets": {}}
    for t, df in frames.items():
        grid = pd.date_range(df.index.min(), df.index.max(), freq=STEP)
        g = df.reindex(grid).reindex(columns=[*LIVE_COLUMNS, "env"])
        target_runs = runs[runs.target.map(SHORT) == t].sort_values("run")
        last = target_runs.iloc[-1]
        data["targets"][t] = {
            "start": secs(grid[0]),
            # q*: continuous forecast at the gate; s*: what Predico registered; b*: climatology
            **{c: arr(g[c]) for c in LIVE_COLUMNS},
            "model": last.model,
            "env": str(df.env.dropna().iloc[-1]) if "env" in df else "playground",
            "runs": len(target_runs),
            "last_run": str(last.run),
            "features": [f for f in str(last.features).split(",") if f],
        }
    write(out, "live.json", data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "data")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--demo-live", action="store_true")
    a = p.parse_args()
    if a.upload and a.demo_live:
        raise SystemExit("--demo-live is for local previews; never upload it")
    tz = load_targets()["spain_mfrr_energy"]["timezone"]
    meta = backtest(a.out, tz)
    target_series(a.out)
    catalog(a.out)
    live(a.out, a.demo_live)
    meta |= {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
        "baseline": BASELINE,
        "live_config": LIVE_CONFIG,
        "units": UNITS,
    }
    write(a.out, "meta.json", meta)
    if a.upload:
        HfApi().upload_folder(
            repo_id=f"{ORG}/dashboard",
            repo_type="space",
            folder_path=HERE,
            allow_patterns=["index.html", "app.js", "README.md", "data/**"],
            delete_patterns=["app.py", "requirements.txt"],  # the old stlite app
            commit_message=f"Dashboard data {meta['generated']}",
        )


if __name__ == "__main__":
    main()
