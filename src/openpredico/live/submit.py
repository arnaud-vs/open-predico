"""Daily live run: baseline forecast D+1..D+10 for every target -> HF live-private -> Predico.

Usage: python -m openpredico.live.submit [--dry-run]   (--dry-run skips HF and Predico)
"""

import sys

import pandas as pd

from openpredico.config import load_targets
from openpredico.data import esios, hf
from openpredico.live.predico import Predico
from openpredico.models.baselines import slot_climatology

HISTORY_DAYS = 35
HORIZON_DAYS = 10


def forecast(target: dict, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    tz = target["timezone"]
    indicator = int(target["truth"].removeprefix("esios:"))
    start = (now - pd.Timedelta(days=HISTORY_DAYS)).tz_convert(tz).normalize()
    history = esios.fetch(indicator, start.tz_convert("UTC").isoformat(), now.isoformat())
    if target["unit"] == "MWh":
        # missing quarter-hour = no activation = 0 MWh (PLAN.md > target data quality)
        grid = pd.date_range(
            start.tz_convert("UTC"), now.floor("15min"), freq="15min", inclusive="left"
        )
        history = history.reindex(grid).fillna(0)
    days = pd.date_range(
        now.tz_convert(tz).normalize() + pd.Timedelta(days=1), periods=HORIZON_DAYS, tz=tz
    )
    fc = slot_climatology(history, days, tz=tz)
    # Predico accepts at most 240 h ahead, so the tail of D+10 (unpaid) is cut
    return fc[fc.index <= now + pd.Timedelta(hours=240)], history


def validate(fc: pd.DataFrame, now: pd.Timestamp) -> None:
    assert fc.notna().all().all(), "NaN in forecast"
    assert (fc.q10 <= fc.q50).all() and (fc.q50 <= fc.q90).all(), "quantiles cross"
    assert fc.index.min() > now and fc.index.max() - now <= pd.Timedelta(hours=240), (
        "outside horizon"
    )
    assert (fc.index.to_series().diff().dropna() == pd.Timedelta("15min")).all(), "grid has gaps"


def main(dry_run: bool) -> None:
    now = pd.Timestamp.now(tz="UTC")
    client = None if dry_run else Predico()
    for name, target in load_targets().items():
        fc, history = forecast(target, now)
        validate(fc, now)
        print(
            f"{name}: {len(fc)} steps {fc.index.min()} .. {fc.index.max()}"
        )  # no values: logs are public
        if dry_run:
            continue
        run = f"{name}/{now:%Y-%m-%dT%H%MZ}"
        hf.upload(fc, "live-private", f"forecasts/{run}.parquet", f"Forecast {run}")
        hf.upload(
            history.to_frame("value"), "live-private", f"inputs/{run}.parquet", f"Inputs {run}"
        )
        client.submit_continuous(client.resource_id(target["predico_resource"]), fc)
        print(f"{name}: submitted")
    if client:
        print("coverage:", client.coverage())


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
