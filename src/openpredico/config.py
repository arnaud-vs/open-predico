"""Read and validate the TOML files in configs/."""

import tomllib
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

# ponytail: assumes a repo checkout (editable install). Ship configs as package data if
# we ever publish wheels.
CONFIGS = Path(__file__).resolve().parents[2] / "configs"

TARGET_KEYS = {
    "predico_resource",
    "truth",
    "column",
    "groups",
    "unit",
    "kind",
    "timezone",
    "origin",
    "gate",
    "test_period",
}
KINDS = {"market", "generation"}


def load_targets(path: Path = CONFIGS / "targets.toml") -> dict[str, dict]:
    """Return {target_name: definition}, raising ValueError on any invalid entry."""
    targets = tomllib.loads(path.read_text())
    for name, t in targets.items():
        _validate_target(name, t)
    return targets


def live_targets() -> dict[str, dict]:
    """The targets live/ submits, checks and publishes: all but those with `live = false`."""
    return {n: t for n, t in load_targets().items() if t.get("live", True)}


def _validate_target(name: str, t: dict) -> None:
    missing = TARGET_KEYS - t.keys()
    if missing:
        raise ValueError(f"target {name!r}: missing keys {sorted(missing)}")
    if t["kind"] not in KINDS:
        raise ValueError(f"target {name!r}: kind must be one of {sorted(KINDS)}")
    if t["kind"] == "generation" and "capacity" not in t:
        raise ValueError(f"target {name!r}: generation targets need a capacity")
    ZoneInfo(t["timezone"])  # raises on unknown timezone
    time.fromisoformat(t["origin"])
    time.fromisoformat(t["gate"])
    start, end = (date.fromisoformat(d) for d in t["test_period"])
    if start >= end:
        raise ValueError(f"target {name!r}: test_period start must be before end")
