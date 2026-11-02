"""Catalog of input series: every ESIOS indicator, marked with where configs/groups.toml uses it.

python -m openpredico.data.catalog --out catalog/inputs.parquet [--cache esios_lists.json]

Two ESIOS calls in total (/indicators and /offer_indicators), 2 s apart. The dashboard's
"Input features" tab reads the result from open-predico/benchmark at catalog/inputs.parquet.
"""

import argparse
import html
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from openpredico.data.esios import BASE_URL
from openpredico.data.groups import load_groups

LISTS = {"esios": "indicators", "esios_offer": "offer_indicators"}
CALENDAR = {
    "qh_of_day": "Quarter-hour of the local (Madrid) day, 0-99.",
    "weekday": "Local day of week, 0 = Monday.",
    "holiday_es": "Spanish national holiday flag (fixed dates plus Good Friday).",
    "dst_switch_day": "Flag for the local days with 92 or 100 quarter-hours (DST switch).",
}
EXTRA = [  # hand-written: sources outside ESIOS's indicator lists
    {
        "source": "entsoe",
        "id": "A37",
        "name": "Balancing energy bids, mFRR, Spain (12.3.B/C)",
        "description": "Planned, parked: per-bid mFRR prices and volumes, to fill missing mFRR "
        "price with the mid-price at 100 MW depth. Needs the ENTSO-E File Library (API keeps "
        "~100 days only). See PLAN.md.",
    },
]


def fetch_lists() -> dict[str, list[dict]]:
    """Raw ESIOS lists {source: [indicator, ...]}; one request each."""
    out = {}
    for source, path in LISTS.items():
        resp = requests.get(
            f"{BASE_URL}/{path}",
            headers={
                "Accept": "application/json; application/vnd.esios-api-v1+json",
                "x-api-key": os.environ["ESIOS_TOKEN"],
            },
            timeout=60,
        )
        resp.raise_for_status()
        out[source] = resp.json()["indicators"]
        time.sleep(2)  # ESIOS's WAF blocks bursts
    return out


def clean(text: str | None, limit: int = 300) -> str:
    """Strip HTML tags and entities, collapse whitespace, trim to `limit` characters."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def build(lists: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per series: source, id, name, description, used, group, column, availability."""
    uses = {}  # (source, id) -> [(group, column, availability)]
    for group, columns in load_groups().items():
        for column, c in columns.items():
            prefix, key = c["source"].split(":")
            uses.setdefault((prefix, key), []).append((group, column, c["availability"]))
    rows = [
        {
            "source": s,
            "id": str(i["id"]),
            "name": i["name"],
            "description": clean(i.get("description")),
        }
        for s, items in lists.items()
        for i in items
    ]
    rows += [
        {"source": "calendar", "id": k, "name": k, "description": v} for k, v in CALENDAR.items()
    ]
    rows += EXTRA
    df = pd.DataFrame(rows)
    # esios_offer never matches: groups.toml only references /indicators ids.
    found = [uses.get((s, i), []) for s, i in zip(df.source, df.id, strict=True)]
    df["used"] = [bool(u) for u in found]
    for k, name in enumerate(["group", "column", "availability"]):
        df[name] = [", ".join(dict.fromkeys(u[k] for u in us)) or None for us in found]
    df["n"] = pd.to_numeric(df.id, errors="coerce")  # numeric order for ESIOS ids
    df = df.sort_values(["used", "source", "n", "id"], ascending=[False, True, True, True])
    return df.drop(columns="n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, help="raw ESIOS lists as JSON; fetched once if missing")
    args = p.parse_args()
    if args.cache and args.cache.exists():
        lists = json.loads(args.cache.read_text())
    else:
        lists = fetch_lists()
        if args.cache:
            args.cache.write_text(json.dumps(lists))
    df = build(lists).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"{len(df)} series, {df.used.sum()} used -> {args.out}")


if __name__ == "__main__":
    main()
