"""Catalog of input series: every ESIOS indicator, marked with where configs/groups.toml uses it.

python -m openpredico.data.catalog --out catalog/inputs.parquet [--cache esios_lists.json]

Five ESIOS calls in total, 2 s apart: /indicators and /offer_indicators in English and Spanish,
plus /archives (bulk files). The dashboard's "Input features" tab reads the result from
open-predico/benchmark at catalog/inputs.parquet.
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
    "holiday_de": "German national holiday flag (fixed dates plus Easter-based days).",
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
SNP = (
    r"\bNPS\b|\bSNP\b|\bNPT\b|\bTNP\b|non.?peninsular|balear|mallorca|menorca|ibiza|formentera"
    r"|canar|tenerife|lanzarote|fuerteventura|gomera|palma|hierro|ceuta|melilla"
)
HORIZON = {"A": "yearly", "T": "quarterly", "M": "monthly", "D": "daily", "H": "hourly"}
# Ordered (category, subcategory, regex on the English name); first match wins, else "Other".
# fmt: off
BAL, XB, SET = "Balancing services", "Cross-border", "Settlement"
SCH, MKT = "Schedules", "Prices & markets"
RULES = [
    (BAL, "Bid curves", r"bids? curves?|curvas? de ofertas"),
    (XB, "Balancing exchange", r"cross.?border balancing|igcc|\bRR\b.*(export|import)"),
    (SET, "Price components", r"component|last resort|competitive suppliers|RD-L|sum of comp"),
    (SET, "Retail tariffs", r"tariff|toll|\bDHA\b|\bDHS\b|2\.0TD|PVPC|marketing cost"),
    (SET, "Costs", r"capacity payment|charge rights|collection rights|cost of|income|quota"
                   r"|losses|settlement|forward"),
    (SET, "Metering", r"clients|customers|border points|access rate|voltage level|final energy"),
    ("Emissions", "CO2", r"emission|CO2"),
    ("Non-peninsular (SNP)", "System", SNP),
    (SCH, "PBF (day-ahead base)", r"\bPBF\b|\bPDBF\b|Base Daily"),
    (SCH, "PVP (viable)", r"\bPVP\b|Viable Daily"),
    (SCH, "P48 (operating)", r"\bP48\b|Operating Hourly"),
    (SCH, "PHF (intraday final)", r"\bPHF|Final Hourly"),
    (BAL, "mFRR", r"\bmFRR\b|tertiary|scheduled activation|direct activation"),
    (BAL, "aFRR", r"\baFRR\b|secondary|BSPs?"),
    (BAL, "RR", r"\bRR\b|replacement reserve"),
    (BAL, "FCR", r"\bFCR\b|primary"),
    (BAL, "Technical constraints", r"technical constr|constraints|restricted energy|curtail"),
    (BAL, "Imbalances", r"imbalance"),
    (BAL, "Other services", r"interruptib|demand response|reserv|security energy|activations"
                            r"|balancing|voltage control|power factor|emergency"),
    (XB, "Auctions", r"auction|congestion rent|holders|participants|long.?term capacity"),
    (XB, "Capacity (NTC/ATC)", r"\bNTC\b|\bATC\b|interconnection capacity"),
    (XB, "Exchanges", r"france|portugal|morocco|andorra|interconnection|import|export"
                      r"|countertrading|net.*position"),
    ("Forecasts", "REE forecasts", r"forecast|D\+1|H\+3"),
    (MKT, "Intraday", r"intraday"),
    (MKT, "Day-ahead", r"day.?ahead|spot|daily market|marginal price"),
    (MKT, "Other prices", r"price"),
    ("Capacity", "Installed / available", r"installed|capacity|enabled"),
    ("Demand", "Demand", r"demand|consumption"),
    ("Generation", "By technology", r"generation|wind|solar|nuclear|hydro|coal|cycle|storage"
                                    r"|renewable|battery|pumping"),
]
AREAS = [  # first match on the name wins, else Peninsula
    ("Ceuta-Melilla", r"ceuta|melilla"),
    ("SNP-islands", SNP),
    ("cross-border", r"france|portugal|morocco|andorra|interconnection|cross.?border|\bNTC\b"
                     r"|\bATC\b|igcc|auction"),
    ("national", r"national|spain|spanish|autonomous|component|tariff|clients|customers"),
]
# fmt: on


def get(path: str, locale: str) -> dict:
    resp = requests.get(
        f"{BASE_URL}/{path}",
        params={"locale": locale},
        headers={
            "Accept": "application/json; application/vnd.esios-api-v1+json",
            "x-api-key": os.environ["ESIOS_TOKEN"],
        },
        timeout=90,
    )
    resp.raise_for_status()
    time.sleep(2)  # ESIOS's WAF blocks bursts
    return resp.json()


def fetch_lists() -> dict:
    """Raw ESIOS lists {"en"|"es": {source: [indicator, ...]}, "esios_archive": [...]}."""
    out = {loc: {s: get(p, loc)["indicators"] for s, p in LISTS.items()} for loc in ("en", "es")}
    return out | {"esios_archive": get("archives", "en")["archives"]}


def clean(text: str | None, limit: int = 300) -> str:
    """Strip HTML tags and entities, collapse whitespace, trim to `limit` characters."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def first(rules: list, text: str, default):
    return next((out for *out, rx in rules if re.search(rx, text, re.I)), default)


def build(lists: dict) -> pd.DataFrame:
    """One row per series: source, category, subcategory, area, id, name, name_es, description,
    used, group, column, availability."""
    uses = {}  # (source, id) -> [(group, column, availability)]
    for group, columns in load_groups().items():
        for column, c in columns.items():
            prefix, key = c["source"].split(":")
            uses.setdefault((prefix, key), []).append((group, column, c["availability"]))
    rows = []
    for s in LISTS:
        en = {str(i["id"]): i for i in lists.get("en", {}).get(s, [])}
        es = {str(i["id"]): i for i in lists.get("es", {}).get(s, [])}
        for k in en | es:
            e, x = en.get(k, {}), es.get(k, {})
            rows.append(
                {
                    "source": s,
                    "id": k,
                    "name": (e.get("name") or x.get("name", "")).strip(),
                    "name_es": x.get("name"),
                    "description": clean(e.get("description") or x.get("description")),
                }
            )
    rows += [
        {"source": "calendar", "id": k, "name": k, "description": v} for k, v in CALENDAR.items()
    ]
    rows += EXTRA
    df = pd.DataFrame(rows)
    df[["category", "subcategory"]] = [first(RULES, n, ("Other", "Other")) for n in df.name]
    df.loc[df.source == "calendar", ["category", "subcategory"]] = ("Calendar", "Calendar")
    archives = pd.DataFrame(
        {
            "source": "esios_archive",
            "id": str(a["id"]),
            "name": a["name"],
            "description": f"{a['archive_type']} file, {HORIZON.get(a['horizon'], a['horizon'])}: "
            + ", ".join(t["name"] for t in a["taxonomy_terms"] if t["name"] != "Downloads"),
            "category": "Bulk files (archives)",
            "subcategory": a["taxonomy_terms"][-1]["name"],
        }
        for a in lists.get("esios_archive", [])
    )
    df = pd.concat([df, archives], ignore_index=True)
    df["area"] = [first(AREAS, n, ["Peninsula"])[0] for n in df.name]
    # esios_offer never matches: groups.toml only references /indicators ids.
    found = [uses.get((s, i), []) for s, i in zip(df.source, df.id, strict=True)]
    df["used"] = [bool(u) for u in found]
    for k, name in enumerate(["group", "column", "availability"]):
        df[name] = [", ".join(dict.fromkeys(u[k] for u in us)) or None for us in found]
    df["n"] = pd.to_numeric(df.id, errors="coerce")  # numeric order for ESIOS ids
    df = df.sort_values(
        ["used", "category", "subcategory", "source", "n", "id"],
        ascending=[False, True, True, True, True, True],
    )
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
    print(df.category.value_counts().to_string())


if __name__ == "__main__":
    main()
