# Data sources

Every source used by open-predico, its license, and whether it may be used live. Live forecasts earn prize money, so **live sources must allow commercial use**.

| Source | Used for | License / terms | Commercial use | Attribution | Live | Snapshot on HF |
|---|---|---|---|---|---|---|
| [ESIOS](https://www.esios.ree.es/) (Red Eléctrica) | mFRR truth, market, forecasts, balancing, cross-border | to verify (REE legal notice) | to verify | to verify | planned | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | mFRR balancing energy bids (12.3.B/C, File Library), group `mfrr_bids` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution; to confirm 12.3.B/C is on ENTSO-E's free re-use list | to confirm | "Source: ENTSO-E Transparency Platform" | not yet | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) REST API | 50Hertz truth and features (`data/entsoe.py`): actual generation A75 (B16 solar, B18 wind offshore, B19 wind onshore), group `target_50hertz` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution | to confirm (A75 is on ENTSO-E's free re-use list, to check) | "Source: ENTSO-E Transparency Platform" | German targets | pending terms |

Add a row when you add a source.

## ENTSO-E REST API (50Hertz)

Free API key in `ENTSOE_API_KEY` (also needed in the Modal secret for the German live run and
`publish`). One request per document, production type and year, 2 s apart. Area
`10YDE-VE-------2` (50Hertz control area), the one Predico's 50Hertz program uses.

Truth parity: Predico scores `germany_wind_50hertz_ca` against A75 B18 + B19 and
`germany_solar_50hertz_ca` against A75 B16, process A16. Curve type A03 omits positions that
repeat the previous value, so Predico carries each series forward on the 15-min grid, then sums
(program page, 2026-11). `entsoe.fetch` does the same, up to the last published period (the
unpublished tail stays NaN). ENTSO-E revises these values over a rolling 35 days; Predico
re-imports them, so the benchmark truth (fetched later) can differ slightly from what Predico
scored at the time.

Publication lag of A75 in 50Hertz: ~25-40 min (2026-11-08, 07:25 UTC: data up to 07:00 UTC),
inside the 2 h `PUBLICATION_LAG` the backtest assumes. The 50Hertz website connector (previous
truth and past features) was dropped: A75 is the scored series and is just as fresh.
