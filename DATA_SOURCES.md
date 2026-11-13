# Data sources

Every source used by open-predico, its license, and whether it may be used live. Live forecasts earn prize money, so **live sources must allow commercial use**.

| Source | Used for | License / terms | Commercial use | Attribution | Live | Snapshot on HF |
|---|---|---|---|---|---|---|
| [ESIOS](https://www.esios.ree.es/) (Red Eléctrica) | mFRR truth, market, forecasts, balancing, cross-border | to verify (REE legal notice) | to verify | to verify | planned | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | mFRR balancing energy bids (12.3.B/C, File Library), group `mfrr_bids` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution; to confirm 12.3.B/C is on ENTSO-E's free re-use list | to confirm | "Source: ENTSO-E Transparency Platform" | not yet | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) REST API | 50Hertz truth and features (`data/entsoe.py`): actual generation A75 (B16 solar, B18 wind offshore, B19 wind onshore), group `target_50hertz`; day-ahead wind/solar (A69) and load (A65) forecasts, group `entsoe_forecasts_50hertz` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution | to confirm (A75/A69/A65 on ENTSO-E's free re-use list, to check) | "Source: ENTSO-E Transparency Platform" | German targets | pending terms |
| [Open-Meteo](https://open-meteo.com/) Previous Runs API | 50Hertz weather (`data/openmeteo.py`): wind 10/100 m, radiation, cloud cover, temperature as forecast 48 h ahead, group `weather_50hertz` | Data CC BY 4.0; the free API is non-commercial only | **licence required for production** (API subscription; the maintainer buys it before production) | "Weather data by Open-Meteo.com" | German targets (playground) | pending terms |
| [Elia Open Data](https://opendata.elia.be/) | Belgian solar truth and Elia's forecast (`data/elia.py`): ods032 `measured`, `dayahead11h*`, region Belgium; groups `target_elia`, `tso_forecast_be` | Elia Open Data Licence (to read: portal was down on 2026-11-13) | to verify | "Source: Elia" | not yet (backtest) | pending terms |

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

Forecast availability (`availability` in groups.toml), from the Transparency Regulation
deadlines and one check on 2026-11-08 (`D+1` = 9 Nov): A65 day-ahead load is due by 10:00 D-1 and
was out at 10:12 (not yet at 08:24), so `future`. A69 day-ahead wind/solar is due by 18:00 D-1,
after the 15:00 gate, and was not out at 10:12, so `past` only. ENTSO-E does not expose
publication times, so these are deadlines plus spot checks, not a vintage record; the live run's
stored inputs are. The German live run also reads `da_market` from ESIOS (indicator 600): on
2026-11-08 the Iberian and French prices for D+1 were there at 13:15 but DE-LU, BE and NL only by
14:05, ten minutes before the 14:15 run (ENTSO-E A44 had DE-LU at 13:16: the fallback if ESIOS
turns out late).

## Open-Meteo

ECMWF IFS (9 km) from the Single Runs API: each local target day takes the 00 UTC run of the day
before (23 to 47 h ahead, published around 08:00 UTC, before the 14:00 origin). The backtest and
the live run fetch the same run, so both see the same vintage (not the archive or reanalysis API,
which is truth-like). ECMWF runs are archived since March 2024; other models only since April
2026. Means over five land points (Brandenburg, Mecklenburg-Vorpommern, Saxony-Anhalt, Saxony,
Thuringia) and two Baltic offshore points. One request per day for all points. Free tier:
non-commercial; buy the API licence before production.

Belgium (`weather_be`): one point per province (11), weighted by Elia's monitored PV capacity per
province on 2026-10-25 (Flanders ~72%). Its 09:00 origin comes before the 00 UTC run lands, so it
reads the 18 UTC run of D-2 (29 to 53 h ahead, out around 02:00 UTC).

## Elia Open Data

Solar, Belgium total, 15 min, MW, from the Explore v2.1 export endpoint (one request per field,
no key). ods032 (historical, since 2020-10-01) stops at 2026-10-25: Elia moved solar to ods209
(historical, `measured_upscaled`) and ods210 (near-real-time, `measured`), both split by
`gridconnectiontype` (Total/TSO/DSO); ods087 (near-real-time) is discontinued. Predico's docs
still name ods032 `measured` (settled) and ods087 `realtime`; check which one Predico scores
before going live. `dayahead11h*` is Elia's forecast issued at 11:00 D-1, an hour after the
10:00 gate; values carry the final version only (no vintages), so it is a benchmark, not an input.
