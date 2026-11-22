# Data sources

Every source used by open-predico, its license, and whether it may be used live. Live forecasts earn prize money, so **live sources must allow commercial use**.

| Source | Used for | License / terms | Commercial use | Attribution | Live | Snapshot on HF |
|---|---|---|---|---|---|---|
| [ESIOS](https://www.esios.ree.es/) (Red Eléctrica) | mFRR truth, market, forecasts, balancing, cross-border | to verify (REE legal notice) | to verify | to verify | planned | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | mFRR balancing energy bids (12.3.B/C, File Library), group `mfrr_bids` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution; to confirm 12.3.B/C is on ENTSO-E's free re-use list | to confirm | "Source: ENTSO-E Transparency Platform" | not yet | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) REST API | 50Hertz truth and features (`data/entsoe.py`): actual generation A75 (B16 solar, B18 wind offshore, B19 wind onshore), group `target_50hertz`; day-ahead wind/solar (A69) and load (A65) forecasts, group `entsoe_forecasts_50hertz` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution | to confirm (A75/A69/A65 on ENTSO-E's free re-use list, to check) | "Source: ENTSO-E Transparency Platform" | German targets | pending terms |
| [Open-Meteo](https://open-meteo.com/) Previous Runs API | 50Hertz and Belgian weather (`data/openmeteo.py`): ECMWF IFS (and ICON-D2 for Belgium) single runs; groups `weather_50hertz`, `weather_be`, `weather_be_d2` | Data CC BY 4.0; the free API is non-commercial only | **licence required for production** (API subscription; the maintainer buys it before production) | "Weather data by Open-Meteo.com" | German targets (playground) | pending terms |
| [WeatherNext 3](https://deepmind.google/science/weathernext/) (Google DeepMind) via BigQuery | Belgian and 50Hertz weather for the backtest (`data/weathernext.py`): ensemble means, groups `weather_be_wn3`, `weather_50hertz_wn3` | WeatherNext data licence of the BigQuery linked dataset (`open-predico.weathernext_3`): no redistribution | to verify | to verify | not yet: code ready, keyless Workload Identity Federation on Modal (docs/weathernext-live.md); off until commercial use is confirmed | **never public**: private HF `open-predico/weather-private` |
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
province on 2026-10-25 (Flanders ~72%). ECMWF runs land ~6h40 after init (06 UTC run: 12:40 UTC
on 2026-11-14), so Belgium's 09:00 origin (07:00 UTC in summer) reads the 00 UTC run of D-1.
`weather_be_d2`: same points from DWD's ICON-D2 (2 km, 48 h, every 3 h), 03 UTC run of D-1 (lands
~1h55 after init); single runs archived since 2026-04-02 only.

## WeatherNext 3 (BigQuery)

Google DeepMind's WeatherNext 3 (0.1 deg, hourly runs since 2026, 48 h ahead; 00/06/12/18 UTC
runs 360 h), table `open-predico.weathernext_3.weathernext_3_0_0_0p1deg` (a view over a linked
dataset), read with Application Default Credentials (`uv sync --extra weathernext`), on Modal
with keyless Workload Identity Federation (docs/weathernext-live.md). Per point
of `openmeteo.POINTS` the nearest 0.1 deg cell (`ST_DISTANCE`), then the same weights as the
Open-Meteo groups; units converted to Open-Meteo's (J/m2 per hour to W/m2, K to degC, 0-1 to %;
diffuse = global - direct).

- **Runs:** Belgium reads the 23 UTC run of D-2, Germany the 04 UTC run of D-1. Runs landed on
  GCS 6h15-7h00 after init (Aug-Nov 2026; e.g. 23 UTC of 2026-11-20 at 05:06 UTC); we assume
  7h45, so both are out at least 15 min before the 09:00 Brussels / 14:00 Berlin origin in
  summer (07:00 / 12:00 UTC), and both cover the whole local target day.
- **Vintage:** runs before 2026-08-28 were all written on 2026-08-27, i.e. reforecasts by the
  current model, not what was available then. Backtest scores before that date may be optimistic
  if that period overlaps the model's training data (to check).
- **Cost:** BigQuery bills scanned bytes. Every query filters one `init_time` and the points'
  cells (`ST_DWITHIN` of a MULTIPOINT, 8 km), and takes the lead from the array offset
  (offset i = i+1 h ahead; selecting `hours` or `time` adds ~125 GB to the estimate). The
  dry-run estimate is ~130 GB per selected column (~750 GB for ours) while a run bills up to ~230 MB;
  BigQuery refuses a `maximum_bytes_billed` below the estimate, so a query may bill up to the
  estimate + 10% and `fetch` stops when one bills over 1 GB. Never query without the
  `init_time` filter. The 2026-04-03 to 2026-11-21 backfill (468 runs) billed 32 GB.
- **Licence:** no redistribution: the groups live in the private dataset `weather-private`
  (`groups.py` routes every group with a `weathernext:` column there); `load_benchmark` reads
  it when the login can, else leaves the groups out.

## Elia Open Data

Solar, Belgium total, 15 min, MW, from the Explore v2.1 export endpoint (one request per field,
no key). ods032 (historical, since 2020-10-01) stops at 2026-10-25: Elia moved solar to ods209
(historical, `measured_upscaled`) and ods210 (near-real-time, `measured`), both split by
`gridconnectiontype` (Total/TSO/DSO); ods087 (near-real-time) is discontinued. Predico's docs
still name ods032 `measured` (settled) and ods087 `realtime`; check which one Predico scores
before going live. `dayahead11h*` is Elia's forecast issued at 11:00 D-1, an hour after the
10:00 gate; values carry the final version only (no vintages), so it is a benchmark, not an input.
