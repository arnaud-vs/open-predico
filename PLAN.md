# open-predico — plan

Open-source forecasting for the [Predico](https://predico-elia.inesctec.pt/docs/) collaborative forecasting platform, and an open benchmark built from it. Anyone can contribute data, features, models or improvements to the repo itself. Prize money is reinvested in better data.

This document records the decisions made before coding and why. Change it via PR like anything else.

## Scope

- **Launch target:** Spain mFRR (EDP program) — `spain_mfrr_energy` (MWh, signed) and `spain_mfrr_price` (EUR/MWh, ~7% exact zeros, almost all in quarter-hours with net downward activation). Day-ahead, 15-min resolution, gate closure 17:00 CET D-1. Ground truth: ESIOS indicators 10250 and 2197.
- **Next:** 50Hertz wind and solar (`germany_wind_50hertz_ca`, `germany_solar_50hertz_ca`, gate 15:00 CET D-1; see Adding a target). More targets are configuration, not code.
- **Main forecasting engines:** time-series foundation models (TSFMs). ML models (LightGBM) and simple baselines are benchmarked alongside.
- No public benchmark exists for Spanish mFRR. Building one is part of the goal.

## Structure

```
open-predico/
├── src/openpredico/
│   ├── config.py         # read + validate TOML configs
│   ├── data/             # sources (esios.py, hf.py), groups.py runner, transforms.py
│   ├── models/           # baselines.py, ml/, tsfm/
│   ├── evaluation/       # backtest.py, metrics.py, score.py
│   └── live/             # predico.py, submit.py, publish.py
├── configs/
│   ├── targets.toml
│   ├── groups.toml
│   ├── experiments/*.toml
│   └── live/<target>.toml
├── results/<config>.csv  # official benchmark results, one file per config
└── tests/
```

`live/` is the only code that touches secrets or the Predico API.

## Targets

Each target is an entry in `configs/targets.toml`: Predico resource name, truth source, kind (`market` = no clipping, may be negative; `generation` = clip to [0, capacity]), timezone, forecast origin, test period. Models, metrics and backtest are target-agnostic.

## Data and features

- **Feature groups** are declared in `configs/groups.toml`: one table per group, one line per column (`source = "esios:<id>"`, optional `geo`, `fill`, and the column's **maximum availability**: `future` = known for the target day before the gate, e.g. day-ahead price; `past` = only known up to the origin, e.g. mFRR actuals). One generic runner (`data/groups.py`) builds any group on the 15-min UTC grid; hourly series are repeated per quarter-hour. A new ESIOS feature is one TOML line; a new source is one fetch module (`entsoe:`, `calendar:` for computed columns).
- The same `build()` produces benchmark history and live inputs, so there is no train/serve skew.
- Groups hold cleaned, near-raw signals. Feature engineering lives in `data/transforms.py` and is selected per experiment.
- **No duplicates:** CI fails if two groups share a source id or a column name.
- **v1 groups (ESIOS):** target history, day-ahead market (OMIE price), REE D+1 forecasts (demand, wind, solar), balancing (aFRR, imbalance; RR dropped: Spain left TERRE end of 2025, no data after 2025-12-30), REE cross-border exchanges (future: PBF net schedule FR/PT/total 10104/10113/10186 and NTC D-1 forecasts 1844/1848/1845/1849; past: P48 net 10015/10014; skip Morocco, Andorra and revised NTC 488), calendar (computed: quarter-hour, weekday, Spanish national holidays, DST-switch days; no regional holidays in v1). Weather and other countries' TSO data later.
- **Time:** UTC everywhere; local time only inside calendar features.

## Experiment configs

One TOML file per experiment; one experiment = one leaderboard row. Each feature is listed individually as `group.column` with its own transforms and a required `availability`:

```toml
target = "spain_mfrr_price"

[model]
name = "chronos2"
context_days = 85

[[features]]
column = "da_market.price"
availability = "future"

[[features]]
column = "target.mfrr_price"
availability = "past"

[[features]]
column = "target.mfrr_price"
transforms = ["lag(7d)"]
availability = "future"   # legal: a 7-day lag is known before the gate
```

Rules enforced by the harness:

- Requesting `future` on a column that is only `past`-available is an error. Requesting `past` on a `future` column is allowed.
- Availability is checked after transforms (`lag(7d)` of a past column can be future; `lag(1d)` cannot).
- **Mask first, transform second:** past values after the origin are set to NaN before any transform runs.
- `group.*` selects a whole group. Raw columns are kept alongside transforms unless `raw = false`.
- TSFM configs normally use no transforms; transforms are mainly for ML models.

## Models

- Every model is a class with `fit(history)` and `predict(past, future) -> q10, q50, q90`. TSFMs load weights in `__init__`; their `fit` does nothing.
- `retrain_every` in the config controls retraining during the backtest.
- Output is always sorted (no quantile crossing); clipping follows the target kind.
- TSFMs forecast from the end of their context, so they roll through the gap between origin and target day; those steps are discarded. ML models predict the target day directly.
- `context_days` is a config choice, capped by each model's maximum context.
- Each TSFM is an optional install (`pip install openpredico[chronos2]`) because their dependencies conflict. Every TSFM runs with covariates (past + known-future); models without covariate support are not in v1. No univariate TSFM runs; baselines are the reference.
- **v1 TSFMs:** Chronos-2, TimesFM 3, Timestral-5M and T0-gamma run (ctx90, with covariates; scores on the dashboard). TimesFM 3 weights are non-commercial: benchmark only, never live. Toto skipped (no zero-shot covariates). Timestral-209M parked: over 5 h per target on a laptop CPU; run it on a cloud GPU.
- **Leakage review (2026-11-02):** 1775/1777/1779 are REE "D+1" forecasts: fine as `future`. 10249 (residual demand forecast) is not D+1 and already holds values two days ahead, so it is probably overwritten: suspect, ablation pending. Imbalance 686/687/1338 appear on ESIOS ~25-40 min after the quarter-hour and aFRR 10252 ~70 min, so the 2 h lag holds (later revisions not checked).
- **Missing price:** quarter-hours without truth drop out of the daily means. Bid mid-price fill (100 MW): ladder built from the ENTSO-E File Library (group `mfrr_bids`, `data/entsoe_bids.py`); `bid_mid_p100` exists on 431 of the 538 test-year quarter-hours without price (80%). Bids are missing 2026-05-03..07-29 (not yet archived by ENTSO-E) and on quarter-hours where every Spanish bid is flagged unavailable (1-3%). Fill not applied yet.
- **Next benchmark:** simple ML (e.g. gradient boosting on the same features) once the leakage review is done.
- **Baselines:** persistence, slot climatology (quantiles per local 15-min slot, last 28 days, weekdays pooled: per-weekday scored worse on 2026 data), zero-inflated climatology.

## Benchmark

- **History start:** 10 Dec 2024 (Spain joins MARI; mFRR price 2197 starts that day). Earlier data is a different market design (old tertiary prices 676/677, ~20% no-activation quarter-hours); optional for volume pretraining only. Day-ahead price is hourly-repeated until 30 Sep 2025. All 15-min series have a gap 28 Apr – 6 May 2025 (Iberian blackout).
- **Test period:** 1 Nov 2025 – 31 Oct 2026, one year, no validation split. Report results per month too: levels and spread shift inside the year (price mean ~100 → ~157 EUR/MWh in Jul–Oct 2026). ML models handle their own train/validation split without leakage.
- **Rolling origin:** for each target day, origin = 14:00 Madrid on D-1 (after OMIE results and PBF at ~13:45). Live runs later (16:15), so it sees at least what the backtest saw; a 16:30-origin test scored slightly better.
- **Storage:** Parquet on Hugging Face (`open-predico/benchmark`), versioned (`v1`, `v2`, …). New feature groups are contributed as build scripts; the maintainer publishes the next dataset version. No data files in git. A version only adds groups and never rewrites an earlier one; `experiment.load_benchmark` takes each group from the latest version folder that holds it. **v1** (2024-12-10 to 2026-10-31): the Spain groups. **v2** (same span, 2026-11-08): the 50Hertz groups `target_50hertz`, `entsoe_forecasts_50hertz`, `weather_50hertz`, `calendar_de`; the German targets reuse v1's `calendar` (Madrid and Berlin share their UTC offsets) and `da_market`.
- **Known limitation:** some ESIOS forecast indicators are overwritten after publication, so their historical values are approximate. Flagged per feature. Live runs store their exact inputs, which gives true vintages going forward.
- **Target data quality (ESIOS, 2025-11 to 2026-10):** ~1.5% of quarter-hours missing in both 10250 and 2197 (263 in Nov 2025, rest scattered, gaps ≤ 2 h). Treated as no activation: volume = 0. Price: fill with the mid-price at 100 MW depth ((price to cover 100 MW up + price to cover 100 MW down) / 2, available bids only) from ENTSO-E Balancing Energy Bids (12.3.B/C, Spanish bids only; see Missing price above). Until the fill is applied, price = NaN (baseline ignores it). Predico's scoring of missing truth is unknown; to confirm. Values hit 1500 (MWh and EUR/MWh) once each in the year; read as a cap, kept as is.

## Evaluation

- **Predico-exact metrics** (from the [evaluation docs](https://predico-elia.inesctec.pt/docs/evaluation/)), per target day:
  - Q50: RMSE = `sqrt(mean_t (o − q50)²)`
  - Q10/Q90: MWI = `mean_t [(q90 − q10) + 10·max(0, q10 − o) + 10·max(0, o − q90)]` (α = 0.2)
- **Diagnostics:** MAE, pinball loss per quantile, interval coverage, max error, skill vs climatology.
- `backtest` writes predictions; `score` writes per-step errors. They are separate so predictions from anywhere can be scored, and live forecasts are scored the same way.
- **Official results** come from the maintainer's full-year runs: `results/<config>.csv` in git, predictions on HF. Anyone can run the backtest locally.

## Live operations

- **Continuous submission only:** one forecast per day covering D+1 to D+10. Predico slices it at each gate closure. If a run fails, yesterday's forecast still covers tomorrow. D+1 comes from the live config (`t0gamma_cov_ctx90`: best Apache-2.0 model in the backtest, 2026-11-03), D+2 to D+10 from the slot-climatology baseline. EDP program pays D+1 only (Predico programs page, Nov 2026), so D+2 to D+10 exist only as a fallback. The fallback is not harmless: on 2026-11-08 the run crashed (HF 429 before submitting) and Predico scored the previous day's climatology D+2 as our D+1. Next: backtest t0 at a D+2 origin (14:00 on D-2) and use it for the tail if it beats climatology there.
- **Scheduler: Modal** (`live/modal_app.py`, Starter plan: 5 crons, $30/month free credit). GitHub Actions cron was dropped (2026-11-05): scheduled runs started 5-7 h late. Three crons serve every target, in Madrid time (Berlin shares the offsets): `submit` at 14:15 and 16:15 forecasts the targets whose session is open (`submit.in_session`: the hour before the gate, so Germany at 14:15 and Spain at 16:15); `check` at :30 and :45 of both hours resubmits only targets whose open challenge isn't `ready` in `market/session/coverage` or that have no t0 run archived in `live-private` today (`submit.missing`; coverage alone counts yesterday's climatology tail as ready); `publish` at 15:30 and 17:30 publishes everything gated and rebuilds the dashboard. `gate_closure` in the API is authoritative. Cost ~$0.65/month.
- **Late or failed features:** left out of the live run (`groups.build(lenient=True)`; known-future columns with gaps on D+1 are dropped in `submit.forecast`), names logged. TSFMs take any covariate set; only the target's own history is required. The latest submission counts.
- **`submit`:** build features → predict → validate (full grid, no NaN, sorted, ≤ 240 h) → submit → save forecast and exact inputs to HF `live-private`. Never prints forecast values (logs may be shared; the repo is public).
- **`publish`:** after gate closure, copy closed days to public HF `live`; fetch actuals and Predico's official scores for us; recompute scores until the month freezes (day 7 of the next month); fail if today's submission is missing.
- **Monitoring:** Modal failure alerts (enable in workspace settings) plus the `publish` check (which catches skipped runs).
- **Secrets:** Modal secret `open-predico`: `HF_TOKEN` (fine-grained, org-scoped), `ESIOS_TOKEN`, `ENTSOE_API_KEY` (German targets), `PREDICO_EMAIL`, `PREDICO_PASSWORD`, `PREDICO_PLAYGROUND_URL` and `PREDICO_URL` (scheme optional). `URL_VAR` in `live/predico.py` picks one: the playground until production unlocks.
- **Before go-live:** dry run on the Predico playground; historical forecast export from the backtest if required.
- **Playground:** production unlocks after 5 days of good playground predictions, so the live record starts on the playground. Every run is stored and published with `env` = `playground` or `production`. Both Spain resources and `germany_wind_50hertz_ca` exist there (checked 2026-11-03). Auth: access token ~1 h, refresh 24 h, no long-lived API key, so the password stays a secret.

## Adding a target

A target is an entry in `targets.toml` (`column`, `groups`, `truth`, `timezone`, `gate`, ...) plus
its feature groups in `groups.toml`. Remaining Spain-specific code: `to_grid` (ESIOS hourly to
15-min switch at Madrid midnight), the mFRR zero fill in `experiment.features`, the single
`PUBLICATION_LAG`, and the Modal crons in `modal_app.py` (one per gate time; Starter allows 5).

### 50Hertz wind and solar (`germany_wind_50hertz_ca`, `germany_solar_50hertz_ca`)

- **Truth:** exactly Predico's: ENTSO-E A75 in `10YDE-VE-------2`, B18 + B19 for wind, B16 for
  solar, each series carried forward on the 15-min grid (curve type A03) before summing
  (`entsoe:` sources, `data/entsoe.py`, DATA_SOURCES.md). ENTSO-E revises it over 35 days.
  The 50Hertz website connector was dropped (2026-11-08): same data, not the scored series.
- **Origin:** 14:00 Berlin, like Spain's 14:00 Madrid; the live run is at 14:15.
- **Capacity:** ENTSO-E A68 installed capacity for 2026 (wind 24.5 GW, solar 25.2 GW); only
  used to clip, the 2025-26 maxima are ~17 GW and ~15 GW.
- **Features:** `target_50hertz` (both targets and the wind components, past; A75 shows up
  ~25-40 min after real time), `entsoe_forecasts_50hertz` (A65 day-ahead load: future, due
  10:00 D-1; A69 day-ahead wind/solar: past, due 18:00 D-1, after the gate), `weather_50hertz`
  (Open-Meteo ECMWF IFS, 00 UTC run of D-1, area means: future), `calendar` (shared, built in
  Europe/Berlin; `holiday_es` comes along) and `calendar_de` (`holiday_de`). `da_market` was
  dropped (2026-11-09): it is Spain's group (ES/PT/FR prices, REE forecasts) and the DE price
  only lands on ESIOS ~14:05. See DATA_SOURCES.md for the availability evidence.
- **Backtest (2026-11-08, test year, ratio to `slot_climatology_28d`):** `t0gamma_cov_ctx90`
  wind RMSE 0.41, Winkler 0.39, coverage 84% (climatology 73%); solar RMSE 0.48, Winkler 0.52,
  coverage 90% (78%). The weather covariates carry it; ablations and the other TSFMs are next.
- **Live:** same config as Spain (`t0gamma_cov_ctx90` D+1, climatology D+2..D+10). `submit` at
  14:15 (session 14:00-15:00), `check` at 14:30/14:45, `publish` at 15:30, all crons shared with
  Spain. Needs `ENTSOE_API_KEY` in the Modal secret, and Open-Meteo's commercial licence before
  production. Risk: ESIOS published the DE-LU/BE/NL prices between 13:15 and 14:05 on
  2026-11-08, close to the 14:15 run; if they arrive late the run logs them as late (NaN). Switch
  `da_price_de` to ENTSO-E A44 (out by 13:16) if that happens.

## Predico rules that shape the repo

- Only the maintainer holds Predico credentials. Contributors never need platform access.
- One team, one account. Anyone else submitting with this repo's models does so openly, with their own model.
- Forecasts are published only after gate closure. Platform information about other forecasters (scores, ranks, ensemble) is never published.
- Every data source used live must allow commercial use (prize money). Each source is documented in `DATA_SOURCES.md`.

## Dashboard

Hugging Face Space (`open-predico/dashboard`), a free static Space: `index.html` + `app.js` (uPlot, no framework) draw JSON that `dashboard/build.py` precomputes from the HF datasets (2026-11-05: replaced the stlite/Streamlit app, which was slow). Gridual palette. A market selector at the top: Spain mFRR, Germany wind and Germany solar (50Hertz) have every tab; Belgium shows "coming soon". Tabs:

- **Backtest:** configs discovered from `predictions/backtest/*/` on HF. Leaderboard modelled on GIFT-Eval: one row per config with type (baseline / foundation model / ML), RMSE and Winkler per target normalised to `slot_climatology_28d` (ratio of mean daily scores on common days), coverage, aggregate score (geometric mean of the normalised scores) and rank. Win-rate matrix per metric (share of common days where A beats B), errors by month/hour/weekday, day viewer.
- **Target data:** the target columns (`v1/target.parquet`, `v2/target_50hertz.parquet`) explored: time series (daily or 15-min window), patterns by quarter-hour/weekday/month, distribution, missing values and zeros per day, price vs energy (Spain).
- **Input features:** Spain: the ESIOS catalog. Germany: the feature columns of `groups.toml` with source, availability and a description.
- **Live:** from HF `live` (written by `live/publish.py` at 17:30 Madrid, after the gate): continuous forecast dotted, days Predico confirmed as submitted solid underneath, actuals, submission record per day, scores vs the same-run climatology baseline, model and input features. Later: Predico's official score, rank if Predico allows publishing it.

## Repo meta

- **License:** Apache-2.0 for code. Data licenses documented separately.
- **DCO:** commits are signed off (`git commit -s`), checked by the DCO GitHub app.
- **Contributions:** any improvement is welcome, including structure, docs, tests and bug fixes. CONTRIBUTING has two example recipes: add data (feature group) and add a model (adapter + config). Open a draft PR early so others see what you're working on.
- **CI (every PR, no secrets):** ruff, offline pytest (metrics, leakage, duplicates, config validation), `backtest --days 2` smoke test on baselines and LightGBM.
- **GitHub settings:** branch protection on `main`, approval required for workflows from outside contributors, CODEOWNERS on `live/`, workflows and live configs.
- **Tooling:** Python 3.12, uv, ruff, pytest.

## Build order

1. Skeleton and repo meta.
2. ESIOS data and benchmark v1.
3. Metrics, baselines, backtest.
4. TSFM adapters.
5. Live submission (after Predico access).

## Open items

- Predico: access to the EDP program and playground; whether we may publish our own rank.
- ESIOS: token; reuse terms for publishing a data snapshot and for commercial use.
- Timestral license ("Modified MIT") to read; T0-gamma weights and Toto CPU support to confirm.
- ESIOS indicator ids for the v1 feature groups.
