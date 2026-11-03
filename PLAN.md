# open-predico — plan

Open-source forecasting for the [Predico](https://predico-elia.inesctec.pt/docs/) collaborative forecasting platform, and an open benchmark built from it. Anyone can contribute data, features, models or improvements to the repo itself. Prize money is reinvested in better data.

This document records the decisions made before coding and why. Change it via PR like anything else.

## Scope

- **Launch target:** Spain mFRR (EDP program) — `spain_mfrr_energy` (MWh, signed) and `spain_mfrr_price` (EUR/MWh, ~7% exact zeros, almost all in quarter-hours with net downward activation). Day-ahead, 15-min resolution, gate closure 17:00 CET D-1. Ground truth: ESIOS indicators 10250 and 2197.
- **Next:** 50Hertz wind (`germany_wind_50hertz_ca`, gate 15:00 CET D-1). More targets are configuration, not code.
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
- **Rolling origin:** for each target day, origin = 14:00 Madrid on D-1 (live run 13:00 UTC = 14:00 CET, after OMIE results and PBF at ~13:45; one hour conservative in summer), so the backtest never sees more than live does.
- **Storage:** Parquet on Hugging Face (`open-predico/benchmark`), versioned (`v1`, `v2`, …). New feature groups are contributed as build scripts; the maintainer publishes the next dataset version. No data files in git.
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

- **Continuous submission only:** one forecast per day covering D+1 to D+10. Predico slices it at each gate closure. If a run fails, yesterday's forecast still covers tomorrow. D+1 comes from the live config (`t0gamma_cov_ctx90`: best Apache-2.0 model in the backtest, 2026-11-03), D+2 to D+10 from the slot-climatology baseline. EDP program pays D+1 only (Predico programs page, Nov 2026), so D+2 to D+10 exist only as a fallback.
- **Schedule (GitHub Actions, UTC):** `submit` at 13:00 (14:00 CET / 15:00 CEST, after OMIE results) with a retry at 14:00 (≥1 h before the gate in both CET and CEST); `publish` at 16:30.
- **Late features:** filled with NaN and recorded; the retry submits the full forecast if the data has arrived. The latest submission counts.
- **`submit`:** build features → predict → validate (full grid, no NaN, sorted, ≤ 240 h) → submit → check coverage → save forecast and exact inputs to HF `live-private`. Never prints forecast values (Actions logs are public).
- **`publish`:** after gate closure, copy closed days to public HF `live`; fetch actuals and Predico's official scores for us; recompute scores until the month freezes (day 7 of the next month); fail if today's submission is missing.
- **Monitoring:** GitHub failure emails plus the `publish` check (which catches skipped scheduled runs).
- **Secrets:** GitHub `production` environment, `main` only: `HF_TOKEN` (fine-grained, org-scoped), `ESIOS_TOKEN`, `ENTSOE_API_KEY`, `PREDICO_EMAIL`, `PREDICO_PASSWORD`.
- **Before go-live:** dry run on the Predico playground; historical forecast export from the backtest if required.

## Predico rules that shape the repo

- Only the maintainer holds Predico credentials. Contributors never need platform access.
- One team, one account. Anyone else submitting with this repo's models does so openly, with their own model.
- Forecasts are published only after gate closure. Platform information about other forecasters (scores, ranks, ensemble) is never published.
- Every data source used live must allow commercial use (prize money). Each source is documented in `DATA_SOURCES.md`.

## Dashboard

Hugging Face Space (`open-predico/dashboard`): Streamlit run in the browser with stlite, so a free static Space (Docker and Gradio Spaces need a paid plan for orgs). Tabs:

- **Backtest:** configs discovered from `predictions/backtest/*/` on HF. Leaderboard modelled on GIFT-Eval: one row per config with type (baseline / foundation model / ML), RMSE and Winkler per target normalised to `slot_climatology_28d` (ratio of mean daily scores on common days), coverage, aggregate score (geometric mean of the normalised scores) and rank. Win-rate matrix per metric (share of common days where A beats B), errors by month/hour/weekday, day viewer.
- **Target data:** `v1/target.parquet` explored: time series (daily or 15-min window), patterns by quarter-hour/weekday/month, distribution, missing values and zeros per day, price vs energy.
- **Live:** submitted forecasts vs actuals for any day, daily scores over time next to Predico's official score, rank if Predico allows publishing it.

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
