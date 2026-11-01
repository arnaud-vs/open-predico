# open-predico — plan

Open-source forecasting for the [Predico](https://predico-elia.inesctec.pt/docs/) collaborative forecasting platform, and an open benchmark built from it. Anyone can contribute data, features, models or improvements to the repo itself. Prize money is reinvested in better data.

This document records the decisions made before coding and why. Change it via PR like anything else.

## Scope

- **Launch target:** Spain mFRR (EDP program) — `spain_mfrr_energy` (MWh, signed) and `spain_mfrr_price` (EUR/MWh, many exact zeros). Day-ahead, 15-min resolution, gate closure 17:00 CET D-1. Ground truth: ESIOS indicators 10250 and 2197.
- **Next:** 50Hertz wind (`germany_wind_50hertz_ca`, gate 15:00 CET D-1). More targets are configuration, not code.
- **Main forecasting engines:** time-series foundation models (TSFMs). ML models (LightGBM) and simple baselines are benchmarked alongside.
- No public benchmark exists for Spanish mFRR. Building one is part of the goal.

## Structure

```
open-predico/
├── src/openpredico/
│   ├── config.py         # read + validate TOML configs
│   ├── data/             # sources (esios.py, hf.py), load(), groups/, transforms.py
│   ├── models/           # baselines.py, ml/, tsfm/
│   ├── evaluation/       # backtest.py, metrics.py, score.py
│   └── live/             # predico.py, submit.py, publish.py
├── configs/
│   ├── targets.toml
│   ├── experiments/*.toml
│   └── live/<target>.toml
├── results/<config>.csv  # official benchmark results, one file per config
└── tests/
```

`live/` is the only code that touches secrets or the Predico API.

## Targets

Each target is an entry in `configs/targets.toml`: Predico resource name, truth source, kind (`market` = no clipping, may be negative; `generation` = clip to [0, capacity]), timezone, forecast origin, test period. Models, metrics and backtest are target-agnostic.

## Data and features

- **Feature group** = one file in `data/groups/` with a `build(start, end)` function returning a frame on the 15-min UTC grid, plus metadata: source ids, license, and each column's **maximum availability** (`future` = known for the target day before the gate, e.g. day-ahead price; `past` = only known up to the origin, e.g. mFRR actuals).
- The same `build()` produces benchmark history and live inputs, so there is no train/serve skew.
- Groups hold cleaned, near-raw signals. Feature engineering lives in `data/transforms.py` and is selected per experiment.
- **No duplicates:** CI fails if two groups share a source id or a column name.
- **v1 groups (ESIOS):** target history, day-ahead market (OMIE price), REE D+1 forecasts (demand, wind, solar), balancing (aFRR/RR, imbalance), REE cross-border exchanges, calendar (computed). Weather and other countries' TSO data later.
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
- Each TSFM is an optional install (`pip install openpredico[chronos2]`) because their dependencies conflict. Models without covariate support run univariate.
- **v1 TSFMs:** Chronos-2, TimesFM 3, Timestral (5M and 209M), T0-gamma (weights to confirm), Toto (CPU support to confirm).
- **Baselines:** persistence, weekly-pattern climatology (quantiles per 15-min slot × weekday), zero-inflated climatology.

## Benchmark

- **Test period:** 1 Nov 2025 – 31 Oct 2026, one year, no validation split. ML models handle their own train/validation split without leakage.
- **Rolling origin:** for each target day, origin = 13:00 Madrid on D-1 (the live run time, not the gate), so the backtest never sees more than live does.
- **Storage:** Parquet on Hugging Face (`open-predico/benchmark`), versioned (`v1`, `v2`, …). New feature groups are contributed as build scripts; the maintainer publishes the next dataset version. No data files in git.
- **Known limitation:** some ESIOS forecast indicators are overwritten after publication, so their historical values are approximate. Flagged per feature. Live runs store their exact inputs, which gives true vintages going forward.

## Evaluation

- **Predico-exact metrics** (from the [evaluation docs](https://predico-elia.inesctec.pt/docs/evaluation/)), per target day:
  - Q50: RMSE = `sqrt(mean_t (o − q50)²)`
  - Q10/Q90: MWI = `mean_t [(q90 − q10) + 10·max(0, q10 − o) + 10·max(0, o − q90)]` (α = 0.2)
- **Diagnostics:** MAE, pinball loss per quantile, interval coverage, max error, skill vs climatology.
- `backtest` writes predictions; `score` writes per-step errors. They are separate so predictions from anywhere can be scored, and live forecasts are scored the same way.
- **Official results** come from the maintainer's full-year runs: `results/<config>.csv` in git, predictions on HF. Anyone can run the backtest locally.

## Live operations

- **Continuous submission only:** one forecast per day covering D+1 to D+10. Predico slices it at each gate closure. If a run fails, yesterday's forecast still covers tomorrow. D+1 comes from the live config, D+2 to D+10 from the weekly-pattern baseline.
- **Schedule (GitHub Actions, UTC):** `submit` at 13:00 (14:00 CET / 15:00 CEST, after OMIE results) with a retry at 14:00 (≥1 h before the gate in both CET and CEST); `publish` at 16:30.
- **Late features:** filled with NaN and recorded; the retry submits the full forecast if the data has arrived. The latest submission counts.
- **`submit`:** build features → predict → validate (full grid, no NaN, sorted, ≤ 240 h) → submit → check coverage → save forecast and exact inputs to HF `live-private`. Never prints forecast values (Actions logs are public).
- **`publish`:** after gate closure, copy closed days to public HF `live`; fetch actuals and Predico's official scores for us; recompute scores until the month freezes (day 7 of the next month); fail if today's submission is missing.
- **Monitoring:** GitHub failure emails plus the `publish` check (which catches skipped scheduled runs).
- **Secrets:** GitHub `production` environment, `main` only: `HF_TOKEN` (fine-grained, org-scoped), `ESIOS_TOKEN`, `PREDICO_EMAIL`, `PREDICO_PASSWORD`.
- **Before go-live:** dry run on the Predico playground; historical forecast export from the backtest if required.

## Predico rules that shape the repo

- Only the maintainer holds Predico credentials. Contributors never need platform access.
- One team, one account. Anyone else submitting with this repo's models does so openly, with their own model.
- Forecasts are published only after gate closure. Platform information about other forecasters (scores, ranks, ensemble) is never published.
- Every data source used live must allow commercial use (prize money). Each source is documented in `DATA_SOURCES.md`.

## Dashboard

Hugging Face Space (`open-predico/dashboard`, Streamlit) with two tabs:

- **Backtest:** leaderboard, paired comparison between configs (share of days won), errors by hour/weekday/month.
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
