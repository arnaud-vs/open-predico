# open-predico

Open-source forecasting for [Predico](https://predico.inesctec.pt/docs/), INESC TEC's collaborative forecasting market, and an open benchmark built from it.

**Why:** Predico pays forecasters for day-ahead quantile forecasts of grid quantities. This repo is one team's entry, built in the open: data pipelines, features, models, backtests and results are public, and anyone can improve them.

**First target:** Spanish mFRR balancing (EDP program): `spain_mfrr_energy` (net activated energy, MWh, signed) and `spain_mfrr_price` (EUR/MWh), 15-min, day-ahead, gate 17:00 CET on D-1. Truth comes from ESIOS (Red Eléctrica).

Status: early, working end to end. The design and every decision so far live in [PLAN.md](PLAN.md).

## How it works

```
ESIOS ─► configs/groups.toml ─► feature frame (15-min UTC) ─► model ─► q10/q50/q90
         (one line per column,                                          │
          future vs past availability)                 backtest ◄───────┤
                                                       live submit ◄────┘ (daily, GitHub Actions)
```

- **Data:** feature groups are declared in `configs/groups.toml`; one runner builds any group. Each column says whether it is known for the target day before the gate (`future`, e.g. day-ahead price) or only up to the forecast time (`past`, e.g. mFRR actuals). The backtest enforces this, so no lookahead.
- **Models:** simple baselines and time-series foundation models (TSFMs), always with covariates.
- **Evaluation:** rolling origin over one test year (Nov 2025 – Oct 2026), forecast made at 14:00 Madrid on D-1, scored exactly like Predico: RMSE on q50, Winkler (α = 0.2) on q10/q90.
- **Live:** one maintainer account submits a continuous forecast daily; forecasts become public only after the gate closes.

## Results (test year, mean daily score, lower is better)

| Model | Energy RMSE | Energy Winkler | Price RMSE | Price Winkler |
|---|---|---|---|---|
| Slot climatology (baseline) | 262.6 | 971.3 | 48.5 | 163.1 |
| Timestral-5M + covariates, 28-day context | **249.5** | **924.4** | **40.7** | **132.3** |

Price is scored on 254 of 365 days: like Predico, a day with any missing truth is skipped. Per-month scores and the exact covariate set will move into `results/` once the benchmark dataset is published.

## Quickstart

```bash
git clone https://github.com/arnaud-vs/open-predico && cd open-predico
uv sync                       # add --extra timestral for the Timestral TSFM
uv run pytest
```

Building data needs a free [ESIOS token](https://api.esios.ree.es/) in `ESIOS_TOKEN`.

## Working with AI agents

Most of this repo was built with Claude Code and agents, and we expect contributors to work the same way. [AGENTS.md](AGENTS.md) is the briefing your agent should read first (Claude Code picks it up through `CLAUDE.md`): repo map, commands, and the rules that matter here (no lookahead, API rate limits, never print secrets or forecast values). Point your agent at PLAN.md for the why.

## Contributing

Any improvement is welcome: data, features, models, docs, tests, or the repo structure itself. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## Disclaimer

Independent project. Not affiliated with INESC TEC, Red Eléctrica, EDP, Elia or 50Hertz.

## License

Code: [Apache-2.0](LICENSE). Data sources keep their own licenses, see [DATA_SOURCES.md](DATA_SOURCES.md).
