# Contributing

Thanks for helping. Any improvement is welcome: new data, features, models, bug fixes, tests, docs, or a better repo structure. If something is confusing, an issue saying so is a contribution too.

Read [PLAN.md](PLAN.md) first for how the pieces fit together.

## Setup

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Workflow

1. Fork, branch, commit.
2. **Sign off every commit** (`git commit -s`). This adds `Signed-off-by: Your Name <email>` and certifies that you have the right to contribute the code under the project license ([Developer Certificate of Origin](https://developercertificate.org/)). If you contribute in a professional context, check your employer allows it.
3. Open a **draft PR early**, so others can see what you're working on and nobody builds the same thing twice.
4. CI must pass. The maintainer reviews and merges.

## Recipes

These are the two most common contributions. Neither is required — any PR is fine.

### Add data (a feature)

- **ESIOS feature:** one line in `configs/groups.toml` (`source = "esios:<id>"`, `availability = "future"` if known for the target day before the gate, otherwise `"past"`). No code.
- **New source:** one fetch module in `src/openpredico/data/` (e.g. `entsoe.py`), then reference it as `source = "<prefix>:<id>"`.

Add the source to [DATA_SOURCES.md](DATA_SOURCES.md). New columns land in the next benchmark dataset version. Data files never go in git.

### Add a model

1. An adapter in `src/openpredico/models/` (TSFMs in `models/tsfm/`, as an optional extra in `pyproject.toml`): `predict(past, future, days)` returning q10/q50/q90 on the 15-min UTC grid of the target days. TSFMs must use covariates.
2. Run the backtest locally on part of the test year to check it works. Official full-year results are produced by the maintainer.

## Rules

- **Never commit data, forecasts or secrets.**
- Contributors never need access to the Predico platform; only the maintainer submits.
- Don't submit this repo's models to Predico under another account without telling the maintainer — one team, one account.
