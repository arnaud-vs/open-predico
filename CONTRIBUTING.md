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

### Add data (a feature group)

One file in `src/openpredico/data/groups/` with:

- `build(start, end)` returning a DataFrame on the 15-min UTC grid.
- Metadata: source ids, license, and each column's maximum availability (`future` if known for the target day before the gate, otherwise `past`).

Add the source to [DATA_SOURCES.md](DATA_SOURCES.md). New groups land in the next benchmark dataset version. Data files never go in git.

### Add a model

1. An adapter in `src/openpredico/models/` with `fit(history)` and `predict(past, future)` returning q10/q50/q90.
2. An experiment config in `configs/experiments/` listing the features (each with `availability`), transforms and model settings.
3. Run a short backtest locally to check it works. Official full-year results are produced by the maintainer.

## Rules

- **Never commit data, forecasts or secrets.**
- Contributors never need access to the Predico platform; only the maintainer submits.
- Don't submit this repo's models to Predico under another account without telling the maintainer — one team, one account.
