---
title: open-predico dashboard
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: Live forecasts, backtest leaderboard and data for open-predico
---

Dashboard for [open-predico](https://github.com/arnaud-vs/open-predico): live forecasts (published after each gate), the backtest leaderboard, the target series and the input catalog. Reads the public datasets [open-predico/benchmark](https://huggingface.co/datasets/open-predico/benchmark) and `open-predico/live`.

A static page: `index.html` + `app.js` draw small JSON files with [uPlot](https://github.com/leeoniya/uPlot). `build.py` writes them to `data/` from the HF datasets (scores via `openpredico.evaluation.metrics`) and `--upload` pushes the Space. Locally: `uv run python dashboard/build.py --demo-live && python -m http.server -d dashboard`.
