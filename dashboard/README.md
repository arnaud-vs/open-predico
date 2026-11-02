---
title: open-predico dashboard
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: Backtest leaderboard and target data for open-predico
---

Dashboard for [open-predico](https://github.com/arnaud-vs/open-predico). Backtest leaderboard, win rates and error breakdowns, an exploration of the target series, and a catalog of input series (every ESIOS indicator, marked when used as a feature). Reads the public dataset [open-predico/benchmark](https://huggingface.co/datasets/open-predico/benchmark).

Streamlit app run in the browser with [stlite](https://github.com/whitphx/stlite) (static Space, no server). Locally: `pip install -r requirements.txt && streamlit run app.py`.
