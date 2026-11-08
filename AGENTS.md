# Agent briefing

Read [PLAN.md](PLAN.md) before changing anything: it records every design decision and why.

## Map

- `configs/targets.toml`: forecasting targets (Predico resource, truth source, origin, test period).
- `configs/groups.toml`: feature columns, one line each, with `availability` = `future` | `past`.
- `src/openpredico/data/`: `esios.py` and `entsoe.py` (API clients), `groups.py` (builds feature frames), `hf.py` (Hugging Face storage).
- `src/openpredico/models/`: `baselines.py`, `tsfm/` (one adapter per foundation model, optional extras).
- `src/openpredico/evaluation/`: `backtest.py` (rolling origin), `metrics.py` (Predico's scores).
- `src/openpredico/live/`: daily submission. The only code that touches Predico or secrets.

Adding a target? Read PLAN.md > Adding a target first: it lists the Spain-specific code left.

## Commands

```bash
uv sync [--extra timestral]
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Rules

- **No lookahead.** A model may only see `past` columns published before the origin (14:00 Madrid on D-1, minus the publication lag) and `future` columns for the target day. Never fill or interpolate truth.
- **Respect API rate limits.** ESIOS blocks IPs that send bursts: at most one request per second, and budget the total. Prefer bulk downloads to paging. ENTSO-E: at most one request every two seconds.
- **Never print secrets or forecast values.** Tokens come from env vars (`ESIOS_TOKEN`, `ENTSOE_API_KEY`, `HF_TOKEN`; Predico credentials are maintainer-only). Don't log request URLs that carry tokens. The repo is public, and live forecasts stay private until the gate closes.
- **Never commit data, forecasts, PDFs or secrets.** Data lives on Hugging Face (`open-predico` org).
- **UTC everywhere**; local time (Europe/Madrid) only for calendar features and day boundaries. DST days have 92 or 100 quarter-hours.
- **Keep it small.** Stdlib and pandas first, no speculative abstractions, one small test for non-trivial logic. Mark deliberate shortcuts with a `# ponytail:` comment naming the limit.
- Sign off commits (`git commit -s`), see CONTRIBUTING.md.
