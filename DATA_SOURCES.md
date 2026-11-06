# Data sources

Every source used by open-predico, its license, and whether it may be used live. Live forecasts earn prize money, so **live sources must allow commercial use**.

| Source | Used for | License / terms | Commercial use | Attribution | Live | Snapshot on HF |
|---|---|---|---|---|---|---|
| [ESIOS](https://www.esios.ree.es/) (Red Eléctrica) | mFRR truth, market, forecasts, balancing, cross-border | to verify (REE legal notice) | to verify | to verify | planned | pending terms |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | mFRR balancing energy bids (12.3.B/C, File Library), group `mfrr_bids` | ENTSO-E Transparency Platform terms of use: reuse allowed with source attribution; to confirm 12.3.B/C is on ENTSO-E's free re-use list | to confirm | "Source: ENTSO-E Transparency Platform" | not yet | pending terms |

| [50Hertz wind power](https://www.50hertz.com/en/Transparency/GridData/Productiongridfeed-in/Windpower) | Control-area wind actuals and published forecasts; `fiftyhertz.py` | Public download; reuse/redistribution terms not verified | to verify | Source: 50Hertz | connector available; not enabled live | pending terms |

Add a row when you add a source.

## 50Hertz connector

No credentials required. `fiftyhertz.fetch("actual_sum", start, end)` returns
onshore + offshore MW over UTC `[start, end)`; `actual_total` preserves the separately
published total. They can differ: the 30 October–5 November 2026 audit found a
maximum 0.26 MW difference. Predico settlement parity has not yet been verified.

`fetch_frame("actual", start, end)` fetches each intersecting German local day
once and returns `total`, `onshore`, `offshore`, `sum` on UTC timestamps. It preserves
missing observations (including -1/n.a.) and the repeated autumn hour. It records
retrieval time in DataFrame attrs; this is not a provider publication timestamp.
Daily calls have retries and two-second spacing. Use annual downloads, not this
daily fetcher, for multi-year backfill. The website endpoint is not a versioned API.

`forecast_<component>` is also supported, but the website documents publication at
18:00 D-1, after the 15:00 D-1 Predico gate. The group validator rejects `future`
availability for this source. Historical forecasts are not issue-time vintages.

See `examples/fiftyhertz_groups.toml` for the source keys. This example is not in
active feature groups: first implement target-specific group selection, target
mapping, publication lags and the renewable submission schedule from PLAN.md.
No live target or submission is enabled by adding the connector.

Offline validation: `PYTHONPATH=src python -m pytest tests/test_fiftyhertz.py`.
Live read-only smoke check (downloads one day; prints metadata, not values):

```python
import pandas as pd
from openpredico.data.fiftyhertz import fetch_frame
frame = fetch_frame("actual", "2026-11-03T23:00Z", "2026-11-04T23:00Z")
assert len(frame) == 96 and frame.index.is_unique
assert frame.index.to_series().diff().dropna().eq(pd.Timedelta("15min")).all()
print(frame.shape, str(frame.index.tz), frame.isna().sum().to_dict())
```
