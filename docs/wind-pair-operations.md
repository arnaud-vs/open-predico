# 50Hertz T0 + Timestral wind pair: deployment and rollback

This branch adds an isolated producer and a guarded handoff into the existing submission job.
It is **not deployed by a merge**. Production is currently blocked on the research seed asset
rights described below. Once resolved, deploy both apps after bootstrap and smoke validation.
No solar/Spain model, existing cron, gate time or public-publishing logic changes.

## What runs automatically

All times below are Europe/Berlin (the same clock as Brussels), including DST:

| Time | Action |
|---|---|
| 13:00 | New `open-predico-wind-pair` worker collects/cache-updates sources and predicts |
| 13:15 | Same serialized worker retries failures; a completed day is not recomputed |
| 13:30 | Frozen model information origin; new source acquisition after this is rejected |
| Before 14:00 | Candidate computation must finish; cached-bundle upload retries stop at 14:00 |
| 14:15 | Existing `open-predico` submission uses a verified same-day pair, otherwise its incumbent |
| 14:30 / 14:45 | Existing checks retry missing coverage; they can upgrade an incumbent run if a valid pair becomes readable |
| 15:00 | Configured German gate; Predico's actual gate remains authoritative |
| 15:30 | Existing publish job releases gated forecasts and refreshes the dashboard |

The producer may compute before its declared 13:30 origin: its data must already have been
acquired before that origin and actual input timestamps are strictly before origin minus two
hours. Thus it uses a subset of information allowed at the tested origin. It cannot backdate
late data. If Modal starts the collection too late, inference fails, sources are incomplete,
calibration is unavailable or HF is inaccessible, wind follows the existing T0 submission path.
Separate apps mean producer latency cannot block the existing submission scheduler.

The worker uses one L4 GPU, 2 CPUs, 16 GiB host memory, a 40-minute job timeout and at most one
container. Bootstrap and daily work are serialized through that worker. The new worker adds
one cron definition (two daily invocations), leaving the existing three cron definitions intact.
A GPU-enabled Modal workspace/payment setup is required; check its resource allowance/usage.

## Models and data

- Two independent component forecasts per model, six points per component, 92 physical days
  of context, full next-local-day 15-minute delivery grid (92/96/100 rows).
- Pinned T0-gamma and Timestral-209M weights, and a separately pinned Linux producer environment
  in `requirements/wind-pair.txt`. Model files are baked into the image, not downloaded each day.
  The incumbent image retains its existing dependency versions; SciPy is added only to dev tests.
- Learned onshore weights: last 60 complete prior days, minimum 21, normalized squared error
  plus 0.1 regularization, convex weights with a 70% cap. Offshore uses the two-model mean.
  Cold start is equal-weight. No XGBoost, TimesFM inference or future TSO forecast inputs.
- Point weather uses explicit ICON-EU/D2 06 UTC and ICON/GFS global 00 UTC runs. Point channels,
  speed-band routing and estimated T100 reproduce the research feature layouts. The three
  North Sea offshore predictors are retained intentionally; this is not a Baltic-only redesign.
- Weather banks are the latest approved causal bank no later than each issue. They are
  initialized from the reviewed study and held until a reviewed asset update; **weather-bank
  refitting is not automatic**. Pair-model weights and conformal calibration update daily.
- Truth is existing `data/entsoe.py`: A75 B19/B18 in the 50Hertz control area. The official
  total is their sum. Recent truth revisions and missing collection periods are refetched;
  every snapshot records acquisition time. Truth is never filled by this feature.
- Q50 is the sum of component ensembles, clipped to target capacity. Total Q10/Q90 come from
  its own rolling signed-error conformal bank (60 days, minimum 30 per local slot), not added
  component quantiles. Incomplete/stale calibration rejects the candidate. Bounds are clipped
  to [0, capacity]. Nominal 80% coverage is not a guarantee under time dependence.

The private research seed contains historical simulated member/ensemble forecasts and research weather
banks. Historical publication/revision vintages are unknown, and that is recorded in audit
metadata. Seed actual labels use the official ENTSO-E benchmark; missing dates are fetched
on bootstrap. New historical catch-up predictions are labelled hindcasts, never submissions.
The producer can recover up to seven missed member days automatically; larger gaps require
manual smoke/catch-up. Future live model inputs and calibration labels retain actual acquisition
times, and saved historical forecasts are not recomputed with later data. This rollout does not
claim a fresh holdout win: the selected pair improved period-average research scores versus
its individual members, but those improvements remain statistically uncertain.

## Production asset prerequisite — currently blocked

The supplied seed is **research-only and deliberately rejected by bootstrap/production**.
Its speed-band weather banks were fitted to TimesFM3 predictions. The TimesFM licence
[restrictions](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
cover production use of outputs and data as well as model inference. Removing TimesFM from
inference alone does not establish permission to use those assets. All seeded pair member
forecasts, totals and calibration also depend on those banks.

Before the deployment commands below, either obtain permission covering that downstream use,
or rebuild the banks **and every dependent member/total/calibration forecast** using permitted
sources (for example T0/Timestral power-error routing). Re-evaluate performance and coverage:
existing research scores cannot be attributed to a replacement bank. After reviewing that
artifact, the maintainer must supply `production_approval` in its checksum manifest with
`approved: true` and a nonempty `reference` identifying the grant or reviewed clean-asset
provenance. This is an explicit asset review, not a flag to set on the supplied research ZIP.
The seed builder intentionally writes `approved: false`; no override is provided. Until this
is resolved, retain the incumbent deployment. The recorded numerical validation below tests
the research math and does not clear asset rights or cloud deployment.

## Maintainer steps (after the production asset prerequisite is resolved)

1. Review/merge the PR, or check out its branch for bootstrap and validation first:

   ```sh
   git fetch origin
   git switch codex/50hertz-wind-t0-timestral
   uv sync --frozen
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   ```

2. Use your existing Modal profile and existing `open-predico` secret. Do **not** replace that
   secret: it supplies HF/ENTSO-E and the maintainer-only Predico credentials. Prepare the new
   weather secret from a file **outside the checkout**, using your provider's licensed
   single-runs HTTPS endpoint and key. The empty template is `docs/wind-weather.env.example`.
   The free Open-Meteo endpoint is deliberately rejected for commercial collection.
   Never paste credentials into chat, commit them, or put keys on the command line.

   ```sh
   uv run --with modal==1.6.3 modal secret create open-predico-wind-weather --from-dotenv /secure/wind-weather.env
   ```

   Required keys: `OPEN_METEO_BASE_URL`, `OPEN_METEO_API_KEY`,
   `WIND_PAIR_WEATHER_LICENSED=1`. The existing HF token needs read/write to the existing
   private dataset `open-predico/live-private`. The Timestral weight licence retains its
   attribution and employer monthly-revenue condition; retain the downloaded licence files.

3. Obtain the separately provided **private** `wind-pair-seed.zip` and `.sha256` file from the
   contributor. Do not commit or put them in a public release. Verify the checksum. The seed
   is bound to the exact `configs/live/wind_pair.toml` fingerprint; changing model/data
   parameters requires a new reviewed seed. To reproduce the artifact from the research
   caches, use `tools/build_wind_pair_seed.py` (requires the research environment's pandas/numpy).

   ```sh
   shasum -a 256 /secure/wind-pair-seed.zip
   uv run --with modal==1.6.3 modal run src/openpredico/live/modal_wind_pair.py::bootstrap --seed /secure/wind-pair-seed.zip
   uv run --with modal==1.6.3 modal run src/openpredico/live/modal_wind_pair.py::smoke
   ```

   `bootstrap` installs checksummed state, backs up the seed privately on HF, fetches missing
   actuals/weather and computes missing member forecasts. It does not produce a ready
   candidate or contact Predico. `smoke` runs both real models and conformal calibration on
   the cloud GPU, even with a full cache; it does not overwrite saved live point forecasts.
   Expect `validation_only: true`, 92/96/100 `model_intervals`, at least 30 calibration days,
   and runtime comfortably below the submission margin. If it fails or exceeds ten minutes
   on an already caught-up state, investigate before deploying. Late-day smoke is validation,
   not evidence of an on-time market submission. Keep the returned status/duration as evidence.

4. After successful cloud smoke validation, deploy **both** apps from this checkout:

   ```sh
   uv run --with modal==1.6.3 modal deploy src/openpredico/live/modal_wind_pair.py
   uv run --with modal==1.6.3 modal deploy src/openpredico/live/modal_app.py
   ```

   The first command adds the new producer schedule. The second updates the existing app
   with the guarded consumer and retry upgrade. Deploying only the producer will not change
   the submitted forecast. Nothing in this PR deploys or merges itself.

5. Observe the next scheduled run. Verify producer status `ready`, then the 14:15 wind
   submission/coverage record and canonical HF archive model `wind_pair_t0_timestral_six_v1`.
   Inspect metadata/status, not forecast values in public logs. At 15:30 the existing publisher
   should show that model on the dashboard. If initialization/deployment finishes after the
   collection cutoff, the pair first becomes eligible on the next scheduled day; the incumbent
   remains active in the meantime. Do not claim success from a scheduled app's existence alone.

   ```sh
   uv run --with modal==1.6.3 modal app logs open-predico-wind-pair
   uv run --with modal==1.6.3 modal app logs open-predico
   ```

## Storage and integrity

Modal volume `open-predico-wind-pair-state` keeps decoded weather, approved banks, member points,
ensemble points, latest actual/acquisition records and immutable pending bundles. Successful
jobs retain 120 days (enough for context and bounded catch-up). HF holds the private seed backup
at `wind-pair/v1/bootstrap/<sha>.zip` and atomic candidate bundles at
`wind-pair/v1/candidates/<delivery-day>.zip`. Each bundle contains exact engineered model inputs,
raw newly fetched weather records, member fitting inputs, conformal inputs and hashes/audit.
The namespace is outside public-publisher and retry-run discovery; it is not a submission.
The existing submit job saves the selected forecast and exact inputs at the standard private
`forecasts/<target>/<run>.parquet` and `inputs/<target>/<run>.parquet` paths. Publishing remains
gate-protected. A private-storage check fails closed rather than falling back to public storage.

## Rollback

Set `enabled = false` in `configs/live/wind_pair.toml`, then redeploy **both** apps. The consumer
immediately returns to the incumbent path and the producer exits without collecting/inference.
If a pair was already submitted today and the German session is still open, explicitly replace
it with the incumbent for **wind only**:

```sh
uv run --with modal==1.6.3 modal deploy src/openpredico/live/modal_wind_pair.py
uv run --with modal==1.6.3 modal deploy src/openpredico/live/modal_app.py
uv run --with modal==1.6.3 modal run src/openpredico/live/modal_wind_pair.py::restore_incumbent
```

The rollback entrypoint refuses to run if the flag is still enabled or the wind session is
closed. Existing gated forecasts cannot be changed retroactively. The new app may then be
stopped in Modal; keep its volume/HF records for audit. Re-enabling the original config restores
its original seed fingerprint. Do not overwrite a different installed seed or reset live state
as a routine retry.

## Verification performed before PR

- Existing and new offline tests, lint and formatting in a clean lockfile-based environment.
- Real cached T0/Timestral recorded-input producer replay for 5 November: 4 weather models,
  six points per component, 96 intervals, 60 calibration days; point difference from reviewed
  research below 0.003 MW. Simulated acquisition times, no source/API writes or submissions.
- Actual Modal definition imported locally using Modal 1.6.3, without deployment credentials.

Cloud image build, live source access, L4 runtime, actual scheduled execution and Predico
coverage **must be verified by the maintainer**. They were not run from the contributor's
workspace, which has no Modal profile. No production deployment is claimed by this PR.
