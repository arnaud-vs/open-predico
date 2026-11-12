# Automatic public wind-pair snapshots

The immutable seed at `assets/wind-pair/wind-pair-clean-seed.zip` remains the initial
bootstrap fallback. Fresh historical state is published as a **daily GitHub Release asset**,
with a small latest-download manifest committed on the public **`codex/wind-pair-snapshots`**
branch. Large ZIPs are not added to daily Git history. This is historical state publication,
not weather-weight retraining and not a change to Predico submission timing.

## Public locations

After the first successful run:

- [Latest manifest](https://github.com/arnaud-vs/open-predico/blob/codex/wind-pair-snapshots/assets/wind-pair/latest.json)
- [Raw latest manifest](https://raw.githubusercontent.com/arnaud-vs/open-predico/refs/heads/codex/wind-pair-snapshots/assets/wind-pair/latest.json)
- Dated release tag: `wind-pair-bootstrap-YYYY-MM-DD`.
- Each release contains `wind-pair-clean-seed.zip`, its `.sha256` file and `manifest.json`.

The manifest includes the public `archive_url`, checksum, configuration fingerprint,
historical cutoff, last fully observed truth day and latest weather-bank applicability date. The pointer branch contains metadata
only and is not a deployment branch. Until activation/first success, use the committed seed.
The GitHub Actions workflow is named **public wind snapshot**.

## Daily flow

1. GitHub Actions runs at **18:30 UTC daily** (19:30 Brussels in winter, 20:30 in summer),
   after the forecasting window. It can also be dispatched manually. GitHub may delay scheduled
   jobs; forecast submission does not depend on this publication schedule.
2. Actions invokes the already deployed Modal `open-predico-wind-pair` / `worker` with
   `mode="export"`. This uses the same serialized worker as collection/inference so it does
   not read a half-written multi-file state. Export does not collect weather, refit models,
   submit forecasts, or alter saved point forecasts.
3. The exporter constructs a fresh archive from an allowlist: actual generation, numeric
   decoded weather, historical member/total points and approved speed banks. It strips extra
   columns and dataframe metadata. It never copies the state directory wholesale.
4. Every time-series data row is **before today's midnight in Europe/Berlin**.
   Thus current/future delivery-day forecasts, pending/ready candidates and credentials are excluded.
   Approved speed-bank parameters are included even when labelled for a later applicability
   date, just like the initial seed: these small convex matrices are not forecast trajectories.
   The archive contains up to 120 physical days of input history and 60 complete historical
   forecast days. Missing observations are preserved; at least 30 fully observed forecast
   days and a fully observed day no older than the day before yesterday are required.
5. Actions downloads only the fixed `exports/wind-pair/YYYY-MM-DD/` files from the Modal
   Volume. It independently checks the archive schema, dates, model/configuration metadata,
   checksums and latest historical forecast/weather day. The deployed configuration must
   match the checkout's fingerprint.
6. Assets are uploaded to a draft dated release and downloaded back for verification before
   publication. Existing assets with a different checksum are never overwritten. Successful
   release publication precedes the latest-pointer commit. Older workflows cannot rewind it.

A failed export, download, validation or upload leaves the last successful public pointer
in place. Re-running the workflow on the same date is idempotent: the export is cached and
that day's published asset is immutable. It does not replace an installed Modal seed or
reset the live Volume. The dated export cache is separate from runtime candidate directories.

## One-time maintainer activation

The contributor prepared and locally tested this path but has no Modal deployment credentials.
A maintainer must perform these steps after review/merge:

1. Redeploy **the producer app only** from current main. This adds `mode="export"`; it preserves
   the colleague's current CPU/image/collector settings and existing producer cron.

   ```sh
   uv run --with modal==1.6.3 modal deploy src/openpredico/live/modal_wind_pair.py
   ```

2. Add repository Actions secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for a dedicated
   Modal token able to invoke this app and read its Volume. Do not paste credentials into
   chat, commit them, or put them in command arguments. Use GitHub Settings → Secrets and
   variables → Actions, or files outside the checkout:

   ```sh
   gh secret set MODAL_TOKEN_ID --repo arnaud-vs/open-predico < /secure/modal-token-id
   gh secret set MODAL_TOKEN_SECRET --repo arnaud-vs/open-predico < /secure/modal-token-secret
   ```

   If the app is deployed in a non-default Modal environment, set the repository Actions
   variable `MODAL_ENVIRONMENT` to that environment's name. No HF or Predico credentials are
   needed in Actions. Its ordinary `GITHUB_TOKEN` receives `contents: write` from the workflow.
   Repository rules must permit that bot to create the named metadata branch and release tags.

3. Ensure bootstrap/collection has populated a recent complete forecast/weather day and
   sufficient actuals. Dispatch and verify the first publication:

   ```sh
   gh workflow run wind-snapshot.yml --ref main --repo arnaud-vs/open-predico
   gh run list --workflow wind-snapshot.yml --repo arnaud-vs/open-predico --limit 3
   ```

   Confirm the run succeeded, the dated release has all three assets, the latest manifest
   points to it, the checksum matches, and the cutoff excludes today. Actual scheduled
   execution and Modal/Volume permissions remain cloud checks; local tests cannot prove them.

## Using a refreshed seed

Open the latest public manifest, then download its `archive_url` and verify `sha256` before
using the existing bootstrap command. Choose a fresh output directory outside the checkout.
For example, after reading `release_tag` from that manifest:

```sh
gh release download wind-pair-bootstrap-YYYY-MM-DD --repo arnaud-vs/open-predico --dir /secure/wind-snapshot
(cd /secure/wind-snapshot && shasum -a 256 -c wind-pair-clean-seed.zip.sha256)
uv run --with modal==1.6.3 modal run src/openpredico/live/modal_wind_pair.py::bootstrap --seed /secure/wind-snapshot/wind-pair-clean-seed.zip
```

Bootstrap is for initializing a new state Volume. The existing install guard refuses to
replace a different already installed seed. Existing production keeps collecting/updating
its own state; it does not reinstall each public snapshot. A changed configuration/model
fingerprint requires review and a matching bootstrap artifact, not bypassing that guard.

## Sources and limits

Weather data by [Open-Meteo](https://open-meteo.com/), using DWD ICON and NOAA GFS;
[data licence and attribution](https://open-meteo.com/en/licence),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Weather transformations include
100 m height conversion, sine/cosine directions, quarter-hour interpolation and estimated
T100 from ICON-EU temperatures at 80/120 m. Generation is from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), 50Hertz A75 B18/B19.
Source API/model terms continue to apply. No pretrained neural checkpoints are redistributed.

Snapshots preserve historical hindcasts and past live model outputs, including their kind
labels. They do not establish official contest submission records. Initial seed publication/
revision vintages remain unverified, as stated in the manifest. Raw-source revisions and
late measurements can change later snapshots; published dated ZIPs remain unchanged.
Daily snapshot publication does **not** implement daily weather-bank refitting: it publishes
the approved parameter banks the pipeline currently has; their applicability dates are explicit.

To stop automatic publication, disable **public wind snapshot** in Actions. Existing releases
and the latest pointer stay available, and forecasting/submission continue independently.
