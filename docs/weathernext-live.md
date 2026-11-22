# WeatherNext 3 on Modal: keyless setup

Modal containers get a short-lived OIDC identity token (`MODAL_IDENTITY_TOKEN`, issuer
`https://oidc.modal.com`, `aud` = `oidc.modal.com`, claims `workspace_id`, `environment_name`,
`app_name`, `function_name`, ...). Google's Workload Identity Federation exchanges it for a token of
`open-predico-wn@open-predico.iam.gserviceaccount.com` (`weathernext.credentials`, config `WIF`).
No key exists anywhere. On a laptop, `weathernext` uses Application Default Credentials as before.

## 1. Find the Modal workspace id

The condition pins our workspace by its immutable id (`ac-...`), not its name (`arnaud-vs`, which
can be renamed and is not in the token): `uv run --with modal modal token info` (Workspace ID) or
https://modal.com/settings/workspaces. The environment is `main` unless `modal environment list`
says otherwise.

## 2. GCP (run once, as project owner)

```bash
PROJECT=open-predico
NUMBER=210233305810   # gcloud projects describe open-predico --format='value(projectNumber)'
SA=open-predico-wn@open-predico.iam.gserviceaccount.com
WORKSPACE_ID=ac-REPLACE_ME   # step 1

gcloud services enable iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  --project=$PROJECT

gcloud iam workload-identity-pools create modal --project=$PROJECT --location=global \
  --display-name="Modal"

gcloud iam workload-identity-pools providers create-oidc modal --project=$PROJECT \
  --location=global --workload-identity-pool=modal \
  --issuer-uri="https://oidc.modal.com" \
  --allowed-audiences="oidc.modal.com" \
  --attribute-mapping="google.subject=assertion.workspace_id + ':' + assertion.app_name + ':' + assertion.function_name,attribute.workspace_id=assertion.workspace_id,attribute.app_name=assertion.app_name" \
  --attribute-condition="assertion.workspace_id == '$WORKSPACE_ID' && assertion.environment_name == 'main' && assertion.app_name == 'open-predico'"

gcloud iam service-accounts add-iam-policy-binding $SA --project=$PROJECT \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/modal/attribute.app_name/open-predico"
```

`google.subject` is not Modal's `sub` (that one carries the container id and can exceed Google's
127-byte limit). Only the `open-predico` app of our workspace's `main` environment can
impersonate the service account; the service account keeps only `bigquery.jobUser` and
`bigquery.dataViewer`.

**WeatherNext access is granted per email:** if the linked dataset was shared with your user
only, request access for the service account email too, or the query fails with 403 Access
Denied on the table.

## 3. Verify (no submission, no archive write)

Locally add `"weather_be_wn3"` to `belgium_solar_opendata_elia.groups` in `configs/targets.toml`
(step 4), then:

```bash
uv run --with modal modal run src/openpredico/live/modal_app.py::dry_run --only belgium_solar_opendata_elia
```

`modal run` uses the local files and app name `open-predico`. Expect `weathernext <run> UTC: N MB
billed` lines and no `weather_be_wn3.*: left out` line. An STS `invalid_grant` / attribute
condition error means a wrong workspace id or environment; `iam.serviceAccounts.getAccessToken`
denied means the binding is missing. The dry run also reads the archive from the private HF
dataset `weather-private`: the Modal `HF_TOKEN` needs read and write access to it.

## 4. Switch on (one line, then deploy)

```toml
groups = ["target_elia", "weather_be", "weather_be_d2", "weather_be_knmi", "weather_be_aifs", "weather_be_wn3", "capacity_be", "calendar"]
```

Then commit and `modal deploy src/openpredico/live/modal_app.py`. The archive
`weather-private/archive/weather_be_wn3.parquet` is seeded from the v6 backfill on the first run.

If BigQuery fails live (auth, quota, run late), the lenient build leaves the WN3 columns out,
`submit.forecast` drops them as missing on D+1 and the run submits with the other four weather
models (test `test_wn3_archive_is_private_and_a_bigquery_failure_drops_it`). Before ~06:45 UTC
the D+1 run is not out: intraday runs then go without WN3 (each tries one query, ~20 s).

Rollback: remove the group from `targets.toml` and redeploy. To revoke access:
`gcloud iam workload-identity-pools providers disable modal --workload-identity-pool=modal --location=global --project=open-predico`.
