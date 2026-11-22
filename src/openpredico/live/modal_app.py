"""Daily Predico submission on Modal. Logic lives in `submit.py`; this file only says where/when.

Deploy:  uv run --with modal modal deploy src/openpredico/live/modal_app.py
Once:    uv run --with modal modal run src/openpredico/live/modal_app.py::tick
         (or ::publish); tick acts only on targets whose session is open
Secrets: Modal secret `open-predico` with ESIOS_TOKEN, ENTSOE_API_KEY, HF_TOKEN, PREDICO_EMAIL,
         PREDICO_PASSWORD, PREDICO_PLAYGROUND_URL and PREDICO_URL (live/predico.py URL_VAR picks
         one). See PLAN.md.
"""

import modal

from openpredico.live.submit import MODEL

image = (
    modal.Image.debian_slim(python_version="3.12")
    # weathernext: BigQuery client (~60 MB next to torch's ~600 MB), imported only when a group
    # reads WeatherNext; auth is keyless (weathernext.credentials, PLAN.md)
    .uv_sync(extras=["t0", "weathernext"])
    .run_commands(
        f"python -c \"import huggingface_hub; huggingface_hub.snapshot_download('{MODEL}')\""
    )
    .add_local_python_source("openpredico")
    .add_local_dir("configs", "/configs")  # config.CONFIGS resolves to /configs in the container
    .add_local_dir("dashboard", "/dashboard", ignore=["data", "__pycache__"])
)
app = modal.App("open-predico", image=image, secrets=[modal.Secret.from_name("open-predico")])
# Each target's D+1 session opens the hour before its gate (Belgium 10:00, Germany 15:00, Spain
# 17:00, Predico: "CET"); Belgium also has hourly intraday challenges. One `tick` cron acts on
# whatever is open (Starter plan: 5 crons; this app uses 2, the wind-pair app 1). Madrid, Berlin
# and Brussels share UTC offsets.
# ponytail: assumes the gates follow local time in summer; check `gate_closure` in CEST.
TZ = "Europe/Madrid"


@app.function(
    schedule=modal.Cron("15,30,45 * * * *", timezone=TZ),
    cpu=1.0,
    memory=4096,
    timeout=1800,
    max_containers=1,  # one tick at a time: an overlapping tick queues, never submits twice
)
def tick() -> None:
    """Submit the targets in session without a run yet (:15), retry what failed (:30, :45).

    Same image as the forecast: a tick with nothing to do imports no torch and loads no model
    (seconds), so a separate light image would save little.
    """
    from openpredico.live import submit

    submit.check_and_retry()


@app.function(
    # 15:30 and 17:30 follow the gates; 09:30 and 21:30 pick up Predico scores that land later
    schedule=modal.Cron("30 9,15,17,21 * * *", timezone=TZ),
    cpu=1.0,
    memory=2048,
    timeout=1800,
)
def publish() -> None:
    """Half an hour after each gate: publish what Predico scored, then rebuild the dashboard."""
    import subprocess
    import sys

    from openpredico.live import publish

    try:
        publish.main()
    finally:  # the dashboard refreshes even if publishing failed (it shows what is there)
        subprocess.run([sys.executable, "/dashboard/build.py", "--upload"], check=True)


@app.function(cpu=1.0, memory=4096, timeout=1800)
def dry_run(only: str) -> None:
    """Forecast `only` (comma-separated targets) without submitting or uploading (no Predico
    login): `modal run src/openpredico/live/modal_app.py::dry_run --only a,b`."""
    from openpredico.live import submit

    submit.main(dry_run=True, only=only.split(","))
