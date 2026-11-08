"""Daily Predico submission on Modal. Logic lives in `submit.py`; this file only says where/when.

Deploy:  uv run --with modal modal deploy src/openpredico/live/modal_app.py
Once:    uv run --with modal modal run src/openpredico/live/modal_app.py::submit
         (or ::check, ::publish); submit acts only on targets whose session is open
Secrets: Modal secret `open-predico` with ESIOS_TOKEN, ENTSOE_API_KEY, HF_TOKEN, PREDICO_EMAIL,
         PREDICO_PASSWORD, PREDICO_PLAYGROUND_URL and PREDICO_URL (live/predico.py URL_VAR picks
         one). See PLAN.md.
"""

import modal

from openpredico.live.submit import MODEL

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["t0"])
    .run_commands(
        f"python -c \"import huggingface_hub; huggingface_hub.snapshot_download('{MODEL}')\""
    )
    .add_local_python_source("openpredico")
    .add_local_dir("configs", "/configs")  # config.CONFIGS resolves to /configs in the container
    .add_local_dir("dashboard", "/dashboard", ignore=["data", "__pycache__"])
)
app = modal.App("open-predico", image=image, secrets=[modal.Secret.from_name("open-predico")])
# Each target's D+1 session opens the hour before its gate (Spain 17:00, Germany 15:00, Predico:
# "CET"). Every job runs in both session hours and acts on what is open, so all targets share
# three crons (Starter plan: 5). Madrid and Berlin share their UTC offsets.
# ponytail: assumes the gates follow local time in summer; check `gate_closure` in CEST.
TZ = "Europe/Madrid"


@app.function(
    schedule=modal.Cron("15 14,16 * * *", timezone=TZ), cpu=1.0, memory=4096, timeout=1800
)
def submit() -> None:
    """Forecast and submit the targets whose session is open (Germany 14:15, Spain 16:15)."""
    import pandas as pd

    from openpredico.live import submit

    if only := submit.in_session(pd.Timestamp.now(tz="UTC")):
        submit.main(dry_run=False, only=only)
    else:
        print("no session of ours open")


@app.function(
    schedule=modal.Cron("30,45 14,16 * * *", timezone=TZ), cpu=1.0, memory=4096, timeout=900
)
def check() -> None:
    """Resubmit the targets whose open challenge isn't covered (their submit run failed)."""
    from openpredico.live import submit

    submit.check_and_retry()


@app.function(
    schedule=modal.Cron("30 15,17 * * *", timezone=TZ), cpu=1.0, memory=2048, timeout=1800
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
