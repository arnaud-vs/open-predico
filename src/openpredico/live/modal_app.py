"""Daily Predico submission on Modal. Logic lives in `submit.py`; this file only says where/when.

Deploy:  uv run --with modal modal deploy src/openpredico/live/modal_app.py
Once:    uv run --with modal modal run src/openpredico/live/modal_app.py::submit  (or ::publish)
Secrets: Modal secret `open-predico` with ESIOS_TOKEN, HF_TOKEN, PREDICO_EMAIL, PREDICO_PASSWORD
         PREDICO_PLAYGROUND_URL and PREDICO_URL (live/predico.py URL_VAR picks one). See PLAN.md.
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
# Gate 17:00 Madrid (Predico: "17:00 CET"); session opens the hour before.
# ponytail: assumes the gate follows Madrid local time in summer; check `gate_closure` in CEST.
TZ = "Europe/Madrid"


@app.function(schedule=modal.Cron("15 16 * * *", timezone=TZ), cpu=1.0, memory=4096, timeout=1800)
def submit() -> None:
    from openpredico.live import submit

    submit.main(dry_run=False)


@app.function(schedule=modal.Cron("30,45 16 * * *", timezone=TZ), cpu=1.0, memory=4096, timeout=900)
def check() -> None:
    """Resubmit only if an open challenge of ours isn't covered (the 16:15 run failed)."""
    from openpredico.live import submit

    submit.check_and_retry()


@app.function(schedule=modal.Cron("30 17 * * *", timezone=TZ), cpu=1.0, memory=2048, timeout=1800)
def publish() -> None:
    """After the gate: publish what Predico scored, then rebuild and push the dashboard Space."""
    import subprocess
    import sys

    from openpredico.live import publish

    try:
        publish.main()
    finally:  # the dashboard refreshes even if publishing failed (it shows what is there)
        subprocess.run([sys.executable, "/dashboard/build.py", "--upload"], check=True)
