"""Isolated producer app. Bootstrap, smoke, then deploy; existing submit/publish crons stay.

See docs/wind-pair-operations.md. Secrets: open-predico and open-predico-wind-weather.
"""

from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements/wind-pair.txt")
    .env({"HF_HOME": "/models", "HF_HUB_CACHE": "/models/hub"})
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download; '
        "snapshot_download('theforecastingcompany/t0-gamma', "
        "revision='e08aee083778f1f32da4974c0a34276776a35c52'); "
        "snapshot_download('mistralai/Timestral-209M', "
        "revision='9874d252ab0aae16f7f7c999f98f98d6ad7d8365')\""
    )
    .add_local_python_source("openpredico")
    .add_local_dir("configs", "/configs")
)
app = modal.App("open-predico-wind-pair", image=image)
volume = modal.Volume.from_name("open-predico-wind-pair-state", create_if_missing=True)


@app.function(
    schedule=modal.Cron("0,15 13 * * *", timezone="Europe/Berlin"),
    gpu="L4",
    cpu=2.0,
    memory=16384,
    timeout=2400,
    max_containers=1,
    volumes={"/state": volume},
    secrets=[
        modal.Secret.from_name("open-predico"),
        modal.Secret.from_name("open-predico-wind-weather"),
    ],
)
def worker(mode: str = "produce"):
    """One serialized worker prevents bootstrap/producer state races."""
    from openpredico.live import wind_pair
    from openpredico.live.wind_pair_state import install

    root = Path("/state/data")
    try:
        if mode == "bootstrap":
            _, sha = wind_pair.config()
            seed = Path("/state/upload/seed.zip").read_bytes()
            install(seed, root, sha)
            from openpredico.live.wind_pair_bundle import backup_seed

            backup_seed(seed)
            result = wind_pair.initialize(root)
        elif mode == "smoke":
            result = wind_pair.smoke(root)
        elif mode == "incumbent":
            cfg, _ = wind_pair.config()
            if cfg["enabled"]:
                raise ValueError("disable the pair before an incumbent-only rollback")
            import pandas as pd

            from openpredico.live import submit

            if "germany_wind_50hertz_ca" not in submit.in_session(pd.Timestamp.now(tz="UTC")):
                raise ValueError("wind session is not open")
            submit.main(dry_run=False, only=["germany_wind_50hertz_ca"])
            result = {"status": "incumbent_submitted"}
        elif mode == "produce":
            result = wind_pair.produce(root)
        else:
            raise ValueError("unknown wind-pair job")
        from openpredico.live.wind_pair_state import compact

        if mode != "incumbent":
            compact(root)
        print(f"wind-pair {mode}: {result}")  # status/counts only, never forecast values
        return result
    except Exception as error:
        raise RuntimeError(
            f"wind-pair job failed ({type(error).__name__}); incumbent retained"
        ) from None
    finally:
        if mode != "incumbent":
            volume.commit()  # source caches and pending bundles survive transient HF errors


@app.local_entrypoint()
def bootstrap(seed: str):
    """Install the private seed and catch up. Does not create a ready forecast or submit."""
    with volume.batch_upload(force=True) as upload:
        upload.put_file(Path(seed), "/upload/seed.zip")
    print(worker.remote(mode="bootstrap"))


@app.local_entrypoint()
def smoke():
    """Operator initialization/catch-up check, no candidate publication or Predico access."""
    print(worker.remote(mode="smoke"))


@app.local_entrypoint()
def restore_incumbent():
    """Wind-only rollback during its open session, after deploying enabled=false to both apps."""
    print(worker.remote(mode="incumbent"))
