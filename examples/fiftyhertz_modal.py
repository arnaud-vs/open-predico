"""Independent collection-only Modal app. Deploying does not alter the Spain app.

From repo root after Modal authentication:
  modal run examples/fiftyhertz_modal.py::collect_daily
  modal deploy examples/fiftyhertz_modal.py
"""

from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas>=2.2", "pyarrow>=17", "requests>=2.32")
    .add_local_python_source("openpredico")
)
app = modal.App("xiaoyujin97-fiftyhertz-inputs", image=image)
volume = modal.Volume.from_name("xiaoyujin97-fiftyhertz-inputs", create_if_missing=True)


@app.function(
    schedule=modal.Cron("30 13 * * *", timezone="Europe/Berlin"),
    volumes={"/inputs": volume},
    timeout=600,
    retries=2,
)
def collect_daily() -> None:
    from openpredico.data.fiftyhertz_collect import collect

    metadata = collect(Path("/inputs"))
    volume.commit()
    print(metadata)  # timestamps, coverage and freshness only; no generation/forecast values
