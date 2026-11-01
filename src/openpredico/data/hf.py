"""Hugging Face storage (org `open-predico`). Token from HF_TOKEN or the local HF login."""

import io

import pandas as pd
from huggingface_hub import HfApi

ORG = "open-predico"


def upload(df: pd.DataFrame, repo: str, path: str, message: str) -> None:
    """Write `df` as Parquet to `path` in dataset `open-predico/<repo>` (one commit)."""
    buf = io.BytesIO()
    df.to_parquet(buf)
    HfApi().upload_file(
        path_or_fileobj=buf.getvalue(),
        path_in_repo=path,
        repo_id=f"{ORG}/{repo}",
        repo_type="dataset",
        commit_message=message,
    )


def download(repo: str, path: str) -> pd.DataFrame:
    return pd.read_parquet(HfApi().hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset"))
