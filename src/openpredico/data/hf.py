"""Hugging Face storage (org `open-predico`). Token from HF_TOKEN or the local HF login."""

import io

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi

ORG = "open-predico"


def upload(df: pd.DataFrame, repo: str, path: str, message: str) -> None:
    """Write `df` as Parquet to `path` in dataset `open-predico/<repo>` (one commit)."""
    upload_many({path: df}, repo, message)


def upload_many(files: dict[str, pd.DataFrame], repo: str, message: str) -> None:
    """Write each frame as Parquet to its path in dataset `open-predico/<repo>`, in one commit."""
    ops = []
    for path, df in files.items():
        buf = io.BytesIO()
        df.to_parquet(buf)
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=buf.getvalue()))
    HfApi().create_commit(
        repo_id=f"{ORG}/{repo}", operations=ops, commit_message=message, repo_type="dataset"
    )


def download(repo: str, path: str) -> pd.DataFrame:
    return pd.read_parquet(HfApi().hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset"))
