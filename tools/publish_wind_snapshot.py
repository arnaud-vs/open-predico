"""Actions-only daily release publication. No access to Predico; prints metadata only."""

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from openpredico.live import wind_pair_snapshot as snapshot
from openpredico.live.wind_pair_state import digest

BRANCH = "codex/wind-pair-snapshots"
POINTER = "assets/wind-pair/latest.json"
BOT = {
    "name": "github-actions[bot]",
    "email": "41898282+github-actions[bot]@users.noreply.github.com",
}


def gh(*args, data=None, missing_ok=False):
    command = ["gh", *args]
    if data is not None:
        if "--method" not in args:
            command += ["--method", "POST"]
        command += ["--input", "-"]
    result = subprocess.run(
        command,
        input=json.dumps(data) if data is not None else None,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        if missing_ok and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(
            "GitHub operation failed; check workflow contents permissions and run logs"
        )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def release_metadata(repo, tag):
    release = gh("api", f"repos/{repo}/releases/tags/{tag}", missing_ok=True)
    if release is not None:
        return release
    # A draft need not have a published git tag yet; resumable uploads still find it.
    recent = gh("api", f"repos/{repo}/releases?per_page=100")
    return next((item for item in recent if item["tag_name"] == tag), None)


def published_archive_matches(repo, tag, public):
    with TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                "gh",
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--dir",
                tmp,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError("Could not verify existing release archive")
        directory = Path(tmp)
        try:
            data = (directory / snapshot.ARCHIVE).read_bytes()
            metadata = json.loads((directory / "manifest.json").read_text())
            sidecar = (directory / f"{snapshot.ARCHIVE}.sha256").read_text()
        except (OSError, ValueError):
            return False
        return (
            digest(data) == public["sha256"]
            and metadata == public
            and sidecar == f"{public['sha256']}  {snapshot.ARCHIVE}\n"
        )


def pointer_update(repo, pointer):
    content = json.dumps(pointer, indent=2) + "\n"
    message = f"Update public wind snapshot {pointer['snapshot_date']}\n\n" + (
        "Signed-off-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"
    )
    ref = gh("api", f"repos/{repo}/git/ref/heads/{BRANCH}", missing_ok=True)
    if ref is None:
        tree = gh(
            "api",
            f"repos/{repo}/git/trees",
            data={
                "tree": [
                    {
                        "path": POINTER,
                        "mode": "100644",
                        "type": "blob",
                        "content": content,
                    }
                ]
            },
        )
        commit = gh(
            "api",
            f"repos/{repo}/git/commits",
            data={
                "message": message,
                "tree": tree["sha"],
                "parents": [],
                "author": BOT,
                "committer": BOT,
            },
        )
        gh(
            "api",
            f"repos/{repo}/git/refs",
            data={
                "ref": f"refs/heads/{BRANCH}",
                "sha": commit["sha"],
            },
        )
        return
    current = gh("api", f"repos/{repo}/contents/{POINTER}?ref={BRANCH}")
    previous = json.loads(base64.b64decode(current["content"]))
    if previous["snapshot_date"] > pointer["snapshot_date"]:
        raise RuntimeError("Refusing to move the public latest pointer backwards")
    if previous["snapshot_date"] == pointer["snapshot_date"]:
        if previous["sha256"] != pointer["sha256"]:
            raise RuntimeError("Daily public snapshot is immutable; checksum changed")
        return
    gh(
        "api",
        f"repos/{repo}/contents/{POINTER}",
        "--method",
        "PUT",
        data={
            "branch": BRANCH,
            "message": message,
            "sha": current["sha"],
            "content": base64.b64encode(content.encode()).decode(),
        },
    )


def publish(folder, public, repo, revision):
    tag = f"wind-pair-bootstrap-{public['snapshot_date']}"
    print(f"Verifying release assets for {tag}")
    existing = release_metadata(repo, tag)
    if existing and not existing["draft"]:
        if not published_archive_matches(repo, tag, public):
            raise RuntimeError("Existing published release has a different checksum")
    else:
        if existing is None:
            notes = folder / "release-notes.md"
            notes.write_text(
                f"Historical wind-pair bootstrap through {public['historical_through']}.\n\n"
                f"SHA256: `{public['sha256']}`. Configuration: `{public['config_sha256']}`.\n\n"
                "Past-only model/input rows plus approved routing parameters; no current/future "
                "delivery-day forecast rows, secrets or "
                "pretrained checkpoints. Historical vintages remain unverified. "
                f"See [snapshot operations](https://github.com/{repo}/blob/{revision}/"
                "docs/wind-pair-snapshots.md). "
                "Weather data by [Open-Meteo](https://open-meteo.com/) "
                "([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), using DWD ICON "
                "and NOAA GFS. Generation: [ENTSO-E](https://transparency.entsoe.eu/). "
                "Transformation/source attribution is retained in the archive manifest.\n"
            )
            command = [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                repo,
                "--draft",
                "--target",
                revision,
                "--title",
                f"Historical wind bootstrap {public['snapshot_date']}",
                "--notes-file",
                str(notes),
            ]
            if subprocess.run(command, capture_output=True).returncode:
                raise RuntimeError("Could not create draft historical release")
        present = {asset["name"] for asset in (release_metadata(repo, tag) or {}).get("assets", [])}
        for name in [snapshot.ARCHIVE, f"{snapshot.ARCHIVE}.sha256", "manifest.json"]:
            if name in present:
                with TemporaryDirectory() as temporary:
                    command = [
                        "gh",
                        "release",
                        "download",
                        tag,
                        "--repo",
                        repo,
                        "--pattern",
                        name,
                        "--dir",
                        temporary,
                    ]
                    if subprocess.run(command, capture_output=True).returncode:
                        raise RuntimeError("Could not verify existing draft asset")
                    if digest((Path(temporary) / name).read_bytes()) != digest(
                        (folder / name).read_bytes()
                    ):
                        raise RuntimeError("Existing draft asset has a different checksum")
                continue
            command = ["gh", "release", "upload", tag, "--repo", repo, str(folder / name)]
            if subprocess.run(command, capture_output=True).returncode:
                raise RuntimeError("Could not upload draft release asset")
        if not published_archive_matches(repo, tag, public):
            raise RuntimeError("Uploaded release checksum mismatch")
        if subprocess.run(
            ["gh", "release", "edit", tag, "--repo", repo, "--draft=false", "--latest=false"],
            capture_output=True,
        ).returncode:
            raise RuntimeError("Could not publish verified historical release")
    pointer = {
        **public,
        "release_tag": tag,
        "release_url": f"https://github.com/{repo}/releases/tag/{tag}",
        "archive_url": f"https://github.com/{repo}/releases/download/{tag}/{snapshot.ARCHIVE}",
    }
    pointer_update(repo, pointer)
    print(f"Published historical snapshot {public['snapshot_date']}; SHA256 {public['sha256']}")


async def download(volume, prefix, folder):
    for name in [snapshot.ARCHIVE, "manifest.json", f"{snapshot.ARCHIVE}.sha256"]:
        with (folder / name).open("wb") as target:
            size = 0
            async for chunk in volume.read_file.aio(f"{prefix}/{name}"):
                size += len(chunk)
                limit = snapshot.MAX_BYTES if name == snapshot.ARCHIVE else 1024 * 1024
                if size > limit:
                    raise RuntimeError("Snapshot download exceeds size limit")
                target.write(chunk)


def main(folder):
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise RuntimeError("Publication runs only in the main-branch Actions workflow")
    repo, revision = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"]
    if repo != "arnaud-vs/open-predico" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Unexpected publication repository or revision")
    import modal

    environment = os.getenv("MODAL_ENVIRONMENT") or None
    worker = modal.Function.from_name(
        "open-predico-wind-pair", "worker", environment_name=environment
    )
    print("Requesting a past-only Modal snapshot")
    result = worker.remote(mode="export")
    if result.get("status") != "exported":
        reason = result.get("reason", "unavailable")
        if re.fullmatch(r"[A-Za-z_]{1,64}", reason):
            print(f"Snapshot unavailable: {reason}")
        raise RuntimeError(
            "Historical export unavailable; check pipeline history and exporter deployment"
        )
    day = result["snapshot_date"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise RuntimeError("Unexpected snapshot date")
    prefix = f"exports/wind-pair/{day}"
    if result.get("volume_prefix") != prefix:
        raise RuntimeError("Refusing to download non-export runtime paths")
    volume = modal.Volume.from_name("open-predico-wind-pair-state", environment_name=environment)
    folder.mkdir(parents=True, exist_ok=True)
    asyncio.run(download(volume, prefix, folder))
    data = (folder / snapshot.ARCHIVE).read_bytes()
    if digest(data) != result["sha256"] or len(data) != result["bytes"]:
        raise RuntimeError("Downloaded archive does not match the export receipt")
    print(f"Downloaded snapshot {day}; checking public schema and cutoff")
    expected_config = digest(Path("configs/live/wind_pair.toml").read_bytes())
    manifest = snapshot.inspect(
        data, config_hash=expected_config, now=pd.Timestamp(datetime.now(UTC))
    )
    public = json.loads((folder / "manifest.json").read_text())
    keys = {
        "schema_version",
        "snapshot_date",
        "historical_through",
        "last_complete_truth_day",
        "last_weather_bank_day",
        "cutoff_utc",
        "created_at_utc",
        "config_sha256",
        "pipeline",
        "sha256",
        "bytes",
    }
    if set(public) != keys:
        raise RuntimeError("Unexpected public metadata fields")
    expected = {k: result[k] for k in public}
    if public != expected or digest(data) != public["sha256"] or len(data) != public["bytes"]:
        raise RuntimeError("Export/download metadata mismatch")
    if public["historical_through"] != manifest["last_member_day"]:
        raise RuntimeError("Historical cutoff mismatch")
    if (
        folder / f"{snapshot.ARCHIVE}.sha256"
    ).read_text() != f"{public['sha256']}  {snapshot.ARCHIVE}\n":
        raise RuntimeError("Checksum sidecar mismatch")
    publish(folder, public, repo, revision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    try:
        main(parser.parse_args().output_dir)
    except Exception as error:
        print(
            f"Snapshot publication failed ({type(error).__name__}); check setup and stage status."
        )
        sys.exit(1)
