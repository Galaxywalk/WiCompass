#!/usr/bin/env python3
"""Build one Zenodo-ready source archive from the current clean Git checkout.

The generated archive contains all tracked files from ``HEAD``, the checked-out
contents of every recursive Git submodule, and a manifest recording the exact
repository and submodule commits.  Dataset, checkpoint, and log archives are
intentionally excluded because they are published as separate Zenodo files.

Usage:
    python tools/build_zenodo_source_archive.py
    python tools/build_zenodo_source_archive.py --output-dir /path/to/output

The checkout must be clean and all submodules must be initialized at the
commits recorded by the superproject.  This prevents publishing a source bundle
that cannot be traced back to one Git revision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def run_git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    """Run Git and return text output with a useful failure message."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result.stdout.strip()


def git_archive(commit: str, prefix: str, cwd: Path) -> bytes:
    """Return a tar archive for one repository or submodule revision."""
    result = subprocess.run(
        ["git", "archive", "--format=tar", f"--prefix={prefix}", commit],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git archive failed in {cwd}: {message}")
    return result.stdout


def extract_archive(archive_bytes: bytes, destination: Path) -> None:
    """Extract a Git-produced tar archive without allowing path traversal."""
    destination_root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Refusing unsafe path in git archive: {member.name}")
        archive.extractall(destination)


def submodule_revisions() -> list[dict[str, str]]:
    """Return initialized recursive submodules at their recorded revisions."""
    output = run_git("submodule", "status", "--recursive")
    revisions: list[dict[str, str]] = []

    for line in output.splitlines():
        if not line:
            continue
        state = line[0]
        parts = line[1:].strip().split()
        if len(parts) < 2:
            raise RuntimeError(f"Could not parse submodule status: {line}")
        commit, relative_path = parts[0], parts[1]

        if state == "-":
            raise RuntimeError(
                f"Submodule {relative_path} is not initialized. Run "
                "`git submodule update --init --recursive` first."
            )
        if state in {"+", "U"}:
            raise RuntimeError(
                f"Submodule {relative_path} is not at the commit recorded by HEAD. "
                "Update or reset the submodule before packaging."
            )

        submodule_dir = REPOSITORY_ROOT / relative_path
        actual_commit = run_git("rev-parse", "HEAD", cwd=submodule_dir)
        if actual_commit != commit:
            raise RuntimeError(
                f"Submodule {relative_path} is at {actual_commit}, expected {commit}."
            )
        revisions.append({"path": relative_path, "commit": commit})

    return revisions


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_checkout() -> None:
    """Ensure that the archive maps exactly to ``HEAD``."""
    status = run_git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "Working tree is not clean. Commit, stash, or remove local changes "
            "before creating a Zenodo source archive."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single source archive suitable for a Zenodo artifact release."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "dist",
        help="Directory for the generated .tar.gz file (default: %(default)s)",
    )
    args = parser.parse_args()

    require_clean_checkout()
    commit = run_git("rev-parse", "HEAD")
    short_commit = run_git("rev-parse", "--short=12", "HEAD")
    revision_label = run_git("describe", "--tags", "--always", "--dirty=+dirty")
    origin_url = run_git("remote", "get-url", "origin")
    submodules = submodule_revisions()

    archive_root = f"WiCompass-source-{short_commit}"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{archive_root}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="wicompass-zenodo-") as temporary_dir:
        staging_dir = Path(temporary_dir)
        extract_archive(git_archive(commit, f"{archive_root}/", REPOSITORY_ROOT), staging_dir)

        for submodule in submodules:
            submodule_dir = REPOSITORY_ROOT / submodule["path"]
            submodule_prefix = f"{archive_root}/{submodule['path']}/"
            extract_archive(
                git_archive(submodule["commit"], submodule_prefix, submodule_dir),
                staging_dir,
            )

        manifest = {
            "archive_format": "WiCompass source snapshot",
            "repository": origin_url,
            "commit": commit,
            "revision": revision_label,
            "submodules": submodules,
            "included": [
                "Tracked repository source files at HEAD",
                "Tracked files for every recursive Git submodule",
                "This manifest",
            ],
            "excluded": [
                "Datasets, checkpoints, and logs (published as separate workspace archives)",
                "Git history and local untracked files",
            ],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = staging_dir / archive_root / "SOURCE_ARCHIVE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        with tarfile.open(output_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(staging_dir / archive_root, arcname=archive_root)

    print(f"Created: {output_path}")
    print(f"Commit:  {commit}")
    print(f"SHA256:  {sha256(output_path)}")
    print("Upload this single .tar.gz file as a new Zenodo record version.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
