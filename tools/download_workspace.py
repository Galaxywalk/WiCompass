#!/usr/bin/env python3
"""
Download workspace archives from Google Drive using rclone.

Prerequisites:
    1. Install rclone: https://rclone.org/install/
    2. Configure rclone for Google Drive (see README for details)
    3. Add the shared WiCompass folder to your Google Drive

Usage:
    python tools/download_workspace.py --workspace ~/wicompass_workspace
    python tools/download_workspace.py --workspace ~/wicompass_workspace --minimal
    python tools/download_workspace.py --workspace ~/wicompass_workspace --components model_zoo mmfi
    python tools/download_workspace.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

RCLONE_REMOTE = "gdrive"
REMOTE_DIR = "WiCompass"

COMPONENTS = [
    ("model_zoo",           "1.6 GB",  "Pre-trained model weights"),
    ("wicompass_logs",      "34 GB",   "Training logs and results"),
    ("AMASS_preproc",       "3.2 GB",  "Processed AMASS dataset"),
    ("mmfi",                "1.6 GB",  "MMFi dataset"),
    ("mmbody",              "34 GB",   "mmBody dataset"),
    ("real_world",          "429 MB",  "Real-world collected dataset"),
    ("simulation_datasets", "8.2 GB",  "Simulated mmWave dataset"),
]

MINIMAL = ["model_zoo", "wicompass_logs"]


def download_component(name: str, workspace: Path) -> bool:
    """Download and extract one component."""
    component_dir = workspace / name
    if component_dir.exists() and any(component_dir.iterdir()):
        print(f"[SKIP] {name}: already exists")
        return True

    archive = workspace / f"{name}.tar.gz"
    remote = f"{RCLONE_REMOTE}:{REMOTE_DIR}/{name}.tar.gz"

    if not archive.exists():
        print(f"Downloading {name} ...")
        ret = subprocess.run(["rclone", "copyto", "-P", remote, str(archive)]).returncode
        if ret != 0:
            print(f"  -> download FAILED")
            return False

    print(f"Extracting {name} ...")
    ret = subprocess.run(["tar", "-xzf", str(archive), "-C", str(workspace)]).returncode
    if ret != 0:
        print(f"  -> extraction FAILED")
        return False

    archive.unlink()
    print(f"  -> done")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download workspace from Google Drive")
    parser.add_argument("--workspace", "-w", type=Path, help="Local workspace directory")
    parser.add_argument("--components", "-c", nargs="+", help="Download specific components only")
    parser.add_argument("--minimal", action="store_true",
                        help="Download minimal set (model_zoo + wicompass_logs) for reproducing paper figures")
    parser.add_argument("--list", action="store_true", help="List available components")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':25s} {'Size':>8s}  Description")
        print(f"{'-'*25} {'-'*8}  {'-'*30}")
        for name, size, desc in COMPONENTS:
            tag = " *" if name in MINIMAL else ""
            print(f"{name:25s} {size:>8s}  {desc}{tag}")
        print(f"\n  * = minimal set (~36 GB, enough to reproduce paper figures)")
        return

    if not args.workspace:
        parser.error("--workspace is required (use --list to see available components)")

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    all_names = [c[0] for c in COMPONENTS]

    if args.minimal:
        targets = MINIMAL
    elif args.components:
        targets = args.components
        for t in targets:
            if t not in all_names:
                print(f"Error: unknown component '{t}'")
                sys.exit(1)
    else:
        targets = all_names

    for name in targets:
        download_component(name, workspace)

    print()
    print(f"Next: python tools/setup_workspace.py --workspace {workspace}")


if __name__ == "__main__":
    main()
