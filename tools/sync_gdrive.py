#!/usr/bin/env python3
"""
Upload/download a single file to/from Google Drive using rclone.

Prerequisites:
    1. Install rclone: https://rclone.org/install/
    2. Configure: rclone config  (create a remote named "gdrive" for Google Drive)

Usage:
    python tools/sync_gdrive.py upload   --file ~/wicompass_archives/model_zoo.tar.gz
    python tools/sync_gdrive.py download --file ~/model_zoo.tar.gz
"""

import argparse
import subprocess
import sys
from pathlib import Path

RCLONE_REMOTE = "gdrive"
REMOTE_DIR = "WiCompass"


def main():
    parser = argparse.ArgumentParser(description="Transfer a file to/from Google Drive")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--file", "-f", required=True, type=Path, help="Local file path")
    args = parser.parse_args()

    if args.action == "upload" and not args.file.exists():
        print(f"Error: {args.file} not found")
        sys.exit(1)

    remote = f"{RCLONE_REMOTE}:{REMOTE_DIR}/{args.file.name}"
    if args.action == "upload":
        cmd = ["rclone", "copyto", "-P", str(args.file), remote]
    else:
        cmd = ["rclone", "copyto", "-P", remote, str(args.file)]

    print(f"{args.action.title()}: {args.file.name}")
    ret = subprocess.run(cmd).returncode
    sys.exit(0 if ret == 0 else 1)


if __name__ == "__main__":
    main()
