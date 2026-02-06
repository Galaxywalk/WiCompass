#!/usr/bin/env python3
"""
Workspace Setup Tool

This script sets up the project workspace by creating necessary directories
and symbolic links after cloning the repository and downloading wicompass_workspace.

Prerequisites:
    1. Clone the Wi-compass repository
    2. Download wicompass_workspace to your preferred location

Usage:
    # Default: workspace at ~/wicompass_workspace
    python tools/setup_workspace.py

    # Custom workspace path
    python tools/setup_workspace.py --workspace /path/to/wicompass_workspace

    # Dry run (show what would be done)
    python tools/setup_workspace.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Default workspace path
DEFAULT_WORKSPACE = os.path.expanduser("~/wicompass_workspace")

# Symbolic link mappings: (link_path_in_repo, target_subdir_in_workspace)
SYMLINK_MAPPINGS = [
    # Root level links
    ("logs", "wicompass_logs"),
    ("model_zoo", "model_zoo"),
    # Dataset links
    ("datasets/AMASS_preproc", "AMASS_preproc"),
    ("datasets/MMFi", "mmfi"),
    ("datasets/mmBody", "mmbody"),
    ("datasets/real_world", "real_world"),
    ("datasets/simulation_datasets", "simulation_datasets"),
]


def setup_symlink(repo_root: Path, workspace: Path, link_rel: str, target_subdir: str, dry_run: bool) -> bool:
    """Create a symbolic link."""
    link_path = repo_root / link_rel
    target_path = workspace / target_subdir

    # Ensure parent directory exists (e.g., create 'datasets/' for 'datasets/MMFi')
    parent_dir = link_path.parent
    if parent_dir != repo_root and not parent_dir.exists():
        if dry_run:
            print(f"  📁 Would create directory: {parent_dir.relative_to(repo_root)}/")
        else:
            parent_dir.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created directory: {parent_dir.relative_to(repo_root)}/")

    # Check if target exists in workspace
    if not target_path.exists():
        print(f"  ⚠️  Skip: {link_rel} (target not found: {target_path})")
        return False

    # Remove existing link or empty directory
    if link_path.is_symlink():
        if dry_run:
            print(f"  🔗 Would update: {link_rel} → {target_path}")
        else:
            link_path.unlink()
    elif link_path.is_dir():
        # Only remove if empty
        try:
            if dry_run:
                print(f"  📁 Would replace empty dir: {link_rel} → {target_path}")
            else:
                link_path.rmdir()
        except OSError:
            print(f"  ❌ Error: {link_rel} is a non-empty directory")
            return False
    elif link_path.exists():
        print(f"  ❌ Error: {link_rel} exists and is not a symlink")
        return False
    else:
        if dry_run:
            print(f"  🔗 Would create: {link_rel} → {target_path}")

    if not dry_run:
        link_path.symlink_to(target_path)
        print(f"  ✅ {link_rel} → {target_path}")

    return True


def verify_workspace(workspace: Path) -> bool:
    """Verify workspace structure."""
    expected_dirs = [
        "wicompass_logs",
        "model_zoo",
        "AMASS_preproc",
        "mmfi",
        "mmbody",
        "real_world",
        "simulation_datasets",
    ]
    missing = []

    for d in expected_dirs:
        if not (workspace / d).exists():
            missing.append(d)

    if missing:
        print(f"⚠️  Warning: Some directories missing in workspace:")
        for d in missing:
            print(f"   - {d}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup Wi-compass workspace with symbolic links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Setup with default workspace path
    python tools/setup_workspace.py

    # Setup with custom workspace path
    python tools/setup_workspace.py --workspace ~/wicompass_workspace

    # Show what would be done without making changes
    python tools/setup_workspace.py --dry-run
        """
    )
    parser.add_argument(
        "--workspace", "-w",
        type=Path,
        default=Path(DEFAULT_WORKSPACE),
        help=f"Path to wicompass_workspace (default: {DEFAULT_WORKSPACE})"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    # Get repo root (parent of tools directory)
    repo_root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("Wi-compass Workspace Setup")
    print("=" * 60)
    print(f"Repository: {repo_root}")
    print(f"Workspace:  {args.workspace}")
    if args.dry_run:
        print("Mode:       DRY RUN")
    print()

    # Check workspace exists
    if not args.workspace.exists():
        print(f"❌ Workspace not found: {args.workspace}")
        print()
        print("Please download wicompass_workspace first:")
        print("  python tools/sync_gdrive.py download")
        sys.exit(1)

    # Verify workspace structure
    print("Checking workspace structure...")
    verify_workspace(args.workspace)
    print()

    # Create symlinks
    print("Setting up symbolic links...")
    success = 0
    total = len(SYMLINK_MAPPINGS)

    for link_rel, target_subdir in SYMLINK_MAPPINGS:
        if setup_symlink(repo_root, args.workspace, link_rel, target_subdir, args.dry_run):
            success += 1

    print()
    print("=" * 60)
    print(f"Done: {success}/{total} links created")

    if success < total:
        print("\n⚠️  Some links were skipped. Check the output above.")
        sys.exit(1)

    print("\n✅ Workspace setup complete!")


if __name__ == "__main__":
    main()

