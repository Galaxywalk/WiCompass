#!/usr/bin/env python3
"""
Extract only the necessary data from MMFi and MMBody datasets.

This script copies only the files actually used by the pose estimation models,
significantly reducing storage requirements by skipping unused modalities.

MMFi: Only keeps mmwave/ and ground_truth.npy (skips depth, rgb, lidar, etc.)
MMBody: Only keeps radar/ and mesh/ (skips image/, depth/)

Usage:
    python tools/extract_datasets.py --dry-run  # Preview what will be copied
    python tools/extract_datasets.py            # Actually copy files
"""

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


# Source dataset paths (original full datasets)
# Modify these to point to your original downloaded datasets
MMFI_SRC = Path("mmfi/Unzipfiles")
MMBODY_SRC = Path("mmbody")

# Destination root (wicompass_workspace for extracted data)
DEST_ROOT = Path("wicompass_workspace")


def get_size_str(size_bytes):
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_dir_size(path):
    """Get total size of a directory."""
    total = 0
    for p in Path(path).rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total


def extract_mmfi(dest_root: Path, dry_run: bool = False):
    """
    Extract MMFi dataset - only mmwave/ folder and ground_truth.npy.
    
    Structure: E{01-04}/S{01-40}/A{01-27}/
        - mmwave/*.npy (radar point clouds) - KEEP
        - ground_truth.npy (pose labels) - KEEP
        - depth/, rgb/, infra1/, infra2/, lidar/, wifi-csi/ - SKIP
    """
    print("\n" + "=" * 60)
    print("Extracting MMFi Dataset")
    print("=" * 60)
    print(f"Source: {MMFI_SRC}")
    print(f"Dest:   {dest_root / 'mmfi'}")
    
    if not MMFI_SRC.exists():
        print(f"⚠️  Source not found: {MMFI_SRC}")
        return
    
    dest = dest_root / "mmfi"
    total_copied = 0
    file_count = 0
    
    # Iterate through all scene/subject/action combinations
    scenes = sorted([d for d in MMFI_SRC.iterdir() if d.is_dir() and d.name.startswith('E')])
    
    for scene in tqdm(scenes, desc="Scenes"):
        subjects = sorted([d for d in scene.iterdir() if d.is_dir() and d.name.startswith('S')])
        
        for subject in subjects:
            actions = sorted([d for d in subject.iterdir() if d.is_dir() and d.name.startswith('A')])
            
            for action in actions:
                rel_path = action.relative_to(MMFI_SRC)
                dest_action = dest / rel_path
                
                # Copy ground_truth.npy
                gt_src = action / "ground_truth.npy"
                if gt_src.exists():
                    gt_dst = dest_action / "ground_truth.npy"
                    if not dry_run:
                        gt_dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(gt_src, gt_dst)
                    total_copied += gt_src.stat().st_size
                    file_count += 1
                
                # Copy mmwave folder
                mmwave_src = action / "mmwave"
                if mmwave_src.exists():
                    mmwave_dst = dest_action / "mmwave"
                    if not dry_run:
                        if mmwave_dst.exists():
                            shutil.rmtree(mmwave_dst)
                        shutil.copytree(mmwave_src, mmwave_dst)
                    total_copied += get_dir_size(mmwave_src)
                    file_count += len(list(mmwave_src.glob('*')))
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}MMFi: {file_count} files, {get_size_str(total_copied)}")


def extract_mmbody(dest_root: Path, dry_run: bool = False):
    """
    Extract MMBody dataset - only radar/ and mesh/ folders.
    
    Structure: {train|test}/sequence_*/
        - radar/*.npy (radar point clouds) - KEEP
        - mesh/*.npz (SMPL mesh with joints) - KEEP  
        - image/, depth/ - SKIP (image is 11GB per sequence!)
    """
    print("\n" + "=" * 60)
    print("Extracting MMBody Dataset")
    print("=" * 60)
    print(f"Source: {MMBODY_SRC}")
    print(f"Dest:   {dest_root / 'mmbody'}")
    
    if not MMBODY_SRC.exists():
        print(f"⚠️  Source not found: {MMBODY_SRC}")
        return
    
    dest = dest_root / "mmbody"
    total_copied = 0
    file_count = 0
    
    for split in ['train', 'test']:
        split_src = MMBODY_SRC / split
        if not split_src.exists():
            continue
        
        # Find all sequence directories (handle nested structure for test)
        if split == 'train':
            sequences = sorted([d for d in split_src.iterdir() if d.is_dir() and d.name.startswith('sequence_')])
        else:
            # test has scenario/sequence structure
            sequences = []
            for scenario in split_src.iterdir():
                if scenario.is_dir():
                    for seq in scenario.iterdir():
                        if seq.is_dir() and seq.name.startswith('sequence_'):
                            sequences.append(seq)
            sequences = sorted(sequences)
        
        for seq in tqdm(sequences, desc=f"{split}"):
            rel_path = seq.relative_to(MMBODY_SRC)
            dest_seq = dest / rel_path
            
            # Copy radar folder
            radar_src = seq / "radar"
            if radar_src.exists():
                radar_dst = dest_seq / "radar"
                if not dry_run:
                    radar_dst.parent.mkdir(parents=True, exist_ok=True)
                    if radar_dst.exists():
                        shutil.rmtree(radar_dst)
                    shutil.copytree(radar_src, radar_dst)
                total_copied += get_dir_size(radar_src)
                file_count += len(list(radar_src.glob('*')))
            
            # Copy mesh folder
            mesh_src = seq / "mesh"
            if mesh_src.exists():
                mesh_dst = dest_seq / "mesh"
                if not dry_run:
                    if mesh_dst.exists():
                        shutil.rmtree(mesh_dst)
                    shutil.copytree(mesh_src, mesh_dst)
                total_copied += get_dir_size(mesh_src)
                file_count += len(list(mesh_src.glob('*')))
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}MMBody: {file_count} files, {get_size_str(total_copied)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract only necessary data from datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be copied without actually copying",
    )
    parser.add_argument(
        "--dest", "-d",
        type=Path,
        default=DEST_ROOT,
        help=f"Destination directory (default: {DEST_ROOT})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mmfi", "mmbody", "all"],
        default=["all"],
        help="Which datasets to extract (default: all)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Extraction Tool")
    print("=" * 60)
    print(f"Destination: {args.dest}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be copied")
    
    datasets = args.datasets if "all" not in args.datasets else ["mmfi", "mmbody"]
    
    if "mmfi" in datasets:
        extract_mmfi(args.dest, args.dry_run)
    
    if "mmbody" in datasets:
        extract_mmbody(args.dest, args.dry_run)
    
    print("\n" + "=" * 60)
    print("Done!")
    if args.dry_run:
        print("💡 Remove --dry-run to actually copy files")
    print("=" * 60)


if __name__ == "__main__":
    main()

