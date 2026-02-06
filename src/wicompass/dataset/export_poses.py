#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export MMBody train poses to npy format.

MMBody uses Z-up coordinate system. Empirically, keeping Z-up gives 
better VQVAE reconstruction (14.6mm vs 49.1mm) than converting to Y-down.

Usage:
    python export_poses.py
    python export_poses.py --split test
    python export_poses.py --no-normalize
    python export_poses.py --coord-transform  # Convert to Y-down (NOT recommended)
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm


def z_up_to_y_down(joints: np.ndarray) -> np.ndarray:
    """
    Convert from Z-up to Y-down coordinate system.
    
    MMBody uses Z-up (head has positive Z, feet have negative Z).
    AMASS uses Y-down (head has negative Y, feet have positive Y).
    
    Transform: (X, Y, Z) -> (X, -Z, Y)
    After transform: head has negative Y, feet have positive Y.
    
    Args:
        joints: np.ndarray of shape (..., 3)
    Returns:
        Transformed joints with same shape
    """
    transformed = np.empty_like(joints)
    transformed[..., 0] = joints[..., 0]      # X stays the same
    transformed[..., 1] = -joints[..., 2]     # New Y = -Z (head gets negative Y)
    transformed[..., 2] = joints[..., 1]      # New Z = Y (original Y becomes depth)
    return transformed


def main():
    parser = argparse.ArgumentParser(description="Export MMBody poses to npy")
    parser.add_argument("--root", type=str, default="datasets/mmBody")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output", type=str, default="logs/wicompass/exported_mmwave_poses")
    parser.add_argument("--no-normalize", action="store_true", 
                        help="Skip pelvis normalization")
    parser.add_argument("--coord-transform", action="store_true",
                        help="Convert to Y-down (NOT recommended - gives worse VQVAE results)")
    args = parser.parse_args()

    root = Path(args.root) / args.split
    files = sorted(root.rglob("mesh/frame_*.npz"))
    print(f"Found {len(files)} frames in {root}")
    
    if args.coord_transform:
        print("Coordinate transform: Z-up -> Y-down")
    else:
        print("Keeping original Z-up coordinates (better for VQVAE)")

    poses = np.empty((len(files), 22, 3), dtype=np.float32)
    for i, fp in enumerate(tqdm(files, desc="Loading")):
        with np.load(fp) as data:
            joints = data['joints'][:22].astype(np.float32)
            
            # Convert from Z-up to Y-down (only if explicitly requested)
            if args.coord_transform:
                joints = z_up_to_y_down(joints)
            
            # Pelvis normalization (default)
            if not args.no_normalize:
                joints = joints - joints[0:1]
            
            poses[i] = joints

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "mmbody_poses.npy"
    np.save(out_file, poses)
    print(f"✅ Saved {poses.shape} to {out_file}")


if __name__ == "__main__":
    main()
