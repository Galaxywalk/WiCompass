#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export MMFi ground truth poses (17 joints) to npy format.

This script loads all ground_truth.npy files from the MMFi dataset
and exports them as a single npy file for use with the VQVAE model.

Usage:
    python export_mmfi_poses.py
    python export_mmfi_poses.py --output logs/wicompass/exported_mmwave_poses/mmfi_poses_17joints.npy
    python export_mmfi_poses.py --no-normalize
    python export_mmfi_poses.py --scenes E01 E02 E03  # Export specific scenes only

Joint Format (Human3.6M 17 joints):
    0: Hip/Pelvis (center)
    1-3: RHip -> RKnee -> RAnkle (right leg)
    4-6: LHip -> LKnee -> LAnkle (left leg)
    7-10: Spine -> Thorax -> Neck -> Head (spine chain)
    11-13: LShoulder -> LElbow -> LWrist (left arm)
    14-16: RShoulder -> RElbow -> RWrist (right arm)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# Add src to path for imports
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# MMFi dataset structure constants (from mmfi.py)
SCENE_SUBJECTS = {
    'E01': [f'S{i:02d}' for i in range(1, 11)],
    'E02': [f'S{i:02d}' for i in range(11, 21)],
    'E03': [f'S{i:02d}' for i in range(21, 31)],
    'E04': [f'S{i:02d}' for i in range(31, 41)],
}
ALL_ACTIONS = [f'A{i:02d}' for i in range(1, 28)]


def export_mmfi_poses(
    root: Path,
    output: Path,
    scenes: list = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Export all MMFi ground truth poses to a single npy file.
    
    Args:
        root: Path to MMFi dataset root (e.g., datasets/MMFi)
        output: Output npy file path
        scenes: List of scenes to export (default: all)
        normalize: Whether to apply pelvis normalization
        
    Returns:
        Exported poses array with shape (N, 17, 3)
    """
    if scenes is None:
        scenes = list(SCENE_SUBJECTS.keys())
    
    print(f"MMFi Pose Export")
    print(f"=" * 50)
    print(f"Root:      {root}")
    print(f"Scenes:    {scenes}")
    print(f"Normalize: {normalize}")
    print(f"Output:    {output}")
    print()
    
    # Collect all ground_truth.npy files
    all_poses = []
    stats = {'scenes': {}, 'total_frames': 0}
    
    for scene in scenes:
        if scene not in SCENE_SUBJECTS:
            print(f"  [WARN] Unknown scene: {scene}, skipping")
            continue
            
        subjects = SCENE_SUBJECTS[scene]
        scene_frames = 0
        
        for subject in tqdm(subjects, desc=f"Scene {scene}", leave=False):
            for action in ALL_ACTIONS:
                gt_path = root / scene / subject / action / 'ground_truth.npy'
                
                if not gt_path.exists():
                    continue
                
                # Load ground truth poses
                gt_data = np.load(gt_path)  # (N_frames, 17, 3)
                
                if normalize:
                    # Pelvis normalization: subtract joint 0 position from all joints
                    pelvis = gt_data[:, 0:1, :]  # (N, 1, 3)
                    gt_data = gt_data - pelvis
                
                all_poses.append(gt_data.astype(np.float32))
                scene_frames += len(gt_data)
        
        stats['scenes'][scene] = scene_frames
        stats['total_frames'] += scene_frames
        print(f"  {scene}: {scene_frames:,} frames")
    
    if not all_poses:
        print("[ERROR] No poses found!")
        return None
    
    # Concatenate all poses
    poses = np.concatenate(all_poses, axis=0)
    print()
    print(f"Total: {poses.shape[0]:,} frames")
    print(f"Shape: {poses.shape}")
    
    # Save to npy file
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, poses)
    print(f"\n✅ Saved to {output}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Min: {poses.min():.4f}")
    print(f"  Max: {poses.max():.4f}")
    print(f"  Mean: {poses.mean():.4f}")
    print(f"  Std: {poses.std():.4f}")
    
    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Export MMFi ground truth poses (17 joints) to npy format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export all scenes with pelvis normalization (default)
    python export_mmfi_poses.py
    
    # Export without normalization
    python export_mmfi_poses.py --no-normalize
    
    # Export specific scenes only
    python export_mmfi_poses.py --scenes E01 E02 E03
    
    # Custom output path
    python export_mmfi_poses.py --output /path/to/output.npy
        """
    )
    parser.add_argument(
        "--root", type=str, default="datasets/MMFi",
        help="Path to MMFi dataset root (default: datasets/MMFi)"
    )
    parser.add_argument(
        "--output", "-o", type=str, 
        default="logs/wicompass/exported_mmwave_poses/mmfi_poses_17joints.npy",
        help="Output npy file path"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scenes to export (default: all - E01, E02, E03, E04)"
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Disable pelvis normalization"
    )
    args = parser.parse_args()
    
    # Convert paths
    root = Path(args.root)
    output = Path(args.output)
    
    # Check root exists
    if not root.exists():
        print(f"[ERROR] Dataset root not found: {root}")
        sys.exit(1)
    
    # Export poses
    export_mmfi_poses(
        root=root,
        output=output,
        scenes=args.scenes,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()

