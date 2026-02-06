# -*- coding: utf-8 -*-
"""
Real-world dataset for mmWave radar-based human pose estimation.

Supports flexible directory structure:
    {data_root}/{folder_name}/radar/{frame}.npy
    {data_root}/{folder_name}/label/{frame}.npy

Each dataset folder contains:
- radar/: (N, 3) point cloud [x, y, z]
- label/: (22, 3) joint positions

Config allows specifying arbitrary folders for train/val splits.
"""

import os
import numpy as np
from .base_dataset import BaseRadarPoseDataset, cropping


def coordinate_transform_y_to_z(joints: np.ndarray) -> np.ndarray:
    """Transform from Y-down to Z-up coordinate system."""
    # Y-down: X=right, Y=down, Z=forward -> Z-up: X=forward, Y=left, Z=up
    new_joints = np.zeros_like(joints)
    new_joints[..., 0] = joints[..., 2]   # X_new = Z_old (forward)
    new_joints[..., 1] = -joints[..., 0]  # Y_new = -X_old (left)
    new_joints[..., 2] = -joints[..., 1]  # Z_new = -Y_old (up)
    return new_joints


def rotate_z_90_cw(joints: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise around Z-axis (facing +Y -> facing +X)."""
    new_joints = np.zeros_like(joints)
    new_joints[..., 0] = joints[..., 1]   # X_new = Y_old
    new_joints[..., 1] = -joints[..., 0]  # Y_new = -X_old
    new_joints[..., 2] = joints[..., 2]   # Z unchanged
    return new_joints


class RealWorldDataset(BaseRadarPoseDataset):
    """
    Real-world dataset with flexible folder-based train/val splits.
    
    Directory structure:
        {data_root}/
            {folder_1}/
                radar/{frame}.npy  # (N, 3) point cloud
                label/{frame}.npy  # (22, 3) joints
            {folder_2}/
                radar/...
                label/...
    
    Config example:
        data_root: datasets/real_world
        train_folders: [A_dance_train]
        val_folders: [A_dance_val]
        coord_transform: y_to_z_rotate
    """
    
    def __init__(self, config, split, device=None):
        self.data_root = config.get("data_root", config.get("dataset_path", ""))
        self.coord_transform = config.get("coord_transform", None)
        
        # Get folders for this split
        if split == "train":
            self.folders = config.get("train_folders", [])
        else:
            self.folders = config.get("val_folders", [])
        
        super().__init__(config, split, device)
        
        # Cache frame ranges for merging
        self._frame_range_cache = {}
    
    def _load_data_list(self):
        """Load file paths from all folders for this split."""
        data_list = []
        
        for folder_name in self.folders:
            folder_path = os.path.join(self.data_root, folder_name)
            radar_dir = os.path.join(folder_path, "radar")
            label_dir = os.path.join(folder_path, "label")
            
            if not os.path.exists(radar_dir) or not os.path.exists(label_dir):
                print(f"[WARN] Skipping {folder_name}: missing radar/ or label/")
                continue
            
            # Get frame numbers (filename without extension)
            radar_files = {}
            for f in os.listdir(radar_dir):
                if f.endswith(".npy"):
                    frame_num = int(f.replace(".npy", ""))
                    radar_files[frame_num] = os.path.join(radar_dir, f)
            
            label_files = {}
            for f in os.listdir(label_dir):
                if f.endswith(".npy"):
                    frame_num = int(f.replace(".npy", ""))
                    label_files[frame_num] = os.path.join(label_dir, f)
            
            # Match frames with both radar and label
            common_frames = sorted(set(radar_files.keys()) & set(label_files.keys()))
            
            # Apply per-folder sample limit if specified
            if self.split == "train":
                max_samples_per_folder = self.config.get("max_samples_per_folder", None)
            else:
                max_samples_per_folder = self.config.get("max_val_samples_per_folder", None)
            
            if max_samples_per_folder and len(common_frames) > max_samples_per_folder:
                common_frames = common_frames[:max_samples_per_folder]
                print(f"[RealWorldDataset] {folder_name}: Limited to first {max_samples_per_folder} samples")
            
            for frame in common_frames:
                data_list.append({
                    "frame": frame,
                    "radar": radar_files[frame],
                    "label": label_files[frame],
                    "radar_dir": radar_dir,
                    "folder": folder_name,
                })
            
            print(f"[RealWorldDataset] {folder_name}: {len(common_frames)} samples")
        
        # Apply total max samples filter based on split (fallback if per-folder limit not used)
        if self.split == "train":
            max_samples = self.config.get("max_train_samples", None)
        else:
            max_samples = self.config.get("max_val_samples", None)
        
        # Only apply total limit if per-folder limit was not used
        if max_samples and len(data_list) > max_samples and not self.config.get("max_samples_per_folder"):
            data_list = data_list[:max_samples]
            print(f"[RealWorldDataset] Limited {self.split} to first {max_samples} samples")
        
        if self.used_data_quantity < 1.0 and len(data_list) > 0:
            n = max(1, int(len(data_list) * self.used_data_quantity))
            indices = np.random.choice(len(data_list), n, replace=False)
            data_list = [data_list[i] for i in sorted(indices)]
            print(f"[RealWorldDataset] Using {n} samples ({self.used_data_quantity*100:.0f}%)")
        
        return data_list
    
    def _load_sample(self, idx):
        """Load and process a single sample."""
        item = self.data_list[idx]
        frame = item["frame"]
        radar_dir = item["radar_dir"]
        
        # Cache frame range for this directory
        if radar_dir not in self._frame_range_cache:
            frames = [int(f.replace(".npy", "")) 
                     for f in os.listdir(radar_dir) if f.endswith(".npy")]
            self._frame_range_cache[radar_dir] = (min(frames), max(frames))
        
        min_frame, max_frame = self._frame_range_cache[radar_dir]
        half = self.merge_nframes // 2
        start = max(min_frame, frame - half)
        end = min(max_frame, frame + half) + 1
        
        # Load label
        gt_joints = np.load(item["label"]).astype(np.float32)
        
        # Apply coordinate transform to label
        if self.coord_transform == "y_to_z":
            gt_joints = coordinate_transform_y_to_z(gt_joints)
        elif self.coord_transform == "y_to_z_rotate":
            gt_joints = coordinate_transform_y_to_z(gt_joints)
            gt_joints = rotate_z_90_cw(gt_joints)
        
        # Load and merge radar frames
        radar_list = []
        for f in range(start, end):
            path = os.path.join(radar_dir, f"{f}.npy")
            if os.path.exists(path):
                data = np.load(path).astype(np.float32)
                
                # Ensure only xyz (in case there are extra channels)
                if data.ndim == 2 and data.shape[1] >= 3:
                    data = data[:, :3]
                
                # Apply coordinate transform to radar
                if self.coord_transform == "y_to_z":
                    data = coordinate_transform_y_to_z(data)
                elif self.coord_transform == "y_to_z_rotate":
                    data = coordinate_transform_y_to_z(data)
                    data = rotate_z_90_cw(data)
                
                radar_list.append(data)
        
        # Merge frames
        if len(radar_list) > 0:
            merged_radar = np.concatenate(radar_list, axis=0)
        else:
            merged_radar = np.zeros((0, 3), dtype=np.float32)
        
        # Normalize relative to pelvis and crop
        if self.normalized:
            pelvis = gt_joints[0].copy()
            merged_radar = merged_radar - pelvis
            merged_radar = cropping(merged_radar, x_range=[-1, 1], y_range=[-1, 1], z_range=[-1, 1])
        
        # Return as list for base class compatibility
        return [merged_radar], gt_joints
    
    def get_data_info(self):
        """Get information about the loaded data."""
        return {
            "split": self.split,
            "folders": self.folders,
            "num_samples": len(self),
            "n_points": self.n_points,
            "merge_nframes": self.merge_nframes,
            "normalized": self.normalized,
            "coord_transform": self.coord_transform,
        }
