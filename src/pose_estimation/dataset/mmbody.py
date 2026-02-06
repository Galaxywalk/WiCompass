# -*- coding: utf-8 -*-
"""MMBody dataset for mmWave radar-based human pose estimation."""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for direct execution
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pose_estimation.dataset.base_dataset import BaseRadarPoseDataset, cropping


class MMBodyDataset(BaseRadarPoseDataset):
    """MMBody dataset with train/test split support."""
    
    ALL_TEST_SCENARIOS = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]
    ALL_TRAIN_SCENARIOS = [f"sequence_{i}" for i in range(20)]
    
    def __init__(self, config, split, device=None):
        super().__init__(config, split, device)
    
    def _load_data_list(self):
        """Build DataFrame of radar/mesh paths."""
        scenarios = self._get_scenarios()
        split_path = os.path.join(self.root, self.split)
        
        df_list = []
        if self.split == "train":
            for seq in scenarios:
                seq_path = os.path.join(split_path, seq)
                df_list.append(self._load_sequence(seq_path))
        else:  # test
            for scenario in scenarios:
                base = os.path.join(split_path, scenario)
                seq_dirs = sorted(
                    [d for d in os.listdir(base) if d.startswith("sequence_")],
                    key=lambda x: int(x.split("_")[-1])
                )
                for seq in seq_dirs:
                    df_list.append(self._load_sequence(os.path.join(base, seq)))
        
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    
    def _get_scenarios(self):
        """Get scenario list based on split and config."""
        if self.split == "train":
            return self.ALL_TRAIN_SCENARIOS
        
        scenes_cfg = self.config.get("val_dataset", {}).get("scenes", "all")
        if scenes_cfg == "all":
            return self.ALL_TEST_SCENARIOS
        if isinstance(scenes_cfg, list):
            return scenes_cfg
        if isinstance(scenes_cfg, dict) and scenes_cfg.get("split") == "random":
            ratio = scenes_cfg.get("ratio", 0.5)
            n = int(len(self.ALL_TEST_SCENARIOS) * ratio)
            idx = np.random.permutation(len(self.ALL_TEST_SCENARIOS))[-n:]
            return sorted(np.array(self.ALL_TEST_SCENARIOS)[idx].tolist())
        return self.ALL_TEST_SCENARIOS
    
    def _load_sequence(self, seq_path):
        """Load radar/mesh paths for a sequence."""
        seq_id = int(os.path.basename(seq_path).split("_")[-1])
        
        # Load radar files
        radar_dir = os.path.join(seq_path, "radar")
        radar_files = sorted(
            [f for f in os.listdir(radar_dir) if f.endswith(".npy")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        n_files = int(len(radar_files) * self.used_data_quantity)
        
        # Load mesh files
        mesh_dir = os.path.join(seq_path, "mesh")
        mesh_files = sorted(
            [f for f in os.listdir(mesh_dir) if f.endswith(".npz")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        
        # Build DataFrame
        data = []
        for f in radar_files[:n_files]:
            frame = int(f.split("_")[1].split(".")[0])
            mesh_file = f"frame_{frame}.npz"
            if mesh_file in mesh_files:
                data.append({
                    "Sequence": seq_id,
                    "Frame": frame,
                    "Radar": os.path.join(radar_dir, f),
                    "Mesh": os.path.join(mesh_dir, mesh_file)
                })
        return pd.DataFrame(data)
    
    def _load_sample(self, idx):
        """Load radar frames and ground truth joints."""
        row = self.data_list.iloc[int(idx)]
        frame = row["Frame"]
        
        # Load ground truth (22 joints)
        mesh = np.load(row["Mesh"])
        gt_joints = mesh["joints"][:22].astype(np.float32)
        
        # Load merged radar frames
        radar_list = []
        start = max(0, frame - self.merge_nframes // 2)
        end = start + self.merge_nframes
        
        for f in range(start, end):
            path = row["Radar"].replace(f"frame_{frame}", f"frame_{f}")
            if os.path.exists(path):
                data = np.load(path).astype(np.float32)
                data[:, 3] *= 1e38  # Scale intensity (following mmdiff)
                data[:, 5] /= 100   # Scale velocity
                data = data[:, [0, 1, 2, 3, 5]]  # xyz, intensity, velocity
                
                if self.normalized:
                    data[:, :3] -= gt_joints[0]  # Pelvis normalization
                    data = cropping(data, x_range=[-1.0, 1.0])
                radar_list.append(data)
        
        return radar_list, gt_joints


if __name__ == "__main__":
    # Use path relative to this file
    config_path = Path(__file__).resolve().parent.parent / "configs" / "mmbody.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset = MMBodyDataset(config, split='train', device='cuda')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        radar, joints = dataset[0]
        print(f"Radar shape: {radar.shape}")
        print(f"Joints shape: {joints.shape}")
