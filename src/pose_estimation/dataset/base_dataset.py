# -*- coding: utf-8 -*-
"""Base dataset class for mmWave radar-based human pose estimation."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def cropping(radar_pc, x_range=[-1.67, 1.67], y_range=[-1, 1], z_range=[-1.67, 1.67]):
    """Crop radar point cloud to a bounding box."""
    mask = (
        (radar_pc[:, 0] >= x_range[0]) & (radar_pc[:, 0] <= x_range[1]) &
        (radar_pc[:, 1] >= y_range[0]) & (radar_pc[:, 1] <= y_range[1]) &
        (radar_pc[:, 2] >= z_range[0]) & (radar_pc[:, 2] <= z_range[1])
    )
    return radar_pc[mask]


class BaseRadarPoseDataset(Dataset):
    """
    Base class for mmWave radar pose estimation datasets.
    
    Subclasses should implement:
    - _load_data_list(): Build list of data samples
    - _load_sample(idx): Load (radar_frames, gt_joints) for a sample
    """
    
    def __init__(self, config, split, device=None):
        super().__init__()
        self.config = config
        self.split = split
        self.device = device
        self.root = config.get("dataset_path", "")
        
        # Common data format params
        fmt = config.get("data_format", {})
        self.n_points = fmt.get("n_points", 200)
        self.merge_nframes = fmt.get("merge_nframes", 1)
        self.normalized = fmt.get("normalized", True)
        self.input_channels = fmt.get("radar_input_c", 5)
        self.num_joints = fmt.get("num_joints", 17)
        
        # Random seed
        self.random_seed = config.get("random_seed", 42)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Data quantity (for training subset)
        self.used_data_quantity = 1.0
        if split == "train" and "used_data_quantity" in config:
            self.used_data_quantity = config["used_data_quantity"]
        
        # Experiment name
        self._exp_name = config.get("recorder", {}).get("exp_name", "experiment")
        
        # Load data list (implemented by subclass)
        self.data_list = self._load_data_list()
    
    def _load_data_list(self):
        """Build list of data samples. Override in subclass."""
        raise NotImplementedError
    
    def _load_sample(self, idx):
        """Load raw sample data. Override in subclass. Returns (radar_data_list, gt_joints)."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """Returns (radar_tensor, gt_tensor) both on self.device."""
        radar_list, gt_joints = self._load_sample(idx)
        
        # Merge frames
        if len(radar_list) > 0:
            radar_data = np.concatenate(radar_list, axis=0)
        else:
            radar_data = np.zeros((0, self.input_channels))
        
        # Convert to tensor
        radar_tensor = torch.from_numpy(radar_data).float()
        
        if self.device:
            radar_tensor = radar_tensor.to(self.device)
        
        # Downsample/pad to n_points
        radar_tensor = self._process_points(radar_tensor)
        
        # Normalize gt_joints (radar is already normalized in _load_sample if needed)
        if self.normalized:
            pelvis = gt_joints[0]
            gt_joints = gt_joints - pelvis
        
        gt_tensor = torch.from_numpy(gt_joints).float()
        if self.device:
            gt_tensor = gt_tensor.to(self.device)
        
        return radar_tensor, gt_tensor
    
    def _process_points(self, points):
        """Downsample or pad point cloud to n_points."""
        n = points.shape[0]
        
        if n == 0:
            return torch.zeros((self.n_points, self.input_channels), device=self.device)
        
        if n < self.n_points:
            # Pad with zeros
            pad = torch.zeros((self.n_points - n, self.input_channels), device=self.device)
            return torch.cat([points, pad], dim=0)
        
        if n == self.n_points:
            return points
        
        # Downsample with FPS
        try:
            from pointnet2_ops import pointnet2_utils
            xyz = points[:, :3].unsqueeze(0).contiguous()
            idx = pointnet2_utils.furthest_point_sample(xyz, self.n_points)
            idx = idx.squeeze(0).to(dtype=torch.int64)
            return points[idx]
        except ImportError:
            # Fallback: random sampling
            idx = torch.randperm(n, device=self.device)[:self.n_points]
            return points[idx]
    
    def exp_name(self):
        return self._exp_name
