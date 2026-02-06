# -*- coding: utf-8 -*-
"""
Evaluator for mmWave radar-based human pose estimation.
"""

import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from pathlib import Path

from pose_estimation.model import PointTransformer, mpjpe, p_mpjpe
from pose_estimation.dataset import get_dataset


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to model checkpoint
        device: Torch device
    
    Returns:
        Loaded model in eval mode
    """
    model = PointTransformer(config, device=device).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    batch_size: int = 16,
) -> Optional[Dict]:
    """
    Evaluate model on a dataset split defined by config.
    
    Args:
        model: Model to evaluate
        config: Configuration with val_dataset settings
        device: Torch device
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with evaluation results or None if dataset is empty
    """
    dataset = get_dataset(config, 'test', device)
    
    if len(dataset) == 0:
        return None
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    mpjpe_values = []
    p_mpjpe_values = []
    
    for data in loader:
        radar_data, gt_data = data[0], data[1]
        gt_data = gt_data.to(device)
        
        outputs, _ = model(radar_data)
        outputs = outputs.type(torch.FloatTensor).to(device)
        
        # Per-sample metrics
        for i in range(outputs.shape[0]):
            sample_mpjpe = mpjpe(outputs[i:i+1], gt_data[i:i+1]).item() * 1e3
            sample_p_mpjpe = p_mpjpe(outputs[i:i+1], gt_data[i:i+1]).item() * 1e3
            mpjpe_values.append(sample_mpjpe)
            p_mpjpe_values.append(sample_p_mpjpe)
    
    return {
        'mpjpe_mean': np.mean(mpjpe_values),
        'mpjpe_std': np.std(mpjpe_values),
        'p_mpjpe_mean': np.mean(p_mpjpe_values),
        'p_mpjpe_std': np.std(p_mpjpe_values),
        'num_samples': len(dataset)
    }


class Evaluator:
    """
    Evaluator for pose estimation models.
    
    Supports evaluation by subject, scene, action, or custom splits.
    """
    
    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        device: Union[str, torch.device] = 'cuda',
        batch_size: int = 16,
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Base configuration
            checkpoint_path: Path to model checkpoint
            device: Torch device
            batch_size: Batch size for evaluation
        """
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = load_model(config, checkpoint_path, device)
        print(f"Loaded model from: {checkpoint_path}")
    
    def _create_split_config(
        self,
        split_type: str,
        split_value: str,
    ) -> dict:
        """Create config for evaluating a specific split.
        
        Preserves original val_dataset settings (scenes, subjects) and only
        overrides the specific split_type.
        """
        config = deepcopy(self.config)
        
        # Start with original val_dataset settings, or default to 'all'
        if 'val_dataset' not in config:
            config['val_dataset'] = {}
        
        # Ensure all keys exist with defaults
        config['val_dataset'].setdefault('scenes', 'all')
        config['val_dataset'].setdefault('subjects', 'all')
        config['val_dataset'].setdefault('actions', 'all')
        
        # Override only the specific split type
        if split_type == 'subject':
            config['val_dataset']['subjects'] = [split_value]
        elif split_type == 'scene':
            config['val_dataset']['scenes'] = [split_value]
        elif split_type == 'action':
            config['val_dataset']['actions'] = [split_value]
        
        return config
    
    def evaluate_by_split(
        self,
        split_type: str,
        split_values: List[str],
        desc: Optional[str] = None,
    ) -> List[Dict]:
        """
        Evaluate model on multiple splits of a given type.
        
        Args:
            split_type: 'subject', 'scene', or 'action'
            split_values: List of values to evaluate
            desc: Description for progress bar
        
        Returns:
            List of result dicts with split info included
        """
        results = []
        desc = desc or split_type.capitalize() + 's'
        
        for value in tqdm(split_values, desc=desc):
            config = self._create_split_config(split_type, value)
            result = evaluate_split(self.model, config, self.device, self.batch_size)
            
            if result:
                result[split_type] = value
                results.append(result)
        
        return results
    
    def evaluate_subjects(self, subjects: List[str]) -> List[Dict]:
        """Evaluate on specific subjects."""
        return self.evaluate_by_split('subject', subjects, 'Subjects')
    
    def evaluate_scenes(self, scenes: List[str]) -> List[Dict]:
        """Evaluate on specific scenes."""
        return self.evaluate_by_split('scene', scenes, 'Scenes')
    
    def evaluate_actions(self, actions: List[str]) -> List[Dict]:
        """Evaluate on specific actions."""
        return self.evaluate_by_split('action', actions, 'Actions')
    
    def evaluate_custom(self, config: dict) -> Optional[Dict]:
        """Evaluate on a custom config."""
        return evaluate_split(self.model, config, self.device, self.batch_size)

