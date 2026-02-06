# -*- coding: utf-8 -*-
"""
Dataset module for mmWave radar-based human pose estimation.
"""

from .base_dataset import BaseRadarPoseDataset, cropping
from .mmbody import MMBodyDataset
from .mmfi import MMFiDataset
from .simulation_mmbody import SimulationMMBody
from .real_world import RealWorldDataset


def get_dataset(config, split, device=None):
    """
    Factory function to create dataset based on config.
    
    Args:
        config: Configuration dictionary with 'dataset_name' key
        split: 'train' or 'test'
        device: Torch device
    
    Returns:
        Dataset instance
    """
    dataset_name = config.get('dataset_name', 'SimulationMMBody')
    
    dataset_classes = {
        'MMBodyDataset': MMBodyDataset,
        'MMFiDataset': MMFiDataset,
        'SimulationMMBody': SimulationMMBody,
        'RealWorldDataset': RealWorldDataset,
    }
    
    if dataset_name not in dataset_classes:
        available = list(dataset_classes.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return dataset_classes[dataset_name](config, split, device)


__all__ = [
    'BaseRadarPoseDataset',
    'MMBodyDataset',
    'MMFiDataset', 
    'SimulationMMBody',
    'RealWorldDataset',
    'cropping',
    'get_dataset'
]
