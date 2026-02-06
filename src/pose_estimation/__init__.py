# -*- coding: utf-8 -*-
"""
Pose estimation module for mmWave radar-based human pose estimation.

Sub-modules:
    - model: PointTransformer and loss functions
    - dataset: Dataset classes for radar data
    - train: Training scripts
    - evaluate: Evaluation utilities
"""

# Use lazy imports to avoid loading heavy dependencies
__all__ = [
    'model',
    'dataset',
    'train',
    'evaluate',
]
