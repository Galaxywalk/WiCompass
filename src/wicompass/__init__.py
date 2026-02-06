# -*- coding: utf-8 -*-
"""
Wi-compass VQ-VAE module for joint tokenization.

This package previously imported model and dataset modules eagerly, which
pulled in heavy dependencies like torch even when only lightweight utilities
were needed (e.g., visualization). Imports are now lazy via ``__getattr__`` so
that modules are loaded only when their symbols are accessed.
"""

__all__ = [
    'create_joint_tokenizer',
    'JointVQVAELoss',
    'create_dataset',
    'get_available_datasets',
    'load_dataset_configs',
    # Sub-modules
    'model',
    'dataset',
    'evaluation',
    'train',
]


def __getattr__(name):
    """Lazily import heavy modules only when requested."""
    if name in {'create_joint_tokenizer', 'JointVQVAELoss'}:
        from .model import create_joint_tokenizer, JointVQVAELoss

        return create_joint_tokenizer if name == 'create_joint_tokenizer' else JointVQVAELoss

    if name in {'create_dataset', 'get_available_datasets', 'load_dataset_configs'}:
        from .dataset import create_dataset, get_available_datasets, load_dataset_configs

        mapping = {
            'create_dataset': create_dataset,
            'get_available_datasets': get_available_datasets,
            'load_dataset_configs': load_dataset_configs,
        }
        return mapping[name]

    raise AttributeError(f"module 'wicompass' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
