# -*- coding: utf-8 -*-
"""
Evaluation Core Module

Configuration loading, model loading, dataset utilities, and base classes.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, random_split

from wicompass.model import create_joint_tokenizer
from wicompass.dataset import create_dataset, get_available_datasets

# Import visualization constants
from wicompass.visualization import JOINT_NAMES, BONE_CONNECTIONS


# =============================================================================
# Configuration Loading Functions
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load config file, supports JSON and YAML formats, handles encoding automatically"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    return yaml.safe_load(f)
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == 'utf-8':
                raise e
    
    # If all encodings fail, try binary mode
    try:
        with open(config_path, 'rb') as f:
            content = f.read()
            for encoding in encodings:
                try:
                    text_content = content.decode(encoding)
                    if config_path.endswith(('.yaml', '.yml')):
                        import yaml
                        return yaml.safe_load(text_content)
                    return json.loads(text_content)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    except Exception:
        pass
    
    raise ValueError(f"Unable to load config file {config_path} with any supported encoding")


# =============================================================================
# Dataset Utilities
# =============================================================================

def get_dataset_configs(data_config: Dict) -> List[Dict]:
    """Get dataset configuration list from data config"""
    datasets_config = data_config.get('datasets')
    
    if datasets_config:
        all_available = get_available_datasets(['amass', 'mmbody', 'mmfi', 'wicompass'])
        enabled_datasets = [ds for ds in all_available if ds['name'] in datasets_config]
    else:
        enabled_datasets = get_available_datasets(['amass', 'mmbody', 'mmfi', 'wicompass'])
    
    return enabled_datasets


def create_dataloader(dataset, batch_size: int = 64, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """Create data loader, optimized for GPU datasets"""
    # Check if data is already on GPU
    if hasattr(dataset, 'joints_data') and dataset.joints_data.is_cuda:
        def gpu_collate_fn(batch):
            joints, labels = zip(*batch)
            return torch.stack(joints), torch.tensor(labels, dtype=torch.long, device=joints[0].device)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=0, collate_fn=gpu_collate_fn, pin_memory=False)
    else:
        def cpu_collate_fn(batch):
            joints, labels = zip(*batch)
            return torch.stack(joints), torch.tensor(labels, dtype=torch.long)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=cpu_collate_fn)


def split_dataset(dataset, train_ratio: float = 0.9, seed: int = 42) -> Tuple:
    """Split dataset into train and test sets"""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, test_size], generator=generator)


def create_evaluation_dataset(config_path: str, num_joints: int = 22, device: str = 'cuda'):
    """Create evaluation dataset"""
    config = load_config(config_path)
    dataset_configs = get_dataset_configs(config.get('data', {}))
    enabled_datasets = [cfg['name'] for cfg in dataset_configs]
    
    master_dataset, dataset_info = create_dataset(dataset_configs, num_joints, device)
    
    return master_dataset, dataset_info, enabled_datasets


# =============================================================================
# Device Utilities
# =============================================================================

def get_device(device: str = 'cuda') -> torch.device:
    """Get the appropriate torch device"""
    if torch.cuda.is_available() and 'cuda' in device:
        return torch.device(device)
    return torch.device('cpu')


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_config: Dict, checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load model from checkpoint"""
    print(f"🔧 Loading model from {checkpoint_path}")
    model = create_joint_tokenizer(model_config)
    
    # Determine target device
    if torch.cuda.is_available() and 'cuda' in device:
        device_obj = torch.device(device)
    else:
        device_obj = torch.device('cpu')
    
    # Force map to correct device when loading checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle DataParallel saved models
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()
    print("✅ Model loaded successfully")
    
    return model


# =============================================================================
# Base Visualizer Class
# =============================================================================

class BaseVisualizer:
    """
    Base class for visualization tools.
    Provides common functionality for loading config, model, and datasets.
    """
    
    def __init__(self, config_path: str, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize base visualizer.
        
        Args:
            config_path: Path to config file
            model_path: Path to model checkpoint (optional)
            device: Compute device ('cuda' or 'cpu')
        """
        self.config_path = str(config_path)
        self.device = get_device(device)
        self.device_str = str(self.device)
        
        # Load configuration
        print(f"📁 Loading config from {config_path}")
        self.config = load_config(self.config_path)
        self.model_cfg = self.config.get('model', {})
        self.data_cfg = self.config.get('data', {})
        
        # Load model if path provided
        self.model = None
        if model_path:
            self.model = load_model(self.model_cfg, model_path, device)
    
    def get_dataset_configs(self) -> List[Dict]:
        """Get dataset configurations from config"""
        return get_dataset_configs(self.data_cfg)
    
    def create_dataset(self, device: Optional[str] = None):
        """
        Create evaluation dataset.
        
        Args:
            device: Device for dataset (defaults to 'cpu' for memory efficiency)
            
        Returns:
            Tuple of (dataset, dataset_info, enabled_datasets)
        """
        device = device or 'cpu'
        return create_evaluation_dataset(
            self.config_path,
            self.model_cfg.get('num_joints', 22),
            device
        )
