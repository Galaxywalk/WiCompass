# -*- coding: utf-8 -*-
"""
Model Encoder Module

Contains ModelEncoder class and dataset encoding functions.
"""

import numpy as np
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader

from .core import (
    load_config,
    load_model,
    create_dataloader,
    create_evaluation_dataset,
)


class ModelEncoder:
    """Unified model encoder for converting poses to tokens"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def encode_batch(self, joints: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode single batch"""
        # Check if data is already on correct device, avoid unnecessary transfers
        if joints.device != torch.device(self.device):
            joints = joints.to(self.device)
        
        joints_visible = torch.ones(joints.shape[:2], device=self.device).bool()
        
        with torch.no_grad():
            encoding_indices, _, _ = self.model.encode(joints, joints_visible, train=False)
            return encoding_indices
    
    def encode_dataset(self, dataloader: DataLoader) -> Dict:
        """Encode entire dataset, optimized for GPU memory usage and transfer efficiency"""
        all_tokens_gpu = []
        all_labels_gpu = []
        total_samples = 0
        
        with torch.no_grad():
            for joints, labels in tqdm(dataloader, desc="Encoding"):
                encoding_indices = self.encode_batch(joints)
                
                if encoding_indices is not None:
                    # Accumulate data on GPU to reduce transfer overhead
                    all_tokens_gpu.append(encoding_indices)
                    all_labels_gpu.append(labels)
                    total_samples += joints.shape[0]
        
        # Concatenate on GPU once, then transfer to CPU
        if all_tokens_gpu:
            tokens_gpu = torch.cat(all_tokens_gpu, dim=0)
            labels_gpu = torch.cat(all_labels_gpu, dim=0)
            
            # Transfer to CPU and convert to numpy in one go
            tokens = tokens_gpu.cpu().numpy()
            labels = labels_gpu.cpu().numpy()
        else:
            tokens = np.array([])
            labels = np.array([])
        
        return {
            'tokens': tokens,
            'labels': labels,
            'total_samples': total_samples
        }


# =============================================================================
# Token I/O Functions
# =============================================================================

def save_tokens(tokens: np.ndarray, labels: np.ndarray, output_path: str, metadata: Dict = None):
    """Save tokens to HDF5 file"""
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('tokens', data=tokens, compression='gzip', compression_opts=9)
        f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_tokens(file_path: str) -> Dict:
    """Load tokens from HDF5 file"""
    with h5py.File(file_path, 'r') as f:
        tokens = f['tokens'][:]
        labels = f['labels'][:]
        metadata = dict(f.attrs)
    
    return {'tokens': tokens, 'labels': labels, 'metadata': metadata}


# =============================================================================
# High-level Interface
# =============================================================================

def encode_dataset(model_path: str, config_path: str, output_path: str, 
                  batch_size: int = 128, device: str = 'cuda') -> Dict:
    """One-click dataset encoding, optimized version"""
    # Auto-adjust batch size based on GPU memory
    if device.startswith('cuda') and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if gpu_memory_gb > 12:  # High-end GPU
            suggested_batch_size = min(256, batch_size * 2)
        elif gpu_memory_gb > 8:   # Mid-range GPU
            suggested_batch_size = min(192, batch_size)  
        else:  # Low-end GPU
            suggested_batch_size = min(128, batch_size)
        
        if suggested_batch_size != batch_size:
            print(f"💡 GPU Memory: {gpu_memory_gb:.1f}GB, adjusting batch size: {batch_size} -> {suggested_batch_size}")
            batch_size = suggested_batch_size
    
    # Load configuration and model
    config = load_config(config_path)
    model = load_model(config['model'], model_path, device)
    
    # Create dataset
    dataset, dataset_info, enabled_datasets = create_evaluation_dataset(
        config_path, config['model'].get('num_joints', 22), device
    )
    
    dataloader = create_dataloader(dataset, batch_size, shuffle=False)
    
    print(f"🚀 Starting encoding with batch_size={batch_size}, device={device}")
    print(f"📊 Dataset: {len(dataset):,} samples")
    
    # Encode
    encoder = ModelEncoder(model, device)
    results = encoder.encode_dataset(dataloader)
    
    # Save
    tokens = results['tokens'].astype(np.int8)
    metadata = {
        'total_samples': results['total_samples'],
        'num_tokens_per_sample': tokens.shape[1] if len(tokens) > 0 else 0,
        'codebook_size': config['model'].get('token_class_num', 512),
        'dataset_name': "_".join(enabled_datasets),
        'batch_size_used': batch_size,
        'device_used': device
    }
    
    save_tokens(tokens, results['labels'], output_path, metadata)
    
    return {
        'total_samples': results['total_samples'],
        'output_path': str(output_path),
        'file_size_mb': float(Path(output_path).stat().st_size / (1024 * 1024)),
        'token_range': [int(tokens.min()), int(tokens.max())] if len(tokens) > 0 else [0, 0],
        'batch_size_used': batch_size
    }

