#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token to Pose Conversion Tool

Converts sampled tokens (from any sampling method: FPS, PPS, random, etc.) 
to human poses using a VQ-VAE model's codebook and decoder.

Usage:
    python src/wicompass/token_space_sampling/convert_sampled_tokens_to_poses.py \
        --tokens-npy logs/wicompass/sampled_tokens/pps_sampling_k12_quantile85/capped_pps_selected_vectors.npy \
        --model logs/vqvae/vqvae_tokennum16_tokenclass64/best_model.pth \
        --config src/wicompass/configs/joint_vae_base_tokennum16_tokenclass64.json \
        --output-dir logs/wicompass/sampled_poses/pps_sampling_k12_quantile85
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Add src directory to path for direct script execution
_src_dir = str(Path(__file__).parent.parent.parent)  # src/wicompass/token_space_sampling -> src
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wicompass.evaluation.core import load_config, load_model


# =============================================================================
# Data Loading
# =============================================================================

def load_tokens(tokens_path: Path) -> np.ndarray:
    """
    Load sampled tokens from file.
    
    Args:
        tokens_path: Path to tokens file (.npy format)
        
    Returns:
        tokens: (N, token_num) array of token indices
    """
    tokens = np.load(tokens_path)
    print(f"📖 Loaded tokens: {tokens.shape}, dtype: {tokens.dtype}")
    
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D tokens array (N, token_num), got shape: {tokens.shape}")
    
    return tokens


def load_vqvae_model(config_path: Path, model_path: Path, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict]:
    """
    Load VQ-VAE model using the unified interface from evaluation.core.
    
    Args:
        config_path: Path to model config file
        model_path: Path to model weights file
        device: Computation device
        
    Returns:
        model: Loaded VQ-VAE model
        config: Full configuration dict
    """
    config = load_config(str(config_path))
    model_config = config.get('model', config)  # Support both nested and flat configs
    model = load_model(model_config, str(model_path), device)
    
    print(f"📋 Model config: token_num={model_config.get('token_num')}, "
          f"token_class_num={model_config.get('token_class_num')}, "
          f"num_joints={model_config.get('num_joints')}")
    
    return model, config


# =============================================================================
# Token to Pose Conversion
# =============================================================================

def tokens_to_poses(
    tokens: np.ndarray, 
    model: torch.nn.Module, 
    device: str = 'cuda', 
    batch_size: int = 64
) -> np.ndarray:
    """
    Convert token sequences to poses using VQ-VAE model.
    
    The conversion process:
    1. Look up codebook vectors for each token index
    2. Decode quantized features to poses using model.decode()
    
    Args:
        tokens: (N, token_num) token index array
        model: VQ-VAE model with codebook and decode() method
        device: Computation device
        batch_size: Batch size for processing
        
    Returns:
        poses: (N, num_joints, joint_dim) pose array
    """
    N = len(tokens)
    model.eval()
    poses_list = []
    
    print(f"🔄 Converting {N} token sequences to poses...")
    
    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Converting"):
            batch_tokens = tokens[start:start + batch_size]
            batch_tensor = torch.from_numpy(batch_tokens).long().to(device)
            
            # Look up codebook vectors: (B, token_num) -> (B, token_num, token_dim)
            quantized = torch.nn.functional.embedding(batch_tensor, model.codebook)
            
            # Decode to poses: (B, token_num, token_dim) -> (B, num_joints, joint_dim)
            poses = model.decode(quantized)
            poses_list.append(poses.cpu().numpy())
    
    all_poses = np.concatenate(poses_list, axis=0)
    print(f"✅ Conversion completed: {all_poses.shape}")
    return all_poses


# =============================================================================
# Output Saving
# =============================================================================

def compute_pose_stats(poses: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute statistics for converted poses."""
    stats = {
        'num_samples': int(poses.shape[0]),
        'num_joints': int(poses.shape[1]),
        'joint_dim': int(poses.shape[2]),
        'poses_range': {
            'min': float(poses.min()),
            'max': float(poses.max()),
            'mean': float(poses.mean()),
            'std': float(poses.std())
        },
        'per_axis_stats': {
            axis: [float(poses[:, :, i].min()), float(poses[:, :, i].max())]
            for i, axis in enumerate(['x', 'y', 'z'][:poses.shape[2]])
        }
    }
    
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        stats['label_distribution'] = {str(u): int(c) for u, c in zip(unique, counts)}
    
    return stats


def save_poses(
    poses: np.ndarray, 
    output_dir: Path, 
    labels: Optional[np.ndarray] = None,
    format_type: str = 'npy'
) -> None:
    """
    Save converted poses and statistics.
    
    Args:
        poses: (N, num_joints, joint_dim) pose array
        output_dir: Output directory
        labels: Optional (N,) labels array
        format_type: Output format ('npy' or 'h5')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'npy':
        np.save(output_dir / "converted_poses.npy", poses)
        print(f"✅ Poses saved: {output_dir / 'converted_poses.npy'}")
        
        if labels is not None:
            np.save(output_dir / "converted_labels.npy", labels)
            print(f"✅ Labels saved: {output_dir / 'converted_labels.npy'}")
    
    elif format_type == 'h5':
        try:
            import h5py
            h5_path = output_dir / "converted_poses.h5"
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('poses', data=poses, compression='gzip')
                if labels is not None:
                    f.create_dataset('labels', data=labels, compression='gzip')
                f.attrs['num_samples'] = poses.shape[0]
                f.attrs['num_joints'] = poses.shape[1]
                f.attrs['joint_dim'] = poses.shape[2]
                f.attrs['description'] = 'Poses converted from sampled tokens'
            
            print(f"✅ Poses saved (H5): {h5_path}")
            
        except ImportError:
            print("⚠️  h5py not available, falling back to npy format")
            save_poses(poses, output_dir, labels, 'npy')
            return
    
    # Save statistics
    stats = compute_pose_stats(poses, labels)
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved: {stats_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert sampled tokens to poses using VQ-VAE model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert FPS-sampled tokens
  python convert_further_sampling_to_pose.py \\
      --tokens-npy fps_selected_vectors.npy --model model.pth --config config.json
  
  # Convert with labels and save as H5
  python convert_further_sampling_to_pose.py \\
      --tokens-npy tokens.npy --labels-npy labels.npy \\
      --model model.pth --config config.json --format h5
        """
    )
    
    # Input files
    parser.add_argument('--tokens-npy', required=True, 
                        help='Path to sampled tokens (.npy file)')
    parser.add_argument('--labels-npy', 
                        help='Path to corresponding labels (.npy file, optional)')
    
    # Model
    parser.add_argument('--model', required=True, 
                        help='Path to VQ-VAE model (.pth file)')
    parser.add_argument('--config', required=True,
                        help='Path to model config (.json file)')
    
    # Output
    parser.add_argument('--output-dir', default='converted_poses',
                        help='Output directory for poses (default: converted_poses)')
    parser.add_argument('--format', default='npy', choices=['npy', 'h5'],
                        help='Output format (default: npy)')
    
    # Computation
    parser.add_argument('--device', default='cuda', 
                        help='Device for computation (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for conversion (default: 64)')
    
    args = parser.parse_args()
    
    # Header
    print("🔄 Token → Pose Conversion")
    print("=" * 60)
    print(f"Tokens: {args.tokens_npy}")
    print(f"Model:  {args.model}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    try:
        # Check CUDA availability
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            args.device = 'cpu'
        
        # Load tokens
        tokens_path = Path(args.tokens_npy)
        if not tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {tokens_path}")
        tokens = load_tokens(tokens_path)
        
        # Load labels (optional)
        labels = None
        if args.labels_npy:
            labels_path = Path(args.labels_npy)
            if labels_path.exists():
                labels = np.load(labels_path)
                print(f"📖 Loaded labels: {labels.shape}")
                if len(labels) != len(tokens):
                    print(f"⚠️  Size mismatch: tokens={len(tokens)}, labels={len(labels)}")
            else:
                print(f"⚠️  Labels file not found: {labels_path}")
        
        # Load model
        model_path = Path(args.model)
        config_path = Path(args.config)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        model, config = load_vqvae_model(config_path, model_path, args.device)
        
        # Validate token dimension
        model_config = config.get('model', config)
        expected_token_num = model_config.get('token_num', 64)
        if tokens.shape[1] != expected_token_num:
            raise ValueError(
                f"Token dimension mismatch: model expects {expected_token_num}, "
                f"got {tokens.shape[1]}"
            )
        
        # Convert tokens to poses
        poses = tokens_to_poses(tokens, model, args.device, args.batch_size)
        
        # Save results
        output_dir = Path(args.output_dir)
        save_poses(poses, output_dir, labels, args.format)
        
        print(f"\n✅ Conversion completed!")
        print(f"📁 Output: {output_dir.resolve()}")
        print(f"📊 Converted {len(poses)} token sequences to poses")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
