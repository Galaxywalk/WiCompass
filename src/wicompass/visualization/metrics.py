# -*- coding: utf-8 -*-
"""
Metrics Visualization Module

Functions for visualizing evaluation metrics, errors, and distributions.
No model dependencies - works with raw metric data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Union

from .constants import JOINT_NAMES


def plot_token_heatmap(
    token_code_counter: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (20, 10),
    cmap: str = 'viridis'
):
    """
    Plot Token-Codebook usage heatmap.
    
    Args:
        token_code_counter: (token_num, codebook_size) usage count matrix
        save_path: Path to save figure (optional)
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Log transform for better visualization
    log_counter = np.log1p(token_code_counter)
    usage_rate = log_counter / (log_counter.sum(axis=1, keepdims=True) + 1e-10)
    
    im = plt.imshow(usage_rate, cmap=cmap, aspect='auto', vmin=0)
    plt.colorbar(im, label='Log-normalized Usage Rate')
    
    plt.xlabel(f"Codebook Index (0~{token_code_counter.shape[1]-1})")
    plt.ylabel("Token Position")
    plt.title("Token-Codebook Usage Heatmap")
    
    # Add statistics info
    total_usage = np.sum(token_code_counter)
    max_usage = np.max(token_code_counter)
    min_nonzero = np.min(token_code_counter[token_code_counter > 0]) if np.any(token_code_counter > 0) else 0
    
    stats_text = f"Total: {total_usage:,}\nMax: {max_usage:,}\nMin(>0): {min_nonzero:,}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Token heatmap saved: {save_path}")
    
    return fig


def plot_joint_errors(
    joint_stats: Dict,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (15, 8)
):
    """
    Plot per-joint reconstruction error distribution.
    
    Args:
        joint_stats: Dict with joint names as keys, containing 'mean_error' and 'std_error'
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        fig: matplotlib figure
    """
    joint_names = list(joint_stats.keys())
    mean_errors = [joint_stats[name]['mean_error'] for name in joint_names]
    std_errors = [joint_stats[name]['std_error'] for name in joint_names]
    
    fig = plt.figure(figsize=figsize)
    x_pos = np.arange(len(joint_names))
    
    plt.bar(x_pos, mean_errors, yerr=std_errors, capsize=5, alpha=0.7, color='steelblue')
    plt.xlabel('Joints')
    plt.ylabel('Mean Reconstruction Error (MSE)')
    plt.title('Joint Reconstruction Error Distribution')
    plt.xticks(x_pos, joint_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Joint error plot saved: {save_path}")
    
    return fig


def plot_loss_distribution(
    sample_losses: np.ndarray,
    sample_stats: Optional[Dict] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 12)
):
    """
    Plot sample loss distribution with multiple views.
    
    Args:
        sample_losses: Array of per-sample losses
        sample_stats: Optional pre-computed statistics dict
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        fig: matplotlib figure
    """
    # Compute stats if not provided
    if sample_stats is None:
        sample_stats = compute_sample_stats(sample_losses)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Histogram
    ax1 = axes[0, 0]
    n_bins = min(50, len(sample_losses) // 10)
    ax1.hist(sample_losses, bins=n_bins, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(sample_stats['mean_loss'], color='red', linestyle='--', 
                label=f'Mean: {sample_stats["mean_loss"]:.6f}')
    ax1.axvline(sample_stats['median_loss'], color='green', linestyle='--', 
                label=f'Median: {sample_stats["median_loss"]:.6f}')
    ax1.set_xlabel('Reconstruction Loss (MSE)')
    ax1.set_ylabel('Count')
    ax1.set_title('Sample Loss Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    box_data = ax2.boxplot(sample_losses, patch_artist=True)
    box_data['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Reconstruction Loss (MSE)')
    ax2.set_title('Sample Loss Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_losses = np.sort(sample_losses)
    cumulative_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    ax3.plot(sorted_losses, cumulative_prob, linewidth=2, color='steelblue')
    ax3.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95th percentile')
    ax3.axvline(sample_stats['q95_loss'], color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Reconstruction Loss (MSE)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics info
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""Sample Statistics:
Total: {sample_stats['total_samples']:,}
Mean: {sample_stats['mean_loss']:.6f}
Std: {sample_stats['std_loss']:.6f}
Min: {sample_stats['min_loss']:.6f}
Max: {sample_stats['max_loss']:.6f}
Median: {sample_stats['median_loss']:.6f}
95th: {sample_stats['q95_loss']:.6f}
99th: {sample_stats['q99_loss']:.6f}"""
    
    if 'outliers' in sample_stats:
        stats_text += f"\nOutliers: {sample_stats['outliers']['count']} ({sample_stats['outliers']['percentage']:.1f}%)"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             verticalalignment='top', fontsize=12, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss distribution plot saved: {save_path}")
    
    return fig


def compute_sample_stats(sample_losses: np.ndarray) -> Dict:
    """
    Compute statistics for sample losses.
    
    Args:
        sample_losses: Array of per-sample losses
        
    Returns:
        Dict with statistics
    """
    q95 = np.percentile(sample_losses, 95)
    q99 = np.percentile(sample_losses, 99)
    iqr = np.percentile(sample_losses, 75) - np.percentile(sample_losses, 25)
    outlier_threshold = np.percentile(sample_losses, 75) + 1.5 * iqr
    outliers = sample_losses > outlier_threshold
    
    return {
        'total_samples': len(sample_losses),
        'mean_loss': float(np.mean(sample_losses)),
        'std_loss': float(np.std(sample_losses)),
        'min_loss': float(np.min(sample_losses)),
        'max_loss': float(np.max(sample_losses)),
        'median_loss': float(np.median(sample_losses)),
        'q95_loss': float(q95),
        'q99_loss': float(q99),
        'outliers': {
            'count': int(np.sum(outliers)),
            'percentage': float(100 * np.mean(outliers)),
            'threshold': float(outlier_threshold)
        }
    }


def compute_joint_stats(joint_losses: np.ndarray, joint_names: list = None) -> Dict:
    """
    Compute per-joint error statistics.
    
    Args:
        joint_losses: (num_samples, num_joints) array of per-joint losses
        joint_names: List of joint names (defaults to JOINT_NAMES)
        
    Returns:
        Dict with per-joint statistics
    """
    joint_names = joint_names or JOINT_NAMES
    num_joints = joint_losses.shape[1]
    
    stats = {}
    for i in range(min(num_joints, len(joint_names))):
        name = joint_names[i]
        losses = joint_losses[:, i]
        stats[name] = {
            'mean_error': float(np.mean(losses)),
            'std_error': float(np.std(losses)),
            'min_error': float(np.min(losses)),
            'max_error': float(np.max(losses)),
        }
    
    return stats

