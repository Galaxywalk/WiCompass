# -*- coding: utf-8 -*-
"""
Pose Visualization Module

Functions for visualizing human poses with skeleton connections.
No model dependencies - works with raw pose data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .constants import (
    JOINT_NAMES,
    BONE_CONNECTIONS,
    BODY_PART_COLORS,
    BONE_PART_MAPPING,
    DEFAULT_PLOT_SETTINGS,
)


def plot_single_pose(
    joints: np.ndarray,
    ax=None,
    title: str = "",
    show_axes: bool = False,
    joint_size: int = None,
    line_width: float = None,
    elev: int = None,
    azim: int = None,
    tokens: Optional[np.ndarray] = None,
):
    """
    Plot a single pose on 3D axis.
    
    Args:
        joints: (num_joints, 3) array of joint positions
        ax: matplotlib 3D axis (creates new figure if None)
        title: Plot title
        show_axes: Whether to show axis labels and ticks
        joint_size: Size of joint markers
        line_width: Width of bone lines
        elev: Elevation angle for view
        azim: Azimuth angle for view
        tokens: Optional token sequence to display
        
    Returns:
        ax: The matplotlib axis
    """
    # Use defaults if not specified
    joint_size = joint_size or DEFAULT_PLOT_SETTINGS['joint_size']
    line_width = line_width or DEFAULT_PLOT_SETTINGS['line_width']
    elev = elev if elev is not None else DEFAULT_PLOT_SETTINGS['elev']
    azim = azim if azim is not None else DEFAULT_PLOT_SETTINGS['azim']
    
    # Convert tensor to numpy if needed
    if hasattr(joints, 'numpy'):
        joints = joints.numpy()
    elif hasattr(joints, 'cpu'):
        joints = joints.cpu().numpy()
    
    # Create figure if no axis provided
    if ax is None:
        fig = plt.figure(figsize=DEFAULT_PLOT_SETTINGS['figsize_single'])
        ax = fig.add_subplot(111, projection='3d')
    
    # Style background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
    else:
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
              c=DEFAULT_PLOT_SETTINGS['joint_color'], 
              s=joint_size, 
              alpha=DEFAULT_PLOT_SETTINGS['joint_alpha'],
              edgecolors='white', linewidth=1.5)
    
    # Plot bones with body-part colors
    for j1, j2 in BONE_CONNECTIONS:
        if j1 >= len(joints) or j2 >= len(joints):
            continue
        body_part = BONE_PART_MAPPING.get((j1, j2), 'spine')
        color = BODY_PART_COLORS[body_part]
        ax.plot([joints[j1, 0], joints[j2, 0]],
               [joints[j1, 1], joints[j2, 1]],
               [joints[j1, 2], joints[j2, 2]],
               color=color, linewidth=line_width, 
               alpha=DEFAULT_PLOT_SETTINGS['bone_alpha'])
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Equal aspect ratio
    max_range = np.array([
        joints[:, 0].max() - joints[:, 0].min(),
        joints[:, 1].max() - joints[:, 1].min(),
        joints[:, 2].max() - joints[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=elev, azim=azim)
    
    # Add token info display
    if tokens is not None:
        tokens_per_line = 16
        token_lines = []
        for i in range(0, len(tokens), tokens_per_line):
            line_tokens = tokens[i:i+tokens_per_line]
            token_lines.append(' '.join(f"{t:3d}" for t in line_tokens))
        token_text = "Tokens:\n" + '\n'.join(token_lines)
        ax.text2D(0.02, 0.98, token_text, transform=ax.transAxes,
                 fontsize=6, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    
    return ax


def plot_pose_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    original_title: str = "Original",
    reconstructed_title: str = "Reconstructed",
    show_axes: bool = False,
    **kwargs
):
    """
    Plot original and reconstructed poses side by side.
    
    Args:
        original: Original pose joints
        reconstructed: Reconstructed pose joints
        original_title: Title for original pose
        reconstructed_title: Title for reconstructed pose
        show_axes: Whether to show axis labels
        **kwargs: Additional arguments for plot_single_pose
        
    Returns:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=(16, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_single_pose(original, ax=ax1, title=original_title, show_axes=show_axes, **kwargs)
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_single_pose(reconstructed, ax=ax2, title=reconstructed_title, show_axes=show_axes, **kwargs)
    
    plt.tight_layout()
    return fig


def plot_poses_grid(
    poses: np.ndarray,
    titles: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = None,
    show_axes: bool = False,
    **kwargs
):
    """
    Plot multiple poses in a grid layout.
    
    Args:
        poses: (N, num_joints, 3) array of poses
        titles: List of titles for each pose
        n_cols: Number of columns in grid
        figsize_per_plot: Size of each subplot
        show_axes: Whether to show axis labels
        **kwargs: Additional arguments passed to plot_single_pose
        
    Returns:
        fig: matplotlib figure
    """
    figsize_per_plot = figsize_per_plot or DEFAULT_PLOT_SETTINGS['figsize_per_subplot']
    n_poses = len(poses)
    n_rows = (n_poses + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    
    for i, pose in enumerate(poses):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        title = titles[i] if titles and i < len(titles) else f"#{i}"
        plot_single_pose(pose, ax=ax, title=title, show_axes=show_axes, **kwargs)
    
    plt.tight_layout()
    return fig


def plot_multiple_poses(
    poses_data: List,
    titles: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize_per_plot: Tuple[int, int] = (4, 4),
    show_axes: bool = True
):
    """
    Plot overview of multiple poses (backward compatible alias for plot_poses_grid).
    
    Args:
        poses_data: List of (joints, tokens) tuples or List of joints arrays
        titles: List of titles (optional)
        n_cols: Number of columns per row
        figsize_per_plot: Size of each subplot
        show_axes: Whether to show axes, default True
        
    Returns:
        matplotlib figure
    """
    n_poses = len(poses_data)
    n_rows = (n_poses + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    
    for i, pose_data in enumerate(poses_data):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        
        # Handle different input formats
        if isinstance(pose_data, tuple):
            joints, tokens = pose_data
        else:
            joints, tokens = pose_data, None
        
        title = titles[i] if titles and i < len(titles) else f"Pose {i+1}"
        plot_single_pose(joints, ax=ax, title=title, tokens=tokens, show_axes=show_axes)
    
    plt.tight_layout()
    return fig


def create_color_legend(output_path: Optional[Union[str, Path]] = None):
    """
    Create a color legend showing body part colors.
    
    Args:
        output_path: Path to save the legend (optional)
        
    Returns:
        fig: matplotlib figure
    """
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    
    legend_elements = [
        Line2D([0], [0], color=color, lw=4, label=part.replace('_', ' ').title())
        for part, color in BODY_PART_COLORS.items()
    ]
    
    ax.legend(handles=legend_elements, loc='center', fontsize=12,
             title="Body Part Colors", title_fontsize=14)
    
    plt.title("Pose Visualization Color Legend", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DEFAULT_PLOT_SETTINGS['dpi'], bbox_inches='tight')
        print(f"Color legend saved: {output_path}")
    
    return fig


def save_pose_plot(fig, filepath: Union[str, Path], dpi: int = None):
    """
    Save pose figure to file.
    
    Args:
        fig: matplotlib figure
        filepath: Save path
        dpi: Image resolution
    """
    dpi = dpi or DEFAULT_PLOT_SETTINGS['dpi']
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def visualize_poses_from_file(
    poses_path: Union[str, Path],
    num_samples: int = 16,
    start_idx: int = 0,
    n_cols: int = 4,
    show_axes: bool = False,
    random_sample: bool = False,
    seed: int = 42
):
    """
    Load poses from file and visualize them.
    
    Args:
        poses_path: Path to .npy file containing poses
        num_samples: Number of poses to show
        start_idx: Starting index (ignored if random_sample=True)
        n_cols: Number of columns in grid
        show_axes: Whether to show axis labels
        random_sample: Whether to randomly sample poses
        seed: Random seed for sampling
        
    Returns:
        fig: matplotlib figure
    """
    poses = np.load(poses_path)
    total = len(poses)
    print(f"📖 Loaded: {poses.shape} (N={total}, J={poses.shape[1]}, dim={poses.shape[2]})")
    
    if random_sample:
        np.random.seed(seed)
        indices = np.random.choice(total, min(num_samples, total), replace=False)
        indices = np.sort(indices)
    else:
        end_idx = min(start_idx + num_samples, total)
        indices = np.arange(start_idx, end_idx)
    
    poses_to_show = poses[indices]
    titles = [f"#{i}" for i in indices]
    
    print(f"📊 Showing {len(poses_to_show)} poses")
    
    fig = plot_poses_grid(poses_to_show, titles=titles, n_cols=n_cols, show_axes=show_axes)
    return fig
