# -*- coding: utf-8 -*-
"""
Visualization Module

Model-independent visualization utilities for poses, skeletons, and metrics.

Usage:
    from wicompass.visualization import plot_single_pose, plot_poses_grid, create_color_legend
    from wicompass.visualization import plot_token_heatmap, plot_joint_errors
"""

from .constants import (
    JOINT_NAMES,
    BONE_CONNECTIONS,
    BODY_PART_COLORS,
    BONE_PART_MAPPING,
    DEFAULT_PLOT_SETTINGS,
)

from .pose import (
    plot_single_pose,
    plot_pose_comparison,
    plot_poses_grid,
    plot_multiple_poses,  # backward compatible alias
    create_color_legend,
    save_pose_plot,
    visualize_poses_from_file,
)

from .metrics import (
    plot_token_heatmap,
    plot_joint_errors,
    plot_loss_distribution,
    compute_sample_stats,
    compute_joint_stats,
)

__all__ = [
    # Constants
    'JOINT_NAMES',
    'BONE_CONNECTIONS',
    'BODY_PART_COLORS',
    'BONE_PART_MAPPING',
    'DEFAULT_PLOT_SETTINGS',
    # Pose visualization
    'plot_single_pose',
    'plot_pose_comparison',
    'plot_poses_grid',
    'plot_multiple_poses',  # backward compatible alias
    'create_color_legend',
    'save_pose_plot',
    'visualize_poses_from_file',
    # Metrics visualization
    'plot_token_heatmap',
    'plot_joint_errors',
    'plot_loss_distribution',
    'compute_sample_stats',
    'compute_joint_stats',
]
