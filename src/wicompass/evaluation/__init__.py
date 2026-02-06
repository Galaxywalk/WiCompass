# -*- coding: utf-8 -*-
"""
VQ-VAE Model Evaluation Package

Model-aware evaluation, encoding, and visualization.
For model-independent visualization utilities, use wicompass.visualization.
"""

# Core utilities
from .core import (
    # Configuration
    load_config,
    get_dataset_configs,
    # Dataset utilities
    create_dataloader,
    split_dataset,
    create_evaluation_dataset,
    # Device utilities
    get_device,
    # Model loading
    load_model,
    # Base class
    BaseVisualizer,
)

# Evaluator
from .evaluator import (
    ModelEvaluator,
    analyze_joint_errors,
    analyze_sample_errors,
    evaluate_model,
    create_evaluation_report,
)

# Encoder
from .encoder import (
    ModelEncoder,
    save_tokens,
    load_tokens,
    encode_dataset,
)

# Pose visualizer
from .pose_visualizer import (
    PoseVisualizer,
    visualize_dataset_poses,
)

# Token visualizer
from .token_visualizer import (
    TokenVisualizer,
    extract_token_representations,
    apply_dimensionality_reduction,
    create_token_distribution_visualization,
)

# Re-export visualization functions and constants for convenience
from wicompass.visualization import (
    # Constants
    JOINT_NAMES,
    BONE_CONNECTIONS,
    BODY_PART_COLORS,
    BONE_PART_MAPPING,
    DEFAULT_PLOT_SETTINGS,
    # Pose visualization
    plot_single_pose,
    plot_pose_comparison,
    plot_poses_grid,
    plot_multiple_poses,
    create_color_legend,
    save_pose_plot,
    visualize_poses_from_file,
    # Metrics visualization
    plot_token_heatmap,
    plot_joint_errors,
    plot_loss_distribution,
    compute_sample_stats,
    compute_joint_stats,
)

__all__ = [
    # Core
    'load_config',
    'get_dataset_configs',
    'create_dataloader',
    'split_dataset',
    'create_evaluation_dataset',
    'get_device',
    'load_model',
    'BaseVisualizer',
    
    # Evaluator
    'ModelEvaluator',
    'analyze_joint_errors',
    'analyze_sample_errors',
    'evaluate_model',
    'create_evaluation_report',
    
    # Encoder
    'ModelEncoder',
    'save_tokens',
    'load_tokens',
    'encode_dataset',
    
    # Pose visualizer
    'PoseVisualizer',
    'visualize_dataset_poses',
    
    # Token visualizer
    'TokenVisualizer',
    'extract_token_representations',
    'apply_dimensionality_reduction',
    'create_token_distribution_visualization',
    
    # Re-exports from visualization
    'JOINT_NAMES',
    'BONE_CONNECTIONS',
    'BODY_PART_COLORS',
    'BONE_PART_MAPPING',
    'DEFAULT_PLOT_SETTINGS',
    'plot_single_pose',
    'plot_pose_comparison',
    'plot_poses_grid',
    'plot_multiple_poses',
    'create_color_legend',
    'save_pose_plot',
    'visualize_poses_from_file',
    'plot_token_heatmap',
    'plot_joint_errors',
    'plot_loss_distribution',
    'compute_sample_stats',
    'compute_joint_stats',
]
