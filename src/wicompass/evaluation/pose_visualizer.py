# -*- coding: utf-8 -*-
"""
Pose Visualizer Module (Model-dependent)

Model-aware pose visualization for reconstruction comparison.
Reuses base visualization functions from wicompass.visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import random

from .core import load_config, get_dataset_configs, BaseVisualizer
from wicompass.dataset import create_dataset
from wicompass.visualization import (
    plot_single_pose,
    plot_pose_comparison,
    create_color_legend,
    save_pose_plot,
)


__all__ = [
    'PoseVisualizer',
    'visualize_dataset_poses',
]


def visualize_dataset_poses(
    config_path: str, 
    num_samples: int = 6, 
    output_dir: str = "simple_poses", 
    random_seed: int = 42, 
    show_axes: bool = True
):
    """
    Simple dataset pose visualization (no model required)
    
    Args:
        config_path: Config file path
        num_samples: Number of samples
        output_dir: Output directory
        random_seed: Random seed
        show_axes: Whether to show axes, default True
    """
    print(f"📁 Loading config from {config_path}")
    config = load_config(config_path)
    model_cfg = config['model']
    data_cfg = config['data']
    
    # Create dataset configuration
    dataset_configs = get_dataset_configs(data_cfg)
    enabled_datasets = [cfg['name'] for cfg in dataset_configs]
    print(f"Enabled datasets: {enabled_datasets}")
    
    num_joints = model_cfg.get('num_joints', 22)
    master_dataset, dataset_info = create_dataset(dataset_configs, num_joints, 'cpu')
    
    print(f"Total samples: {len(master_dataset)}")
    
    # Random sampling
    random.seed(random_seed)
    sample_indices = random.sample(range(len(master_dataset)), min(num_samples, len(master_dataset)))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get samples
    samples = []
    label_to_name = dataset_info.get('label_to_name', {})
    
    for idx in sample_indices:
        joints, label = master_dataset[idx]
        label_value = label.item() if hasattr(label, 'item') else label
        dataset_name = label_to_name.get(label_value, f"Dataset_{label_value}")
        samples.append((joints.numpy(), dataset_name, idx))
    
    print(f"✅ Selected {len(samples)} samples")
    
    # Create overview plot
    n_cols = min(3, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    
    for i, (joints, dataset_name, sample_idx) in enumerate(samples):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        title = f"{dataset_name}\n(Sample {sample_idx})"
        plot_single_pose(joints, ax, title=title, show_axes=show_axes)
    
    plt.tight_layout()
    overview_file = output_path / "pose_samples_overview.png"
    fig.savefig(overview_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual images
    for i, (joints, dataset_name, sample_idx) in enumerate(samples):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        title = f"Sample {sample_idx} from {dataset_name}"
        plot_single_pose(joints, ax, title=title, show_axes=show_axes)
        
        individual_file = output_path / f"pose_{i+1:02d}_{dataset_name}_{sample_idx}.png"
        fig.savefig(individual_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {individual_file.name}")
    
    # Save color legend
    legend_path = output_path / "pose_color_legend.png"
    create_color_legend(legend_path)
    
    print(f"📈 Overview saved: {overview_file}")
    print(f"📁 All files saved to: {output_path}")


class PoseVisualizer(BaseVisualizer):
    """Human pose visualizer with optional reconstruction comparison"""
    
    def __init__(self, config_path: str, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize pose visualizer.
        
        Args:
            config_path: Path to config file
            model_path: Path to model checkpoint (optional, enables reconstruction)
            device: Compute device ('cuda' or 'cpu')
        """
        super().__init__(config_path, model_path, device)
    
    def load_dataset_samples(
        self, 
        num_samples: int = 24, 
        random_seed: int = 42, 
        extract_tokens: bool = True
    ) -> List[Tuple]:
        """
        Load samples from dataset.
        
        Args:
            num_samples: Number of samples to load
            random_seed: Random seed for reproducibility
            extract_tokens: Whether to extract token sequence (requires model)
            
        Returns:
            List of (joints, label, dataset_name, sample_idx, tokens) tuples
        """
        print(f"📊 Loading dataset samples...")
        
        # Create dataset configuration
        dataset_configs = self.get_dataset_configs()
        enabled_datasets = [cfg['name'] for cfg in dataset_configs]
        print(f"Enabled datasets: {enabled_datasets}")
        
        # Create dataset
        num_joints = self.model_cfg.get('num_joints', 22)
        master_dataset, dataset_info = create_dataset(
            dataset_configs,
            num_joints,
            'cpu'  # Load on CPU to save GPU memory
        )
        
        print(f"Total samples in dataset: {len(master_dataset)}")
        
        # Random sampling
        random.seed(random_seed)
        sample_indices = random.sample(range(len(master_dataset)), min(num_samples, len(master_dataset)))
        
        samples = []
        label_to_name = dataset_info.get('label_to_name', {})
        
        for idx in sample_indices:
            joints, label = master_dataset[idx]
            label_value = label.item() if hasattr(label, 'item') else label
            dataset_name = label_to_name.get(label_value, f"Dataset_{label_value}")
            
            tokens = None
            if extract_tokens and self.model is not None:
                with torch.no_grad():
                    joints_tensor = torch.from_numpy(joints.numpy()).float().unsqueeze(0).to(self.device)
                    joints_visible = torch.ones(joints_tensor.shape[:2], device=self.device).bool()
                    encoding_indices, _, _ = self.model.encode(joints_tensor, joints_visible, train=False)
                    tokens = encoding_indices.cpu().numpy().flatten()
            
            samples.append((joints.numpy(), label_value, dataset_name, idx, tokens))
        
        print(f"✅ Loaded {len(samples)} samples" + (f" with tokens" if extract_tokens and self.model else ""))
        return samples
    
    def visualize_sample_poses(
        self, 
        samples: List[Tuple], 
        output_dir: str = "pose_visualizations", 
        show_reconstruction: bool = True, 
        save_individual: bool = True
    ):
        """
        Visualize poses of multiple samples
        
        Args:
            samples: List of (joints, label, dataset_name, sample_idx, tokens) tuples
            output_dir: Output directory
            show_reconstruction: Whether to show reconstruction results
            save_individual: Whether to save individual sample images
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"🎨 Visualizing {len(samples)} pose samples...")
        
        # Create color legend
        legend_path = output_path / "pose_color_legend.png"
        create_color_legend(legend_path)
        
        # Create individual visualization for each sample
        if save_individual:
            for i, (joints, label, dataset_name, sample_idx, tokens) in enumerate(samples):
                title = f"Sample {sample_idx} from {dataset_name}"
                
                if show_reconstruction and self.model is not None:
                    with torch.no_grad():
                        joints_tensor = torch.from_numpy(joints).float().unsqueeze(0).to(self.device)
                        joints_visible = torch.ones(joints_tensor.shape[:2], device=self.device).bool()
                        recovered_joints, _, _ = self.model(joints=joints_tensor, joints_visible=joints_visible, train=False)
                        recovered_joints = recovered_joints.cpu().numpy()[0]
                    
                    fig = plot_pose_comparison(
                        joints, recovered_joints,
                        f"{title} - Original", f"{title} - Reconstructed"
                    )
                else:
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    plot_single_pose(joints, ax, title=f"{title} - Original", tokens=tokens)
                
                filename = f"pose_sample_{i:02d}_{dataset_name}_{sample_idx}.png"
                save_pose_plot(fig, output_path / filename)
                print(f"  ✅ Saved: {filename}")
        
        # Create overview plot
        self._create_overview_plot(samples, output_path, show_reconstruction)
        
        print(f"📁 All visualizations saved to: {output_path}")
    
    def _create_overview_plot(self, samples: List[Tuple], output_path: Path, show_reconstruction: bool):
        """Create multi-sample overview plot"""
        n_samples = len(samples)
        
        if show_reconstruction and self.model is not None:
            n_cols = min(3, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols * 2
            
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            
            for i, (joints, label, dataset_name, sample_idx, tokens) in enumerate(samples):
                row = (i // n_cols) * 2
                col = i % n_cols
                
                subplot_idx1 = row * n_cols + col + 1
                ax1 = fig.add_subplot(n_rows, n_cols, subplot_idx1, projection='3d')
                plot_single_pose(joints, ax1, title=f"Original - {dataset_name}", tokens=tokens)
                
                subplot_idx2 = (row + 1) * n_cols + col + 1
                ax2 = fig.add_subplot(n_rows, n_cols, subplot_idx2, projection='3d')
                
                with torch.no_grad():
                    joints_tensor = torch.from_numpy(joints).float().unsqueeze(0).to(self.device)
                    joints_visible = torch.ones(joints_tensor.shape[:2], device=self.device).bool()
                    recovered_joints, _, _ = self.model(joints=joints_tensor, joints_visible=joints_visible, train=False)
                    recovered_joints = recovered_joints.cpu().numpy()[0]
                
                plot_single_pose(recovered_joints, ax2, title=f"Reconstructed - {dataset_name}", tokens=tokens)
        
        else:
            n_cols = min(4, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            
            for i, (joints, label, dataset_name, sample_idx, tokens) in enumerate(samples):
                ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                title = f"{dataset_name}\n(Sample {sample_idx})"
                plot_single_pose(joints, ax, title=title, tokens=tokens)
        
        plt.tight_layout()
        filename = "pose_overview.png"
        save_pose_plot(fig, output_path / filename)
        print(f"  ✅ Overview plot saved: {filename}")

