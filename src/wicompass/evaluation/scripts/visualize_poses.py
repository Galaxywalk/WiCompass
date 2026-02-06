#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQ-VAE Pose Visualization Script
Unified pose visualization script with optional model reconstruction

Examples:
    # Simple visualization (no model required)
    python -m wicompass.evaluation.scripts.visualize_poses --config configs/joint_vae_base.json --simple

    # Full visualization with reconstruction comparison (requires model)
    python -m wicompass.evaluation.scripts.visualize_poses --config configs/joint_vae_base.json --model work_dirs/best_model.pth

    # Customize output
    python -m wicompass.evaluation.scripts.visualize_poses --config configs/joint_vae_base.json --model work_dirs/best_model.pth --num-samples 12 --output-dir pose_results
"""

import argparse
from pathlib import Path
import sys

# Add src/ to Python path for absolute imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wicompass.evaluation.pose_visualizer import PoseVisualizer, visualize_dataset_poses


def main():
    parser = argparse.ArgumentParser(
        description='VQ-VAE Pose Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Simple visualization (no model required)
  python -m wicompass.evaluation.scripts.visualize_poses --config config.json --simple

  # With reconstruction comparison
  python -m wicompass.evaluation.scripts.visualize_poses --config config.json --model model.pth
        '''
    )
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model', help='Model weights path (optional, enables reconstruction comparison)')
    parser.add_argument('--output-dir', default='pose_visualizations', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--device', default='cuda', help='Compute device')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--simple', action='store_true', 
                       help='Simple mode: visualize raw poses without model (ignores --model)')
    parser.add_argument('--no-reconstruction', action='store_true', 
                       help='Do not show reconstruction results even if model is provided')
    parser.add_argument('--no-individual', action='store_true', 
                       help='Do not save individual sample images')
    parser.add_argument('--hide-axes', action='store_true', help='Hide coordinate axes in plots')
    
    args = parser.parse_args()
    
    print("🎨 VQ-VAE Pose Visualization")
    print("=" * 60)
    print(f"Config: {args.config}")
    
    # Simple mode: just visualize dataset poses without model
    if args.simple:
        print("Mode: Simple (no model)")
        visualize_dataset_poses(
            config_path=args.config,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            show_axes=not args.hide_axes
        )
        print("✅ Visualization complete!")
        return 0
    
    # Full mode: use PoseVisualizer with optional model
    if args.model:
        print(f"Model: {args.model}")
    else:
        print("Mode: Dataset only (no reconstruction)")
    
    visualizer = PoseVisualizer(
        config_path=args.config,
        model_path=args.model,
        device=args.device
    )
    
    # Load dataset samples
    samples = visualizer.load_dataset_samples(
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        extract_tokens=args.model is not None
    )
    
    # Print sample info
    print(f"\n📋 Selected {len(samples)} samples:")
    for i, (joints, label, dataset_name, sample_idx, tokens) in enumerate(samples):
        token_info = f", tokens: {len(tokens)}" if tokens is not None else ""
        print(f"  {i+1}. Sample {sample_idx} from {dataset_name} (shape: {joints.shape}{token_info})")
    
    # Create visualization
    show_recon = not args.no_reconstruction and args.model is not None
    visualizer.visualize_sample_poses(
        samples,
        output_dir=args.output_dir,
        show_reconstruction=show_recon,
        save_individual=not args.no_individual
    )
    
    print("\n✅ Pose visualization complete!")
    return 0


if __name__ == "__main__":
    exit(main())

