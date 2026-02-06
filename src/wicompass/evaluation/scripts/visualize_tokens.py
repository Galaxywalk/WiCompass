#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQ-VAE Token Visualization Script
Visualize token distributions using t-SNE/PCA dimensionality reduction

Examples:
    python -m wicompass.evaluation.scripts.visualize_tokens \
        --model work_dirs/best_model.pth \
        --config configs/joint_vae_base.json

    python -m wicompass.evaluation.scripts.visualize_tokens \
        --model work_dirs/best_model.pth \
        --config configs/joint_vae_base.json \
        --sampling-mode total \
        --max-samples 3000
"""

import argparse
from pathlib import Path
import sys

# Add src/ to Python path for absolute imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wicompass.evaluation.token_visualizer import TokenVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='VQ-VAE Token Distribution Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', required=True, help='VQ-VAE model path')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--output-dir', default='token_visualization', 
                       help='Output directory')
    parser.add_argument('--sampling-mode', default='per-dataset', 
                       choices=['per-dataset', 'total'],
                       help='Sampling mode: per-dataset or total')
    parser.add_argument('--samples-per-dataset', type=int, default=1000,
                       help='Samples per dataset (for per-dataset mode)')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Max total samples (for total mode)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Compute device')
    parser.add_argument('--method', default='tsne', choices=['tsne', 'pca'],
                       help='Dimensionality reduction method')
    parser.add_argument('--random-seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("🔍 VQ-VAE Token Distribution Visualization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Method: {args.method.upper()}")
    print(f"Sampling mode: {args.sampling_mode}")
    
    try:
        # Create visualizer
        visualizer = TokenVisualizer(
            config_path=args.config,
            model_path=args.model,
            device=args.device
        )
        
        # Create token visualization
        df = visualizer.create_token_visualization(
            output_dir=args.output_dir,
            sampling_mode=args.sampling_mode,
            samples_per_dataset=args.samples_per_dataset,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            method=args.method,
            random_seed=args.random_seed
        )
        
        print("\n✅ Token visualization complete!")
        return 0
        
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install: pip install scikit-learn plotly")
        return 1
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

