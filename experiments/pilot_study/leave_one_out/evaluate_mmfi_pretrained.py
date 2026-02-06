#!/usr/bin/env python3
"""
Evaluate pre-trained MMFi model on each action.

This generates baseline data for comparison with leave-one-out training results.

Usage:
    python evaluate_mmfi_pretrained.py [--config_path PATH] [--checkpoint_path PATH]
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_estimation.evaluate import Evaluator


# Default paths
DEFAULT_CONFIG = Path(__file__).parent / "mmfi_leave_one_out_config.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / \
    "logs/pose_estimation/mmfi/leave_one_out/checkpoints/PointTransformer-baseline_all_actions_repeat00-2025-12-20_14-55-53/best_model.pth"

# MMFi actions (A01-A27)
ALL_ACTIONS = [f'A{i:02d}' for i in range(1, 28)]


def main():
    parser = argparse.ArgumentParser(description='Evaluate pre-trained MMFi model on each action')
    parser.add_argument('--config_path', type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint_path', type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert relative dataset_path to absolute path
    if not os.path.isabs(config['dataset_path']):
        config['dataset_path'] = str(PROJECT_ROOT / config['dataset_path'])
    
    # Override val_dataset to only use E04 (standard test set) for faster evaluation
    config['val_dataset'] = {
        'scenes': ['E04'],  # Only test set
        'subjects': 'all',
        'actions': 'all',
    }
    
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset path: {config['dataset_path']}")
    print(f"Test scenes: {config['val_dataset']['scenes']}")
    
    # Create evaluator
    evaluator = Evaluator(
        config=config,
        checkpoint_path=args.checkpoint_path,
        device='cuda',
        batch_size=16,
    )
    
    # Evaluate each action on test set (E04 only)
    print(f"\n=== Evaluating {len(ALL_ACTIONS)} Actions on E04 (10 subjects) ===")
    action_results = evaluator.evaluate_actions(ALL_ACTIONS)
    
    for r in action_results:
        print(f"{r['action']}: MPJPE={r['mpjpe_mean']:.2f}mm, Samples={r['num_samples']}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save JSON (for analyze_results.ipynb)
    output = {
        'actions': action_results,
        'model_info': {
            'checkpoint_path': args.checkpoint_path,
            'config_path': args.config_path,
        }
    }
    output_path = os.path.join(args.output_dir, 'mmfi_pretrained_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n=== Summary ===")
    mpjpes = [r['mpjpe_mean'] for r in action_results]
    print(f"Actions: {len(action_results)} evaluated")
    
    if len(mpjpes) > 0:
        print(f"MPJPE: {np.mean(mpjpes):.2f} ± {np.std(mpjpes):.2f} mm")
        print(f"Range: {np.min(mpjpes):.2f} - {np.max(mpjpes):.2f} mm")
    else:
        print("WARNING: No actions were evaluated! Check dataset path and configuration.")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
