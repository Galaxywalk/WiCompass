#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQ-VAE Model Evaluation Script
Evaluate model reconstruction quality and generate visualization reports

Examples:
    python -m wicompass.evaluation.scripts.evaluate_model \
        --model work_dirs/best_model.pth \
        --config configs/joint_vae_base.json

    python -m wicompass.evaluation.scripts.evaluate_model \
        --model work_dirs/best_model.pth \
        --config configs/joint_vae_base.json \
        --output-dir results/
"""

import argparse
from pathlib import Path
import sys

# Add src/ to Python path for absolute imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wicompass.evaluation.evaluator import evaluate_model, create_evaluation_report


def main():
    parser = argparse.ArgumentParser(
        description='VQ-VAE Model Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', required=True, help='Model weights path')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--output-dir', default='evaluation_results', 
                       help='Output directory for evaluation results')
    parser.add_argument('--test-ratio', type=float, default=0.1, 
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Compute device')
    
    args = parser.parse_args()
    
    print("🚀 VQ-VAE Model Evaluation")
    print("=" * 60)
    
    try:
        # Run evaluation
        print("📈 Running model evaluation...")
        eval_results = evaluate_model(
            model_path=args.model,
            config_path=args.config,
            test_ratio=args.test_ratio,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Generate report
        print("📊 Generating evaluation report...")
        report = create_evaluation_report(eval_results, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 Evaluation Results Summary")
        print("=" * 60)
        
        avg_losses = eval_results['avg_losses']
        print(f"📊 Datasets: {eval_results.get('enabled_datasets', [])}")
        print(f"📈 Test samples: {eval_results['total_samples']}")
        print(f"🎯 Average losses:")
        print(f"   - Reconstruction loss: {avg_losses['recon']:.6f}")
        print(f"   - VQ loss: {avg_losses['vq']:.6f}")
        print(f"   - Total loss: {avg_losses['total']:.6f}")
        
        sample_stats = report['model_evaluation']['sample_statistics']
        print(f"\n📊 Sample-level analysis:")
        print(f"   - Mean error: {sample_stats['mean_loss']:.6f}")
        print(f"   - Std dev: {sample_stats['std_loss']:.6f}")
        print(f"   - 95th percentile: {sample_stats['q95_loss']:.6f}")
        print(f"   - Outliers: {sample_stats['outliers']['count']} "
              f"({sample_stats['outliers']['percentage']:.1f}%)")
        
        # Joint analysis
        joint_stats = report['model_evaluation']['joint_statistics']
        if joint_stats:
            print(f"\n🦴 Joint analysis (Top-5 highest errors):")
            sorted_joints = sorted(
                joint_stats.items(), 
                key=lambda x: x[1]['mean_error'], 
                reverse=True
            )
            for i, (joint_name, stats) in enumerate(sorted_joints[:5]):
                print(f"   {i+1}. {joint_name}: {stats['mean_error']:.6f}")
        
        print(f"\n📁 Detailed report saved to: {args.output_dir}")
        print("=" * 60)
        print("✅ Evaluation complete!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

