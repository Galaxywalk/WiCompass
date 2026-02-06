#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch evaluate all VQ-VAE models on MMBody dataset.

Usage:
    conda activate wicompass
    python experiments/vqvae_mircobenchmark/batch_evaluate_models.py
"""

import json
import time
from pathlib import Path
import sys
import torch

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from wicompass.evaluation.evaluator import evaluate_model

# VQ-VAE models are stored in logs/vqvae/
VQVAE_LOGS = project_root / 'logs' / 'vqvae'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_all_models():
    """Evaluate all vqvae_* models on MMBody dataset."""
    print(f"Device: {DEVICE}")
    print(f"Scanning: {VQVAE_LOGS}\n" + "=" * 60)
    
    for model_dir in sorted(VQVAE_LOGS.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith('vqvae_'):
            continue
        
        # Find model weights and config
        model_path = model_dir / 'best_model.pth'
        if not model_path.exists():
            model_path = model_dir / 'final_model.pth'
        config_path = model_dir / 'config.json'
        
        if not model_path.exists() or not config_path.exists():
            print(f"[SKIP] {model_dir.name}: missing files")
            continue
        
        print(f"[EVAL] {model_dir.name}...", end=" ", flush=True)
        
        # Create MMBody-only config
        with open(config_path) as f:
            config = json.load(f)
        config['data']['datasets'] = ['MMBody']
        test_config = model_dir / 'test_mmbody_config.json'
        with open(test_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run evaluation
        results = evaluate_model(
            model_path=str(model_path),
            config_path=str(test_config),
            test_ratio=1.0,
            batch_size=256,
            device=DEVICE
        )
        
        # Determine output filename
        results_file = model_dir / 'testing_results.json'
        if results_file.exists():
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            results_file = model_dir / f'testing_results_{timestamp}.json'
        
        # Save results
        output = {
            'mmbody_evaluation': {
                'test_samples': results['test_samples'],
                'total_samples': results.get('total_samples', results['test_samples']),
                'enabled_datasets': results['enabled_datasets'],
                'avg_losses': {
                    'total_loss': results['avg_losses']['total'],
                    'recon_loss': results['avg_losses']['recon'],
                    'vq_loss': results['avg_losses']['vq']
                },
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"recon={results['avg_losses']['recon']:.6f} -> {results_file.name}")
    
    print("=" * 60 + "\nDone!")


if __name__ == "__main__":
    evaluate_all_models()
