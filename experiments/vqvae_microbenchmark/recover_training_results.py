#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to recover training_results.json for work_dirs that are missing it.

This script reads training metrics from tensorboard logs to create comprehensive
training_results.json files compatible with the format used by train_vqvae.py.

Usage:
    conda activate wicompass
    python experiments/vqvae_microbenchmark/recover_training_results.py
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_checkpoint_info(ckpt_path: Path) -> dict:
    """Load epoch and loss from checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {
        'epoch': ckpt.get('epoch', -1),
        'loss': ckpt.get('loss', -1)
    }


def load_tensorboard_metrics(tb_dir: Path) -> dict:
    """
    Load training metrics from tensorboard logs.
    
    Returns dict with structure:
    {
        'train': {epoch: {metric_name: value, ...}, ...},
        'val': {epoch: {metric_name: value, ...}, ...},
        'lr': {epoch: value, ...}
    }
    """
    if not tb_dir.exists():
        return None
    
    # Find event files
    event_files = list(tb_dir.glob('events.out.tfevents.*'))
    if not event_files:
        return None
    
    # Load tensorboard data with size guidance to speed up loading
    # Only load scalars, skip images/histograms/etc
    size_guidance = {
        'scalars': 0,  # 0 means load all scalars
        'images': 0,   # Skip images
        'histograms': 0,
        'compressedHistograms': 0,
        'tensors': 0,
    }
    ea = EventAccumulator(str(tb_dir), size_guidance=size_guidance)
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        return None
    
    metrics = {
        'train': {},
        'val': {},
        'lr': {}
    }
    
    # Map tensorboard tags to our metric names
    tag_mapping = {
        'Train/total_loss': ('train', 'total_loss'),
        'Train/recon_loss': ('train', 'recon_loss'),
        'Train/vq_loss': ('train', 'vq_loss'),
        'Train/commitment_loss': ('train', 'commitment_loss'),
        'Val/total_loss': ('val', 'total_loss'),
        'Val/recon_loss': ('val', 'recon_loss'),
        'Val/vq_loss': ('val', 'vq_loss'),
        'Val/commitment_loss': ('val', 'commitment_loss'),
        'LearningRate': ('lr', 'value'),
    }
    
    for tag in scalar_tags:
        if tag not in tag_mapping:
            continue
        
        category, metric_name = tag_mapping[tag]
        events = ea.Scalars(tag)
        
        for event in events:
            epoch = event.step
            value = event.value
            
            if category == 'lr':
                metrics['lr'][epoch] = value
            else:
                if epoch not in metrics[category]:
                    metrics[category][epoch] = {}
                metrics[category][epoch][metric_name] = value
    
    return metrics


def get_best_and_final_metrics(tb_metrics: dict) -> tuple:
    """
    Extract best and final metrics from tensorboard data.
    
    Returns:
        (best_train, best_val, final_train, final_val, best_epoch, final_epoch)
    """
    if not tb_metrics or not tb_metrics['val']:
        return None, None, None, None, -1, -1
    
    # Find best epoch (lowest validation total_loss)
    best_epoch = -1
    best_val_loss = float('inf')
    
    for epoch, val_metrics in tb_metrics['val'].items():
        if 'total_loss' in val_metrics and val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
    
    # Final epoch is the last one
    final_epoch = max(tb_metrics['val'].keys()) if tb_metrics['val'] else -1
    
    # Get metrics for best and final epochs
    best_train = tb_metrics['train'].get(best_epoch, {})
    best_val = tb_metrics['val'].get(best_epoch, {})
    final_train = tb_metrics['train'].get(final_epoch, {})
    final_val = tb_metrics['val'].get(final_epoch, {})
    
    return best_train, best_val, final_train, final_val, best_epoch, final_epoch


def format_metrics(metrics_dict: dict) -> dict:
    """Format metrics dict to match expected structure."""
    return {
        'total_loss': metrics_dict.get('total_loss'),
        'recon_loss': metrics_dict.get('recon_loss'),
        'vq_loss': metrics_dict.get('vq_loss'),
        'commitment_loss': metrics_dict.get('commitment_loss')
    }


def recover_training_results(work_dir: Path, force: bool = False) -> bool:
    """
    Recover training_results.json for a single work_dir.
    
    Args:
        work_dir: Path to the work directory
        force: If True, overwrite existing training_results.json
        
    Returns:
        True if successful, False otherwise.
    """
    results_file = work_dir / 'training_results.json'
    config_file = work_dir / 'config.json'
    best_model = work_dir / 'best_model.pth'
    final_model = work_dir / 'final_model.pth'
    tb_dir = work_dir / 'tensorboard'
    
    # Skip if training_results.json already exists (unless force=True)
    if results_file.exists() and not force:
        print(f"  [SKIP] {work_dir.name}: training_results.json already exists")
        return False
    
    # Check required files
    if not config_file.exists():
        print(f"  [ERROR] {work_dir.name}: config.json not found")
        return False
    
    if not best_model.exists() and not final_model.exists():
        print(f"  [ERROR] {work_dir.name}: no model checkpoint found")
        return False
    
    print(f"  [PROCESSING] {work_dir.name}...", end=" ", flush=True)
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Try to load tensorboard metrics
    tb_metrics = None
    source = "checkpoint"
    
    if tb_dir.exists():
        try:
            tb_metrics = load_tensorboard_metrics(tb_dir)
            if tb_metrics and tb_metrics['train'] and tb_metrics['val']:
                source = "tensorboard"
        except Exception as e:
            print(f"\n  [WARN] {work_dir.name}: Failed to read tensorboard: {e}", end=" ")
    
    # Get metrics from tensorboard or fallback to checkpoint
    if source == "tensorboard":
        best_train, best_val, final_train, final_val, best_epoch, final_epoch = \
            get_best_and_final_metrics(tb_metrics)
        
        best_train_formatted = format_metrics(best_train)
        best_val_formatted = format_metrics(best_val)
        final_train_formatted = format_metrics(final_train)
        final_val_formatted = format_metrics(final_val)
    else:
        # Fallback to checkpoint info (limited data)
        best_info = load_checkpoint_info(best_model) if best_model.exists() else {'epoch': -1, 'loss': None}
        final_info = load_checkpoint_info(final_model) if final_model.exists() else {'epoch': -1, 'loss': None}
        
        best_epoch = best_info['epoch']
        final_epoch = final_info['epoch']
        
        best_train_formatted = format_metrics({})
        best_val_formatted = format_metrics({'total_loss': best_info['loss']})
        final_train_formatted = format_metrics({})
        final_val_formatted = format_metrics({'total_loss': final_info['loss']})
    
    # Build results structure (compatible with existing format)
    results = {
        'training_info': {
            'total_epochs': config.get('training', {}).get('epochs', final_epoch + 1),
            'timestamp': datetime.now().isoformat(),
            'config_file': f"configs/joint_vae_base_tokennum{config['model']['token_num']}_tokenclass{config['model']['token_class_num']}.json",
            'source': source,
            'note': f'Recovered from {source}'
        },
        'best_metrics': {
            'train': best_train_formatted,
            'validation': best_val_formatted
        },
        'final_metrics': {
            'train': final_train_formatted,
            'validation': final_val_formatted
        },
        'model_info': {
            'num_joints': config['model'].get('num_joints', 22),
            'token_num': config['model']['token_num'],
            'token_class_num': config['model']['token_class_num'],
            'token_dim': config['model'].get('token_dim', 256)
        },
        'checkpoint_info': {
            'best_epoch': best_epoch,
            'final_epoch': final_epoch
        }
    }
    
    # Save results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Format output message
    best_val_loss = best_val_formatted.get('total_loss')
    loss_str = f"{best_val_loss:.6f}" if best_val_loss is not None else "N/A"
    print(f"OK from {source} (best_val_loss={loss_str})")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Recover training_results.json from tensorboard logs')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Force overwrite existing training_results.json')
    parser.add_argument('--work-dirs', type=str, default=None,
                        help='Path to vqvae logs directory (default: logs/vqvae)')
    args = parser.parse_args()
    
    # Find vqvae logs directory
    if args.work_dirs:
        vqvae_logs_path = Path(args.work_dirs).expanduser()
    else:
        # This file lives in <repo>/experiments/vqvae_microbenchmark/.
        # Going up three parents points outside the repository and makes the
        # default resolve to <parent-of-repo>/logs/vqvae.
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        vqvae_logs_path = project_root / 'logs' / 'vqvae'

    vqvae_logs_path = vqvae_logs_path.resolve()
    
    if not vqvae_logs_path.exists():
        raise SystemExit(
            f"Error: VQ-VAE logs not found at {vqvae_logs_path}.\n"
            "Run tools/setup_workspace.py after extracting the "
            "wicompass_logs workspace component, or pass --work-dirs explicitly."
        )
    
    print(f"Scanning: {vqvae_logs_path}")
    if args.force:
        print("Mode: Force overwrite enabled")
    print("-" * 60)
    
    # Find all vqvae model directories
    model_dirs = sorted([
        d for d in vqvae_logs_path.iterdir()
        if d.is_dir() and d.name.startswith('vqvae_')
    ])

    if not model_dirs:
        raise SystemExit(
            f"Error: no vqvae_* experiment directories found in {vqvae_logs_path}."
        )
    
    recovered = 0
    skipped = 0
    errors = 0
    
    for model_dir in model_dirs:
        result = recover_training_results(model_dir, force=args.force)
        if result:
            recovered += 1
        elif (model_dir / 'training_results.json').exists():
            skipped += 1
        else:
            errors += 1
    
    print("-" * 60)
    print(f"Summary: {recovered} recovered, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
