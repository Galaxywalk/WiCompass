#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AMASS Joint VQ-VAE Training

Examples:
python src/train/train_vqvae.py --config configs/joint_vae_base.json --work-dir work_dirs/vqvae_resume --resume work_dirs/vqvae_exp5/best.pth 
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum8_tokenclass32.json --work-dir work_dirs/vqvae_tokennum8_tokenclass32 --gpus "1"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum16_tokenclass32.json --work-dir work_dirs/vqvae_tokennum16_tokenclass32 --gpus "2"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum16_tokenclass64.json --work-dir work_dirs/vqvae_tokennum16_tokenclass64 --gpus "3"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum16_tokenclass128.json --work-dir work_dirs/vqvae_tokennum16_tokenclass128 --gpus "4"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum32_tokenclass128.json --work-dir work_dirs/vqvae_tokennum32_tokenclass128 --gpus "5"

python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum32_tokenclass64.json --work-dir work_dirs/vqvae_tokennum32_tokenclass64 --gpus "0"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum32_tokenclass32.json --work-dir work_dirs/vqvae_tokennum32_tokenclass32 --gpus "1"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum4_tokenclass32.json --work-dir work_dirs/vqvae_tokennum4_tokenclass32 --gpus "2"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum4_tokenclass64.json --work-dir work_dirs/vqvae_tokennum4_tokenclass64 --gpus "3"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum4_tokenclass128.json --work-dir work_dirs/vqvae_tokennum4_tokenclass128 --gpus "4"
python src/train/train_vqvae.py --config configs/joint_vae_base_tokennum8_tokenclass64.json --work-dir work_dirs/vqvae_tokennum8_tokenclass64 --gpus "5"




python src/train/train_vqvae.py --config configs/joint_vae_base.json --gpus "4,5,6,7"
"""

import argparse
import os
import json
import yaml
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add src to Python path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from wicompass.dataset import create_dataset, get_available_datasets
from wicompass.model import create_joint_tokenizer, JointVQVAELoss
from wicompass.evaluation import ModelEvaluator, load_config


def set_seed(seed: int):
    """Set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['loss']


class LossCalculator:
    """Unified loss computation utility."""
    
    def __init__(self, criterion):
        self.criterion = criterion
    
    def compute_batch_loss(self, model, batch, device, train_mode=True):
        """
        Compute loss for a single batch.
        
        Args:
            model: Model instance
            batch: Data batch
            device: Computing device
            train_mode: Whether in training mode
            
        Returns:
            loss_dict: Loss dictionary
            encoding_indices: Encoding indices
        """
        joints, labels = batch
        # Data is already on GPU from dataset, no need to transfer
        # joints = joints.to(device, dtype=torch.float32)  # Remove unnecessary transfer
        joints_visible = torch.ones(joints.shape[:2], device=device).bool()
        
        # Forward pass
        recovered, encoding_indices, e_loss = model(
            joints=joints, 
            joints_visible=joints_visible, 
            train=train_mode
        )
        
        # Compute loss
        loss_dict = self.criterion(recovered, joints, e_loss)
        
        return loss_dict, encoding_indices


class MetricsTracker:
    """Training metrics tracker."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.recon_losses = []
        self.vq_losses = []
        self.commitment_losses = []
    
    def update(self, loss_dict):
        """Update metrics."""
        self.losses.append(loss_dict['total_loss'].item())
        self.recon_losses.append(loss_dict['recon_loss'].item())
        if 'vq_loss' in loss_dict:
            self.vq_losses.append(loss_dict['vq_loss'].item())
        if 'commitment_loss' in loss_dict:
            self.commitment_losses.append(loss_dict['commitment_loss'].item())
    
    def get_averages(self):
        """Get averages."""
        return {
            'total_loss': np.mean(self.losses) if self.losses else 0.0,
            'recon_loss': np.mean(self.recon_losses) if self.recon_losses else 0.0,
            'vq_loss': np.mean(self.vq_losses) if self.vq_losses else 0.0,
            'commitment_loss': np.mean(self.commitment_losses) if self.commitment_losses else 0.0
        }


def save_training_results(work_dir, best_train_metrics, best_val_metrics, final_train_metrics, final_val_metrics, config, total_epochs):
    """
    Save training results to JSON file.
    
    Args:
        work_dir: Working directory
        best_train_metrics: Best training metrics
        best_val_metrics: Best validation metrics  
        final_train_metrics: Final training metrics
        final_val_metrics: Final validation metrics
        config: Configuration information
        total_epochs: Total number of training epochs
    """
    results = {
        'training_info': {
            'total_epochs': total_epochs,
            'timestamp': datetime.now().isoformat(),
            'config_file': config.get('config_file', 'unknown'),
        },
        'best_metrics': {
            'train': best_train_metrics,
            'validation': best_val_metrics
        },
        'final_metrics': {
            'train': final_train_metrics,
            'validation': final_val_metrics
        },
        'model_info': {
            'num_joints': config['model']['num_joints'],
            'token_num': config['model']['token_num'],
            'token_class_num': config['model']['token_class_num'],
            'token_dim': config['model']['token_dim']
        }
    }
    
    results_file = Path(work_dir) / 'training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Training results saved to: {results_file}")






def setup_model(config, device):
    """Setup model and device."""
    model = create_joint_tokenizer(config['model']).float().to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    """Train one epoch."""
    model.train()
    
    loss_calculator = LossCalculator(criterion)
    metrics_tracker = MetricsTracker()
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Compute loss
        loss_dict, encoding_indices = loss_calculator.compute_batch_loss(
            model, batch, device, train_mode=True
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(loss_dict)
        
        # Monitor codebook usage
        if encoding_indices is not None and batch_idx % 100 == 0:
            _log_codebook_usage(model, encoding_indices, writer, epoch, len(loader), batch_idx)
        
        # Update progress bar
        current_metrics = metrics_tracker.get_averages()
        progress_bar.set_postfix({
            'Loss': f'{current_metrics["total_loss"]:.6f}',
            'Recon': f'{current_metrics["recon_loss"]:.6f}',
            'VQ': f'{current_metrics["vq_loss"]:.6f}' if current_metrics["vq_loss"] > 0 else 'N/A'
        })
        
        # Log to tensorboard
        if writer:
            global_step = epoch * len(loader) + batch_idx
            for k, v in loss_dict.items():
                writer.add_scalar(f'Train/{k}', v.item(), global_step)
    
    return metrics_tracker.get_averages()


def _log_codebook_usage(model, encoding_indices, writer, epoch, loader_len, batch_idx):
    """Log codebook usage."""
    unique_codes = torch.unique(encoding_indices)
    if hasattr(model, 'module'):
        codebook_usage = len(unique_codes) / model.module.token_class_num
    else:
        codebook_usage = len(unique_codes) / model.token_class_num
    
    if writer:
        global_step = epoch * loader_len + batch_idx
        writer.add_scalar('Train/CodebookUsage', codebook_usage, global_step)


def validate_epoch(model, loader, criterion, device):
    """Validate one epoch."""
    model.eval()
    
    loss_calculator = LossCalculator(criterion)
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in loader:
            # Compute loss
            loss_dict, _ = loss_calculator.compute_batch_loss(
                model, batch, device, train_mode=False
            )
            
            # Update metrics
            metrics_tracker.update(loss_dict)
    
    return metrics_tracker.get_averages()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--work-dir', default='./work_dirs', help='Working directory')
    parser.add_argument('--output-dir', help='Output directory (overrides work-dir setting)')
    parser.add_argument('--resume', help='Resume training checkpoint')
    parser.add_argument('--gpus', help='Specify GPUs, e.g. "0,1"')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config file setting)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config file setting)')
    parser.add_argument('--device', help='Computing device')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Command line arguments can override config file settings
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.output_dir is not None:
        args.work_dir = args.output_dir
    
    set_seed(config.get('seed', 42))
    
    # Setup GPU and device
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    if args.device:
        device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read dataset list from config file
    datasets_config = config['data'].get('datasets')
    
    # Get available datasets
    all_available_datasets = get_available_datasets(['amass', 'mmbody', 'mmfi', 'wicompass'])
    
    if args.list_datasets:
        print("Supported datasets:")
        for ds in all_available_datasets:
            path = Path(ds['path'])
            file_count = len(list(path.rglob("*.npz"))) if path.exists() else 0
            status = "✅" if file_count > 0 else "❌"
            print(f"  {status} {ds['name']}: {file_count:,} files ({ds['type']})")
        return
    
    # Select datasets according to configuration
    if datasets_config:
        enabled_datasets = [ds for ds in all_available_datasets if ds['name'] in datasets_config]
    else:
        # Default to include all available datasets
        enabled_datasets = all_available_datasets
    
    if not enabled_datasets:
        raise ValueError("No available datasets! Please check dataset configuration.")
    
    print("\n--- Dataset List ---")
    for ds in enabled_datasets:
        path = Path(ds['path'])
        file_count = len(list(path.rglob("*.npz"))) if path.exists() else 0
        print(f"  {ds['name']}: {file_count} files")
    
    # Create dataset and model
    model = setup_model(config, device)
    train_dataset, dataset_info = create_dataset(enabled_datasets, config['model']['num_joints'], device)
    
    train_size = int(config['data']['train_ratio'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders optimized for GPU datasets
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0,  # GPU dataset doesn't need multi-processing
        pin_memory=False  # Data already on GPU
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    criterion = JointVQVAELoss(**config['loss'])
    optimizer = optim.AdamW(model.parameters(), **config['training']['optimizer'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['training']['scheduler'])
    
    # Working directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file to work directory
    config_dst = work_dir / 'config.json'
    try:
        shutil.copy2(args.config, config_dst)
        print(f"Config file copied to: {config_dst}")
    except Exception as e:
        print(f"Failed to copy config file: {e}")
    
    writer = SummaryWriter(work_dir / 'tensorboard')
    
    # Training loop
    start_epoch = 0
    best_val_loss = float('inf')
    best_train_metrics = None
    best_val_metrics = None
    
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
    
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log validation metrics
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Print training results
        print(f'Epoch {epoch:03d}: '
              f'Train Loss={train_metrics["total_loss"]:.6f} '
              f'(Recon={train_metrics["recon_loss"]:.6f}, '
              f'VQ={train_metrics["vq_loss"]:.6f}, '
              f'Commit={train_metrics["commitment_loss"]:.6f}) | '
              f'Val Loss={val_metrics["total_loss"]:.6f} '
              f'(Recon={val_metrics["recon_loss"]:.6f}, '
              f'VQ={val_metrics["vq_loss"]:.6f}, '
              f'Commit={val_metrics["commitment_loss"]:.6f}) | '
              f'LR={current_lr:.2e}')
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = work_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            save_checkpoint(model, optimizer, epoch, val_metrics['total_loss'], checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_train_metrics = train_metrics.copy()
            best_val_metrics = val_metrics.copy()
            best_model_path = work_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_metrics['total_loss'], best_model_path)
            print(f'Saved best model: {best_model_path}')
    
    # Save final model and training results
    final_model_path = work_dir / 'final_model.pth'
    save_checkpoint(model, optimizer, config['training']['epochs']-1, val_metrics['total_loss'], final_model_path)
    
    # Save training results to JSON file
    config['config_file'] = args.config  # Add config file path
    save_training_results(
        work_dir=work_dir,
        best_train_metrics=best_train_metrics if best_train_metrics else train_metrics,
        best_val_metrics=best_val_metrics if best_val_metrics else val_metrics,
        final_train_metrics=train_metrics,
        final_val_metrics=val_metrics,
        config=config,
        total_epochs=config['training']['epochs']
    )
    
    writer.close()
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
