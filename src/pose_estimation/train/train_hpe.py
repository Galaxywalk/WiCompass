# -*- coding: utf-8 -*-
"""
Training script for mmWave radar-based human pose estimation.

Usage:
    python train_hpe.py --config_path configs/mmbody.yaml --gpu 0
"""

import argparse
import os
import json
import yaml
import logging
import sys
import datetime
from pathlib import Path

# Add src to Python path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pose_estimation.model import PointTransformer, mpjpe, p_mpjpe
from pose_estimation.dataset import MMBodyDataset, MMFiDataset, get_dataset


def get_current_time() -> str:
    """Get current timestamp as formatted string."""
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_device():
    """Setup and return device. Uses GPU 0 by default, or as specified by CUDA_VISIBLE_DEVICES."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def get_model(model_name, config, device):
    """Get model instance by name."""
    model_map = {
        'PointTransformer': PointTransformer,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name](config, device=device).to(device)


def train_epoch(model, train_loader, optimizer, criterion, device, grad_norm, epoch, epochs, 
                writer, log_interval, logger):
    """Train for one epoch."""
    model.train()
    epoch_loss = AverageMeter()
    epoch_mpjpe = AverageMeter()
    epoch_p_mpjpe = AverageMeter()
    
    with tqdm(total=len(train_loader), desc=f'Train epoch {epoch+1}/{epochs}', unit='batch') as pbar:
        for data in train_loader:
            radar_data, gt_data = data[0], data[1]
            gt_data = gt_data.to(device)
            
            optimizer.zero_grad()
            outputs, feat = model(radar_data)
            outputs = outputs.type(torch.FloatTensor).to(device)
            
            loss = criterion(outputs, gt_data)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            
            # Compute metrics (in millimeters)
            batch_mpjpe = mpjpe(outputs, gt_data).item() * 1e3
            batch_p_mpjpe = p_mpjpe(outputs, gt_data).item() * 1e3
            
            epoch_loss.update(loss.item(), gt_data.size(0))
            epoch_mpjpe.update(batch_mpjpe, gt_data.size(0))
            epoch_p_mpjpe.update(batch_p_mpjpe, gt_data.size(0))
            
            pbar.update(1)
            pbar.set_postfix(**{'loss': loss.item(), 'mpjpe': batch_mpjpe, 'p-mpjpe': batch_p_mpjpe})
            
            if pbar.n % log_interval == 0:
                step = epoch * len(train_loader) + pbar.n
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/mpjpe', batch_mpjpe, step)
                writer.add_scalar('train/p-mpjpe', batch_p_mpjpe, step)
    
    logger.info(f'Epoch:{epoch+1}, Loss:{epoch_loss.avg:.9f}, MPJPE:{epoch_mpjpe.avg:.4f}mm, P-MPJPE:{epoch_p_mpjpe.avg:.4f}mm')
    writer.add_scalar('train/epoch_loss', epoch_loss.avg, epoch)
    writer.add_scalar('train/epoch_mpjpe', epoch_mpjpe.avg, epoch)
    writer.add_scalar('train/epoch_p-mpjpe', epoch_p_mpjpe.avg, epoch)
    
    return epoch_loss.avg, epoch_mpjpe.avg, epoch_p_mpjpe.avg


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, epoch, writer, logger):
    """Evaluate model on test set."""
    model.eval()
    test_loss = AverageMeter()
    test_mpjpe = AverageMeter()
    test_p_mpjpe = AverageMeter()
    
    for data in test_loader:
        radar_data, gt_data = data[0], data[1]
        
        outputs, feat = model(radar_data)
        outputs = outputs.type(torch.FloatTensor).to(device)
        loss = criterion(outputs, gt_data)
        
        test_loss.update(loss.item(), gt_data.size(0))
        test_mpjpe.update(mpjpe(outputs, gt_data).item() * 1e3, gt_data.size(0))
        test_p_mpjpe.update(p_mpjpe(outputs, gt_data).item() * 1e3, gt_data.size(0))
    
    logger.info(f'Test Loss:{test_loss.avg:.9f}, MPJPE:{test_mpjpe.avg:.4f}mm, P-MPJPE:{test_p_mpjpe.avg:.4f}mm')
    writer.add_scalar('test/loss', test_loss.avg, epoch)
    writer.add_scalar('test/mpjpe', test_mpjpe.avg, epoch)
    writer.add_scalar('test/p-mpjpe', test_p_mpjpe.avg, epoch)
    
    return test_loss.avg, test_mpjpe.avg, test_p_mpjpe.avg


def main(args):
    # Load configuration
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    
    # Setup device
    device = setup_device()
    
    # Get names from config
    dataset_name = config['dataset_name']
    model_name = config.get('model_name', 'PointTransformer')
    
    # Create datasets
    train_dataset = get_dataset(config, 'train', device)
    test_dataset = get_dataset(config, 'test', device)
    
    experiment_name = train_dataset.exp_name()
    
    # Setup logging
    current_time = get_current_time()
    log_dir = config["recorder"]["log_dir"]
    checkpoint_root_dir = config["recorder"]["checkpoint_root_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_root_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/{current_time}.txt", mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    # Checkpoint directory with model_name and experiment_name
    if experiment_name:
        checkpoint_dir = os.path.join(checkpoint_root_dir, f'{model_name}-{experiment_name}-{current_time}')
    else:
        checkpoint_dir = os.path.join(checkpoint_root_dir, f'{model_name}-{dataset_name}-{current_time}')
    
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    # Training parameters
    batch_size = config["loader"]["batch_size"]
    epochs = config["loader"]["epochs"]
    log_interval = config["recorder"]["log_interval"]
    test_freq = config["recorder"]["test_freq"]
    
    # Optimizer parameters
    trainer_cfg = config["trainer"]
    lr = trainer_cfg["lr"]
    grad_norm = trainer_cfg.get("grad_norm", 1.0)
    weight_decay = trainer_cfg.get("weight_decay", 0.01)
    
    # Create data loaders with optional subset sampling
    if config.get('full_scaling_ratio') and config['full_scaling_ratio'] < 1.0:
        dataset_ratio = config['full_scaling_ratio']
        total_len = len(train_dataset)
        subset_len = int(total_len * dataset_ratio)
        # Set random seed for reproducible subset selection
        random_seed = config.get('random_seed', 42)
        torch.manual_seed(random_seed)
        # Randomly select subset indices (not just first N samples)
        indices = torch.randperm(total_len)[:subset_len]
        logger.info(f"Using {dataset_ratio*100:.1f}% of training data: {subset_len}/{total_len} samples (seed={random_seed})")
        sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")
    
    # Create model using model_name from config
    model = get_model(model_name, config, device)
    
    # Setup optimizer with proper parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    criterion = nn.MSELoss()
    
    logger.info(f"Optimizer: AdamW(lr={lr}, weight_decay={weight_decay})")
    
    # Option to track best metrics separately for train and test
    separate_best_metrics = config.get("recorder", {}).get("separate_best_metrics", False)
    
    # Option to log epoch-by-epoch history
    log_epoch_history = config.get("recorder", {}).get("log_epoch_history", False)
    epoch_history = [] if log_epoch_history else None
    
    # Training loop
    best_train_mpjpe = float('inf')
    best_test_mpjpe = float('inf')
    
    if separate_best_metrics:
        # Separate tracking: train and test best metrics tracked independently
        best_metrics = {
            "model_name": model_name,
            "model_scale": config.get("model", {}).get("scale", "unknown"),
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "separate_best_metrics": True,
            "train": {"best_epoch": -1, "loss": None, "mpjpe": None, "p_mpjpe": None},
            "test": {"best_epoch": -1, "loss": None, "mpjpe": None, "p_mpjpe": None},
        }
    else:
        # Joint tracking: record both train and test metrics at the epoch with best test performance
        best_metrics = {
            "model_name": model_name,
            "model_scale": config.get("model", {}).get("scale", "unknown"),
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "separate_best_metrics": False,
            "best_epoch": -1,
            "train": {"loss": None, "mpjpe": None, "p_mpjpe": None},
            "test": {"loss": None, "mpjpe": None, "p_mpjpe": None},
        }
    
    for epoch in range(epochs):
        train_loss, train_mpjpe_val, train_p_mpjpe_val = train_epoch(
            model, train_loader, optimizer, criterion, device, grad_norm, 
            epoch, epochs, writer, log_interval, logger
        )
        
        # Update best train metrics (only in separate mode)
        if separate_best_metrics and train_mpjpe_val < best_train_mpjpe:
            best_train_mpjpe = train_mpjpe_val
            best_metrics["train"]["best_epoch"] = epoch + 1
            best_metrics["train"]["loss"] = train_loss
            best_metrics["train"]["mpjpe"] = train_mpjpe_val
            best_metrics["train"]["p_mpjpe"] = train_p_mpjpe_val
            logger.info(f"New best train MPJPE: {train_mpjpe_val:.4f}mm at epoch {epoch+1}")
        
        if (epoch + 1) % test_freq == 0:
            test_loss, test_mpjpe_val, test_p_mpjpe_val = evaluate(
                model, test_loader, criterion, device, epoch, writer, logger
            )
            
            # Update best test metrics and save model
            if test_mpjpe_val < best_test_mpjpe:
                best_test_mpjpe = test_mpjpe_val
                
                if separate_best_metrics:
                    # Separate mode: only update test metrics
                    best_metrics["test"]["best_epoch"] = epoch + 1
                    best_metrics["test"]["loss"] = test_loss
                    best_metrics["test"]["mpjpe"] = test_mpjpe_val
                    best_metrics["test"]["p_mpjpe"] = test_p_mpjpe_val
                else:
                    # Joint mode: update both train and test metrics at this epoch
                    best_metrics["best_epoch"] = epoch + 1
                    best_metrics["train"]["loss"] = train_loss
                    best_metrics["train"]["mpjpe"] = train_mpjpe_val
                    best_metrics["train"]["p_mpjpe"] = train_p_mpjpe_val
                    best_metrics["test"]["loss"] = test_loss
                    best_metrics["test"]["mpjpe"] = test_mpjpe_val
                    best_metrics["test"]["p_mpjpe"] = test_p_mpjpe_val
                
                # Save best model (based on test performance)
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
                logger.info(f"New best test MPJPE: {test_mpjpe_val:.4f}mm at epoch {epoch+1}")
            
            # Save/update best metrics to JSON after each evaluation
            metrics_path = os.path.join(checkpoint_dir, 'best_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=2)
            
            # Log epoch history if enabled
            if log_epoch_history:
                epoch_history.append({
                    "epoch": epoch + 1,
                    "train": {"loss": train_loss, "mpjpe": train_mpjpe_val, "p_mpjpe": train_p_mpjpe_val},
                    "test": {"loss": test_loss, "mpjpe": test_mpjpe_val, "p_mpjpe": test_p_mpjpe_val},
                })
                history_path = os.path.join(checkpoint_dir, 'epoch_history.json')
                with open(history_path, 'w') as f:
                    json.dump(epoch_history, f, indent=2)
    
    writer.close()
    logger.info(f"Training complete!")
    if separate_best_metrics:
        logger.info(f"Best train MPJPE: {best_train_mpjpe:.4f}mm (epoch {best_metrics['train']['best_epoch']})")
        logger.info(f"Best test MPJPE: {best_test_mpjpe:.4f}mm (epoch {best_metrics['test']['best_epoch']})")
    else:
        logger.info(f"Best test MPJPE: {best_test_mpjpe:.4f}mm at epoch {best_metrics['best_epoch']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for mmWave pose estimation')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=str, default=None, help='GPU id(s) to use, e.g. "0" or "0,1"')
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    main(args)
