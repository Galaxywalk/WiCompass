# -*- coding: utf-8 -*-
"""
Model Evaluator Module

Contains ModelEvaluator class and evaluation-related functions.
"""

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader

from wicompass.model import JointVQVAELoss
from wicompass.visualization import JOINT_NAMES

from .core import (
    load_config,
    load_model,
    create_dataloader,
    split_dataset,
    create_evaluation_dataset,
)


class ModelEvaluator:
    """Unified model evaluator"""
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.model.eval()
    
    def evaluate_batch(self, joints: torch.Tensor) -> Dict:
        """Evaluate single batch"""
        joints = joints.to(self.device, dtype=torch.float32)
        joints_visible = torch.ones(joints.shape[:2], device=self.device).bool()
        
        with torch.no_grad():
            recovered, encoding_indices, e_loss = self.model(
                joints=joints, joints_visible=joints_visible, train=False
            )
            loss_dict = self.criterion(recovered, joints, e_loss)
            
            # Compute sample-level MSE
            sample_mse = ((recovered - joints) ** 2).mean(dim=(1, 2)).cpu().numpy()
            
        return {
            'loss_dict': loss_dict,
            'recovered_joints': recovered.cpu().numpy(),
            'target_joints': joints.cpu().numpy(),
            'encoding_indices': encoding_indices.cpu().numpy() if encoding_indices is not None else None,
            'reconstruction_errors': sample_mse
        }
    
    def evaluate_dataset(self, dataloader: DataLoader) -> Dict:
        """Evaluate entire dataset"""
        all_sample_losses = []
        all_joint_losses = []
        all_encoding_indices = []
        
        # Get model parameters
        token_num = getattr(self.model, 'token_num', 64)
        codebook_size = getattr(self.model, 'token_class_num', 512)
        token_code_counter = np.zeros((token_num, codebook_size), dtype=np.int64)
        
        total_loss, total_recon, total_vq, total_samples = 0, 0, 0, 0
        
        with torch.no_grad():
            for joints, labels in tqdm(dataloader, desc="Evaluating"):
                batch_results = self.evaluate_batch(joints)
                
                loss_dict = batch_results['loss_dict']
                recovered_joints = torch.from_numpy(batch_results['recovered_joints']).to(self.device)
                joints = torch.from_numpy(batch_results['target_joints']).to(self.device)
                encoding_indices = batch_results['encoding_indices']
                
                batch_size = joints.shape[0]
                
                # Accumulate losses
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_recon += loss_dict['recon_loss'].item() * batch_size
                if 'vq_loss' in loss_dict:
                    total_vq += loss_dict['vq_loss'].item() * batch_size
                total_samples += batch_size
                
                # Sample-level errors
                all_sample_losses.extend(batch_results['reconstruction_errors'])
                
                # Joint-level errors
                joint_mse = ((recovered_joints - joints) ** 2).mean(dim=2)
                all_joint_losses.append(joint_mse.cpu().numpy())
                
                # Token distribution statistics
                if encoding_indices is not None:
                    all_encoding_indices.append(encoding_indices)
                    for pos in range(min(token_num, encoding_indices.shape[1])):
                        codes = encoding_indices[:, pos]
                        for code in codes:
                            if 0 <= code < codebook_size:
                                token_code_counter[pos, code] += 1
        
        # Merge results
        results = {
            'avg_losses': {
                'total': total_loss / total_samples,
                'recon': total_recon / total_samples,
                'vq': total_vq / total_samples if total_vq > 0 else 0.0
            },
            'sample_losses': np.array(all_sample_losses),
            'joint_losses': np.concatenate(all_joint_losses, axis=0) if all_joint_losses else None,
            'encoding_indices': np.concatenate(all_encoding_indices, axis=0) if all_encoding_indices else None,
            'token_code_counter': token_code_counter,
            'total_samples': total_samples
        }
        
        return results


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_joint_errors(joint_losses: np.ndarray) -> Dict:
    """Analyze joint errors"""
    joint_stats = {}
    mean_errors = np.mean(joint_losses, axis=0)
    std_errors = np.std(joint_losses, axis=0)
    
    for i, joint_name in enumerate(JOINT_NAMES):
        joint_stats[joint_name] = {
            'mean_error': float(mean_errors[i]),
            'std_error': float(std_errors[i]),
            'max_error': float(np.max(joint_losses[:, i])),
            'min_error': float(np.min(joint_losses[:, i]))
        }
    
    return joint_stats


def analyze_sample_errors(sample_losses: np.ndarray) -> Dict:
    """Analyze sample-level errors"""
    sample_losses = np.array(sample_losses)
    
    stats = {
        'total_samples': int(len(sample_losses)),
        'mean_loss': float(np.mean(sample_losses)),
        'std_loss': float(np.std(sample_losses)),
        'min_loss': float(np.min(sample_losses)),
        'max_loss': float(np.max(sample_losses)),
        'median_loss': float(np.median(sample_losses)),
        'q95_loss': float(np.percentile(sample_losses, 95)),
        'q99_loss': float(np.percentile(sample_losses, 99)),
    }
    
    # Outlier detection using 1.5*IQR rule
    q25 = np.percentile(sample_losses, 25)
    q75 = np.percentile(sample_losses, 75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    outlier_indices = np.where(sample_losses > upper_bound)[0]
    
    stats['outliers'] = {
        'count': int(len(outlier_indices)),
        'percentage': float(len(outlier_indices) / len(sample_losses) * 100),
        'threshold': float(upper_bound)
    }
    
    return stats


# =============================================================================
# High-level Interface
# =============================================================================

def evaluate_model(model_path: str, config_path: str, test_ratio: float = 0.1, 
                  batch_size: int = 64, device: str = 'cuda') -> Dict:
    """One-click model evaluation"""
    # Load configuration
    config = load_config(config_path)
    model_cfg = config['model']
    loss_cfg = config.get('loss', {})
    
    # Load model
    model = load_model(model_cfg, model_path, device)
    criterion = JointVQVAELoss(
        recon_weight=loss_cfg.get('recon_weight', 1.0),
        vq_weight=loss_cfg.get('vq_weight', 1.0),
        commitment_weight=loss_cfg.get('commitment_weight', 0.25),
        recon_loss_type=loss_cfg.get('recon_loss_type', 'mse')
    )
    
    # Create dataset
    dataset, dataset_info, enabled_datasets = create_evaluation_dataset(
        config_path, model_cfg.get('num_joints', 22), device
    )
    
    # Handle dataset splitting
    if test_ratio >= 1.0:
        test_dataset = dataset
        print(f"📊 Using full dataset for evaluation: {len(dataset):,} samples")
    else:
        _, test_dataset = split_dataset(dataset, 1.0 - test_ratio)
        print(f"📊 Using test set for evaluation: {len(test_dataset):,} samples ({test_ratio*100:.1f}%)")
    
    test_loader = create_dataloader(test_dataset, batch_size, shuffle=False)
    
    # Evaluate
    evaluator = ModelEvaluator(model, criterion, device)
    results = evaluator.evaluate_dataset(test_loader)
    
    # Add configuration info
    results['config'] = config
    results['dataset_info'] = dataset_info
    results['enabled_datasets'] = enabled_datasets
    results['test_samples'] = len(test_dataset)
    
    return results


# =============================================================================
# Evaluation Report Generation
# =============================================================================

def create_evaluation_report(eval_results: Dict, output_dir: str) -> Dict:
    """
    Create complete evaluation report with visualizations.
    
    Args:
        eval_results: Evaluation results from ModelEvaluator
        output_dir: Output directory for report files
        
    Returns:
        Detailed results dictionary
    """
    import matplotlib.pyplot as plt
    import json
    from wicompass.visualization import (
        plot_token_heatmap,
        plot_joint_errors,
        plot_loss_distribution,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("📊 Generating evaluation report...")
    
    # Analyze results
    joint_stats = None
    sample_stats = analyze_sample_errors(eval_results['sample_losses'])
    
    if eval_results.get('joint_losses') is not None:
        joint_stats = analyze_joint_errors(eval_results['joint_losses'])
    
    # Generate visualizations
    if eval_results.get('token_code_counter') is not None:
        fig = plot_token_heatmap(eval_results['token_code_counter'])
        fig.savefig(output_dir / "token_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Token heatmap saved")
    
    if joint_stats:
        fig = plot_joint_errors(joint_stats)
        fig.savefig(output_dir / "joint_errors.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Joint errors plot saved")
    
    # Sample errors distribution
    fig = plot_loss_distribution(eval_results['sample_losses'], sample_stats)
    fig.savefig(output_dir / "sample_errors.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Sample errors plot saved")
    
    # Save detailed results
    detailed_results = {
        'model_evaluation': {
            'avg_losses': eval_results['avg_losses'],
            'total_samples': eval_results['total_samples'],
            'sample_statistics': sample_stats,
            'joint_statistics': joint_stats,
        },
        'configuration': {
            'enabled_datasets': eval_results.get('enabled_datasets', []),
            'model_config': eval_results.get('config', {}).get('model', {}),
            'data_config': eval_results.get('config', {}).get('data', {}),
            'loss_config': eval_results.get('config', {}).get('loss', {})
        }
    }
    
    with open(output_dir / "evaluation_report.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save numpy data
    np.save(output_dir / "sample_losses.npy", eval_results['sample_losses'])
    if eval_results.get('joint_losses') is not None:
        np.save(output_dir / "joint_losses.npy", eval_results['joint_losses'])
    if eval_results.get('token_code_counter') is not None:
        np.save(output_dir / "token_code_counter.npy", eval_results['token_code_counter'])
    
    print(f"✅ Evaluation report saved to: {output_dir}")
    
    return detailed_results

