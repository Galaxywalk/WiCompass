#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQ-VAE Codebook Coverage Analysis

Analyzes token distribution and coverage across different datasets using K-Means
clustering on token sequences. Measures how well each dataset covers the 
universal token sequence space.

Usage:
    python src/wicompass/latent_space_analysis/vqvae_coverage_calculation.py \
        --model_path logs/vqvae/best_model.pth \
        --config_path configs/vqvae_config.json \
        --output-dir coverage_analysis
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directories to path for direct script execution
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wicompass.model import create_joint_tokenizer
from wicompass.dataset import create_dataset
from wicompass.evaluation.core import load_config, get_dataset_configs


class CodebookCoverageAnalyzer:
    """Analyzes VQ-VAE codebook coverage and token distribution across datasets."""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to VQ-VAE model checkpoint
            config_path: Path to model configuration file
            device: Computation device ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model and config
        self._load_model()
        
        # Storage for analysis results
        self.dataset_token_usage: Dict[str, np.ndarray] = {}
        self.dataset_features: Dict[str, np.ndarray] = {}
        
    def _load_model(self):
        """Load model and configuration."""
        print(f"📁 Loading config: {self.config_path}")
        self.config = load_config(self.config_path)
        self.model_cfg = self.config['model']
        self.data_cfg = self.config['data']
        
        print(f"🔧 Loading model: {self.model_path}")
        self.model = create_joint_tokenizer(self.model_cfg)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle DataParallel prefix
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Extract codebook vectors
        self.codebook_vectors = self.model.codebook.data.cpu().numpy()
        print(f"✅ Codebook shape: {self.codebook_vectors.shape}")
        
    def extract_token_usage(
        self, 
        samples_per_dataset: int = 5000, 
        batch_size: int = 32
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract token usage from datasets.
        
        Args:
            samples_per_dataset: Max samples per dataset (-1 for all)
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (token_usage, features) dictionaries
        """
        print(f"\n🔍 Extracting token usage from datasets...")
        use_all = (samples_per_dataset == -1)
        
        # Create dataset
        dataset_configs = get_dataset_configs(self.data_cfg)
        num_joints = self.model_cfg.get('num_joints', 22)
        master_dataset, dataset_info = create_dataset(dataset_configs, num_joints, 'cpu')
        
        print(f"Total samples: {len(master_dataset)}")
        if use_all:
            print("⚠️  Extracting ALL samples (may require significant memory)")
        else:
            print(f"Extracting up to {samples_per_dataset} samples per dataset")
        
        label_to_name = dataset_info.get('label_to_name', {})
        
        # Create dataloader
        def collate_fn(batch):
            joints, labels = zip(*batch)
            return torch.stack(joints), torch.tensor(labels, dtype=torch.long)
        
        dataloader = DataLoader(
            master_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        # Collect tokens
        dataset_tokens = defaultdict(list)
        sample_counts = defaultdict(int)
        
        with torch.no_grad():
            for batch_joints, batch_labels in tqdm(dataloader, desc="Extracting tokens"):
                batch_joints = batch_joints.to(self.device)
                encoding_indices, _, _ = self.model.encode(batch_joints, train=False)
                encoding_indices = encoding_indices.cpu().numpy()
                
                for i, label in enumerate(batch_labels):
                    label_id = label.item()
                    
                    # Skip if we've collected enough (unless using all)
                    if not use_all and sample_counts[label_id] >= samples_per_dataset:
                        continue
                    
                    dataset_tokens[label_id].append(encoding_indices[i])
                    sample_counts[label_id] += 1
                
                # Early exit if all datasets have enough samples
                if not use_all and all(c >= samples_per_dataset for c in sample_counts.values()):
                    break
        
        # Consolidate results
        print("\n🔄 Consolidating results...")
        for label_id, tokens_list in dataset_tokens.items():
            name = label_to_name.get(label_id, f"Dataset_{label_id}")
            
            if tokens_list:
                tokens_array = np.vstack(tokens_list)
                print(f"  📊 {name}: {tokens_array.shape} ({sample_counts[label_id]} samples)")
            else:
                tokens_array = np.array([])
                print(f"  📊 {name}: No samples")
            
            self.dataset_token_usage[name] = tokens_array
            self.dataset_features[name] = tokens_array
        
        # Cleanup
        del dataset_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.dataset_token_usage, self.dataset_features
    
    def analyze_coverage(self, K: int = 1024) -> Tuple[Dict, object]:
        """
        Analyze token sequence space coverage using K-Means clustering.
        
        Args:
            K: Number of clusters
            
        Returns:
            Tuple of (results dict, kmeans model)
        """
        print(f"\n🎯 Analyzing coverage with K={K} clusters...")
        
        if not self.dataset_features:
            raise ValueError("Run extract_token_usage() first")
        
        # Collect all token sequences
        print("Collecting token sequences...")
        all_features = []
        all_labels = []
        dataset_names = list(self.dataset_features.keys())
        
        for i, (name, tokens) in enumerate(self.dataset_token_usage.items()):
            if tokens.ndim == 2 and tokens.shape[0] > 0:
                all_features.append(tokens)
                all_labels.extend([i] * len(tokens))
                print(f"  {name}: {len(tokens)} samples")
        
        if not all_features:
            raise ValueError("No valid token sequences found")
        
        universe_vectors = np.concatenate(all_features, axis=0)
        universe_labels = np.array(all_labels)
        print(f"Total: {universe_vectors.shape}")
        
        # K-Means clustering
        print(f"Running K-Means (K={K})...")
        start_time = time.time()
        
        n_samples = len(universe_vectors)
        if n_samples > 1_000_000:
            kmeans = MiniBatchKMeans(
                n_clusters=K, random_state=42,
                batch_size=min(10000, n_samples // 100),
                max_iter=100, n_init=1, verbose=1
            )
        elif n_samples > 100_000:
            kmeans = MiniBatchKMeans(
                n_clusters=K, random_state=42,
                batch_size=1000, max_iter=300, n_init=3
            )
        else:
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=3)
        
        kmeans.fit(universe_vectors)
        cluster_labels = kmeans.labels_
        
        print(f"Clustering completed in {time.time() - start_time:.2f}s")
        
        # Compute distributions
        universe_dist = np.histogram(cluster_labels, bins=np.arange(K + 1))[0]
        universe_prob = universe_dist / universe_dist.sum()
        
        # Analyze each dataset
        results = {}
        for i, name in enumerate(dataset_names):
            mask = universe_labels == i
            dataset_clusters = cluster_labels[mask]
            
            dataset_dist = np.histogram(dataset_clusters, bins=np.arange(K + 1))[0]
            dataset_prob = dataset_dist / dataset_dist.sum()
            
            # Metrics
            jsd = jensenshannon(universe_prob, dataset_prob, base=2)
            used_clusters = np.sum(dataset_dist > 0)
            coverage_rate = used_clusters / K
            
            # Occupancy ratio
            epsilon = 1e-10
            occupancy_ratio = dataset_prob / (universe_prob + epsilon)
            
            results[name] = {
                'jsd_divergence': jsd,
                'cluster_distribution': dataset_prob,
                'occupancy_ratio': occupancy_ratio,
                'coverage_rate': coverage_rate,
                'used_clusters': used_clusters,
                'total_samples': mask.sum()
            }
            
            print(f"\n📈 {name}:")
            print(f"  JSD: {jsd:.4f}, Coverage: {coverage_rate:.2%} ({used_clusters}/{K})")
        
        # Plot histograms
        self._plot_distributions(results, universe_prob, dataset_names, K)
        
        return results, kmeans
    
    def _plot_distributions(
        self, 
        results: Dict, 
        universe_prob: np.ndarray,
        dataset_names: list,
        K: int
    ):
        """Plot cluster distribution histograms."""
        print(f"\n📊 Plotting distributions...")
        
        n_plots = len(dataset_names) + 1
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.array(axes).flatten() if n_plots > 1 else [axes]
        
        cluster_ids = np.arange(K)
        
        # Find valid range
        non_zero = np.where(universe_prob > 0)[0]
        x_min = max(0, non_zero.min() - 10) if len(non_zero) > 0 else 0
        x_max = min(K-1, non_zero.max() + 10) if len(non_zero) > 0 else K-1
        
        # Universal distribution
        axes[0].bar(cluster_ids, universe_prob, alpha=0.7, color='gray', width=0.8)
        axes[0].set_title('Universal Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Cluster ID')
        axes[0].set_ylabel('Probability')
        axes[0].set_xlim(x_min, x_max)
        
        # Dataset distributions
        colors = plt.cm.Set1(np.linspace(0, 1, len(dataset_names)))
        
        for i, name in enumerate(dataset_names):
            ax = axes[i + 1]
            r = results[name]
            
            ax.bar(cluster_ids, r['cluster_distribution'], alpha=0.7, color=colors[i], width=0.8)
            ax.set_title(f'{name}\n(Coverage: {r["coverage_rate"]:.1%}, JSD: {r["jsd_divergence"]:.3f})')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Probability')
            ax.set_xlim(x_min, x_max)
            ax.text(0.02, 0.98, f'Clusters: {r["used_clusters"]}\nSamples: {r["total_samples"]}',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
        # Hide extra subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('cluster_distributions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Overlay plot
        self._plot_overlay(results, universe_prob, dataset_names, K, colors, x_min, x_max)
    
    def _plot_overlay(
        self,
        results: Dict,
        universe_prob: np.ndarray,
        dataset_names: list,
        K: int,
        colors: np.ndarray,
        x_min: int,
        x_max: int
    ):
        """Plot overlay comparison of all distributions."""
        plt.figure(figsize=(15, 8))
        cluster_ids = np.arange(K)
        
        plt.bar(cluster_ids, universe_prob, alpha=0.3, color='gray',
                label='Universal', width=0.8)
        
        for i, name in enumerate(dataset_names):
            r = results[name]
            plt.plot(cluster_ids, r['cluster_distribution'],
                    color=colors[i], linewidth=2, marker='o', markersize=3,
                    label=f'{name} (JSD: {r["jsd_divergence"]:.3f})')
        
        plt.title('Cluster Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.xlim(x_min, x_max)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_distributions_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Plots saved: cluster_distributions_comparison.png, cluster_distributions_overlay.png")
    
    def visualize_results(self, coverage_results: Dict, output_dir: str):
        """
        Save all visualizations to output directory.
        
        Args:
            coverage_results: Results from analyze_coverage()
            output_dir: Output directory path
        """
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_names = list(coverage_results.keys())
        
        # JSD and coverage bar plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        jsd_values = [coverage_results[n]['jsd_divergence'] for n in dataset_names]
        ax1.bar(dataset_names, jsd_values)
        ax1.set_title('Jensen-Shannon Divergence from Universal')
        ax1.set_ylabel('JSD')
        ax1.tick_params(axis='x', rotation=45)
        
        coverage_rates = [coverage_results[n]['coverage_rate'] for n in dataset_names]
        ax2.bar(dataset_names, coverage_rates)
        ax2.set_title('Token Space Coverage Rate')
        ax2.set_ylabel('Coverage')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_analysis.png'), dpi=300)
        plt.close()
        
        # Overlap heatmap
        if len(dataset_names) > 1:
            self._plot_overlap_heatmap(coverage_results, dataset_names, output_dir)
        
        # Move histogram plots
        for filename in ['cluster_distributions_comparison.png', 'cluster_distributions_overlay.png']:
            if os.path.exists(filename):
                shutil.move(filename, os.path.join(output_dir, filename))
        
        print(f"📈 Visualizations saved to {output_dir}")
    
    def _plot_overlap_heatmap(self, results: Dict, dataset_names: list, output_dir: str):
        """Plot dataset overlap heatmap."""
        n = len(dataset_names)
        overlap_matrix = np.zeros((n, n))
        
        for i, d1 in enumerate(dataset_names):
            for j, d2 in enumerate(dataset_names):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    dist1 = results[d1]['cluster_distribution']
                    dist2 = results[d2]['cluster_distribution']
                    overlap_matrix[i, j] = np.sum(np.minimum(dist1, dist2))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, xticklabels=dataset_names, yticklabels=dataset_names,
                   annot=True, fmt='.3f', cmap='Blues',
                   cbar_kws={'label': 'Token Space Overlap'})
        plt.title('Dataset Token Space Overlap Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overlap_heatmap.png'), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='VQ-VAE Codebook Coverage Analysis')
    parser.add_argument('--model_path', required=True, help='VQ-VAE model checkpoint')
    parser.add_argument('--config_path', required=True, help='Model config file')
    parser.add_argument('--output-dir', default='codebook_analysis', help='Output directory')
    parser.add_argument('--samples-per-dataset', type=int, default=5000,
                        help='Samples per dataset (-1 for all)')
    parser.add_argument('--clusters', type=int, default=1024, help='Number of clusters')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    print("🚀 VQ-VAE Codebook Coverage Analysis")
    print("=" * 60)
    
    analyzer = CodebookCoverageAnalyzer(args.model_path, args.config_path, args.device)
    
    token_usage, features = analyzer.extract_token_usage(
        samples_per_dataset=args.samples_per_dataset,
        batch_size=args.batch_size
    )
    
    coverage_results, kmeans = analyzer.analyze_coverage(K=args.clusters)
    
    analyzer.visualize_results(coverage_results, args.output_dir)
    
    print(f"\n✅ Analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
