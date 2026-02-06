# -*- coding: utf-8 -*-
"""
Token Visualizer Module

VQ-VAE token visualization functions, including t-SNE dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. t-SNE/PCA visualization disabled.")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive plots disabled.")

from .core import create_dataloader, BaseVisualizer


__all__ = [
    'TokenVisualizer',
    'extract_token_representations',
    'apply_dimensionality_reduction',
    'create_token_distribution_visualization',
]


def extract_token_representations(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: str,
    sampling_mode: str = 'per-dataset', 
    samples_per_dataset: int = 1000, 
    max_samples: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract token-level representations from VQ-VAE model for visualization
    
    Args:
        model: VQ-VAE model
        dataloader: Data loader
        device: Compute device
        sampling_mode: 'per-dataset' or 'total'
        samples_per_dataset: Number of samples per dataset (for per-dataset mode)
        max_samples: Max total samples across all datasets (for total mode)
    
    Returns: 
        features: (num_samples, token_num * hidden_dim), labels: (num_samples,)
    """
    model.eval()
    dataset_features = {}
    dataset_labels = {}
    dataset_counts = {}
    total_samples = 0
    
    if sampling_mode == 'per-dataset':
        print(f"Extracting token representations (max {samples_per_dataset} per dataset)...")
    else:
        print(f"Extracting token representations (max {max_samples} total across all datasets)...")
    
    with torch.no_grad():
        for joints, labels in tqdm(dataloader, desc="Extracting token features"):
            joints = joints.to(device)
            joints_visible = torch.ones(joints.shape[:2], device=device).bool()
            
            encoding_indices, quantized, _ = model.encode(joints, joints_visible, train=False)
            batch_size, token_num, hidden_dim = quantized.shape
            
            sample_features = quantized.reshape(batch_size, token_num * hidden_dim).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(batch_size):
                label = labels_np[i]
                
                if label not in dataset_features:
                    dataset_features[label] = []
                    dataset_labels[label] = []
                    dataset_counts[label] = 0
                
                should_add = False
                if sampling_mode == 'per-dataset':
                    should_add = dataset_counts[label] < samples_per_dataset
                else:
                    should_add = total_samples < max_samples
                
                if should_add:
                    dataset_features[label].append(sample_features[i])
                    dataset_labels[label].append(label)
                    dataset_counts[label] += 1
                    total_samples += 1
            
            if sampling_mode == 'per-dataset':
                all_datasets_full = all(count >= samples_per_dataset for count in dataset_counts.values())
                if all_datasets_full:
                    print(f"All datasets reached sampling limit ({samples_per_dataset}), stopping extraction.")
                    break
            else:
                if total_samples >= max_samples:
                    print(f"Reached total sampling limit ({max_samples}), stopping extraction.")
                    break
    
    all_sample_features = []
    all_sample_labels = []
    
    for label in sorted(dataset_features.keys()):
        features = np.array(dataset_features[label])
        labels_arr = np.array(dataset_labels[label])
        
        all_sample_features.append(features)
        all_sample_labels.append(labels_arr)
        
        print(f"Dataset {label}: collected {len(features)} samples")
    
    all_sample_features = np.concatenate(all_sample_features, axis=0)
    all_sample_labels = np.concatenate(all_sample_labels, axis=0)
    
    print(f"Extracted {len(all_sample_features)} samples total")
    print(f"Token features shape: {all_sample_features.shape}")
    
    return all_sample_features, all_sample_labels


def apply_dimensionality_reduction(
    features: np.ndarray, 
    method: str = 'tsne', 
    n_components: int = 2, 
    random_state: int = 42
) -> np.ndarray:
    """Apply dimensionality reduction algorithm"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for dimensionality reduction")
    
    print(f"Applying {method.upper()} dimensionality reduction...")
    
    if method.lower() == 'tsne':
        if features.shape[1] > 50:
            print("High dimensional features detected, applying PCA preprocessing...")
            pca = PCA(n_components=50, random_state=random_state)
            features = pca.fit_transform(features)
            print(f"PCA reduced to {features.shape[1]} dimensions")
        
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=30,
            max_iter=1000,
            learning_rate=200,
            verbose=1
        )
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    reduced_features = reducer.fit_transform(features)
    print(f"Reduced features shape: {reduced_features.shape}")
    
    return reduced_features


def create_token_distribution_visualization(
    reduced_features: np.ndarray, 
    labels: np.ndarray, 
    dataset_configs: List[Dict], 
    output_path: Union[str, Path],
    method: str = 't-SNE'
) -> pd.DataFrame:
    """Create token distribution visualization charts"""
    output_path = Path(output_path)
    
    label_to_name = {}
    for i, config in enumerate(dataset_configs):
        label_to_name[i] = config['name']
    
    df = pd.DataFrame({
        'x': reduced_features[:, 0],
        'y': reduced_features[:, 1],
        'dataset_label': labels
    })
    
    df['dataset_name'] = df['dataset_label'].map(label_to_name)
    
    dataset_counts = df['dataset_name'].value_counts()
    print(f"\nToken sample distribution by dataset:")
    for name, count in dataset_counts.items():
        print(f"  {name}: {count:,} samples")
    
    unique_datasets = df['dataset_name'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_datasets)))
    color_map = dict(zip(unique_datasets, colors))
    
    # Create matplotlib static plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for dataset_name in unique_datasets:
        mask = df['dataset_name'] == dataset_name
        count = mask.sum()
        ax1.scatter(
            df[mask]['x'], df[mask]['y'],
            label=f'{dataset_name} ({count:,})',
            alpha=0.8,
            s=3,
            color=color_map[dataset_name],
            edgecolors='none'
        )
    
    ax1.set_xlabel(f'{method} Component 1')
    ax1.set_ylabel(f'{method} Component 2')
    ax1.set_title(f'VQ-VAE Token Distribution ({method}) - Each point = 1 Sample')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    for dataset_name in unique_datasets:
        mask = df['dataset_name'] == dataset_name
        if mask.sum() > 10:
            ax2.scatter(
                df[mask]['x'], df[mask]['y'],
                alpha=0.4,
                s=2,
                color=color_map[dataset_name],
                edgecolors='none'
            )
    
    ax2.set_xlabel(f'{method} Component 1')
    ax2.set_ylabel(f'{method} Component 2')
    ax2.set_title(f'VQ-VAE Token Density ({method})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    static_path = output_path.with_suffix('.png')
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Static token visualization saved to: {static_path}")
    
    # Create interactive plotly chart (if available)
    if PLOTLY_AVAILABLE:
        fig_interactive = px.scatter(
            df,
            x='x',
            y='y',
            color='dataset_name',
            title=f'VQ-VAE Token Distribution ({method}) - Interactive (Each point = 1 Sample)',
            labels={
                'x': f'{method} Component 1',
                'y': f'{method} Component 2'
            },
            opacity=0.7
        )
        
        fig_interactive.update_traces(marker=dict(size=4))
        fig_interactive.update_layout(
            width=1000,
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        interactive_path = output_path.with_suffix('.html')
        fig_interactive.write_html(interactive_path)
        print(f"Interactive token visualization saved to: {interactive_path}")
    
    return df


def deduplicate_features_per_label(
    features: np.ndarray, 
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate features per label based on feature values."""
    dedup_features_list = []
    dedup_labels_list = []
    total_removed = 0

    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        rows = features[mask]
        if rows.size == 0:
            continue
        
        _, unique_idx = np.unique(rows, axis=0, return_index=True)
        unique_idx_sorted = np.sort(unique_idx)

        kept = rows[unique_idx_sorted]
        dedup_features_list.append(kept)
        dedup_labels_list.append(np.full(len(unique_idx_sorted), label, dtype=labels.dtype))

        removed = rows.shape[0] - kept.shape[0]
        if removed > 0:
            print(f"Deduplicated label {label}: removed {removed} duplicates (kept {kept.shape[0]}/{rows.shape[0]})")
            total_removed += removed

    dedup_features = np.concatenate(dedup_features_list, axis=0) if dedup_features_list else features
    dedup_labels = np.concatenate(dedup_labels_list, axis=0) if dedup_labels_list else labels
    print(f"Total duplicates removed: {total_removed}. Samples after dedup: {len(dedup_features)}")
    return dedup_features, dedup_labels


class TokenVisualizer(BaseVisualizer):
    """Token distribution visualizer with t-SNE/PCA dimensionality reduction"""
    
    def __init__(self, config_path: str, model_path: str, device: str = 'cuda'):
        """
        Initialize token visualizer.
        
        Args:
            config_path: Path to config file
            model_path: Path to model checkpoint (required for token extraction)
            device: Compute device ('cuda' or 'cpu')
        """
        if not model_path:
            raise ValueError("model_path is required for TokenVisualizer")
        super().__init__(config_path, model_path, device)
    
    def create_token_visualization(
        self, 
        output_dir: str = "token_visualization",
        sampling_mode: str = 'per-dataset', 
        samples_per_dataset: int = 1000,
        max_samples: int = 5000, 
        batch_size: int = 64,
        method: str = 'tsne', 
        random_seed: int = 42
    ):
        """
        Create complete token distribution visualization
        
        Args:
            output_dir: Output directory
            sampling_mode: Sampling mode ('per-dataset' or 'total')
            samples_per_dataset: Sample count per dataset
            max_samples: Max total sample count
            batch_size: Batch size
            method: Dimensionality reduction method ('tsne' or 'pca')
            random_seed: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for token visualization")
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("📊 Creating token visualization...")
        
        dataset_configs = self.get_dataset_configs()
        dataset, dataset_info, enabled_datasets = self.create_dataset(device=str(self.device))
        
        print(f"Enabled datasets: {enabled_datasets}")
        print(f"Total samples in dataset: {len(dataset)}")
        
        dataloader = create_dataloader(dataset, batch_size, shuffle=True)
        
        token_features, token_labels = extract_token_representations(
            self.model, dataloader, str(self.device),
            sampling_mode=sampling_mode,
            samples_per_dataset=samples_per_dataset,
            max_samples=max_samples
        )
        
        print(f"Final samples for visualization: {len(token_features)}")

        token_features, token_labels = deduplicate_features_per_label(token_features, token_labels)

        reduced_features = apply_dimensionality_reduction(
            token_features,
            method=method,
            random_state=random_seed
        )
        
        df = create_token_distribution_visualization(
            reduced_features,
            token_labels,
            dataset_configs,
            output_dir / f"token_visualization_{method}",
            method=method.upper()
        )
        
        np.savez(
            output_dir / "token_data.npz",
            token_features=token_features,
            reduced_features=reduced_features,
            token_labels=token_labels,
            dataset_configs=[cfg['name'] for cfg in dataset_configs]
        )
        
        import json
        visualization_config = {
            'config_path': self.config_path,
            'enabled_datasets': enabled_datasets,
            'dataset_info': dataset_info,
            'total_samples': len(token_features),
            'sampling_mode': sampling_mode,
            'samples_per_dataset': samples_per_dataset if sampling_mode == 'per-dataset' else None,
            'max_samples': max_samples if sampling_mode == 'total' else None,
            'method': method,
            'random_seed': random_seed,
            'feature_dim': token_features.shape[1],
        }
        
        with open(output_dir / "token_visualization_config.json", 'w') as f:
            json.dump(visualization_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Token visualization completed! Results saved to: {output_dir}")
        print(f"📊 Generated files:")
        print(f"  - token_visualization_{method}.png (static plot)")
        if PLOTLY_AVAILABLE:
            print(f"  - token_visualization_{method}.html (interactive plot)")
        print(f"  - token_data.npz (raw data)")
        print(f"  - token_visualization_config.json (configuration)")
        
        return df

