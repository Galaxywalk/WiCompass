#!/usr/bin/env python3
"""
Paper Token Visualization Tool
Generate visualization images and data of token variations for paper
Select a random action, specify token position, change token value multiple times, save visualizations and joint data
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from pathlib import Path
import matplotlib.pyplot as plt

from wicompass.model import create_joint_tokenizer
from wicompass.dataset import create_dataset
from ..evaluation.core import load_config, get_dataset_configs
from wicompass.visualization import (
    plot_single_pose, plot_pose_comparison, plot_multiple_poses,
    save_pose_plot, create_color_legend
)


class PaperTokenVisualizer:
    """Tool for generating token variation visualizations for paper"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize visualizer
        
        Args:
            model_path: Model weights path
            config_path: Config file path
            device: Device type
        """
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"📁 Loading config from {config_path}")
        self.config = load_config(config_path)
        self.model_cfg = self.config['model']
        self.data_cfg = self.config['data']
        
        print(f"🔧 Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Infer model config from checkpoint and update
        inferred_config = self._infer_model_config_from_checkpoint(state_dict)
        print(f"📋 Inferred model config from checkpoint:")
        for key, value in inferred_config.items():
            if key in self.model_cfg and self.model_cfg[key] != value:
                print(f"   {key}: {self.model_cfg[key]} -> {value}")
            self.model_cfg[key] = value
        
        # Create model with updated config
        self.model = create_joint_tokenizer(self.model_cfg)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Get codebook size from actually loaded model
        self.codebook_size = self.model.codebook.shape[0]
        self.token_dim = self.model.codebook.shape[1]
        self.token_num = self.model_cfg.get('num_tokens', 32)
        print(f"✅ Model loaded successfully (codebook size: {self.codebook_size}, token dim: {self.token_dim}, token num: {self.token_num})")
    
    def _infer_model_config_from_checkpoint(self, state_dict):
        """
        Infer model config from checkpoint
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Inferred config dictionary
        """
        inferred_config = {}
        
        # Infer codebook size from codebook
        if 'codebook' in state_dict:
            codebook_size = state_dict['codebook'].shape[0]
            inferred_config['codebook_size'] = codebook_size
        
        # Infer token number from token_mlp
        if 'token_mlp.weight' in state_dict:
            num_tokens = state_dict['token_mlp.weight'].shape[0]
            inferred_config['num_tokens'] = num_tokens
        
        # Infer embedding dimension from codebook
        if 'codebook' in state_dict:
            embed_dim = state_dict['codebook'].shape[1]
            inferred_config['embed_dim'] = embed_dim
        
        # Infer joint number from token_mlp
        if 'token_mlp.weight' in state_dict:
            num_joints = state_dict['token_mlp.weight'].shape[1]
            inferred_config['num_joints'] = num_joints
        
        return inferred_config
    
    def get_random_sample(self, random_seed: int = 42):
        """
        Get a random sample from dataset
        
        Args:
            random_seed: Random seed
            
        Returns:
            (joints, label, dataset_name, sample_idx, tokens, reconstructed_joints)
        """
        print(f"📊 Loading dataset...")
        
        # Create dataset
        dataset_configs = get_dataset_configs(self.data_cfg)
        num_joints = self.model_cfg.get('num_joints', 22)
        master_dataset, dataset_info = create_dataset(
            dataset_configs,
            num_joints,
            'cpu'
        )
        
        print(f"Total samples in dataset: {len(master_dataset)}")
        
        # Randomly select sample
        random.seed(random_seed)
        sample_idx = random.randint(0, len(master_dataset) - 1)
        
        joints, label = master_dataset[sample_idx]
        label_value = label.item() if hasattr(label, 'item') else label
        label_to_name = dataset_info.get('label_to_name', {})
        dataset_name = label_to_name.get(label_value, f"Dataset_{label_value}")
        
        # Use model to extract token sequence and reconstruct pose
        with torch.no_grad():
            joints_tensor = torch.from_numpy(joints.numpy()).float().unsqueeze(0).to(self.device)
            joints_visible = torch.ones(joints_tensor.shape[:2], device=self.device).bool()
            
            # Encode to get tokens
            encoding_indices, _, _ = self.model.encode(joints_tensor, joints_visible, train=False)
            tokens = encoding_indices.cpu().numpy().flatten()
            
            # Decode to reconstruct pose
            reconstructed_joints, _, _ = self.model(joints=joints_tensor, joints_visible=joints_visible, train=False)
            reconstructed_joints = reconstructed_joints.cpu().numpy()[0]
        
        print(f"✅ Selected sample {sample_idx} from {dataset_name}")
        print(f"   Original joints shape: {joints.numpy().shape}")
        print(f"   Tokens shape: {tokens.shape}, range: {tokens.min()}-{tokens.max()}")
        
        return joints.numpy(), label_value, dataset_name, sample_idx, tokens, reconstructed_joints
    
    def generate_token_variations(self, original_joints, original_tokens, token_position: int, 
                                 n_variations: int = 10, output_dir: str = "paper_token_viz"):
        """
        Generate multiple variations for specified token position
        
        Args:
            original_joints: Original joint data
            original_tokens: Original token sequence
            token_position: Token position to modify
            n_variations: Number of variations
            output_dir: Output directory
            
        Returns:
            List of variation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if token_position >= len(original_tokens):
            raise ValueError(f"Token position {token_position} exceeds sequence length {len(original_tokens)}")
        
        original_token_value = original_tokens[token_position]
        print(f"🔍 Generating {n_variations} variations for token at position {token_position}")
        print(f"   Original token value: {original_token_value}")
        
        # Randomly select token values for variation
        random.seed(42)  # Fix random seed for reproducibility
        available_values = list(range(self.codebook_size))
        available_values.remove(original_token_value)  # Remove original value
        selected_values = random.sample(available_values, min(n_variations, len(available_values)))
        
        print(f"   Selected token values: {selected_values}")
        
        # Generate modified poses
        variations = []
        joint_data_list = []
        
        # Add original pose as reference
        variations.append({
            'joints': original_joints,
            'tokens': original_tokens.copy(),
            'token_value': original_token_value,
            'is_original': True,
            'difference': 0.0
        })
        joint_data_list.append(('original', original_joints))
        
        print(f"🔄 Generating {len(selected_values)} token variations...")
        
        with torch.no_grad():
            for i, new_value in enumerate(selected_values):
                try:
                    # Create modified token sequence
                    modified_tokens = original_tokens.copy()
                    modified_tokens[token_position] = new_value
                    
                    # Convert tokens to tensor and decode
                    tokens_tensor = torch.from_numpy(modified_tokens).long().unsqueeze(0).to(self.device)
                    
                    # Use model's forward method, pass token info through cls_logits parameter
                    B, token_num = tokens_tensor.shape
                    token_indices_flat = tokens_tensor.view(-1)  # (B*token_num,)
                    
                    # Create one-hot encoding as cls_logits
                    cls_logits = torch.zeros(token_indices_flat.shape[0], self.codebook_size, 
                                           device=self.device, dtype=torch.float32)
                    cls_logits.scatter_(1, token_indices_flat.unsqueeze(1), 1.0)
                    
                    # Use model's forward method to decode
                    decoded_joints, _, _ = self.model(joints=None, cls_logits=cls_logits, train=False)
                    decoded_joints = decoded_joints.cpu().numpy()[0]
                    
                    # Calculate difference from original pose
                    joint_diff = np.linalg.norm(decoded_joints - original_joints, axis=1).mean()
                    
                    # Save variation result
                    variation = {
                        'joints': decoded_joints,
                        'tokens': modified_tokens.copy(),
                        'token_value': new_value,
                        'is_original': False,
                        'difference': joint_diff
                    }
                    variations.append(variation)
                    joint_data_list.append((f'variation_{i+1}_token_{new_value}', decoded_joints))
                    
                    print(f"   Generated variation {i+1}/{len(selected_values)}: token[{token_position}]={new_value}, diff={joint_diff:.4f}")
                    
                except Exception as e:
                    print(f"Warning: Failed to process token value {new_value}: {str(e)}")
                    continue
        
        if len(variations) <= 1:  # Only original pose
            raise RuntimeError("Failed to generate any token variations successfully")
        
        print(f"📈 Generated {len(variations)-1} successful variations")
        
        # Save joint data as npy files
        joints_dir = output_path / "joint_data"
        joints_dir.mkdir(exist_ok=True)
        
        for name, joints in joint_data_list:
            npy_path = joints_dir / f"{name}.npy"
            np.save(npy_path, joints)
            print(f"   Saved joint data: {npy_path}")
        
        # Create visualization
        self._create_paper_visualization(variations, token_position, original_token_value, output_path)
        
        # Save detailed information
        info_data = {
            'token_position': token_position,
            'original_token_value': int(original_token_value),
            'variations': [
                {
                    'token_value': int(var['token_value']),
                    'difference': float(var['difference']),
                    'is_original': var['is_original']
                }
                for var in variations
            ],
            'model_info': {
                'codebook_size': self.codebook_size,
                'token_num': self.token_num
            }
        }
        
        with open(output_path / "variations_info.json", 'w') as f:
            json.dump(info_data, f, indent=2)
        
        return variations
    
    def _create_paper_visualization(self, variations, token_position, original_token_value, output_path):
        """Create visualization images for paper"""
        
        print("🎨 Creating paper-ready visualizations...")
        
        try:
            # 1. Create main variation display
            poses_data = []
            titles = []
            
            # Sort by difference, but keep original pose first
            original_var = variations[0]  # First is original pose
            other_vars = sorted(variations[1:], key=lambda x: x['difference'])
            
            # Limit display count to ensure image clarity
            max_show = min(12, len(variations))  # Including original pose, display at most 12
            if max_show > 1:
                selected_vars = [original_var] + other_vars[:max_show-1]
            else:
                selected_vars = [original_var]
            
            for i, var in enumerate(selected_vars):
                poses_data.append((var['joints'], var['tokens']))
                if var['is_original']:
                    titles.append(f"Original\nToken[{token_position}]={var['token_value']}")
                else:
                    titles.append(f"Variation {i}\nToken[{token_position}]={var['token_value']}\nDiff: {var['difference']:.3f}")
            
            # Create multi-pose visualization
            n_cols = 4
            fig = plot_multiple_poses(poses_data, titles, n_cols=n_cols, figsize_per_plot=(4, 4))
            fig.suptitle(f"Token Position {token_position} Variations\nOriginal Value: {original_token_value}", 
                        fontsize=16, fontweight='bold')
            
            # Save high-resolution images for paper
            plt.figure(fig.number)
            plt.savefig(output_path / "paper_token_variations.png", dpi=300, bbox_inches='tight')
            plt.savefig(output_path / "paper_token_variations.pdf", bbox_inches='tight')
            
            print(f"📁 Paper visualization saved to: {output_path / 'paper_token_variations.png'}")
            print(f"📁 PDF version saved to: {output_path / 'paper_token_variations.pdf'}")
            
        except Exception as e:
            print(f"Warning: Failed to create paper visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        
        try:
            # 2. Create difference distribution plot
            print("   Creating difference distribution plot...")
            differences = [var['difference'] for var in variations[1:]]  # Exclude original pose
            token_values = [var['token_value'] for var in variations[1:]]
            
            if differences:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(differences)), differences, alpha=0.7, color='skyblue', edgecolor='navy')
                
                # Add value labels
                for i, (bar, diff) in enumerate(zip(bars, differences)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(differences)*0.01, 
                           f'{diff:.3f}', ha='center', va='bottom', fontsize=10)
                
                ax.set_xlabel('Variation Index')
                ax.set_ylabel('Pose Difference (Joint Distance)')
                ax.set_title(f'Token Position {token_position} Variation Impact\nOriginal Value: {original_token_value}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add token values as x-axis labels
                ax.set_xticks(range(len(token_values)))
                ax.set_xticklabels([f'T{tv}' for tv in token_values], rotation=45)
                
                plt.tight_layout()
                plt.savefig(output_path / "difference_distribution.png", dpi=300, bbox_inches='tight')
                plt.savefig(output_path / "difference_distribution.pdf", bbox_inches='tight')
                plt.close()
                
                print(f"📁 Difference plot saved to: {output_path / 'difference_distribution.png'}")
            
        except Exception as e:
            print(f"Warning: Failed to create difference plot: {str(e)}")
        
        try:
            # 3. Create color legend
            create_color_legend(output_path)
            
        except Exception as e:
            print(f"Warning: Failed to create color legend: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Paper Token Visualization Tool')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model weights path')
    parser.add_argument('--output-dir', type=str, default='paper_token_visualization', help='Output directory')
    parser.add_argument('--token-position', type=int, required=True, help='Token position to analyze')
    parser.add_argument('--n-variations', type=int, default=10, help='Number of variations to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate input files
    config_path = Path(args.config)
    model_path = Path(args.model)
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return 1
    
    if args.n_variations <= 0:
        print(f"❌ Number of variations must be positive")
        return 1
    
    print("🎨 Starting Paper Token Visualization...")
    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Token position: {args.token_position}")
    print(f"Number of variations: {args.n_variations}")
    
    try:
        # Create visualizer
        visualizer = PaperTokenVisualizer(
            model_path=str(model_path),
            config_path=str(config_path),
            device=args.device
        )
        
        # Get random sample
        joints, label, dataset_name, sample_idx, tokens, reconstructed = visualizer.get_random_sample(
            random_seed=args.random_seed
        )
        
        # Validate token position
        if args.token_position < 0 or args.token_position >= len(tokens):
            print(f"❌ Token position {args.token_position} is out of range [0, {len(tokens)-1}]")
            return 1
        
        print(f"\n📋 Selected sample info:")
        print(f"   Sample {sample_idx} from {dataset_name}")
        print(f"   Joints shape: {joints.shape}")
        print(f"   Tokens length: {len(tokens)}")
        print(f"   Token position {args.token_position} value: {tokens[args.token_position]}")
        
        # Generate token variations
        variations = visualizer.generate_token_variations(
            joints, tokens, args.token_position,
            n_variations=args.n_variations,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Paper token visualization completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Generated {len(variations)-1} variations (plus original)")
        print(f"💾 Joint data saved as .npy files in: {args.output_dir}/joint_data/")
        print(f"🎨 High-resolution images saved for paper use")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
