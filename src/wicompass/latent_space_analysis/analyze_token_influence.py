#!/usr/bin/env python3
"""
Token Influence Analysis Tool
Analyze the influence of individual tokens on pose reconstruction
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from pathlib import Path

from wicompass.model import create_joint_tokenizer
from wicompass.dataset import create_dataset
from ..evaluation.core import load_config, get_dataset_configs
from wicompass.visualization import (
    plot_single_pose, plot_pose_comparison, plot_multiple_poses,
    save_pose_plot, create_color_legend
)


class TokenInfluenceAnalyzer:
    """Token influence analyzer"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize analyzer
        
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
    
    def get_sample_from_dataset(self, sample_idx: int = None, random_seed: int = 42):
        """
        Get a sample from dataset
        
        Args:
            sample_idx: Specify sample index, if None then randomly select
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
        
        # Select sample
        if sample_idx is None:
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
    
    def analyze_all_tokens_influence(self, original_joints, original_tokens, 
                                   value_range: list = None, output_dir: str = "token_influence"):
        """
        Analyze influence of all token positions on pose, only generate overview images
        
        Args:
            original_joints: Original joint data
            original_tokens: Original token sequence
            value_range: Range of token values to vary, if None then use full codebook range
            output_dir: Output directory
            
        Returns:
            Analysis results dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"🔍 Analyzing influence for all {len(original_tokens)} token positions")
        
        # Determine token value variation range
        if value_range is None:
            # Use all possible values in codebook, but limit count for visualization
            step = max(1, self.codebook_size // 15)  # At most 15 different values
            value_range = list(range(0, self.codebook_size, step))
            print(f"Using {len(value_range)} values from codebook (step={step})")
        
        # Validate and filter token value range
        valid_value_range = []
        for v in value_range:
            if 0 <= v < self.codebook_size:
                valid_value_range.append(v)
            else:
                print(f"Warning: Token value {v} is out of range [0, {self.codebook_size}), skipping")
        
        value_range = valid_value_range
        if not value_range:
            raise ValueError(f"No valid token values found in range. Codebook size is {self.codebook_size}")
        
        # Analyze influence for each token position
        all_results = {}
        all_overview_poses = []  # Store most representative poses for each position
        
        for token_position in range(len(original_tokens)):
            original_token_value = original_tokens[token_position]
            print(f"🔄 Analyzing token position {token_position}/{len(original_tokens)-1} (original value: {original_token_value})")
            
            # Ensure original value is included in range
            current_value_range = value_range.copy()
            if original_token_value not in current_value_range:
                if 0 <= original_token_value < self.codebook_size:
                    current_value_range.append(original_token_value)
                    current_value_range.sort()
                else:
                    print(f"Warning: Original token value {original_token_value} is out of valid range")
            
            # Generate modified poses
            modified_poses = []
            pose_differences = []
            
            with torch.no_grad():
                for new_value in current_value_range:
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
                        
                        modified_poses.append((decoded_joints, modified_tokens.copy(), new_value, joint_diff))
                        pose_differences.append(joint_diff)
                        
                    except Exception as e:
                        print(f"Warning: Failed to process token value {new_value} at position {token_position}: {str(e)}")
                        continue
            
            if not modified_poses:
                print(f"Warning: Failed to process any token values for position {token_position}")
                continue
            
            # Sort and select most representative poses
            sorted_indices = np.argsort(pose_differences)
            
            # Select poses with largest differences as representatives
            n_represent = min(3, len(modified_poses))  # Select at most 3 representative poses per position
            if n_represent >= 2:
                # Select minimum and maximum differences
                selected_indices = [sorted_indices[0], sorted_indices[-1]]
                if n_represent == 3:
                    # Add medium difference
                    mid_idx = len(sorted_indices) // 2
                    selected_indices.append(sorted_indices[mid_idx])
            else:
                selected_indices = [sorted_indices[0]]
            
            # Select most influential pose (largest difference)
            max_diff_idx = sorted_indices[-1]
            best_pose = modified_poses[max_diff_idx]
            all_overview_poses.append((token_position, best_pose, original_token_value))
            
            # Save analysis results for this position
            analysis_result = {
                'token_position': token_position,
                'original_token_value': int(original_token_value),
                'mean_difference': float(np.mean(pose_differences)),
                'std_difference': float(np.std(pose_differences)),
                'max_difference': float(np.max(pose_differences)),
                'min_difference': float(np.min(pose_differences)),
                'most_influential_value': int(best_pose[2])
            }
            all_results[token_position] = analysis_result
        
        # Create overall overview visualization
        self._create_all_tokens_overview(original_joints, original_tokens, all_overview_poses, output_path)
        
        # Save all analysis results
        with open(output_path / "all_tokens_analysis.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        all_max_diffs = [result['max_difference'] for result in all_results.values()]
        print(f"📊 Analysis completed for {len(all_results)} token positions:")
        print(f"   Overall max difference: {max(all_max_diffs):.4f}")
        print(f"   Overall mean difference: {np.mean(all_max_diffs):.4f}")
        print(f"   Most influential position: {np.argmax(all_max_diffs)}")
        
        return all_results
    
    def _create_all_tokens_overview(self, original_joints, original_tokens, all_overview_poses, output_path):
        """Create overview visualization for all token position influences"""
        
        print("🎨 Creating overview visualization for all token positions...")
        
        try:
            # Sort by influence
            sorted_poses = sorted(all_overview_poses, key=lambda x: x[1][3], reverse=True)  # Sort by difference
            
            # Select top 12 most influential token positions for visualization
            n_show = min(12, len(sorted_poses))
            top_poses = sorted_poses[:n_show]
            
            poses_data = []
            titles = []
            
            # Add original pose as reference
            poses_data.append((original_joints, original_tokens))
            titles.append("Original\nPose")
            
            for token_pos, (decoded_joints, modified_tokens, new_value, joint_diff), original_value in top_poses:
                poses_data.append((decoded_joints, modified_tokens))
                titles.append(f"Token[{token_pos}]\n{original_value}→{new_value}\nDiff: {joint_diff:.3f}")
            
            # Create grid layout visualization
            n_cols = 4
            n_rows = (len(poses_data) + n_cols - 1) // n_cols
            
            from wicompass.visualization import plot_multiple_poses, save_pose_plot
            
            fig = plot_multiple_poses(poses_data, titles, n_cols=n_cols, figsize_per_plot=(4, 4))
            fig.suptitle(f"Token Influence Analysis Overview\nTop {n_show} Most Influential Tokens", 
                        fontsize=16, fontweight='bold')
            
            save_pose_plot(fig, output_path / "all_tokens_influence_overview.png")
            
            print(f"📁 Overview visualization saved to: {output_path / 'all_tokens_influence_overview.png'}")
            
        except Exception as e:
            print(f"Warning: Failed to create overview visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _visualize_token_influence(self, original_joints, original_tokens, token_position, 
                                 original_token_value, selected_poses, all_differences, output_path):
        """Create visualization of token influence"""
        
        print("🎨 Creating visualizations...")
        
        try:
            # 1. Create comparison of original pose vs reconstructed pose
            print("   Creating original vs reconstructed comparison...")
            original_reconstructed = None
            with torch.no_grad():
                joints_tensor = torch.from_numpy(original_joints).float().unsqueeze(0).to(self.device)
                joints_visible = torch.ones(joints_tensor.shape[:2], device=self.device).bool()
                reconstructed_joints, _, _ = self.model(joints=joints_tensor, joints_visible=joints_visible, train=False)
                original_reconstructed = reconstructed_joints.cpu().numpy()[0]
            
            fig = plot_pose_comparison(
                original_joints, original_reconstructed,
                "Original Pose", "Original Reconstructed",
                original_tokens, original_tokens
            )
            save_pose_plot(fig, output_path / "original_comparison.png")
            
        except Exception as e:
            print(f"Warning: Failed to create original comparison: {str(e)}")
        
        try:
            # 2. Create overview of token modification effects
            print("   Creating token influence overview...")
            poses_data = []
            titles = []
            
            for decoded_joints, modified_tokens, new_value, joint_diff in selected_poses:
                poses_data.append((decoded_joints, modified_tokens))
                titles.append(f"Token[{token_position}]={new_value}\nDiff: {joint_diff:.3f}")
            
            if poses_data:
                fig = plot_multiple_poses(poses_data, titles, n_cols=3, figsize_per_plot=(5, 5))
                fig.suptitle(f"Token Position {token_position} Influence Analysis\n"
                            f"Original Value: {original_token_value}", fontsize=16, fontweight='bold')
                save_pose_plot(fig, output_path / f"token_{token_position}_influence_overview.png")
            
        except Exception as e:
            print(f"Warning: Failed to create influence overview: {str(e)}")
        
        try:
            # 3. Create comparison of maximum difference
            print("   Creating max difference comparison...")
            if selected_poses:
                max_diff_idx = np.argmax([pose[3] for pose in selected_poses])
                max_diff_pose = selected_poses[max_diff_idx]
                
                fig = plot_pose_comparison(
                    original_joints, max_diff_pose[0],
                    f"Original (Token[{token_position}]={original_token_value})",
                    f"Modified (Token[{token_position}]={max_diff_pose[2]})",
                    original_tokens, max_diff_pose[1]
                )
                save_pose_plot(fig, output_path / f"token_{token_position}_max_difference.png")
            
        except Exception as e:
            print(f"Warning: Failed to create max difference comparison: {str(e)}")
        
        try:
            # 4. Create difference distribution plot
            print("   Creating difference curve...")
            import matplotlib.pyplot as plt
            if all_differences:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(all_differences, 'b-', alpha=0.7, linewidth=2)
                mean_diff = np.mean(all_differences)
                ax.axhline(y=mean_diff, color='r', linestyle='--', label=f'Mean: {mean_diff:.3f}')
                ax.set_xlabel('Token Value Index')
                ax.set_ylabel('Pose Difference (Joint Distance)')
                ax.set_title(f'Token Position {token_position} Influence on Pose\n'
                            f'Original Value: {original_token_value}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                save_pose_plot(fig, output_path / f"token_{token_position}_difference_curve.png")
            
        except Exception as e:
            print(f"Warning: Failed to create difference curve: {str(e)}")
        
        try:
            # 5. Create color legend
            print("   Creating color legend...")
            create_color_legend(output_path)
            
        except Exception as e:
            print(f"Warning: Failed to create color legend: {str(e)}")
        
        print(f"📁 Visualizations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Token Influence Analysis Tool')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model weights path')
    parser.add_argument('--output-dir', type=str, default='token_influence_analysis', help='Output directory')
    parser.add_argument('--sample-idx', type=int, help='Specify sample index (if not specified then randomly select)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--value-range', type=str, help='Token value range, format like "0,100,10" means from 0 to 100 with step 10')
    
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
    
    print("🔍 Starting Token Influence Analysis for All Positions...")
    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    
    # Parse value_range
    value_range = None
    if args.value_range:
        try:
            parts = args.value_range.split(',')
            if len(parts) == 3:
                start, end, step = map(int, parts)
                if step <= 0:
                    print("❌ Step must be positive")
                    return 1
                value_range = list(range(start, end + 1, step))
                print(f"Using custom value range: {start}-{end} step {step} ({len(value_range)} values)")
            else:
                print("❌ Invalid value-range format. Use 'start,end,step'")
                return 1
        except ValueError as e:
            print(f"❌ Error parsing value-range: {e}")
            return 1
    
    try:
        # Create analyzer
        analyzer = TokenInfluenceAnalyzer(
            model_path=str(model_path),
            config_path=str(config_path),
            device=args.device
        )
        
        # Get dataset sample
        joints, label, dataset_name, sample_idx, tokens, reconstructed = analyzer.get_sample_from_dataset(
            sample_idx=args.sample_idx,
            random_seed=args.random_seed
        )
        
        print(f"\n📋 Selected sample info:")
        print(f"   Sample {sample_idx} from {dataset_name}")
        print(f"   Joints shape: {joints.shape}")
        print(f"   Tokens length: {len(tokens)}")
        print(f"   Will analyze all {len(tokens)} token positions")
        
        # Execute influence analysis for all token positions
        results = analyzer.analyze_all_tokens_influence(
            joints, tokens,
            value_range=value_range,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Token influence analysis completed for all positions!")
        print(f"📁 Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
