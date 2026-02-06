import faiss
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Reuse existing modules
from .analyze_token_distances import load_codebook_embeddings, load_tokens_from_h5, tokens_to_embeddings
from wicompass.knn_coverage.knn_coverage import build_index
from wicompass.model import create_joint_tokenizer
from wicompass.visualization import plot_single_pose
import mpl_toolkits.mplot3d.axes3d as p3d

# Use bone connections and joint names from evaluation package
from ..evaluation.core import BONE_CONNECTIONS, JOINT_NAMES, load_config

def find_knn_neighbors(data, query_indices, k=5, metric="cosine", gpu=False):
    """
    Find k-NN neighbors for specified query points
    
    Args:
        data: token data (N, dim)
        query_indices: List of indices of points to query
        k: Number of neighbors
        metric: Distance metric ('cosine' or 'l2')
        gpu: Whether to use GPU
    
    Returns:
        neighbor_info: Neighbor information for each query point
    """
    print(f"Building {metric} index for k-NN search...")
    index = build_index(data, metric, gpu)
    
    neighbor_info = {}
    for query_idx in tqdm(query_indices, desc="Finding k-NN neighbors"):
        if query_idx >= len(data):
            print(f"Warning: query_idx {query_idx} >= data length {len(data)}")
            continue
            
        query_point = data[query_idx:query_idx+1]  # (1, dim)
        
        # Normalize query point for cosine distance
        if metric.lower() == 'cosine':
            query_normalized = query_point.copy()
            faiss.normalize_L2(query_normalized)
            search_query = query_normalized
        else:
            search_query = query_point
            
        # Search for k+1 neighbors (including self)
        distances, indices = index.search(search_query, k + 1)
        
        # Remove self (if in results)
        neighbor_indices = []
        neighbor_distances = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx != query_idx:  # Exclude self
                neighbor_indices.append(int(idx))
                # For cosine distance, convert to cosine distance
                if metric.lower() == 'cosine':
                    dist = 1.0 - np.clip(dist, -1.0, 1.0)
                neighbor_distances.append(float(dist))
                
            if len(neighbor_indices) >= k:
                break
                
        neighbor_info[query_idx] = {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'query_token': data[query_idx].copy()
        }
    
    return neighbor_info

def tokens_to_poses(tokens, model, device):
    """
    Decode tokens to pose joint positions
    
    Args:
        tokens: token array (N, token_dim) or (N, num_tokens) 
        model: Trained VQ-VAE model
        device: Device
    
    Returns:
        poses: Decoded joint positions (N, num_joints, 3)
    """
    model.eval()
    poses = []
    
    with torch.no_grad():
        for i in tqdm(range(len(tokens)), desc="Decoding tokens to poses"):
            token_data = tokens[i:i+1]  # (1, token_dim or num_tokens)
            
            try:
                # Check token data shape
                if token_data.shape[1] == model.token_num:
                    # This is token indices (1, token_num), need to convert to quantized features
                    token_indices = torch.from_numpy(token_data).long().to(device)
                    
                    # Get corresponding embeddings from codebook
                    # token_indices: (1, token_num), each position is an index from 0-511
                    quantized_features = model.codebook[token_indices.flatten()]  # (token_num, token_dim)
                    quantized_features = quantized_features.unsqueeze(0)  # (1, token_num, token_dim)
                    
                elif token_data.shape[1] == model.token_dim:
                    # This is already quantized features (1, token_dim)
                    quantized_features = torch.from_numpy(token_data).float().to(device)
                    quantized_features = quantized_features.unsqueeze(1)  # (1, 1, token_dim)
                    
                    # Need to expand to token_num tokens
                    quantized_features = quantized_features.repeat(1, model.token_num, 1)
                else:
                    # Try to reshape it to (1, token_num, token_dim) shape
                    total_elements = token_data.shape[1]
                    if total_elements == model.token_num * model.token_dim:
                        token_tensor = torch.from_numpy(token_data).float().to(device)
                        quantized_features = token_tensor.view(1, model.token_num, model.token_dim)
                    else:
                        print(f"Warning: Unexpected token shape {token_data.shape}, using zero features")
                        quantized_features = torch.zeros(1, model.token_num, model.token_dim).to(device)
                
                # Use model's decode method
                recovered_joints = model.decode(quantized_features)
                poses.append(recovered_joints.cpu().numpy())
                
            except Exception as e:
                print(f"Error decoding token {i}: {e}")
                # Create zero pose as fallback
                zero_pose = torch.zeros(1, model.num_joints, 3).to(device)
                poses.append(zero_pose.cpu().numpy())
    
    return np.concatenate(poses, axis=0)

def calculate_pose_similarity_metrics(pose1, pose2):
    """
    Calculate similarity metrics between two poses
    
    Args:
        pose1, pose2: Joint positions (num_joints, 3)
    
    Returns:
        metrics: Similarity metrics dictionary
    """
    # 1. Joint position MSE
    joint_mse = np.mean((pose1 - pose2) ** 2)
    
    # 2. Bone length difference
    bone_length_errors = []
    for joint1_idx, joint2_idx in BONE_CONNECTIONS:
        bone1_len = np.linalg.norm(pose1[joint1_idx] - pose1[joint2_idx])
        bone2_len = np.linalg.norm(pose2[joint1_idx] - pose2[joint2_idx])
        bone_length_errors.append(abs(bone1_len - bone2_len))
    
    avg_bone_error = np.mean(bone_length_errors)
    
    # 3. Joint angle similarity (simplified: calculate cosine similarity of key joint vectors)
    def get_limb_vectors(pose):
        vectors = []
        limb_pairs = [
            (16, 18), (18, 20),  # Left arm
            (17, 19), (19, 21),  # Right arm
            (1, 4), (4, 7),      # Left leg
            (2, 5), (5, 8),      # Right leg
            (0, 3), (3, 6), (6, 9)  # Spine
        ]
        for start, end in limb_pairs:
            vec = pose[end] - pose[start]
            if np.linalg.norm(vec) > 1e-6:  # Avoid zero vector
                vectors.append(vec / np.linalg.norm(vec))
            else:
                vectors.append(np.zeros(3))
        return np.array(vectors)
    
    vectors1 = get_limb_vectors(pose1)
    vectors2 = get_limb_vectors(pose2)
    
    # Calculate cosine similarity between vector pairs
    cosine_similarities = []
    for v1, v2 in zip(vectors1, vectors2):
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_sim = np.dot(v1, v2)  # Already normalized
                cosine_similarities.append(cos_sim)
    
    avg_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else 0
    
    return {
        'joint_mse': float(joint_mse),
        'avg_bone_length_error': float(avg_bone_error),
        'avg_cosine_similarity': float(avg_cosine_similarity),
        'limb_cosine_similarities': [float(x) for x in cosine_similarities]
    }

def plot_pose_comparison(query_pose, neighbor_poses, neighbor_distances, 
                        query_idx, save_path, similarity_metrics=None):
    """
    Visualize comparison of query pose with its neighbor poses
    """
    n_neighbors = len(neighbor_poses)
    n_cols = min(4, n_neighbors + 1)  # At most 4 columns
    n_rows = (n_neighbors + 1 + n_cols - 1) // n_cols  # Round up
    
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    # Plot query pose
    ax1 = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    plot_single_pose(query_pose, ax1, f"Query (Index: {query_idx})")
    
    # Plot neighbor poses
    for i, (neighbor_pose, distance) in enumerate(zip(neighbor_poses, neighbor_distances)):
        ax = fig.add_subplot(n_rows, n_cols, i + 2, projection='3d')
        
        title = f"Neighbor {i+1}\nDist: {distance:.4f}"
        if similarity_metrics and i < len(similarity_metrics):
            metrics = similarity_metrics[i]
            title += f"\nMSE: {metrics['joint_mse']:.4f}"
            title += f"\nCos: {metrics['avg_cosine_similarity']:.3f}"
        
        plot_single_pose(neighbor_pose, ax, title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pose comparison saved to: {save_path}")

def analyze_knn_pose_similarity(tokens, model, device, query_indices, 
                               k=5, metric="cosine", output_dir="knn_pose_analysis"):
    """
    Main analysis function: analyze similarity of k-NN neighbors in pose space
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing k-NN pose similarity for {len(query_indices)} query points...")
    
    # 1. Find k-NN neighbors
    neighbor_info = find_knn_neighbors(tokens, query_indices, k, metric, gpu=False)
    
    # 2. Collect all token indices that need to be decoded
    all_indices = set(query_indices)
    for info in neighbor_info.values():
        all_indices.update(info['neighbor_indices'])
    all_indices = sorted(list(all_indices))
    
    # 3. Batch decode tokens to poses
    print(f"Decoding {len(all_indices)} tokens to poses...")
    selected_tokens = tokens[all_indices]
    decoded_poses = tokens_to_poses(selected_tokens, model, device)
    
    # Create index mapping
    index_to_pose = {idx: decoded_poses[i] for i, idx in enumerate(all_indices)}
    
    # 4. Analyze each query point
    analysis_results = {}
    
    for query_idx in tqdm(query_indices, desc="Analyzing pose similarities"):
        if query_idx not in neighbor_info:
            continue
            
        info = neighbor_info[query_idx]
        neighbor_indices = info['neighbor_indices']
        neighbor_distances = info['neighbor_distances']
        
        query_pose = index_to_pose[query_idx]
        neighbor_poses = [index_to_pose[idx] for idx in neighbor_indices]
        
        # Calculate pose similarity metrics
        similarity_metrics = []
        for neighbor_pose in neighbor_poses:
            metrics = calculate_pose_similarity_metrics(query_pose, neighbor_pose)
            similarity_metrics.append(metrics)
        
        # Save analysis results
        analysis_results[query_idx] = {
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'similarity_metrics': similarity_metrics,
            'token_distance_vs_pose_similarity': {
                'token_distances': neighbor_distances,
                'pose_mse_errors': [m['joint_mse'] for m in similarity_metrics],
                'pose_cosine_similarities': [m['avg_cosine_similarity'] for m in similarity_metrics],
                'bone_length_errors': [m['avg_bone_length_error'] for m in similarity_metrics]
            }
        }
        
        # Generate visualization
        plot_save_path = output_dir / f"pose_comparison_query_{query_idx}.png"
        plot_pose_comparison(query_pose, neighbor_poses, neighbor_distances, 
                           query_idx, plot_save_path, similarity_metrics)
    
    # 5. Generate summary report
    generate_summary_report(analysis_results, output_dir, metric)
    
    return analysis_results

def generate_summary_report(analysis_results, output_dir, metric):
    """Generate summary report"""
    
    # Collect all statistical data
    all_token_distances = []
    all_pose_mse_errors = []
    all_pose_cosine_sims = []
    all_bone_errors = []
    
    for query_idx, result in analysis_results.items():
        all_token_distances.extend(result['token_distance_vs_pose_similarity']['token_distances'])
        all_pose_mse_errors.extend(result['token_distance_vs_pose_similarity']['pose_mse_errors'])
        all_pose_cosine_sims.extend(result['token_distance_vs_pose_similarity']['pose_cosine_similarities'])
        all_bone_errors.extend(result['token_distance_vs_pose_similarity']['bone_length_errors'])
    
    # Calculate correlation
    token_dist_array = np.array(all_token_distances)
    pose_mse_array = np.array(all_pose_mse_errors)
    pose_cosine_array = np.array(all_pose_cosine_sims)
    bone_error_array = np.array(all_bone_errors)
    
    # Calculate Pearson correlation coefficient
    correlation_token_pose_mse = np.corrcoef(token_dist_array, pose_mse_array)[0, 1]
    correlation_token_pose_cosine = np.corrcoef(token_dist_array, pose_cosine_array)[0, 1]
    correlation_token_bone_error = np.corrcoef(token_dist_array, bone_error_array)[0, 1]
    
    # Generate report
    report = {
        'analysis_summary': {
            'total_queries': len(analysis_results),
            'metric_used': metric,
            'total_neighbor_pairs': len(all_token_distances)
        },
        'correlation_analysis': {
            'token_distance_vs_pose_mse': float(correlation_token_pose_mse),
            'token_distance_vs_pose_cosine_similarity': float(correlation_token_pose_cosine),
            'token_distance_vs_bone_length_error': float(correlation_token_bone_error)
        },
        'statistics': {
            'token_distances': {
                'mean': float(np.mean(token_dist_array)),
                'std': float(np.std(token_dist_array)),
                'min': float(np.min(token_dist_array)),
                'max': float(np.max(token_dist_array))
            },
            'pose_mse_errors': {
                'mean': float(np.mean(pose_mse_array)),
                'std': float(np.std(pose_mse_array)),
                'min': float(np.min(pose_mse_array)),
                'max': float(np.max(pose_mse_array))
            },
            'pose_cosine_similarities': {
                'mean': float(np.mean(pose_cosine_array)),
                'std': float(np.std(pose_cosine_array)),
                'min': float(np.min(pose_cosine_array)),
                'max': float(np.max(pose_cosine_array))
            }
        }
    }
    
    # Save report
    with open(output_dir / "analysis_summary_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate scatter plot
    plot_correlation_analysis(token_dist_array, pose_mse_array, pose_cosine_array, 
                             bone_error_array, output_dir, metric)
    
    # Print brief report
    print("\n" + "="*60)
    print("📊 k-NN Pose Similarity Analysis Summary")
    print("="*60)
    print(f"Analyzed {len(analysis_results)} query points with {metric} metric")
    print(f"Total neighbor pairs: {len(all_token_distances)}")
    print(f"\n📈 Correlation Analysis:")
    print(f"Token distance vs Pose MSE: {correlation_token_pose_mse:.4f}")
    print(f"Token distance vs Pose Cosine Similarity: {correlation_token_pose_cosine:.4f}")
    print(f"Token distance vs Bone Length Error: {correlation_token_bone_error:.4f}")
    print(f"\n💡 Interpretation:")
    if correlation_token_pose_mse > 0.5:
        print("✅ Strong positive correlation: Token similarity → Pose similarity")
    elif correlation_token_pose_mse > 0.3:
        print("⚠️  Moderate correlation: Token similarity somewhat reflects pose similarity")
    else:
        print("❌ Weak correlation: Token similarity poorly reflects pose similarity")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)

def plot_correlation_analysis(token_distances, pose_mse_errors, pose_cosine_sims, 
                            bone_errors, output_dir, metric):
    """Plot correlation analysis graph"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Token distance vs Pose MSE
    ax1 = axes[0, 0]
    ax1.scatter(token_distances, pose_mse_errors, alpha=0.6, s=20)
    ax1.set_xlabel(f'Token Distance ({metric})')
    ax1.set_ylabel('Pose MSE Error')
    ax1.set_title('Token Distance vs Pose MSE Error')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display correlation coefficient
    corr1 = np.corrcoef(token_distances, pose_mse_errors)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr1:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Token distance vs Pose Cosine Similarity
    ax2 = axes[0, 1]
    ax2.scatter(token_distances, pose_cosine_sims, alpha=0.6, s=20, color='green')
    ax2.set_xlabel(f'Token Distance ({metric})')
    ax2.set_ylabel('Pose Cosine Similarity')
    ax2.set_title('Token Distance vs Pose Cosine Similarity')
    ax2.grid(True, alpha=0.3)
    
    corr2 = np.corrcoef(token_distances, pose_cosine_sims)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {corr2:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Token distance vs Bone Length Error
    ax3 = axes[1, 0]
    ax3.scatter(token_distances, bone_errors, alpha=0.6, s=20, color='red')
    ax3.set_xlabel(f'Token Distance ({metric})')
    ax3.set_ylabel('Bone Length Error')
    ax3.set_title('Token Distance vs Bone Length Error')
    ax3.grid(True, alpha=0.3)
    
    corr3 = np.corrcoef(token_distances, bone_errors)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr3:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Comprehensive correlation bar chart
    ax4 = axes[1, 1]
    correlations = [corr1, corr2, corr3]
    labels = ['Token-MSE', 'Token-Cosine', 'Token-Bone']
    colors = ['blue', 'green', 'red']
    
    bars = ax4.bar(labels, correlations, color=colors, alpha=0.7)
    ax4.set_ylabel('Correlation Coefficient')
    ax4.set_title('Token Distance Correlations Summary')
    ax4.set_ylim([-1, 1])
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation analysis plot saved to: {output_dir / 'correlation_analysis.png'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze k-NN pose similarity in token space")
    parser.add_argument('--tokens', type=str, required=True, help='Path to token H5 file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained VQ-VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--query-indices', type=str, help='Query point indices (comma-separated), or "random" for random selection')
    parser.add_argument('--num-queries', type=int, default=10, help='Number of randomly selected query points')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors')
    parser.add_argument('--metric', choices=['cosine', 'l2'], default='cosine', help='Distance metric')
    parser.add_argument('--sample-ratio', type=float, default=0.01, help='Data sampling ratio')
    parser.add_argument('--output-dir', type=str, default='knn_pose_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    model_cfg = config.get('model', {})
    
    # Load model
    print("🚀 Loading VQ-VAE model...")
    model = create_joint_tokenizer(model_cfg)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load token data
    print(f"📊 Loading token data with sample ratio {args.sample_ratio}...")
    tokens, labels = load_tokens_from_h5(Path(args.tokens), args.sample_ratio)
    tokens = tokens.astype(np.float32)
    print(f"Loaded tokens shape: {tokens.shape}")
    
    # Deduplication
    print("🔄 Removing duplicate tokens...")
    original_count = len(tokens)
    
    # Use numpy's unique function for deduplication, return_index=True to preserve original indices
    unique_tokens, unique_indices = np.unique(tokens, axis=0, return_index=True)
    
    # Create mapping from original indices to deduplicated indices
    original_to_unique = {}
    for new_idx, old_idx in enumerate(unique_indices):
        original_to_unique[old_idx] = new_idx
    
    # Update tokens to deduplicated version
    tokens = unique_tokens
    duplicate_count = original_count - len(tokens)
    duplicate_rate = duplicate_count / original_count * 100
    
    print(f"Deduplication results:")
    print(f"  Original tokens: {original_count}")
    print(f"  Unique tokens: {len(tokens)}")
    print(f"  Duplicates removed: {duplicate_count} ({duplicate_rate:.2f}%)")
    print(f"  Final tokens shape: {tokens.shape}")
    
    # Determine query point indices (need to adapt to deduplicated data)
    if args.query_indices:
        if args.query_indices.lower() == 'random':
            query_indices = np.random.choice(len(tokens), args.num_queries, replace=False)
            print(f"Randomly selected query indices (from deduplicated data): {query_indices}")
        else:
            specified_indices = [int(x.strip()) for x in args.query_indices.split(',')]
            # Map original indices to deduplicated indices
            query_indices = []
            for orig_idx in specified_indices:
                if orig_idx in original_to_unique:
                    query_indices.append(original_to_unique[orig_idx])
                else:
                    print(f"Warning: Original index {orig_idx} was removed during deduplication")
            
            if not query_indices:
                print("No valid query indices after deduplication, using random selection")
                query_indices = np.random.choice(len(tokens), min(args.num_queries, len(tokens)), replace=False)
            
            print(f"Using mapped query indices: {query_indices}")
    else:
        query_indices = np.random.choice(len(tokens), args.num_queries, replace=False)
        print(f"Using random query indices: {query_indices}")
    
    # Execute analysis
    results = analyze_knn_pose_similarity(
        tokens, model, device, query_indices,
        k=args.k, metric=args.metric, output_dir=args.output_dir
    )
    
    print(f"✅ Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
