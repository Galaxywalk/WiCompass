# -*- coding: utf-8 -*-
"""
Farthest Point Sampling (FPS) with Deduplication

Iteratively selects the point that is farthest from all already-selected points.
Uses cosine distance metric with FAISS acceleration.

Features:
- Optional outlier filtering based on k-NN radius (removes top 5% by default)
- Batch processing for efficiency
- GPU acceleration support

Usage:
    python src/wicompass/token_space_sampling/fps_sampling.py --A-path tokens.h5 --budget 5000 --multi-gpu
    python src/wicompass/token_space_sampling/fps_sampling.py --A-path tokens.h5 --budget 5000 --remove-outliers
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import faiss

# Add src directory to path for direct script execution
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wicompass.token_space_sampling.base import (
    load_data,
    dedupe_rows,
    to_indices,
    map_to_unique_space,
    compute_quantile,
    compute_knn_radius,
    save_results,
    get_device_info,
)


def _build_ip_index(d: int, use_gpu: bool, gpu_id: int, multi_gpu: bool) -> faiss.Index:
    """Build inner product index with optional GPU acceleration."""
    index = faiss.IndexFlatIP(d)
    if use_gpu and faiss.get_num_gpus() > 0:
        if multi_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    return index


def farthest_point_sampling(
    X: np.ndarray,
    budget: int,
    candidates: np.ndarray = None,
    init_indices: np.ndarray = None,
    batch: int = 64,
    use_gpu: bool = True,
    gpu_id: int = 0,
    multi_gpu: bool = True,
    recompute_only_remain: bool = True,
    verbose: bool = True,
) -> List[int]:
    """
    Farthest Point Sampling with cosine distance.
    
    Args:
        X: Data matrix (N, d), will be L2-normalized internally
        budget: Number of samples to select
        candidates: Candidate indices (None = all)
        init_indices: Seed indices to start with
        batch: Batch size for point selection
        use_gpu: Whether to use GPU
        gpu_id: GPU device ID
        multi_gpu: Whether to use all GPUs
        recompute_only_remain: If True, only recompute distances for remaining points
        verbose: Whether to print progress
        
    Returns:
        List of selected indices
    """
    Nu, d = X.shape
    X = X.astype(np.float32, copy=True)
    faiss.normalize_L2(X)  # Cosine metric
    
    candidates = to_indices(candidates, Nu)
    if len(candidates) == 0:
        if verbose:
            print("[FPS] No candidates to sample from.")
        return []
    
    index_sel = _build_ip_index(d, use_gpu, gpu_id, multi_gpu)
    selected = []
    
    # Initialize with seed points or centroid-farthest point
    if init_indices is not None and len(init_indices) > 0:
        seeds = to_indices(init_indices, Nu)
        index_sel.add(X[seeds])
        selected.extend(seeds.tolist())
    else:
        # Use farthest point from centroid as first seed
        centroid = X[candidates].mean(0, keepdims=True)
        faiss.normalize_L2(centroid)
        sims = X[candidates] @ centroid[0]
        seed = candidates[np.argmin(sims)]
        index_sel.add(X[seed:seed+1])
        selected.append(int(seed))
    
    # Compute initial nearest distances (cosine distance = 1 - similarity)
    D, _ = index_sel.search(X[candidates], 1)
    min_dist = 1.0 - np.clip(D.ravel(), -1.0, 1.0)
    
    is_picked = np.zeros(Nu, dtype=bool)
    is_picked[selected] = True
    
    # Iteratively select farthest points
    while len(selected) < budget:
        remain_mask = ~is_picked[candidates]
        if not np.any(remain_mask):
            break
        
        remain_pos = np.flatnonzero(remain_mask)
        b = min(batch, len(remain_pos))
        
        # Select batch of farthest points
        far_local = np.argpartition(min_dist[remain_pos], -b)[-b:]
        new_pos = remain_pos[far_local]
        new_sel = candidates[new_pos]
        
        if len(selected) + len(new_sel) > budget:
            new_sel = new_sel[:budget - len(selected)]
        
        index_sel.add(X[new_sel])
        is_picked[new_sel] = True
        selected.extend(new_sel.tolist())
        
        # Update nearest distances
        if recompute_only_remain:
            remain_mask = ~is_picked[candidates]
            if np.any(remain_mask):
                D, _ = index_sel.search(X[candidates[remain_mask]], 1)
                min_dist[remain_mask] = 1.0 - np.clip(D.ravel(), -1.0, 1.0)
        else:
            D, _ = index_sel.search(X[candidates], 1)
            min_dist = 1.0 - np.clip(D.ravel(), -1.0, 1.0)
        
        if verbose:
            max_dist = float(min_dist[~is_picked[candidates]].max()) if np.any(~is_picked[candidates]) else 0.0
            print(f"[FPS] picked={len(selected)} / {budget}  max(min_dist)={max_dist:.4f}")
    
    return selected


def filter_outliers_by_radius(
    X: np.ndarray,
    candidates: np.ndarray,
    k: int = 16,
    metric: str = "cosine",
    quantile: float = 0.95,
    use_gpu: bool = True,
    gpu_id: int = 0,
    multi_gpu: bool = True,
    batch_size: int = 8192,
    verbose: bool = True,
) -> np.ndarray:
    """
    Filter out outliers based on k-NN radius.
    
    Args:
        X: Data matrix
        candidates: Candidate indices
        k: k for k-NN
        metric: Distance metric
        quantile: Quantile threshold (default 0.95 = remove top 5%)
        use_gpu: GPU acceleration
        gpu_id: GPU device ID
        multi_gpu: Use all GPUs
        batch_size: Batch size for k-NN computation
        verbose: Print progress
        
    Returns:
        Filtered candidate indices
    """
    r_k = compute_knn_radius(X, k, metric, use_gpu, gpu_id, multi_gpu, batch_size, verbose=False)
    
    r_k_candidates = r_k[candidates]
    threshold = compute_quantile(r_k_candidates, quantile)
    
    keep_mask = r_k_candidates < threshold
    filtered = candidates[keep_mask]
    
    if verbose:
        n_removed = len(candidates) - len(filtered)
        print(f"🔄 Outlier filtering: threshold={threshold:.4f}")
        print(f"   candidates={len(candidates):,}, kept={len(filtered):,}, "
              f"removed={n_removed:,} ({100*n_removed/max(len(candidates),1):.1f}%)")
    
    return filtered


def main():
    ap = argparse.ArgumentParser(
        description="Farthest Point Sampling on token data with deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fps_sampling.py --A-path tokens.h5 --budget 5000 --multi-gpu
  python fps_sampling.py --A-path tokens.h5 --budget 5000 --remove-outliers --batch 128
        """
    )
    
    # Input
    ap.add_argument("--A-path", dest="A_path", required=True,
                    help="Input .h5 or .npy file")
    ap.add_argument("--sample-ratio", type=float, default=1.0,
                    help="Sampling ratio for .h5 loading (default: 1.0)")
    
    # Sampling
    ap.add_argument("--budget", type=int, required=True,
                    help="Number of samples to select")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size for FPS selection (default: 64)")
    
    # Outlier filtering
    ap.add_argument("--remove-outliers", action="store_true",
                    help="Remove top 5%% radius outliers before sampling")
    ap.add_argument("--outlier-k", type=int, default=16,
                    help="k for k-NN radius in outlier detection (default: 16)")
    ap.add_argument("--outlier-metric", choices=["cosine", "l2"], default="cosine",
                    help="Distance metric for outlier detection (default: cosine)")
    
    # Candidates/seeds
    ap.add_argument("--candidates-npy",
                    help="Optional: .npy with candidate indices/mask (original space)")
    ap.add_argument("--init-indices-npy",
                    help="Optional: .npy with seed indices (original space)")
    
    # Device
    ap.add_argument("--cpu-only", action="store_true",
                    help="Force CPU-only computation")
    ap.add_argument("--gpu-id", type=int, default=0,
                    help="GPU device ID (default: 0)")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="Use all available GPUs")
    ap.add_argument("--recompute-only-remain", action="store_true",
                    help="Only recompute distances for remaining candidates")
    
    # Output
    ap.add_argument("--out-dir", default="",
                    help="Output directory (default: derived from input)")
    
    args = ap.parse_args()
    
    # Load data
    A_path = Path(args.A_path)
    print(f"📂 Loading: {A_path}")
    data, format_info = load_data(A_path, args.sample_ratio)
    N0 = data.shape[0]
    print(f"   Shape: {data.shape}, dtype: {format_info['dtype']}")
    
    # Deduplicate
    print("🔄 Deduplicating...")
    data_unique, unique_first_idx, orig_to_unique = dedupe_rows(data)
    Nu = len(data_unique)
    print(f"   {N0:,} -> {Nu:,} unique ({100*(N0-Nu)/max(N0,1):.1f}% duplicates)")
    
    # Map candidates and seeds
    candidates, init_indices = map_to_unique_space(
        orig_to_unique, args.candidates_npy, args.init_indices_npy, N0
    )
    if candidates is None:
        candidates = np.arange(Nu, dtype=np.int64)
    
    # Device setup
    use_gpu = (not args.cpu_only) and (faiss.get_num_gpus() > 0)
    device_info = get_device_info(use_gpu, args.multi_gpu, args.gpu_id)
    print(f"🚀 Using {device_info}")
    
    # Outlier filtering
    outlier_info = {"enabled": False}
    if args.remove_outliers:
        print("🔍 Filtering outliers...")
        candidates_before = len(candidates)
        candidates = filter_outliers_by_radius(
            data_unique, candidates,
            k=args.outlier_k,
            metric=args.outlier_metric,
            use_gpu=use_gpu,
            gpu_id=args.gpu_id,
            multi_gpu=args.multi_gpu,
            verbose=True
        )
        outlier_info = {
            "enabled": True,
            "k": args.outlier_k,
            "metric": args.outlier_metric,
            "candidates_before": candidates_before,
            "candidates_after": len(candidates),
        }
        
        # Filter seeds if they became outliers
        if init_indices is not None:
            valid_mask = np.isin(init_indices, candidates)
            if not np.all(valid_mask):
                n_removed = np.sum(~valid_mask)
                print(f"⚠️  {n_removed} seeds removed as outliers")
            init_indices = init_indices[valid_mask] if np.any(valid_mask) else None
    
    # FPS
    print(f"🎲 Farthest Point Sampling {args.budget:,}...")
    selected = farthest_point_sampling(
        X=data_unique,
        budget=args.budget,
        candidates=candidates,
        init_indices=init_indices,
        batch=args.batch,
        use_gpu=use_gpu,
        gpu_id=args.gpu_id,
        multi_gpu=args.multi_gpu,
        recompute_only_remain=args.recompute_only_remain,
        verbose=True,
    )
    selected = np.asarray(selected, dtype=np.int64)
    
    # Save
    out_dir = Path(args.out_dir) if args.out_dir else A_path.parent / f"{A_path.stem}_fps"
    print(f"💾 Output: {out_dir}")
    
    meta = {
        "input_path": str(A_path.resolve()),
        "sample_ratio": args.sample_ratio,
        "original_size": N0,
        "unique_size": Nu,
        "duplicates_removed": N0 - Nu,
        "budget": args.budget,
        "selected_count": len(selected),
        "method": "farthest_point_sampling",
        "metric": "cosine",
        "batch": args.batch,
        "outlier_filtering": outlier_info,
        "device": device_info,
        "recompute_only_remain": args.recompute_only_remain,
    }
    
    save_results(selected, data_unique, unique_first_idx, format_info, out_dir, meta, prefix="fps")
    print("✅ Done!")


if __name__ == "__main__":
    main()

