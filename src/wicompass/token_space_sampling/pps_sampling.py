# -*- coding: utf-8 -*-
"""
Capped-PPS Sampling (Probability Proportional to Size)

Points in sparser regions (higher k-NN radius) have higher selection probability,
enabling better coverage of the latent space while avoiding over-sampling of outliers.

Uses Efraimidis-Spirakis PPSWOR algorithm for efficient weighted sampling:
    priority_i = -log(u_i) / weight_i, select K smallest priorities

Usage:
    python src/wicompass/token_space_sampling/pps_sampling.py --A-path tokens.h5 --budget 40000 --k 8 --cap-quantile 0.9
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import faiss

# Add src directory to path for direct script execution
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wicompass.token_space_sampling.base import (
    load_data,
    dedupe_rows,
    map_to_unique_space,
    compute_quantile,
    compute_knn_radius,
    save_results,
    get_device_info,
)


def sample_capped_pps(
    candidates: np.ndarray,
    weights: np.ndarray,
    budget: int,
    cap_quantile: float = 0.95,
    seeds: Optional[np.ndarray] = None,
    random_seed: int = 42,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Capped-PPS sampling using Efraimidis-Spirakis PPSWOR algorithm.
    
    Args:
        candidates: Candidate indices
        weights: Weight for each point (indexed by candidates)
        budget: Number of samples to select
        cap_quantile: Quantile for weight capping (default 0.95)
        seeds: Seed indices to include first (optional)
        random_seed: Random seed
        eps: Small constant to prevent zero weights
        
    Returns:
        selected: Selected indices
        cap_value: The actual cap threshold used
    """
    rng = np.random.default_rng(random_seed)
    n_total = len(weights)
    
    picked = []
    is_picked = np.zeros(n_total, dtype=bool)
    
    # Include seeds first
    if seeds is not None and len(seeds) > 0:
        seeds = np.unique(seeds)
        if len(seeds) > budget:
            seeds = seeds[:budget]
        picked.extend(seeds.tolist())
        is_picked[seeds] = True
    
    remaining = budget - len(picked)
    if remaining <= 0:
        return np.asarray(picked, dtype=np.int64), 0.0
    
    # Filter already picked
    mask = ~is_picked[candidates]
    cand = candidates[mask]
    if len(cand) == 0:
        return np.asarray(picked, dtype=np.int64), 0.0
    
    # Compute capped weights
    cand_weights = weights[cand]
    cap_value = compute_quantile(cand_weights, cap_quantile)
    capped = np.minimum(cand_weights, cap_value).astype(np.float64)
    capped = np.maximum(capped, eps)
    
    # Efraimidis-Spirakis: priority = -log(u) / weight
    priorities = -np.log(rng.random(len(cand))) / capped
    
    # Select smallest K priorities
    k = min(remaining, len(cand))
    selected_idx = np.argpartition(priorities, kth=k-1)[:k]
    picked.extend(cand[selected_idx].tolist())
    
    return np.asarray(picked, dtype=np.int64), cap_value


def main():
    ap = argparse.ArgumentParser(
        description="Capped-PPS sampling on token data with deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pps_sampling.py --A-path tokens.h5 --budget 40000 --k 16
  python pps_sampling.py --A-path tokens.h5 --budget 40000 --cap-quantile 0.9 --multi-gpu
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
    ap.add_argument("--k", type=int, default=16,
                    help="k for k-NN radius (default: 16)")
    ap.add_argument("--metric", choices=["cosine", "l2"], default="cosine",
                    help="Distance metric (default: cosine)")
    ap.add_argument("--cap-quantile", type=float, default=0.95,
                    help="Quantile for weight capping (default: 0.95)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--search-batch", type=int, default=8192,
                    help="Batch size for k-NN search (default: 8192)")
    
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
    
    # Output
    ap.add_argument("--out-dir", default="",
                    help="Output directory (default: derived from input)")
    
    args = ap.parse_args()
    
    # Validate
    if not 0 < args.cap_quantile <= 1:
        raise ValueError(f"--cap-quantile must be in (0, 1], got {args.cap_quantile}")
    
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
    candidates, seeds = map_to_unique_space(
        orig_to_unique, args.candidates_npy, args.init_indices_npy, N0
    )
    if candidates is None:
        candidates = np.arange(Nu, dtype=np.int64)
    
    # Device setup
    use_gpu = (not args.cpu_only) and (faiss.get_num_gpus() > 0)
    device_info = get_device_info(use_gpu, args.multi_gpu, args.gpu_id)
    print(f"🚀 Using {device_info}")
    
    # Compute k-NN radius
    print(f"⏳ Computing k-NN radius (k={args.k}, metric={args.metric})...")
    r_k = compute_knn_radius(
        data_unique, args.k, args.metric,
        use_gpu, args.gpu_id, args.multi_gpu,
        args.search_batch, verbose=True
    )
    print(f"   r_k: min={r_k.min():.4f}, max={r_k.max():.4f}, "
          f"mean={r_k.mean():.4f}, std={r_k.std():.4f}")
    
    # Sample
    print(f"🎲 Capped-PPS sampling {args.budget:,} (cap_quantile={args.cap_quantile})...")
    selected, cap_value = sample_capped_pps(
        candidates, r_k, args.budget,
        cap_quantile=args.cap_quantile,
        seeds=seeds,
        random_seed=args.seed
    )
    print(f"   Selected: {len(selected):,}, cap_value={cap_value:.6f}")
    
    # Save
    out_dir = Path(args.out_dir) if args.out_dir else A_path.parent / f"{A_path.stem}_pps"
    print(f"💾 Output: {out_dir}")
    
    meta = {
        "input_path": str(A_path.resolve()),
        "sample_ratio": args.sample_ratio,
        "original_size": N0,
        "unique_size": Nu,
        "duplicates_removed": N0 - Nu,
        "budget": args.budget,
        "selected_count": len(selected),
        "method": "capped-pps",
        "k": args.k,
        "metric": args.metric,
        "cap_quantile": args.cap_quantile,
        "cap_value": float(cap_value),
        "seed": args.seed,
        "device": device_info,
    }
    
    save_results(selected, data_unique, unique_first_idx, format_info, out_dir, meta, prefix="capped_pps")
    print("✅ Done!")


if __name__ == "__main__":
    main()

