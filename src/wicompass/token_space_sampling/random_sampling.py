# -*- coding: utf-8 -*-
"""
Random Sampling with Deduplication

Simple random sampling without replacement on deduplicated token space.

Usage:
    python src/wicompass/token_space_sampling/random_sampling.py --A-path tokens.h5 --budget 500 --seed 123
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src directory to path for direct script execution
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wicompass.token_space_sampling.base import (
    load_data,
    dedupe_rows,
    to_indices,
    map_to_unique_space,
    save_results,
)


def random_sampling(
    Nu: int,
    budget: int,
    candidates: np.ndarray = None,
    init_indices: np.ndarray = None,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Random sampling without replacement on deduplicated index domain [0, Nu).
    
    Args:
        Nu: Number of unique points
        budget: Number of samples to select
        candidates: Candidate indices (None = all)
        init_indices: Seed indices to include first
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Selected indices (seeds first, then random)
    """
    rng = np.random.default_rng(seed)
    candidates = to_indices(candidates, Nu)
    
    if len(candidates) == 0:
        if verbose:
            print("[RANDOM] No candidates to sample from.")
        return np.array([], dtype=np.int64)
    
    selected = []
    is_picked = np.zeros(Nu, dtype=bool)
    
    # Include seeds first
    if init_indices is not None and len(init_indices) > 0:
        seeds = to_indices(init_indices, Nu)
        if len(seeds) > budget:
            seeds = seeds[:budget]
        selected.extend(seeds.tolist())
        is_picked[seeds] = True
        if verbose:
            print(f"[RANDOM] Seeded: {len(seeds)} / budget={budget}")
    
    # Fill remaining budget randomly
    remain_budget = budget - len(selected)
    if remain_budget <= 0:
        return np.asarray(selected, dtype=np.int64)
    
    # Exclude already picked from candidates
    mask_remain = ~is_picked[candidates]
    remain_positions = np.flatnonzero(mask_remain)
    
    if len(remain_positions) == 0:
        if verbose:
            print("[RANDOM] No remaining candidates after seeding.")
        return np.asarray(selected, dtype=np.int64)
    
    k = min(remain_budget, len(remain_positions))
    picked_pos = rng.choice(remain_positions, size=k, replace=False)
    picked_idx = candidates[picked_pos]
    selected.extend(picked_idx.tolist())
    
    if verbose:
        print(f"[RANDOM] Total selected: {len(selected)} / {budget}")
    
    return np.asarray(selected, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(
        description="Random sampling on token data with deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input
    ap.add_argument("--A-path", dest="A_path", required=True,
                    help="Input .h5 or .npy file")
    ap.add_argument("--sample-ratio", type=float, default=1.0,
                    help="Sampling ratio for .h5 loading (default: 1.0)")
    
    # Sampling
    ap.add_argument("--budget", type=int, required=True,
                    help="Number of samples to select")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    
    # Candidates/seeds
    ap.add_argument("--candidates-npy",
                    help="Optional: .npy with candidate indices/mask (original space)")
    ap.add_argument("--init-indices-npy",
                    help="Optional: .npy with seed indices (original space)")
    
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
    
    # Map candidates and seeds to unique space
    candidates, init_indices = map_to_unique_space(
        orig_to_unique, args.candidates_npy, args.init_indices_npy, N0
    )
    
    # Sample
    print(f"🎲 Random sampling {args.budget:,}...")
    selected = random_sampling(
        Nu=Nu,
        budget=args.budget,
        candidates=candidates,
        init_indices=init_indices,
        seed=args.seed,
        verbose=True,
    )
    
    # Save
    out_dir = Path(args.out_dir) if args.out_dir else A_path.parent / f"{A_path.stem}_random"
    print(f"💾 Output: {out_dir}")
    
    meta = {
        "input_path": str(A_path.resolve()),
        "sample_ratio": args.sample_ratio,
        "original_size": N0,
        "unique_size": Nu,
        "duplicates_removed": N0 - Nu,
        "budget": args.budget,
        "selected_count": len(selected),
        "method": "random_no_replacement",
        "seed": args.seed,
    }
    
    save_results(selected, data_unique, unique_first_idx, format_info, out_dir, meta, prefix="rand")
    print("✅ Done!")


if __name__ == "__main__":
    main()

