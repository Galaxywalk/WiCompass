# -*- coding: utf-8 -*-
"""
Common utilities for token space sampling methods.

This module provides shared functionality for:
- Data loading (H5/NPY files)
- Row deduplication
- Index manipulation
- Result saving with format preservation
- FAISS index building
"""

import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import faiss

# Add src directory to path for direct script execution
_src_dir = str(Path(__file__).parent.parent.parent)  # src/wicompass/token_space_sampling -> src
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wicompass.latent_space_analysis.analyze_token_distances import load_tokens_from_h5


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path: Path, sample_ratio: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load latent vectors from .npy or .h5 file.
    
    Args:
        path: Path to input file
        sample_ratio: Sampling ratio for .h5 files (default 1.0 = full dataset)
        
    Returns:
        data: Float32 matrix of shape (N, d)
        format_info: Metadata for preserving output format
    """
    suffix = path.suffix.lower()
    
    if suffix == ".npy":
        data = np.load(path)
        format_info = {"type": "npy", "dtype": str(data.dtype), "original_data": data.copy()}
    elif suffix == ".h5":
        data, labels = load_tokens_from_h5(path, sample_ratio)
        format_info = {
            "type": "h5_tokens",
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "labels": labels,
        }
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .npy or .h5")
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D (N, d), got shape {data.shape}")
    
    return data.astype(np.float32, copy=False), format_info


def load_indices_from_npy(path: str, max_idx: int) -> np.ndarray:
    """
    Load indices from .npy file (supports bool mask or int array).
    
    Args:
        path: Path to .npy file
        max_idx: Maximum valid index (exclusive)
        
    Returns:
        Valid indices as int64 array
    """
    arr = np.load(path)
    if arr.dtype == bool:
        return np.where(arr)[0].astype(np.int64)
    idx = arr.astype(np.int64).ravel()
    return idx[(idx >= 0) & (idx < max_idx)]


# =============================================================================
# Deduplication and Index Utilities
# =============================================================================

def dedupe_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicate rows in a 2D array.
    
    Args:
        X: Input matrix of shape (N, d)
        
    Returns:
        X_unique: Deduplicated matrix of shape (M, d) where M <= N
        first_idx: Indices of first occurrence of each unique row in original X
        inv_idx: Mapping from original indices to unique indices
    """
    X_unique, first_idx = np.unique(X, axis=0, return_index=True)
    _, inv_idx = np.unique(X, axis=0, return_inverse=True)
    return X_unique, first_idx.astype(np.int64), inv_idx.astype(np.int64)


def to_indices(arr, N: int) -> np.ndarray:
    """
    Convert bool mask or index array to sorted unique indices within [0, N).
    
    Args:
        arr: Bool mask, index array, or None
        N: Maximum valid index (exclusive)
        
    Returns:
        Sorted unique indices as int64 array
    """
    if arr is None:
        return np.arange(N, dtype=np.int64)
    arr = np.asarray(arr)
    if arr.dtype == bool:
        idx = np.where(arr)[0].astype(np.int64)
    else:
        idx = arr.astype(np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < N)]
    return np.unique(idx)


def map_to_unique_space(
    orig_to_unique: np.ndarray,
    candidates_npy: Optional[str],
    init_indices_npy: Optional[str],
    N_original: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Map candidate and seed indices from original space to deduplicated space.
    
    Args:
        orig_to_unique: Mapping from original to unique indices
        candidates_npy: Path to candidates .npy file (or None)
        init_indices_npy: Path to initial seeds .npy file (or None)
        N_original: Original data size
        
    Returns:
        candidates: Unique candidate indices (or None)
        init_indices: Unique seed indices (or None)
    """
    candidates = None
    if candidates_npy:
        cand_orig = load_indices_from_npy(candidates_npy, N_original)
        candidates = np.unique(orig_to_unique[cand_orig])
    
    init_indices = None
    if init_indices_npy:
        seeds_orig = load_indices_from_npy(init_indices_npy, N_original)
        init_indices = np.unique(orig_to_unique[seeds_orig])
    
    return candidates, init_indices


# =============================================================================
# Math Utilities
# =============================================================================

def compute_quantile(values: np.ndarray, q: float) -> float:
    """Compute quantile, compatible with different numpy versions."""
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="linear"))


# =============================================================================
# FAISS Utilities
# =============================================================================

def build_faiss_index(
    X: np.ndarray,
    metric: str,
    use_gpu: bool = True,
    gpu_id: int = 0,
    multi_gpu: bool = True
) -> faiss.Index:
    """
    Build FAISS index for similarity search.
    
    Args:
        X: Data matrix of shape (N, d)
        metric: Distance metric ('cosine' or 'l2')
        use_gpu: Whether to use GPU acceleration
        gpu_id: GPU device ID (if not multi_gpu)
        multi_gpu: Whether to use all available GPUs
        
    Returns:
        FAISS index with data added
    """
    d = X.shape[1]
    X_idx = X.copy()
    
    if metric == "cosine":
        faiss.normalize_L2(X_idx)
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"metric must be 'cosine' or 'l2', got '{metric}'")
    
    if use_gpu and faiss.get_num_gpus() > 0:
        if multi_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    
    index.add(X_idx)
    return index


def compute_knn_radius(
    X: np.ndarray,
    k: int,
    metric: str = "cosine",
    use_gpu: bool = True,
    gpu_id: int = 0,
    multi_gpu: bool = True,
    batch_size: int = 8192,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute k-NN radius (distance to k-th nearest neighbor) for each point.
    Points in sparser regions have larger values.
    
    Args:
        X: Data matrix of shape (N, d)
        k: Number of neighbors
        metric: Distance metric ('cosine' or 'l2')
        use_gpu: Whether to use GPU
        gpu_id: GPU device ID
        multi_gpu: Whether to use all GPUs
        batch_size: Batch size for search
        verbose: Whether to print progress
        
    Returns:
        r_k: Array of k-NN radius for each point
    """
    N = X.shape[0]
    Xq = X.copy()
    if metric == "cosine":
        faiss.normalize_L2(Xq)
    
    index = build_faiss_index(X, metric, use_gpu, gpu_id, multi_gpu)
    r_k = np.empty(N, dtype=np.float32)
    search_k = k + 1  # +1 for self-match
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        D, _ = index.search(Xq[start:end], search_k)
        
        # Convert to distance
        if metric == "cosine":
            r_k[start:end] = 1.0 - np.clip(D[:, -1], -1.0, 1.0)
        else:
            r_k[start:end] = np.sqrt(np.maximum(D[:, -1], 0.0))
        
        if verbose and (start // batch_size) % 50 == 0:
            print(f"  [k-NN radius] Processed {end:,}/{N:,} ({100*end/N:.1f}%)")
    
    return r_k


# =============================================================================
# Result Saving
# =============================================================================

def save_results(
    selected: np.ndarray,
    data_unique: np.ndarray,
    unique_first_idx: np.ndarray,
    format_info: Dict[str, Any],
    out_dir: Path,
    meta: Dict[str, Any],
    prefix: str = "sampled"
) -> None:
    """
    Save selected samples and metadata.
    
    Args:
        selected: Selected indices in deduplicated space
        data_unique: Deduplicated data matrix
        unique_first_idx: Mapping from unique to original indices
        format_info: Original format information
        out_dir: Output directory
        meta: Metadata dictionary
        prefix: Filename prefix for outputs
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vec_file = f"{prefix}_selected_vectors.npy"
    lab_file = f"{prefix}_selected_labels.npy"
    meta_file = f"{prefix}_meta.json"
    
    # Save vectors with original dtype
    selected_data = data_unique[selected]
    if "original_data" in format_info:
        save_dtype = format_info["original_data"].dtype
    else:
        save_dtype = format_info["dtype"]
    
    np.save(out_dir / vec_file, selected_data.astype(save_dtype))
    print(f"✅ Saved vectors ({save_dtype}): {selected_data.shape} -> {out_dir / vec_file}")
    
    # Save labels if available
    if "labels" in format_info and format_info["labels"] is not None:
        original_idx = unique_first_idx[selected]
        labels = format_info["labels"][original_idx]
        np.save(out_dir / lab_file, labels)
        print(f"✅ Saved labels: {labels.shape} -> {out_dir / lab_file}")
    
    # Save metadata (ensure JSON-safe)
    safe_fmt = {k: v for k, v in format_info.items() if k not in ("labels", "original_data")}
    meta["input_format"] = safe_fmt
    meta["outputs"] = {
        vec_file: "Selected tokens (same dtype as input)",
        lab_file: "Corresponding labels (if available)"
    }
    
    with open(out_dir / meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata saved -> {out_dir / meta_file}")


def get_device_info(use_gpu: bool, multi_gpu: bool, gpu_id: int) -> str:
    """Get device description string for metadata."""
    if use_gpu and faiss.get_num_gpus() > 0:
        if multi_gpu:
            return f"multi-GPU ({faiss.get_num_gpus()})"
        return f"GPU {gpu_id}"
    return "CPU"

