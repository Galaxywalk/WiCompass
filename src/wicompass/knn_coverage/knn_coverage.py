"""
k-NN Coverage Analysis for Token Embeddings

Usage:
    python src/wicompass/knn_coverage/knn_coverage.py \
        --dataset-A logs/wicompass/encoded_tokens/BMLmovi-BMLrub-CMU-GRAB-KIT-MOYO-MoSh-PosePrior-WEIZMANN_tokens.h5 \
        --dataset-B logs/wicompass/encoded_tokens/MMBody_tokens.h5 \
        --metric cosine --k 10 --multi-gpu \
        --sample-ratio 1.0 \
        --output logs/wicompass/knn_coverage/mmbody/k10

Default token dataset location: logs/wicompass/encoded_tokens/

The script always runs both deduplication and non-deduplication versions, 
generating separate output files for each.
"""

import faiss
import argparse, json
from pathlib import Path
import numpy as np
import torch, h5py
import sys

# Store GPU resources globally to prevent garbage collection
_gpu_resources = {}

# Threshold for storing per-neighbor data (indices/distances arrays)
# When k > K_STORAGE_THRESHOLD, intra_knn data (size O(N*k)) will not be stored
# to avoid excessive memory and disk usage
K_STORAGE_THRESHOLD = 16

# Import from latent_space_analysis module
from wicompass.latent_space_analysis.analyze_token_distances import load_codebook_embeddings, load_tokens_from_h5, tokens_to_embeddings
from tqdm import tqdm

def _prepare_query_data(data: np.ndarray, metric: str) -> np.ndarray:
    """Prepare query data for FAISS search (normalize for cosine metric)"""
    query_data = np.ascontiguousarray(data, dtype=np.float32)
    if metric.lower() == 'cosine':
        faiss.normalize_L2(query_data)
    return query_data


def _convert_distances(D: np.ndarray, metric: str) -> np.ndarray:
    """Convert FAISS distances to actual distances (handle cosine metric)"""
    if metric.lower() == 'cosine':
        # FAISS returns inner product for cosine, convert to distance
        return 1.0 - np.clip(D, -1.0, 1.0)
    return D


def _save_h5_dataset(group, name: str, data, dtype, compression="gzip", compression_opts=4):
    """Helper to save dataset with consistent compression settings"""
    group.create_dataset(
        name,
        data=np.asarray(data, dtype=dtype),
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )


def build_index(data: np.ndarray, metric: str, gpu: bool, gpu_id: int = None, multi_gpu: bool = False):
    """Build FAISS index for similarity search"""
    dim = data.shape[1]
    data_copy = _prepare_query_data(data.copy(), metric)
    
    if metric.lower() == 'l2':
        cpu = faiss.IndexFlatL2(dim)
    elif metric.lower() == 'cosine':
        cpu = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    cpu.add(data_copy)
    
    if not gpu or faiss.get_num_gpus() == 0:
        print(f"🖥️  Using CPU index")
        return cpu
    
    try:
        if multi_gpu:
            # Use all available GPUs (original behavior)
            gpu_index = faiss.index_cpu_to_all_gpus(cpu)
            num_gpus = faiss.get_num_gpus()
            print(f"✅ Successfully created multi-GPU index using {num_gpus} GPUs")
            return gpu_index
        else:
            # Use specified single GPU
            # Store gpu_res globally to prevent garbage collection
            target_gpu = gpu_id or 0
            if target_gpu not in _gpu_resources:
                _gpu_resources[target_gpu] = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(_gpu_resources[target_gpu], target_gpu, cpu)
            print(f"✅ Successfully created GPU index on GPU {target_gpu}")
            return gpu_index
    except Exception as e:
        print(f"⚠️  Failed to create GPU index: {e}")
        print("Falling back to CPU index")
        return cpu

def intra_knn_search(data, k, index, metric="cosine", batch=200_000):
    """
    Compute intra-dataset k-NN for each point (vectorized implementation).
    Returns all k neighbor indices and all k distances, excluding self.
    
    Args:
        data: (N, d) data matrix
        k: number of neighbors
        index: FAISS index built from the same data
        metric: 'cosine' or 'l2'
        batch: batch size for processing
        
    Returns:
        indices: (N, k) array of neighbor indices (excluding self, -1 if not enough neighbors)
        distances: (N, k) array of distances to neighbors (inf if not enough neighbors)
    """
    N = len(data)
    k_eff = min(k, N - 1)  # Cannot have more neighbors than N-1
    
    # Initialize output arrays
    all_indices = np.full((N, k_eff), -1, dtype=np.int32)
    all_distances = np.full((N, k_eff), np.inf, dtype=np.float32)
    
    query_data = _prepare_query_data(data, metric)
    
    for i in tqdm(range(0, N, batch), desc=f"Computing intra-{k}-NN"):
        end_idx = min(i + batch, N)
        block = query_data[i:end_idx]
        batch_size = end_idx - i
        global_indices = np.arange(i, end_idx)
        
        # Search for k+2 neighbors (extra buffer for self-exclusion)
        D, I = index.search(block, k_eff + 2)
        
        # Vectorized self-exclusion: create mask where neighbor != self
        self_mask = I != global_indices[:, np.newaxis]  # (batch_size, k+2)
        
        # For each row, select first k_eff non-self neighbors
        for j in range(batch_size):
            valid_mask = self_mask[j] & (np.abs(D[j]) <= 1e38)  # Exclude sentinel
            valid_indices = np.where(valid_mask)[0][:k_eff]
            n_valid = len(valid_indices)
            
            if n_valid > 0:
                all_indices[i + j, :n_valid] = I[j, valid_indices]
                all_distances[i + j, :n_valid] = D[j, valid_indices]
        
        # Convert distances for cosine metric (batch operation)
        all_distances[i:end_idx] = _convert_distances(all_distances[i:end_idx], metric)
        all_distances[i:end_idx] = np.clip(all_distances[i:end_idx], 0, None)
    
    return all_indices, all_distances

def coverage_flags(query, target_index, radii, batch=200_000, return_indices=False):
    """Check coverage status for query points against target index"""
    N = len(query)
    flags = np.zeros(N, dtype=bool)
    nearest_indices = np.zeros(N, dtype=np.int32) if return_indices else None
    
    is_cosine = (target_index.metric_type == faiss.METRIC_INNER_PRODUCT)
    metric = "cosine" if is_cosine else "l2"
    query_data = _prepare_query_data(query, metric)
    
    for i in tqdm(range(0, N, batch), desc="coverage checking"):
        end_idx = min(i + batch, N)
        block = query_data[i:end_idx]
        R = radii[i:end_idx]
        
        D, I = target_index.search(block, 1)
        D = D.ravel()
        
        if is_cosine:
            # For cosine: similarity >= (1 - radius) means covered
            flags[i:end_idx] = D >= (1.0 - R)
        else:
            # For L2: distance <= radius means covered
            flags[i:end_idx] = D <= R
        
        if return_indices:
            nearest_indices[i:end_idx] = I.ravel()
    
    return (flags, nearest_indices.tolist()) if return_indices else flags

def analyze_knn_coverage(A, B, k, metric, gpu, batch, gpu_id=None, multi_gpu=False):
    """Analyze k-NN coverage between two datasets"""
    print(f"Building indices for {metric} metric...")
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"Faiss GPUs available: {faiss.get_num_gpus()}")
    
    idxA = build_index(A.copy(), metric, gpu, gpu_id, multi_gpu)
    idxB = build_index(B.copy(), metric, gpu, gpu_id, multi_gpu)

    # Compute intra-dataset k-NN for A (all k neighbors and distances)
    print("Computing intra-dataset k-NN in A...")
    intra_A_indices, intra_A_distances = intra_knn_search(A, k, idxA, metric, batch)
    radA = intra_A_distances[:, -1] if intra_A_distances.shape[1] > 0 else np.zeros(len(A), dtype=np.float32)
    print(f"A radii stats: mean={radA.mean():.6f}, max={radA.max():.6f}, nonzero={np.count_nonzero(radA)}")
    
    print("Checking B covers A...")
    flagA, knn_indices_A_to_B = coverage_flags(A, idxB, radA, batch, return_indices=True)

    # Compute intra-dataset k-NN for B (all k neighbors and distances)
    print("Computing intra-dataset k-NN in B...")
    intra_B_indices, intra_B_distances = intra_knn_search(B, k, idxB, metric, batch)
    radB = intra_B_distances[:, -1] if intra_B_distances.shape[1] > 0 else np.zeros(len(B), dtype=np.float32)
    print(f"B radii stats: mean={radB.mean():.6f}, max={radB.max():.6f}, nonzero={np.count_nonzero(radB)}")
    
    print("Checking A covers B...")
    flagB, knn_indices_B_to_A = coverage_flags(B, idxA, radB, batch, return_indices=True)

    def stat(r, f):
        return {
            "radius_mean": float(r.mean()),
            "radius_p95":  float(np.percentile(r, 95)),
            "radius_max":  float(r.max()),
            "holes": int((~f).sum()),
            "hole_rate": float((~f).mean()) }

    # Find indices of covered and uncovered data points
    covered_A_indices = np.where(flagA)[0]  # Indices of points in A covered by B
    uncovered_A_indices = np.where(~flagA)[0]  # Indices of points in A not covered by B
    covered_B_indices = np.where(flagB)[0]  # Indices of points in B covered by A
    uncovered_B_indices = np.where(~flagB)[0]  # Indices of points in B not covered by A

    result = {
        "A_to_B": stat(radA, flagA), 
        "B_to_A": stat(radB, flagB),
        # Coverage index information
        "coverage_indices": {
            "A_covered_by_B": covered_A_indices.tolist(),
            "A_uncovered_by_B": uncovered_A_indices.tolist(),
            "B_covered_by_A": covered_B_indices.tolist(),
            "B_uncovered_by_A": uncovered_B_indices.tolist(),
        },
        "knn_mappings": {
            "A_to_B_nearest": knn_indices_A_to_B,  # Nearest neighbor index in B for each point in A
            "B_to_A_nearest": knn_indices_B_to_A,  # Nearest neighbor index in A for each point in B
        },
        "radii": {
            "A_knn_radii": radA.tolist(),
            "B_knn_radii": radB.tolist(),
        },
    }
    
    # Only store intra-kNN data when k <= K_STORAGE_THRESHOLD to avoid excessive memory/disk usage
    # intra_knn arrays have size O(N * k), which becomes very large for large k
    if k <= K_STORAGE_THRESHOLD:
        result["intra_knn"] = {
            "A_indices": intra_A_indices,   # (N_A, k) - k nearest neighbor indices within A
            "A_distances": intra_A_distances,  # (N_A, k) - distances to k nearest neighbors within A
            "B_indices": intra_B_indices,   # (N_B, k) - k nearest neighbor indices within B
            "B_distances": intra_B_distances,  # (N_B, k) - distances to k nearest neighbors within B
        }
    else:
        print(f"⚠️  k={k} > {K_STORAGE_THRESHOLD}: Skipping intra_knn storage to save memory/disk space")
    
    return result

def generate_output_filename(dataset_A_path, dataset_B_path, metric, k, sample_ratio, dedup=True):
    """Generate output filename based on dataset names and deduplication mode"""
    # Extract dataset names (remove path and extension)
    name_A = Path(dataset_A_path).stem.replace('_tokens', '').replace('dataset_tokens', '')
    name_B = Path(dataset_B_path).stem.replace('_tokens', '').replace('dataset_tokens', '')
    
    # Add dedup suffix to distinguish files
    dedup_suffix = "dedup" if dedup else "nodedup"
    
    # Generate descriptive filename (without extension, .json and .h5 will be added later)
    filename = f"knn_coverage_{name_A}_vs_{name_B}_{metric}_k{k}_sr{sample_ratio:.3f}_{dedup_suffix}"
    return filename

def save_coverage_results(
    results,
    json_path,
    h5_path,
    *,
    A_raw=None,
    B_raw=None,
    compression="gzip",
    compression_opts=4,
):
    """
    Save k-NN coverage results to JSON (statistics) and HDF5 (large arrays).
    """
    # ---- Statistics (write to JSON) ----
    stats_data = {
        "A_to_B": results["A_to_B"],
        "B_to_A": results["B_to_A"],
        "metadata": results["metadata"],
    }
    if "uncovered_counts" in results:
        stats_data["uncovered_counts"] = results["uncovered_counts"]

    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"✅ Statistics saved to: {json_path}")

    # ---- Large arrays (write to HDF5) ----
    def save_group(f, group_name, data_dict, dtype):
        """Helper to save a group of datasets"""
        grp = f.create_group(group_name)
        for key, data in data_dict.items():
            _save_h5_dataset(grp, key, data, dtype, compression, compression_opts)
    
    with h5py.File(h5_path, "w") as f:
        # Save coverage indices, k-NN mappings, radii
        save_group(f, "coverage_indices", results["coverage_indices"], np.int32)
        save_group(f, "knn_mappings", results["knn_mappings"], np.int32)
        save_group(f, "radii", results["radii"], np.float32)
        
        # Save intra-dataset k-NN (mixed dtypes)
        if "intra_knn" in results:
            intra_grp = f.create_group("intra_knn")
            for key, arr in results["intra_knn"].items():
                dtype = np.int32 if "indices" in key else np.float32
                _save_h5_dataset(intra_grp, key, arr, dtype, compression, compression_opts)
        
        # Save processed raw data
        proc_grp = f.create_group("processed/raw")
        if A_raw is not None:
            _save_h5_dataset(proc_grp, "A", A_raw, A_raw.dtype, compression, compression_opts)
        if B_raw is not None:
            _save_h5_dataset(proc_grp, "B", B_raw, B_raw.dtype, compression, compression_opts)

        # Metadata attributes
        meta = results["metadata"]
        for key in ["dataset_A", "dataset_B", "metric", "k", "sample_ratio"]:
            f.attrs[key] = meta[key]
        f.attrs["deduplication_applied"] = str(meta.get("deduplication_applied", False))
        f.attrs["final_shape_A"] = str(meta["final_shapes"]["A"])
        f.attrs["final_shape_B"] = str(meta["final_shapes"]["B"])

    print(f"✅ Array data saved to: {h5_path}")


def load_coverage_results(json_path, h5_path, load_intra_knn=True):
    """
    Load k-NN coverage results from JSON and HDF5 files
    
    Args:
        json_path: path to JSON file with statistics
        h5_path: path to HDF5 file with array data
        load_intra_knn: whether to load intra-dataset k-NN data (can be large)
    """
    # Load statistics
    with open(json_path, "r") as f:
        stats_data = json.load(f)
    
    # Load large array data
    with h5py.File(h5_path, "r") as f:
        array_data = {
            "coverage_indices": {},
            "knn_mappings": {},
            "radii": {}
        }
        
        # Read coverage indices
        coverage_grp = f["coverage_indices"]
        for key in coverage_grp.keys():
            array_data["coverage_indices"][key] = coverage_grp[key][:].tolist()
        
        # Read k-NN mappings
        knn_grp = f["knn_mappings"]
        for key in knn_grp.keys():
            array_data["knn_mappings"][key] = knn_grp[key][:].tolist()
        
        # Read radii arrays
        radii_grp = f["radii"]
        for key in radii_grp.keys():
            array_data["radii"][key] = radii_grp[key][:].tolist()
        
        # Read intra-dataset k-NN data (optional, can be large)
        if load_intra_knn and "intra_knn" in f:
            array_data["intra_knn"] = {}
            intra_grp = f["intra_knn"]
            for key in intra_grp.keys():
                array_data["intra_knn"][key] = intra_grp[key][:]  # Keep as numpy array
    
    # Merge data
    results = {**stats_data, **array_data}
    return results

def example_load_and_analyze():
    """
    Example: How to load and analyze saved k-NN coverage results
    """
    json_path = "knn_coverage_results.json"
    h5_path = "knn_coverage_results.h5"
    
    try:
        results = load_coverage_results(json_path, h5_path)
        
        print("📈 Loaded k-NN Coverage Analysis Results:")
        print(f"Dataset A: {results['metadata']['dataset_A']}")
        print(f"Dataset B: {results['metadata']['dataset_B']}")
        print(f"Metric: {results['metadata']['metric']}, k={results['metadata']['k']}")
        
        # Analyze coverage status
        a_covered = results['coverage_indices']['A_covered_by_B']
        a_uncovered = results['coverage_indices']['A_uncovered_by_B']
        print(f"\nA→B Coverage: {len(a_covered)} covered, {len(a_uncovered)} uncovered")
        
        # Analyze nearest neighbor mappings
        a_to_b_nearest = results['knn_mappings']['A_to_B_nearest']
        print(f"A→B nearest neighbor mappings: {len(a_to_b_nearest)} entries")
        
        return results
    except FileNotFoundError as e:
        print(f"Files not found: {e}")
        return None


def run_single_analysis(A_tok_input, B_tok_input, args, use_gpu, gpu_id, multi_gpu, 
                        do_dedup, output_base=None):
    """
    Run a single k-NN coverage analysis with specified deduplication setting.
    
    Args:
        A_tok_input: Original token data for dataset A
        B_tok_input: Original token data for dataset B
        args: Command line arguments
        use_gpu, gpu_id, multi_gpu: GPU settings
        do_dedup: Whether to perform deduplication
        output_base: Base path for output files (without extension)
        
    Returns:
        tuple: (json_output_path, h5_output_path)
    """
    dedup_mode_str = "dedup" if do_dedup else "nodedup"
    print(f"\n{'='*60}")
    print(f"🔄 Running analysis with mode: {dedup_mode_str.upper()}")
    print(f"{'='*60}")
    
    # Generate output filenames
    if output_base:
        json_output = f"{output_base}_{dedup_mode_str}.json"
        h5_output = f"{output_base}_{dedup_mode_str}.h5"
    else:
        base_filename = generate_output_filename(
            args.dataset_A, args.dataset_B, 
            args.metric, args.k, args.sample_ratio, dedup=do_dedup
        )
        json_output = f"{base_filename}.json"
        h5_output = f"{base_filename}.h5"
    
    print(f"Output files:")
    print(f"  Statistics: {json_output}")
    print(f"  Array data: {h5_output}")
    
    # Process data
    original_A_count, original_B_count = len(A_tok_input), len(B_tok_input)
    
    if do_dedup:
        print("🔄 Performing deduplication...")
        # Deduplicate dataset A
        A_unique, _ = np.unique(A_tok_input, axis=0, return_index=True)
        A_duplicates = original_A_count - len(A_unique)
        A_duplicate_rate = A_duplicates / original_A_count * 100
        
        # Deduplicate dataset B
        B_unique, _ = np.unique(B_tok_input, axis=0, return_index=True)
        B_duplicates = original_B_count - len(B_unique)
        B_duplicate_rate = B_duplicates / original_B_count * 100
        
        print(f"Dataset A: {original_A_count} → {len(A_unique)} ({A_duplicate_rate:.2f}% duplicates removed)")
        print(f"Dataset B: {original_B_count} → {len(B_unique)} ({B_duplicate_rate:.2f}% duplicates removed)")
    else:
        print("⏭️  Skipping deduplication...")
        A_unique, B_unique = A_tok_input, B_tok_input
        A_duplicates = B_duplicates = 0
        A_duplicate_rate = B_duplicate_rate = 0.0
        print(f"Dataset A: {original_A_count} samples")
        print(f"Dataset B: {original_B_count} samples")
    
    # Convert to analysis format
    if args.metric == "l2":
        if not args.model: 
            raise ValueError("--model required for embedding l2")
        print("Converting tokens to embeddings...")
        device_str = f"cuda:{gpu_id}" if use_gpu and not multi_gpu else "cpu"
        codebook = load_codebook_embeddings(Path(args.model), device=device_str)
        A = tokens_to_embeddings(A_unique, codebook, use_fp8=False)
        B = tokens_to_embeddings(B_unique, codebook, use_fp8=False)
        print(f"Embeddings A: {A.shape}, B: {B.shape}")
    else:  # cosine
        A = A_unique.astype("float32")
        B = B_unique.astype("float32")
        print(f"Using tokens directly A: {A.shape}, B: {B.shape}")

    # Run analysis
    res = analyze_knn_coverage(
        A, B, k=args.k, metric=args.metric,
        gpu=use_gpu, batch=args.chunk_size, 
        gpu_id=gpu_id, multi_gpu=multi_gpu)

    # Add metadata
    res["metadata"] = {
        "dataset_A": str(Path(args.dataset_A).name),
        "dataset_B": str(Path(args.dataset_B).name),
        "dataset_A_path": str(args.dataset_A),
        "dataset_B_path": str(args.dataset_B),
        "metric": args.metric,
        "k": args.k,
        "sample_ratio": args.sample_ratio,
        "deduplication_applied": do_dedup,
        "gpu_used": f"Multi-GPU ({torch.cuda.device_count()})" if multi_gpu else (gpu_id if gpu_id is not None else "CPU"),
        "original_counts": {
            "A": original_A_count,
            "B": original_B_count
        },
        "unique_counts": {
            "A": len(A_unique),
            "B": len(B_unique)
        },
        "duplicate_rates": {
            "A": A_duplicate_rate,
            "B": B_duplicate_rate
        },
        "final_shapes": {
            "A": A.shape,
            "B": B.shape
        }
    }

    # Print summary
    print(f"\n📊 k-NN Coverage Summary ({dedup_mode_str})")
    print(f"📁 Dataset A: {res['metadata']['dataset_A']}")
    print(f"📁 Dataset B: {res['metadata']['dataset_B']}")
    print("📈 Coverage Statistics:")
    stats_only = {k: v for k, v in res.items() 
                  if k not in ["coverage_indices", "knn_mappings", "radii", "metadata", "intra_knn"]}
    print(json.dumps(stats_only, indent=2))
    
    # Save results
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(h5_output).parent.mkdir(parents=True, exist_ok=True)
    
    save_coverage_results(
        res,
        json_output,
        h5_output,
        A_raw=A_unique,
        B_raw=B_unique,
        compression="gzip",
        compression_opts=4,
    )
    
    # Print file info
    print(f"\n✅ Files generated for {dedup_mode_str}:")
    print(f"  📄 Statistics (JSON): {json_output}")
    print(f"  🗃️  Array data (HDF5): {h5_output}")
    print(f"Array data sizes:")
    print(f"  Coverage indices: {sum(len(indices) for indices in res['coverage_indices'].values())} integers")
    print(f"  k-NN mappings: {sum(len(mapping) for mapping in res['knn_mappings'].values())} integers") 
    print(f"  Radii arrays: {sum(len(radii) for radii in res['radii'].values())} floats")
    if "intra_knn" in res:
        intra_size = sum(arr.size for arr in res['intra_knn'].values())
        print(f"  Intra-kNN data: {intra_size} elements (indices + distances)")
    else:
        print(f"  Intra-kNN data: Not stored (k={args.k} > {K_STORAGE_THRESHOLD})")
    
    return json_output, h5_output


def _setup_gpu(args) -> tuple:
    """Setup GPU configuration and return (use_gpu, gpu_id, multi_gpu)"""
    if args.cpu_only:
        print("🖥️  Using CPU only")
        return False, None, False
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        return False, None, False
    
    if args.gpu_id is not None:
        if args.gpu_id >= torch.cuda.device_count():
            print(f"⚠️  GPU {args.gpu_id} not available. Available: 0-{torch.cuda.device_count()-1}")
            print("Using multi-GPU instead.")
            return True, None, True
        
        torch.cuda.set_device(args.gpu_id)
        gpu_name = torch.cuda.get_device_name(args.gpu_id)
        gpu_mem = torch.cuda.get_device_properties(args.gpu_id).total_memory / (1024**3)
        print(f"🚀 Using single GPU {args.gpu_id}: {gpu_name} ({gpu_mem:.1f}GB)")
        return True, args.gpu_id, False
    
    # Default: multi-GPU
    print(f"Using multi-GPU with {torch.cuda.device_count()} GPUs")
    return True, None, True


def main():
    ap = argparse.ArgumentParser("k-NN Coverage Analysis")
    ap.add_argument("--dataset-A", "--big-dataset", required=True, help="Dataset A (typically larger)")
    ap.add_argument("--dataset-B", "--small-dataset", required=True, help="Dataset B (typically smaller)")
    ap.add_argument("--model", help="Model path (required for L2 metric)")
    ap.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sample-ratio", type=float, default=0.01)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-id", type=int, help="GPU ID (if not specified, uses multi-GPU)")
    ap.add_argument("--multi-gpu", action="store_true", help="Use all available GPUs")
    ap.add_argument("--output", default="", help="Output filename base")
    ap.add_argument("--chunk-size", type=int, default=200_000)
    args = ap.parse_args()

    use_gpu, gpu_id, multi_gpu = _setup_gpu(args)
    output_base = str(Path(args.output).parent / Path(args.output).stem) if args.output else None

    # Create output directory if specified and doesn't exist
    if output_base:
        output_dir = Path(output_base).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

    # Load datasets once
    print(f"\n📂 Loading datasets (sample_ratio={args.sample_ratio})...")
    A_tok, _ = load_tokens_from_h5(Path(args.dataset_A), args.sample_ratio)
    B_tok, _ = load_tokens_from_h5(Path(args.dataset_B), args.sample_ratio)
    print(f"Loaded A: {A_tok.shape}, B: {B_tok.shape}")

    # Always run both dedup and nodedup versions
    generated_files = []
    for do_dedup in [True, False]:
        files = run_single_analysis(
            A_tok, B_tok, args, use_gpu, gpu_id, multi_gpu,
            do_dedup=do_dedup, output_base=output_base
        )
        generated_files.extend(files)

    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 Analysis Complete! Generated {len(generated_files)} files:")
    print(f"{'='*60}")
    for f in generated_files:
        print(f"  📄 {f}")

if __name__ == "__main__":
    main()