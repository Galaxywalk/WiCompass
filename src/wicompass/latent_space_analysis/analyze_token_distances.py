#!/usr/bin/env python3
"""
Token Nearest-Neighbor Distance Analysis
---------------------------------------
Supports two distance calculation methods:
1. Token-based Cosine Distance: Convert discrete tokens (N,16) → float32 and L2 normalize, use cosine distance
2. Embedding-based L2 Distance: Load codebook embeddings from model, convert tokens to embeddings, compute L2 distance
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
import h5py, matplotlib.pyplot as plt
from tqdm import tqdm
import faiss   # pip install faiss-cpu | conda install -c conda-forge faiss-gpu

# FP8 support (if available)
try:
    # Try to use torch's native FP8 support (PyTorch 2.1+)
    FP8_AVAILABLE = hasattr(torch, 'float8_e4m3fn')
    if FP8_AVAILABLE:
        FP8_DTYPE = torch.float8_e4m3fn
    else:
        FP8_AVAILABLE = False
        FP8_DTYPE = None
except:
    FP8_AVAILABLE = False
    FP8_DTYPE = None

print(f"🔧 FP8 support: {'✅ Available' if FP8_AVAILABLE else '❌ Not available, using FP16'}")


# --------------------------------------------------------------------------- #
# 1. Data Loading
# --------------------------------------------------------------------------- #
def load_tokens_from_h5(path: Path, ratio: float = 0.01):
    print(f"📁 Load {path}")
    with h5py.File(path, "r") as f:
        total = f.attrs["total_samples"]
        keep  = int(total * ratio)
        print(f"   total {total:,}, sample {keep:,} ({ratio*100:.1f}%)")
        tokens = f["tokens"][:keep]          # (N,16) uint8
        labels = f["labels"][:keep]
    return tokens.astype("uint8"), labels


def load_codebook_embeddings(model_path: Path, device: str = "cpu"):
    """Load codebook embeddings from model file"""
    print(f"🔧 Loading codebook from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Remove 'module.' prefix (if exists)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Find codebook weights
    codebook_key = None
    for key in state_dict.keys():
        if 'codebook' in key.lower() and ('weight' in key or 'embedding' in key):
            codebook_key = key
            break
    
    if codebook_key is None:
        # Try other possible key names
        possible_keys = ['quantizer.codebook.weight', 'quantize.embedding.weight', 
                        'vq.embedding.weight', 'codebook', 'embedding.weight']
        for key in possible_keys:
            if key in state_dict:
                codebook_key = key
                break
    
    if codebook_key is None:
        print("Available keys in state_dict:")
        for key in sorted(state_dict.keys()):
            print(f"  {key}: {state_dict[key].shape}")
        raise KeyError("Could not find codebook embeddings in model")
    
    codebook = state_dict[codebook_key].cpu().numpy()  # (codebook_size, embedding_dim)
    print(f"   Codebook shape: {codebook.shape} (key: {codebook_key})")
    
    return codebook


def tokens_to_embeddings(tokens: np.ndarray, codebook: np.ndarray, use_fp8: bool = True, chunk_size: int = 10000) -> np.ndarray:
    """Convert tokens to embeddings and flatten, support FP8 optimization and chunked processing"""
    # tokens: (N, 16), codebook: (64, embedding_dim)
    # Returns: (N, 16 * embedding_dim)
    batch_size, num_tokens = tokens.shape
    embedding_dim = codebook.shape[1]
    
    print(f"🔄 Converting {batch_size:,} samples to embeddings...")
    print(f"   Token shape: {tokens.shape}")
    print(f"   Codebook shape: {codebook.shape}")
    print(f"   Output shape: ({batch_size}, {num_tokens * embedding_dim})")
    
    # Calculate memory requirements
    if use_fp8 and FP8_AVAILABLE:
        bytes_per_element = 1  # FP8
        dtype_str = "FP8"
    else:
        bytes_per_element = 2  # FP16 as alternative
        dtype_str = "FP16"
    
    total_elements = batch_size * num_tokens * embedding_dim
    memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)
    print(f"   Memory requirement: {memory_mb:.1f} MB ({dtype_str})")
    
    # If memory requirement is too large, use chunked processing
    if memory_mb > 2048:  # If exceeds 2GB, use chunked processing
        print(f"   Large memory requirement detected, using chunked processing...")
        return _tokens_to_embeddings_chunked(tokens, codebook, use_fp8, chunk_size)
    
    # Process all at once
    # Use advanced indexing to get embeddings
    embeddings = codebook[tokens]  # (N, 16, embedding_dim)
    
    # Flatten to 1D vector
    embeddings_flat = embeddings.reshape(batch_size, -1)  # (N, 16 * embedding_dim)
    
    # Convert to required precision
    if use_fp8 and FP8_AVAILABLE:
        # Use PyTorch's FP8 (need to convert to torch tensor then back to numpy)
        embeddings_torch = torch.from_numpy(embeddings_flat)
        embeddings_fp8 = embeddings_torch.to(FP8_DTYPE)
        return embeddings_fp8.to(torch.float16).numpy()  # Temporarily convert to FP16 for computation
    else:
        # Use FP16 as alternative
        return embeddings_flat.astype(np.float16)


def _tokens_to_embeddings_chunked(tokens: np.ndarray, codebook: np.ndarray, use_fp8: bool, chunk_size: int) -> np.ndarray:
    """Chunked processing of token to embedding conversion"""
    batch_size, num_tokens = tokens.shape
    embedding_dim = codebook.shape[1]
    output_dim = num_tokens * embedding_dim
    
    # Pre-allocate output array
    if use_fp8 and FP8_AVAILABLE:
        output = np.empty((batch_size, output_dim), dtype=np.float16)  # Use FP16 as computation precision
    else:
        output = np.empty((batch_size, output_dim), dtype=np.float16)
    
    # Chunked processing
    for i in tqdm(range(0, batch_size, chunk_size), desc="Converting tokens to embeddings"):
        end_idx = min(i + chunk_size, batch_size)
        chunk_tokens = tokens[i:end_idx]
        
        # Convert current chunk
        chunk_embeddings = codebook[chunk_tokens]  # (chunk_size, 16, embedding_dim)
        chunk_flat = chunk_embeddings.reshape(end_idx - i, -1)  # (chunk_size, 16 * embedding_dim)
        
        # Store to output array
        output[i:end_idx] = chunk_flat.astype(np.float16)
    
    return output


# --------------------------------------------------------------------------- #
# 2. Distance Calculation Functions
# --------------------------------------------------------------------------- #
def nn_cos_big2small(
    big: np.ndarray,
    small: np.ndarray,
    batch: int = 200_000,
    gpu: bool = True,
) -> np.ndarray:
    """Return (N_big,) cosine distance = 1 - max similarity"""
    assert big.shape[1] == small.shape[1]
    big_f   = big.astype("float32")
    small_f = small.astype("float32")

    # After normalization, inner product = cosine similarity
    faiss.normalize_L2(big_f)
    faiss.normalize_L2(small_f)

    dim = big_f.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)          # Exact cosine
    index = faiss.index_cpu_to_all_gpus(cpu_index) if gpu and faiss.get_num_gpus() else cpu_index

    index.add(small_f)

    out = np.empty(len(big_f), dtype="float32")
    for i in tqdm(range(0, len(big_f), batch), desc="Faiss cosine search"):
        D, _ = index.search(big_f[i:i + batch], k=1)   # Max similarity
        out[i:i + batch] = 1.0 - D.ravel()             # Similarity → distance
    return out


def nn_l2_big2small(
    big: np.ndarray,
    small: np.ndarray,
    batch: int = 200_000,
    gpu: bool = True,
) -> np.ndarray:
    """Return (N_big,) L2 distance to nearest neighbor, support FP16 optimization"""
    assert big.shape[1] == small.shape[1]
    
    # If input is FP16, convert to FP32 for computation (Faiss requires)
    if big.dtype == np.float16:
        print("   Converting FP16 to FP32 for Faiss computation...")
        big_f = big.astype("float32")
        small_f = small.astype("float32")
    else:
        big_f = big.astype("float32")
        small_f = small.astype("float32")

    dim = big_f.shape[1]
    cpu_index = faiss.IndexFlatL2(dim)          # Exact L2 distance
    index = faiss.index_cpu_to_all_gpus(cpu_index) if gpu and faiss.get_num_gpus() else cpu_index

    index.add(small_f)

    out = np.empty(len(big_f), dtype="float32")
    for i in tqdm(range(0, len(big_f), batch), desc="Faiss L2 search"):
        D, _ = index.search(big_f[i:i + batch], k=1)   # Min L2 distance
        out[i:i + batch] = D.ravel()                   # Already distance
    return out


# --------------------------------------------------------------------------- #
# 3. Visualization & Statistics
# --------------------------------------------------------------------------- #
def plot_dist(dist, out_dir: Path, names: dict, metric: str = "cosine"):
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    metric_name = "Cosine" if metric == "cosine" else "L2"
    distance_label = "1-NN Cosine Distance (1-cos)" if metric == "cosine" else "1-NN L2 Distance"
    
    fig.suptitle(f"{metric_name} 1-NN Distance  •  {names['big']} → {names['small']}", fontsize=16)

    # Histogram
    ax[0].hist(dist, bins="auto", alpha=.7, edgecolor="k")
    ax[0].set_xlabel(distance_label)
    ax[0].set_ylabel("Count")
    ax[0].set_title("Histogram")
    ax[0].grid(alpha=.3)
    ax[0].text(.98, .98,
        "\n".join([
            f"Count : {len(dist):,}",
            f"Mean  : {dist.mean():.3f}",
            f"Std   : {dist.std():.3f}",
            f"Min   : {dist.min():.3f}",
            f"Max   : {dist.max():.3f}"]),
        ha="right", va="top", transform=ax[0].transAxes,
        bbox=dict(boxstyle="round", fc="w", alpha=.8))

    # CDF
    s = np.sort(dist)
    ax[1].plot(s, np.arange(1, len(s)+1)/len(s), lw=2)
    ax[1].set_xlabel(distance_label)
    ax[1].set_ylabel("Cumulative Probability")
    ax[1].set_title("CDF")
    ax[1].grid(alpha=.3)
    for p in [0.5, 0.9, 0.95, 0.99]:
        v = np.percentile(dist, p*100)
        ax[1].axvline(v, ls="--", c="red", alpha=.6)
        ax[1].text(v, p, f"{p*100:.0f}%: {v:.3f}", rotation=90, va="bottom", ha="right")

    # Box plot
    bp = ax[2].boxplot(dist, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax[2].set_title("Box Plot")
    ax[2].set_ylabel(distance_label)
    ax[2].grid(alpha=.3)

    plt.tight_layout()
    
    prefix = "cos" if metric == "cosine" else "l2"
    png = out_dir / f"{prefix}_distance_{names['big']}_to_{names['small']}.png"
    plt.savefig(png, dpi=300); plt.close()
    print(f"📊 Plot saved → {png}")

    stats = {
        "datasets": names,
        "metric": metric,
        "count": int(len(dist)),
        "mean":  float(dist.mean()),
        "std":   float(dist.std()),
        "min":   float(dist.min()),
        "max":   float(dist.max()),
        "median": float(np.median(dist)),
        "percentiles": {p: float(np.percentile(dist, p)) for p in [25,50,75,90,95,99]},
    }
    json_path = out_dir / f"{prefix}_stats_{names['big']}_to_{names['small']}.json"
    with open(json_path,"w") as f: json.dump(stats,f,indent=2)
    print(f"📄 Stats saved → {json_path}")
    return stats


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("Token Distance Analysis (Faiss)")
    ap.add_argument("--big-dataset",  required=True, help="Large dataset HDF5 file path")
    ap.add_argument("--small-dataset",required=True, help="Small dataset HDF5 file path")
    ap.add_argument("--output-dir", default="distance_analysis", help="Output directory")
    ap.add_argument("--sample-ratio", type=float, default=0.01, help="Sampling ratio")
    ap.add_argument("--chunk-size",   type=int,   default=200_000, help="Batch size")
    ap.add_argument("--cpu-only",     action="store_true", help="Use CPU only")
    ap.add_argument("--metric", choices=["cosine", "l2", "both"], default="cosine", 
                   help="Distance metric: cosine (token cosine), l2 (embedding L2), both (compute both)")
    ap.add_argument("--model", type=str, help="Model file path (required when using l2/both)")
    ap.add_argument("--use-fp16", action="store_true", help="Force use FP16 instead of FP8")
    ap.add_argument("--embedding-chunk-size", type=int, default=10000, help="Chunk size when converting embeddings")
    args = ap.parse_args()

    big_p, small_p = Path(args.big_dataset), Path(args.small_dataset)
    out_dir = Path(args.output_dir)
    if not big_p.exists() or not small_p.exists():
        raise FileNotFoundError("dataset file missing")

    # Check if model file is needed
    if args.metric in ["l2", "both"] and not args.model:
        raise ValueError("--model is required when using l2 or both metrics")
    
    if args.model and not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"🚀 Distance Analysis   big={big_p.name}   small={small_p.name}")
    print(f"📏 Metric: {args.metric}")
    
    # Load token data
    big_tok,_   = load_tokens_from_h5(big_p,   args.sample_ratio)
    small_tok,_ = load_tokens_from_h5(small_p, args.sample_ratio)

    names = {"big": big_p.stem.replace("_tokens", ""), 
             "small": small_p.stem.replace("_tokens", "")}

    # Compute cosine distance (token-based)
    if args.metric in ["cosine", "both"]:
        print("\n🔄 Computing cosine distances (token-based)...")
        cos_dist = nn_cos_big2small(big_tok, small_tok,
                                    batch=args.chunk_size,
                                    gpu=not args.cpu_only)
        plot_dist(cos_dist, out_dir, names, metric="cosine")

    # Compute L2 distance (embedding-based)
    if args.metric in ["l2", "both"]:
        print("\n🔄 Computing L2 distances (embedding-based)...")
        
        # Load codebook
        codebook = load_codebook_embeddings(Path(args.model))
        
        # Convert to embeddings
        print("🔄 Converting tokens to embeddings...")
        use_fp8 = not args.use_fp16  # If not forced to use FP16, try to use FP8
        
        big_emb = tokens_to_embeddings(big_tok, codebook, use_fp8=use_fp8, 
                                       chunk_size=args.embedding_chunk_size)
        small_emb = tokens_to_embeddings(small_tok, codebook, use_fp8=use_fp8,
                                         chunk_size=args.embedding_chunk_size)
        
        print(f"   Big embeddings shape: {big_emb.shape}, dtype: {big_emb.dtype}")
        print(f"   Small embeddings shape: {small_emb.shape}, dtype: {small_emb.dtype}")
        
        # Calculate memory usage
        big_memory_mb = big_emb.nbytes / (1024 * 1024)
        small_memory_mb = small_emb.nbytes / (1024 * 1024)
        total_memory_mb = big_memory_mb + small_memory_mb
        print(f"   Memory usage: {total_memory_mb:.1f} MB (big: {big_memory_mb:.1f}, small: {small_memory_mb:.1f})")
        
        # Compute L2 distance
        l2_dist = nn_l2_big2small(big_emb, small_emb,
                                  batch=args.chunk_size,
                                  gpu=not args.cpu_only)
        plot_dist(l2_dist, out_dir, names, metric="l2")

    print(f"\n✅ Analysis completed! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()