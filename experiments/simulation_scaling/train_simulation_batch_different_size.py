#!/usr/bin/env python3
"""
Batch training script for SimulationMMBody with different training data sizes.

For each training set, trains models with 100%, 80%, 60%, 40%, 20%, 10% of the data.
All models are evaluated against the same benchmark validation set.

Usage:
    # Train all available training sets with all data sizes
    python train_simulation_batch_different_size.py
    
    # Train specific training sets
    python train_simulation_batch_different_size.py --train_sets k10_q85 k12_q95
    
    # Train with specific data sizes
    python train_simulation_batch_different_size.py --sizes 1.0 0.5 0.1
    
    # Train with custom model scale
    python train_simulation_batch_different_size.py --scale tiny
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy

import ray

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "src/pose_estimation/configs/simulation_mmbody.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src/pose_estimation/train/train_hpe.py"
DATA_ROOT = PROJECT_ROOT / "datasets/simulation_datasets/rfgen_data"
BENCHMARK_PATH = DATA_ROOT / "benchmark0105"

# Default data sizes to train with
DEFAULT_SIZES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
# DEFAULT_SIZES = [0.05]



def get_available_train_sets():
    """Get list of available training sets (all folders except benchmark)."""
    train_sets = []
    for d in DATA_ROOT.iterdir():
        if d.is_dir() and d.name != "benchmark":
            train_sets.append(d.name)
    return sorted(train_sets)


def init_ray():
    """Initialize Ray cluster."""
    ray.init(ignore_reinit_error=True, address="auto")
    print("Ray cluster resources:", ray.cluster_resources())


@ray.remote(num_gpus=1, max_retries=2)
def train_single_job(train_set: str, data_quantity: float, base_cfg: dict, 
                     train_script: str, data_root: str, benchmark_path: str, 
                     model_scale: str = None):
    """Train a single model with specified training set and data quantity."""
    import threading
    
    cfg = deepcopy(base_cfg)
    
    # Configure paths
    cfg["dataset_path"] = os.path.join(data_root, train_set)
    cfg["validation_dataset_path"] = benchmark_path
    
    # Set data quantity
    cfg["used_data_quantity"] = data_quantity
    
    # Set experiment name with data percentage
    scale_str = model_scale or cfg.get("model", {}).get("scale", "base")
    pct_str = f"{int(data_quantity * 100)}pct"
    job_name = f"sim_{train_set}_{pct_str}_{scale_str}"
    cfg["recorder"]["exp_name"] = job_name
    
    # Update model scale if specified
    if model_scale:
        cfg["model"]["scale"] = model_scale
    
    # Write temp config
    tmp_yaml = f"/tmp/{job_name}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(cfg, f)
    
    cmd = [sys.executable, train_script, "--config_path", tmp_yaml]
    
    print(f"🚀 Starting job: {job_name}", flush=True)
    print(f"   Train set: {train_set}", flush=True)
    print(f"   Data quantity: {data_quantity * 100:.0f}%", flush=True)
    print(f"   Validation: benchmark", flush=True)
    print(f"   Model scale: {scale_str}", flush=True)
    print(f"   Command: {' '.join(cmd)}", flush=True)
    
    # Run with streaming output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    
    def stream_stderr():
        for line in process.stderr:
            print(f"[{job_name}] {line}", end="", flush=True)
    
    stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
    stderr_thread.start()
    
    for line in process.stdout:
        print(f"[{job_name}] {line}", end="", flush=True)
    
    process.wait()
    stderr_thread.join(timeout=5)
    
    if process.returncode != 0:
        print(f"❌ {job_name} FAILED with exit code {process.returncode}", flush=True)
        raise RuntimeError(f"Job {job_name} failed with exit code {process.returncode}")
    
    print(f"✅ Finished: {job_name}", flush=True)
    return job_name


def main():
    parser = argparse.ArgumentParser(
        description='Batch training for SimulationMMBody with different data sizes'
    )
    parser.add_argument('--train_sets', nargs='+', default=None,
                        help='Training sets to use (default: all available)')
    parser.add_argument('--sizes', nargs='+', type=float, default=None,
                        help=f'Data sizes to train with (default: {DEFAULT_SIZES})')
    parser.add_argument('--scale', type=str, default=None,
                        help='Model scale (e.g., tiny, small, base, large)')
    parser.add_argument('--list', action='store_true',
                        help='List available training sets and exit')
    args = parser.parse_args()
    
    # Get available training sets
    available_sets = get_available_train_sets()
    
    if args.list:
        print("Available training sets:")
        for s in available_sets:
            print(f"  - {s}")
        print(f"\nDefault data sizes: {DEFAULT_SIZES}")
        return
    
    # Determine which training sets to use
    if args.train_sets:
        train_sets = args.train_sets
        # Validate
        for s in train_sets:
            if s not in available_sets:
                print(f"Error: Training set '{s}' not found.")
                print(f"Available: {available_sets}")
                return
    else:
        train_sets = available_sets
    
    # Determine data sizes
    sizes = args.sizes if args.sizes else DEFAULT_SIZES
    sizes = sorted(sizes, reverse=True)  # Train larger sizes first
    
    print(f"Training sets: {train_sets}")
    print(f"Data sizes: {[f'{s*100:.0f}%' for s in sizes]}")
    print(f"Validation set: benchmark")
    print(f"Model scale: {args.scale or 'base (default)'}")
    print(f"Total jobs: {len(train_sets) * len(sizes)}")
    
    # Initialize Ray
    init_ray()
    
    # Load base config
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    
    print(f"\nBase config: {CONFIG_PATH}")
    print(f"Train script: {TRAIN_SCRIPT}")
    print(f"Data root: {DATA_ROOT}\n")
    
    # Submit jobs for all combinations
    futures = []
    for train_set in train_sets:
        for size in sizes:
            pct_str = f"{int(size * 100)}pct"
            job_name = f"train_{train_set}_{pct_str}"
            fut = train_single_job.options(name=job_name).remote(
                train_set, size, base_cfg, str(TRAIN_SCRIPT),
                str(DATA_ROOT), str(BENCHMARK_PATH), args.scale
            )
            futures.append(fut)
    
    print(f"Submitted {len(futures)} jobs\n")
    
    # Wait for completion
    completed_jobs = ray.get(futures)
    
    print("\n🎉 All jobs completed!")
    print("Completed jobs:")
    for job in completed_jobs:
        print(f"   ✅ {job}")


if __name__ == "__main__":
    main()

