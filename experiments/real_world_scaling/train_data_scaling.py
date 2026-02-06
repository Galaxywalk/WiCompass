#!/usr/bin/env python3
"""
Train real-world datasets with different data scaling (1000, 2000, 4000, 8000 samples).

This script trains baseline, dance, and wicompass configurations with varying amounts
of training data. For configurations with multiple train folders, samples are evenly
distributed across folders.

Example:
    - baseline config has 2 train folders: A_baseline, C_baseline
    - With 1000 total samples: each folder gets 500 samples
    - With 2000 total samples: each folder gets 1000 samples
    - With 4000 total samples: each folder gets 2000 samples
    - With 8000 total samples: each folder gets 4000 samples

Usage:
    python train_data_scaling.py --list                    # List available configs
    python train_data_scaling.py --configs baseline        # Train specific config
    python train_data_scaling.py                           # Train all configs
    python train_data_scaling.py --samples 1000 2000       # Train specific sample sizes
    python train_data_scaling.py --configs baseline dance --samples 1000  # Train specific configs with specific samples
"""

import sys
import argparse
import yaml
import subprocess
from pathlib import Path
from copy import deepcopy

import ray

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TRAIN_SCRIPT = PROJECT_ROOT / "src/pose_estimation/train/train_hpe.py"

# Available sample sizes
AVAILABLE_SAMPLES = [1000, 2000, 4000, 8000]

# Configurations to train
CONFIG_NAMES = ["baseline", "dance", "wicompass"]

N_JOBS_PER_CONFIG = 1  # Number of repeated runs per config


def get_available_configs() -> list[str]:
    """Get list of available config files (without .yaml extension)."""
    if not CONFIG_DIR.exists():
        return []
    return sorted([f.stem for f in CONFIG_DIR.glob("*.yaml")])


def init_ray():
    """Initialize Ray cluster."""
    ray.init(ignore_reinit_error=True, address="auto")
    print("Ray cluster resources:", ray.cluster_resources())


@ray.remote(num_gpus=1, max_retries=2)
def train_single_job(job_id: int, config_name: str, sample_size: int, base_cfg: dict, train_script: str):
    """Train a single model with specified config and sample size.
    
    Output is streamed to stdout/stderr for Ray Dashboard logs.
    """
    import threading
    
    cfg = deepcopy(base_cfg)
    
    # Calculate samples per folder
    train_folders = cfg.get("train_folders", [])
    n_folders = len(train_folders)
    
    if n_folders == 0:
        raise ValueError(f"No train_folders specified in config {config_name}")
    
    # Distribute samples evenly across folders
    samples_per_folder = sample_size // n_folders
    
    # Update config with per-folder sample size limits
    # This ensures each folder gets exactly samples_per_folder samples
    cfg["max_samples_per_folder"] = samples_per_folder
    # Also set total for reference (but per-folder limit takes precedence)
    cfg["max_train_samples"] = sample_size
    
    # Create experiment name
    job_name = f"real_world_{config_name}_{sample_size}_job{job_id:02d}"
    cfg["recorder"]["exp_name"] = job_name
    
    # Write temp config
    tmp_yaml = f"/tmp/{job_name}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(cfg, f)
    
    cmd = [sys.executable, train_script, "--config_path", tmp_yaml]
    
    print(f"🚀 Starting job: {job_name}", flush=True)
    print(f"   Config: {config_name}", flush=True)
    print(f"   Total samples: {sample_size}", flush=True)
    print(f"   Samples per folder: {samples_per_folder} (across {n_folders} folders)", flush=True)
    print(f"   Train folders: {train_folders}", flush=True)
    print(f"   Val folders: {cfg.get('val_folders', [])}", flush=True)
    print(f"   Command: {' '.join(cmd)}", flush=True)
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    
    def stream_stderr():
        """Stream stderr (tqdm uses stderr for progress bars)."""
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
        description="Train real-world datasets with different data scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available configs
    python train_data_scaling.py --list

    # Train all configs with all sample sizes
    python train_data_scaling.py

    # Train specific configs
    python train_data_scaling.py --configs baseline wicompass

    # Train specific sample sizes
    python train_data_scaling.py --samples 1000 2000

    # Train specific config with specific samples
    python train_data_scaling.py --configs baseline --samples 1000 2000
        """
    )
    parser.add_argument(
        "--configs", 
        nargs="+", 
        default=None,
        help="Config names to train (without .yaml). If not specified, trains all: baseline, dance, wicompass"
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=None,
        help=f"Sample sizes to train (default: {AVAILABLE_SAMPLES})"
    )
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=N_JOBS_PER_CONFIG,
        help="Number of repeated runs per config (default: 1)"
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Override model scale (e.g., 'base', 'large')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configs and exit"
    )
    args = parser.parse_args()
    
    available_configs = get_available_configs()
    
    if args.list:
        print("Available configs:")
        for cfg in available_configs:
            print(f"  - {cfg}")
        print(f"\nDefault configs to train: {CONFIG_NAMES}")
        print(f"Default sample sizes: {AVAILABLE_SAMPLES}")
        if not available_configs:
            print("  (no configs found)")
        sys.exit(0)
    
    # Determine which configs to train
    if args.configs:
        selected_configs = []
        for cfg in args.configs:
            if cfg in available_configs:
                selected_configs.append(cfg)
            else:
                print(f"⚠️ Config not found: {cfg}")
        if not selected_configs:
            print("Error: No valid configs selected")
            sys.exit(1)
    else:
        # Use default configs
        selected_configs = []
        for cfg_name in CONFIG_NAMES:
            if cfg_name in available_configs:
                selected_configs.append(cfg_name)
            else:
                print(f"⚠️ Default config not found: {cfg_name}")
        if not selected_configs:
            print("Error: No config files found in", CONFIG_DIR)
            sys.exit(1)
    
    # Determine which sample sizes to train
    if args.samples:
        selected_samples = []
        for s in args.samples:
            if s > 0:
                selected_samples.append(s)
            else:
                print(f"⚠️ Invalid sample size: {s}")
        if not selected_samples:
            print("Error: No valid sample sizes specified")
            sys.exit(1)
    else:
        selected_samples = AVAILABLE_SAMPLES
    
    init_ray()
    
    print(f"\nConfig directory: {CONFIG_DIR}")
    print(f"Train script: {TRAIN_SCRIPT}")
    print(f"Selected configs: {selected_configs}")
    print(f"Selected sample sizes: {selected_samples}")
    print(f"Jobs per config: {args.n_jobs}\n")
    
    # Submit all jobs
    futures = []
    for config_name in selected_configs:
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        
        with open(config_path) as f:
            base_cfg = yaml.safe_load(f)
        
        # Verify train_folders exist
        train_folders = base_cfg.get("train_folders", [])
        if not train_folders:
            print(f"⚠️ Skipping {config_name}: no train_folders specified")
            continue
        
        n_folders = len(train_folders)
        print(f"📁 {config_name}: {n_folders} train folders: {train_folders}")
        
        # Apply overrides
        if args.scale:
            base_cfg["model"]["scale"] = args.scale
        if args.epochs:
            base_cfg["loader"]["epochs"] = args.epochs
        
        for sample_size in selected_samples:
            # Check if sample size is divisible by number of folders
            if sample_size % n_folders != 0:
                print(f"⚠️ Skipping {config_name} with {sample_size} samples: "
                      f"not divisible by {n_folders} folders")
                continue
            
            for job_id in range(args.n_jobs):
                job_name = f"train_{config_name}_{sample_size}_job{job_id:02d}"
                fut = train_single_job.options(name=job_name).remote(
                    job_id, config_name, sample_size, base_cfg, str(TRAIN_SCRIPT)
                )
                futures.append((config_name, sample_size, job_id, fut))
    
    print(f"\nSubmitted {len(futures)} jobs\n")
    
    # Wait for completion
    results = []
    for config_name, sample_size, job_id, fut in futures:
        try:
            result = ray.get(fut)
            results.append((config_name, sample_size, job_id, result, "success"))
        except Exception as e:
            results.append((config_name, sample_size, job_id, str(e), "failed"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    for config_name, sample_size, job_id, result, status in results:
        if status == "success":
            print(f"   ✅ {config_name} {sample_size} job{job_id:02d}: {result}")
            success_count += 1
        else:
            print(f"   ❌ {config_name} {sample_size} job{job_id:02d}: FAILED - {result}")
            fail_count += 1
    
    print(f"\nTotal: {success_count} succeeded, {fail_count} failed")
    
    if fail_count == 0:
        print("\n🎉 All jobs completed successfully!")
    else:
        print(f"\n⚠️ {fail_count} jobs failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
