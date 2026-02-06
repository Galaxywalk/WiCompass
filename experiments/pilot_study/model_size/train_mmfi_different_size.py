#!/usr/bin/env python3
"""
Train MMFi dataset with different model sizes using Ray for distributed training.

Usage:
    python train_mmfi_different_size.py
"""

import sys
import yaml
import subprocess
from pathlib import Path
from copy import deepcopy

import ray

# Project paths
# File is at: experiments/pilot_study/model_size/train_mmfi_different_size.py
# Need 4 levels up to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "src/pose_estimation/configs/mmfi.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src/pose_estimation/train/train_hpe.py"

# Model sizes to train
# SIZE_LIST = ["pico", "nano", "micro", "tiny", "small", "base", "large"]
SIZE_LIST = ["large","xl"]
N_JOBS_PER_SIZE = 1  # Number of jobs per size


def init_ray():
    """Initialize Ray cluster."""
    ray.init(ignore_reinit_error=True, address="auto")
    print("Ray cluster resources:", ray.cluster_resources())


@ray.remote(num_gpus=1, max_retries=2)
def train_single_job(job_id: int, size: str, base_cfg: dict, train_script: str):
    """Train a single model with specified size.
    
    Output is streamed to stdout/stderr so it appears in Ray Dashboard logs.
    """
    import threading
    
    cfg = deepcopy(base_cfg)
    job_name = f"mmfi_{size}_job{job_id:02d}"
    
    # Update config for this job
    cfg["model"]["scale"] = size
    cfg["recorder"]["exp_name"] = job_name
    
    # Write temp config
    tmp_yaml = f"/tmp/{job_name}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(cfg, f)
    
    cmd = [sys.executable, train_script, "--config_path", tmp_yaml]
    
    print(f"🚀 Starting job: {job_name}", flush=True)
    print(f"   Model scale: {size}", flush=True)
    print(f"   Command: {' '.join(cmd)}", flush=True)
    
    # Use Popen to stream output in real-time to Ray Dashboard
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )
    
    def stream_stderr():
        """Stream stderr (tqdm uses stderr for progress bars)."""
        for line in process.stderr:
            # No [ERR] prefix - tqdm output to stderr is normal
            print(f"[{job_name}] {line}", end="", flush=True)
    
    # Start stderr reader thread
    stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
    stderr_thread.start()
    
    # Stream stdout in main thread
    for line in process.stdout:
        print(f"[{job_name}] {line}", end="", flush=True)
    
    # Wait for process to complete
    process.wait()
    stderr_thread.join(timeout=5)
    
    if process.returncode != 0:
        print(f"❌ {job_name} FAILED with exit code {process.returncode}", flush=True)
        raise RuntimeError(f"Job {job_name} failed with exit code {process.returncode}")
    
    print(f"✅ Finished: {job_name}", flush=True)
    return job_name


def main():
    init_ray()
    
    # Load base config
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    
    print(f"\nConfig: {CONFIG_PATH}")
    print(f"Train script: {TRAIN_SCRIPT}")
    print(f"Model sizes: {SIZE_LIST}")
    print(f"Jobs per size: {N_JOBS_PER_SIZE}\n")
    
    # Submit all jobs
    futures = []
    for size in SIZE_LIST:
        for job_id in range(N_JOBS_PER_SIZE):
            job_name = f"train_{size}_job{job_id:02d}"
            fut = train_single_job.options(name=job_name).remote(
                job_id, size, base_cfg, str(TRAIN_SCRIPT)
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
