#!/usr/bin/env python3
"""
Train MMFi dataset with Leave-One-Out action cross-validation using Ray.

Experiment design:
1. Baseline: Train on ALL actions (A01-A27) from scenes E01-E03, test on E04
2. Leave-One-Out: For each action Axx:
   - Train on 26 actions (all except Axx) from scenes E01-E03
   - Test on the held-out action Axx from scenes E01-E03
   
This evaluates generalization to unseen action types.

Usage:
    python train_mmfi_leave_one_out.py
"""

import sys
import yaml
import subprocess
from pathlib import Path
from copy import deepcopy

import ray

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "experiments/pilot_study/leave_one_out/mmfi_leave_one_out_config.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src/pose_estimation/train/train_hpe.py"

# All actions in MMFi dataset
ALL_ACTIONS = [f"A{i:02d}" for i in range(6, 28)]  # A01 to A27

# Experiment settings
N_REPEATS = 1  # Number of repeats per configuration
INCLUDE_BASELINE = False  # Whether to train a baseline model on all actions


def init_ray():
    """Initialize Ray cluster."""
    ray.init(ignore_reinit_error=True, address="auto")
    print("Ray cluster resources:", ray.cluster_resources())


@ray.remote(num_gpus=1, max_retries=2)
def train_single_job(job_name: str, base_cfg: dict, train_script: str,
                     train_actions: list, test_actions: list):
    """Train a single model with specified action configuration."""
    import threading
    
    cfg = deepcopy(base_cfg)
    
    # Update config for this job
    cfg["recorder"]["exp_name"] = job_name
    cfg["train_dataset"]["actions"] = train_actions
    cfg["val_dataset"]["actions"] = test_actions
    
    # Write temp config
    tmp_yaml = f"/tmp/{job_name}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(cfg, f)
    
    cmd = [sys.executable, train_script, "--config_path", tmp_yaml]
    
    print(f"🚀 Starting job: {job_name}", flush=True)
    print(f"   Train actions ({len(train_actions)}): {train_actions[:5]}{'...' if len(train_actions) > 5 else ''}", flush=True)
    print(f"   Test actions ({len(test_actions)}): {test_actions}", flush=True)
    print(f"   Command: {' '.join(cmd)}", flush=True)
    
    # Use Popen to stream output in real-time to Ray Dashboard
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
    init_ray()
    
    # Load base config
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    
    print(f"\nConfig: {CONFIG_PATH}")
    print(f"Train script: {TRAIN_SCRIPT}")
    print(f"All actions: {ALL_ACTIONS}")
    print(f"Repeats per config: {N_REPEATS}")
    print(f"Include baseline: {INCLUDE_BASELINE}\n")
    
    # Submit all jobs
    futures = []
    
    # 1. Baseline jobs: Train on ALL actions, test on E04
    if INCLUDE_BASELINE:
        for repeat_id in range(N_REPEATS):
            job_name = f"baseline_all_actions_repeat{repeat_id:02d}"
            
            baseline_cfg = deepcopy(base_cfg)
            baseline_cfg["val_dataset"]["scenes"] = ["E04"]
            
            fut = train_single_job.options(name=job_name).remote(
                job_name, baseline_cfg, str(TRAIN_SCRIPT),
                train_actions=ALL_ACTIONS,
                test_actions=ALL_ACTIONS
            )
            futures.append(fut)
    
    # 2. Leave-One-Out jobs: For each action, train on 26, test on 1
    for held_out_action in ALL_ACTIONS:
        train_actions = [a for a in ALL_ACTIONS if a != held_out_action]
        test_actions = [held_out_action]
        
        for repeat_id in range(N_REPEATS):
            job_name = f"leave_out_{held_out_action}_repeat{repeat_id:02d}"
            
            fut = train_single_job.options(name=job_name).remote(
                job_name, base_cfg, str(TRAIN_SCRIPT),
                train_actions=train_actions,
                test_actions=test_actions
            )
            futures.append(fut)
    
    print(f"Submitted {len(futures)} jobs\n")
    
    # Wait for completion
    completed_jobs = ray.get(futures)
    
    print("\n🎉 All jobs completed!")
    print("Completed jobs:")
    for job in completed_jobs:
        print(f"   ✅ {job}")
    
    print("\n💡 Results saved in checkpoint directories with best_metrics.json")
    print("   Run evaluate_mmfi_leave_one_out.py to aggregate results.")


if __name__ == "__main__":
    main()
