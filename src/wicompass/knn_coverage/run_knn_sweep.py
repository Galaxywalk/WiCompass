#!/usr/bin/env python3
"""
Run KNN coverage analysis with different k values for MMBody, MMFi, and WiCompass datasets.

This script runs the knn_coverage.py tool sequentially with various k values.
Each run uses all GPUs and takes approximately 5-10 minutes.

Usage:
    python src/wicompass/knn_coverage/run_knn_sweep.py                # Run all experiments
    python src/wicompass/knn_coverage/run_knn_sweep.py --dry-run      # Print commands without running
    python src/wicompass/knn_coverage/run_knn_sweep.py --mmbody-only  # Only run MMBody experiments
    python src/wicompass/knn_coverage/run_knn_sweep.py --mmfi-only    # Only run MMFi experiments
    python src/wicompass/knn_coverage/run_knn_sweep.py --wicompass-only  # Only run WiCompass experiments
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Configuration
DATASET_A = "logs/wicompass/encoded_tokens/BMLmovi-BMLrub-CMU-GRAB-KIT-MOYO-MoSh-PosePrior-WEIZMANN_tokens.h5"
MMBODY_DATASET = "logs/wicompass/encoded_tokens/MMBody_tokens.h5"
MMFI_DATASET = "logs/wicompass/encoded_tokens/MMFi_tokens.h5"
WICOMPASS_DATASET = "logs/wicompass/encoded_tokens/k4_q95_tokens.h5"

# K values to sweep
MMBODY_K_VALUES = [2, 4, 6, 8, 10, 12]
MMFI_K_VALUES = [2, 4, 6, 8, 10, 12, 16, 32, 64, 128]
WICOMPASS_K_VALUES = [2, 4, 6, 8, 10, 12]


def run_knn_coverage(dataset_b: str, dataset_name: str, k: int, dry_run: bool = False) -> bool:
    """
    Run knn_coverage.py with specified parameters.
    
    Args:
        dataset_b: Path to dataset B (target dataset)
        dataset_name: Name for output directory (mmbody or mmfi)
        k: Number of nearest neighbors
        dry_run: If True, only print the command without running
    
    Returns:
        True if successful, False otherwise
    """
    output_dir = f"logs/wicompass/knn_coverage/{dataset_name}/k{k}"
    
    cmd = [
        sys.executable,
        "src/wicompass/knn_coverage/knn_coverage.py",
        "--dataset-A", DATASET_A,
        "--dataset-B", dataset_b,
        "--metric", "cosine",
        "--k", str(k),
        "--multi-gpu",
        "--sample-ratio", "1.0",
        "--output", output_dir,
    ]
    
    print(f"\n{'=' * 60}")
    print(f"Running: {dataset_name} with k={k}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Completed: {dataset_name} k={k}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset_name} k={k} (exit code: {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted: {dataset_name} k={k}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run KNN coverage analysis with different k values"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them"
    )
    parser.add_argument(
        "--mmbody-only", action="store_true",
        help="Only run MMBody experiments"
    )
    parser.add_argument(
        "--mmfi-only", action="store_true",
        help="Only run MMFi experiments"
    )
    parser.add_argument(
        "--wicompass-only", action="store_true",
        help="Only run WiCompass experiments"
    )
    args = parser.parse_args()
    
    # Track results
    results = {"success": [], "failed": []}
    
    print("=" * 60)
    print("KNN Coverage Sweep")
    print("=" * 60)
    
    # Determine which experiments to run
    run_mmbody = not args.mmfi_only and not args.wicompass_only
    run_mmfi = not args.mmbody_only and not args.wicompass_only
    run_wicompass = not args.mmbody_only and not args.mmfi_only
    
    # If a specific only flag is set, only run that one
    if args.mmbody_only:
        run_mmbody, run_mmfi, run_wicompass = True, False, False
    elif args.mmfi_only:
        run_mmbody, run_mmfi, run_wicompass = False, True, False
    elif args.wicompass_only:
        run_mmbody, run_mmfi, run_wicompass = False, False, True
    
    if run_mmbody:
        print(f"\nMMBody k values: {MMBODY_K_VALUES}")
    if run_mmfi:
        print(f"MMFi k values: {MMFI_K_VALUES}")
    if run_wicompass:
        print(f"WiCompass k values: {WICOMPASS_K_VALUES}")
    
    total_runs = 0
    if run_mmbody:
        total_runs += len(MMBODY_K_VALUES)
    if run_mmfi:
        total_runs += len(MMFI_K_VALUES)
    if run_wicompass:
        total_runs += len(WICOMPASS_K_VALUES)
    
    print(f"\nTotal runs: {total_runs}")
    print(f"Estimated time: {total_runs * 5}-{total_runs * 10} minutes")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - Commands will be printed but not executed]")
    
    current_run = 0
    
    try:
        # Run MMBody experiments
        if run_mmbody:
            print("\n" + "=" * 60)
            print("Starting MMBody experiments")
            print("=" * 60)
            
            for k in MMBODY_K_VALUES:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] MMBody k={k}")
                
                success = run_knn_coverage(
                    MMBODY_DATASET, "mmbody", k, dry_run=args.dry_run
                )
                
                key = f"mmbody_k{k}"
                if success:
                    results["success"].append(key)
                else:
                    results["failed"].append(key)
        
        # Run MMFi experiments
        if run_mmfi:
            print("\n" + "=" * 60)
            print("Starting MMFi experiments")
            print("=" * 60)
            
            for k in MMFI_K_VALUES:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] MMFi k={k}")
                
                success = run_knn_coverage(
                    MMFI_DATASET, "mmfi", k, dry_run=args.dry_run
                )
                
                key = f"mmfi_k{k}"
                if success:
                    results["success"].append(key)
                else:
                    results["failed"].append(key)
        
        # Run WiCompass experiments
        if run_wicompass:
            print("\n" + "=" * 60)
            print("Starting WiCompass experiments")
            print("=" * 60)
            
            for k in WICOMPASS_K_VALUES:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] WiCompass k={k}")
                
                success = run_knn_coverage(
                    WICOMPASS_DATASET, "wicompass", k, dry_run=args.dry_run
                )
                
                key = f"wicompass_k{k}"
                if success:
                    results["success"].append(key)
                else:
                    results["failed"].append(key)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Sweep interrupted by user")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successful: {len(results['success'])}")
    for key in results["success"]:
        print(f"  ✓ {key}")
    
    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for key in results["failed"]:
            print(f"  ✗ {key}")
    
    print("\nDone!")
    
    # Return non-zero exit code if any failed
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

