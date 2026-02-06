#!/usr/bin/env python3
"""
Utility script to inspect the contents of .npy and .npz files.

Usage:
    python check_npz.py <file_path>
    python check_npz.py  # Uses default test file
"""

import argparse
import numpy as np
from pathlib import Path


def check_file(file_path: str) -> None:
    """
    Load and display the contents/structure of a .npy or .npz file.
    
    Args:
        file_path: Path to the .npy or .npz file to inspect.
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File '{file_path}' not found.")
        return
    
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {file_path}")
    print(f"{'=' * 60}")
    
    data = np.load(file_path, allow_pickle=True)
    
    if file_path.endswith('.npz'):
        # NPZ file: contains multiple arrays
        print(f"File type: .npz (compressed archive)")
        print(f"Keys: {list(data.files)}")
        print(f"\nArray details:")
        for key in data.files:
            value = data[key]
            print(f"  - {key}:")
            print(f"      shape: {value.shape}")
            print(f"      dtype: {value.dtype}")
            if value.size > 0:
                print(f"      min: {np.min(value):.6f}, max: {np.max(value):.6f}")
    else:
        # NPY file: single array
        print(f"File type: .npy (single array)")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        if data.size > 0:
            print(f"Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
        print(f"\nPreview (first few elements):")
        print(data if data.size <= 20 else data.flatten()[:20])
    
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect .npy and .npz files to view their contents and structure."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to the .npy or .npz file to inspect."
    )
    args = parser.parse_args()
    
    if args.file_path:
        check_file(args.file_path)
    else:
        print("Usage: python check_npz.py <file_path>")
        print("Example: python check_npz.py data/sample.npz")


if __name__ == "__main__":
    main()
