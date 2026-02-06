#!/usr/bin/env python3
"""
Script to clear outputs from all Jupyter notebooks in the repository.
"""

import os
import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("Error: nbformat is not installed. Please install it with: pip install nbformat")
    sys.exit(1)


def clear_notebook_outputs(notebook_path: Path) -> bool:
    """
    Clear all outputs from a Jupyter notebook.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Track if any changes were made
        modified = False
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # Clear outputs
                if cell.outputs:
                    cell.outputs = []
                    modified = True
                # Clear execution count
                if cell.execution_count is not None:
                    cell.execution_count = None
                    modified = True
        
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            return True
        return False
        
    except Exception as e:
        print(f"  Error processing {notebook_path}: {e}")
        return False


def find_and_clear_notebooks(root_dir: Path, exclude_dirs: list = None) -> tuple:
    """
    Find all notebooks in the directory tree and clear their outputs.
    
    Args:
        root_dir: Root directory to search
        exclude_dirs: List of directory names to exclude
        
    Returns:
        Tuple of (total_found, total_cleared)
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'node_modules', '.ipynb_checkpoints', 'venv', '.venv']
    
    total_found = 0
    total_cleared = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove excluded directories from the search
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            if filename.endswith('.ipynb'):
                notebook_path = Path(dirpath) / filename
                total_found += 1
                
                # Get relative path for display
                rel_path = notebook_path.relative_to(root_dir)
                
                if clear_notebook_outputs(notebook_path):
                    print(f"  [Cleared] {rel_path}")
                    total_cleared += 1
                else:
                    print(f"  [Skipped] {rel_path} (no outputs or already clean)")
    
    return total_found, total_cleared


def main():
    # Get the repository root (parent of tools directory)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    print(f"Searching for Jupyter notebooks in: {repo_root}")
    print("-" * 60)
    
    total_found, total_cleared = find_and_clear_notebooks(repo_root)
    
    print("-" * 60)
    print(f"Summary: Found {total_found} notebooks, cleared {total_cleared}")


if __name__ == "__main__":
    main()

