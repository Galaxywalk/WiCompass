# -*- coding: utf-8 -*-
"""
Token Space Sampling Methods

This module provides various sampling strategies for token/latent spaces:

- **Random Sampling**: Simple random selection without replacement
- **PPS Sampling**: Probability Proportional to Size (capped), favors sparse regions
- **FPS Sampling**: Farthest Point Sampling, maximizes coverage

All methods support:
- H5 and NPY input formats
- Automatic deduplication
- Candidate/seed specification
- GPU acceleration via FAISS
- Metadata and format preservation

Usage Examples:
    # Programmatic usage
    from wicompass.token_space_sampling import (
        random_sampling,
        sample_capped_pps,
        farthest_point_sampling,
        load_data,
        dedupe_rows,
    )
    
    # Command-line usage
    python src/wicompass/token_space_sampling/random_sampling.py --A-path tokens.h5 --budget 1000
    python src/wicompass/token_space_sampling/pps_sampling.py --A-path tokens.h5 --budget 40000 --k 8
    python src/wicompass/token_space_sampling/fps_sampling.py --A-path tokens.h5 --budget 5000
    python src/wicompass/token_space_sampling/convert_sampled_tokens_to_poses.py --tokens-npy tokens.npy --model model.pth --config config.json
"""

from .base import (
    # Data loading
    load_data,
    load_indices_from_npy,
    # Deduplication and index utilities
    dedupe_rows,
    to_indices,
    map_to_unique_space,
    # Math utilities
    compute_quantile,
    # FAISS utilities
    build_faiss_index,
    compute_knn_radius,
    # Result saving
    save_results,
    get_device_info,
)

from .random_sampling import random_sampling
from .pps_sampling import sample_capped_pps
from .fps_sampling import farthest_point_sampling, filter_outliers_by_radius
from .convert_sampled_tokens_to_poses import tokens_to_poses, load_tokens

__all__ = [
    # Base utilities
    "load_data",
    "load_indices_from_npy",
    "dedupe_rows",
    "to_indices",
    "map_to_unique_space",
    "compute_quantile",
    "build_faiss_index",
    "compute_knn_radius",
    "save_results",
    "get_device_info",
    # Sampling functions
    "random_sampling",
    "sample_capped_pps",
    "farthest_point_sampling",
    "filter_outliers_by_radius",
    # Token to pose conversion
    "tokens_to_poses",
    "load_tokens",
]

