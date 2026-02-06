"""
Latent Space Analysis Package

Provides tools for analyzing VQ-VAE latent space, token distributions, and similarity metrics.
"""

# Main analysis classes
from .vqvae_coverage_calculation import CodebookCoverageAnalyzer
from .analyze_token_influence import TokenInfluenceAnalyzer

# Utility functions
from .analyze_token_distances import (
    load_codebook_embeddings,
    load_tokens_from_h5,
    tokens_to_embeddings
)

from wicompass.knn_coverage.knn_coverage import (
    build_index,
    intra_knn_search,
    analyze_knn_coverage
)

from .analyze_knn_pose_similarity import (
    analyze_knn_pose_similarity,
    find_knn_neighbors,
    calculate_pose_similarity_metrics
)

__all__ = [
    # Analysis classes
    'CodebookCoverageAnalyzer',
    'TokenInfluenceAnalyzer',
    
    # Token distance analysis
    'load_codebook_embeddings',
    'load_tokens_from_h5', 
    'tokens_to_embeddings',
    
    # k-NN coverage analysis
    'build_index',
    'intra_knn_search',
    'analyze_knn_coverage',
    
    # k-NN pose similarity analysis
    'analyze_knn_pose_similarity',
    'find_knn_neighbors',
    'calculate_pose_similarity_metrics',
]