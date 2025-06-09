"""
Fast implementation of Robust PCA with Python bindings
"""

__version__ = "0.1.0"
__author__ = "Rust RobustPCA"

from .rpca import RobustPCA, robust_pca_with_components

__all__ = ['RobustPCA', 'robust_pca_with_components']