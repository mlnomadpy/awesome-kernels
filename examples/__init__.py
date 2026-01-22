"""
Awesome Kernels Examples - JAX Implementations

This package contains JAX-based implementations of kernel methods
for machine learning.
"""

from .kernel_ridge_regression import KernelRidgeRegression
from .random_fourier_features import RandomFourierFeatures, RFFRidgeRegression
from .mmd_test import MMDTest

__all__ = [
    "KernelRidgeRegression",
    "RandomFourierFeatures",
    "RFFRidgeRegression",
    "MMDTest",
]
