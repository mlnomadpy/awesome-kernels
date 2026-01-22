"""
Unit tests for Kernel Ridge Regression implementation.
"""

import pytest
import jax.numpy as jnp
from jax import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.kernel_ridge_regression import (
    KernelRidgeRegression,
    rbf_kernel,
    linear_kernel,
    polynomial_kernel,
)


class TestKernelFunctions:
    """Tests for individual kernel functions."""

    def test_rbf_kernel_shape(self):
        """Test RBF kernel output shape."""
        X1 = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        X2 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        K = rbf_kernel(X1, X2, gamma=1.0)
        assert K.shape == (3, 2)

    def test_rbf_kernel_self_similarity(self):
        """Test that RBF kernel is 1 for identical points."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        K = rbf_kernel(X, X, gamma=1.0)
        # Diagonal should be 1
        assert jnp.allclose(jnp.diag(K), jnp.ones(2), atol=1e-6)

    def test_rbf_kernel_symmetry(self):
        """Test that RBF kernel matrix is symmetric."""
        key = random.PRNGKey(42)
        X = random.normal(key, (10, 5))
        K = rbf_kernel(X, X, gamma=0.5)
        assert jnp.allclose(K, K.T, atol=1e-6)

    def test_rbf_kernel_positive_semidefinite(self):
        """Test that RBF kernel is positive semi-definite."""
        key = random.PRNGKey(42)
        X = random.normal(key, (10, 5))
        K = rbf_kernel(X, X, gamma=0.5)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-6)  # Allow small numerical error

    def test_linear_kernel_shape(self):
        """Test linear kernel output shape."""
        X1 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        X2 = jnp.array([[1.0, 1.0]])
        K = linear_kernel(X1, X2)
        assert K.shape == (2, 1)

    def test_linear_kernel_correctness(self):
        """Test linear kernel computes correct dot products."""
        X1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = jnp.array([[1.0, 1.0]])
        K = linear_kernel(X1, X2)
        expected = jnp.array([[3.0], [7.0]])  # [1*1 + 2*1, 3*1 + 4*1]
        assert jnp.allclose(K, expected, atol=1e-6)

    def test_polynomial_kernel_shape(self):
        """Test polynomial kernel output shape."""
        X1 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        X2 = jnp.array([[1.0, 1.0], [0.0, 0.0]])
        K = polynomial_kernel(X1, X2, gamma=1.0, coef0=1.0, degree=2)
        assert K.shape == (2, 2)

    def test_polynomial_kernel_correctness(self):
        """Test polynomial kernel computes correct values."""
        X1 = jnp.array([[1.0, 2.0]])
        X2 = jnp.array([[1.0, 1.0]])
        # K = (gamma * <x, y> + coef0)^degree = (1 * 3 + 1)^2 = 16
        K = polynomial_kernel(X1, X2, gamma=1.0, coef0=1.0, degree=2)
        expected = jnp.array([[16.0]])
        assert jnp.allclose(K, expected, atol=1e-6)


class TestKernelRidgeRegression:
    """Tests for KernelRidgeRegression class."""

    def test_fit_predict_shape(self):
        """Test that fit and predict work correctly."""
        key = random.PRNGKey(42)
        X = random.normal(key, (50, 3))
        y = jnp.sum(X, axis=1)

        krr = KernelRidgeRegression(kernel="rbf", gamma=1.0, alpha=0.1)
        krr.fit(X, y)

        y_pred = krr.predict(X)
        assert y_pred.shape == (50,)

    def test_fit_stores_training_data(self):
        """Test that fit stores training data and coefficients."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        y = jnp.array([1.0, 2.0])

        krr = KernelRidgeRegression(kernel="linear", alpha=0.1)
        krr.fit(X, y)

        assert krr.X_train_ is not None
        assert krr.dual_coef_ is not None
        assert len(krr.dual_coef_) == 2

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fit."""
        krr = KernelRidgeRegression()
        X = jnp.array([[0.0, 1.0]])

        with pytest.raises(ValueError, match="Model not fitted"):
            krr.predict(X)

    def test_rbf_kernel_regression(self):
        """Test KRR with RBF kernel on sinusoidal data."""
        # Generate sinusoidal data
        X = jnp.linspace(0, 2 * jnp.pi, 100).reshape(-1, 1)
        y = jnp.sin(X).ravel()

        krr = KernelRidgeRegression(kernel="rbf", gamma=1.0, alpha=0.001)
        krr.fit(X, y)
        y_pred = krr.predict(X)

        # Should fit training data well
        mse = jnp.mean((y - y_pred) ** 2)
        assert mse < 0.01  # Very good fit expected

    def test_linear_kernel_regression(self):
        """Test KRR with linear kernel on linear data."""
        key = random.PRNGKey(42)
        X = random.normal(key, (100, 5))
        true_weights = jnp.array([1.0, 2.0, -1.0, 0.5, -0.5])
        y = jnp.dot(X, true_weights)

        krr = KernelRidgeRegression(kernel="linear", alpha=0.001)
        krr.fit(X, y)
        y_pred = krr.predict(X)

        # Should fit linear data well
        mse = jnp.mean((y - y_pred) ** 2)
        assert mse < 0.01

    def test_polynomial_kernel_regression(self):
        """Test KRR with polynomial kernel on quadratic data."""
        X = jnp.linspace(-2, 2, 50).reshape(-1, 1)
        y = (X**2).ravel()

        krr = KernelRidgeRegression(
            kernel="polynomial", gamma=1.0, degree=2, coef0=0.0, alpha=0.001
        )
        krr.fit(X, y)
        y_pred = krr.predict(X)

        # Should fit quadratic data reasonably well
        mse = jnp.mean((y - y_pred) ** 2)
        assert mse < 0.1

    def test_regularization_effect(self):
        """Test that higher alpha leads to smoother predictions."""
        key = random.PRNGKey(42)
        X = jnp.linspace(0, 2 * jnp.pi, 50).reshape(-1, 1)
        y = jnp.sin(X).ravel() + 0.1 * random.normal(key, (50,))

        # Low regularization
        krr_low = KernelRidgeRegression(kernel="rbf", gamma=5.0, alpha=0.0001)
        krr_low.fit(X, y)

        # High regularization
        krr_high = KernelRidgeRegression(kernel="rbf", gamma=5.0, alpha=1.0)
        krr_high.fit(X, y)

        # Dual coefficients should have smaller norm with higher regularization
        norm_low = jnp.linalg.norm(krr_low.dual_coef_)
        norm_high = jnp.linalg.norm(krr_high.dual_coef_)
        assert norm_high < norm_low

    def test_unknown_kernel_raises(self):
        """Test that unknown kernel raises ValueError."""
        X = jnp.array([[0.0, 1.0]])
        y = jnp.array([1.0])

        krr = KernelRidgeRegression(kernel="unknown")
        with pytest.raises(ValueError, match="Unknown kernel"):
            krr.fit(X, y)

    def test_generalization(self):
        """Test that model generalizes to new data."""
        # Train on subset
        X_train = jnp.linspace(0, 2 * jnp.pi, 50).reshape(-1, 1)
        y_train = jnp.sin(X_train).ravel()

        # Test on different points
        X_test = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 30).reshape(-1, 1)
        y_test = jnp.sin(X_test).ravel()

        krr = KernelRidgeRegression(kernel="rbf", gamma=1.0, alpha=0.001)
        krr.fit(X_train, y_train)
        y_pred = krr.predict(X_test)

        # Should generalize well
        mse = jnp.mean((y_test - y_pred) ** 2)
        assert mse < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
