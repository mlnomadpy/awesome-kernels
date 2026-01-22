"""
Unit tests for MMD Two-Sample Test implementation.
"""

import pytest
import jax.numpy as jnp
from jax import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.mmd_test import MMDTest, rbf_kernel, median_heuristic


class TestKernelFunctions:
    """Tests for kernel helper functions."""

    def test_rbf_kernel_shape(self):
        """Test RBF kernel output shape."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        Y = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        K = rbf_kernel(X, Y, gamma=1.0)
        assert K.shape == (3, 2)

    def test_rbf_kernel_symmetry(self):
        """Test that RBF kernel matrix is symmetric."""
        key = random.PRNGKey(42)
        X = random.normal(key, (10, 5))
        K = rbf_kernel(X, X, gamma=0.5)
        assert jnp.allclose(K, K.T, atol=1e-6)

    def test_rbf_kernel_self_similarity(self):
        """Test that RBF kernel is 1 for identical points."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        K = rbf_kernel(X, X, gamma=1.0)
        assert jnp.allclose(jnp.diag(K), jnp.ones(2), atol=1e-6)

    def test_median_heuristic_positive(self):
        """Test that median heuristic returns positive gamma."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (20, 5))
        Y = random.normal(key2, (20, 5))
        gamma = median_heuristic(X, Y)
        assert gamma > 0


class TestMMDTest:
    """Tests for MMDTest class."""

    def test_compute_mmd_squared_shape(self):
        """Test that compute_mmd_squared returns a scalar."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 5))
        Y = random.normal(key2, (50, 5))

        mmd_test = MMDTest(gamma=1.0)
        mmd_sq = mmd_test.compute_mmd_squared(X, Y)

        assert isinstance(mmd_sq, float)

    def test_mmd_same_distribution_small(self):
        """Test that MMD is small when distributions are the same."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        # Same distribution (standard normal)
        X = random.normal(key1, (100, 5))
        Y = random.normal(key2, (100, 5))

        mmd_test = MMDTest(gamma=0.5)
        mmd_sq = mmd_test.compute_mmd_squared(X, Y)

        # MMD should be close to zero for same distribution
        assert mmd_sq < 0.1

    def test_mmd_different_distribution_large(self):
        """Test that MMD is large when distributions differ significantly."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        # Different distributions (shifted means)
        X = random.normal(key1, (100, 5))
        Y = random.normal(key2, (100, 5)) + 2.0  # Large shift

        mmd_test = MMDTest(gamma=0.5)
        mmd_sq = mmd_test.compute_mmd_squared(X, Y)

        # MMD should be larger for different distributions
        assert mmd_sq > 0.1

    def test_mmd_unbiased_vs_biased(self):
        """Test difference between unbiased and biased estimators."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 5))
        Y = random.normal(key2, (50, 5))

        mmd_test = MMDTest(gamma=0.5)
        mmd_unbiased = mmd_test.compute_mmd_squared(X, Y, unbiased=True)
        mmd_biased = mmd_test.compute_mmd_squared(X, Y, unbiased=False)

        # Both should be similar but not identical
        assert abs(mmd_unbiased - mmd_biased) < 0.1

    def test_permutation_test_same_distribution(self):
        """Test permutation test doesn't reject when distributions are same."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (80, 3))
        Y = random.normal(key2, (80, 3))

        mmd_test = MMDTest()
        result = mmd_test.permutation_test(
            X, Y, n_permutations=200, alpha=0.05, random_state=42
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "reject_null" in result
        assert "threshold" in result
        assert "null_distribution" in result

        # Should not reject null for same distribution (usually)
        # Note: This might occasionally fail due to randomness
        # We use a weaker assertion
        assert result["p_value"] > 0.01

    def test_permutation_test_different_distribution(self):
        """Test permutation test rejects when distributions differ."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (100, 3))
        Y = random.normal(key2, (100, 3)) + 1.0  # Shifted distribution

        mmd_test = MMDTest()
        result = mmd_test.permutation_test(
            X, Y, n_permutations=200, alpha=0.05, random_state=42
        )

        # Should reject null for different distributions
        assert result["p_value"] < 0.1  # More lenient for test stability

    def test_permutation_test_null_distribution_length(self):
        """Test that null distribution has correct length."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (30, 2))
        Y = random.normal(key2, (30, 2))

        n_perms = 150
        mmd_test = MMDTest()
        result = mmd_test.permutation_test(
            X, Y, n_permutations=n_perms, random_state=42
        )

        assert len(result["null_distribution"]) == n_perms

    def test_linear_time_mmd_shape(self):
        """Test that linear_time_mmd returns a scalar."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 5))
        Y = random.normal(key2, (50, 5))

        mmd_test = MMDTest(gamma=1.0)
        mmd_linear = mmd_test.linear_time_mmd(X, Y)

        assert isinstance(mmd_linear, float)

    def test_linear_time_mmd_same_distribution(self):
        """Test linear-time MMD is close to zero for same distribution."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (200, 5))
        Y = random.normal(key2, (200, 5))

        mmd_test = MMDTest(gamma=0.5)
        mmd_linear = mmd_test.linear_time_mmd(X, Y)

        # Should be close to zero
        assert abs(mmd_linear) < 0.2

    def test_linear_time_mmd_different_distribution(self):
        """Test linear-time MMD detects distribution shift."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (200, 5))
        Y = random.normal(key2, (200, 5)) + 1.0

        mmd_test = MMDTest(gamma=0.5)
        mmd_linear = mmd_test.linear_time_mmd(X, Y)

        # Should be positive for different distributions
        assert mmd_linear > 0

    def test_median_heuristic_used_when_gamma_none(self):
        """Test that median heuristic is used when gamma is None."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 5))
        Y = random.normal(key2, (50, 5))

        mmd_test = MMDTest(gamma=None)
        mmd_test.compute_mmd_squared(X, Y)

        assert mmd_test.gamma_ is not None
        assert mmd_test.gamma_ > 0

    def test_custom_gamma(self):
        """Test that custom gamma is used correctly."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 5))
        Y = random.normal(key2, (50, 5))

        custom_gamma = 2.5
        mmd_test = MMDTest(gamma=custom_gamma)
        mmd_test.compute_mmd_squared(X, Y)

        assert mmd_test.gamma_ == custom_gamma

    def test_callable_kernel(self):
        """Test MMD with callable kernel function."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (30, 3))
        Y = random.normal(key2, (30, 3))

        def custom_kernel(X, Y):
            return jnp.dot(X, Y.T)  # Linear kernel

        mmd_test = MMDTest(kernel=custom_kernel)
        mmd_sq = mmd_test.compute_mmd_squared(X, Y)

        assert isinstance(mmd_sq, float)

    def test_unknown_kernel_raises(self):
        """Test that unknown kernel string raises ValueError."""
        X = jnp.array([[0.0, 1.0]])
        Y = jnp.array([[1.0, 0.0]])

        mmd_test = MMDTest(kernel="unknown", gamma=1.0)
        with pytest.raises(ValueError, match="Unknown kernel"):
            mmd_test.compute_mmd_squared(X, Y)

    def test_different_sample_sizes(self):
        """Test MMD with different sample sizes."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 3))
        Y = random.normal(key2, (80, 3))

        mmd_test = MMDTest(gamma=1.0)
        mmd_sq = mmd_test.compute_mmd_squared(X, Y)

        assert isinstance(mmd_sq, float)

    def test_reproducibility_permutation_test(self):
        """Test that random_state produces reproducible results."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (30, 3))
        Y = random.normal(key2, (30, 3))

        mmd_test = MMDTest(gamma=1.0)

        result1 = mmd_test.permutation_test(
            X, Y, n_permutations=100, random_state=42
        )
        result2 = mmd_test.permutation_test(
            X, Y, n_permutations=100, random_state=42
        )

        assert result1["p_value"] == result2["p_value"]
        assert jnp.allclose(
            result1["null_distribution"], result2["null_distribution"], atol=1e-6
        )


class TestMMDStatisticalProperties:
    """Tests for statistical properties of MMD."""

    def test_mmd_symmetry(self):
        """Test that MMD(X, Y) = MMD(Y, X)."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (50, 5))
        Y = random.normal(key2, (50, 5)) + 0.5

        mmd_test = MMDTest(gamma=0.5)
        mmd_xy = mmd_test.compute_mmd_squared(X, Y)
        mmd_yx = mmd_test.compute_mmd_squared(Y, X)

        assert abs(mmd_xy - mmd_yx) < 1e-6

    def test_mmd_non_negative(self):
        """Test that MMD^2 is non-negative (with unbiased estimator this can be slightly negative)."""
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        X = random.normal(key1, (100, 5))
        Y = random.normal(key2, (100, 5))

        mmd_test = MMDTest(gamma=0.5)
        # Biased estimator should always be non-negative
        mmd_biased = mmd_test.compute_mmd_squared(X, Y, unbiased=False)
        assert mmd_biased >= -1e-6  # Allow small numerical error

    def test_mmd_zero_for_identical_samples(self):
        """Test that MMD is zero when X = Y."""
        key = random.PRNGKey(42)
        X = random.normal(key, (50, 5))

        mmd_test = MMDTest(gamma=0.5)
        mmd_sq = mmd_test.compute_mmd_squared(X, X, unbiased=False)

        assert abs(mmd_sq) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
