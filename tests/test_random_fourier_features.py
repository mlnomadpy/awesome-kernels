"""
Unit tests for Random Fourier Features implementation.
"""

import pytest
import jax.numpy as jnp
from jax import random

from examples.random_fourier_features import RandomFourierFeatures, RFFRidgeRegression


class TestRandomFourierFeatures:
    """Tests for RandomFourierFeatures class."""

    def test_fit_transform_shape(self):
        """Test that fit_transform produces correct output shape."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        rff = RandomFourierFeatures(n_features=50, gamma=1.0, random_state=42)
        Z = rff.fit_transform(X)
        assert Z.shape == (3, 50)

    def test_transform_shape(self):
        """Test that transform produces correct output shape."""
        X_train = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        X_test = jnp.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])

        rff = RandomFourierFeatures(n_features=100, gamma=1.0, random_state=42)
        rff.fit(X_train)
        Z_test = rff.transform(X_test)

        assert Z_test.shape == (3, 100)

    def test_fit_stores_parameters(self):
        """Test that fit stores omega and b parameters."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        rff = RandomFourierFeatures(n_features=50, gamma=1.0, random_state=42)
        rff.fit(X)

        assert rff.omega_ is not None
        assert rff.b_ is not None
        assert rff.omega_.shape == (2, 50)  # (n_features_in, n_features)
        assert rff.b_.shape == (50,)

    def test_transform_before_fit_raises(self):
        """Test that transform raises error before fit."""
        rff = RandomFourierFeatures(n_features=50)
        X = jnp.array([[0.0, 1.0]])

        with pytest.raises(ValueError, match="Transformer not fitted"):
            rff.transform(X)

    def test_reproducibility(self):
        """Test that random_state produces reproducible results."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])

        rff1 = RandomFourierFeatures(n_features=50, gamma=1.0, random_state=42)
        Z1 = rff1.fit_transform(X)

        rff2 = RandomFourierFeatures(n_features=50, gamma=1.0, random_state=42)
        Z2 = rff2.fit_transform(X)

        assert jnp.allclose(Z1, Z2, atol=1e-6)

    def test_different_seeds_produce_different_results(self):
        """Test that different random states produce different results."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])

        rff1 = RandomFourierFeatures(n_features=50, gamma=1.0, random_state=42)
        Z1 = rff1.fit_transform(X)

        rff2 = RandomFourierFeatures(n_features=50, gamma=1.0, random_state=123)
        Z2 = rff2.fit_transform(X)

        assert not jnp.allclose(Z1, Z2, atol=1e-3)

    def test_kernel_approximation_quality(self):
        """Test that RFF approximates RBF kernel accurately."""
        key = random.PRNGKey(42)
        X = random.normal(key, (50, 5))
        gamma = 0.5

        # Exact RBF kernel
        sq_norm = jnp.sum(X**2, axis=1, keepdims=True)
        sq_dist = sq_norm + sq_norm.T - 2 * jnp.dot(X, X.T)
        K_exact = jnp.exp(-gamma * sq_dist)

        # Approximate kernel with RFF
        rff = RandomFourierFeatures(n_features=1000, gamma=gamma, random_state=42)
        Z = rff.fit_transform(X)
        K_approx = jnp.dot(Z, Z.T)

        # Check relative error is small
        rel_error = jnp.linalg.norm(K_exact - K_approx, "fro") / jnp.linalg.norm(
            K_exact, "fro"
        )
        assert rel_error < 0.15  # Reasonable approximation

    def test_approximation_improves_with_more_features(self):
        """Test that approximation improves with more random features."""
        key = random.PRNGKey(42)
        X = random.normal(key, (30, 5))
        gamma = 0.5

        # Exact kernel
        sq_norm = jnp.sum(X**2, axis=1, keepdims=True)
        sq_dist = sq_norm + sq_norm.T - 2 * jnp.dot(X, X.T)
        K_exact = jnp.exp(-gamma * sq_dist)

        errors = []
        for n_features in [50, 200, 500]:
            rff = RandomFourierFeatures(
                n_features=n_features, gamma=gamma, random_state=42
            )
            Z = rff.fit_transform(X)
            K_approx = jnp.dot(Z, Z.T)
            error = jnp.linalg.norm(K_exact - K_approx, "fro") / jnp.linalg.norm(
                K_exact, "fro"
            )
            errors.append(error)

        # Error should generally decrease (or at least not increase much)
        assert errors[2] < errors[0]

    def test_laplacian_kernel(self):
        """Test RFF with Laplacian kernel."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        rff = RandomFourierFeatures(
            n_features=100, gamma=1.0, kernel="laplacian", random_state=42
        )
        Z = rff.fit_transform(X)
        assert Z.shape == (2, 100)

    def test_unknown_kernel_raises(self):
        """Test that unknown kernel raises ValueError."""
        X = jnp.array([[0.0, 1.0]])
        rff = RandomFourierFeatures(n_features=50, kernel="unknown")
        with pytest.raises(ValueError, match="Unknown kernel"):
            rff.fit(X)

    def test_feature_values_bounded(self):
        """Test that features are appropriately scaled."""
        key = random.PRNGKey(42)
        X = random.normal(key, (100, 10))

        rff = RandomFourierFeatures(n_features=200, gamma=1.0, random_state=42)
        Z = rff.fit_transform(X)

        # cos values are in [-1, 1], scaled by sqrt(2/D)
        scale = jnp.sqrt(2.0 / 200)
        assert jnp.all(jnp.abs(Z) <= scale + 1e-6)


class TestRFFRidgeRegression:
    """Tests for RFFRidgeRegression class."""

    def test_fit_predict_shape(self):
        """Test that fit and predict work correctly."""
        key = random.PRNGKey(42)
        X = random.normal(key, (50, 3))
        y = jnp.sum(X, axis=1)

        model = RFFRidgeRegression(n_features=100, gamma=1.0, alpha=0.1)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == (50,)

    def test_fit_stores_weights(self):
        """Test that fit stores RFF transformer and weights."""
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        y = jnp.array([1.0, 2.0])

        model = RFFRidgeRegression(n_features=50, alpha=0.1)
        model.fit(X, y)

        assert model.rff_ is not None
        assert model.weights_ is not None
        assert len(model.weights_) == 50

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fit."""
        model = RFFRidgeRegression()
        X = jnp.array([[0.0, 1.0]])

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_sinusoidal_regression(self):
        """Test RFF Ridge Regression on sinusoidal data."""
        X = jnp.linspace(0, 2 * jnp.pi, 100).reshape(-1, 1)
        y = jnp.sin(X).ravel()

        model = RFFRidgeRegression(
            n_features=500, gamma=1.0, alpha=0.001, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        mse = jnp.mean((y - y_pred) ** 2)
        assert mse < 0.05  # Reasonable fit

    def test_linear_regression(self):
        """Test RFF Ridge Regression on linear data."""
        key = random.PRNGKey(42)
        X = random.normal(key, (100, 5))
        true_weights = jnp.array([1.0, 2.0, -1.0, 0.5, -0.5])
        y = jnp.dot(X, true_weights)

        model = RFFRidgeRegression(
            n_features=500, gamma=0.5, alpha=0.001, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)

        mse = jnp.mean((y - y_pred) ** 2)
        assert mse < 0.1

    def test_reproducibility(self):
        """Test that random_state produces reproducible results."""
        key = random.PRNGKey(42)
        X = random.normal(key, (50, 3))
        y = jnp.sum(X, axis=1)

        model1 = RFFRidgeRegression(n_features=100, gamma=1.0, alpha=0.1, random_state=42)
        model1.fit(X, y)
        y_pred1 = model1.predict(X)

        model2 = RFFRidgeRegression(n_features=100, gamma=1.0, alpha=0.1, random_state=42)
        model2.fit(X, y)
        y_pred2 = model2.predict(X)

        assert jnp.allclose(y_pred1, y_pred2, atol=1e-6)

    def test_regularization_effect(self):
        """Test that higher alpha leads to smaller weights."""
        key = random.PRNGKey(42)
        X = jnp.linspace(0, 2 * jnp.pi, 50).reshape(-1, 1)
        y = jnp.sin(X).ravel() + 0.1 * random.normal(key, (50,))

        model_low = RFFRidgeRegression(
            n_features=200, gamma=1.0, alpha=0.0001, random_state=42
        )
        model_low.fit(X, y)

        model_high = RFFRidgeRegression(
            n_features=200, gamma=1.0, alpha=1.0, random_state=42
        )
        model_high.fit(X, y)

        norm_low = jnp.linalg.norm(model_low.weights_)
        norm_high = jnp.linalg.norm(model_high.weights_)
        assert norm_high < norm_low

    def test_generalization(self):
        """Test that model generalizes to new data."""
        X_train = jnp.linspace(0, 2 * jnp.pi, 100).reshape(-1, 1)
        y_train = jnp.sin(X_train).ravel()

        X_test = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 50).reshape(-1, 1)
        y_test = jnp.sin(X_test).ravel()

        model = RFFRidgeRegression(
            n_features=500, gamma=1.0, alpha=0.001, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = jnp.mean((y_test - y_pred) ** 2)
        assert mse < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
