"""
Random Fourier Features implemented in JAX.

Random Fourier Features (RFF) approximate shift-invariant kernels using
explicit random feature maps, enabling linear-time training for kernel methods.

Based on:
    Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel Machines.
    NeurIPS 2007.
"""

import jax
import jax.numpy as jnp
from jax import random, Array
from typing import Optional, Literal

KernelType = Literal["rbf", "laplacian"]


class RandomFourierFeatures:
    """
    Random Fourier Features for kernel approximation in JAX.

    Approximates shift-invariant kernels using explicit feature maps based on
    Bochner's theorem. For an RBF kernel K(x, y) = exp(-gamma * ||x-y||^2),
    the approximation is:

        K(x, y) ≈ z(x)^T z(y)

    where z(x) = sqrt(2/D) * cos(omega^T x + b) with omega sampled from the
    spectral density (Gaussian for RBF kernel).

    Parameters
    ----------
    n_features : int, default=100
        Number of random features (D)
    gamma : float, default=1.0
        Kernel parameter (bandwidth)
    kernel : str, default='rbf'
        Kernel type: 'rbf' or 'laplacian'
    random_state : int, default=None
        Random seed for reproducibility

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from examples import RandomFourierFeatures
    >>> X = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    >>> rff = RandomFourierFeatures(n_features=100, gamma=1.0, random_state=42)
    >>> Z = rff.fit_transform(X)
    >>> K_approx = Z @ Z.T  # Approximate kernel matrix
    """

    def __init__(
        self,
        n_features: int = 100,
        gamma: float = 1.0,
        kernel: KernelType = "rbf",
        random_state: Optional[int] = None,
    ):
        self.n_features = n_features
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state

        # To be set during fit
        self.omega_: Optional[Array] = None
        self.b_: Optional[Array] = None

    def fit(self, X: Array) -> "RandomFourierFeatures":
        """
        Sample random frequencies from the spectral density.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features_in)
            Training data (used only to determine input dimension)

        Returns
        -------
        self : RandomFourierFeatures
            Fitted transformer
        """
        if self.random_state is None:
            key = random.PRNGKey(0)
        else:
            key = random.PRNGKey(self.random_state)

        n_features_in = X.shape[1]
        key1, key2 = random.split(key)

        # Sample frequencies from spectral density
        if self.kernel == "rbf":
            # Gaussian RBF: spectral density is Gaussian with variance 2*gamma
            self.omega_ = (
                random.normal(key1, (n_features_in, self.n_features))
                * jnp.sqrt(2 * self.gamma)
            )
        elif self.kernel == "laplacian":
            # Laplacian kernel: spectral density is Cauchy
            self.omega_ = (
                random.cauchy(key1, (n_features_in, self.n_features)) * self.gamma
            )
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        # Sample phase offsets uniformly from [0, 2*pi]
        self.b_ = random.uniform(key2, (self.n_features,)) * 2 * jnp.pi

        return self

    def transform(self, X: Array) -> Array:
        """
        Compute random Fourier features.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features_in)
            Input data

        Returns
        -------
        Z : Array of shape (n_samples, n_features)
            Random Fourier features
        """
        if self.omega_ is None or self.b_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        # Project data: n_samples x n_features
        projection = jnp.dot(X, self.omega_) + self.b_

        # Apply cosine and scale
        Z = jnp.sqrt(2.0 / self.n_features) * jnp.cos(projection)

        return Z

    def fit_transform(self, X: Array) -> Array:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class RFFRidgeRegression:
    """
    Ridge Regression with Random Fourier Features in JAX.

    Combines Random Fourier Features with ridge regression for scalable
    kernel regression. Training complexity is O(n*D^2) instead of O(n^3).

    Parameters
    ----------
    n_features : int, default=100
        Number of random features
    gamma : float, default=1.0
        RBF kernel parameter
    alpha : float, default=1.0
        Regularization strength
    random_state : int, default=None
        Random seed for reproducibility

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from examples import RFFRidgeRegression
    >>> X = jnp.linspace(0, 2*jnp.pi, 100).reshape(-1, 1)
    >>> y = jnp.sin(X).ravel()
    >>> model = RFFRidgeRegression(n_features=500, gamma=1.0, alpha=0.01)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    """

    def __init__(
        self,
        n_features: int = 100,
        gamma: float = 1.0,
        alpha: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_features = n_features
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state

        # To be set during fit
        self.rff_: Optional[RandomFourierFeatures] = None
        self.weights_: Optional[Array] = None

    def fit(self, X: Array, y: Array) -> "RFFRidgeRegression":
        """
        Fit the RFF Ridge Regression model.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features_in)
            Training data
        y : Array of shape (n_samples,)
            Target values

        Returns
        -------
        self : RFFRidgeRegression
            Fitted model
        """
        # Create and fit random features transformer
        self.rff_ = RandomFourierFeatures(
            n_features=self.n_features,
            gamma=self.gamma,
            random_state=self.random_state,
        )
        Z = self.rff_.fit_transform(X)

        # Solve regularized least squares in feature space
        # (Z^T Z + alpha * I)^{-1} Z^T y
        A = jnp.dot(Z.T, Z) + self.alpha * jnp.eye(self.n_features)
        b = jnp.dot(Z.T, y)
        self.weights_ = jax.scipy.linalg.solve(A, b, assume_a="pos")

        return self

    def predict(self, X: Array) -> Array:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features_in)
            Test data

        Returns
        -------
        y_pred : Array of shape (n_samples,)
            Predicted values
        """
        if self.rff_ is None or self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        Z = self.rff_.transform(X)
        return jnp.dot(Z, self.weights_)
