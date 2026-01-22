"""
Kernel Ridge Regression implemented in JAX.

Kernel Ridge Regression (KRR) combines ridge regression with the kernel trick,
enabling nonlinear regression in a Reproducing Kernel Hilbert Space (RKHS).
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional, Literal

KernelType = Literal["rbf", "linear", "polynomial"]


def rbf_kernel(X1: Array, X2: Array, gamma: float = 1.0) -> Array:
    """
    Compute RBF (Gaussian) kernel matrix.

    K(x, y) = exp(-gamma * ||x - y||^2)

    Parameters
    ----------
    X1 : Array of shape (n, d)
    X2 : Array of shape (m, d)
    gamma : float
        Kernel bandwidth parameter

    Returns
    -------
    K : Array of shape (n, m)
    """
    # Compute squared Euclidean distances
    sq_norm1 = jnp.sum(X1**2, axis=1, keepdims=True)
    sq_norm2 = jnp.sum(X2**2, axis=1, keepdims=True)
    sq_dist = sq_norm1 + sq_norm2.T - 2 * jnp.dot(X1, X2.T)
    return jnp.exp(-gamma * sq_dist)


def linear_kernel(X1: Array, X2: Array) -> Array:
    """
    Compute linear kernel matrix.

    K(x, y) = x^T y

    Parameters
    ----------
    X1 : Array of shape (n, d)
    X2 : Array of shape (m, d)

    Returns
    -------
    K : Array of shape (n, m)
    """
    return jnp.dot(X1, X2.T)


def polynomial_kernel(
    X1: Array, X2: Array, gamma: float = 1.0, coef0: float = 1.0, degree: int = 3
) -> Array:
    """
    Compute polynomial kernel matrix.

    K(x, y) = (gamma * x^T y + coef0)^degree

    Parameters
    ----------
    X1 : Array of shape (n, d)
    X2 : Array of shape (m, d)
    gamma : float
        Scaling factor
    coef0 : float
        Independent term
    degree : int
        Polynomial degree

    Returns
    -------
    K : Array of shape (n, m)
    """
    return (gamma * jnp.dot(X1, X2.T) + coef0) ** degree


class KernelRidgeRegression:
    """
    Kernel Ridge Regression implementation in JAX.

    Solves the regularized least squares problem in a Reproducing Kernel
    Hilbert Space using the representer theorem.

    The optimization problem is:
        min_f (1/n) sum_i (y_i - f(x_i))^2 + alpha * ||f||_H^2

    The solution has the form:
        alpha = (K + n*alpha*I)^{-1} y
        f(x) = sum_i alpha_i K(x_i, x)

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: 'rbf', 'linear', or 'polynomial'
    gamma : float, default=1.0
        RBF/polynomial kernel parameter
    degree : int, default=3
        Polynomial kernel degree
    coef0 : float, default=1.0
        Polynomial kernel coefficient
    alpha : float, default=1.0
        Regularization strength

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from examples import KernelRidgeRegression
    >>> X = jnp.linspace(0, 2*jnp.pi, 100).reshape(-1, 1)
    >>> y = jnp.sin(X).ravel()
    >>> krr = KernelRidgeRegression(kernel='rbf', gamma=1.0, alpha=0.01)
    >>> krr.fit(X, y)
    >>> y_pred = krr.predict(X)
    """

    def __init__(
        self,
        kernel: KernelType = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        alpha: float = 1.0,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha

        # To be set during fit
        self.X_train_: Optional[Array] = None
        self.dual_coef_: Optional[Array] = None

    def _compute_kernel(self, X1: Array, X2: Array) -> Array:
        """Compute kernel matrix between X1 and X2."""
        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, self.gamma)
        elif self.kernel == "linear":
            return linear_kernel(X1, X2)
        elif self.kernel == "polynomial":
            return polynomial_kernel(X1, X2, self.gamma, self.coef0, self.degree)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: Array, y: Array) -> "KernelRidgeRegression":
        """
        Fit the Kernel Ridge Regression model.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features)
            Training data
        y : Array of shape (n_samples,)
            Target values

        Returns
        -------
        self : KernelRidgeRegression
            Fitted model
        """
        self.X_train_ = X
        n = len(X)

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Solve (K + n*alpha*I)^{-1} y using JAX's linear solver
        self.dual_coef_ = jax.scipy.linalg.solve(
            K + n * self.alpha * jnp.eye(n), y, assume_a="pos"
        )

        return self

    def predict(self, X: Array) -> Array:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features)
            Test data

        Returns
        -------
        y_pred : Array of shape (n_samples,)
            Predicted values
        """
        if self.X_train_ is None or self.dual_coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        K = self._compute_kernel(X, self.X_train_)
        return jnp.dot(K, self.dual_coef_)
