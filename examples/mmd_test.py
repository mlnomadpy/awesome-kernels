"""
Maximum Mean Discrepancy (MMD) Two-Sample Test implemented in JAX.

MMD is a kernel-based statistic for comparing two distributions by measuring
the distance between their mean embeddings in a Reproducing Kernel Hilbert Space.

Based on:
    Gretton, A., et al. (2012). A Kernel Two-Sample Test.
    Journal of Machine Learning Research, 13, 723-773.
"""

import jax
import jax.numpy as jnp
from jax import random, Array
from typing import Optional, Dict, Any, Callable, Union

KernelFunction = Callable[[Array, Array], Array]


def rbf_kernel(X: Array, Y: Array, gamma: float) -> Array:
    """
    Compute RBF kernel matrix.

    Parameters
    ----------
    X : Array of shape (n, d)
    Y : Array of shape (m, d)
    gamma : float
        Kernel bandwidth parameter

    Returns
    -------
    K : Array of shape (n, m)
    """
    sq_norm_X = jnp.sum(X**2, axis=1, keepdims=True)
    sq_norm_Y = jnp.sum(Y**2, axis=1, keepdims=True)
    sq_dist = sq_norm_X + sq_norm_Y.T - 2 * jnp.dot(X, Y.T)
    return jnp.exp(-gamma * sq_dist)


def median_heuristic(X: Array, Y: Array) -> float:
    """
    Compute kernel bandwidth using the median heuristic.

    Sets gamma = 1 / (2 * median_distance^2).

    Parameters
    ----------
    X : Array of shape (n, d)
    Y : Array of shape (m, d)

    Returns
    -------
    gamma : float
        Computed bandwidth parameter
    """
    XY = jnp.vstack([X, Y])
    n = len(XY)

    # Compute pairwise squared distances
    sq_norm = jnp.sum(XY**2, axis=1, keepdims=True)
    sq_dist = sq_norm + sq_norm.T - 2 * jnp.dot(XY, XY.T)

    # Get upper triangle (excluding diagonal)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    dists = jnp.sqrt(jnp.maximum(sq_dist[mask], 0.0))

    median_dist = jnp.median(dists)
    return 1.0 / (2 * median_dist**2 + 1e-10)


class MMDTest:
    """
    Maximum Mean Discrepancy Two-Sample Test implemented in JAX.

    Tests whether two samples come from the same distribution by computing
    the MMD statistic and comparing against a permutation null distribution.

    The MMD measures the distance between mean embeddings:
        MMD^2[P, Q] = ||μ_P - μ_Q||_H^2
                    = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel function or name ('rbf')
    gamma : float, default=None
        RBF kernel bandwidth. If None, uses median heuristic.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from examples import MMDTest
    >>> key = random.PRNGKey(42)
    >>> key1, key2 = random.split(key)
    >>> X = random.normal(key1, (100, 2))
    >>> Y = random.normal(key2, (100, 2)) + 0.5  # Shifted distribution
    >>> mmd_test = MMDTest()
    >>> result = mmd_test.permutation_test(X, Y, n_permutations=500)
    >>> print(f"MMD² = {result['statistic']:.6f}, p-value = {result['p_value']:.4f}")
    """

    def __init__(
        self,
        kernel: Union[str, KernelFunction] = "rbf",
        gamma: Optional[float] = None,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.gamma_: Optional[float] = None

    def _compute_kernel(self, X: Array, Y: Array) -> Array:
        """Compute kernel matrix."""
        if callable(self.kernel):
            return self.kernel(X, Y)
        elif self.kernel == "rbf":
            if self.gamma_ is None:
                raise ValueError("Gamma not set. Call compute_mmd_squared first.")
            return rbf_kernel(X, Y, self.gamma_)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def compute_mmd_squared(
        self, X: Array, Y: Array, unbiased: bool = True
    ) -> float:
        """
        Compute the MMD^2 statistic.

        Parameters
        ----------
        X : Array of shape (n, d)
            Samples from distribution P
        Y : Array of shape (m, d)
            Samples from distribution Q
        unbiased : bool, default=True
            Whether to use the unbiased estimator

        Returns
        -------
        mmd_squared : float
            The MMD^2 statistic
        """
        # Set bandwidth if not specified
        if self.gamma is None:
            self.gamma_ = median_heuristic(X, Y)
        else:
            self.gamma_ = self.gamma

        n, m = len(X), len(Y)

        K_XX = self._compute_kernel(X, X)
        K_YY = self._compute_kernel(Y, Y)
        K_XY = self._compute_kernel(X, Y)

        if unbiased:
            # Unbiased U-statistic estimator (exclude diagonal)
            term_XX = (jnp.sum(K_XX) - jnp.trace(K_XX)) / (n * (n - 1))
            term_YY = (jnp.sum(K_YY) - jnp.trace(K_YY)) / (m * (m - 1))
            term_XY = jnp.sum(K_XY) / (n * m)
        else:
            # Biased V-statistic estimator
            term_XX = jnp.mean(K_XX)
            term_YY = jnp.mean(K_YY)
            term_XY = jnp.mean(K_XY)

        return float(term_XX + term_YY - 2 * term_XY)

    def permutation_test(
        self,
        X: Array,
        Y: Array,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform MMD permutation test.

        Under the null hypothesis (P = Q), the samples are exchangeable,
        so we can estimate the null distribution by permuting the combined sample.

        Parameters
        ----------
        X : Array of shape (n, d)
            Samples from distribution P
        Y : Array of shape (m, d)
            Samples from distribution Q
        n_permutations : int, default=1000
            Number of permutations
        alpha : float, default=0.05
            Significance level
        random_state : int, default=None
            Random seed for reproducibility

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'statistic': observed MMD^2 value
            - 'p_value': permutation p-value
            - 'reject_null': whether to reject H0 at level alpha
            - 'threshold': critical value at level alpha
            - 'null_distribution': array of null MMD^2 values
        """
        n, m = len(X), len(Y)

        # Observed statistic
        observed_mmd = self.compute_mmd_squared(X, Y)

        # Pool samples
        combined = jnp.vstack([X, Y])

        # Generate permutations and compute null distribution
        if random_state is None:
            key = random.PRNGKey(0)
        else:
            key = random.PRNGKey(random_state)

        null_distribution = []
        for i in range(n_permutations):
            key, subkey = random.split(key)
            perm = random.permutation(subkey, n + m)
            X_perm = combined[perm[:n]]
            Y_perm = combined[perm[n:]]
            null_distribution.append(self.compute_mmd_squared(X_perm, Y_perm))

        null_distribution = jnp.array(null_distribution)

        # Compute p-value (proportion of null >= observed)
        p_value = float(
            (jnp.sum(null_distribution >= observed_mmd) + 1) / (n_permutations + 1)
        )

        # Critical value
        threshold = float(jnp.quantile(null_distribution, 1 - alpha))

        return {
            "statistic": observed_mmd,
            "p_value": p_value,
            "reject_null": p_value < alpha,
            "threshold": threshold,
            "null_distribution": null_distribution,
        }

    def linear_time_mmd(
        self, X: Array, Y: Array, random_state: Optional[int] = None
    ) -> float:
        """
        Compute linear-time MMD estimator.

        Uses pairing to achieve O(n) complexity instead of O(n^2).

        Parameters
        ----------
        X : Array of shape (n, d)
            Samples from distribution P
        Y : Array of shape (m, d)
            Samples from distribution Q
        random_state : int, default=None
            Random seed for reproducibility (used for shuffling if needed)

        Returns
        -------
        mmd_linear : float
            Linear-time MMD estimate
        """
        n = min(len(X), len(Y))
        n = n - (n % 2)  # Make even

        X, Y = X[:n], Y[:n]

        # Set bandwidth if not specified
        if self.gamma is None:
            self.gamma_ = median_heuristic(X, Y)
        else:
            self.gamma_ = self.gamma

        # Compute h-statistics for each pair
        h_values = []
        for i in range(0, n, 2):
            x1, x2 = X[i : i + 1], X[i + 1 : i + 2]
            y1, y2 = Y[i : i + 1], Y[i + 1 : i + 2]

            k_xx = self._compute_kernel(x1, x2)[0, 0]
            k_yy = self._compute_kernel(y1, y2)[0, 0]
            k_xy1 = self._compute_kernel(x1, y2)[0, 0]
            k_xy2 = self._compute_kernel(x2, y1)[0, 0]

            h = k_xx + k_yy - k_xy1 - k_xy2
            h_values.append(h)

        return float(jnp.mean(jnp.array(h_values)))
