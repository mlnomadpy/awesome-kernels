---
slug: implementing-random-fourier-features
title: Random Fourier Features - Scalable Kernel Approximation
authors: [mlnomadpy]
tags: [kernel, jax, python, implementation, tutorial]
---

# Random Fourier Features: Making Kernel Methods Scalable

Random Fourier Features (RFF) is a beautiful technique that approximates kernel functions with explicit feature maps, enabling linear-time training while maintaining kernel-like performance. Let's implement it from scratch using JAX!

<!--truncate-->

## The Bottleneck of Kernel Methods

Standard kernel methods require:
- Computing and storing an $n \times n$ kernel matrix: $O(n^2)$ space
- Solving linear systems: $O(n^3)$ time

For millions of data points, this becomes impractical.

## Bochner's Theorem to the Rescue

For shift-invariant kernels $K(x, y) = k(x - y)$, Bochner's theorem tells us:

$$k(\delta) = \int p(\omega) e^{i\omega^T\delta} d\omega$$

where $p(\omega)$ is the spectral density (Fourier transform of $k$).

## The Key Insight

We can approximate the integral via Monte Carlo:

$$k(x-y) \approx \frac{1}{D}\sum_{j=1}^D e^{i\omega_j^T(x-y)}$$

where $\omega_1, \ldots, \omega_D \sim p(\omega)$.

Using Euler's formula and taking real parts:

$$k(x-y) \approx z(x)^T z(y)$$

where:

$$z(x) = \sqrt{\frac{2}{D}}[\cos(\omega_1^T x + b_1), \ldots, \cos(\omega_D^T x + b_D)]$$

with $b_j \sim \text{Uniform}(0, 2\pi)$.

## JAX Implementation

```python
import jax
import jax.numpy as jnp
from jax import random, Array
from typing import Optional, Literal

KernelType = Literal["rbf", "laplacian"]


class RandomFourierFeatures:
    """
    Random Fourier Features for kernel approximation in JAX.
    
    Parameters
    ----------
    n_features : int, default=100
        Number of random features
    gamma : float, default=1.0
        RBF kernel parameter
    kernel : str, default='rbf'
        Kernel type: 'rbf', 'laplacian'
    random_state : int, default=None
        Random seed for reproducibility
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
        self.omega_: Optional[Array] = None
        self.b_: Optional[Array] = None
        
    def fit(self, X: Array) -> "RandomFourierFeatures":
        """Sample random frequencies from the spectral density."""
        key = random.PRNGKey(self.random_state or 0)
        n_features_in = X.shape[1]
        key1, key2 = random.split(key)
        
        # Sample frequencies from spectral density
        if self.kernel == "rbf":
            # Gaussian RBF: spectral density is Gaussian
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
        
        # Sample phase offsets uniformly
        self.b_ = random.uniform(key2, (self.n_features,)) * 2 * jnp.pi
        
        return self
    
    def transform(self, X: Array) -> Array:
        """Compute random Fourier features."""
        if self.omega_ is None or self.b_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        projection = jnp.dot(X, self.omega_) + self.b_
        Z = jnp.sqrt(2.0 / self.n_features) * jnp.cos(projection)
        return Z
    
    def fit_transform(self, X: Array) -> Array:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class RFFRidgeRegression:
    """
    Ridge regression with Random Fourier Features in JAX.
    
    Parameters
    ----------
    n_features : int, default=100
        Number of random features
    gamma : float, default=1.0
        RBF kernel parameter
    alpha : float, default=1.0
        Regularization strength
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
        self.rff_: Optional[RandomFourierFeatures] = None
        self.weights_: Optional[Array] = None
        
    def fit(self, X: Array, y: Array) -> "RFFRidgeRegression":
        """Fit the model."""
        self.rff_ = RandomFourierFeatures(
            n_features=self.n_features, 
            gamma=self.gamma,
            random_state=self.random_state,
        )
        Z = self.rff_.fit_transform(X)
        
        # Solve regularized least squares in feature space
        A = jnp.dot(Z.T, Z) + self.alpha * jnp.eye(self.n_features)
        b = jnp.dot(Z.T, y)
        self.weights_ = jax.scipy.linalg.solve(A, b, assume_a="pos")
        
        return self
    
    def predict(self, X: Array) -> Array:
        """Make predictions."""
        if self.rff_ is None or self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        Z = self.rff_.transform(X)
        return jnp.dot(Z, self.weights_)
```

## Comparing Exact vs Approximate Kernels

Let's verify the approximation quality with JAX:

```python
import matplotlib.pyplot as plt
from jax import random

# Generate data
key = random.PRNGKey(42)
n = 500
d = 10
X = random.normal(key, (n, d))

# Exact RBF kernel
gamma = 0.5
sq_norm = jnp.sum(X**2, axis=1, keepdims=True)
sq_dist = sq_norm + sq_norm.T - 2 * jnp.dot(X, X.T)
K_exact = jnp.exp(-gamma * sq_dist)

# Approximate kernel with different D
D_values = [10, 50, 100, 500, 1000]
errors = []

for D in D_values:
    rff = RandomFourierFeatures(n_features=D, gamma=gamma, random_state=42)
    Z = rff.fit_transform(X)
    K_approx = jnp.dot(Z, Z.T)
    
    error = jnp.linalg.norm(K_exact - K_approx, 'fro') / jnp.linalg.norm(K_exact, 'fro')
    errors.append(float(error))
    print(f"D = {D:4d}: Relative error = {error:.4f}")

plt.figure(figsize=(8, 5))
plt.loglog(D_values, errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Random Features (D)')
plt.ylabel('Relative Frobenius Error')
plt.title('RFF Approximation Quality')
plt.grid(True, alpha=0.3)
plt.show()
```

## Example: Regression with RFF

```python
# Generate nonlinear regression data
key = random.PRNGKey(42)
X_train = jnp.sort(random.uniform(key, (200,), minval=0, maxval=2*jnp.pi)).reshape(-1, 1)
key, subkey = random.split(key)
y_train = jnp.sin(X_train).ravel() + 0.1 * random.normal(subkey, (200,))

X_test = jnp.linspace(0, 2*jnp.pi, 300).reshape(-1, 1)

# Compare different numbers of features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, D in zip(axes.flat, [10, 50, 200, 1000]):
    model = RFFRidgeRegression(n_features=D, gamma=1.0, alpha=0.001, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    ax.scatter(X_train, y_train, c='blue', alpha=0.3, s=20)
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='RFF prediction')
    ax.plot(X_test, jnp.sin(X_test), 'g--', linewidth=2, label='True function')
    ax.set_title(f'D = {D} features')
    ax.legend()

plt.tight_layout()
plt.show()
```

## Computational Comparison

| Method | Training | Prediction | Storage |
|--------|----------|------------|---------|
| Exact Kernel | $O(n^3)$ | $O(n)$ per sample | $O(n^2)$ |
| RFF | $O(nD^2)$ | $O(D)$ per sample | $O(nD)$ |

For $D \ll n$, RFF is dramatically faster!

```python
import time

n_samples = [100, 500, 1000, 2000, 5000]
D = 500

times_exact = []
times_rff = []

for n in n_samples:
    key = random.PRNGKey(42)
    X = random.normal(key, (n, 10))
    y = random.normal(key, (n,))
    
    # Exact kernel
    start = time.time()
    sq_norm = jnp.sum(X**2, axis=1, keepdims=True)
    sq_dist = sq_norm + sq_norm.T - 2 * jnp.dot(X, X.T)
    K = jnp.exp(-0.5 * sq_dist)
    alpha = jax.scipy.linalg.solve(K + 0.01 * jnp.eye(n), y)
    jax.block_until_ready(alpha)  # Wait for JAX async computation
    times_exact.append(time.time() - start)
    
    # RFF
    start = time.time()
    model = RFFRidgeRegression(n_features=D, gamma=0.5, alpha=0.01)
    model.fit(X, y)
    jax.block_until_ready(model.weights_)
    times_rff.append(time.time() - start)
    
    print(f"n = {n:5d}: Exact = {times_exact[-1]:.3f}s, RFF = {times_rff[-1]:.3f}s")

plt.figure(figsize=(8, 5))
plt.plot(n_samples, times_exact, 'bo-', label='Exact Kernel', linewidth=2)
plt.plot(n_samples, times_rff, 'ro-', label=f'RFF (D={D})', linewidth=2)
plt.xlabel('Number of Samples')
plt.ylabel('Training Time (seconds)')
plt.legend()
plt.title('Scalability Comparison')
plt.show()
```

## JAX Benefits for RFF

Using JAX for Random Fourier Features provides:
- **JIT Compilation**: Compile the transform function for faster repeated calls
- **Automatic Batching**: Use `vmap` for vectorized operations
- **GPU Acceleration**: Seamlessly run on GPUs for large datasets
- **Composability**: Easy to chain with other JAX transformations

## When to Use RFF

✅ **Use RFF when:**
- Dataset is large ($n > 10,000$)
- Shift-invariant kernels (RBF, Laplacian, Matérn)
- Online/streaming learning needed
- Memory is limited

❌ **Avoid RFF when:**
- Small datasets (exact kernel is fine)
- Non-shift-invariant kernels (polynomial)
- Very high accuracy required

## Conclusion

Random Fourier Features transform kernel methods from $O(n^3)$ to $O(nD^2)$, making them practical for large-scale learning. The key insights are:

1. Bochner's theorem connects kernels to Fourier transforms
2. Monte Carlo approximation gives explicit features
3. Standard linear methods can then be applied

The complete JAX implementation is available in our [examples folder](https://github.com/mlnomadpy/awesome-kernels/tree/main/examples).

---

*Next post: Implementing the Nyström approximation for data-dependent features!*
