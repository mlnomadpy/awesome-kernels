---
slug: implementing-mmd-two-sample-test
title: Implementing the MMD Two-Sample Test in JAX
authors: [mlnomadpy]
tags: [kernel, jax, python, implementation, tutorial, rkhs]
---

# Implementing the MMD Two-Sample Test in JAX

The Maximum Mean Discrepancy (MMD) is a powerful kernel-based statistic for comparing distributions. In this post, we'll implement the MMD two-sample test from scratch using JAX and explore its applications.

<!--truncate-->

## The Two-Sample Testing Problem

Given samples:
- $X = \{x_1, \ldots, x_n\} \sim P$
- $Y = \{y_1, \ldots, y_m\} \sim Q$

We want to test:
- $H_0: P = Q$ (distributions are the same)
- $H_1: P \neq Q$ (distributions differ)

## Maximum Mean Discrepancy

The MMD measures the distance between mean embeddings:

$$\text{MMD}^2[P, Q] = \|\mu_P - \mu_Q\|_{\mathcal{H}}^2$$

Expanding:

$$\text{MMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]$$

where $x, x' \sim P$ and $y, y' \sim Q$.

## Empirical MMD Estimators

### Biased Estimator (V-statistic)

$$\widehat{\text{MMD}}_b^2 = \frac{1}{n^2}\sum_{i,j} k(x_i, x_j) + \frac{1}{m^2}\sum_{i,j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)$$

### Unbiased Estimator (U-statistic)

$$\widehat{\text{MMD}}_u^2 = \frac{1}{n(n-1)}\sum_{i \neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i \neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)$$

### Linear-Time Estimator

$$\widehat{\text{MMD}}_l^2 = \frac{2}{n}\sum_{i=1}^{n/2}[k(x_{2i-1}, x_{2i}) + k(y_{2i-1}, y_{2i}) - k(x_{2i-1}, y_{2i}) - k(x_{2i}, y_{2i-1})]$$

## JAX Implementation

```python
import jax
import jax.numpy as jnp
from jax import random, Array
from typing import Optional, Dict, Any


def rbf_kernel(X: Array, Y: Array, gamma: float) -> Array:
    """Compute RBF kernel matrix."""
    sq_norm_X = jnp.sum(X**2, axis=1, keepdims=True)
    sq_norm_Y = jnp.sum(Y**2, axis=1, keepdims=True)
    sq_dist = sq_norm_X + sq_norm_Y.T - 2 * jnp.dot(X, Y.T)
    return jnp.exp(-gamma * sq_dist)


def median_heuristic(X: Array, Y: Array) -> float:
    """Compute kernel bandwidth using the median heuristic."""
    XY = jnp.vstack([X, Y])
    n = len(XY)
    sq_norm = jnp.sum(XY**2, axis=1, keepdims=True)
    sq_dist = sq_norm + sq_norm.T - 2 * jnp.dot(XY, XY.T)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    dists = jnp.sqrt(jnp.maximum(sq_dist[mask], 0.0))
    median_dist = jnp.median(dists)
    return 1.0 / (2 * median_dist**2 + 1e-10)


class MMDTest:
    """
    Maximum Mean Discrepancy Two-Sample Test in JAX.
    
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel function name
    gamma : float, default=None
        RBF kernel bandwidth. If None, uses median heuristic.
    """
    
    def __init__(self, kernel: str = 'rbf', gamma: Optional[float] = None):
        self.kernel = kernel
        self.gamma = gamma
        self.gamma_: Optional[float] = None
        
    def _compute_kernel(self, X: Array, Y: Array) -> Array:
        """Compute kernel matrix."""
        if self.kernel == 'rbf':
            return rbf_kernel(X, Y, self.gamma_)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def compute_mmd_squared(
        self, X: Array, Y: Array, unbiased: bool = True
    ) -> float:
        """Compute MMD^2 statistic."""
        if self.gamma is None:
            self.gamma_ = median_heuristic(X, Y)
        else:
            self.gamma_ = self.gamma
            
        n, m = len(X), len(Y)
        
        K_XX = self._compute_kernel(X, X)
        K_YY = self._compute_kernel(Y, Y)
        K_XY = self._compute_kernel(X, Y)
        
        if unbiased:
            term_XX = (jnp.sum(K_XX) - jnp.trace(K_XX)) / (n * (n - 1))
            term_YY = (jnp.sum(K_YY) - jnp.trace(K_YY)) / (m * (m - 1))
            term_XY = jnp.sum(K_XY) / (n * m)
        else:
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
        """Perform MMD permutation test."""
        n, m = len(X), len(Y)
        observed_mmd = self.compute_mmd_squared(X, Y)
        combined = jnp.vstack([X, Y])
        
        key = random.PRNGKey(random_state or 0)
        null_distribution = []
        
        for i in range(n_permutations):
            key, subkey = random.split(key)
            perm = random.permutation(subkey, n + m)
            X_perm = combined[perm[:n]]
            Y_perm = combined[perm[n:]]
            null_distribution.append(self.compute_mmd_squared(X_perm, Y_perm))
        
        null_distribution = jnp.array(null_distribution)
        p_value = float(
            (jnp.sum(null_distribution >= observed_mmd) + 1) / (n_permutations + 1)
        )
        threshold = float(jnp.quantile(null_distribution, 1 - alpha))
        
        return {
            'statistic': observed_mmd,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'threshold': threshold,
            'null_distribution': null_distribution,
        }
```

## Example: Testing Identical Distributions

Under $H_0$, the MMD should be close to zero:

```python
import matplotlib.pyplot as plt
from jax import random

key = random.PRNGKey(42)
key1, key2 = random.split(key)

# Same distribution (H0 is true)
n, m = 200, 200
X = random.normal(key1, (n, 2))
Y = random.normal(key2, (m, 2))

mmd_test = MMDTest()
result = mmd_test.permutation_test(X, Y, random_state=42)

print(f"MMD² = {result['statistic']:.6f}")
print(f"p-value = {result['p_value']:.4f}")
print(f"Reject H₀: {result['reject_null']}")

# Plot null distribution
plt.figure(figsize=(10, 5))
plt.hist(result['null_distribution'], bins=50, density=True, alpha=0.7, label='Null distribution')
plt.axvline(result['statistic'], color='red', linestyle='--', linewidth=2, label=f'Observed MMD² = {result["statistic"]:.4f}')
plt.axvline(result['threshold'], color='green', linestyle=':', linewidth=2, label=f'Threshold (α=0.05)')
plt.xlabel('MMD²')
plt.ylabel('Density')
plt.title('MMD Test: Same Distribution (H₀ true)')
plt.legend()
plt.show()
```

## Example: Testing Different Distributions

```python
# Different distributions (H1 is true)
key = random.PRNGKey(42)
key1, key2 = random.split(key)

X = random.normal(key1, (200, 2))
Y = random.normal(key2, (200, 2)) + jnp.array([1.0, 0.5])  # Shifted mean

mmd_test = MMDTest()
result = mmd_test.permutation_test(X, Y, random_state=42)

print(f"MMD² = {result['statistic']:.6f}")
print(f"p-value = {result['p_value']:.4f}")
print(f"Reject H₀: {result['reject_null']}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5, label='P')
axes[0].scatter(Y[:, 0], Y[:, 1], alpha=0.5, label='Q')
axes[0].legend()
axes[0].set_title('Samples from P and Q')
axes[0].set_xlabel('x₁')
axes[0].set_ylabel('x₂')

# Null distribution
axes[1].hist(result['null_distribution'], bins=50, density=True, alpha=0.7, label='Null distribution')
axes[1].axvline(result['statistic'], color='red', linestyle='--', linewidth=2, label=f'Observed MMD²')
axes[1].axvline(result['threshold'], color='green', linestyle=':', linewidth=2, label='Threshold')
axes[1].set_xlabel('MMD²')
axes[1].set_ylabel('Density')
axes[1].set_title('MMD Test: Different Distributions (H₁ true)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Power Analysis

Test power increases with:
1. Sample size
2. Distribution separation
3. Appropriate kernel bandwidth

```python
def compute_power(dist_shift, n_samples, n_trials=100):
    """Estimate test power for given shift and sample size."""
    rejections = 0
    
    for trial in range(n_trials):
        key = random.PRNGKey(trial)
        key1, key2 = random.split(key)
        
        X = random.normal(key1, (n_samples, 2))
        Y = random.normal(key2, (n_samples, 2)) + dist_shift
        
        mmd_test = MMDTest()
        result = mmd_test.permutation_test(X, Y, n_permutations=200, random_state=trial)
        
        if result['reject_null']:
            rejections += 1
    
    return rejections / n_trials

# Power vs shift
shifts = jnp.linspace(0, 1.5, 10)
powers = [compute_power(jnp.array([s, 0]), 100, n_trials=50) for s in shifts]

plt.figure(figsize=(8, 5))
plt.plot(shifts, powers, 'bo-', linewidth=2, markersize=8)
plt.axhline(0.05, color='gray', linestyle='--', label='α = 0.05')
plt.xlabel('Distribution Shift')
plt.ylabel('Power (rejection rate)')
plt.title('MMD Test Power vs Distribution Shift')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## JAX Benefits for MMD

Using JAX for MMD computation provides:
- **JIT Compilation**: Speed up permutation tests with `jax.jit`
- **Parallelization**: Use `jax.vmap` to vectorize over permutations
- **GPU Acceleration**: Compute kernel matrices on GPU for large samples
- **Automatic Differentiation**: Optimize kernel parameters if needed

## Applications

### 1. GAN Evaluation
Compare generated samples to real data:

```python
def evaluate_gan(real_samples, generated_samples):
    """Evaluate GAN using MMD."""
    mmd_test = MMDTest()
    result = mmd_test.permutation_test(real_samples, generated_samples)
    print(f"MMD² = {result['statistic']:.6f}")
    print(f"p-value = {result['p_value']:.4f}")
    return result
```

### 2. Covariate Shift Detection

```python
def detect_covariate_shift(train_features, test_features):
    """Detect if train and test distributions differ."""
    mmd_test = MMDTest()
    result = mmd_test.permutation_test(train_features, test_features)
    
    if result['reject_null']:
        print("⚠️ Covariate shift detected!")
    else:
        print("✓ No significant distribution shift")
    return result
```

### 3. Feature Selection

```python
def feature_importance_mmd(X, Y):
    """Rank features by their contribution to MMD."""
    importances = []
    
    for i in range(X.shape[1]):
        mmd_test = MMDTest()
        mmd_i = mmd_test.compute_mmd_squared(X[:, i:i+1], Y[:, i:i+1])
        importances.append((i, mmd_i))
    
    return sorted(importances, key=lambda x: x[1], reverse=True)
```

## Conclusion

The MMD two-sample test is:
- **Powerful**: Can detect any difference between distributions (with characteristic kernel)
- **Nonparametric**: No distributional assumptions
- **Interpretable**: MMD has clear geometric meaning

Key takeaways:
1. Use unbiased estimator for hypothesis testing
2. Median heuristic is a good default for bandwidth
3. Permutation test controls Type I error correctly

The complete JAX implementation is available in our [examples folder](https://github.com/mlnomadpy/awesome-kernels/tree/main/examples).

---

*Next: Implementing kernel mean embeddings for conditional distributions!*
