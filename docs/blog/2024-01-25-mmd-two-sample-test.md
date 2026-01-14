---
slug: implementing-mmd-two-sample-test
title: Implementing the MMD Two-Sample Test in Python
authors: [mlnomadpy]
tags: [kernel, python, implementation, tutorial, rkhs]
---

# Implementing the MMD Two-Sample Test in Python

The Maximum Mean Discrepancy (MMD) is a powerful kernel-based statistic for comparing distributions. In this post, we'll implement the MMD two-sample test from scratch and explore its applications.

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

## Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

class MMDTest:
    """
    Maximum Mean Discrepancy Two-Sample Test.
    
    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel function or name
    gamma : float, default=None
        RBF kernel bandwidth. If None, uses median heuristic.
    """
    
    def __init__(self, kernel='rbf', gamma=None):
        self.kernel = kernel
        self.gamma = gamma
        
    def _compute_kernel(self, X, Y):
        """Compute kernel matrix."""
        if callable(self.kernel):
            return self.kernel(X, Y)
        elif self.kernel == 'rbf':
            sq_dist = cdist(X, Y, 'sqeuclidean')
            return np.exp(-self.gamma * sq_dist)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _median_heuristic(self, X, Y):
        """Set bandwidth using median heuristic."""
        XY = np.vstack([X, Y])
        dists = cdist(XY, XY, 'euclidean')
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(len(XY), k=1)
        median_dist = np.median(dists[triu_indices])
        # gamma = 1 / (2 * sigma^2), where sigma = median_dist
        return 1.0 / (2 * median_dist**2 + 1e-10)
    
    def compute_mmd_squared(self, X, Y, unbiased=True):
        """
        Compute MMD^2 statistic.
        
        Parameters
        ----------
        X : array of shape (n, d) - samples from P
        Y : array of shape (m, d) - samples from Q
        unbiased : bool - use unbiased estimator
        
        Returns
        -------
        mmd_squared : float
        """
        # Set bandwidth if not specified
        if self.gamma is None:
            self.gamma_ = self._median_heuristic(X, Y)
        else:
            self.gamma_ = self.gamma
            
        n, m = len(X), len(Y)
        
        K_XX = self._compute_kernel(X, X)
        K_YY = self._compute_kernel(Y, Y)
        K_XY = self._compute_kernel(X, Y)
        
        if unbiased:
            # Unbiased estimator (exclude diagonal)
            term_XX = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
            term_YY = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
            term_XY = K_XY.sum() / (n * m)
        else:
            # Biased estimator
            term_XX = K_XX.mean()
            term_YY = K_YY.mean()
            term_XY = K_XY.mean()
            
        return term_XX + term_YY - 2 * term_XY
    
    def permutation_test(self, X, Y, n_permutations=1000, alpha=0.05):
        """
        Perform MMD test with permutation.
        
        Parameters
        ----------
        X, Y : arrays - samples from two distributions
        n_permutations : int - number of permutations
        alpha : float - significance level
        
        Returns
        -------
        result : dict with test results
        """
        n, m = len(X), len(Y)
        
        # Observed statistic
        observed_mmd = self.compute_mmd_squared(X, Y)
        
        # Pool samples
        combined = np.vstack([X, Y])
        
        # Permutation distribution
        null_distribution = []
        for _ in range(n_permutations):
            perm = np.random.permutation(n + m)
            X_perm = combined[perm[:n]]
            Y_perm = combined[perm[n:]]
            null_distribution.append(self.compute_mmd_squared(X_perm, Y_perm))
        
        null_distribution = np.array(null_distribution)
        
        # Compute p-value (proportion of null >= observed)
        p_value = (np.sum(null_distribution >= observed_mmd) + 1) / (n_permutations + 1)
        
        # Critical value
        threshold = np.quantile(null_distribution, 1 - alpha)
        
        return {
            'statistic': observed_mmd,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'threshold': threshold,
            'null_distribution': null_distribution
        }
    
    def linear_time_mmd(self, X, Y):
        """
        Compute linear-time MMD estimator.
        O(n) instead of O(n^2).
        """
        n = min(len(X), len(Y))
        n = n - (n % 2)  # Make even
        
        X, Y = X[:n], Y[:n]
        
        if self.gamma is None:
            self.gamma_ = self._median_heuristic(X, Y)
        else:
            self.gamma_ = self.gamma
        
        h_values = []
        for i in range(0, n, 2):
            x1, x2 = X[i:i+1], X[i+1:i+2]
            y1, y2 = Y[i:i+1], Y[i+1:i+2]
            
            k_xx = self._compute_kernel(x1, x2)[0, 0]
            k_yy = self._compute_kernel(y1, y2)[0, 0]
            k_xy1 = self._compute_kernel(x1, y2)[0, 0]
            k_xy2 = self._compute_kernel(x2, y1)[0, 0]
            
            h = k_xx + k_yy - k_xy1 - k_xy2
            h_values.append(h)
        
        return np.mean(h_values)
```

## Example: Testing Identical Distributions

Under $H_0$, the MMD should be close to zero:

```python
import matplotlib.pyplot as plt

np.random.seed(42)

# Same distribution (H0 is true)
n, m = 200, 200
X = np.random.randn(n, 2)
Y = np.random.randn(m, 2)

mmd_test = MMDTest()
result = mmd_test.permutation_test(X, Y)

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
np.random.seed(42)

X = np.random.randn(200, 2)
Y = np.random.randn(200, 2) + np.array([1.0, 0.5])  # Shifted mean

mmd_test = MMDTest()
result = mmd_test.permutation_test(X, Y)

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
    
    for _ in range(n_trials):
        X = np.random.randn(n_samples, 2)
        Y = np.random.randn(n_samples, 2) + dist_shift
        
        mmd_test = MMDTest()
        result = mmd_test.permutation_test(X, Y, n_permutations=200)
        
        if result['reject_null']:
            rejections += 1
    
    return rejections / n_trials

# Power vs shift
shifts = np.linspace(0, 1.5, 10)
powers = [compute_power(np.array([s, 0]), 100, n_trials=50) for s in shifts]

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

## Kernel Selection

The median heuristic is a good default, but optimal bandwidth depends on the problem:

```python
def optimize_kernel_bandwidth(X, Y, gammas):
    """Find bandwidth maximizing test power."""
    best_power = 0
    best_gamma = None
    
    for gamma in gammas:
        mmd_test = MMDTest(gamma=gamma)
        result = mmd_test.permutation_test(X, Y, n_permutations=100)
        
        # Use statistic/threshold ratio as proxy for power
        power_proxy = result['statistic'] / (result['threshold'] + 1e-10)
        
        if power_proxy > best_power:
            best_power = power_proxy
            best_gamma = gamma
    
    return best_gamma, best_power
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

---

*Next: Implementing kernel mean embeddings for conditional distributions!*
