---
slug: implementing-random-fourier-features
title: Random Fourier Features - Scalable Kernel Approximation
authors: [mlnomadpy]
tags: [kernel, python, implementation, tutorial]
---

# Random Fourier Features: Making Kernel Methods Scalable

Random Fourier Features (RFF) is a beautiful technique that approximates kernel functions with explicit feature maps, enabling linear-time training while maintaining kernel-like performance. Let's implement it from scratch!

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

## Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

class RandomFourierFeatures:
    """
    Random Fourier Features for kernel approximation.
    
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
    
    def __init__(self, n_features=100, gamma=1.0, 
                 kernel='rbf', random_state=None):
        self.n_features = n_features
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        
    def fit(self, X):
        """
        Sample random frequencies.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features_in)
        """
        rng = np.random.RandomState(self.random_state)
        n_features_in = X.shape[1]
        
        # Sample frequencies from spectral density
        if self.kernel == 'rbf':
            # Gaussian RBF: spectral density is Gaussian
            self.omega_ = rng.randn(n_features_in, self.n_features) * np.sqrt(2 * self.gamma)
        elif self.kernel == 'laplacian':
            # Laplacian kernel: spectral density is Cauchy
            self.omega_ = rng.standard_cauchy((n_features_in, self.n_features)) * self.gamma
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # Sample phase offsets uniformly
        self.b_ = rng.uniform(0, 2 * np.pi, self.n_features)
        
        return self
    
    def transform(self, X):
        """
        Compute random Fourier features.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features_in)
        
        Returns
        -------
        Z : array of shape (n_samples, n_features)
        """
        # Project data
        projection = X @ self.omega_ + self.b_
        
        # Apply cosine and scale
        Z = np.sqrt(2.0 / self.n_features) * np.cos(projection)
        
        return Z
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class RFFRidgeRegression:
    """
    Ridge regression with Random Fourier Features.
    
    Parameters
    ----------
    n_features : int, default=100
        Number of random features
    gamma : float, default=1.0
        RBF kernel parameter
    alpha : float, default=1.0
        Regularization strength
    """
    
    def __init__(self, n_features=100, gamma=1.0, alpha=1.0):
        self.n_features = n_features
        self.gamma = gamma
        self.alpha = alpha
        
    def fit(self, X, y):
        """Fit the model."""
        # Create and fit random features
        self.rff_ = RandomFourierFeatures(
            n_features=self.n_features, 
            gamma=self.gamma
        )
        Z = self.rff_.fit_transform(X)
        
        # Solve regularized least squares in feature space
        A = Z.T @ Z + self.alpha * np.eye(self.n_features)
        b = Z.T @ y
        self.weights_ = np.linalg.solve(A, b)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        Z = self.rff_.transform(X)
        return Z @ self.weights_
```

## Comparing Exact vs Approximate Kernels

Let's verify the approximation quality:

```python
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n = 500
d = 10
X = np.random.randn(n, d)

# Exact RBF kernel
gamma = 0.5
K_exact = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))

# Approximate kernel with different D
D_values = [10, 50, 100, 500, 1000]
errors = []

for D in D_values:
    rff = RandomFourierFeatures(n_features=D, gamma=gamma, random_state=42)
    Z = rff.fit_transform(X)
    K_approx = Z @ Z.T
    
    error = np.linalg.norm(K_exact - K_approx, 'fro') / np.linalg.norm(K_exact, 'fro')
    errors.append(error)
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
np.random.seed(42)
X_train = np.sort(np.random.uniform(0, 2*np.pi, 200)).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(200)

X_test = np.linspace(0, 2*np.pi, 300).reshape(-1, 1)

# Compare different numbers of features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, D in zip(axes.flat, [10, 50, 200, 1000]):
    model = RFFRidgeRegression(n_features=D, gamma=1.0, alpha=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    ax.scatter(X_train, y_train, c='blue', alpha=0.3, s=20)
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='RFF prediction')
    ax.plot(X_test, np.sin(X_test), 'g--', linewidth=2, label='True function')
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
    X = np.random.randn(n, 10)
    y = np.random.randn(n)
    
    # Exact kernel
    start = time.time()
    K = np.exp(-0.5 * cdist(X, X, 'sqeuclidean'))
    alpha = np.linalg.solve(K + 0.01 * np.eye(n), y)
    times_exact.append(time.time() - start)
    
    # RFF
    start = time.time()
    model = RFFRidgeRegression(n_features=D, gamma=0.5, alpha=0.01)
    model.fit(X, y)
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

## Orthogonal Random Features

We can improve RFF with orthogonal random features:

```python
def orthogonal_random_features(X, n_features, gamma, random_state=None):
    """
    Orthogonal Random Features for better approximation.
    """
    rng = np.random.RandomState(random_state)
    n_features_in = X.shape[1]
    
    # Sample more frequencies than needed
    n_samples_omega = max(n_features_in, n_features)
    G = rng.randn(n_samples_omega, n_samples_omega)
    
    # Orthogonalize via QR decomposition
    Q, _ = np.linalg.qr(G)
    
    # Scale and select
    S = np.sqrt(np.random.chisquare(n_samples_omega, n_samples_omega))
    omega = (Q * S)[:n_features_in, :n_features]
    omega *= np.sqrt(2 * gamma)
    
    # Phase offsets
    b = rng.uniform(0, 2 * np.pi, n_features)
    
    # Compute features
    projection = X @ omega + b
    Z = np.sqrt(2.0 / n_features) * np.cos(projection)
    
    return Z
```

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

---

*Next post: Implementing the Nyström approximation for data-dependent features!*
