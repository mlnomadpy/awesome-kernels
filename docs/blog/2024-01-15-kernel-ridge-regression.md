---
slug: implementing-kernel-ridge-regression
title: Implementing Kernel Ridge Regression from Scratch
authors: [mlnomadpy]
tags: [kernel, python, implementation, tutorial]
---

# Implementing Kernel Ridge Regression from Scratch in Python

Kernel Ridge Regression (KRR) is one of the most elegant kernel methods, combining the simplicity of ridge regression with the power of the kernel trick. In this post, we'll implement KRR from scratch and understand each component.

<!--truncate-->

## The Problem

Given training data $(x_1, y_1), \ldots, (x_n, y_n)$, we want to find a function $f$ that minimizes:

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$$

where $\mathcal{H}$ is a Reproducing Kernel Hilbert Space.

## The Representer Theorem

Thanks to the representer theorem, we know the solution has the form:

$$f^*(x) = \sum_{i=1}^n \alpha_i K(x_i, x)$$

This transforms our infinite-dimensional optimization into a finite-dimensional one!

## Deriving the Solution

Substituting into the objective:

$$L(\alpha) = \frac{1}{n}\|y - K\alpha\|^2 + \lambda \alpha^T K \alpha$$

Taking the derivative and setting to zero:

$$\frac{\partial L}{\partial \alpha} = -\frac{2}{n}K(y - K\alpha) + 2\lambda K\alpha = 0$$

Solving (assuming $K$ is invertible or using regularization):

$$\alpha = (K + n\lambda I)^{-1}y$$

## Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

class KernelRidgeRegression:
    """
    Kernel Ridge Regression implementation.
    
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: 'rbf', 'linear', 'polynomial'
    gamma : float, default=1.0
        RBF kernel parameter
    degree : int, default=3
        Polynomial kernel degree
    coef0 : float, default=1.0
        Polynomial kernel coefficient
    alpha : float, default=1.0
        Regularization strength
    """
    
    def __init__(self, kernel='rbf', gamma=1.0, degree=3, 
                 coef0=1.0, alpha=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        
    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix between X1 and X2."""
        if self.kernel == 'rbf':
            sq_dist = cdist(X1, X2, 'sqeuclidean')
            return np.exp(-self.gamma * sq_dist)
        elif self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'polynomial':
            return (self.gamma * X1 @ X2.T + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)
        """
        self.X_train_ = X
        n = len(X)
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Solve (K + n*alpha*I)^{-1} y
        self.dual_coef_ = np.linalg.solve(
            K + n * self.alpha * np.eye(n), 
            y
        )
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
        """
        K = self._compute_kernel(X, self.X_train_)
        return K @ self.dual_coef_
```

## Example: Fitting a Nonlinear Function

```python
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# Fit model
krr = KernelRidgeRegression(kernel='rbf', gamma=1.0, alpha=0.01)
krr.fit(X, y)

# Predict
X_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
y_pred = krr.predict(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='blue', alpha=0.5, label='Training data')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='KRR prediction')
plt.plot(X_test, np.sin(X_test), 'g--', linewidth=2, label='True function')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kernel Ridge Regression')
plt.show()
```

## Understanding the Kernel Parameter

The kernel width $\gamma$ controls the flexibility:
- **Large $\gamma$**: Local influence, can fit complex patterns (may overfit)
- **Small $\gamma$**: Global influence, smoother predictions (may underfit)

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, gamma in zip(axes, [0.1, 1.0, 10.0]):
    krr = KernelRidgeRegression(kernel='rbf', gamma=gamma, alpha=0.01)
    krr.fit(X, y)
    y_pred = krr.predict(X_test)
    
    ax.scatter(X, y, c='blue', alpha=0.3)
    ax.plot(X_test, y_pred, 'r-', linewidth=2)
    ax.set_title(f'γ = {gamma}')
    
plt.tight_layout()
plt.show()
```

## Connection to Gaussian Processes

KRR is equivalent to the posterior mean of a Gaussian Process with:
- Prior: $f \sim \mathcal{GP}(0, K/\lambda)$
- Likelihood: $y | f \sim \mathcal{N}(f(X), \sigma^2 I)$

where $\sigma^2 = n\lambda$.

## Computational Considerations

The main bottleneck is solving the $n \times n$ linear system:
- **Time complexity**: $O(n^3)$
- **Space complexity**: $O(n^2)$

For large datasets, consider:
1. **Nyström approximation**: Use subset of data
2. **Random Fourier Features**: Explicit feature map
3. **Iterative solvers**: Conjugate gradient

## Next Steps

In upcoming posts, we'll implement:
- Random Fourier Features for scalable KRR
- Kernel PCA for dimensionality reduction
- MMD for two-sample testing

## Full Code

The complete implementation is available in our [GitHub repository](https://github.com/mlnomadpy/awesome-kernels).

---

*Have questions? Open an issue on GitHub or join the discussion!*
