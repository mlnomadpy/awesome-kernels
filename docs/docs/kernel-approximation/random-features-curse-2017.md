---
sidebar_position: 2
title: "Bach (2017) - Breaking the Curse of Dimensionality"
---

# Breaking the Curse of Dimensionality with Random Features

**Authors:** Francis R. Bach  
**Published:** 2017  
**Venue:** arXiv  
**Link:** [arXiv](https://arxiv.org/abs/1702.05803)

## Summary

This paper analyzes when random feature approximations can achieve optimal learning rates, "breaking" the curse of dimensionality. It shows that for certain kernel-target combinations, random features achieve minimax optimal rates independent of the ambient dimension.

## Key Contributions

### 1. Optimal Rates for Random Features

Under appropriate smoothness conditions, random features achieve:
$$\mathbb{E}\|\hat{f} - f_\rho\|_{L^2}^2 = O(n^{-\frac{2r}{2r+1}})$$

This rate is independent of input dimension $d$!

### 2. Sufficient Conditions

Optimal rates are achieved when:
1. Target function is in the RKHS: $f_\rho \in \mathcal{H}_K$
2. Kernel has sufficient spectral decay
3. Number of features $D = \tilde{O}(n)$

### 3. Comparison with Kernel Ridge Regression

| Method | Rate | Computation |
|--------|------|-------------|
| Full KRR | $O(n^{-\frac{2r}{2r+s}})$ | $O(n^3)$ |
| Random Features | $O(n^{-\frac{2r}{2r+1}})$ | $O(nD^2)$ |

Random features achieve rate with $s=1$ (polynomial eigenvalue decay).

## Mathematical Framework

### Random Feature Map

For shift-invariant kernel with spectral density $p(\omega)$:
$$z(x) = \frac{1}{\sqrt{D}}\begin{pmatrix} \cos(\omega_1^T x) \\ \sin(\omega_1^T x) \\ \vdots \\ \cos(\omega_D^T x) \\ \sin(\omega_D^T x) \end{pmatrix}$$

where $\omega_j \sim p(\omega)$.

### Approximation Kernel

$$\hat{K}(x,y) = z(x)^T z(y) = \frac{1}{D}\sum_{j=1}^D \cos(\omega_j^T(x-y))$$

By law of large numbers: $\mathbb{E}[\hat{K}(x,y)] = K(x,y)$

### Random Feature Ridge Regression

$$\hat{f} = \arg\min_{w \in \mathbb{R}^{2D}} \frac{1}{n}\sum_{i=1}^n (w^T z(x_i) - y_i)^2 + \lambda \|w\|^2$$

Closed form: $\hat{w} = (Z^T Z + n\lambda I)^{-1} Z^T y$

## Theoretical Analysis

### Error Decomposition

$$\|\hat{f} - f_\rho\|^2 \leq \underbrace{\|f_\lambda - f_\rho\|^2}_{\text{approximation}} + \underbrace{\|\hat{f} - f_\lambda\|^2}_{\text{estimation}}$$

### Approximation Error

For $f_\rho \in \mathcal{H}_K$ with $\|f_\rho\|_{\mathcal{H}} \leq R$:
$$\|f_\lambda - f_\rho\|_{L^2}^2 \leq \lambda R^2$$

### Estimation Error

With $D = \tilde{O}(n/\lambda)$ features:
$$\mathbb{E}\|\hat{f} - f_\lambda\|_{L^2}^2 \leq \frac{C}{n\lambda}$$

### Optimal Parameter Choice

Setting $\lambda = n^{-\frac{1}{2}}$ and $D = \tilde{O}(\sqrt{n})$:
$$\mathbb{E}\|\hat{f} - f_\rho\|_{L^2}^2 = O(n^{-1/2})$$

## Beyond Basic Random Features

### 1. Data-Dependent Features

Sample frequencies from empirical distribution:
$$\omega_j \sim \hat{p}(\omega) \propto \sum_i |\hat{f}(x_i)|^2 p(\omega | x_i)$$

### 2. Orthogonal Random Features

Use structured orthogonal matrices:
$$\Omega = \frac{1}{\sqrt{D}} S H G$$

where $H$ is Hadamard, $S$ is scaling, $G$ is random sign.

### 3. Quadrature Features

Use deterministic quadrature rules for integration:
$$K(x,y) \approx \sum_{j=1}^D w_j \cos(\omega_j^T(x-y))$$

## Algorithm

```python
import numpy as np

def random_fourier_features_regression(X, y, D, gamma, lambda_reg):
    """
    Random Fourier Features with ridge regression.
    
    Parameters:
    -----------
    X : training data (n, d)
    y : labels (n,)
    D : number of random features
    gamma : RBF kernel parameter
    lambda_reg : regularization
    
    Returns:
    --------
    omega, b, w : feature parameters and weights
    """
    n, d = X.shape
    
    # Sample random features (for RBF kernel)
    omega = np.random.randn(d, D) * np.sqrt(2 * gamma)
    b = np.random.uniform(0, 2 * np.pi, D)
    
    # Compute feature matrix
    Z = np.sqrt(2.0 / D) * np.cos(X @ omega + b)
    
    # Ridge regression
    w = np.linalg.solve(Z.T @ Z + n * lambda_reg * np.eye(D), Z.T @ y)
    
    return omega, b, w

def predict_rff(X_test, omega, b, w):
    """Predict with RFF model."""
    D = len(w)
    Z_test = np.sqrt(2.0 / D) * np.cos(X_test @ omega + b)
    return Z_test @ w
```

## Dimension-Independence

### When does it hold?

1. **Smooth kernels**: RBF, Matérn with sufficient smoothness
2. **Smooth targets**: $f_\rho \in \mathcal{H}_K$
3. **Low intrinsic dimension**: Data on low-dimensional manifold

### When does it fail?

1. **Non-smooth kernels**: Laplacian, polynomial
2. **Hard targets**: $f_\rho$ not in RKHS
3. **High-frequency components**: Oscillatory targets

## Impact on Machine Learning

This paper:
1. **Theoretical foundation**: Justified random features for large-scale learning
2. **Practical guidance**: Guidelines for choosing $D$
3. **Connection**: Linked random features to classical approximation theory
4. **Scalability**: Enabled kernel-quality predictions at linear cost

## Citation

```bibtex
@article{bach2017breaking,
  title={On the equivalence between kernel quadrature rules and random feature expansions},
  author={Bach, Francis},
  journal={Journal of Machine Learning Research},
  volume={18},
  number={21},
  pages={1--38},
  year={2017}
}
```

## Further Reading

- Rahimi, A. & Recht, B. (2007). Random features for large-scale kernel machines
- Rudi, A. & Rosasco, L. (2017). Generalization properties of learning with random features
- Li, Z., et al. (2019). Towards a unified analysis of random Fourier features
