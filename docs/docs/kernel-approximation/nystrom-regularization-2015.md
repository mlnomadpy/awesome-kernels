---
sidebar_position: 1
title: "Rudi et al. (2015) - Nyström Computational Regularization"
---

# Less Is More: Nyström Computational Regularization

**Authors:** Alessandro Rudi, Raffaello Camoriano, Lorenzo Rosasco  
**Published:** 2015  
**Venue:** NeurIPS  
**Link:** [arXiv](https://arxiv.org/abs/1507.04717)

## Summary

This paper provides a rigorous analysis of Nyström approximation for kernel ridge regression, showing that subsampling acts as implicit regularization. The key insight is that using fewer landmark points can improve generalization, not just computation.

## Key Contributions

### 1. Computational Regularization

The Nyström method with $m$ landmark points introduces implicit regularization equivalent to explicit regularization with:
$$\lambda_{eff} \approx \frac{n}{m}$$

This means fewer points = stronger regularization.

### 2. Optimal Subsampling Rate

For optimal generalization with source condition $r$ and capacity $s$:
$$m = O(n^{\frac{s}{2r+s}})$$

This is fewer than $n$ for smooth targets ($r > 0$).

### 3. Minimax Optimal Rates

The Nyström estimator achieves the same minimax optimal rates as full kernel ridge regression:
$$\mathbb{E}\|f_m - f_\rho\|_{L^2}^2 = O(n^{-\frac{2r}{2r+s}})$$

while computing in $O(nm^2 + m^3)$ instead of $O(n^3)$.

## Mathematical Framework

### Nyström Approximation

Given $n$ training points, select $m$ landmark points. The approximate kernel matrix:
$$\tilde{K} = K_{nm} K_{mm}^{-1} K_{nm}^T$$

where $K_{nm}$ is $n \times m$ and $K_{mm}$ is $m \times m$.

### Nyström Estimator

$$\tilde{f} = \sum_{j=1}^m \tilde{\alpha}_j K(\cdot, x_j)$$

where:
$$\tilde{\alpha} = (K_{mm} + \frac{n\lambda}{m} K_{mm})^{-1} \frac{1}{n} K_{nm}^T y$$

### Equivalent Ridge Problem

The Nyström solution solves a modified ridge regression:
$$\tilde{f} = \arg\min_{f \in \text{span}\{K(\cdot, x_j)\}_{j=1}^m} \frac{1}{n}\sum_i (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}}^2$$

## Theoretical Analysis

### Bias-Variance Decomposition

$$\|\tilde{f} - f_\rho\|^2 \leq \underbrace{\|f_\lambda - f_\rho\|^2}_{\text{explicit reg.}} + \underbrace{\|\tilde{f} - f_\lambda\|^2}_{\text{statistical}} + \underbrace{\|P_m f_\lambda - f_\lambda\|^2}_{\text{computational}}$$

### Computational Error Bound

The projection error onto the Nyström subspace:
$$\|P_m f - f\|^2 \leq \frac{\mathcal{N}(\lambda) \kappa^2}{m\lambda}$$

### Main Theorem

**Theorem:** Under source condition $r \leq 1$, capacity condition $s$, with:
- $\lambda = n^{-\frac{1}{2r+s}}$  
- $m = n^{\frac{s}{2r+s}}$

we have:
$$\mathbb{E}\|\tilde{f} - f_\rho\|_{L^2}^2 = O(n^{-\frac{2r}{2r+s}})$$

## Algorithm

```python
import numpy as np

def nystrom_krr(X, y, X_landmarks, lambda_reg, kernel_func):
    """
    Kernel Ridge Regression with Nyström approximation.
    
    Parameters:
    -----------
    X : training data (n, d)
    y : labels (n,)
    X_landmarks : landmark points (m, d)
    lambda_reg : regularization parameter
    kernel_func : kernel function
    
    Returns:
    --------
    alpha : coefficients for landmarks
    """
    n, m = len(X), len(X_landmarks)
    
    # Compute kernel matrices
    K_nm = kernel_func(X, X_landmarks)  # (n, m)
    K_mm = kernel_func(X_landmarks, X_landmarks)  # (m, m)
    
    # Add regularization to K_mm
    K_mm_reg = K_mm + (n * lambda_reg / m) * np.eye(m)
    
    # Solve for coefficients
    rhs = K_nm.T @ y / n
    alpha = np.linalg.solve(K_mm_reg, rhs)
    
    return alpha

def predict(X_test, X_landmarks, alpha, kernel_func):
    """Make predictions with Nyström model."""
    K_test = kernel_func(X_test, X_landmarks)
    return K_test @ alpha
```

## Sampling Strategies

### 1. Uniform Sampling
Simple random selection. Works well when data is uniform.

### 2. Leverage Score Sampling
Sample proportional to:
$$p_i \propto K_{ii} - K_i^T (K + \lambda I)^{-1} K_i$$

Better for non-uniform data distributions.

### 3. K-means Clustering
Use cluster centers as landmarks. Captures data structure.

## Complexity Comparison

| Method | Training | Prediction | Memory |
|--------|----------|------------|--------|
| Full KRR | $O(n^3)$ | $O(n)$ | $O(n^2)$ |
| Nyström | $O(nm^2 + m^3)$ | $O(m)$ | $O(nm)$ |
| Random Features | $O(nD^2)$ | $O(D)$ | $O(nD)$ |

For $m \ll n$, Nyström is much faster.

## Practical Implications

1. **Don't use all data**: Subsampling can improve generalization
2. **Adaptive $m$**: Choose $m$ based on problem difficulty
3. **Regularization tuning**: Less critical when using Nyström
4. **Scalability**: Enables kernel methods on large datasets

## Citation

```bibtex
@inproceedings{rudi2015less,
  title={Less is more: Nystr{\"o}m computational regularization},
  author={Rudi, Alessandro and Camoriano, Raffaello and Rosasco, Lorenzo},
  booktitle={Advances in Neural Information Processing Systems},
  volume={28},
  year={2015}
}
```

## Further Reading

- Williams, C. & Seeger, M. (2000). Using the Nyström method to speed up kernel machines
- Drineas, P. & Mahoney, M. (2005). On the Nyström method for approximating a Gram matrix
- Alaoui, A. & Mahoney, M. (2015). Fast randomized kernel ridge regression
