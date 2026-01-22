---
sidebar_position: 3
title: "Avron et al. (2017) - Subsampling via Regularized Leverage Scores"
---

# Subsampling for Ridge Regression via Regularized Leverage Scores

**Authors:** Haim Avron, Kenneth L. Clarkson, David P. Woodruff  
**Published:** 2017  
**Venue:** ICML / Journal of Machine Learning Research  
**Link:** [arXiv](https://arxiv.org/abs/1803.05049)

## Summary

This paper introduces regularized leverage score sampling for kernel ridge regression. It shows that sampling training points proportionally to their leverage scores yields estimators with optimal statistical and computational properties.

## Key Contributions

### 1. Regularized Leverage Scores

The regularized leverage score of point $i$ is:
$$\ell_i^\lambda = K_i^T (K + \lambda I)^{-1} K_i = \sum_{j} \frac{\lambda_j}{\lambda_j + \lambda} \phi_j(x_i)^2$$

where $K_i$ is the $i$-th column of the kernel matrix.

### 2. Statistical Leverage Sampling

Sample $m$ points with probabilities:
$$p_i = \frac{\ell_i^\lambda}{\sum_j \ell_j^\lambda} = \frac{\ell_i^\lambda}{\mathcal{N}(\lambda)}$$

where $\mathcal{N}(\lambda) = \text{Tr}(K(K+\lambda I)^{-1})$ is the effective dimension.

### 3. Optimal Subsampling Rate

With leverage score sampling, only need:
$$m = O(\mathcal{N}(\lambda) \log \mathcal{N}(\lambda))$$

samples to achieve nearly the same error as full kernel ridge regression.

## Mathematical Framework

### Ridge Regression

Full kernel ridge regression:
$$\hat{f} = \arg\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}}^2$$

Solution: $\hat{f}(x) = K(x, X)(K + n\lambda I)^{-1} y$

### Subsampled Estimator

Given indices $S = \{i_1, \ldots, i_m\}$ sampled with weights $\{p_i\}$:
$$\tilde{f} = \arg\min_{f \in \mathcal{H}} \sum_{j \in S} \frac{1}{m p_j}(f(x_j) - y_j)^2 + \lambda \|f\|_{\mathcal{H}}^2$$

### Importance Weighting

The reweighting $1/(mp_j)$ ensures unbiasedness:
$$\mathbb{E}\left[\sum_{j \in S} \frac{1}{mp_j}(f(x_j) - y_j)^2\right] = \sum_{i=1}^n (f(x_i) - y_i)^2$$

## Theoretical Guarantees

### Main Theorem

**Theorem:** With leverage score sampling using $m = O(\mathcal{N}(\lambda)/\epsilon^2)$ samples:
$$\|\tilde{f} - \hat{f}\|_{L^2}^2 \leq \epsilon \|\hat{f} - f_\rho\|_{L^2}^2$$

with high probability.

### Matrix Approximation View

The leverage score sampling preserves spectral properties:
$$\|K - \tilde{K}\|_2 \leq \epsilon \lambda$$

where $\tilde{K}$ is the subsampled approximation.

### Statistical Efficiency

The subsampled estimator achieves:
$$\mathbb{E}\|\tilde{f} - f_\rho\|_{L^2}^2 \leq (1 + \epsilon) \cdot \mathbb{E}\|\hat{f} - f_\rho\|_{L^2}^2$$

## Algorithm

```python
import numpy as np

def compute_leverage_scores(K, lambda_reg):
    """
    Compute regularized leverage scores.
    
    Parameters:
    -----------
    K : kernel matrix (n, n)
    lambda_reg : regularization parameter
    
    Returns:
    --------
    scores : leverage scores (n,)
    """
    n = K.shape[0]
    # Compute (K + λI)^{-1}
    K_reg_inv = np.linalg.inv(K + lambda_reg * np.eye(n))
    # Leverage score i = K[i,:] @ K_reg_inv @ K[:,i]
    scores = np.sum(K * K_reg_inv, axis=1)
    return scores

def approximate_leverage_scores(X, kernel_func, lambda_reg, n_samples=100):
    """
    Approximate leverage scores using random projection.
    """
    n = len(X)
    # Use random Nyström approximation
    idx = np.random.choice(n, min(n_samples, n), replace=False)
    K_nm = kernel_func(X, X[idx])
    K_mm = kernel_func(X[idx], X[idx])
    
    # Approximate leverage scores
    L = np.linalg.cholesky(K_mm + 1e-6 * np.eye(len(idx)))
    Z = np.linalg.solve(L, K_nm.T).T  # n x m
    scores = np.sum(Z**2, axis=1) / (lambda_reg + np.sum(Z**2, axis=1) / n_samples)
    return scores

def leverage_score_sampling(X, y, kernel_func, lambda_reg, m):
    """
    Kernel ridge regression with leverage score sampling.
    """
    n = len(X)
    K = kernel_func(X, X)
    
    # Compute leverage scores
    scores = compute_leverage_scores(K, lambda_reg)
    probs = scores / scores.sum()
    
    # Sample m points
    idx = np.random.choice(n, m, replace=False, p=probs)
    weights = 1.0 / (m * probs[idx])
    
    # Weighted kernel ridge regression on subsample
    K_sub = K[np.ix_(idx, idx)]
    y_sub = y[idx]
    W = np.diag(np.sqrt(weights))
    
    alpha_sub = np.linalg.solve(
        W @ K_sub @ W + n * lambda_reg * np.eye(m),
        W @ y_sub
    )
    alpha = W @ alpha_sub
    
    return idx, alpha
```

## Comparison of Sampling Strategies

| Strategy | Sample Size Needed | Computation |
|----------|-------------------|-------------|
| Uniform | $O(n)$ | $O(n)$ |
| Leverage | $O(\mathcal{N}(\lambda) \log n)$ | $O(n^3)$ for exact |
| Approx. Leverage | $O(\mathcal{N}(\lambda) \log n)$ | $O(nm^2)$ |

### Advantages of Leverage Sampling

1. **Optimal sample size**: Matches lower bounds
2. **Adapts to data**: More samples in important regions
3. **Preserves structure**: Maintains spectral properties

### Practical Considerations

1. **Approximate scores**: Use Nyström or RFF to estimate
2. **Iterative refinement**: Recompute scores after initial fit
3. **Streaming**: Can be adapted for online setting

## Applications

1. **Large-scale regression**: Scale kernel methods to big data
2. **Coresets**: Construct data summaries
3. **Active learning**: Prioritize informative samples
4. **Distributed computing**: Reduce communication in distributed KRR

## Citation

```bibtex
@article{avron2017random,
  title={Random Fourier features for kernel ridge regression: Approximation bounds and statistical guarantees},
  author={Avron, Haim and Clarkson, Kenneth L and Woodruff, David P},
  journal={International Conference on Machine Learning},
  pages={253--262},
  year={2017}
}
```

## Further Reading

- Drineas, P. & Mahoney, M. (2005). On the Nyström method for approximating a Gram matrix
- Ma, P., et al. (2015). A statistical perspective on algorithmic leveraging
- Alaoui, A. & Mahoney, M. (2015). Fast randomized kernel ridge regression
