---
sidebar_position: 1
title: "Smola et al. (2007) - Hilbert Space Embedding of Distributions"
---

# A Hilbert Space Embedding of Distributions

**Authors:** Alex Smola, Arthur Gretton, Le Song, Bernhard Schölkopf  
**Published:** 2007  
**Venue:** International Conference on Algorithmic Learning Theory  

## Summary

This paper introduces kernel mean embeddings, representing probability distributions as elements in a Reproducing Kernel Hilbert Space. This powerful technique enables nonparametric statistical inference using kernel methods.

## Key Contributions

### 1. Mean Embedding of Distributions

For a distribution $P$ on $\mathcal{X}$ and kernel $K$, the **mean embedding** is:

$$\mu_P = \mathbb{E}_{x \sim P}[\phi(x)] = \mathbb{E}_{x \sim P}[K(\cdot, x)]$$

This is an element of the RKHS $\mathcal{H}$.

### 2. Reproducing Property for Expectations

For any function $f \in \mathcal{H}$:
$$\mathbb{E}_{x \sim P}[f(x)] = \langle f, \mu_P \rangle_{\mathcal{H}}$$

This allows computing expectations via inner products.

### 3. Characteristic Kernels

A kernel is **characteristic** if the embedding is injective:
$$P = Q \iff \mu_P = \mu_Q$$

Examples: Gaussian RBF, Matérn kernels.

### 4. Empirical Estimation

Given samples $x_1, \ldots, x_n \sim P$:
$$\hat{\mu}_P = \frac{1}{n}\sum_{i=1}^n \phi(x_i) = \frac{1}{n}\sum_{i=1}^n K(\cdot, x_i)$$

## Mathematical Framework

### Bochner Integral

The mean embedding is defined via Bochner integral:
$$\mu_P = \int_{\mathcal{X}} \phi(x) dP(x)$$

exists when $\mathbb{E}_P[\sqrt{K(x,x)}] < \infty$.

### RKHS Distance Between Distributions

$$\|\mu_P - \mu_Q\|_{\mathcal{H}}^2 = \mathbb{E}_{x,x' \sim P}[K(x,x')] + \mathbb{E}_{y,y' \sim Q}[K(y,y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[K(x,y)]$$

This is the **Maximum Mean Discrepancy (MMD)**.

### Covariance Operator

The covariance operator $C_{XX}: \mathcal{H} \to \mathcal{H}$:
$$C_{XX} = \mathbb{E}[(\phi(X) - \mu_X) \otimes (\phi(X) - \mu_X)]$$

Enables covariance-based inference in RKHS.

## Applications

### 1. Two-Sample Testing

Test $H_0: P = Q$ using empirical MMD:
$$\widehat{MMD}^2 = \frac{1}{n^2}\sum_{i,j}K(x_i,x_j) + \frac{1}{m^2}\sum_{i,j}K(y_i,y_j) - \frac{2}{nm}\sum_{i,j}K(x_i,y_j)$$

### 2. Independence Testing

Test independence of $X$ and $Y$ using HSIC:
$$HSIC(P_{XY}) = \|\mu_{P_{XY}} - \mu_{P_X} \otimes \mu_{P_Y}\|^2$$

### 3. Conditional Embeddings

Conditional mean embedding:
$$\mu_{Y|X=x} = C_{YX} C_{XX}^{-1} \phi(x)$$

Enables nonparametric conditional density estimation.

### 4. Kernel Bayes' Rule

Compute posterior embeddings from prior and likelihood embeddings:
$$\mu_{X|Y} \propto C_{XX|Y} C_{YY}^{-1} \phi(Y)$$

## Algorithm: MMD Two-Sample Test

```python
import numpy as np

def compute_mmd(X, Y, kernel):
    """
    Compute Maximum Mean Discrepancy between samples.
    
    Parameters:
    -----------
    X : array of shape (n, d) - samples from P
    Y : array of shape (m, d) - samples from Q
    kernel : function - kernel function K(x, y)
    
    Returns:
    --------
    mmd_squared : float - squared MMD statistic
    """
    n, m = len(X), len(Y)
    
    # Compute kernel matrices
    K_XX = np.array([[kernel(x_i, x_j) for x_j in X] for x_i in X])
    K_YY = np.array([[kernel(y_i, y_j) for y_j in Y] for y_i in Y])
    K_XY = np.array([[kernel(x_i, y_j) for y_j in Y] for x_i in X])
    
    # Unbiased estimator
    mmd_squared = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) +
        (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) -
        2 * K_XY.mean()
    )
    
    return mmd_squared

def mmd_permutation_test(X, Y, kernel, n_permutations=1000):
    """Permutation test for H0: P = Q."""
    combined = np.vstack([X, Y])
    n = len(X)
    
    observed_mmd = compute_mmd(X, Y, kernel)
    
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(combined))
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        
        if compute_mmd(X_perm, Y_perm, kernel) >= observed_mmd:
            count += 1
    
    p_value = count / n_permutations
    return p_value
```

## Theoretical Results

### Convergence Rate

$$\|\hat{\mu}_P - \mu_P\|_{\mathcal{H}} = O_P(n^{-1/2})$$

The empirical embedding converges at parametric rate.

### Test Power

For characteristic kernels, MMD test is consistent:
- Type I error controlled at level $\alpha$
- Power $\to 1$ as $n \to \infty$ for $P \neq Q$

## Extensions

1. **Kernel Adaptive HSIC**: Optimize kernel for independence testing
2. **Distributional Embeddings**: Embed distributions of distributions  
3. **Kernel Mean Matching**: Domain adaptation
4. **Stein Operators**: Combine with score functions

## Impact

This paper has enabled:
- Nonparametric hypothesis testing
- Domain adaptation (covariate shift)
- Generative model evaluation (GAN metrics)
- Causal discovery
- Distributional regression

## Citation

```bibtex
@inproceedings{smola2007hilbert,
  title={A Hilbert space embedding for distributions},
  author={Smola, Alex and Gretton, Arthur and Song, Le and Sch{\"o}lkopf, Bernhard},
  booktitle={International Conference on Algorithmic Learning Theory},
  pages={13--31},
  year={2007}
}
```

## Further Reading

- Gretton, A., et al. (2012). A kernel two-sample test
- Song, L., et al. (2009). Hilbert space embeddings of conditional distributions
- Muandet, K., et al. (2017). Kernel mean embedding of distributions: A review
