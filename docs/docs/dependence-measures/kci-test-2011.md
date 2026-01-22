---
sidebar_position: 3
title: "Zhang et al. (2011) - Kernel-Based Conditional Independence Test"
---

# A Kernel-Based Test for Conditional Independence

**Authors:** Kun Zhang, Jonas Peters, Dominik Janzing, Bernhard Schölkopf  
**Published:** 2011  
**Venue:** UAI  
**Link:** [PDF](https://dl.acm.org/doi/10.5555/2986459.2986567)

## Summary

This paper develops a practical kernel-based test for conditional independence, which is crucial for causal discovery algorithms. It provides both the statistical methodology and computational improvements to make the test scalable.

## Key Contributions

### 1. Kernel Conditional Independence (KCI) Test

The test statistic is based on the Hilbert-Schmidt norm of the conditional cross-covariance operator:
$$T = n \cdot \text{Tr}(K_X \tilde{R}_Z K_Y \tilde{R}_Z)$$

where $\tilde{R}_Z = (K_Z + \epsilon I)^{-1}$ is the regularized "residual-making" matrix.

### 2. Asymptotic Distribution

Under $H_0: X \perp\!\!\!\perp Y | Z$, the test statistic has distribution:
$$T \xrightarrow{d} \sum_{i,j} \lambda_{ij} \chi^2_{ij}$$

a weighted sum of chi-squared random variables.

### 3. Gamma Approximation

For practical computation, approximate the null distribution:
$$T \sim \alpha \Gamma(k, \theta) + \beta$$

where parameters are estimated from the eigenvalues.

## Mathematical Framework

### Problem Setup

Test the hypothesis:
- $H_0$: $X \perp\!\!\!\perp Y | Z$ (conditional independence)
- $H_1$: $X \not\perp\!\!\!\perp Y | Z$ (conditional dependence)

### Regression Residuals in RKHS

The key idea: $X \perp\!\!\!\perp Y | Z$ iff the RKHS projections of $X$ and $Y$ onto $Z$ leave independent residuals.

Define residual operators:
$$\tilde{\phi}(X) = \phi(X) - \mathbb{E}[\phi(X)|Z]$$
$$\tilde{\psi}(Y) = \psi(Y) - \mathbb{E}[\psi(Y)|Z]$$

### Test Statistic Derivation

The conditional cross-covariance operator:
$$\mathcal{C}_{XY|Z} = \mathbb{E}[\tilde{\phi}(X) \otimes \tilde{\psi}(Y)]$$

Test statistic is the empirical Hilbert-Schmidt norm:
$$T = n \|\widehat{\mathcal{C}}_{XY|Z}\|_{HS}^2$$

## Algorithm

```python
import numpy as np
from scipy import stats
from scipy.linalg import eigh

def kci_test(X, Y, Z, kernel_x, kernel_y, kernel_z, epsilon=1e-3, alpha=0.05):
    """
    Kernel Conditional Independence test.
    
    Test H0: X ⊥ Y | Z
    
    Parameters:
    -----------
    X, Y, Z : data arrays (n, d_x), (n, d_y), (n, d_z)
    kernel_x, kernel_y, kernel_z : kernel functions
    epsilon : regularization parameter
    alpha : significance level
    
    Returns:
    --------
    dict with test statistic, p-value, and decision
    """
    n = len(X)
    
    # Compute kernel matrices
    K_X = kernel_x(X, X)
    K_Y = kernel_y(Y, Y)
    K_Z = kernel_z(Z, Z)
    
    # Center kernel matrices
    H = np.eye(n) - np.ones((n, n)) / n
    K_X = H @ K_X @ H
    K_Y = H @ K_Y @ H
    K_Z = H @ K_Z @ H
    
    # Regularized inverse for Z
    R_Z = np.linalg.inv(K_Z + epsilon * np.eye(n))
    
    # Compute residual kernels (remove Z dependence)
    K_X_res = K_X - K_X @ R_Z @ K_Z
    K_Y_res = K_Y - K_Y @ R_Z @ K_Z
    
    # Test statistic: Tr(K_X_res @ K_Y_res) / n
    test_stat = np.trace(K_X_res @ K_Y_res) / n
    
    # Compute p-value using gamma approximation
    p_value = compute_pvalue_gamma(K_X_res, K_Y_res, test_stat, n)
    
    return {
        'statistic': test_stat,
        'p_value': p_value,
        'reject': p_value < alpha,
        'independent': p_value >= alpha
    }

def compute_pvalue_gamma(K_X, K_Y, test_stat, n):
    """
    Compute p-value using gamma approximation to null distribution.
    """
    # Compute eigenvalues of the product
    # The null distribution is sum of lambda_ij * chi^2
    
    # Eigenvalues of centered kernels
    eig_x = np.maximum(np.linalg.eigvalsh(K_X / n), 0)
    eig_y = np.maximum(np.linalg.eigvalsh(K_Y / n), 0)
    
    # Products of eigenvalues
    eig_prod = np.outer(eig_x, eig_y).flatten()
    eig_prod = eig_prod[eig_prod > 1e-10]
    
    # Fit gamma distribution
    # Mean and variance of sum of weighted chi-squares
    mean_null = np.sum(eig_prod)
    var_null = 2 * np.sum(eig_prod**2)
    
    # Gamma parameters
    if var_null > 0:
        k = mean_null**2 / var_null  # shape
        theta = var_null / mean_null  # scale
        
        # P-value from gamma distribution
        p_value = 1 - stats.gamma.cdf(test_stat * n, k, scale=theta)
    else:
        p_value = 1.0
    
    return p_value

def kci_test_permutation(X, Y, Z, kernel_x, kernel_y, kernel_z, 
                         epsilon=1e-3, n_perm=1000, alpha=0.05):
    """
    KCI test with permutation-based p-value.
    """
    n = len(X)
    
    K_X = kernel_x(X, X)
    K_Y = kernel_y(Y, Y)
    K_Z = kernel_z(Z, Z)
    
    H = np.eye(n) - np.ones((n, n)) / n
    K_X = H @ K_X @ H
    K_Y = H @ K_Y @ H  
    K_Z = H @ K_Z @ H
    
    R_Z = np.linalg.inv(K_Z + epsilon * np.eye(n))
    K_X_res = K_X - K_X @ R_Z @ K_Z
    
    # Observed statistic
    K_Y_res = K_Y - K_Y @ R_Z @ K_Z
    observed = np.trace(K_X_res @ K_Y_res) / n
    
    # Permutation null
    null_stats = []
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        K_Y_perm = K_Y[np.ix_(perm, perm)]
        K_Y_res_perm = K_Y_perm - K_Y_perm @ R_Z @ K_Z
        null_stats.append(np.trace(K_X_res @ K_Y_res_perm) / n)
    
    p_value = np.mean(np.array(null_stats) >= observed)
    
    return {
        'statistic': observed,
        'p_value': p_value,
        'reject': p_value < alpha
    }
```

## Regularization Selection

### Effect of $\epsilon$

- **Too small**: Numerical instability, overfitting
- **Too large**: Lose power, don't condition on $Z$ properly

### Adaptive Selection

Use cross-validation or:
$$\epsilon = c \cdot \text{Tr}(K_Z) / n$$

where $c \in [0.01, 0.1]$ typically works well.

## Computational Complexity

| Component | Complexity |
|-----------|------------|
| Kernel matrices | $O(n^2 d)$ |
| Matrix inversion | $O(n^3)$ |
| Eigendecomposition | $O(n^3)$ |
| **Total** | $O(n^3)$ |

### Scalability Improvements

1. **Random Fourier Features**: $O(nD^2)$ where $D \ll n$
2. **Nyström approximation**: $O(nm^2)$ where $m \ll n$
3. **Incomplete Cholesky**: Adaptive low-rank approximation

## Applications

### 1. Causal Discovery

Essential for constraint-based algorithms:
- PC algorithm
- FCI algorithm
- Conditional independence oracle

### 2. Markov Blanket Discovery

Find variables $MB(Y)$ such that $Y \perp\!\!\!\perp X | MB(Y)$ for all $X \notin MB(Y)$.

### 3. Variable Selection

Test if feature $X_i$ provides information beyond other features:
$$X_i \not\perp\!\!\!\perp Y | X_{-i}$$

## Comparison with Other CI Tests

| Test | Nonlinear | High-dim | Sample efficiency |
|------|-----------|----------|-------------------|
| Partial correlation | ✗ | ✓ | High |
| G-test | ✗ | ✗ | Medium |
| CMI estimation | ✓ | ✗ | Low |
| KCI | ✓ | ✓ | Medium |

## Citation

```bibtex
@inproceedings{zhang2011kernel,
  title={Kernel-based conditional independence test and application in causal discovery},
  author={Zhang, Kun and Peters, Jonas and Janzing, Dominik and Sch{\"o}lkopf, Bernhard},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={804--813},
  year={2011}
}
```

## Further Reading

- Fukumizu, K., et al. (2007). Kernel measures of conditional dependence
- Doran, G., et al. (2014). A permutation-based kernel conditional independence test
- Strobl, E., et al. (2019). Approximate kernel-based conditional independence tests
