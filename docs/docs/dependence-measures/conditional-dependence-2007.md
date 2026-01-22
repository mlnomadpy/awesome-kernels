---
sidebar_position: 2
title: "Fukumizu et al. (2007) - Kernel Measures of Conditional Dependence"
---

# Kernel Measures of Conditional Dependence

**Authors:** Kenji Fukumizu, Francis R. Bach, Arthur Gretton  
**Published:** 2007  
**Venue:** NeurIPS  
**Link:** [PDF](https://papers.nips.cc/paper/2007/hash/3a0772443a0739141571f2e4e675a4e2-Abstract.html)

## Summary

This paper extends kernel dependence measures to the conditional setting. It introduces kernel-based measures that characterize conditional independence $X \perp\!\!\!\perp Y | Z$ and provides consistent estimators for testing conditional independence hypotheses.

## Key Contributions

### 1. Conditional Cross-Covariance Operator

Define the conditional cross-covariance operator:
$$\mathcal{C}_{XY|Z} = \mathcal{C}_{XY} - \mathcal{C}_{XZ}\mathcal{C}_{ZZ}^{-1}\mathcal{C}_{ZY}$$

This generalizes the partial correlation to nonlinear dependencies.

### 2. Conditional Independence Characterization

**Theorem:** For characteristic kernels:
$$\|\mathcal{C}_{XY|Z}\|_{HS} = 0 \iff X \perp\!\!\!\perp Y | Z$$

### 3. Normalized Measure

The normalized conditional dependence measure:
$$\text{NOCCO}(X,Y|Z) = \frac{\|\mathcal{C}_{XY|Z}\|_{HS}}{\|\mathcal{C}_{XX|Z}\|_{HS}^{1/2} \|\mathcal{C}_{YY|Z}\|_{HS}^{1/2}}$$

This is analogous to partial correlation for nonlinear relationships.

## Mathematical Framework

### Covariance Operators

In RKHS $\mathcal{H}$ with feature map $\phi$:
- **Mean:** $\mu_X = \mathbb{E}[\phi(X)]$
- **Covariance:** $\mathcal{C}_{XX} = \mathbb{E}[\phi(X) \otimes \phi(X)] - \mu_X \otimes \mu_X$
- **Cross-covariance:** $\mathcal{C}_{XY} = \mathbb{E}[\phi(X) \otimes \psi(Y)] - \mu_X \otimes \mu_Y$

### Conditional Operator Derivation

From the regression interpretation:
$$\mathbb{E}[\phi(X)|Z] = \mathcal{C}_{XZ}\mathcal{C}_{ZZ}^{-1}\phi(Z)$$

The conditional cross-covariance removes the part explained by $Z$:
$$\mathcal{C}_{XY|Z} = \text{Cov}(\phi(X) - \mathbb{E}[\phi(X)|Z], \psi(Y) - \mathbb{E}[\psi(Y)|Z])$$

### Regularized Inverse

In practice, use regularized inverse:
$$\mathcal{C}_{XY|Z}^\epsilon = \mathcal{C}_{XY} - \mathcal{C}_{XZ}(\mathcal{C}_{ZZ} + \epsilon I)^{-1}\mathcal{C}_{ZY}$$

## Empirical Estimation

### Matrix Formulation

Given samples $\{(x_i, y_i, z_i)\}_{i=1}^n$, define centered kernel matrices:
- $\tilde{K} = HKH$ (kernel for $X$)
- $\tilde{L} = HLH$ (kernel for $Y$)  
- $\tilde{M} = HMH$ (kernel for $Z$)

where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$.

### Conditional HSIC Estimator

$$\widehat{\mathcal{C}}_{XY|Z} = \frac{1}{n^2}\tilde{K}(\tilde{L} - \tilde{M}(\tilde{M} + \epsilon n I)^{-1}\tilde{L})$$

The Hilbert-Schmidt norm:
$$\widehat{\text{HSIC}}_{XY|Z} = \|\widehat{\mathcal{C}}_{XY|Z}\|_{HS}^2$$

## Algorithm

```python
import numpy as np

def conditional_hsic(K, L, M, epsilon=0.01):
    """
    Compute conditional HSIC.
    
    Parameters:
    -----------
    K : kernel matrix for X (n, n)
    L : kernel matrix for Y (n, n)
    M : kernel matrix for Z (n, n)
    epsilon : regularization parameter
    
    Returns:
    --------
    hsic_cond : conditional HSIC value
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Center kernel matrices
    K_tilde = H @ K @ H
    L_tilde = H @ L @ H
    M_tilde = H @ M @ H
    
    # Regularized inverse
    M_reg_inv = np.linalg.inv(M_tilde + epsilon * n * np.eye(n))
    
    # Residual kernel for Y after regressing on Z
    L_residual = L_tilde - M_tilde @ M_reg_inv @ L_tilde
    
    # Conditional HSIC
    C_cond = K_tilde @ L_residual / n**2
    hsic_cond = np.trace(C_cond @ C_cond.T)
    
    return hsic_cond

def conditional_independence_test(X, Y, Z, kernel_x, kernel_y, kernel_z,
                                   epsilon=0.01, n_permutations=1000, alpha=0.05):
    """
    Test for conditional independence X ⊥ Y | Z.
    """
    K = kernel_x(X, X)
    L = kernel_y(Y, Y)
    M = kernel_z(Z, Z)
    
    # Observed statistic
    observed = conditional_hsic(K, L, M, epsilon)
    
    # Permutation null (permute Y given Z clusters)
    null_values = []
    for _ in range(n_permutations):
        perm = np.random.permutation(len(Y))
        L_perm = L[np.ix_(perm, perm)]
        null_values.append(conditional_hsic(K, L_perm, M, epsilon))
    
    p_value = np.mean(np.array(null_values) >= observed)
    
    return {
        'statistic': observed,
        'p_value': p_value,
        'reject': p_value < alpha
    }
```

## Theoretical Properties

### Consistency

As $n \to \infty$ and $\epsilon \to 0$ appropriately:
$$\widehat{\text{HSIC}}_{XY|Z} \xrightarrow{p} \|\mathcal{C}_{XY|Z}\|_{HS}^2$$

### Regularization Selection

Optimal $\epsilon$ balances:
- Bias from regularization: $O(\epsilon)$
- Variance from estimation: $O(1/(\epsilon^2 n))$

Choosing $\epsilon = O(n^{-1/3})$ gives rate $O(n^{-1/3})$.

### Convergence Rate

Under regularity conditions:
$$|\widehat{\text{HSIC}}_{XY|Z} - \|\mathcal{C}_{XY|Z}\|_{HS}^2| = O_p(n^{-1/3})$$

## Applications

### 1. Causal Discovery

Test whether $X \to Y$ or $Y \to X$ given $Z$:
- $X \perp\!\!\!\perp Y | Z$ suggests $Z$ is a confounder or mediator
- $X \not\perp\!\!\!\perp Y | Z$ suggests direct causal link

### 2. Feature Selection

Select features $X_i$ such that:
$$X_i \not\perp\!\!\!\perp Y | X_{-i}$$

### 3. Graphical Models

Learn conditional independence structure for kernel graphical models.

### 4. Transfer Learning

Test whether domain shift $Z$ explains difference between $X$ and $Y$.

## Comparison with Parametric Tests

| Test | Assumptions | Nonlinear | Complexity |
|------|-------------|-----------|------------|
| Partial correlation | Gaussian | ✗ | $O(n)$ |
| Conditional MI | Known density | ✓ | $O(n \log n)$ |
| Kernel CI | Characteristic kernel | ✓ | $O(n^3)$ |

## Citation

```bibtex
@inproceedings{fukumizu2007kernel,
  title={Kernel measures of conditional dependence},
  author={Fukumizu, Kenji and Bach, Francis R and Gretton, Arthur},
  booktitle={Advances in Neural Information Processing Systems},
  volume={20},
  year={2007}
}
```

## Further Reading

- Zhang, K., et al. (2011). Kernel-based conditional independence test and application
- Strobl, E., et al. (2019). Approximate kernel-based conditional independence tests
- Park, G. & Muandet, K. (2020). A measure-theoretic approach to kernel CI tests
