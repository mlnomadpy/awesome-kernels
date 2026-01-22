---
sidebar_position: 3
title: "Caponnetto & De Vito (2007) - Optimal Rates for Regularized Least-Squares"
---

# Optimal Rates for the Regularized Least-Squares Algorithm

**Authors:** Andrea Caponnetto, Ernesto De Vito  
**Published:** 2007  
**Journal:** Foundations of Computational Mathematics  
**Link:** [Springer](https://link.springer.com/article/10.1007/s10208-006-0196-8)

## Summary

This influential paper establishes optimal learning rates for kernel ridge regression (regularized least-squares) under various assumptions on the regularity of the target function and the capacity of the hypothesis space. It provides a complete characterization of when fast rates are achievable.

## Key Contributions

### 1. Optimal Rates Framework

The paper shows that the learning rate depends on:
- **Source condition**: How "smooth" the target function is
- **Capacity condition**: How complex the hypothesis space is (eigenvalue decay)

### 2. Source Condition

The target function $f_\rho$ satisfies a source condition of order $r$ if:
$$f_\rho = L_K^r g$$
for some $g \in L^2(\rho_X)$ with $\|g\|_{L^2} \leq R$.

This means $f_\rho$ belongs to the interpolation space $[L^2, \mathcal{H}_K]_{2r}$.

### 3. Capacity Condition  

The kernel $K$ satisfies a capacity condition of order $s$ if:
$$\text{Tr}(L_K) < \infty \quad \text{and} \quad \mathcal{N}(\lambda) \leq c_0 \lambda^{-s}$$

where $\mathcal{N}(\lambda) = \sum_i \frac{\lambda_i}{\lambda_i + \lambda}$ is the effective dimension.

### 4. Main Rate Theorem

**Theorem:** Under source condition of order $r \leq 1$ and capacity condition of order $s$, with $\lambda_n = n^{-\frac{1}{2r+s}}$:

$$\mathbb{E}\|f_{\lambda_n} - f_\rho\|_{L^2}^2 \leq C \cdot n^{-\frac{2r}{2r+s}}$$

This rate is minimax optimal.

## Mathematical Framework

### Regularized Estimator

Given data $(x_1, y_1), \ldots, (x_n, y_n)$, the estimator is:
$$f_z = \arg\min_{f \in \mathcal{H}_K} \frac{1}{n}\sum_{i=1}^n(f(x_i) - y_i)^2 + \lambda\|f\|_{\mathcal{H}_K}^2$$

Closed form solution:
$$f_z = (K + n\lambda I)^{-1} \mathbf{y}$$

### Error Decomposition

$$\|f_z - f_\rho\|_{L^2}^2 \leq 2\underbrace{\|f_\lambda - f_\rho\|_{L^2}^2}_{\text{approximation error}} + 2\underbrace{\|f_z - f_\lambda\|_{L^2}^2}_{\text{sample error}}$$

### Bias-Variance Tradeoff

**Approximation error (bias):**
$$\|f_\lambda - f_\rho\|_{L^2} \leq R\lambda^r$$

**Sample error (variance):**
$$\mathbb{E}\|f_z - f_\lambda\|_{L^2}^2 \leq \frac{C\mathcal{N}(\lambda)}{n}$$

## Special Cases

### Case 1: Finite-Dimensional RKHS
- $s = 0$ (finite effective dimension)
- Rate: $O(n^{-1})$ for any $r > 0$
- Parametric rate achieved

### Case 2: Sobolev Spaces
For Sobolev space $H^s(\mathbb{R}^d)$:
- Eigenvalue decay: $\lambda_i \sim i^{-2s/d}$
- Capacity: $s_{cap} = d/(2s)$
- Rate: $O(n^{-\frac{2r}{2r + d/(2s)}})$

### Case 3: Gaussian RBF Kernel
- Eigenvalues decay exponentially
- Effective dimension: $\mathcal{N}(\lambda) = O(\log^d(1/\lambda))$
- Near-parametric rates for smooth targets

## Rate Summary Table

| Source ($r$) | Capacity ($s$) | Optimal $\lambda$ | Rate |
|--------------|----------------|-------------------|------|
| 1/2 | 1 | $n^{-1/2}$ | $n^{-1/2}$ |
| 1 | 1 | $n^{-1/3}$ | $n^{-2/3}$ |
| 1/2 | 1/2 | $n^{-2/3}$ | $n^{-2/3}$ |
| $\infty$ | 1 | $n^{-1}$ | $n^{-1}$ |

## Minimax Lower Bounds

The paper also proves matching lower bounds:

**Theorem:** For the class of distributions satisfying source and capacity conditions, no estimator can achieve rate faster than $n^{-\frac{2r}{2r+s}}$.

## Impact on Machine Learning

This paper:
1. **Optimal rates**: First complete characterization for kernel ridge regression
2. **Adaptive methods**: Motivated development of adaptive regularization
3. **Unified framework**: Connected statistical learning to approximation theory
4. **Practical guidance**: Provided theoretical basis for hyperparameter selection

## Citation

```bibtex
@article{caponnetto2007optimal,
  title={Optimal rates for the regularized least-squares algorithm},
  author={Caponnetto, Andrea and De Vito, Ernesto},
  journal={Foundations of Computational Mathematics},
  volume={7},
  number={3},
  pages={331--368},
  year={2007}
}
```

## Further Reading

- Smale, S. & Zhou, D.-X. (2007). Learning theory estimates via integral operators
- Steinwart, I., et al. (2009). Optimal rates for regularized least squares regression
- Fischer, S. & Steinwart, I. (2020). Sobolev norm learning rates for regularized least-squares
