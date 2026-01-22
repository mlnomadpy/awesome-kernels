---
sidebar_position: 2
title: "Kanagawa et al. (2018) - Gaussian Processes and Kernel Methods: A Review"
---

# Gaussian Processes and Kernel Methods: A Review on Connections

**Authors:** Motonobu Kanagawa, Bharath K. Sriperumbudur, Kenji Fukumizu, Arthur Gretton  
**Published:** 2018  
**Venue:** arXiv  
**Link:** [arXiv](https://arxiv.org/abs/1807.02582)

## Summary

This comprehensive review paper establishes rigorous connections between Gaussian Processes (GPs) and Reproducing Kernel Hilbert Spaces (RKHS). It clarifies when GP posteriors and RKHS estimators coincide and when they differ, providing deep insights into both frameworks.

## Key Contributions

### 1. Sample Path Properties

The paper clarifies that GP sample paths are typically **not** in the RKHS:

**Theorem:** For a GP with kernel $k$, sample paths $f \sim \mathcal{GP}(0, k)$ satisfy:
$$P(f \in \mathcal{H}_k) = 0$$

unless the RKHS is finite-dimensional.

### 2. RKHS as Support

While samples aren't in RKHS, the RKHS characterizes the support:
$$\text{supp}(\mathcal{GP}(0, k)) = \overline{\mathcal{H}_k}$$

where the closure is in an appropriate topology.

### 3. Posterior Mean Equivalence

The GP posterior mean equals the RKHS interpolant:
$$\mathbb{E}[f(x)|y] = \arg\min_{g \in \mathcal{H}_k} \|g\|_{\mathcal{H}}^2 \text{ s.t. } g(x_i) = \tilde{y}_i$$

where $\tilde{y}$ accounts for noise.

## Mathematical Framework

### RKHS via Integral Operator

The RKHS $\mathcal{H}_k$ can be characterized through the integral operator:
$$T_k: L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu)$$
$$(T_k f)(x) = \int k(x, x') f(x') d\mu(x')$$

Then: $\mathcal{H}_k = T_k^{1/2}(L^2(\mathcal{X}, \mu))$

### Mercer Representation

When $k$ has eigendecomposition $k(x,x') = \sum_i \lambda_i \phi_i(x)\phi_i(x')$:
$$\mathcal{H}_k = \left\{f = \sum_i f_i \phi_i : \sum_i \frac{f_i^2}{\lambda_i} < \infty\right\}$$

GP samples satisfy $\sum_i f_i^2 < \infty$ (weaker condition).

### Cameron-Martin Space

For Gaussian measure $\mathcal{GP}(0, k)$, the Cameron-Martin space is exactly $\mathcal{H}_k$:
- Translations by $h \in \mathcal{H}_k$ give equivalent measures
- Translations by $h \notin \mathcal{H}_k$ give orthogonal measures

## Connections Summary

| Property | GP | RKHS |
|----------|-----|------|
| Function space | $L^2$ or continuous | $\mathcal{H}_k$ |
| Prior/Regularizer | $\mathcal{N}(0, K)$ | $\|f\|_{\mathcal{H}}$ |
| Posterior mean | $K_*(K + \sigma^2 I)^{-1}y$ | Same! |
| Uncertainty | Full posterior | None (point estimate) |
| Convergence | In $L^2$ | In RKHS norm |

## Regularization Correspondence

### Ridge Regression

RKHS formulation:
$$\hat{f} = \arg\min_{f \in \mathcal{H}} \sum_i (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$$

GP formulation:
$$\mathbb{E}[f(x)|y] \text{ with noise variance } \sigma^2 = \lambda$$

**These give the same function!**

### Different Regularization = Different Prior

| Regularization | Equivalent Prior |
|---------------|------------------|
| $\|f\|_{\mathcal{H}}^2$ | $\mathcal{GP}(0, k)$ |
| $\|f\|_{\mathcal{H}}^p$ | Non-Gaussian |
| Elastic net | Mixture prior |

## Convergence Analysis

### RKHS Convergence

The RKHS estimator converges in RKHS norm:
$$\|\hat{f}_n - f^*\|_{\mathcal{H}} \to 0$$

This implies uniform convergence when $\mathcal{H}$ embeds continuously into $C(\mathcal{X})$.

### GP Posterior Contraction

The GP posterior contracts around the truth:
$$\Pi_n(f : \|f - f^*\|_{L^2} > \epsilon | y^{(n)}) \to 0$$

Rate depends on smoothness of $f^*$ relative to RKHS.

## Misspecification

### Well-specified Case

When $f^* \in \mathcal{H}_k$:
- RKHS estimator achieves optimal rate
- GP posterior is consistent

### Misspecified Case

When $f^* \notin \mathcal{H}_k$:
- RKHS estimator converges to projection
- GP posterior may be inconsistent
- Calibration of uncertainty is lost

## Practical Implications

### 1. Model Selection

GP marginal likelihood corresponds to RKHS complexity:
$$\log p(y|X) \propto -\|f\|_{\mathcal{H}}^2 - \text{data fit}$$

### 2. Uncertainty Quantification

GP provides uncertainty; RKHS doesn't:
- For decisions: Use GP variance
- For prediction only: RKHS sufficient

### 3. Computational Tradeoffs

| Method | Training | Uncertainty | Scalability |
|--------|----------|-------------|-------------|
| Full GP | $O(n^3)$ | Exact | Limited |
| RKHS/KRR | $O(n^3)$ | None | Limited |
| Sparse GP | $O(nm^2)$ | Approximate | Good |
| Random features | $O(nD^2)$ | None | Excellent |

## Extensions

### Deep GPs

Stack GPs hierarchically:
$$f_L = g_L \circ g_{L-1} \circ \cdots \circ g_1$$
where each $g_l \sim \mathcal{GP}$.

### GP-RKHS for Non-Gaussian Likelihoods

Use RKHS theory for MAP estimation:
$$\hat{f} = \arg\min_{f \in \mathcal{H}} \sum_i \ell(y_i, f(x_i)) + \frac{1}{2}\|f\|_{\mathcal{H}}^2$$

## Citation

```bibtex
@article{kanagawa2018gaussian,
  title={Gaussian processes and kernel methods: A review on connections},
  author={Kanagawa, Motonobu and Sriperumbudur, Bharath K and Fukumizu, Kenji},
  journal={arXiv preprint arXiv:1807.02582},
  year={2018}
}
```

## Further Reading

- Van der Vaart, A. & Van Zanten, H. (2008). Reproducing kernel Hilbert spaces of Gaussian priors
- Berlinet, A. & Thomas-Agnan, C. (2011). Reproducing kernel Hilbert spaces in probability and statistics
- Karvonen, T. & Särkkä, S. (2020). Worst-case optimal approximation with RKHS
