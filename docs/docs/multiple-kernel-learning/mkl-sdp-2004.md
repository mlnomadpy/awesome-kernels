---
sidebar_position: 1
title: "Lanckriet et al. (2004) - Learning the Kernel Matrix with SDP"
---

# Learning the Kernel Matrix with Semidefinite Programming

**Authors:** Gert R. G. Lanckriet, Nello Cristianini, Peter Bartlett, Laurent El Ghaoui, Michael I. Jordan  
**Published:** 2004  
**Journal:** Journal of Machine Learning Research  
**Link:** [PDF](https://www.jmlr.org/papers/v5/lanckriet04a.html)

## Summary

This foundational paper introduces the framework for learning the kernel matrix directly from data using semidefinite programming (SDP). It shows how to optimize over the cone of positive semidefinite matrices to find an optimal kernel for classification tasks.

## Key Contributions

### 1. Kernel Learning as SDP

The key insight is that valid kernels correspond to positive semidefinite (PSD) matrices:
$$K \succeq 0$$

This constraint defines a convex cone, enabling optimization over kernels.

### 2. Transductive Setting

Given training points $\{x_1, \ldots, x_n\}$ with labels $\{y_1, \ldots, y_n\}$, learn the Gram matrix $K$ directly:

$$\max_{K \succeq 0} \quad \text{margin}(K)$$
$$\text{subject to} \quad \text{trace}(K) \leq c$$

### 3. Multiple Kernel Learning

Restrict to convex combinations of base kernels:
$$K = \sum_{m=1}^M \mu_m K_m, \quad \mu_m \geq 0, \quad \sum_m \mu_m = 1$$

This reduces the infinite-dimensional problem to learning weights $\mu$.

## Mathematical Framework

### SVM Dual with Kernel Learning

The soft-margin SVM dual becomes:

$$\max_{\alpha, K} \quad \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K_{ij}$$
$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$
$$\quad\quad\quad\quad K \succeq 0, \quad \text{trace}(K) \leq c$$

### Reformulation as SDP

Define $t = \alpha \circ y$ (element-wise product). The problem becomes:

$$\min_{K,t} \quad t^T K t$$
$$\text{subject to} \quad K \succeq 0$$
$$\quad\quad\quad\quad \text{trace}(K) \leq c$$
$$\quad\quad\quad\quad \text{linear constraints on } t$$

Using Schur complement, this is equivalent to an SDP.

### Quadratically Constrained QP (QCQP)

For MKL with fixed base kernels:

$$\max_{\alpha,\mu} \quad 2\alpha^T \mathbf{1} - \sum_m \mu_m \alpha^T D_y K_m D_y \alpha$$
$$\text{subject to} \quad \sum_m \mu_m = 1, \quad \mu_m \geq 0$$
$$\quad\quad\quad\quad 0 \leq \alpha \leq C\mathbf{1}, \quad \alpha^T y = 0$$

where $D_y = \text{diag}(y)$.

## Algorithms

### 1. Interior Point Methods

Standard SDP solvers can handle:
- $O(n^3)$ variables for full kernel matrix
- Practical for $n < 1000$

### 2. Alternating Optimization

```
while not converged:
    Fix μ, solve SVM for α
    Fix α, solve convex problem for μ
```

### 3. Sequential Minimal Optimization (SMO)

For large-scale problems, coordinate descent methods are more efficient.

## Margin Definitions

### Hard Margin
$$\rho = \min_i y_i f(x_i) = \min_i y_i \sum_j \alpha_j y_j K(x_i, x_j)$$

### Soft Margin (with slack)
$$\rho = 1 - \frac{1}{C}\sum_i \xi_i$$

### Alignment-Based Criterion

The kernel-target alignment:
$$A(K, yy^T) = \frac{\langle K, yy^T \rangle_F}{\|K\|_F \|yy^T\|_F}$$

Maximizing alignment is convex and can be solved efficiently.

## Experimental Results

The paper demonstrates:
1. **Improved accuracy**: Learned kernels outperform fixed kernels
2. **Automatic feature selection**: MKL weights identify relevant kernels
3. **Interpretability**: Kernel weights reveal data structure

## Impact on Machine Learning

This paper:
1. **Founded MKL field**: Established kernel learning as a discipline
2. **Convex optimization**: Showed connection to SDP
3. **Practical algorithms**: Enabled learning kernels from data
4. **Feature combination**: Provided principled way to combine features

## Citation

```bibtex
@article{lanckriet2004learning,
  title={Learning the kernel matrix with semidefinite programming},
  author={Lanckriet, Gert RG and Cristianini, Nello and Bartlett, Peter and 
          El Ghaoui, Laurent and Jordan, Michael I},
  journal={Journal of Machine Learning Research},
  volume={5},
  pages={27--72},
  year={2004}
}
```

## Further Reading

- Bach, F., et al. (2004). Multiple kernel learning, conic duality, and the SMO algorithm
- Sonnenburg, S., et al. (2006). Large scale multiple kernel learning
- Gönen, M. & Alpaydın, E. (2011). Multiple kernel learning algorithms
