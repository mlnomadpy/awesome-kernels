---
sidebar_position: 1
title: "Cortes & Vapnik (1995) - Support-Vector Networks"
---

# Support-Vector Networks

**Authors:** Corinna Cortes, Vladimir Vapnik  
**Published:** 1995  
**Journal:** Machine Learning  
**Link:** [PDF](https://link.springer.com/article/10.1007/BF00994018)

## Summary

This foundational paper introduces the soft-margin Support Vector Machine (SVM), extending the optimal hyperplane algorithm to handle non-separable data through slack variables. It presents the kernel trick for nonlinear classification, becoming one of the most influential papers in machine learning.

## Key Contributions

### 1. Optimal Separating Hyperplane

For linearly separable data, find the hyperplane that maximizes the margin:

$$\min_{w, b} \frac{1}{2}\|w\|^2$$
$$\text{subject to } y_i(w^\top x_i + b) \geq 1, \quad i = 1, \ldots, n$$

The margin is $\frac{2}{\|w\|}$.

### 2. Soft-Margin SVM

For non-separable data, introduce slack variables $\xi_i \geq 0$:

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i$$
$$\text{subject to } y_i(w^\top x_i + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0$$

Parameter $C$ controls the trade-off between margin and classification errors.

### 3. The Kernel Trick

Replace dot products with kernel functions:
$$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$$

enabling nonlinear decision boundaries in the original space.

### 4. Dual Formulation

The dual problem:
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{subject to } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

### 5. Support Vectors

Points with $\alpha_i > 0$ are **support vectors** — they lie on or within the margin and completely determine the decision boundary.

## Mathematical Framework

### Primal Problem (Hinge Loss Form)

Equivalently:
$$\min_{w, b} \frac{1}{n}\sum_{i=1}^n \max(0, 1 - y_i(w^\top x_i + b)) + \frac{\lambda}{2}\|w\|^2$$

where $\lambda = 1/(nC)$.

### KKT Conditions

For optimal $(\alpha^*, w^*, b^*, \xi^*)$:
1. $\alpha_i^* = 0 \Rightarrow y_i(w^{*\top}x_i + b^*) \geq 1$
2. $0 < \alpha_i^* < C \Rightarrow y_i(w^{*\top}x_i + b^*) = 1$
3. $\alpha_i^* = C \Rightarrow y_i(w^{*\top}x_i + b^*) \leq 1$

### Decision Function

$$f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, x) + b\right)$$

## Common Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $x^\top y$ | High-dimensional sparse data |
| Polynomial | $(x^\top y + c)^d$ | Image features |
| RBF | $\exp(-\gamma\|x-y\|^2)$ | General purpose |
| Sigmoid | $\tanh(\kappa x^\top y + c)$ | Neural network analogy |

## Algorithm: SMO

Sequential Minimal Optimization (Platt, 1998) solves the dual efficiently:

```python
def smo_step(i, j, alpha, K, y, C):
    """One step of SMO algorithm."""
    # Compute bounds for alpha_j
    if y[i] != y[j]:
        L = max(0, alpha[j] - alpha[i])
        H = min(C, C + alpha[j] - alpha[i])
    else:
        L = max(0, alpha[i] + alpha[j] - C)
        H = min(C, alpha[i] + alpha[j])
    
    if L == H:
        return False
    
    # Second derivative of objective
    eta = 2 * K[i,j] - K[i,i] - K[j,j]
    
    if eta >= 0:
        return False
    
    # Compute errors
    E_i = decision_function(i) - y[i]
    E_j = decision_function(j) - y[j]
    
    # Update alpha_j
    alpha_j_new = alpha[j] - y[j] * (E_i - E_j) / eta
    alpha_j_new = np.clip(alpha_j_new, L, H)
    
    # Update alpha_i
    alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)
    
    alpha[i], alpha[j] = alpha_i_new, alpha_j_new
    return True
```

## Theoretical Foundations

### VC Dimension

For SVMs with RBF kernel, the VC dimension is controlled by:
- Number of support vectors
- Margin size
- Data geometry

### Generalization Bound

$$R[f] \leq \hat{R}[f] + O\left(\sqrt{\frac{|\text{SV}|}{n}}\right)$$

where $|\text{SV}|$ is the number of support vectors.

### Maximum Margin Theory

Maximizing the margin minimizes a bound on the VC dimension, leading to better generalization.

## Extensions

1. **$\nu$-SVM**: Parameter $\nu$ directly controls fraction of support vectors
2. **One-class SVM**: Novelty/anomaly detection
3. **SVR**: Support Vector Regression
4. **Multi-class SVM**: One-vs-one, one-vs-all strategies
5. **Structured SVM**: Structured output prediction

## Impact

This paper has been cited over 45,000 times and:
- Established SVMs as a standard ML algorithm
- Introduced the "kernel trick" to wider ML community
- Influenced development of many kernel methods
- Led to theoretical advances in statistical learning theory

## Citation

```bibtex
@article{cortes1995support,
  title={Support-vector networks},
  author={Cortes, Corinna and Vapnik, Vladimir},
  journal={Machine Learning},
  volume={20},
  number={3},
  pages={273--297},
  year={1995}
}
```

## Further Reading

- Boser, B., Guyon, I., Vapnik, V. (1992). A training algorithm for optimal margin classifiers
- Vapnik, V. (1998). Statistical Learning Theory
- Platt, J. (1998). Sequential minimal optimization
