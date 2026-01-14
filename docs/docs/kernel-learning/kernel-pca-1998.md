---
sidebar_position: 1
title: "Schölkopf et al. (1998) - Nonlinear Component Analysis"
---

# Nonlinear Component Analysis as a Kernel Eigenvalue Problem

**Authors:** Bernhard Schölkopf, Alexander Smola, Klaus-Robert Müller  
**Published:** 1998  
**Journal:** Neural Computation  
**Link:** [PDF](https://www.face-rec.org/algorithms/Kernel/kernelPCA_scholkopf.pdf)

## Summary

This seminal paper introduces Kernel PCA, a nonlinear extension of Principal Component Analysis using the kernel trick. It demonstrates how to perform PCA in high-dimensional feature spaces without explicit computation of the mapping.

## Key Contributions

### 1. Kernel PCA Algorithm

Classical PCA finds principal components by computing eigenvectors of the covariance matrix. In feature space $\mathcal{F}$:

$$C = \frac{1}{n} \sum_{i=1}^n \phi(x_i) \phi(x_i)^\top$$

The eigenvector equation $Cv = \lambda v$ can be reformulated using only kernel evaluations.

### 2. The Dual Formulation

Principal components in feature space can be expressed as:
$$v = \sum_{i=1}^n \alpha_i \phi(x_i)$$

This leads to the eigenvalue problem:
$$\mathbf{K} \boldsymbol{\alpha} = n\lambda \boldsymbol{\alpha}$$

where $K_{ij} = K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$.

### 3. Projection onto Principal Components

For a new point $x$, projection onto the $k$-th principal component:
$$\langle v^k, \phi(x) \rangle = \sum_{i=1}^n \alpha_i^k K(x_i, x)$$

### 4. Centering in Feature Space

Since data may not be centered in feature space, the centered kernel matrix is:
$$\tilde{\mathbf{K}} = \mathbf{K} - \mathbf{1}_n \mathbf{K} - \mathbf{K} \mathbf{1}_n + \mathbf{1}_n \mathbf{K} \mathbf{1}_n$$

where $\mathbf{1}_n$ is the $n \times n$ matrix with all entries $1/n$.

## Algorithm

```python
def kernel_pca(X, kernel_func, n_components):
    n = len(X)
    
    # Compute kernel matrix
    K = np.array([[kernel_func(x_i, x_j) for x_j in X] for x_i in X])
    
    # Center the kernel matrix
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Normalize eigenvectors
    alphas = eigenvectors[:, :n_components]
    for i in range(n_components):
        alphas[:, i] /= np.sqrt(eigenvalues[i])
    
    return alphas, eigenvalues[:n_components]
```

## Theoretical Insights

### Connection to Kernel Methods

Kernel PCA reveals:
1. The intrinsic dimensionality of data in feature space
2. Nonlinear structure in the original space
3. Clustering structure via the leading components

### Eigenvalue Decay

The decay rate of eigenvalues indicates:
- **Fast decay**: Low effective dimensionality, good for compression
- **Slow decay**: High complexity, may need many components

### Universal Approximation

With a universal kernel (e.g., RBF), kernel PCA can approximate any continuous function's principal components.

## Applications

### 1. Nonlinear Dimensionality Reduction
Capture nonlinear manifold structure that linear PCA misses.

### 2. Feature Extraction
Extract informative nonlinear features for downstream tasks.

### 3. Denoising
Project noisy data onto principal components to remove noise.

### 4. Anomaly Detection
Reconstruction error in kernel PCA can detect outliers.

## Comparison with Linear PCA

| Aspect | Linear PCA | Kernel PCA |
|--------|-----------|------------|
| Feature space | Original | Implicit (RKHS) |
| Complexity | $O(d^2 n + d^3)$ | $O(n^2 d + n^3)$ |
| Captures | Linear correlations | Nonlinear relationships |
| Scalability | Scales with dimension | Scales with samples |

## Extensions

1. **Incremental Kernel PCA**: Online updates
2. **Sparse Kernel PCA**: Use subset of data points
3. **Supervised Kernel PCA**: Incorporate label information
4. **Kernel ICA**: Independent Component Analysis variant

## Citation

```bibtex
@article{scholkopf1998nonlinear,
  title={Nonlinear component analysis as a kernel eigenvalue problem},
  author={Sch{\"o}lkopf, Bernhard and Smola, Alexander and M{\"u}ller, Klaus-Robert},
  journal={Neural Computation},
  volume={10},
  number={5},
  pages={1299--1319},
  year={1998}
}
```

## Further Reading

- Mika, S., et al. (1999). Kernel PCA and de-noising in feature spaces
- Ham, J., et al. (2004). A kernel view of the dimensionality reduction of manifolds
