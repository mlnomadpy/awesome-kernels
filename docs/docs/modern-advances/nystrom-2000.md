---
sidebar_position: 2
title: "Williams & Seeger (2000) - Nyström Approximation"
---

# Using the Nyström Method to Speed Up Kernel Machines

**Authors:** Christopher K.I. Williams, Matthias Seeger  
**Published:** 2000  
**Venue:** NeurIPS  

## Summary

This paper introduces the Nyström approximation for kernel matrices, enabling scalable kernel methods by approximating the full kernel matrix using a subset of landmark points. This technique is fundamental for large-scale kernel learning.

## Key Contributions

### 1. The Nyström Approximation

For a kernel matrix $\mathbf{K} \in \mathbb{R}^{n \times n}$, select $m \ll n$ landmark points.

Partition:
$$\mathbf{K} = \begin{pmatrix} \mathbf{K}_{mm} & \mathbf{K}_{mn} \\ \mathbf{K}_{nm} & \mathbf{K}_{nn} \end{pmatrix}$$

The Nyström approximation:
$$\tilde{\mathbf{K}} = \mathbf{K}_{nm} \mathbf{K}_{mm}^{-1} \mathbf{K}_{mn}$$

### 2. Low-Rank Interpretation

Equivalently, this is a rank-$m$ approximation:
$$\tilde{\mathbf{K}} = \mathbf{U}_m \mathbf{\Lambda}_m \mathbf{U}_m^\top$$

where $\mathbf{U}_m$ and $\mathbf{\Lambda}_m$ are approximated from the landmark points.

### 3. Explicit Feature Map

The Nyström method provides explicit features:
$$\phi_{nys}(x) = \mathbf{K}_{mm}^{-1/2} [K(x, z_1), \ldots, K(x, z_m)]^\top$$

such that:
$$\tilde{K}(x, y) = \phi_{nys}(x)^\top \phi_{nys}(y)$$

### 4. Eigenvalue Approximation

Landmark eigenvalues $\tilde{\lambda}_i$ approximate true eigenvalues:
$$\tilde{\lambda}_i = \frac{n}{m}\lambda_i^{(m)}$$

where $\lambda_i^{(m)}$ are eigenvalues of $\mathbf{K}_{mm}$.

## Algorithm

```python
import numpy as np
from scipy.linalg import sqrtm, inv

def nystrom_approximation(X, X_landmarks, kernel_func, reg=1e-6):
    """
    Compute Nyström approximation features.
    
    Parameters:
    -----------
    X : array of shape (n, d) - all data points
    X_landmarks : array of shape (m, d) - landmark points
    kernel_func : function - kernel function K(X1, X2)
    reg : float - regularization for numerical stability
    
    Returns:
    --------
    features : array of shape (n, m) - Nyström features
    """
    # Compute kernel matrices
    K_mm = kernel_func(X_landmarks, X_landmarks)
    K_nm = kernel_func(X, X_landmarks)
    
    # Add regularization
    K_mm_reg = K_mm + reg * np.eye(len(K_mm))
    
    # Compute K_mm^{-1/2}
    K_mm_sqrt_inv = inv(sqrtm(K_mm_reg))
    
    # Nyström features
    features = K_nm @ K_mm_sqrt_inv
    
    return features

def nystrom_kernel_regression(X_train, y_train, X_test, 
                               X_landmarks, kernel_func, 
                               lambda_reg=1.0):
    """
    Kernel ridge regression with Nyström approximation.
    """
    # Compute Nyström features
    phi_train = nystrom_approximation(X_train, X_landmarks, kernel_func)
    phi_test = nystrom_approximation(X_test, X_landmarks, kernel_func)
    
    # Linear regression in feature space
    m = phi_train.shape[1]
    A = phi_train.T @ phi_train + lambda_reg * np.eye(m)
    b = phi_train.T @ y_train
    
    w = np.linalg.solve(A, b)
    
    return phi_test @ w
```

## Landmark Selection Strategies

### 1. Uniform Sampling
Randomly select $m$ points from training data.
```python
indices = np.random.choice(n, m, replace=False)
landmarks = X[indices]
```

### 2. K-Means Clustering
Use cluster centers as landmarks:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=m).fit(X)
landmarks = kmeans.cluster_centers_
```

### 3. Leverage Score Sampling
Sample proportionally to statistical leverage:
$$p_i \propto K_{ii} - K_i^\top K^{-1} K_i$$

### 4. Greedy Selection
Iteratively select points maximizing coverage.

## Theoretical Guarantees

### Approximation Error

With uniform sampling, for $m = O(\sqrt{n}/\epsilon)$:
$$\|\mathbf{K} - \tilde{\mathbf{K}}\|_F \leq \epsilon \|\mathbf{K}\|_F$$

### Improved Bounds

With leverage score sampling:
$$\|\mathbf{K} - \tilde{\mathbf{K}}\|_2 \leq (1 + \epsilon) \|\mathbf{K} - \mathbf{K}_k\|_2$$

where $\mathbf{K}_k$ is the best rank-$k$ approximation.

### Generalization

Learning with Nyström features preserves generalization:
$$R[\tilde{f}] - R[f^*] = O\left(\frac{\sqrt{\|\mathbf{K} - \tilde{\mathbf{K}}\|_2}}{\sqrt{n}}\right)$$

## Computational Complexity

| Operation | Exact Kernel | Nyström |
|-----------|-------------|---------|
| Kernel matrix | $O(n^2)$ | $O(nm)$ |
| Training | $O(n^3)$ | $O(nm^2 + m^3)$ |
| Prediction | $O(n)$ | $O(m)$ |
| Storage | $O(n^2)$ | $O(nm)$ |

For $m \ll n$, significant speedup.

## Extensions

### 1. Improved Nyström

Multiple passes over data:
$$\tilde{\mathbf{K}}^{(2)} = \tilde{\mathbf{K}} \mathbf{K}_{mm}^{-1} \tilde{\mathbf{K}}$$

### 2. Ensemble Nyström

Average multiple Nyström approximations:
$$\tilde{\mathbf{K}}_{ensemble} = \frac{1}{T}\sum_{t=1}^T \tilde{\mathbf{K}}^{(t)}$$

### 3. Structured Nyström

Exploit structure in kernel (e.g., for graphs).

### 4. Memory Efficient

Streaming/online Nyström updates.

## Applications

1. **Large-scale SVM**: Train on millions of samples
2. **Gaussian Processes**: Scalable GP regression/classification
3. **Kernel PCA**: Approximate principal components
4. **Two-sample testing**: Fast MMD computation

## Connection to Other Methods

| Method | Type | Complexity |
|--------|------|------------|
| Nyström | Data-dependent | $O(nm^2)$ |
| Random Features | Data-independent | $O(nD^2)$ |
| Inducing Points | Variational | $O(nm^2)$ |

## Citation

```bibtex
@inproceedings{williams2000using,
  title={Using the Nystr{\"o}m method to speed up kernel machines},
  author={Williams, Christopher KI and Seeger, Matthias},
  booktitle={Advances in Neural Information Processing Systems},
  volume={13},
  year={2000}
}
```

## Further Reading

- Drineas, P. & Mahoney, M. (2005). On the Nyström method for approximating a Gram matrix
- Kumar, S., et al. (2012). Sampling methods for the Nyström method
- Zhang, K., et al. (2008). Improved Nyström low-rank approximation
