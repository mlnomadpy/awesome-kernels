---
sidebar_position: 3
title: "Cuturi (2013) - Sinkhorn Distances"
---

# Sinkhorn Distances: Lightspeed Computation of Optimal Transport

**Authors:** Marco Cuturi  
**Published:** 2013  
**Venue:** NeurIPS  
**Link:** [PDF](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)

## Summary

This paper introduces Sinkhorn distances, an efficient approximation to optimal transport (Wasserstein) distances using entropic regularization. The key insight is that adding entropy to the transport problem makes it solvable via simple matrix scaling iterations, enabling fast computation of distances between probability distributions.

## Key Contributions

### 1. Entropic Regularization

Add entropy term to optimal transport:
$$d_\lambda(r, c) = \min_{P \in U(r,c)} \langle P, M \rangle - \frac{1}{\lambda} h(P)$$

where $h(P) = -\sum_{ij} P_{ij} \log P_{ij}$ is the entropy.

### 2. Sinkhorn's Algorithm

The optimal $P$ has the form $P = \text{diag}(u) K \text{diag}(v)$ where $K = e^{-\lambda M}$.

Iterate until convergence:
$$u \leftarrow r \oslash (Kv), \quad v \leftarrow c \oslash (K^T u)$$

### 3. Computational Speedup

Sinkhorn complexity: $O(n^2 / \epsilon^2)$ vs. $O(n^3 \log n)$ for exact OT.

For large $n$, this is orders of magnitude faster.

## Mathematical Framework

### Optimal Transport Problem

Given:
- Source distribution $r \in \Delta_n$ (probability simplex)
- Target distribution $c \in \Delta_m$
- Cost matrix $M \in \mathbb{R}^{n \times m}$

Find optimal transport plan:
$$W(r, c) = \min_{P \in U(r,c)} \langle P, M \rangle$$

where $U(r,c) = \{P \geq 0 : P\mathbf{1} = r, P^T\mathbf{1} = c\}$.

### Entropic OT

The regularized problem:
$$W_\lambda(r, c) = \min_{P \in U(r,c)} \langle P, M \rangle + \frac{1}{\lambda} \text{KL}(P \| rc^T)$$

### Optimal Solution Structure

**Theorem:** The optimal $P^*$ satisfies:
$$P^*_{ij} = u_i K_{ij} v_j$$

where $K_{ij} = e^{-\lambda M_{ij}}$ and $(u, v)$ satisfy:
$$u \odot (Kv) = r, \quad v \odot (K^T u) = c$$

## Sinkhorn Algorithm

### Basic Iteration

```python
import numpy as np

def sinkhorn_distance(M, r, c, lambda_reg=1.0, max_iter=100, tol=1e-9):
    """
    Compute Sinkhorn distance using matrix scaling.
    
    Parameters:
    -----------
    M : cost matrix (n, m)
    r : source distribution (n,)
    c : target distribution (m,)
    lambda_reg : regularization strength
    max_iter : maximum iterations
    tol : convergence tolerance
    
    Returns:
    --------
    distance : Sinkhorn distance
    P : optimal transport plan
    """
    n, m = M.shape
    
    # Gibbs kernel
    K = np.exp(-lambda_reg * M)
    
    # Initialize scaling vectors
    u = np.ones(n)
    v = np.ones(m)
    
    for iteration in range(max_iter):
        u_prev = u.copy()
        
        # Sinkhorn iterations
        u = r / (K @ v + 1e-10)
        v = c / (K.T @ u + 1e-10)
        
        # Check convergence
        if np.max(np.abs(u - u_prev)) < tol:
            break
    
    # Optimal transport plan
    P = np.diag(u) @ K @ np.diag(v)
    
    # Sinkhorn distance
    distance = np.sum(P * M)
    
    return distance, P

def sinkhorn_log_stabilized(M, r, c, lambda_reg=1.0, max_iter=100, tol=1e-9):
    """
    Log-stabilized Sinkhorn for numerical stability.
    """
    n, m = M.shape
    
    # Work in log space
    log_r = np.log(r + 1e-10)
    log_c = np.log(c + 1e-10)
    
    f = np.zeros(n)  # log(u)
    g = np.zeros(m)  # log(v)
    
    for iteration in range(max_iter):
        f_prev = f.copy()
        
        # Log-space iterations
        f = log_r - logsumexp(-lambda_reg * M + g[None, :], axis=1)
        g = log_c - logsumexp(-lambda_reg * M.T + f[None, :], axis=1)
        
        if np.max(np.abs(f - f_prev)) < tol:
            break
    
    # Compute distance
    log_P = f[:, None] + g[None, :] - lambda_reg * M
    P = np.exp(log_P)
    distance = np.sum(P * M)
    
    return distance, P

def logsumexp(x, axis=None):
    """Numerically stable logsumexp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(x - x_max), axis=axis))
    return result + np.squeeze(x_max, axis=axis) if axis is not None else result + x_max.item()
```

### GPU-Accelerated Version

```python
import torch

def sinkhorn_gpu(M, r, c, lambda_reg=1.0, max_iter=100):
    """
    GPU-accelerated Sinkhorn using PyTorch.
    """
    K = torch.exp(-lambda_reg * M)
    u = torch.ones_like(r)
    
    for _ in range(max_iter):
        u = r / (K @ (c / (K.T @ u)))
    
    v = c / (K.T @ u)
    P = torch.diag(u) @ K @ torch.diag(v)
    
    return torch.sum(P * M), P
```

## Sinkhorn Divergence

To remove entropic bias, use Sinkhorn divergence:
$$S_\lambda(r, c) = W_\lambda(r, c) - \frac{1}{2}W_\lambda(r, r) - \frac{1}{2}W_\lambda(c, c)$$

This satisfies $S_\lambda(r, r) = 0$ (unlike raw Sinkhorn distance).

## Sinkhorn Kernel

Define a kernel on probability distributions:
$$K_\lambda(r, c) = e^{-\gamma S_\lambda(r, c)}$$

This is **positive definite** for appropriate $\gamma$.

## Theoretical Properties

### Approximation Quality

$$|W_\lambda(r, c) - W(r, c)| \leq \frac{\log(\min(n,m))}{\lambda}$$

The approximation improves as $\lambda \to \infty$.

### Convergence Rate

Sinkhorn converges linearly:
$$\|P^{(k)} - P^*\|_1 \leq C \cdot \rho^k$$

where $\rho = \exp(-2\lambda \min_{i,j} M_{ij}) < 1$.

### Differentiability

$W_\lambda$ is differentiable w.r.t. $r$, $c$, and $M$:
$$\nabla_r W_\lambda = f, \quad \nabla_c W_\lambda = g$$

where $(f, g)$ are the optimal dual variables.

## Applications

### 1. Domain Adaptation

Align source and target distributions:
$$\min_\theta W_\lambda(p_{source}, p_\theta(target))$$

### 2. Generative Models

Wasserstein GAN with Sinkhorn:
$$\min_G \max_D W_\lambda(p_{data}, p_G)$$

### 3. Document Comparison

Word Mover's Distance:
$$WMD(d_1, d_2) = W(bow(d_1), bow(d_2))$$

with word embedding distances as cost.

### 4. Color Transfer

Transport color histogram from source to target image.

### 5. Shape Matching

Compare point clouds or probability measures on shapes.

## Complexity Comparison

| Method | Time | Space | Differentiable |
|--------|------|-------|----------------|
| LP Solver | $O(n^3 \log n)$ | $O(n^2)$ | ✗ |
| Network Simplex | $O(n^3)$ | $O(n^2)$ | ✗ |
| Sinkhorn | $O(n^2 / \epsilon)$ | $O(n^2)$ | ✓ |
| Sliced OT | $O(n \log n)$ | $O(n)$ | ✓ |

## Extensions

### 1. Unbalanced OT

Allow mass creation/destruction:
$$\min_P \langle P, M \rangle + \text{KL}(P\mathbf{1}|r) + \text{KL}(P^T\mathbf{1}|c)$$

### 2. Gromov-Wasserstein

For comparing metric measure spaces:
$$GW(X, Y) = \min_P \sum_{ijkl} |d_X(i,j) - d_Y(k,l)|^2 P_{ik} P_{jl}$$

### 3. Multi-Marginal OT

Transport between $K > 2$ distributions simultaneously.

## Citation

```bibtex
@inproceedings{cuturi2013sinkhorn,
  title={Sinkhorn distances: Lightspeed computation of optimal transport},
  author={Cuturi, Marco},
  booktitle={Advances in Neural Information Processing Systems},
  volume={26},
  year={2013}
}
```

## Further Reading

- Peyré, G. & Cuturi, M. (2019). Computational optimal transport
- Genevay, A., et al. (2018). Learning generative models with Sinkhorn divergences
- Feydy, J., et al. (2019). Interpolating between optimal transport and MMD
