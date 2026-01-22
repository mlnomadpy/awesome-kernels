---
sidebar_position: 2
title: "Bach et al. (2004) - MKL, Conic Duality, and SMO"
---

# Multiple Kernel Learning, Conic Duality, and the SMO Algorithm

**Authors:** Francis R. Bach, Gert R. G. Lanckriet, Michael I. Jordan  
**Published:** 2004  
**Venue:** ICML  
**Link:** [PDF](https://dl.acm.org/doi/10.1145/1015330.1015424)

## Summary

This paper develops efficient algorithms for multiple kernel learning (MKL) by exploiting conic duality. It shows that the MKL problem can be reformulated as a second-order cone program (SOCP) and solved efficiently using an SMO-style algorithm.

## Key Contributions

### 1. Conic Duality Framework

The paper reformulates MKL using conic duality, revealing the structure:

**Primal problem:**
$$\min_{\mu, w, b} \quad \frac{1}{2}\sum_m \frac{\|w_m\|^2}{\mu_m} + C\sum_i \xi_i$$
$$\text{s.t.} \quad y_i(\sum_m \langle w_m, \phi_m(x_i)\rangle + b) \geq 1 - \xi_i$$
$$\quad\quad \sum_m \mu_m = 1, \quad \mu_m \geq 0$$

### 2. Second-Order Cone Formulation

The constraint $\sum_m \frac{\|w_m\|^2}{\mu_m} \leq t$ with $\sum_m \mu_m = 1$ is equivalent to:
$$\left\|\begin{pmatrix} w_1 \\ \vdots \\ w_M \end{pmatrix}\right\|_2 \leq \sqrt{t}$$

This is a second-order cone constraint!

### 3. Dual Problem

The dual of the MKL problem is:
$$\max_\alpha \quad \sum_i \alpha_i - \frac{1}{2}\max_m (y \circ \alpha)^T K_m (y \circ \alpha)$$
$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

Key insight: The dual involves the maximum over kernels, not a sum.

### 4. SMO Algorithm for MKL

The paper develops an efficient SMO-style algorithm:

```python
def mkl_smo(K_list, y, C, tol=1e-4):
    """
    SMO algorithm for Multiple Kernel Learning.
    """
    n = len(y)
    M = len(K_list)
    alpha = np.zeros(n)
    
    # Compute kernel-weighted combination
    def compute_mu(alpha):
        d = np.array([np.dot(y * alpha, K @ (y * alpha)) 
                      for K in K_list])
        d = np.sqrt(np.maximum(d, 0))
        if d.sum() > 0:
            return d / d.sum()
        return np.ones(M) / M
    
    # Main SMO loop
    while True:
        mu = compute_mu(alpha)
        K_mu = sum(mu[m] * K_list[m] for m in range(M))
        
        # Standard SMO iteration on combined kernel
        alpha_new = smo_step(K_mu, y, alpha, C)
        
        if converged(alpha, alpha_new, tol):
            break
        alpha = alpha_new
    
    return alpha, mu
```

## Mathematical Framework

### Group Lasso Connection

MKL is equivalent to group Lasso:
$$\min_{w_1,\ldots,w_M} \frac{1}{2}\left(\sum_m \|w_m\|\right)^2 + C\sum_i L(y_i, \sum_m \langle w_m, \phi_m(x_i)\rangle)$$

The $\ell_1/\ell_2$ regularization induces sparsity in kernel selection.

### Block Coordinate Descent

Alternating between:
1. **Fix $\mu$**: Solve standard SVM
2. **Fix $\alpha$**: Closed-form update for $\mu$

$$\mu_m = \frac{\sqrt{(y \circ \alpha)^T K_m (y \circ \alpha)}}{\sum_{m'} \sqrt{(y \circ \alpha)^T K_{m'} (y \circ \alpha)}}$$

### Convergence Analysis

The algorithm converges because:
1. Each subproblem is convex
2. Objective decreases monotonically
3. KKT conditions satisfied at convergence

## Complexity Analysis

| Method | Per-iteration | Memory |
|--------|--------------|--------|
| Interior point SDP | $O(M^3 + n^3)$ | $O(Mn^2)$ |
| SMO-MKL | $O(Mn^2)$ | $O(Mn^2)$ |
| Decomposition | $O(Mnp)$ | $O(Mp)$ |

where $p$ is working set size.

## Extensions

### 1. Non-sparse MKL

Replace $\ell_1$ norm with $\ell_p$ norm:
$$\sum_m \mu_m^p = 1$$

For $p > 1$, get dense kernel combinations.

### 2. Localized MKL

Allow sample-dependent weights:
$$K(x_i, x_j) = \sum_m \mu_m(x_i) \mu_m(x_j) K_m(x_i, x_j)$$

### 3. Two-Stage MKL

Learn kernels and classifier simultaneously or sequentially.

## Experimental Results

The paper shows:
1. **Efficiency**: SMO-MKL much faster than SDP
2. **Scalability**: Handles thousands of samples
3. **Accuracy**: Matches or exceeds fixed kernels
4. **Sparsity**: Automatically selects relevant kernels

## Impact on Machine Learning

This paper:
1. **Efficient algorithms**: Made MKL practical for large datasets
2. **Theoretical foundations**: Connected MKL to conic optimization
3. **Software**: Led to development of SHOGUN, SimpleMKL
4. **Applications**: Enabled kernel combination in many domains

## Citation

```bibtex
@inproceedings{bach2004multiple,
  title={Multiple kernel learning, conic duality, and the SMO algorithm},
  author={Bach, Francis R and Lanckriet, Gert RG and Jordan, Michael I},
  booktitle={International Conference on Machine Learning},
  pages={6--13},
  year={2004}
}
```

## Further Reading

- Rakotomamonjy, A., et al. (2008). SimpleMKL
- Sonnenburg, S., et al. (2006). Large scale multiple kernel learning
- Kloft, M., et al. (2011). $\ell_p$-norm multiple kernel learning
