---
sidebar_position: 2
title: "Rahimi & Recht (2007) - Random Features for Large-Scale Kernel Machines"
---

# Random Features for Large-Scale Kernel Machines

**Authors:** Ali Rahimi, Benjamin Recht  
**Published:** 2007  
**Venue:** NeurIPS (Test of Time Award 2017)  
**Link:** [PDF](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)

## Summary

This influential paper introduces Random Fourier Features (RFF), a method to approximate shift-invariant kernels with explicit low-dimensional feature maps. This enables linear methods to be applied with kernel-like performance at dramatically reduced computational cost.

## Key Contributions

### 1. Bochner's Theorem Connection

For a continuous shift-invariant kernel $K(x, y) = k(x - y)$, Bochner's theorem states:

$$k(\delta) = \int_{\mathbb{R}^d} p(\omega) e^{i\omega^\top \delta} d\omega$$

where $p(\omega)$ is a probability distribution (the spectral density).

### 2. Random Fourier Features

Approximate the kernel integral via Monte Carlo:

$$k(x - y) \approx \frac{1}{D} \sum_{j=1}^D e^{i\omega_j^\top (x-y)}$$

where $\omega_1, \ldots, \omega_D \sim p(\omega)$.

Using Euler's formula and symmetry:

$$z(x) = \sqrt{\frac{2}{D}} \left[\cos(\omega_1^\top x), \sin(\omega_1^\top x), \ldots, \cos(\omega_D^\top x), \sin(\omega_D^\top x) \right]^\top$$

Then: $K(x, y) \approx z(x)^\top z(y)$

### 3. Spectral Densities of Common Kernels

| Kernel | $k(\delta)$ | Spectral Density $p(\omega)$ |
|--------|-------------|------------------------------|
| Gaussian RBF | $e^{-\gamma\|\delta\|^2}$ | $\mathcal{N}(0, 2\gamma I)$ |
| Laplacian | $e^{-\gamma\|\delta\|_1}$ | Cauchy distribution |
| Matérn | $(1 + \sqrt{3}\gamma\|\delta\|)e^{-\sqrt{3}\gamma\|\delta\|}$ | Student-t |

## Algorithm

```python
import numpy as np

def random_fourier_features(X, D, gamma=1.0):
    """
    Compute Random Fourier Features for RBF kernel.
    
    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
    D : number of random features
    gamma : RBF kernel parameter
    
    Returns:
    --------
    Z : array of shape (n_samples, 2*D)
    """
    n_samples, n_features = X.shape
    
    # Sample frequencies from spectral density (Gaussian for RBF)
    omega = np.random.randn(n_features, D) * np.sqrt(2 * gamma)
    
    # Sample phases uniformly
    b = np.random.uniform(0, 2 * np.pi, D)
    
    # Compute features
    projection = X @ omega + b
    Z = np.sqrt(2.0 / D) * np.cos(projection)
    
    return Z, omega, b
```

## Theoretical Guarantees

### Approximation Error

With probability at least $1 - \delta$:
$$\sup_{x, y \in \mathcal{M}} |k(x-y) - z(x)^\top z(y)| \leq \epsilon$$

when $D = O\left(\frac{d}{\epsilon^2} \log\frac{1}{\delta}\right)$ for compact domain $\mathcal{M}$.

### Uniform Convergence

The approximation converges uniformly:
$$\mathbb{E}[z(x)^\top z(y)] = k(x - y)$$
$$\text{Var}[z(x)^\top z(y)] = O(1/D)$$

## Computational Benefits

| Method | Training | Prediction | Storage |
|--------|----------|------------|---------|
| Exact kernel | $O(n^3)$ | $O(n)$ | $O(n^2)$ |
| RFF | $O(nD^2)$ | $O(D)$ | $O(nD)$ |

For $D \ll n$, this is dramatically faster.

## Extensions and Variants

### 1. Orthogonal Random Features
Use orthogonal $\omega$ for lower variance:
```python
omega = np.linalg.qr(np.random.randn(max(n_features, D), D))[0][:n_features, :]
```

### 2. Fastfood (Le et al., 2013)
Use structured random matrices for $O(n D \log D)$ complexity.

### 3. Random Kitchen Sinks
The linear model with RFF:
$$f(x) = w^\top z(x)$$

trained with standard linear methods (SGD, closed-form).

### 4. Nyström Features
Alternative approximation using data-dependent features:
$$\phi(x) = \mathbf{K}_{nm}^{-1/2} [K(x, x_1), \ldots, K(x, x_m)]^\top$$

## Impact on Machine Learning

This paper enabled:
1. **Scalable kernel methods**: Train on millions of samples
2. **Online learning with kernels**: SGD with random features
3. **Deep learning connections**: Links between random features and neural networks
4. **Theoretical insights**: Understanding generalization in overparameterized models

## Test of Time Award (2017)

The NeurIPS Test of Time Award citation noted:
> "This paper introduced a simple yet powerful idea that has had lasting impact on the field."

The paper's influence extends to:
- Understanding neural network initialization
- Lazy training regime analysis
- Double descent phenomena
- Kernel approximation theory

## Citation

```bibtex
@inproceedings{rahimi2007random,
  title={Random features for large-scale kernel machines},
  author={Rahimi, Ali and Recht, Benjamin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={20},
  year={2007}
}
```

## Further Reading

- Le, Q., et al. (2013). Fastfood—approximating kernel expansions in loglinear time
- Yu, F., et al. (2016). Orthogonal random features
- Sutherland, D. & Schneider, J. (2015). On the error of random Fourier features
