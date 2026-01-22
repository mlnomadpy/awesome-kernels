---
sidebar_position: 3
title: "Daniely et al. (2016) - Power of Initialization in Neural Networks"
---

# Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity

**Authors:** Amit Daniely, Roy Frostig, Yoram Singer  
**Published:** 2016  
**Venue:** NeurIPS  
**Link:** [PDF](https://proceedings.mlr.press/v49/daniely16.html)

## Summary

This paper establishes a fundamental connection between randomly initialized neural networks and kernel methods. It shows that the functions computed by random networks at initialization can be characterized by a kernel, providing theoretical foundations for understanding deep learning through the lens of kernel theory.

## Key Contributions

### 1. Random Network Kernel

For a randomly initialized network $f(x; \theta)$ with $\theta \sim \mathcal{N}(0, I)$:
$$K(x, x') = \mathbb{E}_\theta[f(x; \theta) f(x'; \theta)]$$

This kernel captures the correlations induced by random initialization.

### 2. Expressivity via Kernels

**Theorem:** A function $g$ can be approximated by a random network of width $m$ if and only if $g$ has small norm in the RKHS of $K$:
$$\|g\|_{\mathcal{H}_K} \leq R \implies \exists w: \mathbb{E}[\|f_w - g\|^2] \leq \frac{R^2}{m}$$

### 3. Depth Increases Expressivity

Deeper networks have more expressive kernels:
$$\mathcal{H}_{K^{(L)}} \supset \mathcal{H}_{K^{(L-1)}}$$

The kernel of a deeper network contains that of a shallower one.

## Mathematical Framework

### Network Model

Consider a fully-connected network:
$$f(x) = v^T \sigma(W^{(L)} \sigma(W^{(L-1)} \cdots \sigma(W^{(1)} x)))$$

with:
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ weight matrices
- $\sigma$: activation function (e.g., ReLU)
- $v \in \mathbb{R}^{n_L}$: output weights

### Covariance Kernel Construction

Define recursively:
$$\Sigma^{(0)}(x, x') = x^T x'$$
$$\Sigma^{(l)}(x, x') = \mathbb{E}_{u,v \sim \mathcal{N}(0, \Sigma^{(l-1)})}[\sigma(u)\sigma(v)]$$

The network kernel is:
$$K^{(L)}(x, x') = \Sigma^{(L)}(x, x')$$

### ReLU Activation

For ReLU $\sigma(z) = \max(0, z)$:
$$\Sigma^{(l)}(x, x') = \frac{1}{2\pi}\sqrt{\Sigma^{(l-1)}_{xx}\Sigma^{(l-1)}_{x'x'}}\left(\sin\theta^{(l-1)} + (\pi - \theta^{(l-1)})\cos\theta^{(l-1)}\right)$$

where $\theta^{(l)} = \arccos\left(\frac{\Sigma^{(l)}_{xx'}}{\sqrt{\Sigma^{(l)}_{xx}\Sigma^{(l)}_{x'x'}}}\right)$

## Key Results

### Theorem 1: Approximation Bound

If $g: \mathbb{R}^d \to \mathbb{R}$ satisfies $\|g\|_{\mathcal{H}_K} \leq R$, then there exist weights such that:
$$\mathbb{E}_{x \sim \mathcal{D}}[(f(x) - g(x))^2] \leq \frac{R^2}{m}$$

where $m$ is the width of the network.

### Theorem 2: Universality

For any continuous function $g$ on a compact domain:
$$\lim_{L \to \infty} \min_{h \in \mathcal{H}_{K^{(L)}}} \|h - g\|_{\infty} = 0$$

Deep networks are universal approximators.

### Theorem 3: Separation

There exist functions easy for deep networks but hard for shallow:
$$\|g\|_{\mathcal{H}_{K^{(L)}}} = O(1) \quad \text{but} \quad \|g\|_{\mathcal{H}_{K^{(1)}}} = \Omega(d^{L/2})$$

## Algorithm

```python
import numpy as np

def compute_random_network_kernel(X1, X2, depth, activation='relu'):
    """
    Compute the kernel corresponding to a random neural network.
    
    Parameters:
    -----------
    X1, X2 : arrays of shape (n1, d) and (n2, d)
    depth : number of hidden layers
    activation : 'relu' or 'erf'
    
    Returns:
    --------
    K : kernel matrix (n1, n2)
    """
    # Initialize with linear kernel
    Sigma = X1 @ X2.T
    diag1 = np.sum(X1**2, axis=1)
    diag2 = np.sum(X2**2, axis=1)
    
    for l in range(depth):
        # Normalize
        norm = np.sqrt(np.outer(diag1, diag2)) + 1e-10
        cos_theta = np.clip(Sigma / norm, -1, 1)
        theta = np.arccos(cos_theta)
        
        if activation == 'relu':
            # ReLU covariance
            Sigma_new = (1/(2*np.pi)) * norm * (
                np.sin(theta) + (np.pi - theta) * cos_theta
            )
        elif activation == 'erf':
            # Erf covariance (closed form)
            Sigma_new = (2/np.pi) * np.arcsin(
                Sigma / np.sqrt((1 + diag1[:, None]) * (1 + diag2[None, :]))
            )
        
        # Update for next layer
        Sigma = Sigma_new
        diag1 = np.diag(compute_random_network_kernel(X1, X1, 1, activation))
        diag2 = np.diag(compute_random_network_kernel(X2, X2, 1, activation))
        
    return Sigma

def random_network_kernel_efficient(X1, X2, depth):
    """Efficient implementation using recursion."""
    n1, n2 = len(X1), len(X2)
    
    # Track both cross-covariance and diagonals
    S = X1 @ X2.T
    S11 = np.sum(X1**2, axis=1)
    S22 = np.sum(X2**2, axis=1)
    
    for _ in range(depth):
        norm = np.sqrt(np.outer(S11, S22)) + 1e-10
        theta = np.arccos(np.clip(S / norm, -1, 1))
        
        # Update cross-covariance
        S = (1/(2*np.pi)) * norm * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
        
        # Update diagonals
        S11 = S11 / 2  # For normalized initialization
        S22 = S22 / 2
    
    return S
```

## Implications for Deep Learning

### 1. Initialization Matters

Random initialization implicitly defines a prior over functions:
- Good initialization = good implicit regularization
- Different initializations = different kernels

### 2. Depth Helps Optimization

Deeper networks have:
- More expressive kernels
- Better conditioning for gradient descent
- Implicit feature learning

### 3. Width vs. Depth Tradeoff

| Property | Wide Networks | Deep Networks |
|----------|--------------|---------------|
| Kernel | Fixed | Hierarchical |
| Approximation | $O(1/m)$ | Exponential in $L$ |
| Optimization | Lazy regime | Feature learning |

## Connection to Later Work

This paper laid groundwork for:
- **Neural Tangent Kernel** (Jacot et al., 2018)
- **Mean-field theory** (Mei et al., 2018)
- **Tensor programs** (Yang, 2019)

## Citation

```bibtex
@inproceedings{daniely2016toward,
  title={Toward deeper understanding of neural networks: The power of 
         initialization and a dual view on expressivity},
  author={Daniely, Amit and Frostig, Roy and Singer, Yoram},
  booktitle={Advances in Neural Information Processing Systems},
  volume={29},
  year={2016}
}
```

## Further Reading

- Neal, R. (1996). Priors for infinite networks
- Lee, J., et al. (2018). Deep neural networks as Gaussian processes
- Matthews, A., et al. (2018). Gaussian process behaviour in wide deep neural networks
