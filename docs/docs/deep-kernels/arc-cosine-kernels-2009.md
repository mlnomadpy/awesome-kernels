---
sidebar_position: 1
title: "Cho & Saul (2009) - Kernel Methods for Deep Learning"
---

# Kernel Methods for Deep Learning

**Authors:** Youngmin Cho, Lawrence K. Saul  
**Published:** 2009  
**Venue:** NeurIPS  
**Link:** [PDF](https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html)

## Summary

This paper introduces arc-cosine kernels, which correspond to the computation in multilayer neural networks. These kernels provide a bridge between deep learning and kernel methods, showing how to capture hierarchical representations in a kernel framework.

## Key Contributions

### 1. Arc-Cosine Kernels

Define a family of kernels mimicking neural network layers:
$$k_n(x, x') = \frac{1}{\pi}\|x\|^n \|x'\|^n J_n(\theta)$$

where $\theta = \cos^{-1}\left(\frac{x \cdot x'}{\|x\|\|x'\|}\right)$ and:

$$J_n(\theta) = (-1)^n (\sin\theta)^{2n+1} \left(\frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\right)^n \left(\frac{\pi - \theta}{\sin\theta}\right)$$

### 2. Correspondence to Neural Networks

**Single layer:**
$$k_n(x, x') = 2 \int \Theta(w \cdot x) \Theta(w \cdot x') (w \cdot x)^n (w \cdot x')^n \frac{e^{-\|w\|^2/2}}{(2\pi)^{d/2}} dw$$

where $\Theta$ is the Heaviside step function.

This is the expected dot product of hidden representations after ReLU-like activations.

### 3. Multilayer Composition

Stack kernels by recursive composition:
$$k^{(l+1)}(x, x') = \frac{1}{\pi}\sqrt{k^{(l)}(x,x) k^{(l)}(x',x')} J_n(\theta^{(l)})$$

where $\theta^{(l)} = \cos^{-1}\left(\frac{k^{(l)}(x,x')}{\sqrt{k^{(l)}(x,x)k^{(l)}(x',x')}}\right)$

## Special Cases

### Order $n=0$ (Step Function)

$$k_0(x, x') = 1 - \frac{\theta}{\pi}$$

Corresponds to:
- Activation: Heaviside $\Theta(z)$
- Linear separator in feature space

### Order $n=1$ (ReLU)

$$k_1(x, x') = \frac{1}{\pi}\|x\|\|x'\| \left(\sin\theta + (\pi - \theta)\cos\theta\right)$$

Corresponds to:
- Activation: ReLU $\max(0, z)$
- Most commonly used in deep learning

### Order $n=2$ (Squared ReLU)

$$k_2(x, x') = \frac{1}{\pi}\|x\|^2\|x'\|^2 \left(3\sin\theta\cos\theta + (\pi-\theta)(1+2\cos^2\theta)\right)$$

Corresponds to:
- Activation: $\max(0, z)^2$

## Algorithm

```python
import numpy as np

def arc_cosine_kernel(X1, X2, order=1, depth=1):
    """
    Compute arc-cosine kernel.
    
    Parameters:
    -----------
    X1, X2 : arrays of shape (n1, d) and (n2, d)
    order : n parameter (0, 1, or 2)
    depth : number of layers
    
    Returns:
    --------
    K : kernel matrix (n1, n2)
    """
    # Initialize with linear kernel
    K = X1 @ X2.T
    norm1 = np.sqrt(np.sum(X1**2, axis=1, keepdims=True))
    norm2 = np.sqrt(np.sum(X2**2, axis=1))
    
    for _ in range(depth):
        # Compute angles
        cos_theta = K / (norm1 @ norm2[np.newaxis, :] + 1e-10)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        
        # Compute J_n(theta)
        if order == 0:
            J = 1 - theta / np.pi
        elif order == 1:
            J = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
        elif order == 2:
            J = (3 * np.sin(theta) * cos_theta + 
                 (np.pi - theta) * (1 + 2 * cos_theta**2)) / np.pi
        
        # Update kernel and norms
        K_new = (norm1 ** order) @ ((norm2 ** order)[np.newaxis, :]) * J
        
        # Update norms for next layer
        norm1 = np.sqrt(np.diag(K_new))[:, np.newaxis]
        norm2 = np.sqrt(np.diag(K_new))
        K = K_new
    
    return K

def multilayer_arc_cosine(X1, X2, order=1, depth=3):
    """Convenience function for deep arc-cosine kernel."""
    return arc_cosine_kernel(X1, X2, order=order, depth=depth)
```

## Theoretical Properties

### Positive Definiteness

**Theorem:** Arc-cosine kernels $k_n$ are positive definite for all $n \geq 0$.

Proof: They arise as inner products of random features with Gaussian weights.

### Universal Approximation

The composed kernel $k^{(L)}$ becomes increasingly discriminative:
- As $L \to \infty$, kernel becomes a delta function
- Finite depth gives tradeoff between smoothness and discrimination

### Eigenvalue Decay

Eigenvalues of arc-cosine kernel matrices decay polynomially:
$$\lambda_j \sim j^{-\alpha}$$

where $\alpha$ depends on input dimension and order.

## Comparison with Neural Networks

| Property | Arc-Cosine Kernel | Neural Network |
|----------|------------------|----------------|
| Features | Infinite | Finite width |
| Weights | Integrated out | Learned |
| Training | Kernel methods | Backpropagation |
| Depth effect | Composition | Layer stacking |

### Key Difference

Arc-cosine kernels correspond to **random** neural networks where weights are not learned, just integrated over.

## Applications

1. **Deep kernel SVM**: Use composed kernel in SVM
2. **Feature analysis**: Study representations without training
3. **Initialization theory**: Understanding random networks
4. **Kernel design**: Principled construction of deep kernels

## Connection to NTK

Arc-cosine kernels are related to but different from Neural Tangent Kernels:
- Arc-cosine: Dot product of hidden representations
- NTK: Dot product of gradients

At infinite width, both converge to deterministic kernels.

## Citation

```bibtex
@inproceedings{cho2009kernel,
  title={Kernel methods for deep learning},
  author={Cho, Youngmin and Saul, Lawrence K},
  booktitle={Advances in Neural Information Processing Systems},
  volume={22},
  year={2009}
}
```

## Further Reading

- Neal, R. (1996). Priors for infinite networks
- Williams, C. K. I. (1997). Computing with infinite networks
- Jacot, A., et al. (2018). Neural tangent kernel
