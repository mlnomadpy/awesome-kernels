---
sidebar_position: 1
title: "Arora et al. (2019) - Exact Computation with Infinitely Wide Neural Net"
---

# On Exact Computation with an Infinitely Wide Neural Network

**Authors:** Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov  
**Published:** 2019  
**Venue:** NeurIPS  
**Link:** [arXiv](https://arxiv.org/abs/1904.11955)

## Summary

This paper provides efficient algorithms for computing the Neural Tangent Kernel (NTK) and Conjugate Kernel (CK) for various architectures. It shows that infinite-width networks can be simulated exactly using kernel methods, enabling large-scale experiments comparing neural networks to their kernel counterparts.

## Key Contributions

### 1. Efficient NTK Computation

The paper derives closed-form recursive formulas for computing NTK for:
- Fully-connected networks
- Convolutional networks (CNTK)
- Residual networks

### 2. Conjugate Kernel (Neural Network GP)

The Conjugate Kernel corresponds to the prior over functions:
$$\Sigma^{(L)}(x, x') = \mathbb{E}_{f \sim \text{NN}}[f(x)f(x')]$$

This is the kernel of the Gaussian Process that infinite-width networks converge to at initialization.

### 3. Empirical Findings

Key experimental results:
- NTK closely matches finite-width networks
- CNTK for CNNs achieves state-of-the-art among kernel methods on CIFAR-10
- Gap between NTK and trained networks indicates feature learning

## Mathematical Framework

### NTK Recursion

For a depth-$L$ fully-connected network, the NTK is computed recursively:

**Base case:**
$$\Sigma^{(0)}(x, x') = x^T x'$$
$$\Theta^{(0)}(x, x') = \Sigma^{(0)}(x, x')$$

**Recursive case:**
$$\Sigma^{(l)}(x, x') = c_\sigma \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Lambda^{(l-1)})}[\sigma(u)\sigma(v)]$$

$$\dot{\Sigma}^{(l)}(x, x') = c_\sigma \mathbb{E}_{(u,v) \sim \mathcal{N}(0, \Lambda^{(l-1)})}[\sigma'(u)\sigma'(v)]$$

$$\Theta^{(l)}(x, x') = \Sigma^{(l)}(x, x') + \Theta^{(l-1)}(x, x') \cdot \dot{\Sigma}^{(l)}(x, x')$$

where $\Lambda^{(l-1)} = \begin{pmatrix} \Sigma^{(l-1)}(x,x) & \Sigma^{(l-1)}(x,x') \\ \Sigma^{(l-1)}(x',x) & \Sigma^{(l-1)}(x',x') \end{pmatrix}$

### ReLU Specific Formulas

For ReLU activation:
$$\Sigma^{(l)} = \frac{1}{2\pi}\sqrt{\Sigma^{(l-1)}_{xx} \Sigma^{(l-1)}_{x'x'}}(\sin\theta + (\pi-\theta)\cos\theta)$$

$$\dot{\Sigma}^{(l)} = \frac{1}{2\pi}(\pi - \theta)$$

where $\theta = \arccos\left(\frac{\Sigma^{(l-1)}_{xx'}}{\sqrt{\Sigma^{(l-1)}_{xx}\Sigma^{(l-1)}_{x'x'}}}\right)$

## Convolutional NTK (CNTK)

### Patch-Based Formulation

For image input $x$ with patches $\{p_i\}$:
$$\Sigma^{(0)}(x, x') = \frac{1}{P}\sum_{i=1}^P p_i^T p_i'$$

Apply the same recursion but with local connectivity.

### Global Average Pooling

Final layer uses global average pooling:
$$\Theta^{(L)}_{GAP}(x, x') = \frac{1}{H \cdot W}\sum_{h,w} \Theta^{(L)}_{h,w}(x, x')$$

## Algorithm

```python
import numpy as np
from scipy.special import erf

def compute_ntk_relu(X1, X2, depth):
    """
    Compute NTK for fully-connected ReLU network.
    
    Parameters:
    -----------
    X1, X2 : arrays (n1, d) and (n2, d)
    depth : number of layers
    
    Returns:
    --------
    Theta : NTK matrix (n1, n2)
    Sigma : Conjugate kernel matrix (n1, n2)
    """
    n1, n2 = len(X1), len(X2)
    
    # Initialize
    Sigma = X1 @ X2.T / X1.shape[1]
    Sigma_11 = np.sum(X1**2, axis=1) / X1.shape[1]
    Sigma_22 = np.sum(X2**2, axis=1) / X1.shape[1]
    
    Theta = Sigma.copy()
    
    for l in range(depth):
        # Compute angle
        norm = np.sqrt(np.outer(Sigma_11, Sigma_22)) + 1e-10
        cos_theta = np.clip(Sigma / norm, -1, 1)
        theta = np.arccos(cos_theta)
        
        # Derivative kernel
        Sigma_dot = (np.pi - theta) / (2 * np.pi)
        
        # Update Sigma
        Sigma_new = (1 / (2 * np.pi)) * norm * (
            np.sin(theta) + (np.pi - theta) * cos_theta
        )
        
        # Update Theta (NTK recursion)
        Theta = Sigma_new + Theta * Sigma_dot
        
        # Update diagonals for next iteration
        Sigma = Sigma_new
        Sigma_11 = np.diag(Sigma_new) if n1 == n2 else compute_diag(X1, l+1)
        Sigma_22 = np.diag(Sigma_new) if n1 == n2 else compute_diag(X2, l+1)
    
    return Theta, Sigma

def ntk_predict(K_train, K_test, y_train, reg=1e-6):
    """
    Make predictions using NTK.
    
    Equivalent to infinite-width neural network trained to convergence.
    """
    alpha = np.linalg.solve(K_train + reg * np.eye(len(K_train)), y_train)
    return K_test @ alpha

def compute_cntk(images1, images2, depth, filter_size=3):
    """
    Compute Convolutional NTK.
    
    Parameters:
    -----------
    images1, images2 : arrays (n1, H, W, C) and (n2, H, W, C)
    depth : number of convolutional layers
    filter_size : size of convolutional filter
    """
    # Implementation involves convolution-like operations
    # on the kernel matrices
    pass  # Full implementation in paper
```

## Experimental Results

### CIFAR-10 Classification

| Method | Test Accuracy |
|--------|--------------|
| Gaussian kernel | 55% |
| Polynomial kernel | 62% |
| Arc-cosine kernel | 66% |
| NTK (FC) | 64% |
| **CNTK** | **77%** |
| ResNet-34 | 95% |

### Key Observations

1. **CNTK outperforms all prior kernels** on CIFAR-10
2. **Gap to trained CNNs**: ~18% indicates importance of feature learning
3. **NTK ≈ trained network** for overparameterized regime

## Implications

### Understanding Neural Networks

1. **Lazy training**: Wide networks barely move from initialization
2. **Feature learning**: Finite-width networks learn better features
3. **Architecture matters**: CNTK vs NTK shows importance of convolutions

### Practical Applications

1. **Kernel design**: NTK provides principled kernel construction
2. **Infinite ensembles**: Average over all random initializations
3. **Theoretical analysis**: Rigorous tools for studying deep learning

## Software

The paper released the **Neural Tangents** library:
```python
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512), stax.Relu(),
    stax.Dense(512), stax.Relu(),
    stax.Dense(10)
)

# Compute NTK
ntk = kernel_fn(X1, X2, 'ntk')
# Compute NNGP kernel
nngp = kernel_fn(X1, X2, 'nngp')
```

## Citation

```bibtex
@inproceedings{arora2019exact,
  title={On exact computation with an infinitely wide neural net},
  author={Arora, Sanjeev and Du, Simon S and Hu, Wei and Li, Zhiyuan 
          and Salakhutdinov, Ruslan and Wang, Ruosong},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

## Further Reading

- Jacot, A., et al. (2018). Neural tangent kernel
- Lee, J., et al. (2019). Wide neural networks evolve as linear models
- Novak, R., et al. (2020). Neural tangents: Fast and easy infinite neural networks in Python
