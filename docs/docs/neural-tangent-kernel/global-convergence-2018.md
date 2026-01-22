---
sidebar_position: 2
title: "Chizat & Bach (2018) - Global Convergence for Over-Parameterized Models"
---

# On the Global Convergence of Gradient Descent for Over-Parameterized Models Using Optimal Transport

**Authors:** Lénaïc Chizat, Francis R. Bach  
**Published:** 2018  
**Venue:** NeurIPS  
**Link:** [arXiv](https://arxiv.org/abs/1805.00915)

## Summary

This paper provides a theoretical framework for understanding the global convergence of gradient descent in over-parameterized neural networks. Using optimal transport theory, it shows that gradient descent on two-layer networks converges to global optima when the number of neurons tends to infinity.

## Key Contributions

### 1. Mean-Field Limit

In the infinite-width limit, the distribution of neurons $\rho$ evolves according to:
$$\partial_t \rho = \nabla \cdot (\rho \nabla V_\rho)$$

where $V_\rho(w) = \int L'(f_\rho(x), y) \sigma(w \cdot x) d\mathcal{D}(x, y)$ is the gradient field.

### 2. Global Convergence Result

**Theorem:** For two-layer networks with infinite width, gradient flow converges to a global minimizer under mild conditions on the activation function and data distribution.

### 3. Lazy vs. Active Regime

The paper distinguishes:
- **Lazy regime**: Weights barely move, NTK applies
- **Active regime**: Weights evolve significantly, mean-field applies

## Mathematical Framework

### Two-Layer Network

Consider a network:
$$f_m(x) = \frac{1}{m}\sum_{j=1}^m a_j \sigma(w_j \cdot x)$$

with $m$ neurons, output weights $a_j$, and hidden weights $w_j$.

### Empirical Measure

Represent the network as a probability measure:
$$\rho_m = \frac{1}{m}\sum_{j=1}^m \delta_{(a_j, w_j)}$$

The function becomes:
$$f_{\rho_m}(x) = \int a \cdot \sigma(w \cdot x) d\rho_m(a, w)$$

### Loss Functional

The optimization objective:
$$\mathcal{R}(\rho) = \int L(f_\rho(x), y) d\mathcal{D}(x, y)$$

### Wasserstein Gradient Flow

The mean-field dynamics is the Wasserstein gradient flow:
$$\partial_t \rho_t = \nabla \cdot \left(\rho_t \nabla \frac{\delta \mathcal{R}}{\delta \rho}(\rho_t)\right)$$

## Key Results

### Theorem 1: Global Convergence

Under assumptions:
1. Activation $\sigma$ is smooth and non-polynomial
2. Data distribution has full support
3. Loss is convex and smooth

Then: $\mathcal{R}(\rho_t) \to \inf_\rho \mathcal{R}(\rho)$ as $t \to \infty$.

### Theorem 2: No Spurious Local Minima

Every local minimum of $\mathcal{R}$ in the space of probability measures is a global minimum.

### Theorem 3: Finite-Width Approximation

For width $m$ and time $T$:
$$|\mathcal{R}(\rho_m^{(T)}) - \mathcal{R}(\rho_\infty^{(T)})| \leq O\left(\frac{1}{\sqrt{m}}\right)$$

## Lazy vs. Active Regimes

### Lazy Regime (NTK)

When initialization scale is large:
- Weights move by $O(1/\sqrt{m})$
- Function changes by $O(1)$ 
- NTK is approximately constant
- Kernel regression behavior

### Active Regime (Mean-Field)

When initialization scale is small:
- Weights move by $O(1)$
- Features adapt to the problem
- Mean-field theory applies
- True feature learning

### Scaling Analysis

| Initialization | Weight change | Regime |
|---------------|---------------|--------|
| $O(1)$ | $O(1/\sqrt{m})$ | Lazy/NTK |
| $O(1/\sqrt{m})$ | $O(1)$ | Active/Mean-field |

## Algorithm

```python
import numpy as np
from scipy.special import expit as sigmoid

class MeanFieldTwoLayerNN:
    """
    Two-layer neural network in the mean-field parameterization.
    """
    def __init__(self, n_particles, input_dim, activation='relu'):
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.activation = activation
        
        # Initialize particles (a, w) uniformly
        self.a = np.random.randn(n_particles)  # Output weights
        self.w = np.random.randn(n_particles, input_dim)  # Hidden weights
        
    def forward(self, X):
        """Compute network output."""
        if self.activation == 'relu':
            h = np.maximum(0, X @ self.w.T)
        elif self.activation == 'sigmoid':
            h = sigmoid(X @ self.w.T)
        return h @ self.a / self.n_particles
    
    def compute_gradients(self, X, y):
        """Compute gradients for all particles."""
        n = len(X)
        
        # Forward pass
        if self.activation == 'relu':
            h = np.maximum(0, X @ self.w.T)  # (n, m)
            h_grad = (X @ self.w.T > 0).astype(float)
        
        # Prediction and residual
        pred = h @ self.a / self.n_particles
        residual = pred - y  # (n,)
        
        # Gradients
        grad_a = h.T @ residual / (n * self.n_particles)  # (m,)
        grad_w = np.einsum('ni,n,nj->ij', h_grad, residual, X) * self.a[:, None]
        grad_w /= (n * self.n_particles)
        
        return grad_a, grad_w
    
    def train_gradient_descent(self, X, y, lr=0.01, n_steps=1000):
        """Train using gradient descent."""
        losses = []
        
        for step in range(n_steps):
            pred = self.forward(X)
            loss = np.mean((pred - y)**2) / 2
            losses.append(loss)
            
            grad_a, grad_w = self.compute_gradients(X, y)
            
            self.a -= lr * grad_a
            self.w -= lr * grad_w
            
        return losses

def wasserstein_distance(rho1, rho2):
    """
    Compute 2-Wasserstein distance between particle distributions.
    Uses linear assignment for discrete distributions.
    """
    from scipy.optimize import linear_sum_assignment
    
    # Cost matrix
    C = np.sum((rho1[:, None, :] - rho2[None, :, :])**2, axis=2)
    
    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(C)
    
    return np.sqrt(C[row_ind, col_ind].sum() / len(rho1))
```

## Theoretical Implications

### 1. Landscape Analysis

The mean-field loss landscape has:
- No spurious local minima
- Connected level sets
- Benign saddle points

### 2. Implicit Regularization

Mean-field gradient descent implicitly regularizes:
$$\rho_t \to \arg\min_\rho \mathcal{R}(\rho) + \text{entropy}(\rho)$$

### 3. Comparison to Convex Optimization

| Property | Convex | Mean-Field |
|----------|--------|------------|
| Local = Global | ✓ | ✓ |
| Unique solution | ✓ | ✗ |
| Rate guarantee | ✓ | Partial |

## Extensions

### 1. Multi-Layer Networks

The framework extends to deeper networks using:
- Layer-wise mean fields
- Coupled transport equations

### 2. Regularization

Adding regularization modifies the gradient flow:
$$\partial_t \rho = \nabla \cdot \left(\rho \nabla \left(\frac{\delta \mathcal{R}}{\delta \rho} + \lambda \log \rho\right)\right)$$

### 3. Discrete Time

SGD approximates the continuous flow with:
$$\rho_{k+1} = (I - \eta \nabla^2 V)_\# \rho_k$$

## Citation

```bibtex
@inproceedings{chizat2018global,
  title={On the global convergence of gradient descent for over-parameterized 
         models using optimal transport},
  author={Chizat, L{\'e}na{\"\i}c and Bach, Francis},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```

## Further Reading

- Mei, S., et al. (2018). A mean field view of the landscape of two-layer neural networks
- Rotskoff, G. & Vanden-Eijnden, E. (2018). Neural networks as interacting particle systems
- Sirignano, J. & Spiliopoulos, K. (2020). Mean field analysis of neural networks
