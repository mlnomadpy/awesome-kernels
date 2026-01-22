---
sidebar_position: 2
title: "Wilson et al. (2016) - Deep Kernel Learning"
---

# Deep Kernel Learning

**Authors:** Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, Eric P. Xing  
**Published:** 2016  
**Venue:** AISTATS  
**Link:** [PDF](https://proceedings.mlr.press/v51/wilson16.html)

## Summary

This paper introduces Deep Kernel Learning (DKL), which combines the representational power of deep neural networks with the uncertainty quantification of Gaussian Processes. The key idea is to use a neural network to learn features, then apply a GP on top of these learned features.

## Key Contributions

### 1. Deep Kernel Architecture

Define a deep kernel:
$$k_{DK}(x, x') = k_{base}(g(x; w), g(x'; w))$$

where:
- $g(x; w)$: Deep neural network with parameters $w$
- $k_{base}$: Base kernel (e.g., RBF) on learned features

### 2. End-to-End Training

Train both the neural network $g$ and GP hyperparameters by maximizing the log marginal likelihood:
$$\log p(y | X, w, \theta) = -\frac{1}{2}y^T K_{DK}^{-1} y - \frac{1}{2}\log|K_{DK}| - \frac{n}{2}\log 2\pi$$

### 3. Scalable Inference

Use inducing points and KISS-GP for scalability:
$$K \approx W K_{UU} W^T$$

where $K_{UU}$ is kernel on inducing points and $W$ is interpolation matrix.

## Architecture

### Feature Extractor

The neural network $g: \mathbb{R}^d \to \mathbb{R}^h$ maps inputs to feature space:
```
Input → Conv/FC layers → BatchNorm → ReLU → ... → h-dim features
```

### Base Kernel

Typically use RBF kernel on features:
$$k_{base}(g(x), g(x')) = \sigma^2 \exp\left(-\frac{\|g(x) - g(x')\|^2}{2\ell^2}\right)$$

With Automatic Relevance Determination (ARD):
$$k_{ARD}(g(x), g(x')) = \sigma^2 \exp\left(-\sum_i \frac{(g_i(x) - g_i(x'))^2}{2\ell_i^2}\right)$$

## Training Algorithm

```python
import torch
import torch.nn as nn
import gpytorch

class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF kernel on learned features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # GP on features
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, x):
        return self.network(x)

def train_dkl(model, likelihood, train_x, train_y, epochs=100, lr=0.01):
    """Train Deep Kernel Learning model."""
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
    return model
```

## Theoretical Perspective

### Why Deep Kernels Work

1. **Feature learning**: NN learns task-relevant features
2. **Uncertainty**: GP provides calibrated uncertainty
3. **Flexibility**: Adapts to complex input spaces

### Interpretation

The deep kernel can be viewed as:
$$k_{DK}(x, x') = \langle \phi(g(x)), \phi(g(x')) \rangle_{\mathcal{H}}$$

where $\phi$ is the feature map of $k_{base}$.

### Relationship to Neural Networks

| Aspect | Standard NN | DKL |
|--------|-------------|-----|
| Output | Point prediction | Distribution |
| Training | Cross-entropy/MSE | Marginal likelihood |
| Last layer | Linear/Softmax | Gaussian Process |
| Uncertainty | None/Dropout | Exact posterior |

## Scalability: KISS-GP

For large datasets, use Structured Kernel Interpolation:

$$K_{nm} \approx W K_{mm} W^T$$

where:
- $m$: Number of inducing points (grid)
- $W$: Sparse interpolation weights
- Complexity: $O(n + m \log m)$ per iteration

### Grid Inducing Points

Place inducing points on a regular grid in feature space:
- Enables Kronecker and Toeplitz structure
- FFT-based fast matrix-vector products

## Experimental Results

The paper demonstrates:

1. **MNIST/CIFAR**: Competitive with deep networks + uncertainty
2. **Airline delays**: Better calibrated predictions
3. **Molecular activity**: Improved extrapolation

### Key Finding

DKL often matches or exceeds pure neural networks while providing **calibrated uncertainty estimates**.

## Extensions

### 1. Convolutional DKL

Use CNN as feature extractor:
```python
feature_extractor = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 5 * 5, 128)
)
```

### 2. Additive Deep Kernels

Sum of component kernels:
$$k(x, x') = \sum_i k_i(g_i(x), g_i(x'))$$

### 3. Spectral Mixture DKL

Use spectral mixture kernel as base:
$$k_{SM}(z, z') = \sum_q w_q \cos(2\pi(z-z')^T \mu_q) \exp(-2\pi^2 (z-z')^T \Sigma_q (z-z'))$$

## Citation

```bibtex
@inproceedings{wilson2016deep,
  title={Deep kernel learning},
  author={Wilson, Andrew Gordon and Hu, Zhiting and 
          Salakhutdinov, Ruslan and Xing, Eric P},
  booktitle={Artificial Intelligence and Statistics},
  pages={370--378},
  year={2016}
}
```

## Further Reading

- Calandra, R., et al. (2016). Manifold Gaussian processes for regression
- Bradshaw, J., et al. (2017). Adversarial examples, uncertainty, and deep kernel learning
- Ober, S. W., et al. (2021). The promises and pitfalls of deep kernel learning
