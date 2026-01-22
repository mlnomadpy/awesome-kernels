---
sidebar_position: 3
title: "Mei & Montanari (2019) - Generalization Error of Random Features"
---

# The Generalization Error of Random Features Regression: Precise Asymptotics and the Double Descent Curve

**Authors:** Song Mei, Andrea Montanari  
**Published:** 2019  
**Venue:** Communications on Pure and Applied Mathematics  
**Link:** [arXiv](https://arxiv.org/abs/1905.10451)

## Summary

This paper provides a precise asymptotic characterization of the generalization error for random features regression. It reveals the "double descent" phenomenon where test error first decreases, increases near the interpolation threshold, then decreases again in the overparameterized regime.

## Key Contributions

### 1. Precise Asymptotic Formula

For random features model with $n$ samples and $N$ features:
$$R(n, N) = \mathcal{B}(n, N) + \mathcal{V}(n, N)$$

where bias $\mathcal{B}$ and variance $\mathcal{V}$ have explicit formulas in terms of $\gamma = N/n$.

### 2. Double Descent Curve

The test error exhibits three phases:
1. **Underparameterized** ($N < n$): Classical bias-variance tradeoff
2. **Interpolation threshold** ($N \approx n$): Peak in error
3. **Overparameterized** ($N > n$): Error decreases again

### 3. Implicit Regularization

In the overparameterized regime, minimum-norm interpolation provides implicit regularization equivalent to ridge regression with $\lambda \to 0^+$.

## Mathematical Framework

### Random Features Model

Given data $(x_i, y_i)_{i=1}^n$ with $x_i \in \mathbb{R}^d$, compute random features:
$$\phi_j(x) = \sigma(w_j^T x), \quad w_j \sim \mathcal{N}(0, I_d/d)$$

The model:
$$f(x) = \sum_{j=1}^N \beta_j \phi_j(x) = \beta^T \phi(x)$$

### Ridgeless Regression

The minimum-norm interpolator:
$$\hat{\beta} = \Phi^+ y = \Phi^T(\Phi \Phi^T)^{-1} y$$

when $N > n$ (overparameterized).

### Asymptotic Regime

Consider $n, N, d \to \infty$ with fixed ratios:
$$\gamma = N/n, \quad \psi = n/d$$

## Main Results

### Theorem: Asymptotic Risk

For $\gamma > 1$ (overparameterized), the asymptotic test error is:
$$R_\infty(\gamma) = \frac{\sigma^2}{\gamma - 1} + \mathcal{B}(\gamma)$$

where:
- $\frac{\sigma^2}{\gamma - 1}$: Variance term (decreasing in $\gamma$)
- $\mathcal{B}(\gamma)$: Bias term (depends on target function)

### Interpolation Peak

At $\gamma = 1$:
$$R_\infty(1^+) = +\infty \quad \text{(blows up)}$$

This is the interpolation threshold where the model just fits the data.

### Double Descent Formula

For ridge regression with parameter $\lambda$:
$$R(\gamma, \lambda) = \sigma^2 \frac{\gamma}{(\gamma - 1 + \lambda)^2} + \mathcal{B}(\gamma, \lambda)$$

Setting $\lambda \to 0$:
- $\gamma < 1$: Finite risk
- $\gamma = 1$: Infinite risk  
- $\gamma > 1$: Finite risk (decreasing)

## Algorithm

```python
import numpy as np

def random_features_regression(X_train, y_train, X_test, y_test, 
                                N_features, sigma_noise=0.1, reg=0.0):
    """
    Random features regression with optional regularization.
    
    Parameters:
    -----------
    X_train, X_test : arrays (n, d) and (n_test, d)
    y_train, y_test : arrays (n,) and (n_test,)
    N_features : number of random features
    sigma_noise : noise level in data
    reg : ridge regularization parameter
    
    Returns:
    --------
    train_error, test_error, bias, variance
    """
    n, d = X_train.shape
    
    # Generate random weights
    W = np.random.randn(d, N_features) / np.sqrt(d)
    
    # Compute features
    Phi_train = np.maximum(0, X_train @ W)  # ReLU
    Phi_test = np.maximum(0, X_test @ W)
    
    # Regression
    if N_features <= n or reg > 0:
        # Standard ridge regression
        beta = np.linalg.solve(
            Phi_train.T @ Phi_train + reg * np.eye(N_features),
            Phi_train.T @ y_train
        )
    else:
        # Minimum norm interpolation
        beta = Phi_train.T @ np.linalg.solve(
            Phi_train @ Phi_train.T + 1e-10 * np.eye(n),
            y_train
        )
    
    # Predictions
    y_pred_train = Phi_train @ beta
    y_pred_test = Phi_test @ beta
    
    # Errors
    train_error = np.mean((y_pred_train - y_train)**2)
    test_error = np.mean((y_pred_test - y_test)**2)
    
    return train_error, test_error

def double_descent_experiment(X, y, X_test, y_test, 
                               N_features_list, n_trials=10):
    """
    Run double descent experiment across feature dimensions.
    """
    n = len(X)
    results = {'N': [], 'train': [], 'test': [], 'gamma': []}
    
    for N in N_features_list:
        train_errors = []
        test_errors = []
        
        for _ in range(n_trials):
            train_err, test_err = random_features_regression(
                X, y, X_test, y_test, N
            )
            train_errors.append(train_err)
            test_errors.append(test_err)
        
        results['N'].append(N)
        results['gamma'].append(N / n)
        results['train'].append(np.mean(train_errors))
        results['test'].append(np.mean(test_errors))
    
    return results

def theoretical_risk(gamma, sigma_sq, signal_strength):
    """
    Compute theoretical asymptotic risk for overparameterized regime.
    
    Parameters:
    -----------
    gamma : N/n ratio
    sigma_sq : noise variance
    signal_strength : ||f*||^2
    """
    if gamma <= 1:
        # Underparameterized: classical bias-variance
        variance = sigma_sq * gamma / (1 - gamma) if gamma < 1 else np.inf
        bias = signal_strength * (1 - gamma)
    else:
        # Overparameterized
        variance = sigma_sq / (gamma - 1)
        bias = signal_strength * (gamma - 1) / gamma  # Simplified
    
    return bias + variance
```

## Double Descent Visualization

```python
import matplotlib.pyplot as plt

def plot_double_descent():
    gammas = np.linspace(0.1, 3.0, 100)
    risks = []
    
    for g in gammas:
        if abs(g - 1) < 0.05:
            risks.append(np.nan)  # Near interpolation threshold
        else:
            risks.append(theoretical_risk(g, sigma_sq=0.1, signal_strength=1.0))
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(gammas, risks)
    plt.axvline(x=1, color='r', linestyle='--', label='Interpolation threshold')
    plt.xlabel('γ = N/n')
    plt.ylabel('Test Error')
    plt.title('Double Descent Curve')
    plt.legend()
    plt.show()
```

## Theoretical Implications

### 1. Interpolation Can Generalize

Classical wisdom: Interpolating training data leads to overfitting.

New insight: With enough parameters, interpolation can generalize well.

### 2. More Parameters Can Help

In overparameterized regime:
$$\frac{\partial R}{\partial N} < 0 \quad \text{for } N > n$$

Adding parameters reduces test error!

### 3. Implicit Regularization

Minimum-norm solution has implicit bias:
$$\|\hat{\beta}\|^2 = \min_{\beta: \Phi\beta = y} \|\beta\|^2$$

This prevents overfitting in high dimensions.

## Connection to Neural Networks

| Aspect | Random Features | Neural Networks |
|--------|-----------------|-----------------|
| Feature learning | Fixed (random) | Learned |
| Double descent | ✓ | ✓ |
| Interpolation | Minimum norm | Implicit |
| Theory | Complete | Partial |

### Implications for Deep Learning

1. **Overparameterization helps**: More parameters can improve generalization
2. **Interpolation is OK**: Zero training error doesn't mean overfitting
3. **Implicit regularization**: Optimization algorithm matters

## Citation

```bibtex
@article{mei2019generalization,
  title={The generalization error of random features regression: 
         Precise asymptotics and the double descent curve},
  author={Mei, Song and Montanari, Andrea},
  journal={Communications on Pure and Applied Mathematics},
  year={2019}
}
```

## Further Reading

- Belkin, M., et al. (2019). Reconciling modern machine learning and bias-variance tradeoff
- Hastie, T., et al. (2019). Surprises in high-dimensional ridgeless least squares interpolation
- Bartlett, P., et al. (2020). Benign overfitting in linear regression
