---
sidebar_position: 2
title: "Mercer (1909) - Functions of Positive and Negative Type"
---

# Functions of Positive and Negative Type and their Connection with the Theory of Integral Equations

**Authors:** James Mercer  
**Published:** 1909  
**Journal:** Philosophical Transactions of the Royal Society of London  

## Summary

Mercer's theorem provides the spectral decomposition of positive definite kernels, showing that any continuous positive definite kernel can be represented as an infinite series of eigenfunctions. This theorem is fundamental to understanding the feature space representation of kernels.

## Key Contributions

### 1. Mercer's Theorem

For a continuous, symmetric, positive definite kernel $K: [a,b] \times [a,b] \to \mathbb{R}$, there exists an orthonormal basis $\{\phi_i\}$ of $L^2([a,b])$ consisting of eigenfunctions of the integral operator $T_K$:

$$(T_K f)(x) = \int_a^b K(x, y) f(y) dy$$

such that:

$$K(x, y) = \sum_{i=1}^{\infty} \lambda_i \phi_i(x) \phi_i(y)$$

where $\lambda_i \geq 0$ are the corresponding eigenvalues, and the convergence is absolute and uniform.

### 2. Feature Space Interpretation

Mercer's theorem reveals the **explicit feature map** for kernels:

$$\phi(x) = \left(\sqrt{\lambda_1}\phi_1(x), \sqrt{\lambda_2}\phi_2(x), \ldots \right)$$

such that:

$$K(x, y) = \langle \phi(x), \phi(y) \rangle_{\ell^2}$$

## Mathematical Details

### The Integral Operator

The integral operator $T_K$ associated with kernel $K$ is a compact, self-adjoint operator on $L^2([a,b])$. By the spectral theorem for compact self-adjoint operators:

- All eigenvalues are real
- Eigenvalues can be ordered: $\lambda_1 \geq \lambda_2 \geq \cdots \geq 0$
- Eigenfunctions form an orthonormal basis

### Conditions for Mercer's Theorem

The original theorem requires:
1. **Continuity**: $K$ is continuous on $[a,b] \times [a,b]$
2. **Positive definiteness**: For all $f \in L^2([a,b])$:
   $$\int_a^b \int_a^b K(x,y) f(x) f(y) dx dy \geq 0$$

### Generalizations

Modern versions extend Mercer's theorem to:
- Compact metric spaces with Borel measures
- Locally compact spaces
- Abstract topological spaces

## Examples

### RBF Kernel on $\mathbb{R}$

For the Gaussian RBF kernel $K(x,y) = e^{-\gamma\|x-y\|^2}$, the Mercer expansion involves Hermite functions as eigenfunctions.

### Polynomial Kernel

For the polynomial kernel $K(x,y) = (x \cdot y + c)^d$, the feature map is finite-dimensional, consisting of all monomials up to degree $d$.

## Impact on Machine Learning

### 1. Kernel Trick Justification
Mercer's theorem provides theoretical justification for the kernel trick by showing that kernel evaluation corresponds to an inner product in a (possibly infinite-dimensional) feature space.

### 2. Approximation Methods
The eigenfunction expansion enables:
- **Nyström approximation**: Approximate kernels using top eigenfunctions
- **Random Fourier Features**: Approximate shift-invariant kernels
- **Kernel PCA**: Finds principal components in feature space

### 3. Universality Analysis
The decay rate of eigenvalues $\lambda_i$ determines:
- Approximation properties of the RKHS
- Complexity of learning with the kernel
- Optimal convergence rates

## Connection to Modern Methods

### Random Fourier Features

For shift-invariant kernels $K(x,y) = k(x-y)$, Bochner's theorem relates to Mercer:

$$K(x,y) = \int_{\mathbb{R}^d} e^{i\omega^\top(x-y)} p(\omega) d\omega$$

where $p(\omega)$ is the spectral density.

### Neural Tangent Kernels

The infinite-width limit of neural networks corresponds to a kernel whose Mercer decomposition reveals the function space of the network.

## Citation

```bibtex
@article{mercer1909functions,
  title={Functions of positive and negative type and their connection 
         with the theory of integral equations},
  author={Mercer, James},
  journal={Philosophical Transactions of the Royal Society of London},
  volume={209},
  pages={415--446},
  year={1909}
}
```

## Further Reading

- Cucker, F., & Smale, S. (2002). On the mathematical foundations of learning
- Williams, C., & Seeger, M. (2000). Using the Nyström method to speed up kernel machines
