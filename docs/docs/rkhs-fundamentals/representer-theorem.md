---
sidebar_position: 3
title: "Representer Theorem"
---

# The Representer Theorem

**Key Papers:**
- Kimeldorf & Wahba (1971) - "Some Results on Tchebycheffian Spline Functions"
- Schölkopf, Herbrich & Smola (2001) - "A Generalized Representer Theorem"

## Summary

The Representer Theorem is one of the most important results in kernel methods, stating that the solution to regularized optimization problems in RKHS can always be expressed as a linear combination of kernel evaluations at the training points. This makes infinite-dimensional optimization tractable.

## The Classical Result

### Theorem (Representer Theorem)

Let $\mathcal{H}$ be an RKHS with kernel $K$. Consider the optimization problem:

$$\min_{f \in \mathcal{H}} \left[ \sum_{i=1}^n L(y_i, f(x_i)) + \Omega(\|f\|_{\mathcal{H}}) \right]$$

where:
- $L$ is an arbitrary loss function
- $\Omega: [0, \infty) \to \mathbb{R}$ is strictly increasing

Then any minimizer $f^*$ admits a representation:

$$f^*(x) = \sum_{i=1}^n \alpha_i K(x, x_i)$$

for some $\alpha_1, \ldots, \alpha_n \in \mathbb{R}$.

## Proof Sketch

The proof relies on the orthogonal decomposition of the RKHS. Any function $f \in \mathcal{H}$ can be written as:

$$f = f_{\parallel} + f_{\perp}$$

where:
- $f_{\parallel} \in \text{span}\{K(\cdot, x_1), \ldots, K(\cdot, x_n)\}$
- $f_{\perp} \perp \text{span}\{K(\cdot, x_1), \ldots, K(\cdot, x_n)\}$

**Key observations:**
1. By the reproducing property: $f(x_i) = \langle f, K(\cdot, x_i) \rangle = f_{\parallel}(x_i)$
2. Therefore: $\|f\|_{\mathcal{H}}^2 = \|f_{\parallel}\|^2 + \|f_{\perp}\|^2 \geq \|f_{\parallel}\|^2$

Since the loss depends only on $f(x_i) = f_{\parallel}(x_i)$ and $\|f\| \geq \|f_{\parallel}\|$, setting $f_{\perp} = 0$ never increases the objective.

## Generalized Representer Theorem

Schölkopf, Herbrich & Smola (2001) extended the theorem:

### Theorem (Generalized)

For the problem:
$$\min_{f \in \mathcal{H}} c\left((x_1, y_1, f(x_1)), \ldots, (x_n, y_n, f(x_n)), \|f\|_{\mathcal{H}}\right)$$

where $c$ is strictly increasing in its last argument, the minimizer has the form:
$$f^* = \sum_{i=1}^n \alpha_i K(\cdot, x_i)$$

## Applications

### 1. Kernel Ridge Regression

Problem:
$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$$

By the representer theorem, $f^*(x) = \sum_j \alpha_j K(x, x_j)$. Substituting:
$$\min_{\boldsymbol{\alpha}} \frac{1}{n}\|\mathbf{y} - \mathbf{K}\boldsymbol{\alpha}\|^2 + \lambda \boldsymbol{\alpha}^\top \mathbf{K} \boldsymbol{\alpha}$$

Solution: $\boldsymbol{\alpha} = (\mathbf{K} + n\lambda \mathbf{I})^{-1} \mathbf{y}$

### 2. Support Vector Machines

For SVMs with hinge loss:
$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \max(0, 1 - y_i f(x_i)) + \lambda \|f\|_{\mathcal{H}}^2$$

The representer theorem reduces this to finite-dimensional quadratic programming.

### 3. Kernel Logistic Regression

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \log(1 + e^{-y_i f(x_i)}) + \lambda \|f\|_{\mathcal{H}}^2$$

### 4. Gaussian Processes

The representer theorem explains why GP posterior means are linear combinations of kernel functions centered at observation points.

## Computational Implications

The representer theorem transforms:
- **Infinite-dimensional optimization** → **$n$-dimensional optimization**
- **Function optimization** → **Coefficient optimization**

This enables:
1. Standard convex optimization techniques
2. $O(n^3)$ complexity for kernel methods
3. Sparse approximations via support vectors

## Semi-Parametric Extensions

For problems of the form:
$$\min_{f = f_0 + g} \left[ \text{Loss}(f) + \lambda \|g\|_{\mathcal{H}}^2 \right]$$

where $f_0$ is in a finite-dimensional unpenalized space (e.g., polynomials), the solution is:
$$f^* = f_0^* + \sum_{i=1}^n \alpha_i K(\cdot, x_i)$$

## Limitations and Extensions

### When Representer Theorem Doesn't Apply

- Non-strictly-increasing regularizers
- Multiple kernel learning (modified versions needed)
- Online/streaming settings

### Modern Extensions

1. **Indefinite kernels**: Extended to Krein spaces
2. **Vector-valued RKHS**: Multi-output regression
3. **Operator-valued kernels**: Structured prediction

## Citation

```bibtex
@inproceedings{scholkopf2001generalized,
  title={A generalized representer theorem},
  author={Sch{\"o}lkopf, Bernhard and Herbrich, Ralf and Smola, Alex J},
  booktitle={International Conference on Computational Learning Theory},
  pages={416--426},
  year={2001}
}

@article{kimeldorf1971some,
  title={Some results on Tchebycheffian spline functions},
  author={Kimeldorf, George S and Wahba, Grace},
  journal={Journal of Mathematical Analysis and Applications},
  volume={33},
  pages={82--95},
  year={1971}
}
```
