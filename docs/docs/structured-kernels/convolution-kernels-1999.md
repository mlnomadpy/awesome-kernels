---
sidebar_position: 1
title: "Haussler (1999) - Convolution Kernels on Discrete Structures"
---

# Convolution Kernels on Discrete Structures

**Authors:** David Haussler  
**Published:** 1999  
**Journal:** Technical Report, UC Santa Cruz  
**Link:** [PDF](https://www.jmlr.org/papers/v1/haussler99a.html)

## Summary

This foundational paper introduces convolution kernels, a general framework for defining kernels on structured objects like strings, trees, and graphs. The key idea is to decompose structures into parts and combine kernels on the parts to define a kernel on the whole structure.

## Key Contributions

### 1. R-Convolution Framework

For structures $x$ and $y$ decomposable into parts:
$$K(x, y) = \sum_{\bar{x} \in R^{-1}(x)} \sum_{\bar{y} \in R^{-1}(y)} \prod_{i=1}^D K_i(x_i, y_i)$$

where $R^{-1}(x)$ gives all ways to decompose $x$ into parts.

### 2. Kernel Closure Properties

The paper proves that if $K_i$ are valid kernels, so is the R-convolution $K$.

### 3. Application to Sequences and Trees

Provides specific instantiations for:
- String kernels
- Tree kernels
- Set kernels

## Mathematical Framework

### Decomposition Relation

A **relation** $R$ from $\mathcal{X}_1 \times \cdots \times \mathcal{X}_D$ to $\mathcal{X}$ defines how structures decompose:
$$R(x_1, \ldots, x_D, x) = 1 \iff x \text{ decomposes into } (x_1, \ldots, x_D)$$

### R-Convolution Definition

Given:
- Base spaces $\mathcal{X}_1, \ldots, \mathcal{X}_D$
- Kernels $K_1, \ldots, K_D$ on these spaces
- Relation $R$

The R-convolution kernel on $\mathcal{X}$:
$$K(x, y) = \sum_{\substack{\bar{x}: R(\bar{x}, x)=1 \\ \bar{y}: R(\bar{y}, y)=1}} \prod_{i=1}^D K_i(x_i, y_i)$$

### Positive Definiteness

**Theorem:** If each $K_i$ is positive definite, then $K$ is positive definite.

**Proof sketch:** The sum of tensor products of PSD kernels is PSD.

## String Kernels

### Substring Kernel

Decomposition: All contiguous substrings
$$R^{-1}(s) = \{(s[i:j]) : 1 \leq i \leq j \leq |s|\}$$

Kernel:
$$K(s, t) = \sum_{u \in \Sigma^*} \phi_u(s) \phi_u(t)$$

where $\phi_u(s)$ counts occurrences of substring $u$ in $s$.

### Gap-Weighted Subsequence Kernel

Allow gaps with decay parameter $\lambda$:
$$K_n(s, t) = \sum_{u \in \Sigma^n} \sum_{\mathbf{i}: s[\mathbf{i}]=u} \sum_{\mathbf{j}: t[\mathbf{j}]=u} \lambda^{l(\mathbf{i}) + l(\mathbf{j})}$$

where $l(\mathbf{i})$ is the span of indices.

## Tree Kernels

### Subtree Kernel

For rooted trees $T_1, T_2$:
$$K(T_1, T_2) = \sum_{n_1 \in T_1} \sum_{n_2 \in T_2} \Delta(n_1, n_2)$$

where $\Delta(n_1, n_2)$ is the number of common subtrees rooted at $n_1$ and $n_2$.

### Recursive Computation

$$\Delta(n_1, n_2) = \begin{cases}
0 & \text{if } label(n_1) \neq label(n_2) \\
\lambda & \text{if } n_1, n_2 \text{ are leaves} \\
\lambda \prod_{j=1}^{nc} (1 + \Delta(ch_j(n_1), ch_j(n_2))) & \text{otherwise}
\end{cases}$$

## Algorithm

```python
import numpy as np
from collections import defaultdict

def substring_kernel(s1, s2, max_length=None):
    """
    Compute substring kernel (spectrum kernel).
    
    Parameters:
    -----------
    s1, s2 : strings
    max_length : maximum substring length (None = all)
    
    Returns:
    --------
    kernel value (dot product in substring feature space)
    """
    if max_length is None:
        max_length = max(len(s1), len(s2))
    
    # Count substrings
    def get_substrings(s, k):
        counts = defaultdict(int)
        for i in range(len(s) - k + 1):
            counts[s[i:i+k]] += 1
        return counts
    
    # Kernel is sum over all lengths
    kernel_val = 0
    for k in range(1, max_length + 1):
        counts1 = get_substrings(s1, k)
        counts2 = get_substrings(s2, k)
        for sub in counts1:
            if sub in counts2:
                kernel_val += counts1[sub] * counts2[sub]
    
    return kernel_val

def subsequence_kernel(s1, s2, n, lambda_decay=0.5):
    """
    Gap-weighted subsequence kernel using dynamic programming.
    
    Parameters:
    -----------
    s1, s2 : strings
    n : subsequence length
    lambda_decay : gap penalty (0 < lambda < 1)
    """
    len1, len2 = len(s1), len(s2)
    
    # DP tables
    K = np.zeros((n + 1, len1 + 1, len2 + 1))
    K[0, :, :] = 1  # Base case
    
    for i in range(1, n + 1):
        for s in range(i, len1 + 1):
            for t in range(i, len2 + 1):
                if s1[s-1] == s2[t-1]:
                    K[i, s, t] = lambda_decay * (
                        K[i, s-1, t] + K[i, s, t-1] - 
                        lambda_decay * K[i, s-1, t-1] +
                        lambda_decay ** 2 * K[i-1, s-1, t-1]
                    )
                else:
                    K[i, s, t] = lambda_decay * (
                        K[i, s-1, t] + K[i, s, t-1] - 
                        lambda_decay * K[i, s-1, t-1]
                    )
    
    return K[n, len1, len2]

class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

def tree_kernel(t1, t2, lambda_decay=1.0, cache=None):
    """
    Compute convolution tree kernel.
    
    Parameters:
    -----------
    t1, t2 : TreeNode objects
    lambda_decay : decay factor for subtree depth
    """
    if cache is None:
        cache = {}
    
    def delta(n1, n2):
        """Compute matching subtrees at nodes n1, n2."""
        key = (id(n1), id(n2))
        if key in cache:
            return cache[key]
        
        # Different labels = no match
        if n1.label != n2.label:
            cache[key] = 0
            return 0
        
        # Different number of children = no match for this kernel variant
        if len(n1.children) != len(n2.children):
            cache[key] = 0
            return 0
        
        # Leaves
        if len(n1.children) == 0:
            cache[key] = lambda_decay
            return lambda_decay
        
        # Internal nodes
        result = lambda_decay
        for c1, c2 in zip(n1.children, n2.children):
            result *= (1 + delta(c1, c2))
        
        cache[key] = result
        return result
    
    # Sum over all node pairs
    def all_nodes(t):
        nodes = [t]
        for c in t.children:
            nodes.extend(all_nodes(c))
        return nodes
    
    nodes1 = all_nodes(t1)
    nodes2 = all_nodes(t2)
    
    return sum(delta(n1, n2) for n1 in nodes1 for n2 in nodes2)
```

## Set Kernels

### Intersection Kernel

For sets $A, B$:
$$K(A, B) = |A \cap B|$$

### Subset Kernel

Count common subsets:
$$K(A, B) = 2^{|A \cap B|}$$

### Weighted Sum Kernel

$$K(A, B) = \sum_{a \in A} \sum_{b \in B} k(a, b)$$

## Applications

### 1. Text Classification

String kernels for:
- Document categorization
- Spam detection
- Authorship attribution

### 2. Bioinformatics

- Protein classification (sequence kernels)
- Gene function prediction
- RNA structure comparison

### 3. Natural Language Processing

Tree kernels for:
- Parse tree comparison
- Relation extraction
- Semantic similarity

### 4. Chemoinformatics

Graph kernels for:
- Molecular property prediction
- Drug discovery
- Toxicity prediction

## Complexity Analysis

| Kernel Type | Naive | With DP |
|-------------|-------|---------|
| Substring (length k) | $O(n^2 k)$ | $O(nk)$ |
| Subsequence | $O(n^2 2^n)$ | $O(n^2 k)$ |
| Tree | $O(|T_1| |T_2|)$ | $O(|T_1| |T_2|)$ |

## Citation

```bibtex
@article{haussler1999convolution,
  title={Convolution kernels on discrete structures},
  author={Haussler, David},
  journal={Technical report, UC Santa Cruz},
  year={1999}
}
```

## Further Reading

- Lodhi, H., et al. (2002). Text classification using string kernels
- Collins, M. & Duffy, N. (2002). Convolution kernels for natural language
- Vishwanathan, S. V. N., et al. (2010). Graph kernels
