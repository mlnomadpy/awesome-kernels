---
sidebar_position: 2
title: "Gärtner et al. (2003) - Graph Kernels"
---

# On Graph Kernels: Hardness Results and Efficient Alternatives

**Authors:** Thomas Gärtner, Peter Flach, Stefan Wrobel  
**Published:** 2003  
**Venue:** COLT/Learning Theory  
**Link:** [Springer](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_11)

## Summary

This paper provides foundational results on graph kernels, showing that computing certain natural graph kernels is NP-hard (related to graph isomorphism). It then proposes efficient alternatives based on random walks and subgraph patterns that can be computed in polynomial time.

## Key Contributions

### 1. Hardness Results

**Theorem:** Computing the complete graph kernel (counting all isomorphic subgraphs) is at least as hard as the graph isomorphism problem.

### 2. Random Walk Kernel

Efficient kernel based on counting common walks:
$$K(G, G') = \sum_{i=0}^{\infty} \lambda^i |\{(w, w'): w \sim_i w'\}|$$

where $w \sim_i w'$ means walks of length $i$ match.

### 3. Weisfeiler-Lehman Subtree Kernel

Polynomial-time kernel based on neighborhood aggregation:
$$K_{WL}^{(h)}(G, G') = \sum_{i=0}^{h} K_{\delta}(G_i, G'_i)$$

where $G_i$ is the graph with labels from iteration $i$ of WL coloring.

## Mathematical Framework

### Graph Representation

A labeled graph $G = (V, E, \ell)$ where:
- $V$: vertices
- $E \subseteq V \times V$: edges
- $\ell: V \to \Sigma$: vertex labels

### Adjacency Matrix

$A_{ij} = 1$ if $(i,j) \in E$, else 0.

### Complete Graph Kernel

The "ideal" kernel would count isomorphic substructures:
$$K_{complete}(G, G') = |\{(S, S'): S \subseteq G, S' \subseteq G', S \cong S'\}|$$

This is computationally intractable.

## Random Walk Kernel

### Definition

Count matching walks between two graphs:
$$K_{RW}(G, G') = \sum_{p \in walks(G)} \sum_{q \in walks(G')} k_{walk}(p, q)$$

### Direct Product Graph

Compute via the direct product graph $G_\times = G \times G'$:
- Vertices: $V_\times = V \times V'$
- Edges: $((u,u'), (v,v')) \in E_\times$ iff $(u,v) \in E$ and $(u',v') \in E'$

### Matrix Formula

$$K_{RW}(G, G') = \sum_{i,j} [(\mathbf{I} - \lambda A_\times)^{-1}]_{ij}$$

where $A_\times$ is the adjacency matrix of $G_\times$.

### Efficient Computation

Using the identity $(\mathbf{I} - \lambda A)^{-1} = \sum_{i=0}^{\infty} \lambda^i A^i$:

$$K_{RW}(G, G') = \mathbf{e}^T (\mathbf{I} - \lambda A_\times)^{-1} \mathbf{e}$$

Complexity: $O(n^6)$ for naive, $O(n^3)$ with conjugate gradient.

## Weisfeiler-Lehman Kernel

### WL Color Refinement

Iteratively update vertex labels:
$$\ell^{(i+1)}(v) = \text{hash}\left(\ell^{(i)}(v), \{\!\!\{ \ell^{(i)}(u) : u \in N(v) \}\!\!\}\right)$$

where $\{\!\!\{ \cdot \}\!\!\}$ denotes multiset.

### WL Subtree Kernel

Count matching label histograms:
$$K_{WL}(G, G') = \sum_{i=0}^{h} \langle \phi^{(i)}(G), \phi^{(i)}(G') \rangle$$

where $\phi^{(i)}(G)$ is the histogram of labels at iteration $i$.

## Algorithm

```python
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def random_walk_kernel(G1, G2, lambda_decay=0.1):
    """
    Compute random walk kernel between two graphs.
    
    Parameters:
    -----------
    G1, G2 : adjacency matrices (sparse or dense)
    lambda_decay : decay factor (must satisfy λ < 1/max_degree)
    
    Returns:
    --------
    kernel value
    """
    n1, n2 = G1.shape[0], G2.shape[0]
    
    # Direct product graph
    # A_x[i*n2+j, k*n2+l] = G1[i,k] * G2[j,l]
    A_x = np.kron(G1, G2)
    
    n_x = A_x.shape[0]
    I = np.eye(n_x)
    
    # Compute (I - λA)^{-1}
    try:
        M_inv = np.linalg.inv(I - lambda_decay * A_x)
    except np.linalg.LinAlgError:
        # Use pseudoinverse if singular
        M_inv = np.linalg.pinv(I - lambda_decay * A_x)
    
    # Sum all entries
    return np.sum(M_inv)

def weisfeiler_lehman_kernel(graphs, h=3):
    """
    Compute Weisfeiler-Lehman kernel matrix.
    
    Parameters:
    -----------
    graphs : list of (adjacency_matrix, node_labels) tuples
    h : number of WL iterations
    
    Returns:
    --------
    K : kernel matrix
    """
    n_graphs = len(graphs)
    
    # Initialize labels
    all_labels = [list(labels) for _, labels in graphs]
    
    # Feature vectors for each iteration
    features = [defaultdict(lambda: np.zeros(n_graphs)) for _ in range(h + 1)]
    
    for iteration in range(h + 1):
        # Count label occurrences
        for g_idx, labels in enumerate(all_labels):
            for label in labels:
                features[iteration][label][g_idx] += 1
        
        if iteration < h:
            # Update labels (WL refinement)
            new_all_labels = []
            for (adj, _), labels in zip(graphs, all_labels):
                new_labels = []
                for v in range(len(labels)):
                    neighbors = np.where(adj[v] > 0)[0]
                    neighbor_labels = sorted([labels[u] for u in neighbors])
                    new_label = hash((labels[v], tuple(neighbor_labels)))
                    new_labels.append(new_label)
                new_all_labels.append(new_labels)
            all_labels = new_all_labels
    
    # Compute kernel matrix
    K = np.zeros((n_graphs, n_graphs))
    for iteration in range(h + 1):
        feat_matrix = np.array(list(features[iteration].values())).T
        K += feat_matrix @ feat_matrix.T
    
    return K

def shortest_path_kernel(G1, G2, labels1=None, labels2=None):
    """
    Compute shortest path kernel.
    """
    from scipy.sparse.csgraph import shortest_path
    
    # Compute all-pairs shortest paths
    sp1 = shortest_path(G1)
    sp2 = shortest_path(G2)
    
    # Count matching path lengths (with node labels if provided)
    kernel_val = 0
    n1, n2 = G1.shape[0], G2.shape[0]
    
    for i in range(n1):
        for j in range(i+1, n1):
            for p in range(n2):
                for q in range(p+1, n2):
                    if sp1[i,j] == sp2[p,q] and np.isfinite(sp1[i,j]):
                        # Check label match if labels provided
                        if labels1 is None or (
                            labels1[i] == labels2[p] and labels1[j] == labels2[q]
                        ):
                            kernel_val += 1
    
    return kernel_val

def graphlet_kernel(G1, G2, k=3):
    """
    Count common graphlets (small subgraphs) of size k.
    """
    # Enumerate all k-node subgraphs and compare
    # This is exponential in k but polynomial in |V| for fixed k
    pass  # Full implementation involves subgraph enumeration
```

## Kernel Comparison

| Kernel | Complexity | Expressiveness | Scalability |
|--------|------------|----------------|-------------|
| Complete | NP-hard | Maximal | Intractable |
| Random Walk | $O(n^3)$ | High | Moderate |
| Shortest Path | $O(n^4)$ | Medium | Limited |
| WL Subtree | $O(hm)$ | High | Excellent |
| Graphlet | $O(n^k)$ | Low | Limited |

## Expressiveness Analysis

### What Random Walk Captures

- Local connectivity patterns
- Path structures
- Weighted by length (via $\lambda$)

### What WL Captures

- Neighborhood aggregation patterns
- Subtree isomorphism (up to depth $h$)
- Distinguishes non-isomorphic graphs (with exceptions)

### Limitations

Neither can distinguish:
- Regular graphs with same parameters
- Strongly regular graphs
- Certain counter-examples to WL

## Applications

### 1. Chemoinformatics

- Molecular property prediction
- Drug-drug interaction
- Toxicity classification

### 2. Bioinformatics

- Protein structure comparison
- Metabolic network analysis
- Gene regulatory networks

### 3. Social Networks

- Community detection
- Influence prediction
- Network classification

## Citation

```bibtex
@inproceedings{gartner2003graph,
  title={On graph kernels: Hardness results and efficient alternatives},
  author={G{\"a}rtner, Thomas and Flach, Peter and Wrobel, Stefan},
  booktitle={Learning Theory and Kernel Machines},
  pages={129--143},
  year={2003}
}
```

## Further Reading

- Vishwanathan, S. V. N., et al. (2010). Graph kernels
- Shervashidze, N., et al. (2011). Weisfeiler-Lehman graph kernels
- Kriege, N., et al. (2020). A survey on graph kernels
