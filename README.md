# Awesome Kernels 🎯

[![Deploy to GitHub Pages](https://github.com/mlnomadpy/awesome-kernels/actions/workflows/deploy.yml/badge.svg)](https://github.com/mlnomadpy/awesome-kernels/actions/workflows/deploy.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A curated collection of papers on **Kernel Methods** and **Reproducing Kernel Hilbert Spaces (RKHS)** in machine learning theory.

📖 **[View the Documentation](https://mlnomadpy.github.io/awesome-kernels/)**

## 📚 Papers by Year

### Foundational Works (1900-1970)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 1909 | [Functions of Positive and Negative Type](https://doi.org/10.1098/rsta.1909.0016) | Mercer | Mercer's theorem, kernel eigendecomposition |
| 1950 | [Theory of Reproducing Kernels](https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/) | Aronszajn | RKHS foundations |
| 1963 | [On the Uniform Convergence of Relative Frequencies](https://link.springer.com/article/10.1007/BF02921311) | Vapnik & Chervonenkis | VC theory foundations |
| 1971 | [Some Results on Tchebycheffian Spline Functions](https://doi.org/10.1016/0022-247X(71)90184-3) | Kimeldorf & Wahba | Representer theorem |

### Support Vector Machines Era (1990-2000)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 1992 | [A Training Algorithm for Optimal Margin Classifiers](https://dl.acm.org/doi/10.1145/130385.130401) | Boser, Guyon, Vapnik | Hard-margin SVM |
| 1995 | [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) | Cortes & Vapnik | Soft-margin SVM |
| 1998 | [Statistical Learning Theory](https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034) | Vapnik | VC theory, SRM, SVM theory |
| 1998 | [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.face-rec.org/algorithms/Kernel/kernelPCA_scholkopf.pdf) | Schölkopf, Smola, Müller | Kernel PCA |
| 1999 | [Kernel PCA and De-Noising in Feature Spaces](https://proceedings.neurips.cc/paper/1998/hash/226d1f15ecd35f784d2a20c3ecf56d7f-Abstract.html) | Mika et al. | Kernel PCA for denoising |
| 2000 | [Using the Nyström Method to Speed Up Kernel Machines](https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html) | Williams & Seeger | Nyström approximation |

### Kernel Methods Golden Age (2001-2010)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 2001 | [A Generalized Representer Theorem](https://link.springer.com/chapter/10.1007/3-540-44581-1_27) | Schölkopf, Herbrich, Smola | Generalized representer theorem |
| 2002 | [On the Mathematical Foundations of Learning](https://www.ams.org/journals/bull/2002-39-01/S0273-0979-01-00923-5/) | Cucker & Smale | Learning theory foundations |
| 2002 | [Regularization Networks and Support Vector Machines](https://direct.mit.edu/neco/article/13/6/1443/6505) | Evgeniou, Pontil, Poggio | Unified framework |
| 2004 | [Learning with Kernels](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) | Schölkopf & Smola | Comprehensive textbook |
| 2005 | [On the Nyström Method for Approximating a Gram Matrix](https://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf) | Drineas & Mahoney | Nyström analysis |
| 2006 | [A Kernel Method for the Two-Sample Problem](https://proceedings.neurips.cc/paper/2006/hash/e9fb2eda3d9c55a0d89c98d6c54b5b3e-Abstract.html) | Gretton et al. | MMD introduction |
| 2007 | [A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) | Smola et al. | Kernel mean embeddings |
| 2007 | [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) | Rahimi & Recht | Random Fourier features |
| 2008 | [Support Vector Machines](https://www.springer.com/gp/book/9780387772417) | Steinwart & Christmann | SVM theory textbook |

### Kernel Embeddings & Modern Methods (2011-2020)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 2012 | [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | Gretton et al. | Comprehensive MMD treatment |
| 2012 | [Optimal Rates for the Regularized Least-Squares Algorithm](https://link.springer.com/article/10.1007/s10208-007-9002-3) | Caponnetto & De Vito | Optimal learning rates |
| 2013 | [Fastfood - Approximating Kernel Expansions in Loglinear Time](https://proceedings.mlr.press/v28/le13.html) | Le et al. | Fast random features |
| 2015 | [On the Error of Random Fourier Features](https://arxiv.org/abs/1506.02785) | Sutherland & Schneider | RFF error analysis |
| 2016 | [Orthogonal Random Features](https://proceedings.neurips.cc/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html) | Yu et al. | Improved random features |
| 2017 | [Kernel Mean Embedding of Distributions: A Review and Beyond](https://arxiv.org/abs/1605.09522) | Muandet et al. | Comprehensive review |
| 2018 | [Neural Tangent Kernel](https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html) | Jacot, Gabriel, Hongler | NTK theory |
| 2019 | [Wide Neural Networks of Any Depth Evolve as Linear Models](https://proceedings.neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html) | Lee et al. | Infinite-width networks |

### Recent Advances (2020-Present)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 2020 | [Tensor Programs I-IV](https://arxiv.org/abs/2006.14548) | Yang | Feature learning theory |
| 2020 | [Learning Deep Kernels for Non-Parametric Two-Sample Tests](https://proceedings.mlr.press/v119/liu20m.html) | Liu et al. | Deep kernel MMD |
| 2021 | [Kernel Methods in Machine Learning](https://arxiv.org/abs/2011.00883) | Hofmann, Schölkopf, Smola | Survey paper |
| 2022 | [The Implicit Regularization of Ordinary Least Squares Ensembles](https://proceedings.mlr.press/v151/lejeune22a.html) | Lejeune et al. | Ensemble kernel methods |
| 2023 | [Scaling Laws for Neural Language Models Through the Lens of NTK](https://arxiv.org/abs/2305.16701) | Various | NTK scaling analysis |

## 📖 Topics Covered

### RKHS Fundamentals
- Reproducing property and feature maps
- Moore-Aronszajn theorem
- Mercer's theorem and kernel eigendecomposition
- Representer theorem

### Kernel Learning
- Kernel PCA and dimensionality reduction
- Random Fourier Features
- Nyström approximation
- Multiple kernel learning

### Support Vector Machines
- Hard and soft margin SVMs
- VC dimension and generalization bounds
- Structural risk minimization
- Sequential minimal optimization

### Kernel Embeddings
- Mean embeddings of distributions
- Maximum Mean Discrepancy (MMD)
- Hilbert-Schmidt Independence Criterion (HSIC)
- Conditional mean embeddings

### Modern Advances
- Neural Tangent Kernel (NTK)
- Lazy training and feature learning
- Scalable kernel methods
- Deep kernels

## 🚀 Getting Started

### Local Development

```bash
# Clone the repository
git clone https://github.com/mlnomadpy/awesome-kernels.git
cd awesome-kernels/docs

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

### Documentation Structure

```
docs/
├── docs/
│   ├── intro.md              # Introduction to kernel methods
│   ├── rkhs-fundamentals/    # RKHS theory papers
│   ├── kernel-learning/      # Kernel algorithms
│   ├── support-vector-machines/  # SVM papers
│   ├── kernel-embeddings/    # Distribution embeddings
│   └── modern-advances/      # Recent developments
├── blog/                     # Python implementations
└── src/                      # Website components
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. **Add papers**: Submit PRs with new paper explanations
2. **Improve explanations**: Enhance existing documentation
3. **Add implementations**: Contribute Python tutorials
4. **Fix issues**: Report bugs or suggest improvements

## 📜 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The kernel methods research community
- Authors of all papers included in this collection
- [Docusaurus](https://docusaurus.io/) for the documentation framework

---

**Star ⭐ this repository if you find it useful!**
