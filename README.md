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
| 1968 | [On the Uniform Convergence of Relative Frequencies](https://link.springer.com/article/10.1007/BF02921311) | Vapnik & Chervonenkis | VC theory foundations (English 1971) |
| 1971 | [Some Results on Tchebycheffian Spline Functions](https://doi.org/10.1016/0022-247X(71)90184-3) | Kimeldorf & Wahba | Representer theorem |

### Support Vector Machines Era (1990-2000)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 1992 | [A Training Algorithm for Optimal Margin Classifiers](https://dl.acm.org/doi/10.1145/130385.130401) | Boser, Guyon, Vapnik | Hard-margin SVM |
| 1995 | [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) | Cortes & Vapnik | Soft-margin SVM |
| 1998 | [Statistical Learning Theory](https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034) | Vapnik | VC theory, SRM, SVM theory |
| 1998 | [Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.face-rec.org/algorithms/Kernel/kernelPCA_scholkopf.pdf) | Schölkopf, Smola, Müller | Kernel PCA |
| 1999 | [Kernel PCA and De-Noising in Feature Spaces](https://proceedings.neurips.cc/paper/1998/hash/226d1f15ecd35f784d2a20c3ecf56d7f-Abstract.html) | Mika et al. | Kernel PCA for denoising |
| 1999 | [Convolution Kernels on Discrete Structures](https://www.jmlr.org/papers/v1/haussler99a.html) | Haussler | Convolution kernels |
| 2000 | [Using the Nyström Method to Speed Up Kernel Machines](https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html) | Williams & Seeger | Nyström approximation |

### Kernel Methods Golden Age (2001-2010)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 2001 | [A Generalized Representer Theorem](https://link.springer.com/chapter/10.1007/3-540-44581-1_27) | Schölkopf, Herbrich, Smola | Generalized representer theorem |
| 2002 | [Stability and Generalization](https://www.jmlr.org/papers/v2/bousquet02a.html) | Bousquet, Elisseeff | Algorithmic stability and generalization |
| 2002 | [On the Mathematical Foundations of Learning](https://www.ams.org/journals/bull/2002-39-01/S0273-0979-01-00923-5/) | Cucker & Smale | Learning theory foundations |
| 2002 | [Regularization Networks and Support Vector Machines](https://direct.mit.edu/neco/article/13/6/1443/6505) | Evgeniou, Pontil, Poggio | Unified framework |
| 2004 | [Learning the Kernel Matrix with Semidefinite Programming](https://www.jmlr.org/papers/v5/lanckriet04a.html) | Lanckriet, Cristianini, Bartlett, El Ghaoui, Jordan | Multiple kernel learning via SDP |
| 2004 | [Multiple Kernel Learning, Conic Duality, and the SMO Algorithm](https://dl.acm.org/doi/10.1145/1015330.1015424) | Bach, Lanckriet, Jordan | MKL optimization |
| 2004 | [Learning with Kernels](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) | Schölkopf & Smola | Comprehensive textbook |
| 2005 | [Learning Theory Estimates via Integral Operators](https://link.springer.com/article/10.1007/s00365-006-0659-y) | Smale, Zhou | Integral operator framework |
| 2005 | [On the Nyström Method for Approximating a Gram Matrix](https://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf) | Drineas & Mahoney | Nyström analysis |
| 2005 | [Measuring Statistical Dependence with Hilbert-Schmidt Norms](https://link.springer.com/chapter/10.1007/11564089_7) | Gretton, Bousquet, Smola, Schölkopf | HSIC introduction |
| 2006 | [A Kernel Method for the Two-Sample Problem](https://proceedings.neurips.cc/paper/2006/hash/e9fb2eda3d9c55a0d89c98d6c54b5b3e-Abstract.html) | Gretton et al. | MMD introduction |
| 2006 | [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/) | Rasmussen, Williams | GP textbook |
| 2007 | [A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) | Smola et al. | Kernel mean embeddings |
| 2007 | [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) | Rahimi & Recht | Random Fourier features |
| 2007 | [Optimal Rates for the Regularized Least-Squares Algorithm](https://link.springer.com/article/10.1007/s10208-006-0196-8) | Caponnetto, De Vito | Optimal learning rates |
| 2008 | [Support Vector Machines](https://www.springer.com/gp/book/9780387772417) | Steinwart & Christmann | SVM theory textbook |
| 2008 | [Kernel Measures of Conditional Dependence](https://papers.nips.cc/paper/2007/hash/3a0772443a0739141571f2e4e675a4e2-Abstract.html) | Fukumizu, Bach, Gretton | Conditional dependence measures |
| 2009 | [Kernel Methods for Deep Learning](https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html) | Cho, Saul | Arc-cosine kernels / compositional kernels |

### Kernel Embeddings & Modern Methods (2011-2020)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 2011 | [A Kernel-Based Test for Conditional Independence](https://dl.acm.org/doi/10.5555/2986459.2986567) | Zhang, Peters, Janzing, Schölkopf | Conditional independence testing |
| 2011 | [Multiple Kernel Learning Algorithms](https://www.jmlr.org/papers/v12/gonen11a.html) | Gönen, Alpaydın | MKL survey |
| 2012 | [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | Gretton et al. | Comprehensive MMD treatment |
| 2013 | [Fastfood - Approximating Kernel Expansions in Loglinear Time](https://proceedings.mlr.press/v28/le13.html) | Le et al. | Fast random features |
| 2013 | [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | Cuturi | Optimal transport computation |
| 2013 | [On Graph Kernels: Hardness Results and Efficient Alternatives](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_11) | Gärtner, Flach, Wrobel | Graph kernels |
| 2015 | [On the Error of Random Fourier Features](https://arxiv.org/abs/1506.02785) | Sutherland & Schneider | RFF error analysis |
| 2015 | [Less Is More: Nyström Computational Regularization](https://arxiv.org/abs/1507.04717) | Rudi, Camoriano, Rosasco | Nyström regularization |
| 2016 | [Orthogonal Random Features](https://proceedings.neurips.cc/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html) | Yu et al. | Improved random features |
| 2016 | [Deep Kernel Learning](https://proceedings.mlr.press/v51/wilson16.html) | Wilson, Hu, Salakhutdinov, Xing | Deep kernels |
| 2016 | [Toward Deeper Understanding of Neural Networks: Power of Initialization](https://proceedings.mlr.press/v49/daniely16.html) | Daniely, Frostig, Singer | NN initialization theory |
| 2017 | [Kernel Mean Embedding of Distributions: A Review and Beyond](https://arxiv.org/abs/1605.09522) | Muandet et al. | Comprehensive review |
| 2017 | [Breaking the Curse of Dimensionality with Random Features](https://arxiv.org/abs/1702.05803) | Bach | Random features analysis |
| 2018 | [Neural Tangent Kernel](https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html) | Jacot, Gabriel, Hongler | NTK theory |
| 2018 | [On the Global Convergence of Gradient Descent for Over-Parameterized Models](https://arxiv.org/abs/1805.00915) | Chizat, Bach | Global convergence analysis |
| 2018 | [Subsampling for Ridge Regression via Regularized Leverage Scores](https://arxiv.org/abs/1803.05049) | Avron, Clarkson, Woodruff | Leverage score sampling |
| 2018 | [Gaussian Processes and Kernel Methods: A Review](https://arxiv.org/abs/1807.02582) | Kanagawa, Sriperumbudur, Fukumizu | GP-RKHS connections |
| 2019 | [Wide Neural Networks of Any Depth Evolve as Linear Models](https://proceedings.neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html) | Lee et al. | Infinite-width networks |
| 2019 | [On Exact Computation with an Infinitely Wide Neural Network](https://arxiv.org/abs/1904.11955) | Arora, Du, Hu, Li, Salakhutdinov | NTK computation |
| 2019 | [The Generalization Error of Random Features Regression](https://arxiv.org/abs/1905.10451) | Mei, Montanari | Random features generalization |

### Recent Advances (2020-Present)

| Year | Paper | Authors | Topic |
|------|-------|---------|-------|
| 2020 | [Tensor Programs I-IV](https://arxiv.org/abs/2006.14548) | Yang | Feature learning theory |
| 2020 | [Learning Deep Kernels for Non-Parametric Two-Sample Tests](https://proceedings.mlr.press/v119/liu20m.html) | Liu et al. | Deep kernel MMD |
| 2021 | [Kernel Methods in Machine Learning](https://arxiv.org/abs/2011.00883) | Hofmann, Schölkopf, Smola | Survey paper |
| 2022 | [The Implicit Regularization of Ordinary Least Squares Ensembles](https://proceedings.mlr.press/v151/lejeune22a.html) | Lejeune et al. | Ensemble kernel methods |
| 2023 | [Scaling Laws for Neural Language Models Through the Lens of the NTK](https://arxiv.org/abs/2305.16701) | Anagnostidis, Bachmann, Malach, Kaddour, Noci | NTK scaling analysis |

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

### Stability & Generalization Theory
- Algorithmic stability bounds
- Optimal learning rates
- Regularized learning algorithms

### Multiple Kernel Learning
- Kernel matrix learning
- Conic duality methods
- MKL algorithms and optimization

### Dependence Measures
- Conditional dependence testing
- Independence measures with kernels
- Statistical hypothesis testing

### Gaussian Processes
- GP-RKHS connections
- Bayesian kernel methods
- Probabilistic inference

### Structured Kernels
- Graph kernels
- Convolution kernels
- Optimal transport kernels

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

### Python Examples (JAX)

The repository includes JAX implementations of kernel methods in the `examples/` folder:

```bash
# Install Python dependencies
pip install jax jaxlib pytest

# Run tests
pytest tests/ -v

# Use the implementations
python -c "from examples import KernelRidgeRegression, RandomFourierFeatures, MMDTest"
```

**Available modules:**
- `KernelRidgeRegression` - Kernel Ridge Regression with RBF, linear, and polynomial kernels
- `RandomFourierFeatures` - Random Fourier Features for scalable kernel approximation
- `RFFRidgeRegression` - Ridge regression with Random Fourier Features
- `MMDTest` - Maximum Mean Discrepancy two-sample test

### Documentation Structure

```
awesome-kernels/
├── examples/                 # JAX implementations
│   ├── kernel_ridge_regression.py
│   ├── random_fourier_features.py
│   └── mmd_test.py
├── tests/                    # Unit tests
├── docs/
│   ├── docs/
│   │   ├── intro.md              # Introduction to kernel methods
│   │   ├── rkhs-fundamentals/    # RKHS theory papers
│   │   ├── kernel-learning/      # Kernel algorithms
│   │   ├── support-vector-machines/  # SVM papers
│   │   ├── kernel-embeddings/    # Distribution embeddings
│   │   └── modern-advances/      # Recent developments
│   ├── blog/                     # Python/JAX implementations
│   └── src/                      # Website components
└── pyproject.toml            # Python project configuration
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
