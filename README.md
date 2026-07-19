# BKP: Beta Kernel Process Modeling Website

[![Deploy to GitHub Pages](https://github.com/Jiangyan-Zhao/BKP-website/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/Jiangyan-Zhao/BKP-website/actions/workflows/deploy-pages.yml) 
[![CRAN status](https://www.r-pkg.org/badges/version/BKP)](https://cran.r-project.org/web/packages/BKP/index.html) 
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

This repository contains the companion website for **BKP: Beta Kernel Process Modeling**, an R package for nonparametric modeling of spatially varying binary, binomial, categorical, and multinomial probabilities.

The website provides an accessible introduction to the Beta Kernel Process (BKP), its Dirichlet extension (DKP), and the scalable TwinBKP and TwinDKP approximations. It combines methodological explanations, installation guidance, interactive visualizations, reproducible examples, and links to the associated software and paper.

**Live website:** <https://jiangyan-zhao.github.io/BKP-website/>

## Overview

The Beta Kernel Process directly models an input-dependent probability function using kernel-weighted observations and conjugate Beta updates. For binary or binomial data, the posterior distribution at an input location $\boldsymbol{x}$ is

```math
\begin{aligned}
\pi(\boldsymbol{x}) \mid \mathcal{D}_n
&\sim \mathrm{Beta}\!\left(
\alpha_n(\boldsymbol{x}),
\beta_n(\boldsymbol{x})
\right), \\
\alpha_n(\boldsymbol{x})
&= \alpha_0(\boldsymbol{x})
+ \sum_{i=1}^{n}
k(\boldsymbol{x},\boldsymbol{x}_i)y_i, \\
\beta_n(\boldsymbol{x})
&= \beta_0(\boldsymbol{x})
+ \sum_{i=1}^{n}
k(\boldsymbol{x},\boldsymbol{x}_i)(m_i-y_i).
\end{aligned}
```

This construction yields closed-form posterior summaries on the probability scale without introducing a latent Gaussian process or requiring MCMC, Laplace approximation, or variational inference.

The framework is extended in two directions:

- **DKP** uses Dirichlet–multinomial updating for categorical and multinomial responses.
- **TwinBKP and TwinDKP** use a global–local approximation based on twinning and nearest-neighbor selection for scalable modeling with large datasets.

## Methodological lineage

For the broader methodological context—from Continuous Correlated Beta Processes and beta-kernel classification to the Smooth Beta Process, BKP/DKP, and scalable Twin variants—see the [visual methodological lineage on the live website](https://jiangyan-zhao.github.io/BKP-website/?section=method).

## Website contents

The website is organized into the following sections:

### Overview

A concise introduction to BKP and its main modeling objectives, including direct probability-scale inference and conjugate posterior updating.

### Install and quick start

Installation instructions for the CRAN and development versions of the `BKP` package, followed by a minimal reproducible workflow.

### Method

An explanation of:

- Beta Kernel Process posterior updating;
- Dirichlet Kernel Process modeling;
- supported kernel functions;
- kernel hyperparameter tuning;
- prior specification;
- posterior prediction and uncertainty quantification;
- global–local approximation in TwinBKP and TwinDKP.

### Interactive demonstrations

Interactive visualizations show how kernel parameters and prediction locations affect:

- kernel weights;
- posterior means;
- posterior uncertainty;
- global and local information sharing;
- TwinBKP approximation behavior.

### Examples

The website presents selected synthetic and real-data examples from the BKP software paper and its reproducibility repository.

### Resources

Direct links are provided to the R package, CRAN release, software paper, source code, and reproducibility materials.

## Related resources

| Resource | Description |
|----------------------------------|-------------------------------------|
| [BKP website](https://jiangyan-zhao.github.io/BKP-website/) | Interactive companion website |
| [BKP on CRAN](https://cran.r-project.org/web/packages/BKP/index.html) | Stable release of the R package |
| [BKP package repository](https://github.com/Jiangyan-Zhao/BKP) | Package source code, documentation, tests, and development version |
| [BKP paper repository](https://github.com/Jiangyan-Zhao/BKP-paper) | Paper source, replication scripts, data instructions, figures, and slides |
| [Software paper on arXiv](https://arxiv.org/abs/2508.10447) | Methodological and software description |

## Installing the BKP package

Install the stable version from CRAN:

``` r
install.packages("BKP")
```

Install the development version from GitHub using `pak`:

``` r
install.packages("pak")
pak::pak("Jiangyan-Zhao/BKP")
```

A minimal BKP workflow is:

``` r
library(BKP)

set.seed(123)

X <- matrix(seq(-2, 2, length.out = 30), ncol = 1)
m <- rep(100, nrow(X))
true_probability <- 1 / (1 + exp(-3 * X[, 1]))
y <- rbinom(nrow(X), size = m, prob = true_probability)

fit <- fit_BKP(
  X = X,
  y = y,
  m = m,
  Xbounds = matrix(c(-2, 2), nrow = 1)
)

pred <- predict(
  fit,
  Xnew = matrix(seq(-2, 2, length.out = 200), ncol = 1)
)

plot(fit)
```

For complete package documentation, examples, and function references, see the [BKP package repository](https://github.com/Jiangyan-Zhao/BKP) and the [CRAN reference manual](https://cran.r-project.org/web/packages/BKP/BKP.pdf).

## Technology

The website is implemented using:

- [React](https://react.dev/);
- [TypeScript](https://www.typescriptlang.org/);
- [Vite](https://vite.dev/);
- [KaTeX](https://katex.org/) for mathematical typesetting;
- GitHub Actions and GitHub Pages for continuous deployment.

## Local development

### Requirements

- Node.js 22 or later;
- npm.

Clone the repository:

``` bash
git clone https://github.com/Jiangyan-Zhao/BKP-website.git
cd BKP-website
```

Install the dependencies:

``` bash
npm install
```

Start the local development server:

``` bash
npm run dev
```

Open the local URL printed by Vite in a web browser.

## Production build

Create an optimized production build:

``` bash
npm run build
```

The generated files are written to the `dist/` directory.

Preview the production build locally:

``` bash
npm run preview
```

## Deployment

The website is deployed automatically through GitHub Actions.

Each push to the `main` branch triggers the deployment workflow. The workflow:

1. runs an external-link check for the website and README;
2. installs the project dependencies and creates the production build;
3. uploads the contents of `dist/`;
4. deploys the resulting artifact to GitHub Pages.

The workflow can also be run manually from the **Actions** tab of the repository.

## Reproducibility

The figures and statistical examples presented on the website are based on the BKP software paper and its associated reproducibility repository:

<https://github.com/Jiangyan-Zhao/BKP-paper>

The replication repository contains:

- R scripts for the synthetic examples;
- scripts and instructions for the real-data applications;
- generated figures;
- paper source files;
- presentation slides;
- dependency information.

The website itself is intended for presentation and interaction. The complete statistical analyses should be reproduced from the `BKP-paper` repository rather than from this frontend repository.

## Citation

When referring to the BKP methodology or software paper, please cite:

``` bibtex
@Misc{Zhao2025BKP,
  title  = {BKP: An R Package for Beta Kernel Process Modeling},
  author = {Jiangyan Zhao and Kunhai Qing and Jin Xu},
  year   = {2025},
  note   = {arXiv:2508.10447},
  url    = {https://arxiv.org/abs/2508.10447},
  doi    = {10.48550/arXiv.2508.10447}
}
```

When referring specifically to the R package, please cite:

``` bibtex
@Manual{BKP2026,
  title  = {BKP: Beta Kernel Process Modeling},
  author = {Jiangyan Zhao and Kunhai Qing and Jin Xu},
  year   = {2026},
  note   = {R package version 0.3.1},
  url    = {https://cran.r-project.org/package=BKP},
  doi    = {10.32614/CRAN.package.BKP}
}
```

The recommended citation for the installed package can also be obtained in R:

``` r
citation("BKP")
```

## Contributing

Bug reports, documentation corrections, and suggestions for improving the website may be submitted through the repository’s [issue tracker](https://github.com/Jiangyan-Zhao/BKP-website/issues).

For issues concerning the statistical methodology or R package implementation, use the [BKP package issue tracker](https://github.com/Jiangyan-Zhao/BKP/issues).

## License

Copyright © 2026 Jiangyan Zhao.

The source code of the BKP website is licensed under the GNU General Public License version 3 or, at your option, any later version (`GPL-3.0-or-later`). See [LICENSE](LICENSE) for details.
