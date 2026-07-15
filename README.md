# BKP website

Static React website for **Bayesian Kernel Projections (BKP)**. It includes an interactive posterior explorer, installation guidance, real results from the BKP paper, and links to the paper, GitHub repository, and CRAN.

## Local development

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

## Production build

```bash
npm run build
npm run preview
```

The GitHub Actions workflow deploys `dist/` to GitHub Pages after every push to `main`.

Live site: <https://jiangyan-zhao.github.io/BKP-website/>

[![Deploy to GitHub Pages](https://github.com/Jiangyan-Zhao/BKP-website/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/Jiangyan-Zhao/BKP-website/actions/workflows/deploy-pages.yml)
