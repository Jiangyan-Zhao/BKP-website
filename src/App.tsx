import { useMemo, useState } from "react";

type Mode = "Binary" | "Binomial" | "Multinomial";

const modeData: Record<
  Mode,
  { alpha: number; beta: number; interval: [number, number]; observations: number[]; label: string }
> = {
  Binary: {
    alpha: 8,
    beta: 5,
    interval: [0.42, 0.78],
    observations: [0.29, 0.35, 0.4, 0.52, 0.54, 0.7, 0.75, 0.79, 0.87, 0.92],
    label: "Bernoulli responses",
  },
  Binomial: {
    alpha: 13,
    beta: 7,
    interval: [0.51, 0.76],
    observations: [0.33, 0.4, 0.47, 0.51, 0.58, 0.64, 0.72, 0.81, 0.88],
    label: "Aggregated counts",
  },
  Multinomial: {
    alpha: 6,
    beta: 8,
    interval: [0.24, 0.59],
    observations: [0.12, 0.2, 0.27, 0.34, 0.39, 0.45, 0.53, 0.61, 0.72],
    label: "Class-one marginal",
  },
};

const chart = { left: 56, top: 24, width: 612, height: 248 };

function density(x: number, alpha: number, beta: number) {
  if (x <= 0 || x >= 1) return 0;
  return Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1);
}

function GithubIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="20" height="20">
      <path
        fill="currentColor"
        d="M12 .7a11.5 11.5 0 0 0-3.64 22.42c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.36-3.9-1.36-.52-1.34-1.28-1.69-1.28-1.69-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.47.11-3.05 0 0 .97-.31 3.16 1.18A10.95 10.95 0 0 1 12 6.13c.98 0 1.96.13 2.88.39 2.19-1.49 3.15-1.18 3.15-1.18.63 1.58.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.06.79 2.14v3.26c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"
      />
    </svg>
  );
}

function PosteriorChart() {
  const [mode, setMode] = useState<Mode>("Binary");
  const [query, setQuery] = useState(0.63);
  const current = modeData[mode];

  const curves = useMemo(() => {
    const points = Array.from({ length: 101 }, (_, i) => i / 100);
    const posteriorRaw = points.map((x) => density(x, current.alpha, current.beta));
    const priorRaw = points.map((x) => density(x, 2, 2));
    const posteriorMax = Math.max(...posteriorRaw);
    const priorMax = Math.max(...priorRaw);
    const makePath = (values: number[], max: number) =>
      values
        .map((value, i) => {
          const x = chart.left + (i / 100) * chart.width;
          const y = chart.top + chart.height - (value / max) * (chart.height - 18);
          return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
    return {
      posterior: makePath(posteriorRaw, posteriorMax),
      prior: makePath(priorRaw, priorMax * 1.7),
      queryDensity: density(query, current.alpha, current.beta) / posteriorMax,
    };
  }, [current, query]);

  const queryX = chart.left + query * chart.width;
  const queryY = chart.top + chart.height - curves.queryDensity * (chart.height - 18);

  function setFromPointer(clientX: number, element: SVGSVGElement) {
    const rect = element.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * 720;
    const value = (svgX - chart.left) / chart.width;
    setQuery(Math.min(0.95, Math.max(0.05, Number(value.toFixed(2)))));
  }

  return (
    <div className="chart-shell" aria-label="Interactive posterior probability explorer">
      <div className="chart-topline">
        <div className="mode-switch" role="tablist" aria-label="Response type">
          {(Object.keys(modeData) as Mode[]).map((item) => (
            <button
              className={mode === item ? "active" : ""}
              key={item}
              onClick={() => setMode(item)}
              role="tab"
              aria-selected={mode === item}
            >
              {item}
            </button>
          ))}
        </div>
        <div className="legend" aria-hidden="true">
          <span><i className="legend-line" />Posterior</span>
          <span><i className="legend-band" />95% CrI</span>
        </div>
      </div>

      <div className="chart-stage">
        <svg
          className="posterior-chart"
          viewBox="0 0 720 330"
          role="img"
          aria-label={`${mode} posterior density. Query probability ${query.toFixed(2)}.`}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            setFromPointer(event.clientX, event.currentTarget);
          }}
          onPointerMove={(event) => {
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              setFromPointer(event.clientX, event.currentTarget);
            }
          }}
        >
          <defs>
            <linearGradient id="credible" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#c8f135" stopOpacity=".42" />
              <stop offset="1" stopColor="#c8f135" stopOpacity=".08" />
            </linearGradient>
            <filter id="tooltipShadow" x="-20%" y="-20%" width="140%" height="150%">
              <feDropShadow dx="0" dy="8" stdDeviation="8" floodOpacity=".16" />
            </filter>
          </defs>

          {[0, 1, 2, 3, 4].map((tick) => {
            const y = chart.top + (tick / 4) * chart.height;
            return <line key={tick} x1={chart.left} x2={chart.left + chart.width} y1={y} y2={y} className="grid-line" />;
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const x = chart.left + tick * chart.width;
            return (
              <g key={tick}>
                <line x1={x} x2={x} y1={chart.top} y2={chart.top + chart.height} className="grid-line" />
                <text x={x} y="302" textAnchor="middle" className="axis-label">{tick.toFixed(2)}</text>
              </g>
            );
          })}

          <rect
            x={chart.left + current.interval[0] * chart.width}
            y={chart.top}
            width={(current.interval[1] - current.interval[0]) * chart.width}
            height={chart.height}
            fill="url(#credible)"
          />
          <path d={curves.prior} className="prior-path" />
          <path d={curves.posterior} className="posterior-path" />

          {current.observations.map((x, index) => (
            <circle
              key={`${mode}-${index}`}
              cx={chart.left + x * chart.width}
              cy={chart.top + chart.height - 7 - (index % 2) * 5}
              r="5"
              className="observation"
            />
          ))}

          <line x1={queryX} x2={queryX} y1={chart.top} y2={chart.top + chart.height} className="query-line" />
          <circle cx={queryX} cy={queryY} r="8" className="query-point" />
          <circle cx={queryX} cy={chart.top + chart.height} r="8" className="query-point" />

          <g
            className="tooltip-card"
            transform={`translate(${Math.min(queryX + 18, 492)}, ${Math.max(queryY - 28, 34)})`}
            filter="url(#tooltipShadow)"
          >
            <rect width="172" height="76" rx="9" />
            <text x="15" y="27" className="tooltip-value">p = {query.toFixed(2)}</text>
            <text x="15" y="50" className="tooltip-label">{current.label}</text>
            <text x="15" y="67" className="tooltip-label">95% CrI [{current.interval[0]}, {current.interval[1]}]</text>
          </g>
          <text x="362" y="326" textAnchor="middle" className="axis-title">Probability</text>
        </svg>
      </div>

      <label className="query-control">
        <span>Explore probability</span>
        <input
          type="range"
          min="5"
          max="95"
          value={Math.round(query * 100)}
          onChange={(event) => setQuery(Number(event.target.value) / 100)}
          aria-label="Posterior query probability"
        />
        <output>{query.toFixed(2)}</output>
      </label>
    </div>
  );
}

type ExampleKey = "curve" | "surface" | "warbler";

const examples: Record<
  ExampleKey,
  {
    title: string;
    description: string;
    tag: string;
    figure: string;
    alt: string;
    source: string;
    sourceUrl: string;
  }
> = {
  curve: {
    title: "Nonlinear BKP posterior",
    description: "Example 2 shows the fitted posterior mean, pointwise 95% credible interval, observed proportions, and the true nonlinear probability function.",
    tag: "BKP · Example 2",
    figure: "/results/ex2.pdf",
    alt: "BKP posterior mean and 95 percent credible interval for the nonlinear binomial example",
    source: "s4_ex2_bkp_1d_nonlinear.R",
    sourceUrl: "https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s4_ex2_bkp_1d_nonlinear.R",
  },
  surface: {
    title: "Iris multiclass classification",
    description: "Example 7 visualizes DKP classification regions and the maximum predicted class probability, revealing where class assignments are less decisive.",
    tag: "DKP · Example 7",
    figure: "/results/ex7.pdf",
    alt: "DKP predicted classes and maximum predicted probability for the Iris data",
    source: "s4_ex7_dkp_iris_classification.R",
    sourceUrl: "https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s4_ex7_dkp_iris_classification.R",
  },
  warbler: {
    title: "Mourning Warbler distribution",
    description: "The real-data application maps training presences, training absences, and spatially withheld test observations across North America.",
    tag: "BKP · Real data",
    figure: "/results/mourning-warbler-distribution.pdf",
    alt: "Map of Mourning Warbler training presences, training absences, and withheld test locations",
    source: "s5_app2_mourning_warbler_sdm.R",
    sourceUrl: "https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s5_app2_mourning_warbler_sdm.R",
  },
};

function ExampleExplorer() {
  const [selected, setSelected] = useState<ExampleKey>("curve");
  const figureUrl = `${import.meta.env.BASE_URL}${examples[selected].figure.replace(/^\//, "")}`;
  return (
    <div className="example-explorer">
      <div className="example-tabs" role="tablist" aria-label="Visualization examples">
        {(Object.keys(examples) as ExampleKey[]).map((key, index) => (
          <button key={key} role="tab" aria-selected={selected === key} className={selected === key ? "active" : ""} onClick={() => setSelected(key)}>
            <span>0{index + 1}</span>{examples[key].title}
          </button>
        ))}
      </div>
      <article className="example-display">
        <div className="example-visual">
          <object
            key={selected}
            className={`paper-figure ${selected}`}
            data={`${figureUrl}#view=FitH&toolbar=0&navpanes=0`}
            type="application/pdf"
            aria-label={examples[selected].alt}
          >
            <a href={figureUrl} target="_blank" rel="noreferrer">Open the original PDF figure ↗</a>
          </object>
        </div>
        <div className="example-caption">
          <span>{examples[selected].tag}</span>
          <h3>{examples[selected].title}</h3>
          <p>{examples[selected].description}</p>
          <a href={figureUrl} target="_blank" rel="noreferrer">
            <b>Open original PDF ↗</b>
            <small>Vector figure from BKP-paper</small>
          </a>
          <a href={examples[selected].sourceUrl} target="_blank" rel="noreferrer">
            <b>Reproduce this figure ↗</b>
            <small>{examples[selected].source}</small>
          </a>
        </div>
      </article>
    </div>
  );
}

function InstallPanel() {
  const [source, setSource] = useState<"CRAN" | "GitHub">("CRAN");
  const [copied, setCopied] = useState(false);
  const command = source === "CRAN" ? 'install.packages("BKP")' : 'pak::pak("Jiangyan-Zhao/BKP")';

  async function copyCommand() {
    await navigator.clipboard.writeText(command);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <div className="install-panel">
      <div className="terminal-bar">
        <div><i /><i /><i /></div>
        <span>R console</span>
        <div className="source-toggle" role="tablist" aria-label="Install source">
          {(["CRAN", "GitHub"] as const).map((item) => (
            <button key={item} role="tab" aria-selected={source === item} className={source === item ? "active" : ""} onClick={() => setSource(item)}>{item}</button>
          ))}
        </div>
      </div>
      <div className="install-command">
        <code><span>&gt;</span> {command}</code>
        <button onClick={copyCommand} aria-label="Copy installation command">{copied ? "Copied" : "Copy"}</button>
      </div>
      <pre aria-label="BKP quick start"><code><span># fit a probability surface</span>{`\nlibrary(BKP)\nfit <- fit_BKP(X, y, m, Xbounds = Xbounds)\npred <- predict(fit, Xnew = Xnew)\nplot(fit)`}</code></pre>
    </div>
  );
}

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#overview" aria-label="BKP home">BKP</a>
        <nav aria-label="Primary navigation">
          <a className="active" href="#overview">Overview</a>
          <a href="#method">Method</a>
          <a href="#examples">Examples</a>
          <a href="#install">Install</a>
        </nav>
        <div className="header-links">
          <a href="https://github.com/Jiangyan-Zhao/BKP" target="_blank" rel="noreferrer">
            <GithubIcon /> GitHub
          </a>
          <a href="https://cran.r-project.org/package=BKP" target="_blank" rel="noreferrer">
            <span className="r-mark">R</span> R package
          </a>
        </div>
      </header>

      <section className="hero" id="overview">
        <div className="hero-copy">
          <a className="status-pill" href="https://cran.r-project.org/package=BKP" target="_blank" rel="noreferrer">
            <span>✓</span> CRAN · stable <b>0.3.1</b>
          </a>
          <p className="eyebrow">Beta Kernel Process</p>
          <h1>BKP</h1>
          <h2>Closed-form inference,<br />directly on the probability scale.</h2>
          <p className="hero-description">
            Model binary, binomial, categorical, and multinomial outcomes with
            localized kernels and conjugate Beta or Dirichlet updates.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#install">Get started <span>→</span></a>
            <a className="button secondary" href="https://github.com/Jiangyan-Zhao/BKP" target="_blank" rel="noreferrer">
              <GithubIcon /> View on GitHub
            </a>
          </div>
          <div className="hero-meta" aria-label="Project highlights">
            <span><i>β</i> Closed-form</span>
            <span><i>R</i> R package</span>
            <span><i>◇</i> Four model families</span>
          </div>
        </div>

        <div className="hero-visual">
          <span className="formula-note" aria-hidden="true">
            p ∈ (0, 1)<br />θ ~ Beta(a₀, b₀)<br />yᵢ ~ Bernoulli(pᵢ)
          </span>
          <PosteriorChart />
        </div>
      </section>

      <section className="method-strip" id="method" aria-labelledby="method-title">
        <div className="strip-heading">
          <span className="binder-hole" aria-hidden="true" />
          <h2 id="method-title">Method in a nutshell</h2>
        </div>
        <ol>
          <li><span>1</span><p>Choose a Beta or Dirichlet prior at each input.</p></li>
          <li><span>2</span><p>Borrow nearby information with kernel weights.</p></li>
          <li><span>3</span><p>Obtain a closed-form posterior and credible interval.</p></li>
        </ol>
        <div className="mini-density" aria-hidden="true">⌁</div>
      </section>

      <section className="examples-section" id="examples" aria-labelledby="examples-title">
        <div className="section-intro">
          <p className="section-kicker">One grammar, four paths</p>
          <h2 id="examples-title">Pick the model that matches the outcome.</h2>
          <p>BKP keeps the workflow familiar while changing the response model and computational strategy.</p>
        </div>

        <div className="model-grid">
          <article><span>01</span><h3>BKP</h3><p>Binary or aggregated binomial responses.</p><code>fit_BKP()</code></article>
          <article><span>02</span><h3>DKP</h3><p>Categorical or multinomial responses.</p><code>fit_DKP()</code></article>
          <article><span>03</span><h3>TwinBKP</h3><p>Scalable global–local binomial modeling.</p><code>fit_TwinBKP()</code></article>
          <article><span>04</span><h3>TwinDKP</h3><p>Scalable global–local multiclass modeling.</p><code>fit_TwinDKP()</code></article>
        </div>

        <ExampleExplorer />
      </section>

      <section className="start-section" id="install" aria-labelledby="install-title">
        <div className="start-copy">
          <p className="section-kicker">Start in R</p>
          <h2 id="install-title">From data to posterior in a few lines.</h2>
          <p>
            The stable release is available on CRAN. Fit the model, predict at new inputs,
            and visualize posterior summaries through familiar S3 methods.
          </p>
          <InstallPanel />
        </div>

        <aside className="resource-stack" aria-label="Project resources">
          <a className="paper-card" href="https://arxiv.org/abs/2508.10447" target="_blank" rel="noreferrer">
            <span className="resource-type">Paper · arXiv</span>
            <h3>BKP: An R Package for Beta Kernel Process Modeling</h3>
            <p>Jiangyan Zhao, Kunhai Qing, and Jin Xu</p>
            <b>Read the paper <i>↗</i></b>
          </a>
          <a className="resource-card blue" href="https://github.com/Jiangyan-Zhao/BKP" target="_blank" rel="noreferrer">
            <span>GitHub</span><b>Source, issues & contributions</b><i>↗</i>
          </a>
          <a className="resource-card lime" href="https://cran.r-project.org/package=BKP" target="_blank" rel="noreferrer">
            <span>CRAN · 0.3.1</span><b>Stable package & manual</b><i>↗</i>
          </a>
          <a className="resource-card coral" href="https://github.com/Jiangyan-Zhao/BKP-paper" target="_blank" rel="noreferrer">
            <span>Reproduce</span><b>Code, figures & manuscript</b><i>↗</i>
          </a>
        </aside>
      </section>

      <footer>
        <a className="footer-mark" href="#overview">BKP</a>
        <p>Beta Kernel Process Modeling in R.</p>
        <div>
          <span>GPL-3.0</span>
          <a href="https://github.com/Jiangyan-Zhao/BKP/issues" target="_blank" rel="noreferrer">Report an issue ↗</a>
        </div>
      </footer>
    </main>
  );
}
