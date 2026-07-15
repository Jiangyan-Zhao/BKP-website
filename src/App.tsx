import { useMemo, useState } from "react";

type Mode = "Binary" | "Binomial" | "Multinomial";

type ModeDataset = {
  model: "BKP" | "DKP";
  label: string;
  responseLabel: string;
  xs: number[];
  counts: number[][];
  theta: number;
};

const modeData: Record<Mode, ModeDataset> = {
  Binary: {
    model: "BKP",
    label: "Binary · BKP",
    responseLabel: "Bernoulli observations",
    xs: [0.06, 0.13, 0.2, 0.28, 0.35, 0.43, 0.5, 0.58, 0.65, 0.72, 0.8, 0.87, 0.94],
    counts: [[0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0]],
    theta: 0.2,
  },
  Binomial: {
    model: "BKP",
    label: "Binomial · BKP",
    responseLabel: "Observed proportions",
    xs: [0.05, 0.14, 0.23, 0.32, 0.41, 0.5, 0.59, 0.68, 0.77, 0.86, 0.95],
    counts: [[2, 18], [3, 17], [5, 15], [9, 11], [14, 6], [17, 3], [16, 4], [12, 8], [7, 13], [4, 16], [3, 17]],
    theta: 0.18,
  },
  Multinomial: {
    model: "DKP",
    label: "Multinomial · DKP",
    responseLabel: "Class 1 proportions",
    xs: [0.05, 0.14, 0.23, 0.32, 0.41, 0.5, 0.59, 0.68, 0.77, 0.86, 0.95],
    counts: [[16, 3, 1], [15, 4, 1], [12, 6, 2], [9, 8, 3], [6, 10, 4], [4, 11, 5], [3, 9, 8], [2, 7, 11], [2, 5, 13], [1, 4, 15], [1, 3, 16]],
    theta: 0.2,
  },
};

const chart = { left: 58, top: 22, width: 604, height: 226, weightTop: 286, weightHeight: 32 };

function logGamma(z: number): number {
  const coefficients = [
    676.5203681218851, -1259.1392167224028, 771.3234287776531,
    -176.6150291621406, 12.507343278686905, -0.13857109526572012,
    9.984369578019572e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  let x = 0.9999999999998099;
  const shifted = z - 1;
  coefficients.forEach((coefficient, index) => { x += coefficient / (shifted + index + 1); });
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
}

function betaContinuedFraction(a: number, b: number, x: number) {
  const maxIterations = 100;
  const epsilon = 3e-9;
  const tiny = 1e-30;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const m2 = 2 * iteration;
    let aa = (iteration * (b - iteration) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = -((a + iteration) * (qab + iteration) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < epsilon) break;
  }
  return h;
}

function regularizedBeta(x: number, a: number, b: number) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const factor = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return (factor * betaContinuedFraction(a, b, x)) / a;
  return 1 - (factor * betaContinuedFraction(b, a, 1 - x)) / b;
}

function betaQuantile(probability: number, a: number, b: number) {
  let low = 0;
  let high = 1;
  for (let iteration = 0; iteration < 38; iteration += 1) {
    const midpoint = (low + high) / 2;
    if (regularizedBeta(midpoint, a, b) < probability) low = midpoint;
    else high = midpoint;
  }
  return (low + high) / 2;
}

function kernelWeight(x: number, observedX: number, theta: number) {
  return Math.exp(-Math.pow((x - observedX) / theta, 2));
}

function posteriorAt(x: number, dataset: ModeDataset, theta: number) {
  const classCount = dataset.counts[0].length;
  const shapes = Array.from({ length: classCount }, () => 1);
  let localCount = 0;
  dataset.xs.forEach((observedX, observationIndex) => {
    const weight = kernelWeight(x, observedX, theta);
    dataset.counts[observationIndex].forEach((count, classIndex) => {
      shapes[classIndex] += weight * count;
      localCount += classIndex === 0 ? weight * count : 0;
    });
  });
  const total = shapes.reduce((sum, value) => sum + value, 0);
  const alpha = shapes[0];
  const beta = total - alpha;
  return {
    mean: alpha / total,
    low: betaQuantile(0.025, alpha, beta),
    high: betaQuantile(0.975, alpha, beta),
    shapes,
    pseudoCount: total - classCount,
    weightedSuccesses: localCount,
  };
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
  const [query, setQuery] = useState(0.58);
  const [theta, setTheta] = useState(modeData.Binary.theta);
  const current = modeData[mode];

  const curve = useMemo(() => {
    const points = Array.from({ length: 101 }, (_, i) => i / 100);
    const summaries = points.map((x) => posteriorAt(x, current, theta));
    const makePath = (values: number[]) => values.map((value, index) => {
      const px = chart.left + points[index] * chart.width;
      const py = chart.top + (1 - value) * chart.height;
      return `${index === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(" ");
    const upper = summaries.map((summary) => summary.high);
    const lower = summaries.map((summary) => summary.low);
    const band = `${makePath(upper)} ${lower.map((value, reverseIndex) => {
      const index = lower.length - 1 - reverseIndex;
      const px = chart.left + points[index] * chart.width;
      const py = chart.top + (1 - value) * chart.height;
      return `L${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(" ")} Z`;
    return {
      mean: makePath(summaries.map((summary) => summary.mean)),
      band,
    };
  }, [current, theta]);

  const queryPosterior = useMemo(() => posteriorAt(query, current, theta), [query, current, theta]);
  const queryX = chart.left + query * chart.width;
  const queryY = chart.top + (1 - queryPosterior.mean) * chart.height;
  const queryLowY = chart.top + (1 - queryPosterior.low) * chart.height;
  const queryHighY = chart.top + (1 - queryPosterior.high) * chart.height;

  function setFromPointer(clientX: number, element: SVGSVGElement) {
    const rect = element.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * 720;
    const value = (svgX - chart.left) / chart.width;
    setQuery(Math.min(0.95, Math.max(0.05, Number(value.toFixed(2)))));
  }

  return (
    <div className="chart-shell" aria-label="Interactive Beta Kernel Process explorer">
      <div className="chart-topline">
        <div className="mode-switch" role="tablist" aria-label="Response type">
          {(Object.keys(modeData) as Mode[]).map((item) => (
            <button
              className={mode === item ? "active" : ""}
              key={item}
              onClick={() => {
                setMode(item);
                setTheta(modeData[item].theta);
              }}
              role="tab"
              aria-selected={mode === item}
            >
              {modeData[item].label}
            </button>
          ))}
        </div>
        <div className="legend" aria-hidden="true">
          <span><i className="legend-line" />Posterior mean</span>
          <span><i className="legend-band" />95% pointwise CrI</span>
          <span><i className="legend-weight" />Kernel weight</span>
        </div>
      </div>

      <div className="chart-stage">
        <svg
          className="posterior-chart"
          viewBox="0 0 720 382"
          role="img"
          aria-label={`${current.label} probability surface. Query input ${query.toFixed(2)}.`}
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

          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const y = chart.top + (1 - tick) * chart.height;
            return (
              <g key={tick}>
                <line x1={chart.left} x2={chart.left + chart.width} y1={y} y2={y} className="grid-line" />
                <text x={chart.left - 12} y={y + 4} textAnchor="end" className="axis-label">{tick.toFixed(2)}</text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const x = chart.left + tick * chart.width;
            return (
              <g key={tick}>
                <line x1={x} x2={x} y1={chart.top} y2={chart.top + chart.height} className="grid-line" />
                <text x={x} y="270" textAnchor="middle" className="axis-label">{tick.toFixed(2)}</text>
              </g>
            );
          })}

          <path d={curve.band} className="posterior-band" />
          <path d={curve.mean} className="posterior-path" />

          {current.xs.map((x, index) => {
            const total = current.counts[index].reduce((sum, value) => sum + value, 0);
            const proportion = current.counts[index][0] / total;
            const weight = kernelWeight(query, x, theta);
            return (
            <g key={`${mode}-${index}`}>
              <rect
                x={chart.left + x * chart.width - 3}
                y={chart.weightTop + chart.weightHeight * (1 - weight)}
                width="6"
                height={chart.weightHeight * weight}
                rx="3"
                className="weight-bar"
              />
            <circle
              cx={chart.left + x * chart.width}
              cy={chart.top + (1 - proportion) * chart.height}
              r={current.model === "BKP" && total === 1 ? 5.5 : Math.min(8, 4 + total / 10)}
              className="observation"
            />
            </g>
          );})}

          <line x1={queryX} x2={queryX} y1={chart.top} y2={chart.top + chart.height} className="query-line" />
          <line x1={queryX} x2={queryX} y1={queryHighY} y2={queryLowY} className="interval-line" />
          <line x1={queryX - 7} x2={queryX + 7} y1={queryHighY} y2={queryHighY} className="interval-line" />
          <line x1={queryX - 7} x2={queryX + 7} y1={queryLowY} y2={queryLowY} className="interval-line" />
          <circle cx={queryX} cy={queryY} r="8" className="query-point" />

          <g
            className="tooltip-card"
            transform={`translate(${Math.min(queryX + 18, 458)}, ${Math.max(queryY - 42, 28)})`}
            filter="url(#tooltipShadow)"
          >
            <rect width="205" height="96" rx="9" />
            <text x="15" y="25" className="tooltip-value">x₀ = {query.toFixed(2)}</text>
            <text x="15" y="48" className="tooltip-label">π̂(x₀) = {queryPosterior.mean.toFixed(3)}</text>
            <text x="15" y="66" className="tooltip-label">95% CrI [{queryPosterior.low.toFixed(3)}, {queryPosterior.high.toFixed(3)}]</text>
            <text x="15" y="84" className="tooltip-label">weighted trials = {queryPosterior.pseudoCount.toFixed(1)}</text>
          </g>
          <text x="18" y="136" textAnchor="middle" transform="rotate(-90 18 136)" className="axis-title">Probability</text>
          <text x="360" y="341" textAnchor="middle" className="weight-label">Gaussian weights k(x₀, xᵢ)</text>
          <text x="360" y="374" textAnchor="middle" className="axis-title">Input x</text>
        </svg>
      </div>

      <div className="chart-controls">
        <label className="query-control">
          <span>Query input x₀</span>
          <input type="range" min="2" max="98" value={Math.round(query * 100)} onChange={(event) => setQuery(Number(event.target.value) / 100)} aria-label="BKP query input" />
          <output>{query.toFixed(2)}</output>
        </label>
        <label className="query-control">
          <span>Length scale θ</span>
          <input type="range" min="8" max="38" value={Math.round(theta * 100)} onChange={(event) => setTheta(Number(event.target.value) / 100)} aria-label="Gaussian kernel length scale" />
          <output>{theta.toFixed(2)}</output>
        </label>
      </div>
      <p className="chart-equation">
        {current.model === "BKP"
          ? "αₙ(x₀)=α₀+Σ k(x₀,xᵢ)yᵢ · βₙ(x₀)=β₀+Σ k(x₀,xᵢ)(mᵢ−yᵢ)"
          : "DKP class-1 marginal · αₙⱼ(x₀)=α₀ⱼ+Σ k(x₀,xᵢ)yᵢⱼ"}
      </p>
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
            π(x) | 𝒟ₙ ~ Beta(αₙ(x), βₙ(x))<br />αₙ(x) = α₀(x) + Σ k(x,xᵢ)yᵢ<br />βₙ(x) = β₀(x) + Σ k(x,xᵢ)(mᵢ−yᵢ)
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
