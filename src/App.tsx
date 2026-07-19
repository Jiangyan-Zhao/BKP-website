import { useEffect, useMemo, useState } from "react";
import katex from "katex";

const packageVersion = "0.3.1";

type Example2Observation = {
  x: number;
  trials: number;
  successes: number;
};

// Exact data used by BKP-paper/code/s4_ex2_bkp_1d_nonlinear.R (set.seed(123)).
const example2Data: Example2Observation[] = [
  { x: -0.19398072, trials: 14, successes: 13 },
  { x: -0.70462216, trials: 67, successes: 11 },
  { x: -0.63037211, trials: 42, successes: 7 },
  { x: -1.74727409, trials: 50, successes: 28 },
  { x: 0.24049041, trials: 43, successes: 26 },
  { x: -0.86055712, trials: 14, successes: 2 },
  { x: 1.9804429, trials: 25, successes: 13 },
  { x: -0.08221174, trials: 90, successes: 84 },
  { x: 1.4936816, trials: 91, successes: 52 },
  { x: -1.16105319, trials: 69, successes: 36 },
  { x: -1.46745774, trials: 91, successes: 49 },
  { x: 0.75452924, trials: 57, successes: 10 },
  { x: -1.423341, trials: 92, successes: 47 },
  { x: -0.3557257, trials: 9, successes: 5 },
  { x: 0.82813919, trials: 93, successes: 29 },
  { x: -1.30019818, trials: 99, successes: 54 },
  { x: 0.52700304, trials: 72, successes: 13 },
  { x: 1.41500019, trials: 26, successes: 17 },
  { x: -1.64618964, trials: 7, successes: 4 },
  { x: -1.88528663, trials: 42, successes: 20 },
  { x: -0.93352701, trials: 9, successes: 3 },
  { x: 1.12523789, trials: 83, successes: 46 },
  { x: 1.26768178, trials: 36, successes: 20 },
  { x: 1.73026749, trials: 78, successes: 42 },
  { x: -0.50043451, trials: 81, successes: 24 },
  { x: 0.54768276, trials: 43, successes: 10 },
  { x: 0.28751215, trials: 76, successes: 45 },
  { x: 0.96260663, trials: 15, successes: 6 },
  { x: 1.81017985, trials: 32, successes: 15 },
  { x: 0.08972538, trials: 7, successes: 7 },
];

const example2Fit = {
  gamma: -1.4130744966,
  theta: 0.0386300707,
  brier: 0.0061712436,
  initialGamma: [-1.3494850022, 1] as const,
  optimizerGamma: [-3, 3] as const,
};

const chart = { left: 62, top: 50, width: 648, height: 258, weightTop: 348, weightHeight: 34 };

function MathFormula({ children, display = false, label }: { children: string; display?: boolean; label?: string }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: display, throwOnError: false, strict: "ignore" }),
    [children, display],
  );
  return (
    <span
      className={display ? "math-formula display" : "math-formula"}
      aria-label={label}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

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

function trueExample2(x: number) {
  const expNegativeX = Math.exp(-x);
  return (1 + Math.exp(-x * x) * Math.cos((10 * (1 - expNegativeX)) / (1 + expNegativeX))) / 2;
}

function kernelWeight(x: number, observedX: number, theta: number) {
  // fit_BKP() first rescales Xbounds = [-2, 2] to the unit interval.
  const normalizedDistance = (x - observedX) / 4;
  return Math.exp(-Math.pow(normalizedDistance / theta, 2));
}

function posteriorAt(x: number, theta: number) {
  let alpha = 1;
  let beta = 1;
  let weightedTrials = 0;
  example2Data.forEach((observation) => {
    const weight = kernelWeight(x, observation.x, theta);
    alpha += weight * observation.successes;
    beta += weight * (observation.trials - observation.successes);
    weightedTrials += weight * observation.trials;
  });
  return {
    mean: alpha / (alpha + beta),
    low: betaQuantile(0.025, alpha, beta),
    high: betaQuantile(0.975, alpha, beta),
    weightedTrials,
  };
}

function loocvBrier(gamma: number) {
  const theta = Math.pow(10, gamma);
  const score = example2Data.reduce((sum, observation, observationIndex) => {
    let alpha = 1;
    let beta = 1;
    example2Data.forEach((neighbor, neighborIndex) => {
      if (neighborIndex === observationIndex) return;
      const weight = kernelWeight(observation.x, neighbor.x, theta);
      alpha += weight * neighbor.successes;
      beta += weight * (neighbor.trials - neighbor.successes);
    });
    const prediction = alpha / (alpha + beta);
    return sum + Math.pow(prediction - observation.successes / observation.trials, 2);
  }, 0);
  return score / example2Data.length;
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
  const [query, setQuery] = useState(-0.1);
  const [gamma, setGamma] = useState(example2Fit.gamma);
  const theta = Math.pow(10, gamma);
  const originalFigureUrl = `${import.meta.env.BASE_URL}results/ex2.pdf`;

  const curve = useMemo(() => {
    const points = Array.from({ length: 161 }, (_, i) => -2 + (4 * i) / 160);
    const summaries = points.map((x) => posteriorAt(x, theta));
    const makePath = (values: number[]) => values.map((value, index) => {
      const px = chart.left + ((points[index] + 2) / 4) * chart.width;
      const py = chart.top + (1 - value) * chart.height;
      return `${index === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(" ");
    const upper = summaries.map((summary) => summary.high);
    const lower = summaries.map((summary) => summary.low);
    const band = `${makePath(upper)} ${lower.map((value, reverseIndex) => {
      const index = lower.length - 1 - reverseIndex;
      const px = chart.left + ((points[index] + 2) / 4) * chart.width;
      const py = chart.top + (1 - value) * chart.height;
      return `L${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(" ")} Z`;
    return {
      mean: makePath(summaries.map((summary) => summary.mean)),
      truth: makePath(points.map(trueExample2)),
      band,
    };
  }, [theta]);

  const queryPosterior = useMemo(() => posteriorAt(query, theta), [query, theta]);
  const queryTruth = trueExample2(query);
  const currentBrier = useMemo(() => loocvBrier(gamma), [gamma]);
  const queryX = chart.left + ((query + 2) / 4) * chart.width;
  const queryY = chart.top + (1 - queryPosterior.mean) * chart.height;
  const queryLowY = chart.top + (1 - queryPosterior.low) * chart.height;
  const queryHighY = chart.top + (1 - queryPosterior.high) * chart.height;

  function setFromPointer(clientX: number, element: SVGSVGElement) {
    const rect = element.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * 772;
    const value = ((svgX - chart.left) / chart.width) * 4 - 2;
    setQuery(Math.min(2, Math.max(-2, Number(value.toFixed(2)))));
  }

  function formatTheta(value: number) {
    if (value < 0.01) return value.toExponential(2);
    if (value < 1) return value.toFixed(3);
    if (value < 100) return value.toFixed(2);
    return value.toFixed(0);
  }

  return (
    <div className="chart-shell" aria-label="Interactive Beta Kernel Process explorer">
      <div className="chart-topline">
        <div className="paper-example-heading">
          <p><span>Paper · Example 2</span> Nonlinear binomial curve</p>
          <h3>Explore the fitted BKP probability surface</h3>
          <small>
            <MathFormula>{"n=30"}</MathFormula><i>·</i>
            <MathFormula>{"x\\in[-2,2]"}</MathFormula><i>·</i>
            noninformative <MathFormula>{"\\operatorname{Beta}(1,1)"}</MathFormula> prior
          </small>
          <a className="paper-example-link" href={originalFigureUrl} target="_blank" rel="noreferrer">Original PDF ↗</a>
        </div>
        <div className="legend" aria-hidden="true">
          <span><i className="legend-line" />Posterior mean</span>
          <span><i className="legend-truth" />True <MathFormula>{"\\pi_2(x)"}</MathFormula></span>
          <span><i className="legend-band" />95% pointwise CrI</span>
          <span><i className="legend-dot" />Observed <MathFormula>{"y_i/m_i"}</MathFormula></span>
        </div>
      </div>

      <div className="chart-stage">
        <svg
          className="posterior-chart"
          viewBox="0 0 772 424"
          role="img"
          aria-label={`BKP-paper Example 2 probability surface. Query input ${query.toFixed(2)} and theta ${theta.toPrecision(3)}.`}
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
            <clipPath id="plotClip">
              <rect x={chart.left} y={chart.top} width={chart.width} height={chart.height} />
            </clipPath>
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
          {[-2, -1, 0, 1, 2].map((tick) => {
            const x = chart.left + ((tick + 2) / 4) * chart.width;
            return (
              <g key={tick}>
                <line x1={x} x2={x} y1={chart.top} y2={chart.top + chart.height} className="grid-line" />
                <text x={x} y="330" textAnchor="middle" className="axis-label">{tick}</text>
              </g>
            );
          })}

          <g clipPath="url(#plotClip)">
            <path d={curve.band} className="posterior-band" />
            <path d={curve.truth} className="truth-path" />
            <path d={curve.mean} className="posterior-path" pathLength="1" />
          </g>

          {example2Data.map((observation, index) => {
            const proportion = observation.successes / observation.trials;
            const weight = kernelWeight(query, observation.x, theta);
            const x = chart.left + ((observation.x + 2) / 4) * chart.width;
            return (
            <g key={index}>
              <rect
                x={x - 2.5}
                y={chart.weightTop + chart.weightHeight * (1 - weight)}
                width="5"
                height={chart.weightHeight * weight}
                rx="2.5"
                className="weight-bar"
              />
              <circle
                cx={x}
                cy={Math.min(chart.top + chart.height - 5, Math.max(chart.top + 5, chart.top + (1 - proportion) * chart.height))}
                r={3.4 + Math.sqrt(observation.trials) / 2.8}
                className="observation"
              >
                <title>{`x = ${observation.x.toFixed(3)} · y/m = ${observation.successes}/${observation.trials}`}</title>
              </circle>
            </g>
          );})}

          <line x1={queryX} x2={queryX} y1={chart.top} y2={chart.top + chart.height} className="query-line" />
          <line x1={queryX} x2={queryX} y1={queryHighY} y2={queryLowY} className="interval-line" />
          <line x1={queryX - 7} x2={queryX + 7} y1={queryHighY} y2={queryHighY} className="interval-line" />
          <line x1={queryX - 7} x2={queryX + 7} y1={queryLowY} y2={queryLowY} className="interval-line" />
          <circle cx={queryX} cy={queryY} r="8" className="query-point" />

          <line x1={chart.left} x2={chart.left + chart.width} y1={chart.weightTop + chart.weightHeight} y2={chart.weightTop + chart.weightHeight} className="weight-baseline" />
          <g transform="translate(19 180) rotate(-90)" aria-hidden="true">
            <foreignObject className="chart-axis-math" x="-48" y="-17" width="96" height="34">
              <div><MathFormula>{"\\pi_2(x)"}</MathFormula></div>
            </foreignObject>
          </g>
          <foreignObject className="chart-axis-math weight-axis-math" x={chart.left} y="388" width="104" height="28" aria-hidden="true">
            <div><MathFormula>{"k(x_0,x_i)"}</MathFormula></div>
          </foreignObject>
          <foreignObject className="chart-axis-math" x={chart.left + chart.width / 2 - 40} y="394" width="80" height="28" aria-hidden="true">
            <div><MathFormula>{"x"}</MathFormula></div>
          </foreignObject>
        </svg>
      </div>

      <div className="query-readout" aria-label="Current BKP query summary">
        <span><small><MathFormula>{"x_0"}</MathFormula></small><b>{query.toFixed(2)}</b></span>
        <span><small>BKP mean <MathFormula>{"\\widehat{\\pi}(x_0)"}</MathFormula></small><b>{queryPosterior.mean.toFixed(3)}</b></span>
        <span><small>True <MathFormula>{"\\pi_2(x_0)"}</MathFormula></small><b>{queryTruth.toFixed(3)}</b></span>
        <span><small>95% CrI</small><b>[{queryPosterior.low.toFixed(3)}, {queryPosterior.high.toFixed(3)}]</b></span>
        <span><small>Weighted trials</small><b>{queryPosterior.weightedTrials.toFixed(1)}</b></span>
      </div>

      <div className="chart-controls">
        <label className="query-control">
          <span>Query input <MathFormula>{"x_0"}</MathFormula></span>
          <input type="range" min="-200" max="200" value={Math.round(query * 100)} onChange={(event) => setQuery(Number(event.target.value) / 100)} aria-label="BKP query input" />
          <output>{query.toFixed(2)}</output>
        </label>
        <label className="query-control gamma-control">
          <span>Log length scale <MathFormula>{"\\gamma"}</MathFormula></span>
          <input type="range" min="-300" max="300" value={Math.round(gamma * 100)} onChange={(event) => setGamma(Number(event.target.value) / 100)} aria-label="Log base ten Gaussian kernel length scale" />
          <output>{gamma.toFixed(2)}</output>
        </label>
      </div>
      <div className="fit-summary">
        <div>
          <span><MathFormula>{"\\theta=10^\\gamma"}</MathFormula> = <b>{formatTheta(theta)}</b></span>
          <span>LOOCV Brier = <b>{currentBrier.toFixed(5)}</b></span>
        </div>
        <button type="button" onClick={() => setGamma(example2Fit.gamma)} disabled={Math.abs(gamma - example2Fit.gamma) < 0.005}>
          Use paper fit <MathFormula>{"\\widehat{\\theta}"}</MathFormula> = {example2Fit.theta.toFixed(4)}
        </button>
      </div>
      <p className="range-note">
        <span>Initial LHD <MathFormula>{"\\Omega_0"}</MathFormula>: <MathFormula>{`\\gamma\\in[${example2Fit.initialGamma[0].toFixed(2)},${example2Fit.initialGamma[1].toFixed(2)}]`}</MathFormula> · <MathFormula>{"\\theta\\in[0.0447,10]"}</MathFormula></span>
        <span>Optimizer <MathFormula>{"\\Omega"}</MathFormula>: <MathFormula>{`\\gamma\\in[${example2Fit.optimizerGamma[0]},${example2Fit.optimizerGamma[1]}]`}</MathFormula> · <MathFormula>{"\\theta\\in[0.001,1000]"}</MathFormula></span>
      </p>
    </div>
  );
}

const twinBkpGlobalIndices = [
  6, 52, 79, 89, 124, 142, 155, 177, 190, 217, 232,
  259, 274, 318, 337, 370, 381, 388, 395, 450, 478, 495,
] as const;

type TwinBkpObservation = {
  x: number;
  trials: number;
  successes: number;
  proportion: number;
};

function createSeededRandom(seed: number) {
  let state = seed;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function makeTwinBkpObservations(): TwinBkpObservation[] {
  const random = createSeededRandom(123);
  return Array.from({ length: 500 }, (_, index) => {
    // Example 8 uses a one-dimensional LHS over [-2, 2].
    const x = -2 + ((index + random()) * 4) / 500;
    const trials = 1 + Math.floor(random() * 100);
    const probability = trueExample2(x);
    let successes = 0;
    for (let trial = 0; trial < trials; trial += 1) successes += random() < probability ? 1 : 0;
    return { x, trials, successes, proportion: successes / trials };
  });
}

const twinBkpObservations = makeTwinBkpObservations();
const twinChart = { left: 84, top: 36, width: 824, height: 324 };

function TwinBkpExplorer() {
  const [query, setQuery] = useState(0.15);
  const globalSet = useMemo(() => new Set<number>(twinBkpGlobalIndices), []);
  const localIndices = useMemo(
    () => twinBkpObservations
      .map((observation, index) => ({ x: observation.x, index, distance: Math.abs(observation.x - query) }))
      .filter(({ index }) => !globalSet.has(index))
      .sort((left, right) => left.distance - right.distance)
      .slice(0, 25)
      .map(({ index }) => index),
    [globalSet, query],
  );
  const localSet = useMemo(() => new Set(localIndices), [localIndices]);
  const paperUrl = `${import.meta.env.BASE_URL}results/ex8.pdf`;
  const xPosition = (x: number) => twinChart.left + ((x + 2) / 4) * twinChart.width;
  const yPosition = (probability: number) => twinChart.top + (1 - probability) * twinChart.height;
  const setQueryFromPlot = (clientX: number, svg: SVGSVGElement) => {
    const bounds = svg.getBoundingClientRect();
    if (bounds.width === 0) return;

    const svgX = ((clientX - bounds.left) / bounds.width) * 1160;
    const value = ((svgX - twinChart.left) / twinChart.width) * 4 - 2;
    setQuery(Math.min(2, Math.max(-2, Number(value.toFixed(2)))));
  };
  const truthPath = useMemo(
    () => Array.from({ length: 321 }, (_, index) => {
      const x = -2 + (index * 4) / 320;
      return `${index === 0 ? "M" : "L"}${xPosition(x).toFixed(2)},${yPosition(trueExample2(x)).toFixed(2)}`;
    }).join(" "),
    [],
  );

  return (
    <article className="twin-showcase" aria-labelledby="twin-title">
      <header className="twin-heading">
        <div>
          <p className="section-kicker">Paper · Example 8</p>
          <h3 id="twin-title">Global coverage, local refinement.</h3>
        </div>
        <p>
          Blue circles represent the shared global subset; green diamonds follow the
          testing location as its query-specific nearest neighbours.
        </p>
      </header>

      <figure className="twin-figure">
        <svg viewBox="0 0 1160 420" role="img" aria-label={`TwinBKP global and local training points for query x zero equals ${query.toFixed(2)}`}>
          <rect className="twin-plot-background" x={twinChart.left} y={twinChart.top} width={twinChart.width} height={twinChart.height} />
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={`y-${tick}`}>
              <line className="twin-plot-grid" x1={twinChart.left} x2={twinChart.left + twinChart.width} y1={yPosition(tick)} y2={yPosition(tick)} />
              <text className="twin-plot-tick" x={twinChart.left - 16} y={yPosition(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text>
            </g>
          ))}
          {[-2, -1, 0, 1, 2].map((tick) => (
            <g key={`x-${tick}`}>
              <line className="twin-plot-grid" x1={xPosition(tick)} x2={xPosition(tick)} y1={twinChart.top} y2={twinChart.top + twinChart.height} />
              <text className="twin-plot-tick" x={xPosition(tick)} y={twinChart.top + twinChart.height + 26} textAnchor="middle">{tick}</text>
            </g>
          ))}

          <path className="twin-true-curve" d={truthPath} />
          {twinBkpObservations.map((observation, index) => (
            <circle
              key={`training-${index}`}
              className="twin-training-point"
              cx={xPosition(observation.x)}
              cy={yPosition(observation.proportion)}
              r="1.55"
            />
          ))}

          <line className="twin-query-guide" x1={xPosition(query)} x2={xPosition(query)} y1={twinChart.top} y2={twinChart.top + twinChart.height} />
          {twinBkpGlobalIndices.map((index) => {
            const observation = twinBkpObservations[index];
            return <circle key={`global-${index}`} className="twin-global-point" cx={xPosition(observation.x)} cy={yPosition(observation.proportion)} r="6.8" />;
          })}
          {twinBkpObservations.map((observation, index) => {
            if (!localSet.has(index)) return null;
            const x = xPosition(observation.x);
            const y = yPosition(observation.proportion);
            return <polygon key={`local-${index}`} className="twin-local-point" points={`${x},${y - 6.5} ${x + 6.5},${y} ${x},${y + 6.5} ${x - 6.5},${y}`} />;
          })}

          <rect
            x={twinChart.left}
            y={twinChart.top}
            width={twinChart.width}
            height={twinChart.height}
            fill="transparent"
            cursor="crosshair"
            onPointerDown={(event) => {
              const svg = event.currentTarget.ownerSVGElement;
              if (svg) setQueryFromPlot(event.clientX, svg);
            }}
          />

          <foreignObject className="twin-axis-math" x="456" y={twinChart.top + twinChart.height + 29} width="80" height="28" aria-hidden="true">
            <div><MathFormula>{"x"}</MathFormula></div>
          </foreignObject>
          <g transform={`translate(28 ${twinChart.top + twinChart.height / 2}) rotate(-90)`} aria-hidden="true">
            <foreignObject className="twin-axis-math" x="-60" y="-18" width="120" height="36">
              <div><MathFormula>{"y_i/m_i"}</MathFormula></div>
            </foreignObject>
          </g>

          <g className="twin-plot-legend" transform="translate(962 160)">
            <rect x="-15" y="-24" width="190" height="139" rx="7" />
            <circle className="twin-training-point legend-point" cx="0" cy="0" r="3" /><text x="20" y="5">Training data</text>
            <path className="twin-test-point" d="M-4,19 L4,27 M4,19 L-4,27" /><text x="20" y="27">Testing location</text>
            <line className="twin-true-curve" x1="-6" x2="7" y1="47" y2="47" /><text x="20" y="52">True function</text>
            <circle className="twin-global-point" cx="0" cy="72" r="6" /><text x="20" y="77">Global point · 22</text>
            <polygon className="twin-local-point" points="0,90 6,96 0,102 -6,96" /><text x="20" y="101">Local point · 25</text>
          </g>
        </svg>
        <div className="twin-slider-row">
          <label className="twin-slider">
            <span>Click the plot or move testing location <MathFormula>{"x_0"}</MathFormula></span>
            <input
              type="range"
              min="-2"
              max="2"
              step="0.01"
              value={query}
              onChange={(event) => setQuery(Number(event.target.value))}
            />
            <output>{query.toFixed(2)}</output>
          </label>
        </div>
        <figcaption>
          <b>Interactive reconstruction.</b> This browser-generated illustration follows the settings of
          Example 8 but is not the paper&apos;s exact R random sample. The exact fitted result is shown in the
          original vector PDF alongside it.
        </figcaption>
      </figure>

      <div className="twin-lower">
        <section className="twin-math-panel" aria-labelledby="twin-math-title">
          <p className="section-kicker" id="twin-math-title">Local–global posterior update</p>
          <MathFormula display label="Prediction subset is the union of global and local subsets">
            {"\\mathcal I(x_0)=\\mathcal G\\cup\\mathcal L(x_0),\\qquad |\\mathcal G|=22,\\quad|\\mathcal L(x_0)|=25"}
          </MathFormula>
          <MathFormula display label="TwinBKP posterior conditional on the prediction subset is a beta distribution">
            {"\\pi(x_0)\\mid\\mathcal D_{\\mathcal I(x_0)}\\sim\\operatorname{Beta}\\!\\left(\\alpha_T(x_0),\\beta_T(x_0)\\right)"}
          </MathFormula>
          <div className="twin-math-stages">
            <div>
              <span>01 · Global Gaussian</span>
              <MathFormula display>{"\\begin{aligned}\\alpha_G&=\\alpha_0+\\sum_{i\\in\\mathcal G} k_g(x_0,x_i)y_i\\\\\\beta_G&=\\beta_0+\\sum_{i\\in\\mathcal G} k_g(x_0,x_i)(m_i-y_i)\\end{aligned}"}</MathFormula>
            </div>
            <div>
              <span>02 · Local Wendland</span>
              <MathFormula display>{"\\begin{aligned}\\alpha_T&=\\alpha_G+\\sum_{i\\in\\mathcal L(x_0)} k_\\ell(x_0,x_i)y_i\\\\\\beta_T&=\\beta_G+\\sum_{i\\in\\mathcal L(x_0)} k_\\ell(x_0,x_i)(m_i-y_i)\\end{aligned}"}</MathFormula>
            </div>
          </div>
          <p className="twin-parameters"><span>Gaussian <MathFormula>{"\\theta_g=0.05"}</MathFormula></span><span>Wendland <MathFormula>{"\\theta_\\ell=0.0545"}</MathFormula></span><span>5 twinning runs</span></p>
        </section>

        <aside className="twin-paper">
          <div className="twin-paper-label">
            <span>Original vector result</span>
            <b>Example 8 · TwinBKP fit</b>
          </div>
          <object
            className="twin-paper-figure"
            data={`${paperUrl}#view=FitH&toolbar=0&navpanes=0`}
            type="application/pdf"
            aria-label="Original TwinBKP result from paper Example 8"
          >
            <a href={paperUrl} target="_blank" rel="noreferrer">Open the original PDF figure ↗</a>
          </object>
          <div className="twin-paper-links">
            <a href={paperUrl} target="_blank" rel="noreferrer">Open original PDF ↗</a>
            <a href="https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s4_ex8_twinbkp_1d_nonlinear.R" target="_blank" rel="noreferrer">Reproduce Example 8 ↗</a>
          </div>
        </aside>
      </div>
    </article>
  );
}

type ExampleKey = "surface" | "twindkp" | "warbler";

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
  surface: {
    title: "Iris multiclass classification",
    description: "The original Example 7 figure shows the DKP classification regions and maximum predicted class probability, making uncertain decision boundaries visible.",
    tag: "DKP · Example 7",
    figure: "/results/ex7.pdf",
    alt: "DKP predicted classes and maximum predicted probability for the Iris data",
    source: "s4_ex7_dkp_iris_classification.R",
    sourceUrl: "https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s4_ex7_dkp_iris_classification.R",
  },
  twindkp: {
    title: "TwinDKP multinomial curves",
    description: "Example 9 applies the scalable TwinDKP global–local approximation to 500 observations with three-class multinomial responses.",
    tag: "TwinDKP · Example 9",
    figure: "/results/ex9.pdf",
    alt: "TwinDKP posterior summaries and true probability curves for three multinomial classes",
    source: "s4_ex9_twindkp_1d_multinomial.R",
    sourceUrl: "https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s4_ex9_twindkp_1d_multinomial.R",
  },
  warbler: {
    title: "Mourning Warbler predictions",
    description: "This real-data result compares BKP, TwinBKP, and LGP predictions of presence probability and their posterior uncertainty across North America.",
    tag: "BKP · Real data",
    figure: "/results/mourning-warbler-predictions.pdf",
    alt: "BKP, TwinBKP, and LGP predictions of Mourning Warbler presence probability and posterior uncertainty",
    source: "s5_app2_mourning_warbler_sdm.R",
    sourceUrl: "https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/code/s5_app2_mourning_warbler_sdm.R",
  },
};

function ExampleExplorer() {
  const [selected, setSelected] = useState<ExampleKey>("surface");
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
  const quickStartComment = "# complete reproducible example";
  const quickStartBody = `library(BKP)
set.seed(123)
X <- matrix(seq(-2, 2, length.out = 30), ncol = 1)
m <- rep(50, nrow(X))
prob <- plogis(2 * X[, 1])
y <- rbinom(nrow(X), size = m, prob = prob)
bounds <- matrix(c(-2, 2), nrow = 1)
fit <- fit_BKP(X, y, m, Xbounds = bounds)
Xnew <- matrix(seq(-2, 2, length.out = 100), ncol = 1)
pred <- predict(fit, Xnew = Xnew)
head(pred)
plot(fit, engine = "ggplot")`;
  const completeExample = `${command}

${quickStartComment}
${quickStartBody}`;

  async function copyExample() {
    await navigator.clipboard.writeText(completeExample);
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
        <button onClick={copyExample} aria-label="Copy complete R example">{copied ? "Copied" : "Copy all"}</button>
      </div>
      <pre aria-label="Complete reproducible BKP quick start"><code><span>{quickStartComment}</span>{`\n${quickStartBody}`}</code></pre>
    </div>
  );
}

const bkpBibtex = `@Misc{Zhao2025BKP,
  title  = {BKP: An R Package for Beta Kernel Process Modeling},
  author = {Jiangyan Zhao and Kunhai Qing and Jin Xu},
  year   = {2025},
  note   = {arXiv:2508.10447},
  url    = {https://arxiv.org/abs/2508.10447},
  doi    = {10.48550/arXiv.2508.10447}
}

@Manual{Zhao2026BKPpackage,
  title  = {BKP: Beta Kernel Process Modeling},
  author = {Jiangyan Zhao and Kunhai Qing and Jin Xu},
  year   = {2026},
  note   = {R package version 0.3.1},
  url    = {https://cran.r-project.org/package=BKP},
  doi    = {10.32614/CRAN.package.BKP}
}`;

function CitationPanel() {
  const [copied, setCopied] = useState(false);

  async function copyBibtex() {
    await navigator.clipboard.writeText(bkpBibtex);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <article className="citation-panel" aria-labelledby="citation-title">
      <div className="citation-intro">
        <p className="section-kicker">Cite the project</p>
        <h3 id="citation-title">Using BKP in your work?</h3>
        <p>
          Cite the paper for the methodology and the package for the software version.
          If your work relies on both, include both entries.
        </p>
        <a href="https://cran.r-project.org/web/packages/BKP/citation.html" target="_blank" rel="noreferrer">
          View CRAN citation information ↗
        </a>
      </div>
      <div className="citation-code">
        <div>
          <span>BibTeX · paper + package</span>
          <button type="button" onClick={copyBibtex} aria-label="Copy BKP paper and package BibTeX">
            {copied ? "Copied" : "Copy both"}
          </button>
        </div>
        <pre><code>{bkpBibtex}</code></pre>
      </div>
    </article>
  );
}

export default function Home() {
  const [activeSection, setActiveSection] = useState("overview");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let frame = 0;
    let timer = 0;

    const getRequestedSection = () => {
      const rawHash = window.location.hash.slice(1);

      if (rawHash) {
        try {
          return decodeURIComponent(rawHash);
        } catch {
          return rawHash;
        }
      }

      return new URLSearchParams(window.location.search).get("section") ?? "";
    };

    const scrollToRequestedSection = (behavior: ScrollBehavior) => {
      const sectionId = getRequestedSection();
      if (!sectionId) return;

      const section = document.getElementById(sectionId);
      if (!section) return;

      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        section.scrollIntoView({
          behavior,
          block: "start",
        });
      });
    };

    // Run after React has rendered the page.
    scrollToRequestedSection("auto");

    // Repeat after fonts, figures, and other layout-dependent content settle.
    timer = window.setTimeout(() => {
      scrollToRequestedSection("auto");
    }, 300);

    const handleHashChange = () => scrollToRequestedSection("smooth");
    window.addEventListener("hashchange", handleHashChange);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(timer);
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  useEffect(() => {
    const sectionIds = ["overview", "install", "method", "examples", "resources"];
    let frame = 0;
    const updateActiveSection = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const marker = window.scrollY + Math.min(180, window.innerHeight * 0.28);
        let current = sectionIds[0];
        sectionIds.forEach((id) => {
          const section = document.getElementById(id);
          if (section && section.offsetTop <= marker) current = id;
        });
        setActiveSection(current);
      });
    };

    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, { passive: true });
    window.addEventListener("resize", updateActiveSection);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", updateActiveSection);
      window.removeEventListener("resize", updateActiveSection);
    };
  }, []);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileMenuOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#overview" aria-label="BKP home">BKP</a>
        <nav
          id="primary-navigation"
          className={mobileMenuOpen ? "open" : ""}
          aria-label="Primary navigation"
          onClick={() => setMobileMenuOpen(false)}
        >
          <a className={activeSection === "overview" ? "active" : ""} href="#overview">Overview</a>
          <a className={activeSection === "install" ? "active" : ""} href="#install">Install</a>
          <a className={activeSection === "method" ? "active" : ""} href="#method">Method</a>
          <a className={activeSection === "examples" ? "active" : ""} href="#examples">Examples</a>
          <a className={activeSection === "resources" ? "active" : ""} href="#resources">Resources</a>
          <a className="mobile-nav-external" href="https://cran.r-project.org/package=BKP" target="_blank" rel="noreferrer">R package ↗</a>
        </nav>
        <div className="header-links">
          <a href="https://github.com/Jiangyan-Zhao/BKP" target="_blank" rel="noreferrer">
            <GithubIcon /> GitHub
          </a>
          <a href="https://cran.r-project.org/package=BKP" target="_blank" rel="noreferrer">
            <img className="r-project-logo" src="https://www.r-project.org/logo/Rlogo.svg" alt="" aria-hidden="true" /> R package
          </a>
        </div>
        <button
          className="menu-toggle"
          type="button"
          aria-expanded={mobileMenuOpen}
          aria-controls="primary-navigation"
          aria-label={mobileMenuOpen ? "Close navigation" : "Open navigation"}
          onClick={() => setMobileMenuOpen((open) => !open)}
        >
          <i /><i /><i />
        </button>
      </header>

      <section className="hero" id="overview">
        <div className="hero-copy">
          <a className="status-pill" href="https://cran.r-project.org/package=BKP" target="_blank" rel="noreferrer">
            <span>✓</span> CRAN · stable <b>{packageVersion}</b>
          </a>
          <p className="eyebrow">Beta Kernel Process</p>
          <h1>BKP</h1>
          <h2>Closed-form inference,<br />directly on the probability scale.</h2>
          <p className="hero-description">
            Fit covariate-dependent binary, binomial, categorical, and multinomial
            probabilities with localized kernels and conjugate Beta or Dirichlet updates—
            without a latent Gaussian layer or MCMC.
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
          <div className="formula-note" aria-hidden="true">
            <b>Paper · Example 2</b>
            <MathFormula display>{"\\pi_2(x)=\\tfrac12\\!\\left[1+e^{-x^2}\\cos\\!\\left(10\\tanh(x/2)\\right)\\right]"}</MathFormula>
            <small><MathFormula>{"x\\in[-2,2],\\qquad n=30"}</MathFormula></small>
          </div>
          <PosteriorChart />
        </div>
      </section>

      <section className="install-section" id="install" aria-labelledby="install-title">
        <div className="install-intro">
          <p className="section-kicker">Start in R</p>
          <h2 id="install-title">From data to posterior in a few lines.</h2>
          <p>
            Install the stable CRAN release, then fit, predict, simulate, and visualize
            posterior summaries through a consistent set of S3 methods.
          </p>
        </div>
        <InstallPanel />
      </section>

      <section className="method-section" id="method" aria-labelledby="method-title">
        <div className="method-strip">
          <div className="strip-heading">
            <span className="binder-hole" aria-hidden="true" />
            <h2 id="method-title">Method in a nutshell</h2>
          </div>
          <ol>
            <li><span>1</span><p>Place a Beta or Dirichlet prior directly on the response probability.</p></li>
            <li><span>2</span><p>Borrow evidence through localized, kernel-weighted pseudo-counts.</p></li>
            <li><span>3</span><p>Return closed-form posterior summaries, predictions, and uncertainty.</p></li>
          </ol>
          <div className="mini-density" aria-hidden="true">⌁</div>
        </div>

        <div className="method-body">
          <div className="lineage-block" aria-labelledby="lineage-title">
            <header className="lineage-heading">
              <div>
                <p className="section-kicker">Methodological lineage</p>
                <h2 id="lineage-title">From smoothed Beta distributions to a unified, scalable R framework.</h2>
              </div>
              <p>
                BKP is not a Gaussian process with its distribution swapped. It works directly
                on the probability scale: kernel weights aggregate neighboring Bernoulli or
                binomial observations as fractional pseudo-counts, followed by pointwise Beta
                or Dirichlet conjugate updating. The milestones below are related developments
                across several research communities rather than a strictly linear sequence.
              </p>
            </header>

            <dl className="lineage-terms" aria-label="Meaning of the Beta Kernel Process name">
              <div>
                <dt>Beta / Dirichlet</dt>
                <dd>Pointwise posterior-style distributions for local binomial or multinomial probabilities.</dd>
              </div>
              <div>
                <dt>Kernel</dt>
                <dd>Similarity-weighted borrowing of neighboring observations as fractional pseudo-counts.</dd>
              </div>
              <div>
                <dt>Process</dt>
                <dd>An input-indexed collection of distributions—not a GP-like joint stochastic-process posterior.</dd>
              </div>
            </dl>

            <ol className="lineage-timeline">
              <li className="lineage-step history">
                <a href="https://hal.science/hal-00637575/" target="_blank" rel="noreferrer">
                  <span className="lineage-label">2009–2012 · Robotics</span>
                  <h3>Smoothed beta distributions</h3>
                  <p>Used nonparametric smoothed Beta distributions to learn grasping probabilities from Bernoulli trials and guide active exploration.</p>
                  <small>Montesano and Lopes ↗</small>
                </a>
              </li>
              <li className="lineage-step history">
                <a href="https://dl.acm.org/doi/10.5555/2283516.2283608" target="_blank" rel="noreferrer">
                  <span className="lineage-label">2011 · AI and decision making</span>
                  <h3>Continuous Correlated Beta Processes</h3>
                  <p>Formalized kernel-weighted evidence sharing across a continuum of Bernoulli experiments, with a natural Dirichlet extension.</p>
                  <small>Goetschalckx et al. ↗</small>
                </a>
              </li>
              <li className="lineage-step history">
                <a href="https://doi.org/10.1002/sam.11241" target="_blank" rel="noreferrer">
                  <span className="lineage-label">2014 · Data mining</span>
                  <h3>Bayesian beta kernel model</h3>
                  <p>Developed probability-scale beta-kernel updating for binary classification, imbalanced data, and online learning.</p>
                  <small>MacKenzie et al. ↗</small>
                </a>
              </li>
              <li className="lineage-step history">
                <a href="https://proceedings.mlr.press/v97/rolland19a.html" target="_blank" rel="noreferrer">
                  <span className="lineage-label">2019 · Statistical ML</span>
                  <h3>Smooth Beta Process</h3>
                  <p>Studied sample-dependent local sharing and established convergence guarantees for learning smooth Bernoulli probability functions.</p>
                  <small>Rolland et al. ↗</small>
                </a>
              </li>
              <li className="lineage-step current">
                <a href="https://arxiv.org/abs/2508.10447" target="_blank" rel="noreferrer">
                  <span className="lineage-label">2025–2026 · Unified software</span>
                  <h3>BKP and DKP</h3>
                  <p>Consolidate, formalize, extend, and implement the method family for binomial and multinomial data, with priors, loss-based tuning, ESS calibration, and posterior tools.</p>
                  <small>BKP software paper ↗</small>
                </a>
              </li>
              <li className="lineage-step scalable">
                <a href="https://github.com/Jiangyan-Zhao/BKP" target="_blank" rel="noreferrer">
                  <span className="lineage-label">2026 · Scalable extension</span>
                  <h3>TwinBKP and TwinDKP</h3>
                  <p>Combine a representative global subset selected by twinning with prediction-specific nearest neighbors while retaining the conjugate pseudo-count update.</p>
                  <small>Package implementation ↗</small>
                </a>
              </li>
            </ol>

            <aside>
              <a
                className="lineage-frontier"
                href="https://arxiv.org/abs/2407.06321"
                target="_blank"
                rel="noreferrer"
                aria-label="Read Mussi et al. (2024) on the theoretical frontier"
              >
                <span>Open theoretical frontier · 2024</span>
                <p>
                  <strong>Mussi et al. ↗</strong>
                  {" "}explained why learning from kernelized Bernoulli observations is not a routine
                  Gaussian-to-Bernoulli substitution and highlighted the open challenge of tight
                  Bernoulli-specific guarantees for kernelized bandits.
                </p>
              </a>
            </aside>

            <section className="lineage-locality" aria-labelledby="locality-title">
              <header>
                <span>Localization clarified</span>
                <div>
                  <h3 id="locality-title">Locality at three levels.</h3>
                  <p>
                    “Local” may describe evidence weighting, compact kernel support, or
                    prediction-specific data subsets. TwinBKP is analogous in spirit to localized
                    GP ideas, but is adapted to BKP&apos;s conjugate pseudo-count structure.
                  </p>
                </div>
              </header>
              <ol>
                <li>
                  <small>Weight level</small>
                  <b>Soft local borrowing</b>
                  <span>Gaussian and Matérn kernels are globally supported but assign smoothly decaying weights.</span>
                </li>
                <li>
                  <small>Kernel level</small>
                  <b>Hard support</b>
                  <span>The Wendland kernel assigns exactly zero weight outside its support radius.</span>
                </li>
                <li>
                  <small>Subset level</small>
                  <b>Global–local approximation</b>
                  <span>Twin variants combine a representative global subset with prediction-specific nearest neighbors.</span>
                </li>
              </ol>
            </section>
          </div>

          <div className="section-intro">
            <p className="section-kicker">One grammar, four paths</p>
            <h2>Pick the model that matches the outcome.</h2>
            <p>Choose BKP or DKP by response type, then use a Twin variant when the data call for a scalable global–local approximation.</p>
          </div>

          <div className="capability-grid" aria-label="BKP modeling capabilities">
            <article><span>Kernels</span><b>Gaussian · Matérn 5/2 · Matérn 3/2 · Wendland</b></article>
            <article><span>Length scales</span><b>Isotropic or anisotropic</b></article>
            <article><span>Tuning</span><b>LOOCV · Brier score or log-loss</b></article>
            <article><span>ESS calibration</span><b>Optional Shepard scaling for BKP / DKP</b></article>
          </div>

          <div className="model-grid">
            <article><span>01</span><h3>BKP</h3><p>Full modeling for binary or aggregated binomial responses.</p><code>fit_BKP()</code></article>
            <article><span>02</span><h3>DKP</h3><p>Full modeling for categorical or multinomial responses.</p><code>fit_DKP()</code></article>
            <article><span>03</span><h3>TwinBKP</h3><p>A global–local approximation for larger binomial datasets.</p><code>fit_TwinBKP()</code></article>
            <article><span>04</span><h3>TwinDKP</h3><p>A global–local approximation for larger multiclass datasets.</p><code>fit_TwinDKP()</code></article>
          </div>
        </div>
      </section>

      <section className="examples-section" id="examples" aria-labelledby="examples-title">
        <div className="section-intro">
          <p className="section-kicker">Interactive example</p>
          <h2 id="examples-title">From mechanism to published evidence.</h2>
          <p>Move the testing location to see the local subset update around a fixed global design, then compare the reconstruction with the original paper results.</p>
        </div>

        <TwinBkpExplorer />

        <div className="results-intro">
          <div>
            <p className="section-kicker">Selected paper results</p>
            <h3>Compare fitted results. Reproduce each analysis.</h3>
          </div>
          <p>
            Each panel embeds an original vector PDF from BKP-paper. Open a figure at
            full resolution or follow its R script to reproduce the published result.
          </p>
        </div>
        <ExampleExplorer />
      </section>

      <section className="resources-section" id="resources" aria-labelledby="resources-title">
        <header className="resources-heading">
          <div>
            <p className="section-kicker">Project resources</p>
            <h2 id="resources-title">Read, reproduce, and contribute.</h2>
          </div>
          <p>
            Follow the paper, inspect the source, install the stable package, or reproduce
            the published analyses from the companion repository.
          </p>
        </header>
        <aside className="resource-stack" aria-label="Project resources">
          <article className="paper-card">
            <span className="resource-type">Software paper</span>
            <h3>BKP: An R Package for Beta Kernel Process Modeling</h3>
            <p>Jiangyan Zhao, Kunhai Qing, and Jin Xu</p>
            <div className="paper-card-links">
              <a href="https://arxiv.org/abs/2508.10447" target="_blank" rel="noreferrer"><b>Read on arXiv</b><i>↗</i></a>
              <a href="https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/paper/TR_BKP.pdf" target="_blank" rel="noreferrer"><b>Latest manuscript PDF</b><i>↗</i></a>
            </div>
          </article>
          <a className="resource-card blue" href="https://github.com/Jiangyan-Zhao/BKP" target="_blank" rel="noreferrer">
            <span>GitHub</span>
            <div className="resource-card-copy">
              <b>Source, issues & contributions</b>
              <small>If BKP supports your work, consider starring the repository.</small>
            </div>
            <i aria-hidden="true">★</i>
          </a>
          <a className="resource-card lime" href="https://cran.r-project.org/web/packages/BKP/refman/BKP.html" target="_blank" rel="noreferrer">
            <span>Documentation · {packageVersion}</span><b>Function reference & package manual</b><i>↗</i>
          </a>
          <a className="resource-card coral" href="https://github.com/Jiangyan-Zhao/BKP-paper" target="_blank" rel="noreferrer">
            <span>Reproduce</span><b>Code, figures & manuscript</b><i>↗</i>
          </a>
          <a className="resource-card soft" href="https://github.com/Jiangyan-Zhao/BKP-paper/blob/master/slides/BKP_Slides.pdf" target="_blank" rel="noreferrer">
            <span>Slides · PDF</span><b>Presentation deck</b><i>↗</i>
          </a>
        </aside>

        <section className="followup-research-section" aria-labelledby="followup-research-title">
          <header className="followup-research-heading">
            <div>
              <p className="section-kicker">Downstream applications</p>
              <h3 id="followup-research-title">BKP-based follow-up research</h3>
            </div>
            <p>
              Selected methodological work that carries BKP into domain-specific problems,
              beginning with Bayesian dose-finding in Phase I trials.
            </p>
          </header>

          <article className="followup-project-card" aria-labelledby="skbd-title">
            <div className="followup-project-copy">
              <span className="followup-project-domain">Phase I dose-finding</span>
              <h4 id="skbd-title">Shared Keyboard: An Improved Bayesian Design for Phase I Clinical Trials via Beta Kernel Process</h4>
              <p>
                SKBD brings the Beta kernel process into Phase I dose-finding. By borrowing
                information smoothly across neighboring dose levels, it supports coherent
                escalation and de-escalation decisions while preserving interpretable,
                dose-specific toxicity estimates.
              </p>
              <div className="followup-project-meta" aria-label="SKBD research scope">
                <span><small>Application</small><b>Phase I clinical trials</b></span>
                <span><small>Built on</small><b>Beta kernel process</b></span>
              </div>
            </div>

            <div className="followup-project-links" aria-label="SKBD research links">
              <a href="https://arxiv.org/abs/2605.25043" target="_blank" rel="noreferrer">
                <div><span>Paper</span><b>Read the preprint</b><small>arXiv:2605.25043</small></div>
                <i aria-hidden="true">↗</i>
              </a>
              <a href="https://github.com/Jiangyan-Zhao/SKBD" target="_blank" rel="noreferrer">
                <div><span>Method code</span><b>SKBD implementation</b><small>Jiangyan-Zhao/SKBD</small></div>
                <i aria-hidden="true">↗</i>
              </a>
              <a href="https://github.com/Jiangyan-Zhao/SKBD-paper" target="_blank" rel="noreferrer">
                <div><span>Reproduce</span><b>Analysis repository</b><small>Jiangyan-Zhao/SKBD-paper</small></div>
                <i aria-hidden="true">↗</i>
              </a>
            </div>
          </article>
        </section>

        <CitationPanel />
      </section>

      <footer>
        <a className="footer-mark" href="#overview">BKP</a>
        <p>Beta Kernel Process Modeling in R.</p>
        <div>
          <span>BKP website · GPL ≥ 3</span>
          <a href="https://github.com/Jiangyan-Zhao/BKP/issues" target="_blank" rel="noreferrer">Report an issue ↗</a>
        </div>
      </footer>
    </main>
  );
}
