// Measure experiments for N-body (js/examples/nbody.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureForces, measureEnergyDrift), which
// runs the shipped pipelines without a canvas; the CPU reference is the float64 direct sum of
// js/refs/nbody.js. Every sweep is a fixed list with hard caps (exported so test/nbody-measure.test.mjs
// can check them): Barnes-Hut N ≤ 65,536, brute force N ≤ 16,384, ≤ 400 leapfrog steps per run.
//
// Timing follows the shared protocol (PROTOCOL: 1 discarded warm-up, then 5 independent repeats, each a
// fresh measureForces call that sets up, uploads, dispatches once and reads back); timed columns are
// medians with their interquartile range. Errors and interaction counts are deterministic and come from
// the last repeat. Each experiment's verify() checks its rows against CPU references and CI bounds.

import { generateBodies, measureForces, measureEnergyDrift } from '../../examples/nbody.js';
import { directForcesAt, relErrors } from '../../refs/nbody.js';
import { summarize, withSpread, fitPowerLaw, slopeAgrees } from '../stats.js';
import { check } from '../result-record.js';

export const BH_MAX_N = 65536;
export const BRUTE_MAX_N = 16384;
export const SOFTENING = 0.05;      // the example's default ε
export const SEED = 5;              // the bodies of the CI force test (test/gpu/nbody.gpu.test.mjs)
export const PROTOCOL = { warmup: 1, repeats: 5 };

// ---- 1. Opening angle vs force error ----------------------------------------------------------------
export const THETAS = [0.3, 0.4, 0.5, 0.7, 0.9, 1.2];
export const ERROR_COUNTS = [4096, 16384];
export const SAMPLE = 512;          // target bodies checked against the direct sum (512 × N in float64)
// Measured in CI (Plummer, N = 4,096, seed 5, ε = 0.05): the CPU replica of cs_bh_force and the shipped
// shader on a software GPU agree on the median error at θ = 0.5
export const CI_MEDIAN_AT_HALF = 4.2e-3;
// Median relative force error on the SAMPLE targets, per θ in THETAS, from the CPU replica of cs_bh_force
// (test/helpers/nbody-refs.mjs) on these exact bodies (seed 5, ε = 0.05); test/nbody-measure.test.mjs
// recomputes them. The shipped shader matched the replica in CI (4.2e-3 at θ = 0.5, Plummer 4,096).
export const REPLICA_MEDIANS = {
    'plummer/4096': [1.51e-3, 2.58e-3, 4.19e-3, 8.60e-3, 1.50e-2, 2.61e-2],
    'plummer/16384': [8.58e-4, 1.65e-3, 2.64e-3, 5.27e-3, 9.16e-3, 1.70e-2],
    'spiral/4096': [2.70e-3, 5.26e-3, 9.16e-3, 1.83e-2, 2.78e-2, 4.82e-2],
    'spiral/16384': [1.92e-3, 3.68e-3, 6.01e-3, 1.17e-2, 2.04e-2, 3.43e-2],
};
// A GPU median within this factor of the replica's passes (the same factor as test/nbody.test.mjs's bounds;
// f32 vs float64 and tree-build order move the median by far less)
export const REPLICA_FACTOR = 1.7;

// Bodies spread evenly through the generation order, so bulge and disk are both represented
export function sampleTargets(n, count = SAMPLE) {
    return Array.from({ length: Math.min(count, n) }, (_, k) => Math.floor(k * n / Math.min(count, n)));
}

const presetSetting = {
    key: 'preset', label: 'Bodies',
    options: [{ value: 'plummer', label: 'Plummer sphere' }, { value: 'spiral', label: 'Spiral galaxy' }],
    default: 'plummer',
};

const ms = (v) => v.toFixed(2);
const sci = (v) => v.toExponential(2);
const sci1 = (v) => v.toExponential(1);

// Steps i where rows[i][key] does not exceed rows[i − 1][key] (direction +1) or does not fall below it (−1)
function monotoneViolations(rows, key, direction) {
    let bad = 0;
    for (let i = 1; i < rows.length; i++) {
        const a = rows[i - 1][key], b = rows[i][key];
        if (!(typeof a === 'number' && typeof b === 'number')) continue;
        if (direction > 0 ? !(b > a) : !(b < a)) bad++;
    }
    return bad;
}

// ---- Interpretation helpers (measured vs predicted, see "Interpretation" in js/teach/measure.js) -------
// Fits are least-squares lines in log-log space (fitPowerLaw): the slope is the measured exponent, with a
// 95% interval from the scatter of the points about the line (Student t, n − 2 degrees of freedom). With
// fewer than 3 points there is no interval and the verdict is null.

const f2 = (v) => v.toFixed(2);
const count = (n) => n.toLocaleString('en-US');
const pct = (v) => `${(100 * v).toFixed(0)}%`;

// Power-law fit of rows[yKey] against rows[xKey] (rows with a non-positive or missing value are skipped)
export function fitRows(rows, xKey, yKey) {
    return fitPowerLaw(rows.map(r => r[xKey]), rows.map(r => r[yKey]));
}

// "2.09 (95% CI 2.02 … 2.16, 6 points)"
export function slopeText(fit) {
    if (fit.slope === null) return `no fit: ${fit.n} usable point${fit.n === 1 ? '' : 's'}`;
    if (!fit.ci95) return `${f2(fit.slope)} (${fit.n} points: no uncertainty)`;
    return `${f2(fit.slope)} (95% CI ${f2(fit.ci95[0])} … ${f2(fit.ci95[1])}, ${fit.n} points)`;
}

// For predictions that are a range rather than a number: true if the whole 95% interval lies in [lo, hi],
// false if it lies wholly outside, null if it straddles a bound (the data cannot decide) or there is none
export function rangeVerdict(fit, lo, hi) {
    if (!fit.ci95) return null;
    const [a, b] = fit.ci95;
    if (a >= lo && b <= hi) return true;
    if (b < lo || a > hi) return false;
    return null;
}

const tooFew = (fit, what) => `needs at least 3 ${what} for an uncertainty (has ${fit.n})`;

// One force evaluation as an independent measurement for ctx.measure
const forceRun = (ctx, positions, options) => () => measureForces(ctx.gpuDevice,
    { positions, softening: SOFTENING, warmup: 0, repeats: 1, ...options });

const thetaError = {
    id: 'theta-error',
    title: 'Opening angle vs force error',
    question: 'How much accuracy does Barnes-Hut give up as the opening angle θ grows, and how much work does it save?',
    settings: [
        { key: 'n', label: 'Bodies', options: ERROR_COUNTS.map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 16384 },
        presetSetting,
    ],
    budgetMs: 20000,
    columns: [
        { key: 'theta', label: 'θ', format: String },
        { key: 'median', label: 'median rel. error', format: sci },
        { key: 'p99', label: 'p99 rel. error', format: sci },
        { key: 'interactions', label: 'interactions / body', format: (v) => v.toFixed(0) },
        { key: 'traversalMs', label: 'traversal GPU ms', format: ms, unit: 'ms' },
        { key: 'gpuMs', label: 'GPU ms (tree + traversal)', format: ms, unit: 'ms' },
        { key: 'wallMs', label: 'wall ms', format: ms, unit: 'ms' },
    ],
    charts: [
        {
            x: 'theta', xLabel: 'opening angle θ', logX: true,
            y: ['median', 'p99'], yLabel: 'relative force error', logY: true,
            labels: { median: 'median', p99: '99th percentile' },
            // θ² through the CI median at θ = 0.5 (a straight line on log-log axes)
            reference: {
                label: 'θ² (CI)',
                points: [THETAS[0], THETAS[THETAS.length - 1]].map(t => [t, CI_MEDIAN_AT_HALF * (t / 0.5) ** 2]),
            },
        },
        { x: 'theta', xLabel: 'opening angle θ', logX: true, y: ['interactions'], yLabel: 'interactions per body', logY: true },
    ],
    baseline: `CI (shipped shader on a software GPU, Plummer N = 4,096): median ${CI_MEDIAN_AT_HALF.toExponential(1)} at `
        + 'θ = 0.5; 1.5e-3 / 2.6e-3 / 4.2e-3 / 8.6e-3 / 1.5e-2 / 2.6e-2 at θ = 0.3 … 1.2, close to θ², with 997 → 134 '
        + `interactions per body (brute force: 4,095). Errors are checked on ${SAMPLE} bodies against a float64 direct sum.`,
    async *run(ctx, { n, preset }) {
        const N = Math.min(n, BH_MAX_N);
        const { positions } = generateBodies(preset, N, SOFTENING, SEED);
        const targets = sampleTargets(N);
        await ctx.step();
        const ref = directForcesAt(positions, N, SOFTENING * SOFTENING, targets);
        for (const theta of THETAS) {
            await ctx.step();
            const m = await ctx.measure(forceRun(ctx, positions, { mode: 'bh', theta }), PROTOCOL);
            const r = m.samples[m.samples.length - 1];
            const acc = new Float64Array(3 * targets.length);
            targets.forEach((i, t) => { for (let c = 0; c < 3; c++) acc[3 * t + c] = r.accel[4 * i + c]; });
            const e = relErrors(acc, ref.acc, targets.length);
            yield { theta, median: e.median, p99: e.p99, interactions: r.interactionsPerBody,
                ...withSpread('traversalMs', summarize(m.samples.map(s => s.passes?.force))),
                ...withSpread('gpuMs', m.gpuMs), ...withSpread('wallMs', m.wallMs) };
        }
    },
    // Each θ's median error within REPLICA_FACTOR of the CPU replica on the same bodies; error grows with θ
    verify(rows, { n, preset }) {
        const medians = REPLICA_MEDIANS[`${preset}/${n}`];
        const checks = THETAS.map((theta, i) => {
            const row = rows.find(r => r.theta === theta);
            const expected = medians?.[i];
            if (!expected) return { metric: `median rel. error at θ = ${theta}`, status: 'not-checked', value: null,
                tolerance: null, reference: `no replica value for ${preset}, N = ${n}` };
            const lo = +(expected / REPLICA_FACTOR).toPrecision(2), hi = +(expected * REPLICA_FACTOR).toPrecision(2);
            return check(`median rel. error at θ = ${theta}`, row?.median, { min: lo, max: hi,
                tolerance: `${sci1(lo)} … ${sci1(hi)}`,
                reference: `CPU replica of cs_bh_force on the same bodies: ${sci(expected)} (× / ÷ ${REPLICA_FACTOR})` });
        });
        checks.push(rows.length >= 2
            ? check('θ steps where the median error does not grow', monotoneViolations(rows, 'median', +1),
                { max: 0, tolerance: '0', reference: 'the error of the monopole approximation grows with θ (about θ²)' })
            : { metric: 'θ steps where the median error does not grow', value: null, tolerance: '0', reference: 'needs 2 rows', status: 'not-checked' });
        return checks;
    },
    interpret: interpretThetaError,
    scope: {
        shows: [
            'How the median and 99th-percentile relative force error of the shipped Barnes-Hut shader grow with θ, '
                + `measured on ${SAMPLE} bodies against a float64 direct sum.`,
            'How much work the tree saves at each θ (interactions per body; brute force needs N − 1), and what that does to the time.',
            'Whether your GPU computes the same approximation as the float64 CPU replica of the shader on the same bodies.',
        ],
        doesNotShow: [
            'Other distributions or random draws: one set of bodies (seed 5) per preset and N.',
            'Errors of the time integration: each row is a single force evaluation (the energy-drift experiment covers that).',
            'Behaviour outside θ = 0.3 … 1.2, or of other opening tests and higher-order (quadrupole) node expansions.',
            'The f32 rounding floor: it is near 1e-6, about 1,000× below the smallest error here (θ → 0 reaches it in CI).',
        ],
    },
};

// θ-error predictions. The monopole of a node about its center of mass has no dipole term, so the leading
// error of one accepted node is the quadrupole term, smaller than the monopole by about (l/d)², and the
// opening test accepts a node only if l/d ≤ θ. So the error grows as θ² to leading order. The next term
// (octupole, θ³) steepens the fit a little at the large-θ end: the CPU replica gives slopes 2.07 … 2.14 for
// the four preset/N choices, with intervals as narrow as ±0.04, so the prediction is 2 with a model
// tolerance of ±0.25 (not a statistical one).
export const THETA_ORDER = 2;
export const THETA_ORDER_TOLERANCE = 0.25;
// Work: a cell of size l is accepted at distances between about l/θ and 2l/θ, so the number of accepted
// cells per tree level is ∝ θ⁻³ in 3D for a uniform density. That is the large-N limit and an upper bound on
// the rate: at these N the near field (opened down to single bodies at every θ) and the floor of a few cells
// per level (≈ log N levels) flatten it. CI (Plummer N = 4,096): slope −1.47. Predicted: falls, with an
// exponent between −3 and 0.
export const THETA_WORK_RANGE = [-3, 0];

export function interpretThetaError(rows, { n, preset } = {}) {
    const err = fitRows(rows, 'theta', 'median');
    const work = fitRows(rows, 'theta', 'interactions');
    const errAgrees = slopeAgrees(err, THETA_ORDER, THETA_ORDER_TOLERANCE);
    const workAgrees = rangeVerdict(work, ...THETA_WORK_RANGE);
    const claims = [{
        claim: 'Median force error vs θ (log-log slope)',
        predicted: `${THETA_ORDER}: error ∝ θ² (the quadrupole term of a node's monopole about its center of mass, l/d ≤ θ); `
            + `accepted ${THETA_ORDER} ± ${THETA_ORDER_TOLERANCE} for the θ³ term at large θ`,
        measured: slopeText(err),
        agrees: errAgrees,
        note: errAgrees === null ? tooFew(err, 'θ values')
            : errAgrees ? 'The measured exponent is consistent with θ².'
                : err.slope < THETA_ORDER ? 'Flatter than θ²: something other than the multipole truncation sets the error '
                    + '(an f32 floor near 1e-6, or a fault in the tree build or opening test).'
                    : 'Steeper than θ²: higher multipoles dominate, or the error at small θ is unusually low; check the p99 column too.',
    }, {
        claim: 'Interactions per body vs θ (log-log slope)',
        predicted: 'falls, exponent between −3 and 0: θ⁻³ cells per tree level in the large-N, uniform-density limit; '
            + 'flatter at these N, where the near field and the ~log N levels add a θ-independent part (CI, Plummer 4,096: −1.47)',
        measured: slopeText(work),
        agrees: workAgrees,
        note: workAgrees === null ? (work.ci95 ? 'The interval straddles a bound of the predicted range.' : tooFew(work, 'θ values'))
            : workAgrees && n ? `At θ = ${rows.at(-1).theta} each body does ${rows.at(-1).interactions.toFixed(0)} interactions, `
                + `${pct(rows.at(-1).interactions / (n - 1))} of brute force's ${count(n - 1)}.` : null,
    }];
    const medians = REPLICA_MEDIANS[`${preset}/${n}`];
    const ratios = medians ? rows.map(r => r.median / medians[THETAS.indexOf(r.theta)]).filter(v => Number.isFinite(v) && v > 0) : [];
    const within = ratios.every(v => v >= 1 / REPLICA_FACTOR && v <= REPLICA_FACTOR);
    claims.push({
        claim: 'Your GPU vs the float64 CPU replica (same bodies)',
        predicted: `GPU median / replica median = 1 at every θ, within × / ÷ ${REPLICA_FACTOR} (f32 arithmetic, tree-build order)`,
        measured: ratios.length ? `ratio ${f2(Math.min(...ratios))} … ${f2(Math.max(...ratios))} over ${ratios.length} θ value${ratios.length === 1 ? '' : 's'}`
            : medians ? 'no rows yet' : `no replica values for ${preset}, N = ${n}`,
        agrees: ratios.length ? within : null,
        note: !ratios.length ? null : within ? 'Your GPU runs the same approximation as the reference; the trends above are the algorithm\'s, not the device\'s.'
            : 'The shader on this device does not reproduce the reference; treat the slopes above with suspicion and see the verification list.',
    });
    return claims;
}

// ---- 2. Scaling: Barnes-Hut vs brute force ----------------------------------------------------------
export const SCALING_COUNTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536];

const scaling = {
    id: 'scaling',
    title: 'Scaling: Barnes-Hut vs brute force',
    question: 'At what N does the O(N log N) tree, rebuilt every step, beat the O(N²) tiled direct sum on your GPU?',
    settings: [presetSetting],
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'N' },
        { key: 'bhGpu', label: 'Barnes-Hut GPU ms', format: ms, unit: 'ms' },
        { key: 'bruteGpu', label: 'brute force GPU ms', format: ms, unit: 'ms' },
        { key: 'bhWall', label: 'Barnes-Hut wall ms', format: ms, unit: 'ms' },
        { key: 'bruteWall', label: 'brute force wall ms', format: ms, unit: 'ms' },
        { key: 'treeShare', label: 'tree build share', format: (v) => `${(100 * v).toFixed(0)}%` },
        { key: 'interactions', label: 'BH interactions / body', format: (v) => v.toFixed(0) },
        { key: 'compileMs', label: 'BH compile ms', format: ms, unit: 'ms' },
        { key: 'uploadMs', label: 'BH upload ms', format: ms, unit: 'ms' },
        { key: 'readbackMs', label: 'BH readback ms', format: ms, unit: 'ms' },
    ],
    charts: [
        { x: 'n', xLabel: 'bodies N', logX: true, y: ['bhGpu', 'bruteGpu'], yLabel: 'GPU ms per force evaluation', logY: true,
            labels: { bhGpu: 'Barnes-Hut', bruteGpu: 'brute force' },
            title: 'GPU ms per force evaluation vs N (needs timestamp-query)' },
        { x: 'n', xLabel: 'bodies N', logX: true, y: ['bhWall', 'bruteWall'], yLabel: 'wall ms per force evaluation', logY: true,
            labels: { bhWall: 'Barnes-Hut', bruteWall: 'brute force' } },
        { x: 'n', xLabel: 'bodies N', logX: true, y: ['compileMs', 'uploadMs', 'bhWall', 'readbackMs'], yLabel: 'wall ms per phase', logY: true,
            labels: { compileMs: 'compile', uploadMs: 'upload', bhWall: 'dispatch', readbackMs: 'readback' },
            title: 'Barnes-Hut wall-clock phases vs N' },
    ],
    baseline: 'Brute force does exactly N − 1 interactions per body; Barnes-Hut at θ = 0.5 does a few hundred to about a '
        + 'thousand (248 at N = 1,024, 1,140 at 65,536 in CI) plus a 30-dispatch tree build, so it loses at small N and wins '
        + 'at large N. In CI, on a CPU-emulated GPU, the wall times cross between N = 8,192 (brute force about 15% faster) '
        + `and 16,384 (Barnes-Hut about 25% faster). Brute force stops at N = ${BRUTE_MAX_N.toLocaleString('en-US')} here `
        + '(the example allows 32,768). Where the crossover falls depends on the GPU, driver and browser: read it from '
        + 'your medians and their spread, not from one number. Compile is pipeline and buffer creation after the warm-up '
        + '(usually a pipeline-cache hit); the cold compile is in the discarded warm-up.',
    // Budget: 7 Barnes-Hut and 5 brute-force points × (1 warm-up + 5 repeats) = 72 evaluations, each with its own
    // set-up and read-back. On a discrete or recent integrated GPU that is a few seconds; on the CPU-emulated GPU in
    // CI it is 26 s. Every point keeps the full protocol (so the record's warm-up/repeat counts hold for every row);
    // instead, a brute-force point whose projected cost would leave too little of the budget for the remaining
    // Barnes-Hut points is skipped (see skipBrute), and interpret() says so.
    async *run(ctx, { preset }) {
        const perCall = { bh: null, brute: null };   // ms per evaluation at the previous N (set-up to read-back)
        let prevN = null, bruteSkipped = false;
        const timed = async (mode, run) => {
            const t0 = performance.now();
            const m = await ctx.measure(run, PROTOCOL);
            perCall[mode] = (performance.now() - t0) / (PROTOCOL.warmup + PROTOCOL.repeats);
            return m;
        };
        for (const n of SCALING_COUNTS) {
            if (n > BH_MAX_N) break;
            const { positions } = generateBodies(preset, n, SOFTENING, SEED);
            await ctx.step();
            const bh = await timed('bh', forceRun(ctx, positions, { mode: 'bh', theta: 0.5, accel: false }));
            const last = bh.samples[bh.samples.length - 1];
            const share = summarize(bh.samples.map(s => (s.gpuMs ? s.passes.tree / s.gpuMs : null)));
            const row = { n, ...withSpread('bhGpu', bh.gpuMs), ...withSpread('bhWall', bh.wallMs),
                ...withSpread('bruteGpu', null), ...withSpread('bruteWall', null),
                treeShare: share.median, interactions: last.interactionsPerBody,
                ...withSpread('compileMs', bh.phases.compile), ...withSpread('uploadMs', bh.phases.upload),
                ...withSpread('readbackMs', bh.phases.readback) };
            if (n <= BRUTE_MAX_N && !bruteSkipped) {
                bruteSkipped = prevN !== null && skipBrute(n, prevN, perCall, ctx.remainingMs?.() ?? Infinity);
                if (!bruteSkipped) {
                    await ctx.step();
                    const brute = await timed('brute', forceRun(ctx, positions, { mode: 'brute', accel: false }));
                    Object.assign(row, withSpread('bruteGpu', brute.gpuMs), withSpread('bruteWall', brute.wallMs));
                }
            }
            prevN = n;
            yield row;
        }
    },
    // No verify: timings have no reference value, and correctness is the theta-error experiment's job
    interpret: interpretScaling,
    scope: {
        shows: [
            'How the time of one brute-force and one Barnes-Hut (θ = 0.5) force evaluation grows with N on your device, '
                + 'as medians with their interquartile range over 5 repeats.',
            'Where on your device Barnes-Hut starts to win, bracketed between two sweep points.',
            'How much of the Barnes-Hut GPU time is the tree build (needs timestamp-query).',
        ],
        doesNotShow: [
            'A ranking of GPUs or browsers: one device, one browser, one run. Compare medians only where the spreads do not overlap.',
            'Asymptotic complexity: N goes to 65,536 (brute force to 16,384), where fixed per-dispatch and set-up costs '
                + 'still flatten the small-N times, and interactions per body have not reached their log N regime.',
            'Accuracy: every point is θ = 0.5 (the θ-error experiment measures what that costs).',
            'Simulation frame rate: each evaluation here sets up, uploads, dispatches once and reads back; Explore pipelines its steps.',
        ],
    },
};

// Whether to skip brute force at N (and above): its projected cost, from the previous N's per-evaluation time
// scaled by (N / prevN)², plus the remaining Barnes-Hut points (from this N's Barnes-Hut time, measured just
// before, scaled by (M / N)^1.2), would not fit in the remaining budget. Exported for tests.
export function skipBrute(n, prevN, perCall, remainingMs) {
    if (!(perCall.brute > 0) || !Number.isFinite(remainingMs)) return false;
    const calls = PROTOCOL.warmup + PROTOCOL.repeats;
    const brute = calls * perCall.brute * (n / prevN) ** 2;
    const bh = SCALING_COUNTS.filter(m => m > n && m <= BH_MAX_N)
        .reduce((s, m) => s + calls * (perCall.bh ?? 0) * (m / n) ** 1.2, 0);
    return brute + bh > remainingMs;
}

// Scaling predictions.
//   Brute force: exactly N − 1 interactions for each of N bodies, so time ∝ N² once the arithmetic dominates the
//   fixed cost of a submission. Prediction 2, with ±0.1 tolerance for the remaining fixed cost at N = 1,024.
//   Barnes-Hut: O(N log N). Over N = 1,024 … 65,536, N log₂ N has an effective power-law slope of 1.11. At these N
//   the interactions per body grow faster than log N (CI: ∝ N^0.37), which would push the traversal towards 1.4,
//   while the O(N) tree build and fixed per-dispatch costs pull the total down; so the prediction is a band,
//   1.0 … 1.3 (slopeAgrees with 1.15 ± 0.15), not a number. CI (software GPU, wall ms): 1.10 (1.00 … 1.20).
//   Crossover: O(N²) must overtake O(N log N) at some N; where depends on the device. CI: 8,192 … 16,384.
//   Tree-build share: the build is O(N) plus ~30 fixed dispatches, the traversal O(N log N), so the share falls.
export const BRUTE_ORDER = 2, BRUTE_ORDER_TOLERANCE = 0.1;
export const BH_ORDER = 1.15, BH_ORDER_TOLERANCE = 0.15;
export const NLOGN_EFFECTIVE = fitPowerLaw(SCALING_COUNTS, SCALING_COUNTS.map(n => n * Math.log2(n))).slope;   // 1.11

// The times to compare for one method: GPU ms when every measured point has them (timestamp-query), else wall ms
function timeBasis(rows, mode) {
    const measured = rows.filter(r => typeof r[`${mode}Wall`] === 'number');
    const gpu = measured.length > 0 && measured.every(r => r[`${mode}Gpu`] > 0);
    return { key: `${mode}${gpu ? 'Gpu' : 'Wall'}`, label: gpu ? 'GPU ms' : 'wall ms', rows: measured };
}

// The crossover from the medians: the last N where brute force is faster and the first N after which Barnes-Hut
// stays faster, a log-log interpolation between them, and whether the spreads overlap at the two points
export function crossover(rows) {
    const both = rows.filter(r => typeof r.bruteWall === 'number' && typeof r.bhWall === 'number');
    const gpu = both.length > 0 && both.every(r => r.bhGpu > 0 && r.bruteGpu > 0);
    const k = gpu ? 'Gpu' : 'Wall';
    const pts = both.map(r => ({ n: r.n, bh: r[`bh${k}`], brute: r[`brute${k}`],
        bhLo: r[`bh${k}Lo`], bhHi: r[`bh${k}Hi`], bruteLo: r[`brute${k}Lo`], bruteHi: r[`brute${k}Hi`] }));
    const basis = gpu ? 'GPU ms' : 'wall ms';
    if (!pts.length) return { basis, points: 0 };
    let i = pts.length;                       // first index from which Barnes-Hut is faster at every later point
    while (i > 0 && pts[i - 1].bh < pts[i - 1].brute) i--;
    if (i === pts.length) return { basis, points: pts.length, state: 'none', lastN: pts.at(-1).n };
    if (i === 0) return { basis, points: pts.length, state: 'below', firstN: pts[0].n };
    const a = pts[i - 1], b = pts[i];
    const la = Math.log(a.bh / a.brute), lb = Math.log(b.bh / b.brute);
    const estimate = Math.exp(Math.log(a.n) + (Math.log(b.n) - Math.log(a.n)) * la / (la - lb));
    const separated = (p) => [p.bhLo, p.bhHi, p.bruteLo, p.bruteHi].every(Number.isFinite)
        && (p.bhLo > p.bruteHi || p.bhHi < p.bruteLo);
    const monotone = pts.slice(0, i).every(p => p.bh >= p.brute);
    return { basis, points: pts.length, state: 'bracketed', lo: a.n, hi: b.n, estimate,
        clear: separated(a) && separated(b), monotone };
}

export function interpretScaling(rows) {
    const brute = timeBasis(rows, 'brute'), bh = timeBasis(rows, 'bh');
    const bruteFit = fitRows(brute.rows, 'n', brute.key), bhFit = fitRows(bh.rows, 'n', bh.key);
    const bruteAgrees = slopeAgrees(bruteFit, BRUTE_ORDER, BRUTE_ORDER_TOLERANCE);
    const bhAgrees = slopeAgrees(bhFit, BH_ORDER, BH_ORDER_TOLERANCE);
    const skipped = rows.filter(r => r.n <= BRUTE_MAX_N && typeof r.bruteWall !== 'number').map(r => count(r.n));
    const claims = [{
        claim: `Brute-force time vs N (log-log slope, ${brute.label})`,
        predicted: `${BRUTE_ORDER}: N − 1 interactions for each of N bodies, O(N²) (± ${BRUTE_ORDER_TOLERANCE} for fixed costs)`,
        measured: slopeText(bruteFit),
        agrees: bruteAgrees,
        note: [bruteAgrees === null ? tooFew(bruteFit, 'N values') : bruteAgrees ? null
            : bruteFit.slope < BRUTE_ORDER ? 'Below 2: fixed per-evaluation costs (submission, synchronisation, read-back) are still '
                + 'a large part of the time at these N on this device. GPU ms (with timestamp-query) remove most of them.'
                : 'Above 2: the large-N points are slower than N² predicts (clocks, caches or occupancy changing during the run); '
                    + 'check whether the spreads overlap.',
        skipped.length ? `Brute force was skipped at N = ${skipped.join(', ')} to leave budget for the Barnes-Hut points.` : null]
            .filter(Boolean).join(' ') || null,
    }, {
        claim: `Barnes-Hut time vs N (log-log slope, ${bh.label})`,
        predicted: `about N log N: ${f2(NLOGN_EFFECTIVE)} over N = 1,024 … 65,536; accepted ${f2(BH_ORDER - BH_ORDER_TOLERANCE)} … `
            + `${f2(BH_ORDER + BH_ORDER_TOLERANCE)}, because interactions per body still grow faster than log N here`,
        measured: slopeText(bhFit),
        agrees: bhAgrees,
        note: bhAgrees === null ? tooFew(bhFit, 'N values') : bhAgrees ? null
            : bhFit.slope < BH_ORDER ? 'Below 1: fixed costs (about 30 tree-build dispatches per evaluation) dominate over this '
                + 'whole range on this device, so the time barely grows with N.'
                : 'Above 1.3: the traversal grows faster than N log N here (scattered memory access at large N, or the growth of '
                    + 'interactions per body below).',
    }];

    const work = fitRows(rows, 'n', 'interactions');
    const workAgrees = rangeVerdict(work, 0, 1);
    claims.push({
        claim: 'Barnes-Hut interactions per body vs N (log-log slope)',
        predicted: `between 0 and 1: grows, but slower than brute force's N − 1 (→ log N for large N, ${f2(NLOGN_EFFECTIVE - 1)} here; CI: 0.37)`,
        measured: slopeText(work),
        agrees: workAgrees,
        note: workAgrees === null ? (work.ci95 ? 'The interval straddles a bound of the predicted range.' : tooFew(work, 'N values'))
            : 'Deterministic counts (one distribution, one seed): the interval reflects curvature, not noise.',
    });

    const x = crossover(rows);
    const at = (n) => `N = ${count(n)}`;
    let measured, agrees, note = `Compared on ${x.basis}. Device-dependent: compare it with other devices, not with a fixed number.`;
    if (!x.points) { measured = 'no N with both methods'; agrees = null; }
    else if (x.state === 'bracketed') {
        measured = `between ${at(x.lo)} and ${at(x.hi)} (log-interpolated ≈ ${count(Math.round(x.estimate / 100) * 100)})`;
        agrees = true;
        note += x.clear ? ' The interquartile ranges do not overlap at either point, so the bracket is resolved.'
            : ' The interquartile ranges overlap at one of the two points: the bracket could move by one sweep step on a re-run.';
        if (!x.monotone) note += ' The ratio is not monotone below the bracket (noisy small-N times).';
    } else if (x.state === 'below') {
        measured = `below ${at(x.firstN)}: Barnes-Hut is faster at every N measured`;
        agrees = true;
    } else {
        // Brute force faster everywhere it ran: extrapolate the two fits (only meaningful if the slopes differ)
        const sameBasis = brute.label === bh.label && bruteFit.slope !== null && bhFit.slope !== null;
        const extrapolated = sameBasis && bruteFit.slope > bhFit.slope
            ? 10 ** ((bhFit.intercept - bruteFit.intercept) / (bruteFit.slope - bhFit.slope)) : null;
        measured = `above ${at(x.lastN)}: brute force is faster at every N where both ran`
            + (extrapolated ? `; extrapolating the two fits gives N ≈ ${count(Math.round(extrapolated / 1000) * 1000)}` : '');
        agrees = sameBasis && !(bruteFit.slope > bhFit.slope) ? false : null;
        note += agrees === false ? ' The fitted slopes do not cross at all, which contradicts O(N log N) vs O(N²): look at the spreads.'
            : ' The prediction is not tested inside the sweep; the extrapolation assumes both slopes hold beyond it.';
    }
    claims.push({
        claim: 'Crossover N (Barnes-Hut faster from here)',
        predicted: 'exists: O(N²) overtakes O(N log N) at some N (CI, software GPU, wall ms: between 8,192 and 16,384)',
        measured, agrees, note,
    });

    const shares = rows.filter(r => r.treeShare > 0);
    const shareFit = fitRows(shares, 'n', 'treeShare');
    const shareAgrees = rangeVerdict(shareFit, -Infinity, 0);
    claims.push({
        claim: 'Tree-build share of Barnes-Hut GPU time vs N',
        predicted: 'falls with N: the build is O(N) plus about 30 fixed dispatches, the traversal O(N log N)',
        measured: shares.length >= 2 ? `${pct(shares[0].treeShare)} at ${at(shares[0].n)} → ${pct(shares.at(-1).treeShare)} at `
            + `${at(shares.at(-1).n)}; slope ${slopeText(shareFit)}` : 'needs GPU times (timestamp-query)',
        agrees: shareAgrees,
        note: !shares.length ? 'This browser or GPU does not offer timestamp-query, so the split cannot be measured.'
            : shareAgrees === null ? (shareFit.ci95 ? 'The share does not change clearly with N.' : tooFew(shareFit, 'N values'))
                : shareAgrees ? null : 'The share grows with N: the tree build scales worse than the traversal on this device.',
    });
    return claims;
}

// ---- 3. Time step vs energy drift -------------------------------------------------------------------
export const DTS = [0.04, 0.02, 0.01, 0.005];
export const DRIFT_TIME = 2;        // simulated time per run: steps = DRIFT_TIME / dt ≤ 400
export const DRIFT_COUNTS = [2048, 4096];
export const DRIFT_SAMPLES = 25;    // energy readbacks per run (16 bytes each)
// Measured in CI (brute force, Plummer N = 2,048, seed 5): max |ΔE/E₀| over t = 2 at dt = 0.01
export const CI_DRIFT_AT_001 = 1.3e-6;
// Bounds on max |ΔE/E₀|: brute force — a float64 leapfrog on the same bodies gives at most 7.4e-5 at
// dt = 0.04 (spiral, N = 4,096; Plummer 2,048: 3.4e-5, as the GPU in CI); Barnes-Hut θ = 0.5 — the bound of
// test/gpu/nbody-measure.gpu.test.mjs (CI measured a 6.4e-4 … 6.9e-4 floor)
export const DRIFT_BOUNDS = { brute: 1e-3, bh: 1e-2 };

const energyDrift = {
    id: 'energy-drift',
    title: 'Time step vs energy error',
    question: 'How does the leapfrog energy error scale with the time step dt, and what does the Barnes-Hut approximation add to it?',
    settings: [
        { key: 'n', label: 'Bodies', options: DRIFT_COUNTS.map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 2048 },
        { key: 'mode', label: 'Forces', options: [{ value: 'brute', label: 'Brute force (exact)' }, { value: 'bh', label: 'Barnes-Hut θ = 0.5' }], default: 'brute' },
        presetSetting,
    ],
    budgetMs: 20000,
    columns: [
        { key: 'dt', label: 'dt', format: String },
        { key: 'steps', label: 'steps' },
        { key: 'maxDrift', label: 'max |ΔE/E₀|', format: sci },
        { key: 'finalDrift', label: 'final ΔE/E₀', format: sci },
        { key: 'wallMs', label: 'wall ms', format: (v) => v.toFixed(0), unit: 'ms' },
    ],
    charts: [{
        x: 'dt', xLabel: 'time step dt', logX: true, y: ['maxDrift'], yLabel: `max |ΔE/E₀| over t = ${DRIFT_TIME}`, logY: true,
        reference: {
            label: 'dt² (CI)',
            points: [DTS[DTS.length - 1], DTS[0]].map(dt => [dt, CI_DRIFT_AT_001 * (dt / 0.01) ** 2]),
        },
    }],
    baseline: `CI (software GPU, brute force, Plummer N = 2,048, t = ${DRIFT_TIME}): max |ΔE/E₀| 3.4e-5 / 5.5e-6 / 1.3e-6 / `
        + '3.0e-7 at dt = 0.04 / 0.02 / 0.01 / 0.005, about 4× per halving (dt²). With Barnes-Hut at θ = 0.5 it stays at '
        + '6.4e-4 to 6.9e-4 for every dt: the force and potential error of the tree dominates. The f32 energy sum itself is good to about 1e-7. '
        + 'Each dt is one run (a single wall time, not a median over repeats).',
    async *run(ctx, { n, mode, preset }) {
        const N = Math.min(n, mode === 'brute' ? BRUTE_MAX_N : BH_MAX_N);
        const { positions, velocities } = generateBodies(preset, N, SOFTENING, SEED);
        for (const dt of DTS) {
            await ctx.step();
            const steps = Math.round(DRIFT_TIME / dt);
            const r = await measureEnergyDrift(ctx.gpuDevice, { positions, velocities, mode, theta: 0.5,
                softening: SOFTENING, dt, steps, samples: DRIFT_SAMPLES, between: () => ctx.step() });
            yield { dt, steps, maxDrift: r.maxDrift, finalDrift: r.finalDrift, wallMs: r.wallMs };
        }
    },
    // Every dt within the bound; brute force only: the error falls as dt halves, at about second order
    verify(rows, { mode }) {
        const bound = DRIFT_BOUNDS[mode] ?? DRIFT_BOUNDS.bh;
        const reference = mode === 'brute'
            ? 'float64 leapfrog on the same bodies: ≤ 7.4e-5 at dt = 0.04'
            : 'CI bound for Barnes-Hut θ = 0.5 (measured floor 6.4e-4 … 6.9e-4)';
        const checks = DTS.map(dt => check(`max |ΔE/E₀| at dt = ${dt}`, rows.find(r => r.dt === dt)?.maxDrift,
            { min: 0, max: bound, tolerance: `<= ${sci1(bound)}`, reference }));
        if (mode === 'brute') {
            checks.push(rows.length >= 2
                ? check('dt halvings where the error does not fall', monotoneViolations(rows, 'maxDrift', -1),
                    { max: 0, tolerance: '0', reference: 'leapfrog error ∝ dt² for exact forces' })
                : { metric: 'dt halvings where the error does not fall', value: null, tolerance: '0', reference: 'needs 2 rows', status: 'not-checked' });
            const a = rows.find(r => r.dt === 0.02)?.maxDrift, b = rows.find(r => r.dt === 0.01)?.maxDrift;
            checks.push(check('observed order between dt = 0.02 and 0.01', a > 0 && b > 0 ? Math.log2(a / b) : null,
                { min: 1, max: 3, tolerance: '1 … 3', reference: 'second-order leapfrog: 2 (CI: 2.1)' }));
        }
        return checks;
    },
    interpret: interpretEnergyDrift,
    scope: {
        shows: [
            `How the largest energy error over t = ${DRIFT_TIME} of the shipped leapfrog scales with dt when the forces are exact (brute force).`,
            'That with Barnes-Hut forces the error stops depending on dt: the tree\'s force error sets a floor.',
        ],
        doesNotShow: [
            `Long-term behaviour: t = ${DRIFT_TIME} is a few crossing times; it says nothing about secular drift over thousands of steps.`,
            'Statistics over runs: one distribution, one seed, one run per dt (no repeats; the dynamics are deterministic, '
                + 'but a different seed would give somewhat different values).',
            'Errors below about 1e-7: the f32 energy sum sets a floor there, so the brute-force value at dt = 0.005 '
                + '(CI: 3e-7) is already partly rounding.',
            'Accuracy of the trajectories: energy can be conserved while individual orbits diverge (N-body dynamics is chaotic).',
        ],
    },
};

// Energy predictions.
//   Brute force: leapfrog is second order and symplectic, so with exact forces the energy error oscillates with an
//   amplitude ∝ dt² (the shadow energy differs from E by O(dt²)): slope 2 of max |ΔE/E₀| vs dt, no extra tolerance.
//   CI (Plummer 2,048): 2.25 (1.87 … 2.64); dt = 0.04 is a little outside the asymptotic range.
//   Barnes-Hut θ = 0.5: the tree force is not the gradient of one fixed potential (cells open and close as bodies
//   move, the tree is rebuilt every step), so the error is set by the force/potential error, not by dt: slope 0,
//   accepted ±0.2 because a residual dt² part (≈ 5% of the floor at dt = 0.04 in CI) tilts it slightly. CI: 0.04.
//   Floor: of the order of the tree's relative potential error at θ = 0.5 (≈ 4e-3 per body in CI), reduced by
//   averaging over bodies, and far above the exact-force leapfrog error (≤ 7.4e-5 at dt = 0.04 in float64).
//   Predicted range 1e-4 … 1e-2 (an order-of-magnitude scale, not a fit). CI: 6.4e-4 … 6.9e-4.
export const DRIFT_ORDER = 2;
export const FLOOR_SLOPE_TOLERANCE = 0.2;
export const FLOOR_RANGE = [1e-4, 1e-2];

export function interpretEnergyDrift(rows, { mode } = {}) {
    const fit = fitRows(rows, 'dt', 'maxDrift');
    if (mode !== 'bh') {
        const agrees = slopeAgrees(fit, DRIFT_ORDER, 0);
        return [{
            claim: 'Max |ΔE/E₀| vs dt, exact forces (log-log slope = order)',
            predicted: `${DRIFT_ORDER}: leapfrog is second order and symplectic, so the energy error stays in a band ∝ dt²`,
            measured: slopeText(fit),
            agrees,
            note: agrees === null ? tooFew(fit, 'time steps')
                : agrees ? 'The measured order is consistent with second-order leapfrog.'
                    : fit.slope < DRIFT_ORDER ? 'Below 2: a floor flattens the small-dt end (the f32 energy sum near 1e-7), '
                        + 'or the integration is not the kick-drift-kick it should be.'
                        : 'Above 2: the large-dt end is outside the asymptotic range (close encounters at dt = 0.04); '
                            + 'the order between the smaller steps is the better estimate.',
        }];
    }
    const flat = slopeAgrees(fit, 0, FLOOR_SLOPE_TOLERANCE);
    const drifts = rows.map(r => r.maxDrift).filter(v => Number.isFinite(v) && v > 0);
    const floor = drifts.length ? summarize(drifts).median : null;
    const inRange = floor === null ? null : floor >= FLOOR_RANGE[0] && floor <= FLOOR_RANGE[1];
    return [{
        claim: 'Max |ΔE/E₀| vs dt, Barnes-Hut θ = 0.5 (log-log slope)',
        predicted: `0: dt-independent, set by the tree's force error (accepted ± ${FLOOR_SLOPE_TOLERANCE} for a residual dt² part)`,
        measured: slopeText(fit),
        agrees: flat,
        note: flat === null ? tooFew(fit, 'time steps')
            : flat ? 'A smaller dt does not help: the tree, not the integrator, limits the energy error.'
                : fit.slope > 0 ? 'The error still falls with dt: on this run the leapfrog error is not negligible against the tree error.'
                    : 'The error grows as dt shrinks: more steps accumulate more tree error, so the floor is not a fixed level here.',
    }, {
        claim: 'Size of the Barnes-Hut floor',
        predicted: `${sci1(FLOOR_RANGE[0])} … ${sci1(FLOOR_RANGE[1])}: of the order of the tree's relative potential error at θ = 0.5 `
            + '(≈ 4e-3 per body), partly averaged out; far above the exact-force error (≤ 7.4e-5)',
        measured: floor === null ? 'no rows yet' : `${sci(floor)} (median over ${drifts.length} dt; ${sci(Math.min(...drifts))} … ${sci(Math.max(...drifts))})`,
        agrees: inRange,
        note: inRange === null ? null : 'An order-of-magnitude prediction, not a fit; one distribution, one seed (CI, Plummer 2,048: 6.4e-4 … 6.9e-4).',
    }];
}

export default [thetaError, scaling, energyDrift];
