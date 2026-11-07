// Measure experiments for N-body (js/examples/nbody.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureForces, measureEnergyDrift), which
// runs the shipped pipelines without a canvas; the CPU reference is the float64 direct sum of
// js/refs/nbody.js. Every sweep is a fixed list with hard caps (exported so test/nbody-measure.test.mjs
// can check them): Barnes-Hut N ≤ 65,536, brute force N ≤ 16,384 (the scaling sweep's Largest N option: 262,144
// and 65,536), ≤ 400 leapfrog steps per run.
//
// Timing follows the shared protocol (PROTOCOL: 1 discarded warm-up, then 5 independent repeats, each a
// fresh measureForces call that sets up, uploads, dispatches once and reads back); timed columns are
// medians with their interquartile range. Errors and interaction counts are deterministic and come from
// the last repeat. Each experiment's verify() checks its rows against CPU references and CI bounds.

import { generateBodies, measureForces, measureEnergyDrift } from '../../examples/nbody.js';
import { directForcesAt, relErrors } from '../../refs/nbody.js';
import { summarize, withSpread, fitPowerLaw, slopeAgrees } from '../stats.js';
import { check } from '../result-record.js';
import { doublingFactor, timesText, countText } from '../measure-levels.js';

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
        + `interactions per body (brute force: 4,095). Errors are checked on ${SAMPLE} bodies against a float64 direct sum. `
        + 'Fitted slopes (95% CI): error vs θ 2.09 (2.01 … 2.16), interactions vs θ −1.46 (−1.50 … −1.42).',
    plannedSteps: () => THETAS.length,
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
        key: 'theta-error-order',
        claim: 'Median force error vs θ (log-log slope)',
        predicted: `${THETA_ORDER}: error ∝ θ² (the quadrupole term of a node's monopole about its center of mass, l/d ≤ θ); `
            + `accepted ${THETA_ORDER} ± ${THETA_ORDER_TOLERANCE} for the θ³ term at large θ`,
        measured: slopeText(err),
        agrees: errAgrees,
        plain: {
            claim: 'A bigger θ: how far off the pulls are',
            predicted: 'Doubling θ makes the error about 4 times bigger',
            measured: err.slope === null ? 'Not enough θ values yet' : `Doubling θ made the error ${timesText(doublingFactor(err.slope))} times bigger`,
        },
        note: errAgrees === null ? tooFew(err, 'θ values')
            : errAgrees ? 'The measured exponent is consistent with θ².'
                : err.slope < THETA_ORDER ? 'Flatter than θ²: something other than the multipole truncation sets the error '
                    + '(an f32 floor near 1e-6, or a fault in the tree build or opening test).'
                    : 'Steeper than θ²: higher multipoles dominate, or the error at small θ is unusually low; check the p99 column too.',
    }, {
        key: 'theta-work',
        claim: 'Interactions per body vs θ (log-log slope)',
        predicted: 'falls, exponent between −3 and 0: θ⁻³ cells per tree level in the large-N, uniform-density limit; '
            + 'flatter at these N, where the near field and the ~log N levels add a θ-independent part (CI, Plummer 4,096: −1.47)',
        measured: slopeText(work),
        agrees: workAgrees,
        plain: plainWork(rows, n),
        note: workAgrees === null ? (work.ci95 ? 'The interval straddles a bound of the predicted range.' : tooFew(work, 'θ values'))
            : workAgrees && n ? `At θ = ${rows.at(-1).theta} each body does ${rows.at(-1).interactions.toFixed(0)} interactions, `
                + `${pct(rows.at(-1).interactions / (n - 1))} of brute force's ${count(n - 1)}.` : null,
    }];
    const medians = REPLICA_MEDIANS[`${preset}/${n}`];
    const ratios = medians ? rows.map(r => r.median / medians[THETAS.indexOf(r.theta)]).filter(v => Number.isFinite(v) && v > 0) : [];
    const within = ratios.every(v => v >= 1 / REPLICA_FACTOR && v <= REPLICA_FACTOR);
    claims.push({
        key: 'replica-match',
        claim: 'Your GPU vs the float64 CPU replica (same bodies)',
        predicted: `GPU median / replica median = 1 at every θ, within × / ÷ ${REPLICA_FACTOR} (f32 arithmetic, tree-build order)`,
        measured: ratios.length ? `ratio ${f2(Math.min(...ratios))} … ${f2(Math.max(...ratios))} over ${ratios.length} θ value${ratios.length === 1 ? '' : 's'}`
            : medians ? 'no rows yet' : `no replica values for ${preset}, N = ${n}`,
        agrees: ratios.length ? within : null,
        plain: ratios.length ? {
            claim: 'Your GPU vs a careful copy of its program on the CPU',
            predicted: 'The same errors at every θ',
            measured: within ? `The same, to within ${Math.max(1, Math.round(100 * Math.max(...ratios.map(v => Math.abs(v - 1)))))}%`
                : `Different: off by up to ${timesText(Math.max(...ratios.map(v => Math.max(v, 1 / v))))} times`,
        } : undefined,
        note: !ratios.length ? null : within ? 'Your GPU runs the same approximation as the reference; the trends above are the algorithm\'s, not the device\'s.'
            : 'The shader on this device does not reproduce the reference; treat the slopes above with suspicion and see the verification list.',
    });
    return claims;
}

// Plain words for the work claim: the pulls per star at the smallest and largest θ of the rows
function plainWork(rows, n) {
    const r = rows.filter(x => x.interactions > 0);
    if (r.length < 2) return undefined;
    const all = n ? ` (with no shortcut: ${countText(n - 1)})` : '';
    return {
        claim: 'A bigger θ: how much work each star does',
        predicted: 'Less work: more far-away boxes count as one big star',
        measured: `${countText(r[0].interactions)} pulls per star at θ = ${r[0].theta}, ${countText(r.at(-1).interactions)} at θ = ${r.at(-1).theta}${all}`,
    };
}

// ---- 2. Scaling: Barnes-Hut vs brute force ----------------------------------------------------------
export const SCALING_COUNTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
// The maxN setting: the largest Barnes-Hut and brute-force N of the sweep. The default keeps the original sweep
// (it is what CI's software GPU runs); 262,144 gives a fast GPU enough sizes above its fixed cost to fit a slope.
// Caps: Barnes-Hut ≤ MAX_BH = 262,144; brute force ≤ MEASURE_MAX_BRUTE = 65,536, one dispatch of 4.3·10⁹
// interactions (≈ 50 ms on an Apple M4 Pro, under a second on a slow integrated GPU, ≈ 5 s on CI's lavapipe).
export const SCALING_LIMITS = { 65536: { bh: 65536, brute: 16384 }, 262144: { bh: 262144, brute: 65536 } };
export const SCALING_DEFAULT_MAX = 65536;
export const scalingLimits = (maxN) => SCALING_LIMITS[Number(maxN)] ?? SCALING_LIMITS[SCALING_DEFAULT_MAX];
export const scalingCounts = (maxN) => SCALING_COUNTS.filter(n => n <= scalingLimits(maxN).bh);

// Scaling predictions.
//   The fixed floor. Every evaluation pays a cost that does not grow with N: the submission and synchronisation,
//   and for Barnes-Hut 31 dispatches (30 for the tree build, 1 traversal) of which most do little work at small N;
//   a small brute-force dispatch also cannot fill a large GPU. On a fast GPU that floor is a large part of the time
//   over much of the sweep (Apple M4 Pro, GPU ms: Barnes-Hut 0.6 ms at N = 1,024 and only 3–5× that at 16,384),
//   so a log-log fit over every size measures the floor, not the algorithm (slopes 0.7 for Barnes-Hut, 1.2–1.4 for
//   brute force there). The fits therefore (1) estimate the floor as the median at the smallest N (an upper bound on
//   the fixed cost, since it includes that N's work), (2) keep only sizes whose median is at least FLOOR_FACTOR × the
//   floor and at least FIT_MIN_MS, and (3) fit t − floor. At FLOOR_FACTOR = 4 the floor is at most a quarter of each
//   fitted time, so how well it is estimated barely moves the slope; FIT_MIN_MS = 1 keeps the timer's resolution
//   (65 µs GPU ticks in Chrome, 0.1–1 ms wall clocks) small. The floor is an upper bound: where the smallest-N time
//   is really work (a slow GPU), subtracting it steepens the fit a little (≤ +0.1 for N² and N^1.1 power laws with
//   no floor at all, see test/nbody-interpret.test.mjs). With fewer than 3 such sizes, or a 95% interval wider than
//   ± MAX_CI_HALF (which cannot tell N log N from N²), the verdict is undetermined, and the note gives the slope over
//   all sizes for contrast.
//   Brute force: exactly N − 1 interactions for each of N bodies, so time ∝ N²: prediction 2, ± 0.1 for what is left
//   of the fixed cost and for a GPU that is not yet full at the smallest fitted N.
//   Barnes-Hut: O(N log N). N log₂ N has an effective power-law slope over the fitted sizes (nlognSlope: 1.11 over
//   1,024 … 65,536, 1.09 over 32,768 … 262,144). The O(N) tree build pulls the slope towards 1; the interactions per
//   body, which grow faster than log N at these N (CI: ∝ N^0.37), push the traversal up. So the prediction is the
//   band 1.0 … nlognSlope + BH_EXTRA (≈ 1.3), tested as an overlap of the 95% interval with it (slopeAgrees about
//   its center with half its width). CI (software GPU, wall ms): 1.23 (1.16 … 1.30).
//   Crossover: O(N²) must overtake O(N log N) at some N; where depends on the device. CI: 8,192 … 16,384.
//   Tree-build share: the build is O(N) plus ~30 fixed dispatches, the traversal O(N log N), so the share falls.
export const BRUTE_ORDER = 2, BRUTE_ORDER_TOLERANCE = 0.1;
export const BH_LOWER = 1, BH_EXTRA = 0.2;
export const FLOOR_FACTOR = 4;
export const FIT_MIN_MS = 1;
export const MAX_CI_HALF = 0.5;     // a wider 95% interval cannot separate N log N (≈ 1.1) from N² (2): undetermined
export const BH_DISPATCHES = 31;    // per Barnes-Hut force evaluation (measureForces' `dispatches`; brute force: 1)
// Effective log-log slope of N log₂ N over the given sizes (null for fewer than 2)
export const nlognSlope = (ns) => fitPowerLaw(ns, ns.map(n => n * Math.log2(n))).slope;
export const NLOGN_EFFECTIVE = nlognSlope(scalingCounts(SCALING_DEFAULT_MAX));   // 1.11 over 1,024 … 65,536

const scaling = {
    id: 'scaling',
    levels: ['intermediate', 'advanced'],
    title: 'Scaling: Barnes-Hut vs brute force',
    question: 'At what N does the O(N log N) tree, rebuilt every step, beat the O(N²) tiled direct sum on your GPU?',
    settings: [presetSetting, {
        key: 'maxN', label: 'Largest N',
        options: Object.entries(SCALING_LIMITS).map(([max, l]) => ({ value: Number(max),
            label: `Barnes-Hut ${count(l.bh)}, brute force ${count(l.brute)}${Number(max) > SCALING_DEFAULT_MAX ? ' (fast GPUs)' : ''}` })),
        default: SCALING_DEFAULT_MAX,
    }],
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
        + 'at large N. In CI, on a CPU-emulated GPU (no timestamp-query, so wall ms; Largest N 65,536), the wall times cross '
        + 'between N = 8,192 (brute force about 30% faster) and 16,384 (Barnes-Hut about 10% faster; log-interpolated '
        + '≈ 14,000). With the fixed floor (the wall time at N = 1,024: 17 ms Barnes-Hut, 1.4–2.5 ms brute force) subtracted '
        + `and only the sizes at least ${FLOOR_FACTOR}× above it fitted, the slopes (95% CI) were 1.23 (1.16 … 1.30) for Barnes-Hut `
        + 'over N = 4,096 … 65,536, and 1.94 (0.83 … 3.05) for brute force over only 3 sizes (4,096 … 16,384: undetermined); '
        + 'without the correction, over all sizes, an earlier CI run gave 1.11 and 1.86. '
        + 'On an Apple M4 Pro (GPU ms) the floor is 0.6 ms for Barnes-Hut and under 0.1 ms for brute force, and it covers most '
        + 'of the default sweep: choose Largest N 262,144 on a fast GPU. Where the crossover falls depends on the GPU, driver '
        + 'and browser: read it from your medians and their spread, not from one number. Compile is pipeline and buffer '
        + 'creation after the warm-up (usually a pipeline-cache hit); the cold compile is in the discarded warm-up.',
    // Budget: up to 9 Barnes-Hut and 7 brute-force points × (1 warm-up + 5 repeats) = 96 evaluations, each with its own
    // set-up and read-back (default Largest N: 7 and 5, 72 evaluations). On a discrete or recent integrated GPU that is a
    // few seconds; on the CPU-emulated GPU in CI the default is 17–26 s. Every point keeps the full protocol (so the
    // record's warm-up/repeat counts hold for every row); instead, a brute-force point whose projected cost would leave
    // too little of the budget for the remaining Barnes-Hut points is skipped (see skipBrute), and interpret() says so.
    async *run(ctx, { preset, maxN }) {
        const limits = scalingLimits(maxN);
        const perCall = { bh: null, brute: null };   // ms per evaluation at the previous N (set-up to read-back)
        let prevN = null, bruteSkipped = false;
        const timed = async (mode, run) => {
            const t0 = performance.now();
            const m = await ctx.measure(run, PROTOCOL);
            perCall[mode] = (performance.now() - t0) / (PROTOCOL.warmup + PROTOCOL.repeats);
            return m;
        };
        for (const n of scalingCounts(maxN)) {
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
            if (n <= limits.brute && !bruteSkipped) {
                bruteSkipped = prevN !== null && skipBrute(n, prevN, perCall, ctx.remainingMs?.() ?? Infinity, limits.bh);
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
            'How large the fixed cost of one evaluation is (the time at the smallest N), and how much of the Barnes-Hut GPU '
                + 'time is the tree build (needs timestamp-query).',
        ],
        doesNotShow: [
            'A ranking of GPUs or browsers: one device, one browser, one run. Compare medians only where the spreads do not overlap.',
            'Asymptotic complexity: N goes to 65,536 or 262,144 (brute force to 16,384 or 65,536). The slopes are fitted only '
                + `over the sizes at least ${FLOOR_FACTOR}× above the fixed floor, with the floor subtracted, and interactions per `
                + 'body have not reached their log N regime.',
            'Accuracy: every point is θ = 0.5 (the θ-error experiment measures what that costs).',
            'Simulation frame rate: each evaluation here sets up, uploads, dispatches once and reads back; Explore pipelines its steps.',
        ],
    },
};

// Whether to skip brute force at N (and above): its projected cost, from the previous N's per-evaluation time
// scaled by (N / prevN)², plus the remaining Barnes-Hut points up to bhMax (from this N's Barnes-Hut time,
// measured just before, scaled by (M / N)^1.2), would not fit in the remaining budget. Exported for tests.
export function skipBrute(n, prevN, perCall, remainingMs, bhMax = SCALING_DEFAULT_MAX) {
    if (!(perCall.brute > 0) || !Number.isFinite(remainingMs)) return false;
    const calls = PROTOCOL.warmup + PROTOCOL.repeats;
    const brute = calls * perCall.brute * (n / prevN) ** 2;
    const bh = SCALING_COUNTS.filter(m => m > n && m <= bhMax)
        .reduce((s, m) => s + calls * (perCall.bh ?? 0) * (m / n) ** 1.2, 0);
    return brute + bh > remainingMs;
}

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

// The floor-corrected fit of one method (see "Scaling predictions"): the floor (median at the smallest measured N),
// the threshold max(FLOOR_FACTOR × floor, FIT_MIN_MS), the sizes at or above it, the fit of time − floor over them,
// and the plain fit over every size for contrast
export function floorFit(rows, key) {
    const all = rows.filter(r => typeof r[key] === 'number' && Number.isFinite(r[key]) && r[key] >= 0)
        .sort((a, b) => a.n - b.n);
    if (!all.length) return { floor: null, floorN: null, threshold: null, used: [], fit: fitPowerLaw([], []), allFit: fitPowerLaw([], []) };
    const floor = all[0][key], threshold = Math.max(FLOOR_FACTOR * floor, FIT_MIN_MS);
    const used = all.filter(r => r[key] >= threshold);
    return {
        floor, floorN: all[0].n, threshold, used: used.map(r => r.n),
        fit: fitPowerLaw(used.map(r => r.n), used.map(r => r[key] - floor)),
        allFit: fitPowerLaw(all.map(r => r.n), all.map(r => r[key])),
    };
}

// A slope interval wider than ± MAX_CI_HALF cannot tell N log N (≈ 1.1) from N² (2): no verdict
const tooWide = (fit) => fit.ci95 !== null && (fit.ci95[1] - fit.ci95[0]) / 2 > MAX_CI_HALF;
const range = (ns) => (ns.length ? `N = ${count(ns[0])} … ${count(ns.at(-1))}` : 'no sizes');
const floorMs = (v) => (v >= 0.1 ? `${v.toFixed(2)} ms` : `${(1000 * v).toFixed(0)} µs`);

// measured text and the method note shared by the two slope claims
function floorClaimText(f, basis) {
    const measured = f.fit.slope === null
        ? `${f.used.length} size${f.used.length === 1 ? '' : 's'} with a median ≥ ${floorMs(f.threshold ?? FIT_MIN_MS)} (no fit)`
        : `${slopeText(f.fit)} over ${range(f.used)}, floor subtracted`;
    const notes = f.floor === null ? [] : [`Fitted over the sizes whose median ${basis} is at least ${FLOOR_FACTOR}× the floor `
        + `(the median at ${at(f.floorN)}, ${floorMs(f.floor)}) and at least ${FIT_MIN_MS} ms, i.e. ≥ ${floorMs(f.threshold)}: `
        + `${range(f.used)}; the floor is subtracted before fitting. Smaller sizes are mostly fixed cost, so their time does `
        + 'not grow with the algorithm\'s work yet.'];
    if (f.allFit.slope !== null) notes.push(`Over all ${f.allFit.n} sizes, without the correction, the slope is ${f2(f.allFit.slope)}.`);
    if (f.floor !== null && !f.fit.ci95) notes.push('At least 3 sizes above the threshold are needed for an uncertainty, so this '
        + 'stays undetermined; on a fast GPU choose a larger Largest N.');
    else if (tooWide(f.fit)) notes.push(`Too noisy to decide: the 95% interval is wider than ± ${MAX_CI_HALF}, enough to fit both `
        + 'N log N and N²; rerun, or choose a larger Largest N.');
    return { measured, notes };
}
const at = (n) => `N = ${count(n)}`;

// Plain words for a time-vs-N fit: twice the stars took 2^slope times as long
function plainSlope(fit, agrees) {
    if (fit.slope === null) return 'Too few sizes took long enough to time; try a larger Largest N';
    const f = timesText(doublingFactor(fit.slope));
    return agrees === null ? `About ${f} times as long, but the times were too uneven to be sure` : `${f} times as long`;
}

export function interpretScaling(rows, { maxN } = {}) {
    const limits = scalingLimits(maxN);
    const brute = timeBasis(rows, 'brute'), bh = timeBasis(rows, 'bh');
    const bruteF = floorFit(brute.rows, brute.key), bhF = floorFit(bh.rows, bh.key);
    const bruteAgrees = tooWide(bruteF.fit) ? null : slopeAgrees(bruteF.fit, BRUTE_ORDER, BRUTE_ORDER_TOLERANCE);
    // N log N over the fitted sizes (all measured sizes, or the sweep, when there are fewer than 2)
    const bhNs = bhF.used.length >= 2 ? bhF.used : bh.rows.length >= 2 ? bh.rows.map(r => r.n) : scalingCounts(maxN);
    const nlogn = nlognSlope(bhNs), bhUpper = nlogn + BH_EXTRA;
    const bhAgrees = tooWide(bhF.fit) ? null : slopeAgrees(bhF.fit, (BH_LOWER + bhUpper) / 2, (bhUpper - BH_LOWER) / 2);
    const skipped = rows.filter(r => r.n <= limits.brute && typeof r.bruteWall !== 'number').map(r => count(r.n));
    const bruteText = floorClaimText(bruteF, brute.label), bhText = floorClaimText(bhF, bh.label);
    const claims = [{
        key: 'brute-n2',
        claim: `Brute-force time vs N (log-log slope, ${brute.label})`,
        predicted: `${BRUTE_ORDER}: N − 1 interactions for each of N bodies, O(N²) (± ${BRUTE_ORDER_TOLERANCE} for the rest of `
            + `the fixed cost), fitted above ${FLOOR_FACTOR}× the fixed floor`,
        measured: bruteText.measured,
        agrees: bruteAgrees,
        plain: {
            claim: 'Brute force: twice the stars',
            predicted: 'About 4 times as long: twice the stars, each pulled by twice as many others',
            measured: plainSlope(bruteF.fit, bruteAgrees),
        },
        note: [...bruteText.notes, bruteAgrees === null || bruteAgrees ? null
            : bruteF.fit.slope < BRUTE_ORDER ? 'Below 2 even above the floor: at the smallest fitted N the GPU is not yet full '
                + '(too few bodies for its cores), or the floor estimate is off; a larger Largest N decides.'
                : 'Above 2: the large-N points are slower than N² predicts (clocks, caches or occupancy changing during the run); '
                    + 'check whether the spreads overlap.',
        skipped.length ? `Brute force was skipped at N = ${skipped.join(', ')} to leave budget for the Barnes-Hut points.` : null]
            .filter(Boolean).join(' ') || null,
    }, {
        key: 'bh-nlogn',
        claim: `Barnes-Hut time vs N (log-log slope, ${bh.label})`,
        predicted: `about N log N: ${f2(nlogn)} over ${range(bhNs)}${bhF.used.length >= 2 ? ' (the fitted sizes)' : ''}; accepted `
            + `${f2(BH_LOWER)} … ${f2(bhUpper)} (the O(N) tree build pulls it to 1, interactions per body growing faster than `
            + `log N push it up), fitted above ${FLOOR_FACTOR}× the fixed floor`,
        measured: bhText.measured,
        agrees: bhAgrees,
        plain: {
            claim: 'Barnes-Hut: twice the stars',
            predicted: `A bit more than twice as long (2.0 to ${timesText(doublingFactor(bhUpper))} times): far stars are pulled in as groups`,
            measured: plainSlope(bhF.fit, bhAgrees),
        },
        note: [...bhText.notes, bhAgrees === null || bhAgrees ? null
            : bhF.fit.slope < BH_LOWER ? 'Below 1 even above the floor: a fixed or slowly growing cost is still a large part of '
                + 'the time at the fitted sizes; a larger Largest N decides.'
                : `Above ${f2(bhUpper)}: the traversal grows faster than N log N here (scattered memory access once the tree `
                    + 'leaves the caches, or the growth of interactions per body below).'].filter(Boolean).join(' ') || null,
    }];

    const work = fitRows(rows, 'n', 'interactions');
    const workAgrees = rangeVerdict(work, 0, 1);
    const workNs = rows.filter(r => r.interactions > 0).map(r => r.n);
    claims.push({
        key: 'bh-interactions',
        claim: 'Barnes-Hut interactions per body vs N (log-log slope)',
        predicted: `between 0 and 1: grows, but slower than brute force's N − 1 (→ log N for large N, `
            + `${f2((workNs.length >= 2 ? nlognSlope(workNs) : NLOGN_EFFECTIVE) - 1)} here; CI: 0.37)`,
        measured: slopeText(work),
        agrees: workAgrees,
        plain: workNs.length >= 2 ? {
            claim: 'Barnes-Hut: pulls per star as stars are added',
            predicted: 'They grow, but far more slowly than the number of stars',
            measured: `${countText(rows.find(r => r.interactions > 0).interactions)} at ${countText(workNs[0])} stars, `
                + `${countText(rows.filter(r => r.interactions > 0).at(-1).interactions)} at ${countText(workNs.at(-1))} stars`,
        } : undefined,
        note: workAgrees === null ? (work.ci95 ? 'The interval straddles a bound of the predicted range.' : tooFew(work, 'N values'))
            : 'Deterministic counts (one distribution, one seed): the interval reflects curvature, not noise.',
    });

    const x = crossover(rows);
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
        // Brute force faster everywhere it ran: extrapolate the two plain fits (only meaningful if the slopes differ)
        const bruteFit = bruteF.allFit, bhFit = bhF.allFit;
        const sameBasis = brute.label === bh.label && bruteFit.slope !== null && bhFit.slope !== null;
        const extrapolated = sameBasis && bruteFit.slope > bhFit.slope
            ? 10 ** ((bhFit.intercept - bruteFit.intercept) / (bruteFit.slope - bhFit.slope)) : null;
        measured = `above ${at(x.lastN)}: brute force is faster at every N where both ran`
            + (extrapolated ? `; extrapolating the two fits gives N ≈ ${count(Math.round(extrapolated / 1000) * 1000)}` : '');
        agrees = sameBasis && !(bruteFit.slope > bhFit.slope) ? false : null;
        note += agrees === false ? ' The fitted slopes do not cross at all, which contradicts O(N log N) vs O(N²): look at the spreads.'
            : ' The prediction is not tested inside the sweep; the extrapolation assumes both slopes hold beyond it.';
    }
    const plainCross = !x.points ? null : x.state === 'bracketed' ? `Barnes-Hut was faster from between ${countText(x.lo)} and ${countText(x.hi)} stars on`
        : x.state === 'below' ? `Barnes-Hut was faster at every size, from ${countText(x.firstN)} stars`
            : `Brute force was faster at every size tried, up to ${countText(x.lastN)} stars`;
    claims.push({
        key: 'bh-crossover',
        claim: 'Crossover N (Barnes-Hut faster from here)',
        predicted: 'exists: O(N²) overtakes O(N log N) at some N (CI, software GPU, wall ms: between 8,192 and 16,384)',
        measured, agrees, note,
        plain: plainCross ? {
            claim: 'Barnes-Hut beats brute force once there are enough stars',
            predicted: 'Yes, at some number of stars that depends on your GPU',
            measured: plainCross,
        } : undefined,
    });

    const shares = rows.filter(r => r.treeShare > 0);
    const shareFit = fitRows(shares, 'n', 'treeShare');
    const shareAgrees = rangeVerdict(shareFit, -Infinity, 0);
    claims.push({
        key: 'tree-share',
        claim: 'Tree-build share of Barnes-Hut GPU time vs N',
        predicted: 'falls with N: the build is O(N) plus about 30 fixed dispatches, the traversal O(N log N)',
        measured: shares.length >= 2 ? `${pct(shares[0].treeShare)} at ${at(shares[0].n)} → ${pct(shares.at(-1).treeShare)} at `
            + `${at(shares.at(-1).n)}; slope ${slopeText(shareFit)}` : 'needs GPU times (timestamp-query)',
        agrees: shareAgrees,
        plain: shares.length >= 2 ? {
            claim: 'Barnes-Hut: the part of the time spent building the tree',
            predicted: 'It shrinks as stars are added',
            measured: `${pct(shares[0].treeShare)} at ${countText(shares[0].n)} stars, ${pct(shares.at(-1).treeShare)} at ${countText(shares.at(-1).n)} stars`,
        } : undefined,
        note: !shares.length ? 'This browser or GPU does not offer timestamp-query, so the split cannot be measured.'
            : shareAgrees === null ? (shareFit.ci95 ? 'The share does not change clearly with N.' : tooFew(shareFit, 'N values'))
                : shareAgrees ? null : 'The share grows with N: the tree build scales worse than the traversal on this device.',
    });

    // Informational: the floor itself, which the two slope fits subtract
    const floors = [bhF.floor !== null ? `Barnes-Hut ${floorMs(bhF.floor)} at ${at(bhF.floorN)} (${bh.label}; ≤ `
        + `${floorMs(bhF.floor / BH_DISPATCHES)} per dispatch over ${BH_DISPATCHES})` : null,
    bruteF.floor !== null ? `brute force ${floorMs(bruteF.floor)} at ${at(bruteF.floorN)} (${brute.label}; 1 dispatch)` : null]
        .filter(Boolean);
    const largestBh = bh.rows.at(-1);
    claims.push({
        key: 'fixed-floor',
        claim: 'Fixed cost per force evaluation (the time at the smallest N)',
        predicted: `informational, no verdict: a cost that does not grow with N (submission, synchronisation, and for Barnes-Hut `
            + `${BH_DISPATCHES} dispatches, 30 of them the tree build); the time at the smallest N bounds it from above`,
        measured: floors.length ? floors.join('; ') : 'no rows yet',
        agrees: null,
        note: !floors.length ? null : [`The slope fits above use only sizes at least ${FLOOR_FACTOR}× above these (and ≥ ${FIT_MIN_MS} ms) `
            + 'and subtract them.', largestBh && bhF.floor > 0 ? `At the largest Barnes-Hut N (${count(largestBh.n)}) the floor is `
            + `${pct(bhF.floor / largestBh[bh.key])} of the time.` : null].filter(Boolean).join(' '),
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
    levels: ['intermediate', 'advanced'],
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
        + 'Fitted slopes (95% CI): brute force 2.25 (1.87 … 2.64), Barnes-Hut 0.04 (0.01 … 0.07), floor 6.5e-4. '
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
            key: 'drift-order',
            claim: 'Max |ΔE/E₀| vs dt, exact forces (log-log slope = order)',
            predicted: `${DRIFT_ORDER}: leapfrog is second order and symplectic, so the energy error stays in a band ∝ dt²`,
            measured: slopeText(fit),
            agrees,
            plain: {
                claim: 'Exact forces: energy error vs time step',
                predicted: 'Half the time step gives about a quarter of the error',
                measured: fit.slope === null ? 'Not enough time steps yet'
                    : `Half the time step gave an error ${timesText(doublingFactor(fit.slope))} times smaller`,
            },
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
        key: 'drift-bh-flat',
        claim: 'Max |ΔE/E₀| vs dt, Barnes-Hut θ = 0.5 (log-log slope)',
        predicted: `0: dt-independent, set by the tree's force error (accepted ± ${FLOOR_SLOPE_TOLERANCE} for a residual dt² part)`,
        measured: slopeText(fit),
        agrees: flat,
        plain: {
            claim: 'Barnes-Hut forces: energy error vs time step',
            predicted: 'A smaller time step does not help: the tree\'s own error sets a floor',
            measured: fit.slope === null ? 'Not enough time steps yet'
                : `Half the time step changed the error by a factor of ${doublingFactor(fit.slope).toFixed(2)}`,
        },
        note: flat === null ? tooFew(fit, 'time steps')
            : flat ? 'A smaller dt does not help: the tree, not the integrator, limits the energy error.'
                : fit.slope > 0 ? 'The error still falls with dt: on this run the leapfrog error is not negligible against the tree error.'
                    : 'The error grows as dt shrinks: more steps accumulate more tree error, so the floor is not a fixed level here.',
    }, {
        key: 'drift-bh-floor',
        claim: 'Size of the Barnes-Hut floor',
        predicted: `${sci1(FLOOR_RANGE[0])} … ${sci1(FLOOR_RANGE[1])}: of the order of the tree's relative potential error at θ = 0.5 `
            + '(≈ 4e-3 per body), partly averaged out; far above the exact-force error (≤ 7.4e-5)',
        measured: floor === null ? 'no rows yet' : `${sci(floor)} (median over ${drifts.length} dt; ${sci(Math.min(...drifts))} … ${sci(Math.max(...drifts))})`,
        agrees: inRange,
        plain: floor === null ? undefined : {
            claim: 'Barnes-Hut forces: the size of that floor',
            predicted: 'Between 0.0001 and 0.01 of the total energy',
            measured: `About ${+floor.toPrecision(2)} of the total energy`,
        },
        note: inRange === null ? null : 'An order-of-magnitude prediction, not a fit; one distribution, one seed (CI, Plummer 2,048: 6.4e-4 … 6.9e-4).',
    }];
}

export default [thetaError, scaling, energyDrift];
