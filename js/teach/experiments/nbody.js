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
import { summarize, withSpread } from '../stats.js';
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
};

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
    async *run(ctx, { preset }) {
        for (const n of SCALING_COUNTS) {
            if (n > BH_MAX_N) break;
            const { positions } = generateBodies(preset, n, SOFTENING, SEED);
            await ctx.step();
            const bh = await ctx.measure(forceRun(ctx, positions, { mode: 'bh', theta: 0.5, accel: false }), PROTOCOL);
            const last = bh.samples[bh.samples.length - 1];
            const share = summarize(bh.samples.map(s => (s.gpuMs ? s.passes.tree / s.gpuMs : null)));
            const row = { n, ...withSpread('bhGpu', bh.gpuMs), ...withSpread('bhWall', bh.wallMs),
                ...withSpread('bruteGpu', null), ...withSpread('bruteWall', null),
                treeShare: share.median, interactions: last.interactionsPerBody,
                ...withSpread('compileMs', bh.phases.compile), ...withSpread('uploadMs', bh.phases.upload),
                ...withSpread('readbackMs', bh.phases.readback) };
            if (n <= BRUTE_MAX_N) {
                await ctx.step();
                const brute = await ctx.measure(forceRun(ctx, positions, { mode: 'brute', accel: false }), PROTOCOL);
                Object.assign(row, withSpread('bruteGpu', brute.gpuMs), withSpread('bruteWall', brute.wallMs));
            }
            yield row;
        }
    },
    // No verify: timings have no reference value, and correctness is the theta-error experiment's job
};

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
};

export default [thetaError, scaling, energyDrift];
