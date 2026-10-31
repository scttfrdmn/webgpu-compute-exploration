// Measure experiments for N-body (js/examples/nbody.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureForces, measureEnergyDrift), which
// runs the shipped pipelines without a canvas; the CPU reference is the float64 direct sum of
// js/refs/nbody.js. Every sweep is a fixed list with hard caps (exported so test/nbody-measure.test.mjs
// can check them): Barnes-Hut N ≤ 65,536, brute force N ≤ 16,384, ≤ 400 leapfrog steps per run.

import { generateBodies, measureForces, measureEnergyDrift } from '../../examples/nbody.js';
import { directForcesAt, relErrors } from '../../refs/nbody.js';

export const BH_MAX_N = 65536;
export const BRUTE_MAX_N = 16384;
export const SOFTENING = 0.05;      // the example's default ε
export const SEED = 5;              // the bodies of the CI force test (test/gpu/nbody.gpu.test.mjs)

// ---- 1. Opening angle vs force error ----------------------------------------------------------------
export const THETAS = [0.3, 0.4, 0.5, 0.7, 0.9, 1.2];
export const ERROR_COUNTS = [4096, 16384];
export const SAMPLE = 512;          // target bodies checked against the direct sum (512 × N in float64)
// Measured in CI (Plummer, N = 4,096, seed 5, ε = 0.05): the CPU replica of cs_bh_force and the shipped
// shader on a software GPU agree on the median error at θ = 0.5
export const CI_MEDIAN_AT_HALF = 4.2e-3;

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
        { key: 'traversalMs', label: 'traversal GPU ms', format: ms },
        { key: 'gpuMs', label: 'GPU ms (tree + traversal)', format: ms },
        { key: 'wallMs', label: 'wall ms', format: ms },
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
            const r = await measureForces(ctx.gpuDevice, { positions, mode: 'bh', theta, softening: SOFTENING, repeats: 3 });
            const acc = new Float64Array(3 * targets.length);
            targets.forEach((i, t) => { for (let c = 0; c < 3; c++) acc[3 * t + c] = r.accel[4 * i + c]; });
            const e = relErrors(acc, ref.acc, targets.length);
            yield { theta, median: e.median, p99: e.p99, interactions: r.interactionsPerBody,
                traversalMs: r.passes.force, gpuMs: r.gpuMs, wallMs: r.wallMs };
        }
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
        { key: 'bhGpu', label: 'Barnes-Hut GPU ms', format: ms },
        { key: 'bruteGpu', label: 'brute force GPU ms', format: ms },
        { key: 'bhWall', label: 'Barnes-Hut wall ms', format: ms },
        { key: 'bruteWall', label: 'brute force wall ms', format: ms },
        { key: 'treeShare', label: 'tree build share', format: (v) => `${(100 * v).toFixed(0)}%` },
        { key: 'interactions', label: 'BH interactions / body', format: (v) => v.toFixed(0) },
    ],
    charts: [
        { x: 'n', xLabel: 'bodies N', logX: true, y: ['bhGpu', 'bruteGpu'], yLabel: 'GPU ms per force evaluation', logY: true,
            labels: { bhGpu: 'Barnes-Hut', bruteGpu: 'brute force' },
            title: 'GPU ms per force evaluation vs N (needs timestamp-query)' },
        { x: 'n', xLabel: 'bodies N', logX: true, y: ['bhWall', 'bruteWall'], yLabel: 'wall ms per force evaluation', logY: true,
            labels: { bhWall: 'Barnes-Hut', bruteWall: 'brute force' } },
    ],
    baseline: 'Brute force does exactly N − 1 interactions per body; Barnes-Hut at θ = 0.5 does a few hundred to about a '
        + 'thousand (248 at N = 1,024, 1,140 at 65,536 in CI) plus a 30-dispatch tree build, so it loses at small N and wins '
        + 'at large N. In CI, on a CPU-emulated GPU, the wall times cross between N = 8,192 (93 vs 83 ms) and 16,384 '
        + `(212 vs 294 ms). Brute force stops at N = ${BRUTE_MAX_N.toLocaleString('en-US')} here (the example allows 32,768).`,
    async *run(ctx, { preset }) {
        for (const n of SCALING_COUNTS) {
            if (n > BH_MAX_N) break;
            const { positions } = generateBodies(preset, n, SOFTENING, SEED);
            await ctx.step();
            const bh = await measureForces(ctx.gpuDevice, { positions, mode: 'bh', theta: 0.5, softening: SOFTENING, repeats: 3, accel: false });
            const row = { n, bhGpu: bh.gpuMs, bhWall: bh.wallMs, bruteGpu: null, bruteWall: null,
                treeShare: bh.gpuMs ? bh.passes.tree / bh.gpuMs : null, interactions: bh.interactionsPerBody };
            if (n <= BRUTE_MAX_N) {
                await ctx.step();
                const brute = await measureForces(ctx.gpuDevice, { positions, mode: 'brute', softening: SOFTENING, repeats: 3, accel: false });
                row.bruteGpu = brute.gpuMs;
                row.bruteWall = brute.wallMs;
            }
            yield row;
        }
    },
};

// ---- 3. Time step vs energy drift -------------------------------------------------------------------
export const DTS = [0.04, 0.02, 0.01, 0.005];
export const DRIFT_TIME = 2;        // simulated time per run: steps = DRIFT_TIME / dt ≤ 400
export const DRIFT_COUNTS = [2048, 4096];
export const DRIFT_SAMPLES = 25;    // energy readbacks per run (16 bytes each)
// Measured in CI (brute force, Plummer N = 2,048, seed 5): max |ΔE/E₀| over t = 2 at dt = 0.01
export const CI_DRIFT_AT_001 = 1.3e-6;

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
        { key: 'wallMs', label: 'wall ms', format: (v) => v.toFixed(0) },
    ],
    charts: [{
        x: 'dt', xLabel: 'time step dt', logX: true, y: ['maxDrift'], yLabel: `max |ΔE/E₀| over t = ${DRIFT_TIME}`, logY: true,
        reference: {
            label: 'dt² (CI)',
            points: [DTS[DTS.length - 1], DTS[0]].map(dt => [dt, CI_DRIFT_AT_001 * (dt / 0.01) ** 2]),
        },
    }],
    baseline: `CI (software GPU, brute force, Plummer N = 2,048, t = ${DRIFT_TIME}): max |ΔE/E₀| 3.4e-5 / 5.5e-6 / 1.3e-6 / `
        + '3.0e-7 at dt = 0.04 / 0.02 / 0.01 / 0.005, about 4× per halving (dt²). With Barnes-Hut at θ = 0.5 the error '
        + 'stays near 3e-4 whatever dt is: the tree force and potential error dominates. The f32 energy sum itself is good to about 1e-7.',
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
};

export default [thetaError, scaling, energyDrift];
