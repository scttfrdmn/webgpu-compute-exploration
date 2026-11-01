// Measure experiments for k-means (js/examples/kmeans.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureKMeans), which runs the shipped
// cs_assign / cs_update iteration (encodeIteration) and its 80-byte Stats readback without a canvas; the
// CPU reference is the float64 Lloyd step of js/refs/kmeans.js. Points come from the example's own
// seeded generator. Every sweep is a fixed list with hard caps (exported so test/kmeans-measure.test.mjs
// can check them): N ≤ 500,000 (the example's maximum), K ≤ 16, 10 timed iterations per size, and at most
// 100 iterations per run to convergence (the example stops at 300).

import { measureKMeans, generatePoints, initCentroids, mulberry32, MAX_N, MAX_K } from '../../examples/kmeans.js';
import { lloyd64Step } from '../../refs/kmeans.js';

export const SEED = 7;
export const BLOBS = 8;
export const NOISE = 0.05;
export const TIMED_ITERATIONS = 10;
export const MAX_CONVERGE_ITERATIONS = 100;

export const POINT_COUNTS = [10000, 30000, 100000, 300000, 500000];
export const CLUSTER_COUNTS = [2, 4, 8, 12, 16];
export const TRIALS = [1, 2, 3, 4, 5];

// Measured in CI (software GPU, test/gpu/kmeans-measure.gpu.test.mjs): the GPU run vs a float64 Lloyd run
// from the same initial centroids, N = 100,000, K = 5
export const CI_CONVERGENCE = 'labels identical to the float64 run in all 10 trials (k-means++ and random), the same '
    + 'number of iterations, centers within 1e-7, and the f32 inertia within 8e-8 of a float64 sum at the same centers. '
    + 'k-means++ converged in 4–5 iterations in four trials (inertia 3,170) and in 34 in the fifth, at a worse local '
    + 'minimum (7,970); random initialization took 48–79 iterations and ended at about 8,000 in two of five.';

const ms = (v) => v.toFixed(2);
const sci = (v) => v.toExponential(2);

function cpuStepMs(points, n, k, cent) {
    const t0 = performance.now();
    lloyd64Step(points, n, k, cent);
    return performance.now() - t0;
}

// One row of the cost sweeps: TIMED_ITERATIONS GPU iterations (not to convergence), one float64 CPU step
async function costRow(ctx, n, k) {
    const points = generatePoints(n, BLOBS, NOISE, SEED);
    const centroids = initCentroids(points, n, k, 'random', mulberry32(SEED));
    await ctx.step();
    const r = await measureKMeans(ctx.gpuDevice, { points, centroids, iterations: TIMED_ITERATIONS, untilConverged: false });
    await ctx.step();
    const cpuMs = cpuStepMs(points, n, k, centroids);
    const per = r.gpuMs ?? r.wallMs;
    return { n, k, gpuMs: r.gpuMs, wallMs: r.wallMs, cpuMs,
        nsPerDistance: per > 0 ? per * 1e6 / (n * k) : null, speedup: r.wallMs > 0 ? cpuMs / r.wallMs : null };
}

const costColumns = [
    { key: 'n', label: 'points N' },
    { key: 'k', label: 'clusters K' },
    { key: 'gpuMs', label: 'GPU ms / iteration', format: ms },
    { key: 'wallMs', label: 'wall ms / iteration', format: ms },
    { key: 'cpuMs', label: 'CPU float64 ms / iteration', format: ms },
    { key: 'nsPerDistance', label: 'GPU ns per point·cluster', format: (v) => v.toFixed(3) },
    { key: 'speedup', label: 'CPU ms / GPU wall ms', format: (v) => v.toFixed(1) },
];
const costSeries = { y: ['gpuMs', 'wallMs', 'cpuMs'], yLabel: 'ms per Lloyd iteration', logY: true,
    labels: { gpuMs: 'GPU (timestamps)', wallMs: 'GPU wall (submit → Stats read)', cpuMs: 'CPU float64' } };

// ---- 1. Points vs time per iteration ----------------------------------------------------------------
const pointsSweep = {
    id: 'points',
    title: 'Points vs time per iteration',
    question: 'How does the cost of one Lloyd iteration grow with the number of points, on the GPU and on the CPU?',
    settings: [{ key: 'k', label: 'Clusters K', options: [4, 8, 16].map(k => ({ value: k, label: String(k) })), default: 8 }],
    budgetMs: 20000,
    columns: costColumns,
    charts: [{ x: 'n', xLabel: 'points N', logX: true, ...costSeries }],
    baseline: 'CI (CPU-emulated GPU, K = 8): wall 1.7 ms per iteration at N = 10,000, 10.9 ms at 100,000 and 47.6 ms at '
        + '500,000, close to linear in N; one float64 CPU step 0.46, 4.2 and 21 ms. The emulated GPU is about 2–4× slower than '
        + 'the CPU here, because it is itself running on CPU cores; measure a real one.',
    async *run(ctx, { k }) {
        for (const n of POINT_COUNTS) {
            if (n > MAX_N) break;
            yield await costRow(ctx, n, Math.min(k, MAX_K));
        }
    },
};

// ---- 2. Clusters vs time per iteration --------------------------------------------------------------
const clustersSweep = {
    id: 'clusters',
    title: 'Clusters vs time per iteration',
    question: 'The assignment step costs N·K distances; does the GPU time per iteration grow linearly with K?',
    settings: [{ key: 'n', label: 'Points', options: [100000, 500000].map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 100000 }],
    budgetMs: 20000,
    columns: costColumns,
    charts: [{ x: 'k', xLabel: 'clusters K', ...costSeries, logY: false }],
    baseline: 'CI (CPU-emulated GPU, N = 100,000): wall 3.8 ms per iteration at K = 2, 11.8 ms at 8 and 16.9 ms at 16; '
        + 'the float64 CPU step 1.6, 3.9 and 7.3 ms. Both grow with K, the GPU partly because cs_assign runs one tree '
        + 'reduction (8 barrier steps) per cluster.',
    async *run(ctx, { n }) {
        for (const k of CLUSTER_COUNTS) {
            if (k > MAX_K) break;
            yield await costRow(ctx, Math.min(n, MAX_N), k);
        }
    },
};

// ---- 3. Converged inertia vs a float64 Lloyd run ----------------------------------------------------
export const CONVERGE_N = 100000;
export const CONVERGE_K = 5;

// float64 Lloyd from `cent` until no label changes (at most `max` steps): { iterations, inertia, next, labels }
export function lloyd64Converge(points, n, k, cent, max = MAX_CONVERGE_ITERATIONS) {
    let prev = null, step = null, centroids = cent;
    for (let it = 1; it <= max; it++) {
        step = lloyd64Step(points, n, k, centroids);
        if (prev && step.labels.every((l, i) => l === prev[i])) return { iterations: it, converged: true, ...step };
        prev = step.labels;
        centroids = step.next;
    }
    return { iterations: max, converged: false, ...step };
}

const convergence = {
    id: 'convergence',
    title: 'Converged inertia: GPU (f32) vs CPU (float64)',
    question: 'Starting from the same centroids, does the single-precision GPU run reach the same clustering and the same objective as a float64 run?',
    settings: [{ key: 'init', label: 'Initial centroids', options: [{ value: 'kmeans++', label: 'k-means++' }, { value: 'random', label: 'Random (Forgy)' }], default: 'kmeans++' }],
    budgetMs: 20000,
    columns: [
        { key: 'trial', label: 'trial (seed)' },
        { key: 'gpuIterations', label: 'GPU iterations' },
        { key: 'cpuIterations', label: 'CPU iterations' },
        { key: 'gpuInertia', label: 'GPU inertia (f32 sum)', format: (v) => v.toFixed(4) },
        { key: 'cpuInertia', label: 'CPU inertia (float64)', format: (v) => v.toFixed(4) },
        { key: 'inertiaGap', label: 'relative gap', format: sci },
        { key: 'sumError', label: 'f32 sum vs float64 at GPU centroids', format: sci },
        { key: 'labelsDiffer', label: 'labels that differ' },
        { key: 'centroidShift', label: 'max centroid distance', format: sci },
    ],
    charts: [{ x: 'trial', xLabel: 'trial (initialization seed)', y: ['gpuIterations', 'cpuIterations'], yLabel: 'iterations to converge',
        labels: { gpuIterations: 'GPU', cpuIterations: 'CPU float64' } }],
    baseline: `CI (software GPU, N = ${CONVERGE_N.toLocaleString('en-US')}, K = ${CONVERGE_K}): ${CI_CONVERGENCE}`,
    async *run(ctx, { init }) {
        const n = CONVERGE_N, k = CONVERGE_K;
        const points = generatePoints(n, CONVERGE_K, NOISE, SEED);
        for (const trial of TRIALS) {
            const centroids = initCentroids(points, n, k, init, mulberry32(1000 + trial));
            await ctx.step();
            const r = await measureKMeans(ctx.gpuDevice, { points, centroids, iterations: MAX_CONVERGE_ITERATIONS, readLabels: true });
            await ctx.step();
            const cpu = lloyd64Converge(points, n, k, centroids);
            const atGpu = lloyd64Step(points, n, k, r.centroids);
            let labelsDiffer = 0;
            for (let i = 0; i < n; i++) if (r.labels[i] !== cpu.labels[i]) labelsDiffer++;
            let shift = 0;
            for (let j = 0; j < k; j++) shift = Math.max(shift, Math.hypot(r.centroids[2 * j] - cpu.next[2 * j], r.centroids[2 * j + 1] - cpu.next[2 * j + 1]));
            yield {
                trial, gpuIterations: r.converged ? r.iterations : null, cpuIterations: cpu.converged ? cpu.iterations : null,
                gpuInertia: r.inertia, cpuInertia: cpu.inertia,
                inertiaGap: (atGpu.inertia - cpu.inertia) / cpu.inertia,
                sumError: Math.abs(r.inertia - atGpu.inertia) / atGpu.inertia,
                labelsDiffer, centroidShift: shift,
            };
        }
    },
};

export default [pointsSweep, clustersSweep, convergence];
