// Measure experiments for the cloth (js/examples/cloth.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureCloth), which runs the shipped
// compute pass without a canvas, MAX_SUBSTEPS substeps per command buffer as in an animation frame.
// The reference is the rest state: an inextensible cloth has every structural link at its rest
// length, and js/refs/cloth.js measures how far the solved positions are from it. Every sweep is a
// fixed list with hard caps (exported so test/cloth-measure.test.mjs can check them): grid ≤ 128,
// iterations ≤ 30 (the slider's range), ≤ 2 s of simulated time (480 substeps) per row.

import { measureCloth, GRID_SIZES, MAX_ITERATIONS } from '../../examples/cloth.js';
import { stretchStats } from '../../refs/cloth.js';

export const ITERATIONS = [1, 2, 4, 8, 16, 30];
export const SETTLE_SUBSTEPS = 480;      // 2 s of simulated time: the hanging cloth has mostly settled
export const TIMED_SUBSTEPS = 32;        // four timed command buffers of 8 substeps
export const STRETCH_GRIDS = [32, 64, 128];
// Measured in CI (software GPU, 32 × 32, two corners, after 2 s): 99th-percentile structural overstretch
export const CI_P99 = { 1: 7.8e-3, 2: 5.6e-3, 4: 3.9e-3, 8: 2.9e-3, 16: 1.7e-3, 30: 1.0e-3 };

const ms = (v) => v.toFixed(3);
const pct = (v) => `${(100 * v).toFixed(3)}%`;

const presetSetting = {
    key: 'preset', label: 'Cloth',
    options: [{ value: 'corners', label: 'Hang from two corners' }, { value: 'edge', label: 'Hang from top edge' }],
    default: 'corners',
};

const iterationsStretch = {
    id: 'iterations-stretch',
    title: 'Solver iterations vs stretch',
    question: 'How close to inextensible does the cloth get as the Gauss-Seidel solver runs more iterations per substep, and what does each iteration cost?',
    settings: [
        { key: 'n', label: 'Grid', options: STRETCH_GRIDS.map(n => ({ value: n, label: `${n} × ${n}` })), default: 64 },
        presetSetting,
    ],
    budgetMs: 20000,
    columns: [
        { key: 'iterations', label: 'iterations' },
        { key: 'maxStretch', label: 'max stretch', format: pct },
        { key: 'p99Stretch', label: 'p99 stretch', format: pct },
        { key: 'meanStretch', label: 'mean stretch', format: pct },
        { key: 'dispatches', label: 'dispatches / substep' },
        { key: 'gpuMs', label: 'GPU ms / substep', format: ms },
        { key: 'wallMs', label: 'wall ms / substep', format: ms },
    ],
    charts: [
        {
            x: 'iterations', xLabel: 'solver iterations per substep', logX: true,
            y: ['maxStretch', 'p99Stretch'], yLabel: 'structural overstretch (length / rest − 1)', logY: true,
            labels: { maxStretch: 'max', p99Stretch: '99th percentile' },
            // 1/iterations through the visitor's own 99th percentile at 8 iterations
            reference: (rows) => {
                const at8 = rows.find(r => r.iterations === 8)?.p99Stretch;
                return at8 > 0 ? { label: '∝ 1 / iterations', points: [1, MAX_ITERATIONS].map(k => [k, at8 * 8 / k]) } : null;
            },
        },
        { x: 'iterations', xLabel: 'solver iterations per substep', logX: true, y: ['gpuMs', 'wallMs'],
            yLabel: 'ms per substep', logY: true, labels: { gpuMs: 'GPU', wallMs: 'wall' } },
    ],
    baseline: 'CI (shipped shader on a software GPU, 32 × 32, two corners, after 2 s): the 99th percentile of the '
        + `structural overstretch falls from ${pct(CI_P99[1])} at 1 iteration to ${pct(CI_P99[8])} at 8 and ${pct(CI_P99[30])} at 30, `
        + 'slower than 1/iterations at first and close to it later; the maximum (one link, 0.92% → 0.22%) jumps around with '
        + 'the swing of the cloth. Larger grids stretch more: more weight per link, the same iterations. A substep is '
        + '3 + 12 × iterations dispatches (15 at 1 iteration, 363 at 30), so the cost grows linearly with the iterations.',
    async *run(ctx, { n, preset }) {
        for (const iterations of ITERATIONS) {
            await ctx.step();
            const r = await measureCloth(ctx.gpuDevice, { n, preset, iterations, substeps: SETTLE_SUBSTEPS,
                timed: TIMED_SUBSTEPS, between: () => ctx.step() });
            const s = stretchStats(r.positions, r.constraintData);
            yield { iterations, maxStretch: s.max, p99Stretch: s.p99, meanStretch: s.mean,
                dispatches: r.dispatchesPerSubstep, gpuMs: r.gpuMsPerSubstep, wallMs: r.wallMsPerSubstep };
        }
    },
};

// ---- 2. Grid size vs cost ---------------------------------------------------------------------------
export const SCALING_SETTLE = 16;        // two untimed command buffers before the timed ones

const gridScaling = {
    id: 'grid-scaling',
    title: 'Grid size vs time per substep',
    question: 'How does the cost of a substep grow with the number of particles, and when does the GPU stop being limited by dispatch overhead?',
    settings: [
        { key: 'iterations', label: 'Iterations', options: [4, 8, 16].map(k => ({ value: k, label: String(k) })), default: 8 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'grid' },
        { key: 'particles', label: 'particles' },
        { key: 'constraints', label: 'constraints' },
        { key: 'gpuMs', label: 'GPU ms / substep', format: ms },
        { key: 'wallMs', label: 'wall ms / substep', format: ms },
        { key: 'mProjPerS', label: 'M projections / s (wall)', format: (v) => v.toFixed(1) },
        { key: 'realtime', label: 'substeps / frame at 60 FPS', format: (v) => v.toFixed(1) },
    ],
    charts: [
        { x: 'particles', xLabel: 'particles', logX: true, y: ['gpuMs', 'wallMs'], yLabel: 'ms per substep', logY: true,
            labels: { gpuMs: 'GPU', wallMs: 'wall' } },
        { x: 'particles', xLabel: 'particles', logX: true, y: ['mProjPerS'], yLabel: 'million constraint projections per second', logY: true },
    ],
    baseline: 'The example runs 4 substeps per 60 FPS frame (h = 1/240 s, at most 8 per frame). A substep is about 100 '
        + 'small dispatches (8 iterations), most of them a few thousand threads, so on a real GPU expect the time to grow '
        + 'much more slowly than N at the small grids (launch overhead dominates) and the throughput to rise with N. '
        + 'CI runs this on a CPU-emulated GPU, whose timings vary several-fold between runs and say little about a real GPU.',
    async *run(ctx, { iterations }) {
        for (const n of GRID_SIZES) {
            await ctx.step();
            const r = await measureCloth(ctx.gpuDevice, { n, preset: 'corners', iterations, substeps: SCALING_SETTLE,
                timed: TIMED_SUBSTEPS, between: () => ctx.step() });
            const perSubstep = r.wallMsPerSubstep;
            yield { n, particles: r.count, constraints: r.numConstraints, gpuMs: r.gpuMsPerSubstep, wallMs: perSubstep,
                mProjPerS: r.numConstraints * iterations / perSubstep / 1e3,
                realtime: (1000 / 60) / (r.gpuMsPerSubstep ?? perSubstep) };
        }
    },
};

export default [iterationsStretch, gridScaling];
