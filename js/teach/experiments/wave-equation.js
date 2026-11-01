// Measure experiments for the wave equation (js/examples/wave-equation.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureWave), which runs the shipped
// cs_step pipeline without a canvas and reads both time levels back; the CPU reference is the
// scheme's discrete energy and the von Neumann growth factor of js/refs/wave.js. Every sweep is a fixed
// list with hard caps (exported so test/wave-equation-measure.test.mjs can check them): the CFL sweep
// runs at most 1,000 steps per Courant number on 128 × 96 (reading both levels back every 10 steps) and
// stops a run as soon as it diverges; the
// scaling sweep uses grids up to 2048² and at most 1,024 timed steps per grid.

import { measureWave, CFL_LIMIT } from '../../examples/wave-equation.js';
import { energy, growthFactor } from '../../refs/wave.js';

// ---- 1. Courant number vs stability -----------------------------------------------------------------
export const COURANTS = [0.3, 0.5, 0.6, 0.65, 0.7, 0.705, 0.71, 0.72, 0.75];
export const CFL_W = 128, CFL_H = 96;
export const CFL_STEPS = 1000;
export const READ_EVERY = 10;       // readback of both levels (2 × 48 KB) and a CPU energy every 10 steps
export const DIVERGED = 1e3;        // max|u| / max|u₀| above this counts as blown up (the run stops)

// Largest |u|; Infinity once any value has overflowed or become NaN
const maxAbs = (u) => {
    let m = 0;
    for (const v of u) {
        const a = Math.abs(v);
        if (!(a <= 1e30)) return Infinity;
        if (a > m) m = a;
    }
    return m;
};
const sci = (v) => v.toExponential(2);
const ms = (v) => v.toFixed(3);

// A vertical line at the CFL limit spanning the plotted values (log axis: positive values only)
const limitLine = (key) => (rows) => {
    const ys = rows.map(r => r[key]).filter(v => typeof v === 'number' && Number.isFinite(v) && v > 0);
    if (!ys.length) return { label: 'C = 1/√2', points: [] };
    return { label: 'C = 1/√2', points: [[CFL_LIMIT, Math.min(...ys)], [CFL_LIMIT, Math.max(...ys)]] };
};

const cfl = {
    id: 'cfl',
    title: 'Courant number vs stability (CFL)',
    question: 'What happens to the leapfrog scheme just below and just above the CFL limit C = 1/√2 ≈ 0.707?',
    budgetMs: 20000,
    columns: [
        { key: 'courant', label: 'C', format: String },
        { key: 'theory', label: 'max |g| (von Neumann)', format: (v) => v.toFixed(4) },
        { key: 'steps', label: 'steps run' },
        { key: 'growth', label: 'max|u| / max|u₀|', format: sci },
        { key: 'drift', label: 'max |ΔE/E₀|', format: sci },
        { key: 'divergedAt', label: 'blew up at step', format: (v) => (v ? String(v) : 'no') },
    ],
    charts: [
        { x: 'courant', xLabel: 'Courant number C', y: ['growth'], yLabel: 'max|u| / max|u₀|', logY: true,
            reference: limitLine('growth') },
        { x: 'courant', xLabel: 'Courant number C', y: ['drift'], yLabel: 'max relative energy change', logY: true,
            reference: limitLine('drift') },
    ],
    baseline: 'CI (shipped shader on a software GPU, 128 × 96, one drop, no damping): every C ≤ 0.705 runs 1,000 steps with '
        + 'the amplitude never above its starting value and the discrete energy constant to 4e-7. Above the limit the run '
        + 'passes 1,000 × its starting amplitude after 110 steps at C = 0.71 (|g| = 1.20), 50 at 0.72 (|g| = 1.46) and 30 '
        + 'at 0.75 (|g| = 2.0). Nothing in the initial drop excites the checkerboard mode: rounding error seeds it.',
    async *run(ctx) {
        for (const courant of COURANTS) {
            await ctx.step();
            let u0 = null, e0 = null, growth = 0, drift = 0, divergedAt = null;
            const r = await measureWave(ctx.gpuDevice, { width: CFL_W, height: CFL_H, preset: 'open', courant, damping: 0,
                steps: CFL_STEPS, readEvery: READ_EVERY, between: () => ctx.step(),
                onSample: ({ step, cur, prev }) => {
                    const m = maxAbs(cur);
                    const e = energy(prev, cur, CFL_W, CFL_H, Math.fround(courant * courant));
                    if (u0 === null) { u0 = m; e0 = e; }
                    growth = Number.isFinite(m) ? Math.max(growth, m / u0) : Infinity;
                    drift = Number.isFinite(e) ? Math.max(drift, Math.abs(e - e0) / Math.abs(e0)) : Infinity;
                    if (!(m <= DIVERGED * u0)) { divergedAt = step; return false; }
                    return true;
                } });
            yield { courant, theory: growthFactor(courant), steps: r.steps, growth, drift, divergedAt };
        }
    },
};

// ---- 2. Grid size vs time per step ------------------------------------------------------------------
export const SIDES = [128, 256, 512, 1024, 2048];
export const MAX_SIDE = 2048;
export const WARMUP = 16;
export const stepsFor = (side) => Math.max(64, Math.min(1024, Math.round(2 ** 26 / (side * side))));

const scaling = {
    id: 'scaling',
    title: 'Grid size vs time per step',
    question: 'How does the time per leapfrog step grow with the grid, and how much of a small step is launch overhead?',
    settings: [
        { key: 'maxSide', label: 'Largest grid', options: [512, 1024, 2048].map(s => ({ value: s, label: `${s} × ${s}` })), default: 2048 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'side', label: 'grid', format: (v) => `${v} × ${v}` },
        { key: 'cells', label: 'cells' },
        { key: 'steps', label: 'timed steps' },
        { key: 'gpuMsPerStep', label: 'GPU ms / step', format: ms },
        { key: 'wallMsPerStep', label: 'wall ms / step', format: ms },
        { key: 'gpuRate', label: 'M cell updates / s (GPU)', format: (v) => (v / 1e6).toFixed(0) },
        { key: 'wallRate', label: 'M cell updates / s (wall)', format: (v) => (v / 1e6).toFixed(0) },
    ],
    charts: [
        { x: 'cells', xLabel: 'cells', logX: true, y: ['gpuMsPerStep', 'wallMsPerStep'], yLabel: 'ms per step', logY: true,
            labels: { gpuMsPerStep: 'GPU', wallMsPerStep: 'wall' } },
        { x: 'cells', xLabel: 'cells', logX: true, y: ['gpuRate', 'wallRate'], yLabel: 'cell updates per second', logY: true,
            labels: { gpuRate: 'GPU time', wallRate: 'wall time' } },
    ],
    baseline: 'CI, on a CPU-emulated GPU (wall time): about 0.6 / 1.7 / 4 ms per step at 128² / 256² / 512², 26 to 65 million '
        + 'cell updates per second, rising with the grid as the fixed cost per submit is spread over more cells (software '
        + 'timings vary by up to 2× between runs). On a real GPU expect the small grids to cost about the same per step '
        + 'whatever their size.',
    async *run(ctx, { maxSide = MAX_SIDE }) {
        for (const side of SIDES) {
            if (side > Math.min(maxSide, MAX_SIDE)) break;
            await ctx.step();
            const steps = stepsFor(side);
            const r = await measureWave(ctx.gpuDevice, { width: side, height: side, preset: 'open', courant: 0.5,
                damping: 0.004, steps, warmup: WARMUP, between: () => ctx.step() });
            const cells = side * side;
            yield { side, cells, steps, gpuMsPerStep: r.gpuMsPerStep, wallMsPerStep: r.wallMsPerStep,
                gpuRate: r.gpuMsPerStep ? cells / r.gpuMsPerStep * 1e3 : null, wallRate: cells / r.wallMsPerStep * 1e3 };
        }
    },
};

export default [cfl, scaling];
