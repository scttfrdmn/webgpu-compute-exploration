// Measure experiments for reaction-diffusion (js/examples/reaction-diffusion.js). Interface: see the top
// of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureSteps), which runs the shipped
// Gray-Scott pipeline without a canvas; the CPU reference is the float64 step of
// js/refs/reaction-diffusion.js with the example's exported model constants. Every sweep is a fixed
// list with hard caps (exported so test/reaction-diffusion-measure.test.mjs can check them): grids up
// to 2048², at most 1,024 timed steps per grid, at most 4,000 steps (on 128²) for the accuracy run.

import { measureSteps, seedCells, seededRandom, PRESETS, DU, DV, DT, W_EDGE, W_CORNER } from '../../examples/reaction-diffusion.js';
import { stepReference } from '../../refs/reaction-diffusion.js';

export const MODEL = { DU, DV, DT, W_EDGE, W_CORNER };
export const SEED = 1;

// ---- 1. Grid size vs time per step ------------------------------------------------------------------
export const SIDES = [128, 256, 512, 1024, 2048];
export const MAX_SIDE = 2048;
export const WARMUP = 32;
// Timed steps per grid: about 2^26 cell updates, between 64 and 1,024 steps
export const stepsFor = (side) => Math.max(64, Math.min(1024, Math.round(2 ** 26 / (side * side))));

const ms = (v) => v.toFixed(3);
const sci = (v) => v.toExponential(2);
const mcups = (v) => (v / 1e6).toFixed(0);

const scaling = {
    id: 'scaling',
    title: 'Grid size vs time per step',
    question: 'How does the time per Gray-Scott step grow with the number of cells, and when does the GPU stop being limited by launch overhead?',
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
        { key: 'gpuRate', label: 'M cell updates / s (GPU)', format: mcups },
        { key: 'wallRate', label: 'M cell updates / s (wall)', format: mcups },
    ],
    charts: [
        { x: 'cells', xLabel: 'cells', logX: true, y: ['gpuMsPerStep', 'wallMsPerStep'], yLabel: 'ms per step', logY: true,
            labels: { gpuMsPerStep: 'GPU', wallMsPerStep: 'wall' } },
        { x: 'cells', xLabel: 'cells', logX: true, y: ['gpuRate', 'wallRate'], yLabel: 'cell updates per second', logY: true,
            labels: { gpuRate: 'GPU time', wallRate: 'wall time' } },
    ],
    baseline: 'CI, on a CPU-emulated GPU (Mesa lavapipe, wall time only): 0.6 to 1.0 ms per step at 128² and 6 to 10 ms at '
        + '512², about 17 to 43 million cell updates per second (software timings vary by up to 2× between runs). A '
        + 'discrete GPU should be 100 to 1,000× faster at large grids, and much less than that at 128², where one step is '
        + 'only 16,384 threads and the launch overhead dominates.',
    async *run(ctx, { maxSide = MAX_SIDE }) {
        for (const side of SIDES) {
            if (side > Math.min(maxSide, MAX_SIDE)) break;
            await ctx.step();
            const steps = stepsFor(side);
            const r = await measureSteps(ctx.gpuDevice, { width: side, height: side, seed: SEED, steps, warmup: WARMUP,
                between: () => ctx.step() });
            const cells = side * side;
            yield { side, cells, steps, gpuMsPerStep: r.gpuMsPerStep, wallMsPerStep: r.wallMsPerStep,
                gpuRate: r.gpuMsPerStep ? cells / r.gpuMsPerStep * 1e3 : null, wallRate: cells / r.wallMsPerStep * 1e3 };
        }
    },
};

// ---- 2. Single precision vs float64 -----------------------------------------------------------------
export const ACCURACY_SIDE = 128;
export const CHECKPOINTS = [1, 10, 100, 300, 1000, 2000, 3000, 4000];
export const CPU_CHUNK = 50;     // float64 steps between yields to the page (about 15 ms of CPU each)

// Max |GPU − float64| over U and V, the RMS difference, and the fraction of cells where V > 0.1
export function compare(cells, U, V) {
    let max = 0, sq = 0, covered = 0;
    const n = U.length;
    for (let i = 0; i < n; i++) {
        const du = cells[2 * i] - U[i], dv = cells[2 * i + 1] - V[i];
        max = Math.max(max, Math.abs(du), Math.abs(dv));
        sq += du * du + dv * dv;
        if (V[i] > 0.1) covered++;
    }
    return { max, rms: Math.sqrt(sq / (2 * n)), coverage: covered / n };
}

const presetOptions = ['coral', 'mitosis', 'maze', 'chaos'].map(k => ({ value: k, label: PRESETS[k].name }));

const accuracy = {
    id: 'precision',
    title: 'Single precision vs float64',
    question: 'How far does the GPU\'s f32 solution drift from a float64 solution of the same equations, and does the pattern type matter?',
    settings: [{ key: 'preset', label: 'Preset', options: presetOptions, default: 'coral' }],
    budgetMs: 20000,
    columns: [
        { key: 'steps', label: 'steps' },
        { key: 'maxError', label: 'max |f32 − f64|', format: sci },
        { key: 'rmsError', label: 'RMS difference', format: sci },
        { key: 'coverage', label: 'cells with V > 0.1', format: (v) => `${(100 * v).toFixed(0)}%` },
    ],
    charts: [{
        x: 'steps', xLabel: 'Gray-Scott steps', logX: true, y: ['maxError', 'rmsError'], yLabel: '|f32 − f64| in U, V', logY: true,
        labels: { maxError: 'max', rmsError: 'RMS' },
    }],
    baseline: 'CI (shipped shader on a software GPU, 128², seed 1): Coral differs from float64 by 7e-8 after one step, 2.6e-6 '
        + 'after 100, 1.0e-4 after 2,000 and 5.1e-4 after 4,000. Chaos follows it at first, then separates exponentially: '
        + '1.6e-4 after 1,000 steps, 1.7e-3 after 2,000, 1.8e-2 after 3,000 and 0.76 after 4,000, with the same '
        + 'statistics. Same equations, same start: only rounding differs.',
    async *run(ctx, { preset = 'coral' }) {
        const { feed, kill } = PRESETS[preset];
        const w = ACCURACY_SIDE, h = ACCURACY_SIDE;
        const init = seedCells(w, h, seededRandom(SEED));
        const last = CHECKPOINTS[CHECKPOINTS.length - 1];
        await ctx.step();
        const gpu = await measureSteps(ctx.gpuDevice, { width: w, height: h, cells: init, feed, kill, steps: last,
            readAt: CHECKPOINTS, between: () => ctx.step() });
        let U = new Float64Array(w * h), V = new Float64Array(w * h);
        for (let i = 0; i < w * h; i++) { U[i] = init[2 * i]; V[i] = init[2 * i + 1]; }
        let done = 0;
        for (const sample of gpu.samples) {
            while (done < sample.step) {
                const n = Math.min(CPU_CHUNK, sample.step - done);
                for (let s = 0; s < n; s++) [U, V] = stepReference(U, V, w, h, feed, kill, MODEL);
                done += n;
                await ctx.step();
            }
            const c = compare(sample.cells, U, V);
            yield { steps: sample.step, maxError: c.max, rmsError: c.rms, coverage: c.coverage };
        }
    },
};

export default [scaling, accuracy];
