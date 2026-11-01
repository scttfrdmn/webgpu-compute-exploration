// Measure experiments for 3D diffusion-limited aggregation (js/examples/dla-3d.js). Interface: see the
// top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureDLA), which runs the shipped
// walk/stick pipeline without a canvas and reads the walkers and the grid back; the CPU reference
// (js/refs/dla.js) measures the aggregate: radius of gyration, mass-radius fractal dimension and
// 26-connectivity to the seed. Every sweep is a fixed list with hard caps (exported so
// test/dla-measure.test.mjs can check them): grids of 32³ or 64³, at most 5,000 walkers and 20,000 walk
// steps per stickiness value; at most 262,144 walkers and 250 steps per row of the scaling sweep.

import { measureDLA } from '../../examples/dla-3d.js';
import { aggregateStats } from '../../refs/dla.js';

// ---- 1. Stickiness vs structure ---------------------------------------------------------------------
export const STICKINESS = [1, 0.5, 0.2, 0.1, 0.05, 0.03];
export const GRIDS = { 32: 1200, 64: 5000 };       // grid side -> walkers (64³ with 5,000 is the Explore tab)
export const MAX_GROWTH_STEPS = 20000;
export const CHECK_EVERY = 100;                    // count stuck walkers every 100 steps (16 bytes per walker)
export const STOP_AT = 0.95;                       // a run ends once 95% of the walkers have stuck

// First step at which at least `fraction` of the walkers had stuck (from measureDLA's progress)
export function stepsTo(progress, walkers, fraction) {
    return progress.find(p => p.stuck >= fraction * walkers)?.step ?? null;
}

const f2 = (v) => v.toFixed(2);

const stickiness = {
    id: 'stickiness',
    title: 'Stickiness vs structure',
    question: 'How do the sticking probability and the growth time change the shape of the aggregate: its size, density and fractal dimension?',
    settings: [
        { key: 'grid', label: 'Grid', options: [{ value: 32, label: '32³, 1,200 walkers' }, { value: 64, label: '64³, 5,000 walkers' }], default: 64 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'stickiness', label: 'stickiness', format: String },
        { key: 'steps', label: 'walk steps run' },
        { key: 'half', label: 'steps to 50% stuck', format: (v) => (v === null ? '—' : String(v)) },
        { key: 'stuck', label: 'walkers stuck' },
        { key: 'cells', label: 'aggregate cells' },
        { key: 'rg', label: 'radius of gyration', format: f2 },
        { key: 'dimension', label: 'mass-radius dimension D', format: f2 },
        { key: 'connected', label: '26-connected to seed', format: (v) => `${(100 * v).toFixed(1)}%` },
        { key: 'msPerStep', label: 'ms / walk step', format: (v) => v.toFixed(3) },
    ],
    charts: [
        { x: 'stickiness', xLabel: 'stickiness', logX: true, y: ['dimension'], yLabel: 'mass-radius dimension D',
            reference: { label: 'D ≈ 2.5 (3D DLA)', points: [[0.03, 2.5], [1, 2.5]] } },
        { x: 'stickiness', xLabel: 'stickiness', logX: true, y: ['rg'], yLabel: 'radius of gyration (cells)' },
        { x: 'stickiness', xLabel: 'stickiness', logX: true, y: ['half'], yLabel: 'walk steps until half the walkers stuck', logY: true },
    ],
    baseline: 'Software GPU in CI, 64³ with 5,000 walkers: the radius of gyration shrinks from 24.8 cells at stickiness 1 to '
        + '17.4 at 0.03, and half the walkers take 700 → 3,500 steps to stick. D is 2.30 to 2.64 with no clear trend (one '
        + 'aggregate of 5,000 cells is small), and every cell is 26-connected to the seed. On 32³ with 1,200 walkers: R_g '
        + '13.0 → 9.5, 200 → 900 steps.',
    async *run(ctx, { grid = 64 }) {
        const gridSize = grid === 32 ? 32 : 64;
        const walkers = GRIDS[gridSize];
        for (const s of STICKINESS) {
            await ctx.step();
            const r = await measureDLA(ctx.gpuDevice, { gridSize, walkers, stickiness: s, rngSeed: 1, steps: MAX_GROWTH_STEPS,
                checkEvery: CHECK_EVERY, stopAt: STOP_AT, between: () => ctx.step() });
            await ctx.step();
            const a = aggregateStats(r.grid, gridSize);
            yield { stickiness: s, steps: r.steps, half: stepsTo(r.progress, walkers, 0.5), stuck: r.stuck, cells: a.cells,
                rg: a.rg, dimension: a.dimension, connected: a.connected, msPerStep: r.gpuMsPerStep ?? r.wallMsPerStep };
        }
    },
};

// ---- 2. Walkers vs time per step --------------------------------------------------------------------
export const WALKERS = [1024, 4096, 16384, 65536, 262144];
export const SCALING_STEPS = 200;
export const SCALING_WARMUP = 50;

const scaling = {
    id: 'scaling',
    title: 'Walkers vs time per step',
    question: 'How many walkers does it take to keep the GPU busy, and what does one walk step cost per walker?',
    settings: [
        { key: 'maxWalkers', label: 'Most walkers', options: WALKERS.slice(2).map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 262144 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'walkers', label: 'walkers' },
        { key: 'gpuMsPerStep', label: 'GPU ms / step', format: (v) => v.toFixed(4) },
        { key: 'wallMsPerStep', label: 'wall ms / step', format: (v) => v.toFixed(4) },
        { key: 'rate', label: 'M walker steps / s', format: (v) => (v / 1e6).toFixed(1) },
    ],
    charts: [
        { x: 'walkers', xLabel: 'walkers', logX: true, y: ['gpuMsPerStep', 'wallMsPerStep'], yLabel: 'ms per walk step', logY: true,
            labels: { gpuMsPerStep: 'GPU', wallMsPerStep: 'wall' } },
        { x: 'walkers', xLabel: 'walkers', logX: true, y: ['rate'], yLabel: 'walker steps per second', logY: true },
    ],
    baseline: 'CI, on a CPU-emulated GPU (wall time): 0.35 ms per step with 1,024 walkers and 9.4 ms with 65,536, so 3.0 to '
        + '6.9 million walker steps per second. On a real GPU the 5,000 walkers of the Explore tab are only 79 '
        + 'workgroups, far too few to fill it: the time per step should hardly change until tens of thousands of walkers.',
    async *run(ctx, { maxWalkers = WALKERS[WALKERS.length - 1] }) {
        for (const walkers of WALKERS) {
            if (walkers > maxWalkers) break;
            await ctx.step();
            const r = await measureDLA(ctx.gpuDevice, { gridSize: 64, walkers, stickiness: 0.5, rngSeed: 2, steps: SCALING_STEPS,
                warmup: SCALING_WARMUP, readGrid: false, between: () => ctx.step() });
            const perStep = r.gpuMsPerStep ?? r.wallMsPerStep;
            yield { walkers, gpuMsPerStep: r.gpuMsPerStep, wallMsPerStep: r.wallMsPerStep, rate: walkers / perStep * 1e3 };
        }
    },
};

export default [stickiness, scaling];
