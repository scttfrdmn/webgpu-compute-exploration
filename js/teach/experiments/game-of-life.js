// Measure experiments for the Game of Life (js/examples/game-of-life.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureLife(), which runs the shipped stencil
// shader with the example's ping-pong buffers, many generations per compute pass. The reference is the
// exact CPU rule of js/refs/life.js (B3/S23 on a torus): a GPU grid either matches it cell for cell or
// it is wrong. Caps: grids ≤ 2,048², 100 generations per timed batch, 1,000 generations of soup on ≤ 256².

import { measureLife, MAX_LIFE_SIDE } from '../../examples/game-of-life.js';
import { lifeStep } from '../../refs/life.js';
import { mulberry32 } from '../../refs/random.js';

export const SIDES = [128, 256, 512, 1024, 2048];
export const BATCH = 100;                 // generations per timed batch
export const SOUP_SIDES = [128, 256];
export const CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
export const DENSITY = 0.3;               // the example's Random button: P(alive) = 0.3
export const SEED = 3;

export function randomGrid(width, height, density = DENSITY, seed = SEED) {
    const rand = mulberry32(seed);
    return Uint32Array.from({ length: width * height }, () => (rand() < density ? 1 : 0));
}

export const population = (grid) => grid.reduce((s, c) => s + c, 0);

export function mismatches(a, b) {
    let bad = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) bad++;
    return bad;
}

const ms = (v) => v.toFixed(4);
const rate = (cells, t) => (t > 0 ? cells / t / 1e6 : null);   // cell updates per ms → billions per second

const throughput = {
    id: 'throughput',
    title: 'Grid size vs cell updates per second',
    question: 'How many cell updates per second does the stencil reach as the grid grows, and when does the GPU stop being idle?',
    budgetMs: 20000,
    columns: [
        { key: 'side', label: 'grid', format: (v) => `${v} × ${v}` },
        { key: 'cells', label: 'cells' },
        { key: 'gpuMsPerGen', label: 'GPU ms / generation', format: ms },
        { key: 'wallMsPerGen', label: 'wall ms / generation', format: ms },
        { key: 'gcellsGpu', label: 'G cell updates/s (GPU)', format: (v) => v.toFixed(2) },
        { key: 'gcellsWall', label: 'G cell updates/s (wall)', format: (v) => v.toFixed(2) },
        { key: 'gensPerSec', label: 'generations/s (wall)', format: (v) => v.toFixed(0) },
    ],
    charts: [
        {
            x: 'cells', xLabel: 'cells', logX: true, y: ['gcellsGpu', 'gcellsWall'], yLabel: 'billion cell updates per second', logY: true,
            labels: { gcellsGpu: 'GPU time', gcellsWall: 'wall' },
        },
        {
            x: 'cells', xLabel: 'cells', logX: true, y: ['gpuMsPerGen', 'wallMsPerGen'], yLabel: 'ms per generation', logY: true,
            labels: { gpuMsPerGen: 'GPU', wallMsPerGen: 'wall' },
        },
    ],
    baseline: `Each batch runs ${BATCH} generations in one compute pass (the example runs one per animation frame, `
        + 'at most 60 per second). Every cell reads 9 u32 values and writes one, so the stencil is memory-bound; '
        + 'neighbouring threads read overlapping cells, which the caches serve. CI (CPU-emulated GPU): 0.013 G cell '
        + 'updates/s at 128² rising to 0.049 at 2,048² (85 ms per generation); a real GPU is orders of magnitude faster.',
    async *run(ctx) {
        for (const side of SIDES) {
            if (side > MAX_LIFE_SIDE) break;
            const grid = randomGrid(side, side);
            await ctx.step();
            const r = await measureLife(ctx.gpuDevice, { grid, width: side, height: side, generations: BATCH, warmup: 1, repeats: 3, readback: false });
            const cells = side * side;
            const gpuMsPerGen = r.gpuMs === null ? null : r.gpuMs / BATCH;
            const wallMsPerGen = r.wallMs / BATCH;
            yield {
                side, cells, gpuMsPerGen, wallMsPerGen, gcellsGpu: rate(cells, gpuMsPerGen), gcellsWall: rate(cells, wallMsPerGen),
                gensPerSec: wallMsPerGen > 0 ? 1000 / wallMsPerGen : null,
            };
        }
    },
};

const soup = {
    id: 'soup',
    title: 'Random soup vs the CPU rule',
    question: 'Does the GPU reproduce the exact rule for 1,000 generations, and how does a random soup thin out over time?',
    settings: [{ key: 'side', label: 'Grid', options: SOUP_SIDES.map(s => ({ value: s, label: `${s} × ${s}` })), default: 256 }],
    budgetMs: 20000,
    columns: [
        { key: 'generation', label: 'generation' },
        { key: 'live', label: 'live cells' },
        { key: 'density', label: 'live fraction', format: (v) => v.toFixed(4) },
        { key: 'mismatches', label: 'cells ≠ CPU' },
    ],
    charts: [{ x: 'generation', xLabel: 'generation', logX: true, y: ['density'], yLabel: 'live fraction', labels: { density: 'GPU' } }],
    baseline: 'CI (shipped shader on a CPU-emulated GPU, 256 × 256, seed 3): 0 mismatching cells at every checkpoint. '
        + 'The live fraction jumps from 0.30 to 0.34 in the first generation (crowded births), then falls to 0.21 after '
        + '10, 0.095 after 100 and 0.046 after 1,000, still settling toward the stable and oscillating "ash" a soup leaves behind.',
    async *run(ctx, { side }) {
        const s = Math.min(side, 256);
        let gpu = randomGrid(s, s), cpu = gpu;
        let done = 0;
        for (const generation of CHECKPOINTS) {
            await ctx.step();
            const r = await measureLife(ctx.gpuDevice, { grid: gpu, width: s, height: s, generations: generation - done });
            gpu = r.grid;
            for (; done < generation; done++) {
                cpu = lifeStep(cpu, s, s);
                if (done % 50 === 49) await ctx.step();   // the CPU rule is the slow part; keep the page responsive
            }
            const live = population(gpu);
            yield { generation, live, density: live / (s * s), mismatches: mismatches(gpu, cpu) };
        }
    },
};

export default [throughput, soup];
