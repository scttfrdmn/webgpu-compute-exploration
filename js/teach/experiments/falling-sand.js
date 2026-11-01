// Measure experiments for falling sand (js/examples/falling-sand.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureSand), which runs the shipped
// Margolus pipeline without a canvas and reads the cells back; the CPU reference is an exact material
// count (js/refs/falling-sand.js). Every sweep is a fixed list with hard caps (exported so
// test/falling-sand-measure.test.mjs can check them): grids up to 2560 × 1920 with at most 512 timed
// steps each, and at most 2,000 steps of the 320 × 240 sandbox for the conservation check.

import { measureSand, buildSandbox, MATERIAL_IDS, MAX_STEPS } from '../../examples/falling-sand.js';
import { materialCounts, changedCells } from '../../refs/falling-sand.js';

const { STONE, SAND, WATER, OIL, FIRE, PLANT } = MATERIAL_IDS;
export const SANDBOX_W = 320, SANDBOX_H = 240;

// The shipped 320 × 240 sandbox repeated to fill a larger grid (same material mix at every size)
export function tiledSandbox(width, height) {
    const tile = buildSandbox(SANDBOX_W, SANDBOX_H);
    const grid = new Uint32Array(width * height);
    for (let y = 0; y < height; y++) {
        const row = (y % SANDBOX_H) * SANDBOX_W;
        for (let x = 0; x < width; x++) grid[y * width + x] = tile[row + (x % SANDBOX_W)];
    }
    return grid;
}

// Scenes for the conservation check, all 320 × 240
export function scene(key) {
    const grid = buildSandbox(SANDBOX_W, SANDBOX_H);
    if (key === 'inert') {
        // Plants turned to stone: no reaction can happen (they all need fire, acid or a plant)
        for (let i = 0; i < grid.length; i++) if ((grid[i] & 15) === PLANT) grid[i] = (grid[i] & ~15) | STONE;
    } else if (key === 'plants') {
        // A plant patch on the floor of the basin, under water: plants grow into the water around them
        for (let y = 216; y < 226; y++) for (let x = 150; x < 170; x++) grid[y * SANDBOX_W + x] = PLANT | (((x * 5 + y) & 15) << 4);
    } else if (key === 'fire') {
        // A layer of fire (life 120 steps) in the air just above the basin's oil slick
        for (let y = 176; y < 180; y++) for (let x = 101; x < 219; x++) grid[y * SANDBOX_W + x] = FIRE | (((x * 7 + y) & 15) << 4) | (120 << 8);
    }
    return grid;
}

// ---- 1. Grid size and steps per submit vs time per step ---------------------------------------------
export const SIZES = [[160, 120], [320, 240], [640, 480], [1280, 960], [2560, 1920]];
export const MAX_CELLS = 2560 * 1920;
// Timed steps per grid: a power of two near 2^25 / cells, between 32 and 512
export const stepsFor = (cells) => Math.max(32, Math.min(512, 2 ** Math.round(Math.log2(2 ** 25 / cells))));

const ms = (v) => v.toFixed(3);

const scaling = {
    id: 'scaling',
    title: 'Grid size and steps per submit vs time per step',
    question: 'How does the cost of a Margolus step grow with the grid, and how much does batching steps into one submit save?',
    settings: [
        { key: 'maxCells', label: 'Largest grid', options: SIZES.slice(2).map(([w, h]) => ({ value: w * h, label: `${w} × ${h}` })), default: MAX_CELLS },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'size', label: 'grid', format: String },
        { key: 'cells', label: 'cells' },
        { key: 'steps', label: 'timed steps' },
        { key: 'gpuMsPerStep', label: 'GPU ms / step', format: ms },
        { key: 'wall8', label: `wall ms / step (${MAX_STEPS} per submit)`, format: ms },
        { key: 'wall1', label: 'wall ms / step (1 per submit)', format: ms },
        { key: 'rate', label: 'M cell updates / s', format: (v) => (v / 1e6).toFixed(0) },
    ],
    charts: [
        { x: 'cells', xLabel: 'cells', logX: true, y: ['gpuMsPerStep', 'wall8', 'wall1'], yLabel: 'ms per step', logY: true,
            labels: { gpuMsPerStep: 'GPU', wall8: `wall, ${MAX_STEPS} / submit`, wall1: 'wall, 1 / submit' } },
        { x: 'cells', xLabel: 'cells', logX: true, y: ['rate'], yLabel: 'cell updates per second', logY: true },
    ],
    baseline: 'CI, on a CPU-emulated GPU (wall time): 0.74 ms per step at 160 × 120 and 8.9 ms at 640 × 480 with 8 steps per '
        + 'submit (26 to 35 million cell updates per second); one step per submit costs 0.99 and 10.4 ms. On a real GPU '
        + 'the small grids are limited by the per-submit cost, so the gap between the two wall-time lines is much larger.',
    async *run(ctx, { maxCells = MAX_CELLS }) {
        for (const [w, h] of SIZES) {
            const cells = w * h;
            if (cells > Math.min(maxCells, MAX_CELLS)) break;
            const grid = tiledSandbox(w, h);
            const steps = stepsFor(cells);
            await ctx.step();
            const batched = await measureSand(ctx.gpuDevice, { width: w, height: h, grid, steps, warmup: MAX_STEPS,
                stepsPerSubmit: MAX_STEPS, between: () => ctx.step() });
            await ctx.step();
            const single = await measureSand(ctx.gpuDevice, { width: w, height: h, grid, steps: Math.min(steps, 64), warmup: 1,
                stepsPerSubmit: 1, between: () => ctx.step() });
            const perStep = batched.gpuMsPerStep ?? batched.wallMsPerStep;
            yield { size: `${w} × ${h}`, cells, steps, gpuMsPerStep: batched.gpuMsPerStep, wall8: batched.wallMsPerStep,
                wall1: single.wallMsPerStep, rate: cells / perStep * 1e3 };
        }
    },
};

// ---- 2. Conservation --------------------------------------------------------------------------------
export const CHECKPOINTS = [10, 30, 100, 300, 1000, 2000];

const delta = (v) => (v > 0 ? `+${v}` : String(v));

const conservation = {
    id: 'conservation',
    title: 'What is conserved',
    question: 'Movement only swaps cells inside a block, so which material counts stay exactly constant, and which change through reactions?',
    settings: [{
        key: 'scene', label: 'Scene', default: 'plants',
        options: [
            { value: 'inert', label: 'Sandbox, plants turned to stone (no reactions)' },
            { value: 'plants', label: 'Sandbox with plants under water' },
            { value: 'fire', label: 'Sandbox with fire on the oil' },
        ],
    }],
    budgetMs: 20000,
    columns: [
        { key: 'steps', label: 'steps' },
        { key: 'sand', label: 'Δ sand', format: delta },
        { key: 'stone', label: 'Δ stone', format: delta },
        { key: 'oil', label: 'Δ oil', format: delta },
        { key: 'water', label: 'Δ water', format: delta },
        { key: 'plant', label: 'Δ plant', format: delta },
        { key: 'waterPlant', label: 'Δ (water + plant)', format: delta },
        { key: 'fire', label: 'fire cells', format: String },
        { key: 'changed', label: 'cells changed since last row', format: (v) => `${(100 * v).toFixed(2)}%` },
    ],
    charts: [
        { x: 'steps', xLabel: 'steps', logX: true, y: ['sand', 'oil', 'water', 'waterPlant'], yLabel: 'change in cell count',
            labels: { sand: 'sand', oil: 'oil', water: 'water', waterPlant: 'water + plant' } },
        { x: 'steps', xLabel: 'steps', logX: true, y: ['changed'], yLabel: 'fraction of cells changed since the previous row', logY: true },
    ],
    baseline: 'CI (shipped shader on a software GPU): with no reactions every count is exactly unchanged after 2,000 steps. '
        + 'With fire, 408 of the 708 oil cells burn within about 300 steps and the fire is out by step 1,000, while sand, '
        + 'stone and water are untouched. With plants under water, 996 water cells turn into plant in 2,000 steps and '
        + 'water + plant stays exact.',
    async *run(ctx, { scene: key = 'plants' }) {
        const grid = scene(key);
        const c0 = materialCounts(grid);
        await ctx.step();
        const r = await measureSand(ctx.gpuDevice, { width: SANDBOX_W, height: SANDBOX_H, grid,
            steps: CHECKPOINTS[CHECKPOINTS.length - 1], readAt: CHECKPOINTS, between: () => ctx.step() });
        let previous = grid;
        for (const { step, cells } of r.samples) {
            const c = materialCounts(cells);
            const d = (m) => c[m] - c0[m];
            yield { steps: step, sand: d(SAND), stone: d(STONE), oil: d(OIL), water: d(WATER), plant: d(PLANT),
                waterPlant: d(WATER) + d(PLANT), fire: c[FIRE], changed: changedCells(previous, cells) / cells.length };
            previous = cells;
            await ctx.step();
        }
    },
};

export default [scaling, conservation];
