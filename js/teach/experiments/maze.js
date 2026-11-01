// Measure experiments for maze solving (js/examples/maze.js). Interface: see the top of js/teach/measure.js.
//
// Mazes are generated on the CPU with the example's own seeded generators (generateMaze). The GPU work
// goes through the example's compute-only API (measureBFS), which runs the shipped cs_bfs_step pipeline
// in batches with the example's termination readback, without a canvas. The CPU reference is the queue
// BFS of js/refs/maze.js: every distance up to the goal's must match exactly, and the path traced back
// from the goal must have dist[goal] + 1 cells. Every sweep is a fixed list with hard caps (exported so
// test/maze-measure.test.mjs can check them): mazes ≤ 1023 × 767 cells (the example's largest), ≤ 100,000
// BFS levels per run (measureBFS's cap), batches of 1 … 512 levels per submit.

import { measureBFS, generateMaze, extractPath, MAZE_SIZES, GENERATORS, MAX_MEASURE_CELLS } from '../../examples/maze.js';
import { queueBFS, INF } from '../../refs/maze.js';

export const SEED = 12345;
export const SIZE_KEYS = ['small', 'medium', 'large', 'huge'];
export const BRAIDS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3];
export const BATCHES = [1, 4, 16, 64, 256, 512];
export const LEVELS_PER_SUBMIT = 256;

// GPU distances vs the queue BFS: cells up to the goal's level must agree exactly; cells farther than
// the goal were never expanded (the flood stops there) and must still be INF.
export function verifyBFS(gpuDist, ref, levels) {
    let mismatches = 0;
    for (let i = 0; i < ref.length; i++) {
        const expected = ref[i] !== INF && ref[i] <= levels ? ref[i] : INF;
        if (gpuDist[i] !== expected) mismatches++;
    }
    return mismatches;
}

// One maze, one GPU BFS from the top-left to the bottom-right cell (the example's default endpoints),
// checked against the CPU queue BFS
async function solveRow(ctx, { sizeKey, generator, braid, levelsPerSubmit = LEVELS_PER_SUBMIT }) {
    const { w, h } = MAZE_SIZES[sizeKey];
    const t0 = performance.now();
    const { cells } = generateMaze(generator, w, h, braid, SEED);
    const generateMs = performance.now() - t0;
    await ctx.step();
    const r = await measureBFS(ctx.gpuDevice, { cells, width: w, height: h, levelsPerSubmit, between: () => ctx.step() });
    const t1 = performance.now();
    const ref = queueBFS(cells, w, h, 0);
    const cpuMs = performance.now() - t1;
    const levels = r.found ? r.levels : null;
    const mismatches = r.found ? verifyBFS(r.dist, ref, levels) : null;
    const path = r.found ? extractPath(cells, r.dist, w, h, 0, w * h - 1) : [];
    const ok = r.found && levels === ref[w * h - 1] && mismatches === 0 && path.length === levels + 1;
    return {
        cells: w * h, levels, visited: r.visited, dispatches: r.dispatches, submits: r.submits,
        gpuMs: r.gpuMs, wallMs: r.wallMs, cpuMs, generateMs,
        cellsPerSec: r.wallMs > 0 ? r.visited / (r.wallMs / 1000) : null,
        msPerLevel: r.dispatches > 0 ? (r.gpuMs ?? r.wallMs) / r.dispatches : null,
        verified: ok ? `yes (path ${path.length.toLocaleString('en-US')} cells)` : (r.found ? `NO (${mismatches} cells differ)` : 'not reached'),
    };
}

// Keeps only the declared columns of a row (solveRow returns more than each experiment shows)
const pick = (row, columns) => Object.fromEntries(columns.map(c => [c.key, row[c.key] ?? null]));

const generatorSetting = {
    key: 'generator', label: 'Generator',
    options: Object.entries(GENERATORS).map(([value, label]) => ({ value, label })), default: 'kruskal',
};
const ms = (v) => v.toFixed(1);
const big = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)} M` : v.toFixed(0));
const percent = (v) => `${Math.round(100 * v)}%`;

const resultColumns = [
    { key: 'levels', label: 'BFS levels (shortest path)' },
    { key: 'visited', label: 'cells reached' },
    { key: 'dispatches', label: 'dispatches' },
    { key: 'gpuMs', label: 'GPU ms', format: ms },
    { key: 'wallMs', label: 'wall ms (incl. readbacks)', format: ms },
    { key: 'cpuMs', label: 'CPU queue BFS ms', format: ms },
    { key: 'cellsPerSec', label: 'cells reached / s (wall)', format: big },
    { key: 'verified', label: 'verified vs queue BFS', format: String },
];

// ---- 1. Maze size → levels and time -----------------------------------------------------------------
const sizeColumns = [{ key: 'cells', label: 'cells' }, ...resultColumns,
    { key: 'msPerLevel', label: 'ms per level', format: (v) => v.toFixed(3) },
    { key: 'generateMs', label: 'CPU generation ms', format: ms }];
const size = {
    id: 'size',
    title: 'Maze size vs BFS levels and time',
    question: 'A level-synchronous BFS needs one dispatch per level: how many levels do mazes of each size and generator need, and what does that cost?',
    settings: [
        generatorSetting,
        { key: 'braid', label: 'Loops', options: [0, 0.05, 0.2].map(b => ({ value: b, label: percent(b) })), default: 0 },
    ],
    budgetMs: 20000,
    columns: sizeColumns,
    charts: [
        { x: 'cells', xLabel: 'cells', logX: true, y: ['levels'], yLabel: 'BFS levels to the goal', logY: true },
        { x: 'cells', xLabel: 'cells', logX: true, y: ['gpuMs', 'wallMs', 'cpuMs'], yLabel: 'ms per solve', logY: true,
            labels: { gpuMs: 'GPU BFS (timestamps)', wallMs: 'GPU BFS wall', cpuMs: 'CPU queue BFS' } },
    ],
    baseline: 'CI (CPU-emulated GPU, Kruskal, no loops), cells → levels and wall ms: 2,961 → 212 in 27 ms; 48,705 → 822 in '
        + '463 ms; 195,713 → 2,020 in 2.5 s; 784,641 → 4,692 in 22 s, while the CPU queue BFS takes 0.14 to 48 ms. With 20% '
        + 'loops: 114, 456, 928 and 1,840 levels. Every distance matched the queue BFS. A perfect DFS maze of 1023 × 767 '
        + 'needs 73,964 levels: try it on a real GPU.',
    async *run(ctx, { generator, braid }) {
        for (const sizeKey of SIZE_KEYS) {
            const { w, h } = MAZE_SIZES[sizeKey];
            if (w * h > MAX_MEASURE_CELLS) break;
            yield pick(await solveRow(ctx, { sizeKey, generator, braid }), sizeColumns);
        }
    },
};

// ---- 2. Loops (braid) → levels ------------------------------------------------------------------------
const braidColumns = [{ key: 'braid', label: 'walls removed', format: percent }, ...resultColumns];
const braid = {
    id: 'braid',
    title: 'Loops vs BFS levels (511 × 383)',
    question: 'Removing a few walls creates shortcuts: how fast does the shortest path, and with it the number of sequential BFS levels, shrink?',
    settings: [generatorSetting],
    budgetMs: 20000,
    columns: braidColumns,
    charts: [
        { x: 'braid', xLabel: 'share of remaining walls removed', y: ['levels'], yLabel: 'BFS levels to the goal', logY: true },
        { x: 'braid', xLabel: 'share of remaining walls removed', y: ['gpuMs', 'wallMs'], yLabel: 'ms per solve', logY: true,
            labels: { gpuMs: 'GPU (timestamps)', wallMs: 'wall' } },
    ],
    baseline: 'CI (Kruskal, 511 × 383): 2,020 levels without loops, 1,474 with 1% of the walls removed, 1,136 at 5%, '
        + '928 at 20% and 896 at 30%, approaching the grid distance 510 + 382 = 892 that no path can beat. All verified.',
    async *run(ctx, { generator }) {
        for (const b of BRAIDS) {
            yield pick({ braid: b, ...(await solveRow(ctx, { sizeKey: 'large', generator, braid: b })) }, braidColumns);
        }
    },
};

// ---- 3. Levels per submit → wall time ---------------------------------------------------------------
const batchColumns = [{ key: 'batch', label: 'levels per submit' }, { key: 'submits', label: 'submits (readbacks)' },
    { key: 'overshoot', label: 'dispatches past the goal' }, ...resultColumns];
const batch = {
    id: 'batch',
    title: 'Levels per submit vs wall time',
    question: 'The CPU must read back whether the goal was reached: how much does checking after every level cost compared with batching hundreds of levels per submit?',
    settings: [
        { key: 'sizeKey', label: 'Maze', options: [{ value: 'medium', label: '255 × 191' }, { value: 'large', label: '511 × 383' }], default: 'medium' },
        generatorSetting,
    ],
    budgetMs: 20000,
    columns: batchColumns,
    charts: [{ x: 'batch', xLabel: 'levels per submit', logX: true, y: ['gpuMs', 'wallMs'], yLabel: 'ms per solve', logY: true,
        labels: { gpuMs: 'GPU (timestamps)', wallMs: 'wall (incl. readbacks)' } }],
    baseline: 'CI (Kruskal, 255 × 191, 822 levels), wall ms by levels per submit: 826 at 1 (822 readbacks), 471 at 4, '
        + '410 at 16, 297 at 64, 404 at 256 and 428 at 512, where 202 dispatches run past the goal. Fewer readbacks help '
        + 'until the wasted levels of the last batch cost more than the readbacks saved.',
    async *run(ctx, { sizeKey, generator }) {
        for (const levelsPerSubmit of BATCHES) {
            const row = await solveRow(ctx, { sizeKey, generator, braid: 0, levelsPerSubmit });
            const overshoot = row.levels === null ? null : row.dispatches - row.levels;
            yield pick({ batch: levelsPerSubmit, overshoot, ...row }, batchColumns);
        }
    },
};

export default [size, braid, batch];
