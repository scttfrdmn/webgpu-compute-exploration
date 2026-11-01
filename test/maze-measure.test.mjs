// Maze Measure tab, CPU only: the compute-only API of js/examples/maze.js (measureBFS) and the
// experiments in js/teach/experiments/maze.js, driven through the mock device (test/helpers/mock-gpu.mjs),
// which records buffers and uniform writes but runs nothing (its readbacks are zeros).
// Checked: the example and js/refs/maze.js share one cell encoding, also in the shipped WGSL; a
// compute-only MazeSolver creates no render pipeline; the Params bytes of each batch (one slot per level,
// consecutive bfsStep values) match the WGSL struct; walls are uploaded without the path bit; the caps
// are enforced; verifyBFS accepts the exact BFS field and rejects a wrong one; every sweep stays within
// its caps and every row carries the declared columns. The real results are checked on a software GPU
// in test/gpu/maze-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as maze from '../js/examples/maze.js';
import { measureBFS, generateMaze, MazeSolver, MAZE_SIZES, MAX_MEASURE_CELLS, MAX_MEASURE_LEVELS } from '../js/examples/maze.js';
import * as refs from '../js/refs/maze.js';
import experiments, { SIZE_KEYS, BRAIDS, BATCHES, LEVELS_PER_SUBMIT, verifyBFS } from '../js/teach/experiments/maze.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const layout = structLayout(moduleCode(maze, 'compute'), 'Params');
const paramWrites = (device) => device.writes.filter(w => w.buffer.size === 256 * 512 && (w.buffer.usage & GPUBufferUsage.UNIFORM));

test('one cell encoding: example, reference and WGSL constants agree', () => {
    const flags = { INF: 0xffffffff, OPEN_N: 1, OPEN_E: 2, OPEN_S: 4, OPEN_W: 8, ON_PATH: 16 };
    for (const [name, value] of Object.entries(flags)) {
        assert.equal(maze[name], value, `example ${name}`);
        assert.equal(refs[name], value, `reference ${name}`);
        for (const m of maze.wgslModules()) {
            const literal = name === 'INF' ? '0xffffffffu' : `${value}u`;
            assert.ok(m.code.includes(`const ${name}: u32 = ${literal};`), `${m.name} WGSL ${name}`);
        }
    }
});

test('compute-only MazeSolver: no canvas context, no render pipeline', () => {
    let rendered = 0;
    const device = mockDevice();
    const counting = new Proxy(device, { get: (t, p) => (p === 'createRenderPipeline' ? () => { rendered++; } : t[p]) });
    const solver = new MazeSolver(counting, null);
    solver.setupBuffers(10, 10);
    assert.equal(rendered, 0);
    assert.equal(solver.renderBindGroup, undefined);
    assert.ok(solver.computeBindGroup);
    solver.destroy();
});

test('measureBFS on a mock device: Params per level, walls without the path bit, result shape', async () => {
    const device = mockDevice();
    const w = 63, h = 47;
    const { cells } = generateMaze('kruskal', w, h, 0, 9);
    cells[5] |= refs.ON_PATH;
    const r = await measureBFS({ device }, { cells, width: w, height: h, start: 3, goal: 2000, levelsPerSubmit: 40 });
    const writes = paramWrites(device);
    assert.equal(writes.length, 1, 'the mock reports the goal reached after the first batch');
    assert.equal(writes[0].bytes.byteLength, 40 * 256);
    for (let k = 0; k < 40; k++) {
        const p = decodeStruct(writes[0].bytes, layout, 256 * k);
        assert.deepEqual({ size: p.size, startCell: p.startCell, goalCell: p.goalCell, bfsStep: p.bfsStep, scale: p.scale, solved: p.solved, markerRadius: p.markerRadius },
            { size: [w, h], startCell: 3, goalCell: 2000, bfsStep: k, scale: 1, solved: 0, markerRadius: 0 });
    }
    const walls = new Uint32Array(device.writes.find(x => x.bytes.byteLength === 4 * w * h).bytes.buffer);
    assert.ok(walls.every((c, i) => c === (cells[i] & ~refs.ON_PATH)), 'walls uploaded without ON_PATH');
    const dist = new Uint32Array(device.writes.filter(x => x.bytes.byteLength === 4 * w * h)[1].bytes.buffer);
    assert.ok(dist.every((d, i) => d === (i === 3 ? 0 : refs.INF)), 'distances start at INF except the start');
    assert.deepEqual({ found: r.found, dispatches: r.dispatches, submits: r.submits }, { found: true, dispatches: 40, submits: 1 });
    assert.equal(r.dist.length, w * h);
    assert.equal(r.gpuMs, null);
});

test('measureBFS enforces the caps', async () => {
    const device = mockDevice();
    const cells = new Uint32Array(100);
    await assert.rejects(measureBFS({ device }, { cells, width: 10, height: 0 }), RangeError);
    await assert.rejects(measureBFS({ device }, { cells: new Uint32Array(1024 * 769), width: 1024, height: 769 }), RangeError);
    await assert.rejects(measureBFS({ device }, { cells, width: 10, height: 11 }), RangeError);
    await assert.rejects(measureBFS({ device }, { cells, width: 10, height: 10, goal: 100 }), RangeError);
    await assert.rejects(measureBFS({ device }, { cells, width: 10, height: 10, start: -1 }), RangeError);
    // levelsPerSubmit is clamped to the example's 512 slots
    await measureBFS({ device }, { cells, width: 10, height: 10, levelsPerSubmit: 5000 });
    assert.equal(paramWrites(device).at(-1).bytes.byteLength, 512 * 256);
    assert.ok(MAZE_SIZES.huge.w * MAZE_SIZES.huge.h <= MAX_MEASURE_CELLS);
});

test('verifyBFS: the exact field passes, a wrong or over-expanded field fails', () => {
    const w = 63, h = 47;
    const { cells } = generateMaze('dfs', w, h, 0.05, 4);
    const ref = refs.queueBFS(cells, w, h, 0);
    const D = ref[w * h - 1];
    const gpu = ref.map(d => (d <= D ? d : refs.INF));
    assert.equal(verifyBFS(gpu, ref, D), 0);
    const wrong = Uint32Array.from(gpu);
    wrong[10] = wrong[10] === refs.INF ? 1 : wrong[10] + 2;
    assert.equal(verifyBFS(wrong, ref, D), 1);
    assert.ok(verifyBFS(ref, ref, D) > 0, 'cells beyond the goal must not have been expanded');
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(SIZE_KEYS.every(k => MAZE_SIZES[k].w * MAZE_SIZES[k].h <= MAX_MEASURE_CELLS));
    assert.ok(BRAIDS.every(b => b >= 0 && b <= 0.3) && BATCHES.every(b => b >= 1 && b <= 512) && LEVELS_PER_SUBMIT <= 512);
    // the longest shortest path of any sweep (perfect DFS maze, 1023 × 767) fits the level cap
    const { w, h } = MAZE_SIZES.huge;
    const { cells } = generateMaze('dfs', w, h, 0, 12345);
    assert.ok(refs.queueBFS(cells, w, h, 0)[w * h - 1] < MAX_MEASURE_LEVELS);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [
        ['size', { generator: 'kruskal', braid: 0 }, SIZE_KEYS.length],
        ['size', { generator: 'dfs', braid: 0.2 }, SIZE_KEYS.length],
        ['braid', { generator: 'prim' }, BRAIDS.length],
        ['batch', { sizeKey: 'medium', generator: 'kruskal' }, BATCHES.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) {
            assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys`);
            assert.match(row.verified, /^NO/, 'the mock reads back zeros, which must not verify');
        }
    }
});
