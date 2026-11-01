// Maze compute-only API (measureBFS in js/examples/maze.js) and the Measure experiments built on it
// (js/teach/experiments/maze.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/maze.gpu.test.mjs, which rebuilds the dispatch sequence with explicit layouts, this
// runs the example's own pipeline, dynamic-offset bind group, writeParams and termination readback
// through a MazeSolver without a canvas, exactly as the Measure tab does in a browser. Checked against
// the queue BFS of js/refs/maze.js: identical distances up to the goal, INF beyond, a shortest path of
// dist[goal] + 1 cells, and the unreachable case.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureBFS, generateMaze, extractPath, MAZE_SIZES, OPEN_N, OPEN_E, OPEN_S, OPEN_W } from '../../js/examples/maze.js';
import { queueBFS, INF } from '../../js/refs/maze.js';
import experiments, { verifyBFS } from '../../js/teach/experiments/maze.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureBFS: exact BFS distances to the goal for 3 generators, perfect and braided, any batch size', async (t) => {
    const { w, h } = MAZE_SIZES.medium;
    for (const generator of ['dfs', 'kruskal', 'prim']) {
        for (const braid of [0, 0.1]) {
            const { cells } = generateMaze(generator, w, h, braid, 77);
            const start = 1000, goal = w * h - 5;
            const ref = queueBFS(cells, w, h, start);
            for (const levelsPerSubmit of [7, 256]) {
                const r = await measureBFS(gpuDevice, { cells, width: w, height: h, start, goal, levelsPerSubmit });
                const mismatches = verifyBFS(r.dist, ref, r.levels);
                const path = extractPath(cells, r.dist, w, h, start, goal);
                t.diagnostic(`${generator} braid ${braid} batch ${levelsPerSubmit}: levels ${r.levels} (queue ${ref[goal]}), visited ${r.visited}, `
                    + `dispatches ${r.dispatches}, submits ${r.submits}, mismatches ${mismatches}, wall ${r.wallMs.toFixed(1)} ms`);
                assert.ok(r.found);
                assert.equal(r.levels, ref[goal]);
                assert.equal(mismatches, 0);
                assert.equal(path.length, r.levels + 1);
                assert.equal(r.visited, r.dist.filter(d => d !== INF).length);
                assert.ok(r.dispatches >= r.levels && r.dispatches < r.levels + levelsPerSubmit);
                assert.equal(r.submits, Math.ceil(r.dispatches / levelsPerSubmit));
            }
        }
    }
});

test('measureBFS: an unreachable goal stops when a level adds no cell', async (t) => {
    // a 40 × 30 grid with every passage open except a wall column at x = 20
    const w = 40, h = 30;
    const cells = new Uint32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let c = 0;
        if (y > 0) c |= OPEN_N;
        if (y < h - 1) c |= OPEN_S;
        if (x > 0 && x !== 20) c |= OPEN_W;
        if (x < w - 1 && x !== 19) c |= OPEN_E;
        cells[y * w + x] = c;
    }
    const r = await measureBFS(gpuDevice, { cells, width: w, height: h, levelsPerSubmit: 16 });
    const ref = queueBFS(cells, w, h, 0);
    t.diagnostic(`unreachable: found ${r.found}, visited ${r.visited}, dispatches ${r.dispatches}`);
    assert.equal(r.found, false);
    assert.equal(r.visited, 20 * h);
    assert.deepEqual([...r.dist], [...ref]);
});

const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(makeTestContext(gpuDevice), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toPrecision(3)) : String(v);
}).join(' ')).join('; ');

test('experiment size (Kruskal, perfect and 20% loops): every row verified', async (t) => {
    const e = byId.size;
    for (const braid of [0, 0.2]) {
        const { rows, seconds } = await runAll(e, { generator: 'kruskal', braid });
        t.diagnostic(`kruskal braid ${braid} ${seconds.toFixed(1)} s: cells levels dispatches wallMs cpuMs msPerLevel verified: `
            + show(rows, ['cells', 'levels', 'dispatches', 'wallMs', 'cpuMs', 'msPerLevel', 'verified']));
        assert.equal(rows.length, 4);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.match(row.verified, /^yes/, `${row.cells} cells`);
        }
        assert.ok(rows[3].levels > rows[0].levels, 'bigger mazes need more levels');
    }
});

test('experiment braid (Kruskal): loops shorten the path; experiment batch: fewer submits, same answer', async (t) => {
    const b = await runAll(byId.braid, { generator: 'kruskal' });
    t.diagnostic(`braid ${b.seconds.toFixed(1)} s: braid levels visited wallMs: ${show(b.rows, ['braid', 'levels', 'visited', 'wallMs', 'verified'])}`);
    assert.equal(b.rows.length, 7);
    for (const row of b.rows) assert.match(row.verified, /^yes/);
    assert.ok(b.rows.at(-1).levels < b.rows[0].levels, 'loops shorten the shortest path');

    const s = await runAll(byId.batch, { sizeKey: 'medium', generator: 'kruskal' });
    t.diagnostic(`batch ${s.seconds.toFixed(1)} s: batch submits overshoot wallMs gpuMs: ${show(s.rows, ['batch', 'submits', 'overshoot', 'wallMs', 'gpuMs'])}`);
    assert.deepEqual(s.rows.map(r => r.batch), [1, 4, 16, 64, 256, 512]);
    for (const row of s.rows) {
        assert.match(row.verified, /^yes/);
        assert.equal(row.levels, s.rows[0].levels);
        assert.ok(row.overshoot >= 0 && row.overshoot < row.batch);
    }
    assert.equal(s.rows[0].submits, s.rows[0].levels, 'one submit per level at batch 1');
});
