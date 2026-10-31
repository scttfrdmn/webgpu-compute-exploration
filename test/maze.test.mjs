// Maze (example: js/examples/maze.js).
//
// Real code tested directly: generateMaze (all GENERATORS), braidMaze, extractPath, and
// MazeSolver.writeParams (through a mock device; dynamic-offset slots decoded with the WGSL layout).
//
// The BFS part is a CPU replica of maze.js computeShader (cs_bfs_step); validates the algorithm and
// index math, not the WGSL itself — see #51. It is compared against a plain queue BFS.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import * as maze from '../js/examples/maze.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const { generateMaze, braidMaze, extractPath, GENERATORS, MazeSolver } = maze;
import { OPEN_N, OPEN_E, OPEN_S, OPEN_W, INF, openNeighbors, queueBFS, fakeCanvas } from './helpers/maze-refs.mjs';

// openNeighbors / queueBFS (independent queue BFS reference) / fakeCanvas live in
// test/helpers/maze-refs.mjs, shared with test/gpu/maze.gpu.test.mjs.

// Structural checks shared by perfect and braided mazes. Returns the number of undirected passages.
function checkWalls(cells, w, h) {
    let edges = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const c = y * w + x, v = cells[c];
            assert.equal(v & ~15, 0, `cell ${c} has bits outside 0-3`);
            // Outer border stays closed
            if (y === 0) assert.equal(v & OPEN_N, 0, `north border open at ${c}`);
            if (y === h - 1) assert.equal(v & OPEN_S, 0, `south border open at ${c}`);
            if (x === 0) assert.equal(v & OPEN_W, 0, `west border open at ${c}`);
            if (x === w - 1) assert.equal(v & OPEN_E, 0, `east border open at ${c}`);
            // Both sides of a passage carry its bit
            if (x + 1 < w) {
                assert.equal(!!(v & OPEN_E), !!(cells[c + 1] & OPEN_W), `E/W asymmetry at ${c}`);
                if (v & OPEN_E) edges++;
            }
            if (y + 1 < h) {
                assert.equal(!!(v & OPEN_S), !!(cells[c + w] & OPEN_N), `S/N asymmetry at ${c}`);
                if (v & OPEN_S) edges++;
            }
        }
    }
    return edges;
}

// CPU replica of cs_bfs_step: one call = one dispatch for BFS level `level` (pull formulation).
// The GPU reads neighbours that may be written in the same dispatch; the shader argues only L+1 is
// ever written, never L, so reading the in-place array (as here, sequentially) gives the same result.
function bfsStepReplica(walls, dist, stats, w, h, goalCell, level) {
    if (dist[goalCell] <= level) return;          // per-invocation early-out; same for every cell
    for (let gy = 0; gy < h; gy++) {
        for (let gx = 0; gx < w; gx++) {
            const idx = gy * w + gx;
            if (dist[idx] !== INF) continue;
            const open = walls[idx];
            let reached = false;
            if ((open & OPEN_N) && gy > 0 && dist[idx - w] === level) reached = true;
            if ((open & OPEN_E) && gx + 1 < w && dist[idx + 1] === level) reached = true;
            if ((open & OPEN_S) && gy + 1 < h && dist[idx + w] === level) reached = true;
            if ((open & OPEN_W) && gx > 0 && dist[idx - 1] === level) reached = true;
            if (reached) {
                dist[idx] = level + 1;
                stats[0]++;
                stats[1] = Math.max(stats[1], level + 1);
            }
        }
    }
}

// Runs replica dispatches until the goal is reached or a level adds no cells (what the host's
// termination readback decides). Returns { dist, stats, levels }.
function solveReplica(cells, w, h, start, goal) {
    const dist = new Uint32Array(w * h).fill(INF);
    dist[start] = 0;
    const stats = [0, 0];
    let level = 0;
    for (;;) {
        const before = stats[0];
        bfsStepReplica(cells, dist, stats, w, h, goal, level);
        level++;
        if (dist[goal] !== INF && dist[goal] <= level) break;
        if (stats[0] === before) break;
    }
    return { dist, stats, levels: level };
}

// Width 1 is excluded: carve() tests b === a + 1 before b === a + w, so with w = 1 a vertical
// passage is stored as E/W bits (see the todo test at the end). Every MAZE_SIZES entry has w >= 63.
const SIZES = [[1, 1], [9, 1], [2, 2], [2, 9], [17, 11], [63, 47]];

test('every generator builds a perfect maze (spanning tree) for several sizes and seeds', (t) => {
    let checked = 0;
    for (const algorithm of Object.keys(GENERATORS)) {
        let ms = 0;
        for (const [w, h] of SIZES) {
            for (const seed of [1, 0xdeadbeef, 424242]) {
                const t0 = performance.now();
                const { cells, removed } = generateMaze(algorithm, w, h, 0, seed);
                ms += performance.now() - t0;
                assert.equal(removed, 0);
                const edges = checkWalls(cells, w, h);
                // A spanning tree of n cells has exactly n-1 edges and reaches every cell
                assert.equal(edges, w * h - 1, `${algorithm} ${w}x${h}: edges`);
                const dist = queueBFS(cells, w, h, 0);
                assert.ok(dist.every(d => d !== INF), `${algorithm} ${w}x${h}: not connected`);
                checked++;
            }
        }
        t.diagnostic(`${algorithm}: ${SIZES.length * 3} mazes are spanning trees (edges = cells - 1, connected); ${ms.toFixed(1)} ms total`);
    }
    assert.equal(checked, 3 * SIZES.length * 3);
});

test('generation is deterministic per seed and differs between seeds', () => {
    for (const algorithm of Object.keys(GENERATORS)) {
        const a = generateMaze(algorithm, 31, 23, 0.2, 7).cells;
        const b = generateMaze(algorithm, 31, 23, 0.2, 7).cells;
        const c = generateMaze(algorithm, 31, 23, 0.2, 8).cells;
        assert.deepEqual(a, b, `${algorithm}: same seed, different maze`);
        assert.notDeepEqual(a, c, `${algorithm}: different seeds, same maze`);
    }
});

test('braidMaze adds loops, removes dead ends, keeps the walls consistent', (t) => {
    const w = 63, h = 47;
    const deadEnds = cells => cells.reduce((n, v) => n + ([1, 2, 4, 8].includes(v & 15) ? 1 : 0), 0);
    for (const algorithm of Object.keys(GENERATORS)) {
        const perfect = generateMaze(algorithm, w, h, 0, 99).cells;
        const { cells, removed } = generateMaze(algorithm, w, h, 0.25, 99);
        const edges = checkWalls(cells, w, h);
        // braidMaze only adds passages to the same perfect maze: the count it reports is exact
        assert.equal(edges, w * h - 1 + removed);
        assert.ok(removed > 0);
        for (let i = 0; i < w * h; i++) assert.equal(perfect[i] & cells[i], perfect[i], 'braid closed a passage');
        assert.ok(queueBFS(cells, w, h, 0).every(d => d !== INF));
        const before = deadEnds(perfect), after = deadEnds(cells);
        t.diagnostic(`${algorithm}: braid 0.25 removed ${removed} walls; dead ends ${before} -> ${after}`);
        assert.ok(after < before * 0.75, 'braiding should remove a good share of dead ends');
    }
    // braid = 0 is a no-op; braid = 1 opens every interior wall
    const c0 = generateMaze('dfs', 9, 7, 0, 3).cells;
    assert.equal(braidMaze(c0.slice(), 9, 7, 0, 3), 0);
    const all = generateMaze('dfs', 9, 7, 1, 3);
    assert.equal(checkWalls(all.cells, 9, 7), 8 * 7 + 9 * 6);
});

test('BFS replica (cs_bfs_step) matches queue BFS distances; extractPath gives a shortest path', (t) => {
    let compared = 0, maxLevels = 0;
    for (const algorithm of Object.keys(GENERATORS)) {
        for (const braid of [0, 0.15]) {
            const w = 63, h = 47;
            const { cells } = generateMaze(algorithm, w, h, braid, 2024);
            const start = 0, goal = w * h - 1;
            const ref = queueBFS(cells, w, h, start);

            // Full flood: goal = the farthest cell, so every cell gets its distance
            let far = 0;
            for (let i = 0; i < w * h; i++) if (ref[i] > ref[far]) far = i;
            const full = solveReplica(cells, w, h, start, far);
            assert.deepEqual(full.dist, ref, `${algorithm} braid ${braid}: full flood differs`);
            assert.equal(full.stats[0], w * h - 1, 'stats[0] counts reached cells (start excluded)');
            assert.equal(full.stats[1], ref[far], 'stats[1] is the largest distance');
            compared += w * h;

            // Corner-to-corner: early stop at the goal level; every assigned distance is exact and
            // nothing beyond the goal's level is assigned
            const run = solveReplica(cells, w, h, start, goal);
            maxLevels = Math.max(maxLevels, run.levels);
            assert.equal(run.dist[goal], ref[goal]);
            for (let i = 0; i < w * h; i++) {
                if (run.dist[i] !== INF) assert.equal(run.dist[i], ref[i]);
                else assert.ok(ref[i] >= ref[goal], 'a cell closer than the goal was not reached');
            }

            // Shipped extractPath on the replica's distance field
            const path = extractPath(cells, run.dist, w, h, start, goal);
            assert.equal(path.length, ref[goal] + 1, 'path length = shortest distance + 1');
            assert.equal(path[0], goal);
            assert.equal(path[path.length - 1], start);
            for (let k = 0; k + 1 < path.length; k++) {
                assert.ok(openNeighbors(cells, w, h, path[k]).includes(path[k + 1]), 'path step through a wall');
            }
            assert.equal(new Set(path).size, path.length, 'path revisits a cell');
        }
    }
    t.diagnostic(`replica BFS == queue BFS on ${compared} cells (6 mazes, 63x47, perfect + braided); up to ${maxLevels} levels`);
});

test('extractPath rejects a field that is not a BFS distance field', () => {
    const w = 5, h = 5;
    const { cells } = generateMaze('kruskal', w, h, 0, 5);
    const dist = queueBFS(cells, w, h, 0);
    assert.deepEqual(extractPath(cells, new Uint32Array(w * h).fill(INF), w, h, 0, 24), []);
    const broken = dist.slice();
    broken[24] += 3;   // no neighbour at distance d-1 any more
    assert.deepEqual(extractPath(cells, broken, w, h, 0, 24), []);
    assert.deepEqual(extractPath(cells, dist, w, h, 0, 0), [0]);
});

test('MazeSolver.writeParams matches struct Params, one copy per dynamic-offset slot', (t) => {
    const device = mockDevice();
    const solver = new MazeSolver(device, fakeCanvas());
    solver.sizeKey = 'small';
    solver.seed = 77;
    solver.generate(false);
    const { w, h, scale } = maze.MAZE_SIZES.small;
    solver.startCell = 17;
    solver.goalCell = w * h - 5;
    solver.dispatched = 1000;
    solver.phase = 'solved';

    const layout = structLayout(moduleCode(maze, 'compute'), 'Params');
    const renderLayout = structLayout(moduleCode(maze, 'render'), 'Params');
    assert.deepEqual(renderLayout.offsets, layout.offsets);
    assert.equal(layout.size, 32);
    assert.equal(solver.paramsStride % device.limits.minUniformBufferOffsetAlignment, 0);
    assert.ok(solver.paramsStride >= layout.size);

    const slots = 5;
    solver.writeParams(slots);
    const wr = device.lastWrite(solver.paramsBuffer);
    assert.equal(wr.offset, 0);
    assert.equal(wr.bytes.byteLength, slots * solver.paramsStride);
    const markerRadius = Math.max(scale * 1.2, solver.canvas.width / 70);
    for (let k = 0; k < slots; k++) {
        const p = decodeStruct(wr.bytes, layout, k * solver.paramsStride);
        assert.deepEqual(p.size, [w, h]);
        assert.equal(p.startCell, 17);
        assert.equal(p.goalCell, w * h - 5);
        assert.equal(p.bfsStep, 1000 + k, `slot ${k} bfsStep`);
        assert.equal(p.scale, scale);
        assert.equal(p.solved, 1);
        assert.equal(p.markerRadius, Math.fround(markerRadius));
    }
    t.diagnostic(`Params ${layout.size} B, stride ${solver.paramsStride} B; offsets ${JSON.stringify(layout.offsets)}`);
});

test('MAZE_SIZES never uses width 1 (where carve() is ambiguous)', () => {
    for (const { w, h } of Object.values(maze.MAZE_SIZES)) assert.ok(w >= 2 && h >= 1);
});

test('width-1 mazes encode vertical passages as N/S', {
    todo: 'carve() checks b === a + 1 before b === a + w, so w = 1 mazes get E/W bits; unreachable from the UI (MAZE_SIZES)',
}, () => {
    const { cells } = generateMaze('dfs', 1, 9, 0, 1);
    checkWalls(cells, 1, 9);
});
