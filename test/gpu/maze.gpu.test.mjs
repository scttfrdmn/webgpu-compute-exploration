// Maze BFS (js/examples/maze.js) on a software GPU: the shipped cs_bfs_step WGSL, one dispatch per
// BFS level in batches like MazeSolver.frame(), with the termination rule of its stats readback
// (stop when dist[goal] is set, or when the last dispatched level reached no new cell). The walls,
// initial distances and Params slots are the bytes the shipped MazeSolver (generate /
// clearSolution / writeParams) writes, captured through a mock device. CI-only — see
// test/gpu/harness.mjs.
//
// The example binds one Params slot per dispatch at a dynamic uniform offset; here each dispatch gets
// a bind group whose (static) buffer binding starts at that same slot offset.
//
// Compared with: an independent queue BFS (test/helpers/maze-refs.mjs) — distances must match exactly.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, write, read, run } from './harness.mjs';
import * as maze from '../../js/examples/maze.js';
import { moduleCode, structLayout } from '../helpers/wgsl-layout.mjs';
import { mockDevice } from '../helpers/mock-gpu.mjs';
import { INF, OPEN_E, OPEN_S, openNeighbors, queueBFS, fakeCanvas } from '../helpers/maze-refs.mjs';

const { MazeSolver, extractPath, MAZE_SIZES } = maze;
const code = moduleCode(maze, 'compute');
const PARAMS = structLayout(code, 'Params');
const bfs = await computePipeline(code, 'cs_bfs_step');
const BATCH = 512;   // MAX_STEPS_PER_FRAME in maze.js: Params slots (and dispatches) per frame

function makeSolver(sizeKey, algorithm, braid, seed) {
    const device = mockDevice();
    const solver = new MazeSolver(device, fakeCanvas());
    Object.assign(solver, { sizeKey, algorithm, braid, seed });
    solver.generate(false);
    return { device, solver };
}

// Floods on the GPU from solver.startCell until the host's termination rule fires. Returns
// { dist, stats, dispatched }.
async function gpuSolve(device, solver) {
    solver.clearSolution();   // writes walls, dist (INF, start = 0) and zeroed stats
    const { width: w, height: h } = solver;
    const walls = buffer(device.lastWrite(solver.wallBuffer).bytes);
    const dist = buffer(device.lastWrite(solver.distBuffer).bytes);
    const stats = buffer(device.lastWrite(solver.statsBuffer).bytes);
    const params = buffer(solver.paramsStride * BATCH);
    const wg = [Math.ceil(w / 8), Math.ceil(h / 8)];
    const groups = Array.from({ length: BATCH }, (_, i) => bindGroup(bfs, 0, {
        walls, dist, stats, params: { buffer: params, offset: i * solver.paramsStride, size: PARAMS.size },
    }));
    solver.dispatched = 0;
    for (;;) {
        solver.writeParams(BATCH);   // bfsStep = dispatched + k in slot k
        write(params, device.lastWrite(solver.paramsBuffer).bytes);
        await run(groups.map(g => ({ p: bfs, groups: [g], workgroups: wg })), `BFS levels ${solver.dispatched}+`);
        solver.dispatched += BATCH;
        const s = await read(stats, Uint32Array, 8);
        const goal = (await read(dist, Uint32Array, 4, solver.goalCell * 4))[0];
        if (goal !== INF || s[1] < solver.dispatched) break;
        assert.ok(solver.dispatched <= w * h + BATCH, 'BFS did not terminate');
    }
    return {
        dist: await read(dist, Uint32Array, w * h * 4),
        stats: await read(stats, Uint32Array, 8),
        dispatched: solver.dispatched,
    };
}

test('cs_bfs_step on the GPU matches queue BFS exactly (3 generators, perfect + braided)', async (t) => {
    const cases = [];
    for (const algorithm of Object.keys(maze.GENERATORS)) for (const braid of [0, 0.15]) cases.push(['small', algorithm, braid, 2024]);
    cases.push(['medium', 'kruskal', 0.3, 7]);   // 255x191, many loops
    let cells = 0, maxLevel = 0;
    for (const [sizeKey, algorithm, braid, seed] of cases) {
        const { device, solver } = makeSolver(sizeKey, algorithm, braid, seed);
        const { w, h } = MAZE_SIZES[sizeKey];
        assert.equal(solver.width, w);
        const start = solver.startCell, corner = w * h - 1;
        assert.equal(start, 0);
        assert.equal(solver.goalCell, corner);
        const ref = queueBFS(solver.cells, w, h, start);
        let far = 0;
        for (let i = 0; i < w * h; i++) if (ref[i] > ref[far]) far = i;

        // Full flood: goal = the farthest cell, so every cell gets its distance
        solver.goalCell = far;
        const full = await gpuSolve(device, solver);
        let mismatches = 0;
        for (let i = 0; i < w * h; i++) if (full.dist[i] !== ref[i]) mismatches++;
        assert.equal(mismatches, 0, `${sizeKey} ${algorithm} braid ${braid}: ${mismatches} cells differ`);
        assert.equal(full.stats[0], w * h - 1, 'stats[0] counts reached cells (start excluded)');
        assert.equal(full.stats[1], ref[far], 'stats[1] is the largest distance');
        cells += w * h;
        maxLevel = Math.max(maxLevel, ref[far]);

        // Corner to corner: early stop at the goal level; every assigned distance is exact, nothing
        // closer than the goal is missed, and the shipped extractPath yields a shortest path
        solver.goalCell = corner;
        const c2c = await gpuSolve(device, solver);
        assert.equal(c2c.dist[corner], ref[corner]);
        let assigned = 0;
        for (let i = 0; i < w * h; i++) {
            if (c2c.dist[i] !== INF) { assert.equal(c2c.dist[i], ref[i], `cell ${i}`); assigned++; }
            else assert.ok(ref[i] >= ref[corner], 'a cell closer than the goal was not reached');
        }
        const path = extractPath(solver.cells, c2c.dist, w, h, start, corner);
        assert.equal(path.length, ref[corner] + 1, 'path length = shortest distance + 1');
        for (let k = 0; k + 1 < path.length; k++) {
            assert.ok(openNeighbors(solver.cells, w, h, path[k]).includes(path[k + 1]), 'path step through a wall');
        }
        t.diagnostic(`${w}x${h} ${algorithm} braid ${braid}: full flood ${w * h} cells, 0 mismatches, max distance ${ref[far]} ` +
            `(${full.dispatched} dispatches); corner-to-corner distance ${ref[corner]}, ${assigned} cells assigned`);
    }
    t.diagnostic(`GPU BFS == queue BFS on ${cells} cells; deepest level ${maxLevel}`);
});

test('unreachable goal: the flood stops when a level adds no cells', async (t) => {
    const { device, solver } = makeSolver('small', 'dfs', 0, 11);
    const { w, h } = MAZE_SIZES.small;
    // Wall off the bottom-right corner cell: clear its own bits and the matching bit of its west (E)
    // and north (S) neighbours, so both sides of each passage agree
    const g = w * h - 1;
    solver.cells[g - 1] &= ~OPEN_E;
    solver.cells[g - w] &= ~OPEN_S;
    solver.cells[g] &= ~15;
    solver.goalCell = g;
    const ref = queueBFS(solver.cells, w, h, 0);
    assert.equal(ref[g], INF);
    const res = await gpuSolve(device, solver);
    assert.equal(res.dist[g], INF);
    for (let i = 0; i < w * h; i++) assert.equal(res.dist[i], ref[i]);
    const reachable = ref.filter(d => d !== INF).length;
    assert.equal(res.stats[0], reachable - 1, 'stats[0] = reachable cells, start excluded');
    t.diagnostic(`unreachable goal: ${res.stats[0]} cells reached, stopped after ${res.dispatched} dispatches`);
});
