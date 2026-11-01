// Game of Life Measure tab, CPU only: compute-only GameOfLife (canvas = null) and measureLife() in
// js/examples/game-of-life.js, and the experiments in js/teach/experiments/game-of-life.js, on the mock
// device. Checked: no render resources without a canvas, grid buffers readable (COPY_SRC, which
// toggleCell's copy also needs), the dimensions uniform, the grid upload, the caps, rows carry the declared
// columns. Real generations: test/gpu/game-of-life-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GameOfLife, measureLife, MAX_LIFE_SIDE, MAX_LIFE_GENERATIONS, MAX_LIFE_WORK } from '../js/examples/game-of-life.js';
import experiments, { SIDES, BATCH, SOUP_SIDES, CHECKPOINTS, randomGrid, population, mismatches } from '../js/teach/experiments/game-of-life.js';
import { lifeStep } from '../js/refs/life.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });

test('GameOfLife without a canvas: compute only, readable grid buffers, render() is a no-op', () => {
    const device = mockDevice();
    const game = new GameOfLife(device, null, 64, 48);
    assert.equal(game.context, undefined);
    assert.equal(game.renderPipeline, undefined);
    for (const b of game.gridBuffers) assert.ok(b.usage & GPUBufferUsage.COPY_SRC, 'COPY_SRC for read-back and toggleCell');
    game.render();
    game.setGrid(new Uint32Array(64 * 48));
    assert.ok(game.computeBindGroups.length === 2);
});

test('measureLife on a mock device: dimensions, upload, result shape, caps', async () => {
    const device = mockDevice();
    const [w, h] = [50, 30];
    const grid = randomGrid(w, h);
    const r = await measureLife({ device }, { grid, width: w, height: h, generations: 7, repeats: 2 });
    const dims = device.buffers.find(b => b.usage & GPUBufferUsage.UNIFORM);
    assert.deepEqual([...new Uint32Array(dims.getMappedRange())], [w, h], 'vec2<u32> width, height');
    const uploads = device.writes.filter(x => x.bytes.byteLength === 4 * w * h);
    assert.equal(uploads.length, 2, 'one upload per batch');
    assert.deepEqual(new Uint32Array(uploads[0].bytes.buffer), grid);
    assert.deepEqual({ generations: r.generations, gpuMs: r.gpuMs, cells: r.grid.length }, { generations: 7, gpuMs: null, cells: w * h });
    await assert.rejects(measureLife({ device }, { grid, width: w, height: h, generations: MAX_LIFE_GENERATIONS + 1 }), RangeError);
    await assert.rejects(measureLife({ device }, { grid, width: w, height: h, generations: 0 }), RangeError);
    await assert.rejects(measureLife({ device }, { grid: { length: (MAX_LIFE_SIDE + 1) * 4 }, width: MAX_LIFE_SIDE + 1, height: 4 }), RangeError);
    await assert.rejects(measureLife({ device }, { grid, width: w + 1, height: h }), RangeError);
    const big = { length: 2048 * 2048 };
    await assert.rejects(measureLife({ device }, { grid: big, width: 2048, height: 2048, generations: 101 }), RangeError, 'work per pass');
    assert.equal(MAX_LIFE_WORK, 2048 * 2048 * 100);
});

test('helpers: random soup density, population, mismatches against lifeStep', () => {
    const g = randomGrid(200, 150);
    const p = population(g) / g.length;
    assert.ok(Math.abs(p - 0.3) < 0.01, `density ${p}`);
    const next = lifeStep(g, 200, 150);
    assert.equal(mismatches(next, lifeStep(g, 200, 150)), 0);
    assert.ok(mismatches(next, g) > 0);
});

test('experiments on a mock device: rows, declared columns, caps', async () => {
    assert.ok(SIDES.every(s => s <= MAX_LIFE_SIDE && s * s * BATCH <= MAX_LIFE_WORK) && BATCH <= MAX_LIFE_GENERATIONS);
    assert.ok(SOUP_SIDES.every(s => s <= 256));
    for (let i = 1; i < CHECKPOINTS.length; i++) assert.ok((CHECKPOINTS[i] - CHECKPOINTS[i - 1]) * 256 * 256 <= MAX_LIFE_WORK);
    for (const [id, settings, expected] of [['throughput', {}, SIDES.length], ['soup', { side: 128 }, CHECKPOINTS.length]]) {
        const e = experiments.find(x => x.id === id);
        const device = mockDevice();
        const rows = await runExperiment(e, settings, mockCtx(device, e.budgetMs));
        assert.equal(rows.length, expected, id);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort(), id);
        assert.ok(Math.max(...device.buffers.map(b => b.size)) <= 4 * MAX_LIFE_SIDE * MAX_LIFE_SIDE);
    }
});
