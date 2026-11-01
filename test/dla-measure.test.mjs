// 3D DLA Measure tab, CPU only: the compute-only API of js/examples/dla-3d.js (measureDLA) and the
// experiments in js/teach/experiments/dla.js, driven through the mock device (test/helpers/mock-gpu.mjs),
// which records buffers and uniform writes but runs nothing; plus aggregateStats / connectedFraction of
// js/refs/dla.js on shapes with known dimensions. Checked: Params bytes match the WGSL struct; walkers
// are seeded deterministically and uploaded; the stuck count and early stop work; caps are enforced;
// every row carries the declared columns. The real results are checked on a software GPU in
// test/gpu/dla-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as dla from '../js/examples/dla-3d.js';
import { aggregateStats, connectedFraction } from '../js/refs/dla.js';
import experiments, { STICKINESS, GRIDS, MAX_GROWTH_STEPS, CHECK_EVERY, WALKERS, SCALING_STEPS, stepsTo } from '../js/teach/experiments/dla.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const { measureDLA } = dla;
const layout = structLayout(moduleCode(dla, 'particle'), 'Params');
const paramsWrites = (device) => device.writes
    .filter(w => (w.buffer.usage & GPUBufferUsage.UNIFORM) && w.buffer.size === 16 && w.bytes.byteLength === 16)
    .map(w => decodeStruct(w.bytes, layout));

test('measureDLA on a mock device: Params bytes, deterministic walkers, grid seed, progress, between()', async () => {
    assert.equal(layout.size, 16);
    const device = mockDevice();
    let calls = 0;
    const r = await measureDLA({ device }, { gridSize: 32, walkers: 1000, stickiness: 0.25, rngSeed: 7, steps: 120,
        stepsPerSubmit: 20, warmup: 5, checkEvery: 40, between: async () => { calls++; } });
    const params = paramsWrites(device);
    // The mock reads back zeros: every walker looks stuck at the first check (step 40), so the run stops there
    assert.equal(params.length, 1 + 2, 'one Params write per submit (1 warm-up + 2)');
    for (const p of params) {
        assert.deepEqual([p.numParticles, p.gridSize], [1000, 32]);
        assert.equal(p.stickiness, Math.fround(0.25));
    }
    assert.equal(calls, 2);
    assert.deepEqual(r.progress, [{ step: 40, stuck: 1000 }]);
    assert.equal(r.steps, 40);
    assert.equal(r.grid.length, 32 ** 3);
    assert.equal(r.seedCells, 1);
    // walkers: written twice (mapped at creation, then the seeded ones); the seeded set is reproducible
    const uploads = device.writes.filter(w => w.bytes.byteLength === 16 * 1000);
    assert.equal(uploads.length, 1);
    const again = mockDevice();
    await measureDLA({ device: again }, { gridSize: 32, walkers: 1000, rngSeed: 7, steps: 1 });
    assert.deepEqual(again.writes.find(w => w.bytes.byteLength === 16 * 1000).bytes, uploads[0].bytes);
    const words = new Uint32Array(uploads[0].bytes.buffer);
    const f = new Float32Array(uploads[0].bytes.buffer);
    for (let i = 0; i < 1000; i++) {
        assert.ok(words[4 * i + 3] !== 0, 'live walker');
        const d = Math.hypot(f[4 * i] - 16, f[4 * i + 1] - 16, f[4 * i + 2] - 16);
        assert.ok(Math.abs(d - 32 * 0.45) < 1e-3, 'on the spawn sphere');
    }
    const gridUpload = device.writes.find(w => w.bytes.byteLength === 4 * 32 ** 3);
    const g = new Uint32Array(gridUpload.bytes.buffer);
    assert.equal(g.reduce((s, v) => s + v, 0), 1);
    assert.equal(g[16 * 32 * 32 + 16 * 32 + 16], 1, 'single center seed');
    assert.ok(device.buffers.filter(b => b.usage & GPUBufferUsage.COPY_SRC).length >= 2, 'walkers and grid can be read back');
});

test('measureDLA enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measureDLA({ device }, { gridSize: 8, steps: 1 }), RangeError);
    await assert.rejects(measureDLA({ device }, { gridSize: dla.MAX_MEASURE_GRID + 1, steps: 1 }), RangeError);
    await assert.rejects(measureDLA({ device }, { walkers: dla.MAX_MEASURE_WALKERS + 1, steps: 1 }), RangeError);
    await assert.rejects(measureDLA({ device }, { steps: dla.MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureDLA({ device }, { steps: 10, stepsPerSubmit: 51 }), RangeError);
    await assert.rejects(measureDLA({ device }, { steps: 10, stickiness: 0 }), RangeError);
    await assert.rejects(measureDLA({ device }, { steps: 10, seedType: 'cube' }), RangeError);
});

test('aggregateStats: dimension ≈ 3 for a ball, ≈ 1 for a line, ≈ 2 for a disc; connectivity', (t) => {
    const size = 64, c = 32;
    const make = (inside) => {
        const g = new Uint32Array(size ** 3);
        for (let z = 0; z < size; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
            if (inside(x - c, y - c, z - c)) g[z * size * size + y * size + x] = 1;
        }
        return g;
    };
    const ball = aggregateStats(make((x, y, z) => x * x + y * y + z * z <= 20 * 20), size);
    const line = aggregateStats(make((x, y, z) => y === 0 && z === 0 && Math.abs(x) <= 25), size);
    const disc = aggregateStats(make((x, y, z) => z === 0 && x * x + y * y <= 25 * 25), size);
    t.diagnostic(`ball D ${ball.dimension.toFixed(3)} (R_g ${ball.rg.toFixed(2)}), line D ${line.dimension.toFixed(3)}, disc D ${disc.dimension.toFixed(3)}`);
    assert.ok(Math.abs(ball.dimension - 3) < 0.15 && Math.abs(line.dimension - 1) < 0.1 && Math.abs(disc.dimension - 2) < 0.15);
    assert.ok(Math.abs(ball.rg - 20 * Math.sqrt(3 / 5)) < 0.3, 'R_g of a ball of radius R is R·√(3/5)');
    assert.equal(ball.connected, 1);
    // two blobs: only the one containing the seed is connected
    const two = make((x, y, z) => x * x + y * y + z * z <= 9 || (x - 15) ** 2 + y * y + z * z <= 9);
    assert.ok(Math.abs(connectedFraction(two, size, [c, c, c]) - 0.5) < 1e-12);
    assert.equal(aggregateStats(new Uint32Array(size ** 3), size).cells, 0);
    assert.equal(stepsTo([{ step: 100, stuck: 10 }, { step: 200, stuck: 60 }, { step: 300, stuck: 99 }], 100, 0.5), 200);
    assert.equal(stepsTo([{ step: 100, stuck: 10 }], 100, 0.5), null);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(STICKINESS.every(s => s > 0 && s <= 1));
    assert.ok(Object.entries(GRIDS).every(([g, n]) => +g <= dla.MAX_MEASURE_GRID && n <= 5000));
    assert.ok(MAX_GROWTH_STEPS <= dla.MAX_MEASURE_STEPS && MAX_GROWTH_STEPS % CHECK_EVERY === 0 && CHECK_EVERY % dla.MAX_STEPS_PER_SUBMIT === 0);
    assert.ok(WALKERS.every(n => n <= dla.MAX_MEASURE_WALKERS) && SCALING_STEPS <= 250);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [['stickiness', { grid: 32 }, STICKINESS.length], ['scaling', { maxWalkers: 65536 }, 4]];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys = columns`);
        const p = paramsWrites(device);
        assert.ok(p.every(q => q.numParticles <= dla.MAX_MEASURE_WALKERS && q.gridSize <= 64));
    }
});
