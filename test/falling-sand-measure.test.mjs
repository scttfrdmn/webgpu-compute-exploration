// Falling-sand Measure tab, CPU only: the compute-only API of js/examples/falling-sand.js (measureSand)
// and the experiments in js/teach/experiments/falling-sand.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing; plus the
// material counts of js/refs/falling-sand.js. Checked: Params bytes in every dynamic-offset slot match
// the WGSL struct (brush off, simulate on, consecutive step counters); the grid is uploaded unchanged;
// caps are enforced; the scenes are what they claim; every row carries the declared columns. The real
// results are checked on a software GPU in test/gpu/falling-sand-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as sand from '../js/examples/falling-sand.js';
import { materialCounts, changedCells, MATERIAL_NAMES } from '../js/refs/falling-sand.js';
import experiments, { SIZES, MAX_CELLS, stepsFor, CHECKPOINTS, tiledSandbox, scene, SANDBOX_W, SANDBOX_H } from '../js/teach/experiments/falling-sand.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const { measureSand, MATERIAL_IDS: M, MAX_STEPS } = sand;
const layout = structLayout(moduleCode(sand, 'compute'), 'Params');
const STRIDE = 256;
const paramSlots = (device) => device.writes
    .filter(w => (w.buffer.usage & GPUBufferUsage.UNIFORM) && w.buffer.size === STRIDE * MAX_STEPS)
    .flatMap(w => Array.from({ length: w.bytes.byteLength / STRIDE }, (_, k) => decodeStruct(w.bytes, layout, k * STRIDE)));

test('measureSand on a mock device: Params slots, grid upload, samples, batching, between()', async () => {
    const device = mockDevice();
    const w = 64, h = 48, grid = tiledSandbox(w, h);
    let calls = 0;
    const r = await measureSand({ device }, { width: w, height: h, grid, steps: 20, warmup: 2, stepsPerSubmit: 8,
        readAt: [5, 20], between: async () => { calls++; } });
    // warm-up 2, then 5 | 8 + 7 → 3 submits
    assert.equal(calls, 3);
    const slots = paramSlots(device);
    assert.equal(slots.length, 2 + 20);
    slots.forEach((p, k) => {
        assert.deepEqual(p.size, [w, h]);
        assert.equal(p.brushOn, 0);
        assert.equal(p.simulate, 1);
        assert.equal(p.step, k, 'step counters run on across submits');
    });
    assert.deepEqual(r.samples.map(s => s.step), [5, 20]);
    const upload = device.writes.find(wr => wr.bytes.byteLength === 4 * w * h);
    assert.deepEqual(new Uint32Array(upload.bytes.buffer), grid);
    assert.ok(device.buffers.some(b => b.size === 4 * w * h && (b.usage & GPUBufferUsage.STORAGE) && (b.usage & GPUBufferUsage.COPY_SRC)));
    assert.equal(r.gpuMs, null);

    const single = mockDevice();
    await measureSand({ device: single }, { width: w, height: h, grid, steps: 6, stepsPerSubmit: 1 });
    assert.equal(single.writes.filter(wr => wr.buffer.usage & GPUBufferUsage.UNIFORM).length, 6, 'one submit per step');
});

test('measureSand enforces the caps', async () => {
    const device = mockDevice();
    const grid = (w, h) => new Uint32Array(w * h);
    await assert.rejects(measureSand({ device }, { width: 63, height: 48, grid: grid(63, 48), steps: 1 }), RangeError);
    await assert.rejects(measureSand({ device }, { width: 64, height: 48, grid: grid(10, 10), steps: 1 }), RangeError);
    await assert.rejects(measureSand({ device }, { width: 2562, height: 1920, grid: grid(2562, 1920), steps: 1 }), RangeError);
    await assert.rejects(measureSand({ device }, { width: 64, height: 48, grid: grid(64, 48), steps: sand.MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureSand({ device }, { width: 64, height: 48, grid: grid(64, 48), steps: 1, stepsPerSubmit: 9 }), RangeError);
    await assert.rejects(measureSand({ device }, { width: 64, height: 48, grid: grid(64, 48), steps: 1, warmup: 9 }), RangeError);
});

test('material counts and the scenes', (t) => {
    const box = scene('sandbox');
    assert.deepEqual(box, sand.buildSandbox(SANDBOX_W, SANDBOX_H));
    const c = materialCounts(box);
    t.diagnostic(MATERIAL_NAMES.map((n, i) => `${n} ${c[i]}`).join(', '));
    assert.equal(c.reduce((s, x) => s + x, 0), SANDBOX_W * SANDBOX_H);
    assert.ok(c[M.SAND] > 0 && c[M.WATER] > 0 && c[M.OIL] > 0 && c[M.PLANT] > 0 && c[M.FIRE] === 0);
    const inert = materialCounts(scene('inert'));
    assert.equal(inert[M.PLANT], 0);
    assert.equal(inert[M.STONE], c[M.STONE] + c[M.PLANT]);
    const fire = scene('fire');
    const cf = materialCounts(fire);
    assert.equal(cf[M.FIRE], 4 * 118);
    assert.ok(fire.every(v => (v & 15) !== M.FIRE || ((v >>> 8) & 255) === 120), 'fire cells have a life of 120 steps');
    assert.equal(changedCells(box, fire), 4 * 118);
    assert.equal(changedCells(box, box), 0);
    const plants = materialCounts(scene('plants'));
    assert.equal(plants[M.PLANT], c[M.PLANT] + 200);
    assert.equal(plants[M.WATER], c[M.WATER] - 200, 'the patch replaces water');
    // the tiled grid repeats the sandbox
    const big = tiledSandbox(640, 480);
    assert.deepEqual(materialCounts(big), c.map(x => 4 * x));
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(SIZES.every(([w, h]) => w * h <= MAX_CELLS && w * h <= sand.MAX_MEASURE_CELLS && w % 2 === 0 && h % 2 === 0));
    assert.ok(SIZES.every(([w, h]) => stepsFor(w * h) >= 32 && stepsFor(w * h) <= 512 && stepsFor(w * h) % MAX_STEPS === 0));
    assert.ok(CHECKPOINTS[CHECKPOINTS.length - 1] <= 2000);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [
        ['scaling', { maxCells: MAX_CELLS }, SIZES.length], ['scaling', { maxCells: 640 * 480 }, 3],
        ['conservation', { scene: 'inert' }, CHECKPOINTS.length], ['conservation', { scene: 'fire' }, CHECKPOINTS.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys = columns`);
        const cells = paramSlots(device).map(p => p.size[0] * p.size[1]);
        assert.ok(Math.max(...cells) <= MAX_CELLS);
        if (id === 'conservation') assert.deepEqual(rows.map(r => r.steps), CHECKPOINTS);
    }
});
