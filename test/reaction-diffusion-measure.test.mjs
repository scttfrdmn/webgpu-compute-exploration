// Reaction-diffusion Measure tab, CPU only: the compute-only API of js/examples/reaction-diffusion.js
// (measureSteps, seedCells, the exported model constants) and the experiments in
// js/teach/experiments/reaction-diffusion.js, driven through the mock device (test/helpers/mock-gpu.mjs),
// which records buffers and uniform writes but runs nothing. Checked: the exported constants are the
// ones in the WGSL; Params bytes match the WGSL struct; seedCells is the shipped seed(); caps are
// enforced; sweeps stay within their caps; every row carries the declared columns. The real results
// are checked on a software GPU in test/gpu/reaction-diffusion-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as rd from '../js/examples/reaction-diffusion.js';
import { stepReference } from '../js/refs/reaction-diffusion.js';
import experiments, { SIDES, MAX_SIDE, stepsFor, CHECKPOINTS, ACCURACY_SIDE, MODEL, compare } from '../js/teach/experiments/reaction-diffusion.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import * as refs from './helpers/reaction-diffusion-refs.mjs';

const { measureSteps } = rd;
const layout = structLayout(moduleCode(rd, 'compute'), 'Params');
const paramsWrites = (device) => device.writes
    .filter(w => w.buffer.size === layout.size && (w.buffer.usage & GPUBufferUsage.UNIFORM))
    .map(w => decodeStruct(w.bytes, layout));

test('exported model constants are the ones spliced into the WGSL', () => {
    // refs parses DU, DV, DT and the Laplacian weights from the shipped WGSL text
    assert.deepEqual({ DU: rd.DU, DV: rd.DV, DT: rd.DT, W_EDGE: rd.W_EDGE, W_CORNER: rd.W_CORNER },
        { DU: refs.DU, DV: refs.DV, DT: refs.DT, W_EDGE: refs.W_EDGE, W_CORNER: refs.W_CORNER });
    assert.deepEqual(MODEL, refs.MODEL);
    const code = moduleCode(rd, 'compute');
    assert.match(code, /const DU: f32 = 1\.0;/);
    assert.match(code, /let lap = -c \+ 0\.2 \* edges \+ 0\.05 \* corners;/);
});

test('seedCells with a seeded generator is the shipped seed() with the same Math.random stream', () => {
    const w = 61, h = 47;
    // shippedSeed drives seed() with an LCG; feed seedCells the same LCG
    let s = 9 >>> 0;
    const lcg = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
    assert.deepEqual(rd.seedCells(w, h, lcg), refs.shippedSeed(w, h, 9));
    const a = rd.seedCells(128, 128, rd.seededRandom(1)), b = rd.seedCells(128, 128, rd.seededRandom(1));
    assert.deepEqual(a, b, 'seededRandom is deterministic');
    assert.ok(a.some((v, i) => i % 2 === 1 && v > 0.2), 'patches of V seeded');
});

test('measureSteps on a mock device: Params bytes, buffers, state uploaded, samples, between()', async () => {
    const device = mockDevice();
    const w = 40, h = 24, cells = rd.seedCells(w, h, rd.seededRandom(3));
    let calls = 0;
    const r = await measureSteps({ device }, { width: w, height: h, cells, feed: 0.03, kill: 0.06, steps: 70, warmup: 4,
        readAt: [0, 10, 64, 70, 99], between: async () => { calls++; } });
    // warm-up, then chunks of ≤ 32 steps split at the read points: 10, 32, 22, 6 → 4 submits
    const params = paramsWrites(device);
    assert.equal(params.length, 5, 'one Params write per submit (1 warm-up + 4)');
    for (const p of params) {
        assert.deepEqual({ size: p.size, brushOn: p.brushOn, simulate: p.simulate }, { size: [w, h], brushOn: 0, simulate: 1 });
        assert.equal(p.feed, Math.fround(0.03));
        assert.equal(p.kill, Math.fround(0.06));
    }
    assert.equal(calls, 4);
    assert.deepEqual(r.samples.map(s => s.step), [0, 10, 64, 70], 'read points beyond `steps` are dropped');
    for (const s of r.samples) assert.equal(s.cells.length, 2 * w * h);
    const upload = device.writes.find(wr => wr.bytes.byteLength === 8 * w * h);
    assert.deepEqual(new Float32Array(upload.bytes.buffer), cells, 'initial state uploaded unchanged');
    const storage = device.buffers.filter(b => b.size === 8 * w * h && (b.usage & GPUBufferUsage.STORAGE));
    assert.equal(storage.length, 2, 'two ping-pong cell buffers');
    assert.ok(storage.every(b => b.usage & GPUBufferUsage.COPY_SRC), 'cell buffers can be read back');
    assert.equal(r.gpuMs, null, 'no timestamp-query on the mock device');
    assert.ok(r.wallMsPerStep >= 0 && r.steps === 70);
});

test('measureSteps enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measureSteps({ device }, { width: 4, height: 64, steps: 1 }), RangeError);
    await assert.rejects(measureSteps({ device }, { width: rd.MAX_MEASURE_SIDE + 1, height: 64, steps: 1 }), RangeError);
    await assert.rejects(measureSteps({ device }, { width: 64, height: 64, steps: 0 }), RangeError);
    await assert.rejects(measureSteps({ device }, { width: 64, height: 64, steps: rd.MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureSteps({ device }, { width: 64, height: 64, steps: 1, warmup: 33 }), RangeError);
    await assert.rejects(measureSteps({ device }, { width: 64, height: 64, steps: 1, cells: new Float32Array(10) }), RangeError);
    await assert.rejects(measureSteps({ device }, { width: 64, height: 64, steps: 1, feed: 0.5 }), RangeError);
});

test('compare() and stepReference with the exported constants', () => {
    const w = 16, h = 12;
    const init = rd.seedCells(w, h, rd.seededRandom(4));
    const U = Float64Array.from({ length: w * h }, (_, i) => init[2 * i]);
    const V = Float64Array.from({ length: w * h }, (_, i) => init[2 * i + 1]);
    const [U2, V2] = stepReference(U, V, w, h, 0.0545, 0.062, MODEL);
    const [U3, V3] = refs.stepReference(U, V, w, h, 0.0545, 0.062);
    assert.deepEqual([U2, V2], [U3, V3]);
    const c = compare(init, U, V);
    assert.ok(c.max < 1e-8 && c.rms < 1e-8, 'f32 copy of the same state');
    const shifted = Float32Array.from(init, (v, i) => (i === 7 ? v + 0.25 : v));
    assert.ok(Math.abs(compare(shifted, U, V).max - 0.25) < 1e-6);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(SIDES.every(s => s <= MAX_SIDE && s <= rd.MAX_MEASURE_SIDE));
    assert.ok(SIDES.every(s => stepsFor(s) >= 64 && stepsFor(s) <= 1024));
    assert.ok(CHECKPOINTS[CHECKPOINTS.length - 1] <= rd.MAX_MEASURE_STEPS && ACCURACY_SIDE <= 128);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [
        ['scaling', { maxSide: 2048 }, SIDES.length],
        ['scaling', { maxSide: 512 }, 3],
        ['precision', { preset: 'chaos' }, CHECKPOINTS.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected, `${id}: rows`);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys = columns`);
        const sizes = paramsWrites(device).map(p => p.size[0] * p.size[1]);
        assert.ok(Math.max(...sizes) <= MAX_SIDE ** 2);
        if (id === 'precision') assert.deepEqual(rows.map(r => r.steps), CHECKPOINTS);
    }
});
