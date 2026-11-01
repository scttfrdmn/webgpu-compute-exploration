// Path tracer Measure tab, CPU only: the compute-only API of js/examples/path-tracer.js (measurePathTracer,
// PathTracer with canvas = null, furnaceScene) and the experiments in js/teach/experiments/path-tracer.js,
// driven through the mock device (test/helpers/mock-gpu.mjs), which records writes but runs nothing.
// Checked: the Params bytes of every sample (sampleCount counting up, size, bounces, seed, NEE, no depth
// of field); the furnace scene; the caps; imageRmse; every sweep within its caps and every row with the
// declared columns. Real results: test/gpu/path-tracer-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    measurePathTracer, furnaceScene, FURNACE, MEASURE_MAX_PIXELS, MEASURE_MAX_SPP, MEASURE_BATCH, MAX_BOUNCES,
    SCENES, cameraBasis, wgslModules,
} from '../js/examples/path-tracer.js';
import { furnaceRadiance, imageRmse } from '../js/refs/path-tracer.js';
import experiments, { NOISE_SIZE, SPPS, REF_SPP, BOUNCE_SIZE, BOUNCES, BOUNCE_SPP } from '../js/teach/experiments/path-tracer.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const PARAMS = structLayout(wgslModules().find(m => m.name === 'trace').code, 'Params');
const paramWrites = (device) => device.writes
    .filter(w => w.bytes.byteLength === PARAMS.size && (w.buffer.usage & GPUBufferUsage.UNIFORM))
    .map(w => decodeStruct(w.bytes, PARAMS));

test('measurePathTracer on a mock device: one Params write per sample, checkpoints, image size', async () => {
    const device = mockDevice();
    let calls = 0;
    const r = await measurePathTracer({ device }, { scene: 'cornell', width: 40, height: 30, checkpoints: [3, 10, 20],
        bounces: 7, useNEE: false, seed: 99, between: async () => { calls++; } });
    const params = paramWrites(device);
    assert.equal(params.length, 20);
    assert.deepEqual(params.map(p => p.sampleCount), Array.from({ length: 20 }, (_, i) => i));
    const { eye, forward } = cameraBasis(SCENES.cornell.camera);
    const f = Math.fround;
    for (const p of params) {
        assert.deepEqual(p.size, [40, 30]);
        assert.equal(p.maxBounces, 7);
        assert.equal(p.useNEE, 0);
        assert.equal(p.seed, 99);
        assert.equal(p.camForward[3], 0, 'no depth of field');
        assert.equal(p.camUp[3], 0, 'no firefly clamp');
        assert.deepEqual(p.camPos.slice(0, 3), eye.map(f));
        assert.deepEqual(p.camForward.slice(0, 3), forward.map(f));
    }
    // batches of at most MEASURE_BATCH that stop at every checkpoint: 3 | 7 | 8, 2
    assert.equal(calls, 4);
    assert.equal(r.bounces, 7);
    assert.deepEqual(r.checkpoints.map(c => c.spp), [3, 10, 20]);
    for (const c of r.checkpoints) {
        assert.equal(c.image.length, 40 * 30 * 3);
        assert.equal(c.gpuMsPerSample, null);
        assert.ok(c.wallMsPerSample >= 0 && Number.isFinite(c.raysPerPixel));
    }
    assert.ok(device.buffers.some(b => b.size === 40 * 30 * 16 && (b.usage & GPUBufferUsage.MAP_READ)), 'accumulator readback');
    assert.ok(device.buffers.some(b => b.size === 800 * 600 * 16 && (b.usage & GPUBufferUsage.COPY_SRC)), 'accumulator is COPY_SRC');

    const furnace = mockDevice();
    const rf = await measurePathTracer({ device: furnace }, { scene: 'furnace', width: 16, height: 16, checkpoints: [1] });
    const pf = paramWrites(furnace)[0];
    assert.equal(rf.bounces, 5);
    assert.equal(pf.primCount, 6);
    assert.equal(pf.lightCount, 6);
    assert.equal(pf.envType, 0);
});

test('furnaceScene: six inward-facing emissive quads around the camera; furnace radiance series', () => {
    const scene = furnaceScene();
    assert.equal(scene.count, 6);
    assert.equal(scene.lightCount, 6);
    for (let i = 0; i < 6; i++) {
        const o = i * 24;
        const q = [0, 1, 2].map(k => scene.f32[o + k]), u = [4, 5, 6].map(k => scene.f32[o + k]), v = [8, 9, 10].map(k => scene.f32[o + k]);
        const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        const toCenter = [0.5 - q[0], 0.5 - q[1], 0.5 - q[2]];
        assert.ok(n[0] * toCenter[0] + n[1] * toCenter[1] + n[2] * toCenter[2] > 0, `quad ${i} faces inward`);
        assert.deepEqual([scene.f32[o + 12], scene.f32[o + 16]], [FURNACE.rho, FURNACE.le]);
    }
    const { eye } = cameraBasis(FURNACE.camera);
    assert.ok(eye.every(c => c > 0.05 && c < 0.95), 'the camera is inside the box');
    assert.equal(furnaceRadiance(0.5, 1, 1), 1.5);
    assert.ok(Math.abs(furnaceRadiance(0.5, 1, 5) - 1.96875) < 1e-15);
});

test('measurePathTracer enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measurePathTracer({ device }, { scene: 'nope' }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { width: 201, height: 150 }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { width: 4, height: 75 }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { checkpoints: [] }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { checkpoints: [4, 4] }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { checkpoints: [MEASURE_MAX_SPP + 1] }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { bounces: MAX_BOUNCES + 1 }), RangeError);
    await assert.rejects(measurePathTracer({ device }, { bounces: 0 }), RangeError);
});

test('imageRmse: against a constant and an image', () => {
    const img = Float32Array.from([1, 2, 3, 4]);
    const a = imageRmse(img, 2.5);
    assert.ok(Math.abs(a.rmse - Math.sqrt(1.25)) < 1e-12 && a.mean === 2.5 && Math.abs(a.relative - Math.sqrt(1.25) / 2.5) < 1e-12);
    const b = imageRmse(img, [1, 2, 3, 6]);
    assert.ok(Math.abs(b.rmse - 1) < 1e-12 && b.mean === 3);
    assert.equal(imageRmse(img, 0).relative, Infinity);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(NOISE_SIZE[0] * NOISE_SIZE[1] <= MEASURE_MAX_PIXELS && BOUNCE_SIZE[0] * BOUNCE_SIZE[1] <= MEASURE_MAX_PIXELS);
    assert.ok(SPPS[SPPS.length - 1] <= MEASURE_MAX_SPP && REF_SPP <= MEASURE_MAX_SPP && BOUNCE_SPP <= MEASURE_MAX_SPP);
    assert.ok(BOUNCES.every(b => b >= 1 && b <= MAX_BOUNCES));
    assert.ok(MEASURE_BATCH <= 8, 'a few one-sample submits at a time, like the example');
    for (const e of experiments) assert.ok(e.budgetMs <= 20000);
});

test('experiments run on a mock device: one row per sweep value, declared columns, one sample per dispatch', async (t) => {
    const cases = [
        ['spp-noise', { scene: 'furnace', nee: true }, SPPS.length, SPPS[SPPS.length - 1]],
        ['spp-noise', { scene: 'cornell', nee: false }, SPPS.length, REF_SPP + SPPS[SPPS.length - 1]],
        ['bounces', { scene: 'furnace' }, BOUNCES.length, BOUNCES.length * BOUNCE_SPP],
        ['bounces', { scene: 'cornell' }, BOUNCES.length, BOUNCES.length * BOUNCE_SPP],
    ];
    for (const [id, settings, expected, samples] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows, ${paramWrites(device).length} samples`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort());
        const params = paramWrites(device);
        assert.equal(params.length, samples, 'one Params write (one dispatch) per sample per pixel');
        assert.ok(params.every(p => p.size[0] * p.size[1] <= MEASURE_MAX_PIXELS));
        if (id === 'bounces') {
            assert.deepEqual(rows.map(r => r.bounces), BOUNCES);
            if (settings.scene === 'furnace') assert.deepEqual(rows.map(r => r.exact), BOUNCES.map(b => furnaceRadiance(FURNACE.rho, FURNACE.le, b)));
        }
    }
});
