// Procedural-noise Measure tab, CPU only: the compute-only API of js/examples/noise.js (measureNoise,
// SIMPLEX_SCALE / SIMPLEX_FALLOFF) and the experiments in js/teach/experiments/noise.js, driven through
// the mock device (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing;
// plus fbmRef of js/refs/noise.js. Checked: the exported constants are the ones in the WGSL; Params bytes
// match the WGSL struct; the pixel → noise-space mapping the experiment uses is the shader's; caps are
// enforced; every row carries the declared columns. The real results are checked on a software GPU in
// test/gpu/noise-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as noise from '../js/examples/noise.js';
import { perlinRef, simplexRef, fbmRef } from '../js/refs/noise.js';
import experiments, { OCTAVES, SIDES, STAT_SIDE, STAT_STRIDE, STAT_OCTAVES, noisePoint, moments, fbmStdFactor } from '../js/teach/experiments/noise.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const { measureNoise, MAX_OCTAVES } = noise;
const CODE = moduleCode(noise, 'compute');
const layout = structLayout(CODE, 'Params');
const paramsWrites = (device) => device.writes
    .filter(w => (w.buffer.usage & GPUBufferUsage.UNIFORM) && w.buffer.size === layout.size)
    .map(w => decodeStruct(w.bytes, layout));

test('SIMPLEX_SCALE and SIMPLEX_FALLOFF are the values spliced into the WGSL', () => {
    assert.equal(Number(/const SIMPLEX_SCALE: f32 = ([\d.]+);/.exec(CODE)[1]), noise.SIMPLEX_SCALE);
    assert.equal(Number(/let t = ([\d.]+) - dot\(d, d\);/.exec(CODE)[1]), noise.SIMPLEX_FALLOFF);
    assert.match(CODE, /const SIMPLEX_SCALE: f32 = 72\.0;/);
});

test('measureNoise on a mock device: Params bytes, field buffer, repeats, readback', async () => {
    const device = mockDevice();
    const r = await measureNoise({ device }, { width: 200, height: 120, noiseType: 1, octaves: 6, frequency: 3, warp: 0.4,
        ridged: true, warmup: 2, repeats: 3, readback: true });
    const params = paramsWrites(device);
    assert.equal(params.length, 5, 'one write per pass (2 warm-up + 3 timed)');
    for (const p of params) {
        assert.deepEqual(p.size, [200, 120]);
        assert.deepEqual(p.center, [0, 0]);
        assert.equal(p.unitsPerPixel, Math.fround(1 / 120));
        assert.deepEqual([p.frequency, p.octaves, p.noiseType, p.ridged, p.colorMode], [3, 6, 1, 1, 3]);
        assert.equal(p.warp, Math.fround(0.4));
        assert.equal(p.time, 0);
    }
    assert.equal(r.evaluations, 200 * 120 * 6 * 3, 'three fBm per pixel with warp');
    assert.equal(r.field.length, 200 * 120);
    assert.ok(device.buffers.some(b => b.size === 4 * 200 * 120 && (b.usage & GPUBufferUsage.STORAGE) && (b.usage & GPUBufferUsage.COPY_SRC)));
    assert.equal(r.gpuMs, null);
    const plain = await measureNoise({ device: mockDevice() }, { width: 64, height: 64 });
    assert.equal(plain.field, undefined);
});

test('measureNoise enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measureNoise({ device }, { width: 4, height: 64 }), RangeError);
    await assert.rejects(measureNoise({ device }, { width: 64, height: noise.MAX_MEASURE_SIDE + 1 }), RangeError);
    await assert.rejects(measureNoise({ device }, { width: 64, height: 64, octaves: MAX_OCTAVES + 1 }), RangeError);
    await assert.rejects(measureNoise({ device }, { width: 64, height: 64, octaves: 0 }), RangeError);
    await assert.rejects(measureNoise({ device }, { width: 64, height: 64, noiseType: 2 }), RangeError);
    await assert.rejects(measureNoise({ device }, { width: 64, height: 64, warp: 9 }), RangeError);
    const many = mockDevice();
    await measureNoise({ device: many }, { width: 64, height: 64, warmup: 50, repeats: 50 });
    assert.equal(paramsWrites(many).length, 2 * noise.MAX_MEASURE_REPEATS);
});

test('fbmRef: one octave is the base noise; the octave rule matches the shader (p → 2p + offset)', () => {
    const simplex = (x, y, z) => simplexRef(x, y, z, noise.SIMPLEX_SCALE, noise.SIMPLEX_FALLOFF);
    for (const base of [perlinRef, simplex]) {
        assert.equal(fbmRef(base, 1.3, -2.7, 0.5, { octaves: 1 }), base(1.3, -2.7, 0.5));
        const two = fbmRef(base, 1.3, -2.7, 0.5, { octaves: 2, lacunarity: 2, gain: 0.5 });
        const expect = (base(1.3, -2.7, 0.5) + 0.5 * base(2 * 1.3 + 19.1, 2 * -2.7 + 7.3, 2 * 0.5 + 3.7)) / 1.5;
        assert.ok(Math.abs(two - expect) < 1e-15);
        const r = fbmRef(base, 0.3, 0.4, 0.5, { octaves: 3, ridged: true });
        assert.ok(r >= -1 && r <= 1);
    }
});

test('noisePoint is the shader mapping (pixel center, view one unit tall, z = time + 0.5)', () => {
    assert.deepEqual(noisePoint(0, 0, 256, 8), [(0.5 - 128) / 256 * 8, (0.5 - 128) / 256 * 8, 0.5]);
    assert.deepEqual(noisePoint(128, 64, 256, 8), [0.5 / 256 * 8, (64.5 - 128) / 256 * 8, 0.5]);
    const m = moments([1, 2, 3, 4]);
    assert.deepEqual(m, { mean: 2.5, std: Math.sqrt(1.25), min: 1, max: 4 });
    assert.equal(fbmStdFactor(1), 1);
    assert.ok(Math.abs(fbmStdFactor(2) - Math.sqrt(1.25) / 1.5) < 1e-15);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(OCTAVES.every(o => o >= 1 && o <= MAX_OCTAVES) && STAT_OCTAVES.every(o => o <= MAX_OCTAVES));
    assert.ok(SIDES.every(s => s <= noise.MAX_MEASURE_SIDE));
    assert.ok((STAT_SIDE / STAT_STRIDE) ** 2 <= 4096);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [['cost', { side: 2048, warp: 1 }, OCTAVES.length], ['statistics', { noiseType: 1 }, STAT_OCTAVES.length]];
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
        assert.ok(p.every(q => Math.max(...q.size) <= 2048 && q.octaves <= MAX_OCTAVES));
        if (id === 'cost') assert.ok(p.every(q => q.warp === Math.fround(0.4)));
    }
});
