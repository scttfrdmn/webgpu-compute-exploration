// Vector addition Measure tab, CPU only: measureVectorAdd() in js/examples/vector-addition.js and the
// experiment in js/teach/experiments/vector-addition.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and writes but runs nothing. Checked: buffer sizes
// and usages, inputs uploaded unchanged, the caps, rows carry the declared columns, the rounding
// reference. The real sums are checked on a software GPU in test/gpu/vector-addition-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureVectorAdd, MAX_DISPATCH_N, WORKGROUP_SIZE } from '../js/examples/vector-addition.js';
import experiments, { SIZES, makeInputs, roundingMismatches } from '../js/teach/experiments/vector-addition.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });

test('measureVectorAdd on a mock device: buffers, uploads, result shape', async () => {
    const device = mockDevice();
    const n = 1000;   // not a multiple of 64
    const { a, b } = makeInputs(n);
    const r = await measureVectorAdd({ device }, { a, b, warmup: 1, repeats: 3 });
    assert.equal(r.n, n);
    assert.equal(r.workgroups, Math.ceil(n / WORKGROUP_SIZE));
    assert.equal(r.gpuMs, null, 'no timestamp-query on the mock');
    assert.ok(r.kernelWallMs >= 0 && r.roundTripMs >= 0);
    assert.equal(r.sum.length, n);
    const sizes = device.buffers.map(x => x.size);
    assert.equal(sizes.filter(s => s === 4 * n).length, 4, 'two inputs, output, staging');
    assert.ok(device.buffers.some(x => x.size === 4 * n && (x.usage & GPUBufferUsage.MAP_READ)), 'staging buffer');
    const uploads = device.writes.filter(w => w.bytes.byteLength === 4 * n);
    assert.equal(uploads.length, 4, 'inputs uploaded before timing and again for the round trip');
    assert.deepEqual(new Float32Array(uploads[0].bytes.buffer), a);
    assert.deepEqual(new Float32Array(uploads[1].bytes.buffer), b);
});

test('measureVectorAdd enforces the caps', async () => {
    const device = mockDevice();
    assert.equal(MAX_DISPATCH_N, 65535 * 64, 'one 1D dispatch of 64-wide workgroups');
    await assert.rejects(measureVectorAdd({ device }, { a: new Float32Array(0), b: new Float32Array(0) }), RangeError);
    await assert.rejects(measureVectorAdd({ device }, { a: new Float32Array(10), b: new Float32Array(11) }), RangeError);
    const big = { length: MAX_DISPATCH_N + 1 };
    await assert.rejects(measureVectorAdd({ device }, { a: big, b: big }), RangeError);
});

test('rounding reference: fround(a + b) is the correctly rounded f32 sum', () => {
    const { a, b } = makeInputs(10000, 5);
    const sum = Float32Array.from(a, (x, i) => x + b[i]);   // the store rounds once
    assert.equal(roundingMismatches(a, b, sum), 0);
    sum[17] = Math.fround(sum[17] * (1 + 2 ** -22));
    assert.equal(roundingMismatches(a, b, sum), 1);
});

test('experiment on a mock device: one row per size, declared columns, sizes within the cap', async (t) => {
    const [e] = experiments;
    assert.ok(SIZES.every(n => n <= MAX_DISPATCH_N) && SIZES.at(-1) === MAX_DISPATCH_N);
    const device = mockDevice();
    const rows = await runExperiment(e, {}, mockCtx(device, e.budgetMs));
    assert.deepEqual(rows.map(r => r.n), SIZES);
    for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort());
    assert.ok(Math.max(...device.buffers.map(x => x.size)) <= 4 * MAX_DISPATCH_N);
    t.diagnostic(`${rows.length} rows; largest buffer ${Math.max(...device.buffers.map(x => x.size))} bytes`);
});
