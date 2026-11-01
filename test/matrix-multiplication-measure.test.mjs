// Matrix multiplication Measure tab, CPU only: measureMatmul() in js/examples/matrix-multiplication.js
// and the experiment in js/teach/experiments/matrix-multiplication.js, on the mock device. Checked: the
// dimensions uniform (vec3<u32>) bytes, buffer sizes, the caps, rows carry the declared columns, and the
// row-subset float64 reference equals the full one. Real products: test/gpu/matrix-multiplication-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureMatmul, wgslModules, MAX_MATMUL_DIM } from '../js/examples/matrix-multiplication.js';
import experiments, { SIZES, MAX_SIZE, sampleRows, makeMatrix, maxRelError } from '../js/teach/experiments/matrix-multiplication.js';
import { matmulF64, matmulRowsF64 } from '../js/refs/basic.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { typeLayout } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });

test('measureMatmul on a mock device: dimensions bytes, buffer sizes, result shape', async () => {
    const device = mockDevice();
    const [M, N, K] = [37, 23, 19];
    const A = makeMatrix(M, N, 1), B = makeMatrix(N, K, 2);
    const r = await measureMatmul({ device }, { A, B, M, N, K, repeats: 2 });
    assert.match(wgslModules()[0].code, /var<uniform> dimensions: vec3<u32>; \/\/ M, N, K/);
    const dims = device.writes.find(w => w.buffer.usage & GPUBufferUsage.UNIFORM);
    assert.ok(dims.buffer.size >= typeLayout('vec3<u32>', {}).size);
    assert.deepEqual([...new Uint32Array(dims.bytes.buffer)].slice(0, 3), [M, N, K], 'u32 M, N, K (not floats)');
    const sizes = device.buffers.map(b => b.size);
    for (const s of [4 * M * N, 4 * N * K, 4 * M * K]) assert.ok(sizes.includes(s), `a ${s}-byte buffer`);
    assert.deepEqual({ M: r.M, N: r.N, K: r.K, flops: r.flops, gpuMs: r.gpuMs }, { M, N, K, flops: 2 * M * N * K, gpuMs: null });
    assert.equal(r.C.length, M * K);
});

test('measureMatmul enforces the caps', async () => {
    const device = mockDevice();
    const m = (n) => new Float32Array(n * n);
    await assert.rejects(measureMatmul({ device }, { A: m(0), B: m(0), M: 0, N: 0, K: 0 }), RangeError);
    const big = MAX_MATMUL_DIM + 1;
    await assert.rejects(measureMatmul({ device }, { A: { length: big * 4 }, B: { length: 4 * 4 }, M: big, N: 4, K: 4 }), RangeError);
    await assert.rejects(measureMatmul({ device }, { A: m(4), B: m(5), M: 4, N: 4, K: 4 }), RangeError);
});

test('matmulRowsF64 equals the rows of matmulF64; sampled rows are spread and distinct', () => {
    const [M, N, K] = [40, 17, 23];
    const A = makeMatrix(M, N, 3), B = makeMatrix(N, K, 4);
    const full = matmulF64(A, B, M, N, K);
    const rows = sampleRows(M, 7);
    const some = matmulRowsF64(A, B, M, N, K, rows);
    rows.forEach((i, r) => { for (let j = 0; j < K; j++) assert.equal(some[r * K + j], full[i * K + j]); });
    assert.equal(new Set(sampleRows(1024)).size, 16);
    assert.equal(maxRelError(Float32Array.from(full), K, [0], full.subarray(0, K)) < 2 ** -24, true, 'f32 rounding of the exact product');
});

test('experiment on a mock device: one row per size, declared columns, sizes within the cap', async () => {
    const [e] = experiments;
    assert.ok(SIZES.every(n => n <= MAX_SIZE) && MAX_SIZE <= MAX_MATMUL_DIM);
    const device = mockDevice();
    const rows = await runExperiment(e, {}, mockCtx(device, e.budgetMs));
    assert.deepEqual(rows.map(r => r.n), SIZES);
    for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort());
    assert.ok(Math.max(...device.buffers.map(b => b.size)) <= 4 * MAX_SIZE * MAX_SIZE);
});
