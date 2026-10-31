// Matrix multiplication (example #2).
// CPU replica of the `compute` shader; validates the algorithm and index math, not the WGSL itself — see #51.
// The shipped runMatrixMultiplication() is driven with a mock device to capture the matrices and the
// `dimensions: vec3<u32>` uniform it uploads.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runMatrixMultiplication, wgslModules } from '../js/examples/matrix-multiplication.js';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { typeLayout } from './helpers/wgsl-layout.mjs';
import { matmulF64 } from './helpers/basic-refs.mjs';

// Replica of main(): global_id.x = row, global_id.y = col over an 8x8 workgroup grid of
// ceil(M/8) x ceil(K/8); A is MxN row-major, B is NxK row-major, result is MxK row-major.
function replicaInvocation(A, B, M, N, K, row, col, out) {
    if (row >= M || col >= K) return false;
    let sum = 0;
    for (let i = 0; i < N; i++) sum = Math.fround(sum + Math.fround(A[row * N + i] * B[i * K + col]));
    out[row * K + col] = sum;
    return true;
}

function replica(A, B, M, N, K) {
    const out = new Float32Array(M * K).fill(NaN);
    let writes = 0;
    for (let gy = 0; gy < Math.ceil(K / 8) * 8; gy++)
        for (let gx = 0; gx < Math.ceil(M / 8) * 8; gx++)
            if (replicaInvocation(A, B, M, N, K, gx, gy, out)) writes++;
    return { out, writes };
}

// Independent reference: textbook triple loop in float64, i-k-j order (shared with test/gpu/).
const reference = matmulF64;

function lcg(seed) {
    let s = seed >>> 0;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

test('dimensions uniform is vec3<u32> (M, N, K) written as integers', async (t) => {
    const device = mockDevice();
    await runMatrixMultiplication({ device });
    const dimsBuf = device.buffers[3];
    const layout = typeLayout('vec3<u32>');
    assert.equal(dimsBuf.size, layout.size, '12-byte uniform');
    const u = new Uint32Array(dimsBuf.getMappedRange());
    t.diagnostic(`dimensions = [${[...u]}]`);
    assert.deepEqual([...u], [512, 512, 512]);
    assert.match(wgslModules()[0].code, /var<uniform> dimensions: vec3<u32>; \/\/ M, N, K/);
});

test('shipped 512x512 matrices: sampled replica entries vs float64', async (t) => {
    const device = mockDevice();
    await runMatrixMultiplication({ device });
    const A = new Float32Array(device.buffers[0].getMappedRange());
    const B = new Float32Array(device.buffers[1].getMappedRange());
    const [M, N, K] = new Uint32Array(device.buffers[3].getMappedRange());
    assert.equal(A.length, M * N);
    assert.equal(B.length, N * K);
    const rand = lcg(7);
    const out = new Float32Array(M * K);
    let maxRel = 0;
    for (let s = 0; s < 256; s++) {
        const row = Math.floor(rand() * M), col = Math.floor(rand() * K);
        replicaInvocation(A, B, M, N, K, row, col, out);
        let ref = 0;
        for (let i = 0; i < N; i++) ref += A[row * N + i] * B[i * K + col];
        maxRel = Math.max(maxRel, Math.abs(out[row * K + col] - ref) / ref);
    }
    t.diagnostic(`256 sampled entries, max relative error vs float64 = ${maxRel.toExponential(2)}`);
    // Sequential f32 accumulation of 512 positive terms: error grows ~ N * eps / 2 worst case.
    assert.ok(maxRel < 512 * 2 ** -24);
});

test('non-square 37x23 * 23x19 exercises row/col mapping and bounds', (t) => {
    const M = 37, N = 23, K = 19;
    const rand = lcg(42);
    const A = Float32Array.from({ length: M * N }, () => rand() * 2 - 1);
    const B = Float32Array.from({ length: N * K }, () => rand() * 2 - 1);
    const { out, writes } = replica(A, B, M, N, K);
    const ref = reference(A, B, M, N, K);
    let maxAbs = 0;
    for (let i = 0; i < M * K; i++) maxAbs = Math.max(maxAbs, Math.abs(out[i] - ref[i]));
    t.diagnostic(`writes=${writes} (expected ${M * K}) max|err|=${maxAbs.toExponential(2)}`);
    assert.equal(writes, M * K, 'every result element written once; padding invocations masked');
    assert.ok(out.every(Number.isFinite));
    assert.ok(maxAbs < 1e-5, 'f32 dot products of length 23 with |terms| <= 1');
});
