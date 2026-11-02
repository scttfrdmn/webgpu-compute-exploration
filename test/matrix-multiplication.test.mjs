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

// ---- Tiled kernel (`tiled` module, Measure tab) -----------------------------------------------------------
// Replica of the tiled main(): 16 × 16 workgroups over ⌈K/16⌉ × ⌈M/16⌉; thread (lx, ly) of workgroup (wx, wy)
// owns row = 16·wy + ly, col = 16·wx + lx. Per phase t every thread stores one zero-padded element of A and of
// B into the tiles; after the (simulated) barrier each thread does 16 multiply-adds from the tiles in f32.
const T = 16;
function tiledReplica(A, B, M, N, K) {
    const out = new Float32Array(M * K).fill(NaN);
    let writes = 0;
    const wgx = Math.ceil(K / T), wgy = Math.ceil(M / T), phases = Math.ceil(N / T);
    for (let wy = 0; wy < wgy; wy++) for (let wx = 0; wx < wgx; wx++) {
        const sum = new Float32Array(T * T);   // one accumulator per thread, index ly·T + lx
        const tileA = new Float32Array(T * T), tileB = new Float32Array(T * T);
        for (let t = 0; t < phases; t++) {
            for (let ly = 0; ly < T; ly++) for (let lx = 0; lx < T; lx++) {   // every thread loads, then the barrier
                const row = wy * T + ly, col = wx * T + lx, ka = t * T + lx, kb = t * T + ly;
                tileA[ly * T + lx] = row < M && ka < N ? A[row * N + ka] : 0;
                tileB[ly * T + lx] = kb < N && col < K ? B[kb * K + col] : 0;
            }
            for (let ly = 0; ly < T; ly++) for (let lx = 0; lx < T; lx++) {
                let s = sum[ly * T + lx];
                for (let kk = 0; kk < T; kk++) s = Math.fround(s + Math.fround(tileA[ly * T + kk] * tileB[kk * T + lx]));
                sum[ly * T + lx] = s;
            }
        }
        for (let ly = 0; ly < T; ly++) for (let lx = 0; lx < T; lx++) {
            const row = wy * T + ly, col = wx * T + lx;
            if (row < M && col < K) { out[row * K + col] = sum[ly * T + lx]; writes++; }
        }
    }
    return { out, writes };
}

test('tiled module: 16 × 16 tiles in workgroup memory, two barriers, same uniform layout', (t) => {
    const mods = wgslModules();
    assert.deepEqual(mods.map(m => m.name), ['compute', 'tiled'], 'naive stays first (test/gpu/basic.gpu.test.mjs uses index 0)');
    const code = mods[1].code;
    assert.match(code, /const T = 16u;/);
    assert.match(code, /@workgroup_size\(16, 16\)/);
    assert.equal((code.match(/var<workgroup> tile[AB]: array<array<f32, 16>, 16>;/g) ?? []).length, 2);
    assert.equal((code.match(/workgroupBarrier\(\)/g) ?? []).length, 2);
    assert.match(code, /var<uniform> dimensions: vec3<u32>; \/\/ M, N, K/);
    const bytes = 2 * T * T * 4;
    t.diagnostic(`workgroup memory ${bytes} bytes (WebGPU guarantees 16,384)`);
    assert.ok(bytes <= 16384);
    // No barrier under a non-uniform branch: the `if` blocks only guard loads and the final store
    for (const block of code.matchAll(/if \(([^)]*)\) \{([^}]*)\}/g)) assert.doesNotMatch(block[2], /Barrier/);
});

test('tiled replica equals the naive replica bit for bit and writes every element once (non-square, ragged edges)', (t) => {
    for (const [M, N, K] of [[37, 23, 19], [100, 70, 45], [16, 16, 16], [1, 33, 2], [64, 64, 64]]) {
        const rand = lcg(M * 7 + N * 3 + K);
        const A = Float32Array.from({ length: M * N }, () => rand() * 2 - 1);
        const B = Float32Array.from({ length: N * K }, () => rand() * 2 - 1);
        const tiled = tiledReplica(A, B, M, N, K);
        const naive = replica(A, B, M, N, K);
        assert.equal(tiled.writes, M * K, `${M}x${N}x${K}: every element written once`);
        const ref = reference(A, B, M, N, K);
        let same = 0, maxAbs = 0;
        for (let i = 0; i < M * K; i++) {
            if (Object.is(tiled.out[i], naive.out[i])) same++;
            maxAbs = Math.max(maxAbs, Math.abs(tiled.out[i] - ref[i]));
        }
        t.diagnostic(`${M}x${N}*${N}x${K}: ${same}/${M * K} identical to naive, max|err| vs float64 ${maxAbs.toExponential(2)}`);
        assert.equal(same, M * K, 'same summation order k = 0 … N−1; zero padding adds +0');
        assert.ok(maxAbs < 1e-5);
    }
});
