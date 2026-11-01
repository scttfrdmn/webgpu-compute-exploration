// Matrix multiplication Measure tab, CPU only: measureMatmul() in js/examples/matrix-multiplication.js
// (both kernels), the CPU baseline matmulBlockedF32 in js/refs/basic.js and the experiment in
// js/teach/experiments/matrix-multiplication.js, on the mock device. Checked: the dimensions uniform
// (vec3<u32>) bytes, buffer sizes, dispatch grids, phases, the caps, rows carry exactly the declared columns,
// the blocked CPU product against an f32 replica and float64, and verify() / interpret() on synthetic rows.
// Real products: test/gpu/matrix-multiplication-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureMatmul, wgslModules, MAX_MATMUL_DIM, KERNELS, TILED_T } from '../js/examples/matrix-multiplication.js';
import experiments, {
    SIZES, MAX_SIZE, CPU_MAX_N, ERR_FACTOR, FIT_MIN_MS, PROTOCOL, sampleRows, makeMatrix, maxRelError, maxRelDiff, sqrtNu, fitTimes,
} from '../js/teach/experiments/matrix-multiplication.js';
import { matmulF64, matmulRowsF64, matmulBlockedF32, CPU_BLOCK } from '../js/refs/basic.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { PHASES } from '../js/phase-clock.js';
import { buildResultRecord, validateResultRecord, SCHEMA_PATH } from '../js/teach/result-record.js';
import { typeLayout } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });
const [e] = experiments;
const schema = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', SCHEMA_PATH), 'utf8'));

for (const kernel of ['naive', 'tiled']) {
    test(`measureMatmul (${kernel}) on a mock device: dimensions bytes, buffer sizes, phases, result shape`, async () => {
        const device = mockDevice();
        const [M, N, K] = [37, 23, 19];
        const A = makeMatrix(M, N, 1), B = makeMatrix(N, K, 2);
        const r = await measureMatmul({ device }, { A, B, M, N, K, kernel, repeats: 2 });
        const code = wgslModules()[kernel === 'naive' ? 0 : 1].code;
        assert.match(code, /var<uniform> dimensions: vec3<u32>; \/\/ M, N, K/);
        const dims = device.writes.find(w => w.buffer.usage & GPUBufferUsage.UNIFORM);
        assert.ok(dims.buffer.size >= typeLayout('vec3<u32>', {}).size);
        assert.deepEqual([...new Uint32Array(dims.bytes.buffer)].slice(0, 3), [M, N, K], 'u32 M, N, K (not floats)');
        const sizes = device.buffers.map(b => b.size);
        for (const s of [4 * M * N, 4 * N * K, 4 * M * K]) assert.ok(sizes.includes(s), `a ${s}-byte buffer`);
        assert.deepEqual({ M: r.M, N: r.N, K: r.K, kernel: r.kernel, flops: r.flops, gpuMs: r.gpuMs },
            { M, N, K, kernel, flops: 2 * M * N * K, gpuMs: null });
        assert.equal(r.C.length, M * K);
        assert.deepEqual(Object.keys(r.phases).sort(), [...PHASES].sort(), 'compile, upload, dispatch, readback');
        assert.equal(r.phases.dispatch, r.wallMs);
    });
}

test('dispatch grids cover C: naive x = rows (8 × 8), tiled x = columns (16 × 16)', () => {
    assert.equal(TILED_T, 16);
    for (const [M, K] of [[37, 19], [1024, 1024], [1, 17]]) {
        assert.deepEqual(KERNELS.naive.workgroups(M, K), [Math.ceil(M / 8), Math.ceil(K / 8)]);
        const [wx, wy] = KERNELS.tiled.workgroups(M, K);
        assert.ok(wx * 16 >= K && (wx - 1) * 16 < K && wy * 16 >= M && (wy - 1) * 16 < M);
        assert.ok(wx <= 65535 && wy <= 65535);
    }
});

test('measureMatmul enforces the caps', async () => {
    const device = mockDevice();
    const m = (n) => new Float32Array(n * n);
    await assert.rejects(measureMatmul({ device }, { A: m(0), B: m(0), M: 0, N: 0, K: 0 }), RangeError);
    const big = MAX_MATMUL_DIM + 1;
    await assert.rejects(measureMatmul({ device }, { A: { length: big * 4 }, B: { length: 4 * 4 }, M: big, N: 4, K: 4 }), RangeError);
    await assert.rejects(measureMatmul({ device }, { A: m(4), B: m(5), M: 4, N: 4, K: 4 }), RangeError);
    await assert.rejects(measureMatmul({ device }, { A: m(4), B: m(4), M: 4, N: 4, K: 4, kernel: 'strassen' }), RangeError);
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
    assert.equal(maxRelDiff(Float32Array.from(full), Float32Array.from(full), K, [0], full.subarray(0, K)), 0);
});

// The CPU baseline sums k = 0 … N−1 with one rounding to f32 per term (c ← fround(c + a·b), a·b exact in double),
// whatever the block size; so it equals that sequential replica bit for bit and stays within ~√N·u of float64.
test('matmulBlockedF32: equals the sequential f32 replica for any block size; close to float64', (t) => {
    for (const [M, N, K] of [[37, 23, 19], [130, 129, 67], [64, 64, 64], [200, 300, 100]]) {
        const A = makeMatrix(M, N, M + 1), B = makeMatrix(N, K, K + 2);
        const seq = new Float32Array(M * K);
        for (let i = 0; i < M; i++) for (let j = 0; j < K; j++) {
            let c = 0;
            for (let k = 0; k < N; k++) c = Math.fround(c + A[i * N + k] * B[k * K + j]);
            seq[i * K + j] = c;
        }
        const ref = matmulF64(A, B, M, N, K);
        for (const block of [1, 7, 16, CPU_BLOCK, 1000]) {
            const C = matmulBlockedF32(A, B, M, N, K, { block });
            assert.equal(C.length, M * K);
            for (let i = 0; i < M * K; i++) assert.ok(Object.is(C[i], seq[i]), `${M}x${N}x${K} block ${block}: entry ${i}`);
        }
        const out = new Float32Array(M * K).fill(99);
        assert.equal(matmulBlockedF32(A, B, M, N, K, { out }), out, 'reuses and clears out');
        const err = maxRelError(out, K, Array.from({ length: M }, (_, i) => i), ref);
        t.diagnostic(`${M}x${N}*${N}x${K}: max rel. error vs float64 ${err.toExponential(2)} = ${(err / sqrtNu(N)).toFixed(2)} √N·u`);
        assert.ok(err > 0 && err < ERR_FACTOR * sqrtNu(N));
    }
});

test('experiment on a mock device: one row per size, declared columns only, caps, CPU only up to its cap', async () => {
    assert.ok(SIZES.every(n => n <= MAX_SIZE) && MAX_SIZE <= MAX_MATMUL_DIM && CPU_MAX_N <= 512);
    assert.ok(e.budgetMs <= 20000);
    assert.deepEqual(PROTOCOL, { warmup: 1, repeats: 5 });
    const device = mockDevice();
    const ctx = mockCtx(device, e.budgetMs);
    const rows = await runExperiment(e, { maxN: 1024 }, ctx);
    assert.deepEqual(rows.map(r => r.n), SIZES);
    const allowed = new Set(e.columns.flatMap(c => [c.key, `${c.key}Lo`, `${c.key}Hi`]));
    for (const row of rows) {
        for (const k of Object.keys(row)) assert.ok(allowed.has(k), `row key ${k} is a column or its spread`);
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.equal(row.timing, 'wall (submit → done)');
        assert.equal(row.cpuMs === null, row.n > CPU_MAX_N, `n = ${row.n}: CPU measured iff n ≤ ${CPU_MAX_N}`);
        assert.equal(row.naiveVsTiled, 0, 'mock readback: both zero');
    }
    assert.ok(Math.max(...device.buffers.map(b => b.size)) <= 4 * MAX_SIZE * MAX_SIZE);
    assert.deepEqual([ctx.measured.warmup, ctx.measured.repeats], [1, 5]);
    // A record from the mock rows validates (verify fails here: the mock reads back zeros)
    const record = buildResultRecord({ exampleId: 'matrix-multiplication', experiment: e, settings: { maxN: 1024 }, rows,
        measured: ctx.measured, source: { version: 'test', commit: 'local' }, generatedAt: '2026-11-01T00:00:00.000Z' });
    assert.deepEqual(validateResultRecord(record, schema).errors, []);
    assert.ok(record.interpretation.length >= 5);
    assert.ok(record.verification.some(c => c.status === 'fail'), 'zeros are not a product');

    const short = await runExperiment(e, { maxN: 256 }, mockCtx(mockDevice(), e.budgetMs));
    assert.deepEqual(short.map(r => r.n), [64, 128, 256]);
});

// Synthetic rows: times ∝ n³ (with a launch floor), errors as a multiple of √n·u
function synthetic({ errFactor = 0.8, naive = 1e-6, tiled = 2.5e-7, cpu = 4e-6, floor = 0, spread = 0.05, diff = 0 } = {}) {
    return SIZES.map(n => {
        const row = { n, timing: 'timestamps', sqrtNu: sqrtNu(n), naiveVsTiled: diff * sqrtNu(n) };
        for (const [path, c] of [['naive', naive], ['tiled', tiled], ['cpu', cpu]]) {
            const msv = path === 'cpu' && n > CPU_MAX_N ? null : floor + c * n ** 3;
            Object.assign(row, { [`${path}Ms`]: msv, [`${path}MsLo`]: msv && msv * (1 - spread), [`${path}MsHi`]: msv && msv * (1 + spread),
                [`${path}Err`]: msv === null ? null : errFactor * sqrtNu(n) });
        }
        row.speedup = row.naiveMs / row.tiledMs;
        row.speedupLo = row.naiveMsLo / row.tiledMsHi;
        row.speedupHi = row.naiveMsHi / row.tiledMsLo;
        for (const p of ['Upload', 'Dispatch', 'Readback']) { row[`naive${p}`] = 0.1; row[`tiled${p}`] = 0.1; }
        return row;
    });
}
const byClaim = (claims, start) => claims.find(c => c.claim.startsWith(start));

test('verify: passes within ERR_FACTOR·√n·u, fails beyond, not-checked without rows', () => {
    const pass = e.verify(synthetic());
    assert.equal(pass.length, 4);
    for (const c of pass) assert.equal(c.status, 'pass', c.metric);
    assert.ok(Math.abs(pass[0].value - 0.8) < 1e-9);
    const fail = e.verify(synthetic({ errFactor: 10, diff: 5 }));
    for (const c of fail) assert.equal(c.status, 'fail', c.metric);
    for (const c of e.verify([])) assert.equal(c.status, 'not-checked');
    // A run stopped after n = 512 still checks the sizes it has
    assert.ok(e.verify(synthetic().slice(0, 4)).every(c => c.status === 'pass'));
});

test('interpret: n³ data agree, a launch floor or one large size leaves the slope undecided, rankings from spreads', (t) => {
    const claims = e.interpret(synthetic());
    for (const c of claims) {
        for (const k of ['claim', 'predicted', 'measured']) assert.equal(typeof c[k], 'string');
        t.diagnostic(`${c.claim}: ${c.measured} → ${c.agrees}`);
    }
    for (const path of ['naive GPU', 'tiled GPU', 'CPU']) {
        const c = byClaim(claims, path);
        assert.equal(c.agrees, true, c.measured);
        assert.match(c.measured, /^3\.00 \(95% CI/);
    }
    assert.equal(byClaim(claims, 'Tiling').agrees, true);
    assert.match(byClaim(claims, 'Tiling').measured, /^4\.00× \(3\.6\d×–4\.4\d×\) at n = 1024/);
    assert.equal(byClaim(claims, 'The naive kernel').agrees, null);
    const gpu = byClaim(claims, 'The GPU beats');
    assert.equal(gpu.agrees, true);
    assert.match(gpu.measured, /GPU faster from n = 64 on; at n = 512/);

    // Launch-bound GPU: a 0.05 ms floor and tiny n³ term → fewer than 3 sizes ≥ FIT_MIN_MS → undecided
    const fast = e.interpret(synthetic({ naive: 1e-9, tiled: 1.5e-9, floor: 0.05 }));
    assert.equal(byClaim(fast, 'tiled GPU').agrees, null);
    const { used } = fitTimes(synthetic({ tiled: 1.5e-9, floor: 0.05 }), 'tiled');
    assert.deepEqual(used, [1024], `only n = 1024 reaches ${FIT_MIN_MS} ms`);
    assert.match(byClaim(fast, 'tiled GPU').note, /At least 3 sizes/);
    // Slope 2 (e.g. bandwidth-limited growth) disagrees
    const sq = synthetic().map(r => ({ ...r, tiledMs: 1e-3 * r.n ** 2, tiledMsLo: 0.99e-3 * r.n ** 2, tiledMsHi: 1.01e-3 * r.n ** 2 }));
    assert.equal(byClaim(e.interpret(sq), 'tiled GPU').agrees, false);
    // Tiling slower beyond the spread (software GPU) and CPU faster
    const sw = e.interpret(synthetic({ naive: 1e-5, tiled: 3e-5, cpu: 1e-6 }));
    assert.equal(byClaim(sw, 'Tiling').agrees, false);
    assert.equal(byClaim(sw, 'The GPU beats').agrees, false);
    assert.match(byClaim(sw, 'The GPU beats').measured, /^CPU faster at n = 512/);
    // Overlapping spreads → undecided
    assert.equal(byClaim(e.interpret(synthetic({ tiled: 1e-6, spread: 0.3 })), 'Tiling').agrees, null);
    // No rows: every claim undecided, nothing throws
    for (const c of e.interpret([])) assert.equal(c.agrees, null);
    assert.ok(FIT_MIN_MS > 0);
});
