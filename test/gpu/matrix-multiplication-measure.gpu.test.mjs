// Matrix multiplication compute-only API (measureMatmul in js/examples/matrix-multiplication.js: the naive
// and the tiled kernel) and its Measure experiment, on a software GPU — CI only, see harness.mjs. Checked
// against the float64 product (js/refs/basic.js): both kernels on non-square, ragged sizes (every element
// written, within ERR_FACTOR·√N·u), naive == tiled, and a full experiment run whose result record matches
// schema v1 and passes verify(). The printed numbers are the CI baselines quoted on the Understand page;
// software-GPU timings say nothing about a real GPU.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { device } from './harness.mjs';
import { measureMatmul } from '../../js/examples/matrix-multiplication.js';
import { matmulF64 } from '../../js/refs/basic.js';
import experiments, { makeMatrix, sqrtNu, ERR_FACTOR } from '../../js/teach/experiments/matrix-multiplication.js';
import { makeTestContext } from '../../js/teach/measure.js';
import { buildResultRecord, validateResultRecord, SCHEMA_PATH } from '../../js/teach/result-record.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const schema = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../..', SCHEMA_PATH), 'utf8'));

test('measureMatmul naive and tiled: ragged non-square sizes vs float64, naive == tiled', async (t) => {
    for (const [M, N, K] of [[37, 23, 19], [100, 70, 45], [1, 33, 2], [129, 257, 65], [256, 256, 256]]) {
        const A = makeMatrix(M, N, M + N), B = makeMatrix(N, K, K + 1);
        const ref = matmulF64(A, B, M, N, K);
        const out = {};
        for (const kernel of ['naive', 'tiled']) {
            const r = await measureMatmul(gpuDevice, { A, B, M, N, K, kernel, warmup: 0, repeats: 1 });
            assert.equal(r.C.length, M * K);
            let maxRel = 0;
            for (let i = 0; i < M * K; i++) maxRel = Math.max(maxRel, Math.abs(r.C[i] - ref[i]) / ref[i]);
            assert.ok(r.C.every(Number.isFinite), `${kernel}: every element written`);
            assert.ok(maxRel < ERR_FACTOR * sqrtNu(N), `${kernel} ${M}x${N}x${K}: ${maxRel} vs ${ERR_FACTOR}·√N·u`);
            out[kernel] = { C: r.C, maxRel, phases: r.phases };
        }
        let diff = 0, same = 0;
        for (let i = 0; i < M * K; i++) {
            diff = Math.max(diff, Math.abs(out.naive.C[i] - out.tiled.C[i]) / ref[i]);
            if (out.naive.C[i] === out.tiled.C[i]) same++;
        }
        t.diagnostic(`${M}x${N}*${N}x${K}: max rel. error naive ${out.naive.maxRel.toExponential(2)}, tiled ${out.tiled.maxRel.toExponential(2)} `
            + `(${(out.tiled.maxRel / sqrtNu(N)).toFixed(2)} √N·u); naive vs tiled max ${diff.toExponential(2)}, ${same}/${M * K} identical`);
        assert.ok(diff <= ERR_FACTOR * sqrtNu(N), `naive vs tiled ${diff}`);
    }
});

test('experiment cpu-naive-tiled (n ≤ 1,024): record matches schema v1, verify passes, interpretation present', async (t) => {
    const [e] = experiments;
    const settings = { maxN: 1024 };
    const ctx = makeTestContext(gpuDevice);
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(ctx, settings)) {
        rows.push(row);
        t.diagnostic(`n = ${row.n} after ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    }
    const record = buildResultRecord({ exampleId: 'matrix-multiplication', experiment: e, settings, rows, outcome: { status: 'done' },
        measured: ctx.measured, gpuDevice, source: { version: 'ci', commit: 'local', url: null }, userAgent: 'node', budgetMs: e.budgetMs });
    assert.deepEqual(validateResultRecord(record, schema).errors, [], 'record matches schema v1');
    assert.deepEqual(rows.map(r => r.n), [64, 128, 256, 512, 1024]);
    const f = (v, d = 2) => (typeof v === 'number' ? (Math.abs(v) < 1e-2 && v !== 0 ? v.toExponential(2) : v.toFixed(d)) : '—');
    const range = (r, k) => `${f(r[k])} (${f(r[`${k}Lo`])}–${f(r[`${k}Hi`])})`;
    t.diagnostic(`timing: ${rows[0].timing}; protocol ${record.protocol.warmup} + ${record.protocol.repeats}`);
    for (const r of rows) {
        t.diagnostic(`n=${r.n} GFLOPS cpu ${range(r, 'cpuGflops')} naive ${range(r, 'naiveGflops')} tiled ${range(r, 'tiledGflops')} | `
            + `ms cpu ${range(r, 'cpuMs')} naive ${range(r, 'naiveMs')} tiled ${range(r, 'tiledMs')} | speedup ${range(r, 'speedup')}`);
        t.diagnostic(`n=${r.n} err/(√n·u) cpu ${f(r.cpuErr / r.sqrtNu)} naive ${f(r.naiveErr / r.sqrtNu)} tiled ${f(r.tiledErr / r.sqrtNu)} `
            + `naive-vs-tiled ${f(r.naiveVsTiled)} | max rel err cpu ${f(r.cpuErr)} naive ${f(r.naiveErr)} tiled ${f(r.tiledErr)}`);
        t.diagnostic(`n=${r.n} phases ms naive ${['Compile', 'Upload', 'Dispatch', 'Readback'].map(p => f(r[`naive${p}`], 3)).join(' / ')}`
            + ` tiled ${['Compile', 'Upload', 'Dispatch', 'Readback'].map(p => f(r[`tiled${p}`], 3)).join(' / ')}`);
    }
    for (const c of record.verification) t.diagnostic(`verify ${c.status}: ${c.metric} = ${c.value} (${c.tolerance})`);
    for (const c of record.interpretation) t.diagnostic(`interpret ${c.agrees}: ${c.claim} | ${c.measured}`);
    for (const c of record.verification) assert.equal(c.status, 'pass', `${c.metric}: ${c.value} (required ${c.tolerance})`);
    assert.equal(record.interpretation.length, 6);
    for (const col of record.columns.filter(c => c.range)) {
        for (const row of rows) {
            if (row[col.key] === null) continue;
            assert.ok(row[`${col.key}Lo`] <= row[col.key] && row[col.key] <= row[`${col.key}Hi`], `${col.key}: q1 ≤ median ≤ q3`);
        }
    }
    for (const r of rows) {
        assert.ok(r.naiveMs > 0 && r.tiledMs > 0, `n = ${r.n}: GPU times`);
        assert.equal(r.cpuMs === null, r.n > 512);
    }
});
