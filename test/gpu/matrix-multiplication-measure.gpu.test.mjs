// Matrix multiplication compute-only API (measureMatmul in js/examples/matrix-multiplication.js) and its
// Measure experiment, on a software GPU — CI only, see harness.mjs. Checked against the float64 product
// (js/refs/basic.js). The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureMatmul } from '../../js/examples/matrix-multiplication.js';
import { matmulF64 } from '../../js/refs/basic.js';
import experiments, { makeMatrix, UNIT_ROUNDOFF } from '../../js/teach/experiments/matrix-multiplication.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureMatmul: non-square 37×23 · 23×19 vs float64', async (t) => {
    const [M, N, K] = [37, 23, 19];
    const A = makeMatrix(M, N, 1), B = makeMatrix(N, K, 2);
    const r = await measureMatmul(gpuDevice, { A, B, M, N, K });
    const ref = matmulF64(A, B, M, N, K);
    let maxAbs = 0;
    for (let i = 0; i < M * K; i++) maxAbs = Math.max(maxAbs, Math.abs(r.C[i] - ref[i]));
    t.diagnostic(`max |C − C64| = ${maxAbs.toExponential(2)}`);
    assert.ok(maxAbs < 1e-5);
});

test('experiment scaling: error below n·u at every size; GFLOPS reported', async (t) => {
    const [e] = experiments;
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(makeTestContext(gpuDevice), {})) rows.push(row);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: n wallMs GFLOPS(wall) maxRelError (√n·u): ` + rows.map(r => [r.n,
        r.wallMs.toFixed(2), r.gflopsWall?.toFixed(2), r.maxRelError.toExponential(2), (Math.sqrt(r.n) * UNIT_ROUNDOFF).toExponential(2)].join(' ')).join('; '));
    assert.deepEqual(rows.map(r => r.n), [64, 128, 256, 512, 1024]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, c.key);
        assert.ok(row.maxRelError > 0 && row.maxRelError < row.bound, `n = ${row.n}: ${row.maxRelError}`);
        assert.ok(row.wallMs > 0);
    }
});
