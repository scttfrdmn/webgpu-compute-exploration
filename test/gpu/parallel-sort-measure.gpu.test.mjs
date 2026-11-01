// Parallel sort compute-only API (measureSort in js/examples/parallel-sort.js) and the Measure experiment
// built on it (js/teach/experiments/parallel-sort.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/parallel-sort.gpu.test.mjs, which rebuilds the dispatch sequence with explicit
// layouts, this runs the example's own runBenchmark() (pipelines, dynamic-offset bind groups, one compute
// pass per sort, readback, comparison with TypedArray.prototype.sort) on a ParallelSort without a canvas,
// exactly as the Measure tab does in a browser. Every sort must be bit-identical to the CPU sort.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureSort } from '../../js/examples/parallel-sort.js';
import experiments from '../../js/teach/experiments/parallel-sort.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureSort: every algorithm and key type matches the CPU sort (non-power-of-two and power-of-two N)', async (t) => {
    for (const n of [5000, 65536]) {
        for (const dataType of ['u32', 'f32']) {
            for (const algorithm of ['bitonic', 'bitonic-shared', 'radix']) {
                const r = await measureSort(gpuDevice, { algorithm, n, dataType });
                t.diagnostic(`${algorithm} ${dataType} N=${n}: mismatches ${r.mismatches}, inversions ${r.inversions}, dispatches ${r.dispatches}, `
                    + `best wall ${r.bestMs.toFixed(2)} ms, CPU ${r.cpuMs.toFixed(2)} ms`);
                assert.equal(r.mismatches, 0, `${algorithm} ${dataType} N=${n}`);
                assert.equal(r.inversions, 0);
                assert.equal(r.padded, algorithm === 'radix' ? n : 2 ** Math.ceil(Math.log2(n)));
                assert.equal(r.gpuTimes.length, 3);
            }
        }
    }
});

test('experiment scaling (u32 and f32, N ≤ 262,144): every row verified, columns present', async (t) => {
    const e = experiments[0];
    for (const dataType of ['u32', 'f32']) {
        const rows = [];
        const t0 = performance.now();
        for await (const row of e.run(makeTestContext(gpuDevice), { dataType, maxN: 262144 })) rows.push(row);
        const seconds = (performance.now() - t0) / 1000;
        t.diagnostic(`${dataType} ${seconds.toFixed(1)} s: N bitonicWall sharedWall radixWall cpuMs dispatches verified: `
            + rows.map(r => `${r.n} ${r.bitonicWall.toFixed(1)} ${r.sharedWall.toFixed(1)} ${r.radixWall.toFixed(1)} ${r.cpuMs.toFixed(2)} [${r.dispatches}] ${r.verified}`).join('; '));
        assert.deepEqual(rows.map(r => r.n), [16384, 65536, 262144]);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.equal(row.verified, 'yes', `N=${row.n}`);
            assert.ok(row.bitonicWall > 0 && row.sharedWall > 0 && row.radixWall > 0 && row.cpuMs >= 0);
        }
    }
});
