// Parallel sort compute-only API (measureSort in js/examples/parallel-sort.js) and the Measure experiment
// built on it (js/teach/experiments/parallel-sort.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/parallel-sort.gpu.test.mjs, which rebuilds the dispatch sequence with explicit
// layouts, this runs the example's own runBenchmark() (pipelines, dynamic-offset bind groups, one compute
// pass per sort, readback, comparison with TypedArray.prototype.sort) on a ParallelSort without a canvas,
// exactly as the Measure tab does in a browser. Every sort must be bit-identical to the CPU sort; the
// experiment's result record must validate against schema v1, pass verify() and carry the interpretation.
// The printed numbers are the CI baselines quoted in the experiment file and the Understand page.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { device } from './harness.mjs';
import { measureSort, computeSorter, generateKeys } from '../../js/examples/parallel-sort.js';
import experiments, { COUNTS, PROTOCOL } from '../../js/teach/experiments/parallel-sort.js';
import { makeTestContext } from '../../js/teach/measure.js';
import { buildResultRecord, validateResultRecord, SCHEMA_PATH } from '../../js/teach/result-record.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const schema = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../..', SCHEMA_PATH), 'utf8'));

test('measureSort: every algorithm and key type matches the CPU sort (non-power-of-two and power-of-two N)', async (t) => {
    for (const n of [5000, 65536]) {
        for (const dataType of ['u32', 'f32']) {
            for (const algorithm of ['bitonic', 'bitonic-shared', 'radix']) {
                const r = await measureSort(gpuDevice, { algorithm, n, dataType, runs: 2 });
                t.diagnostic(`${algorithm} ${dataType} N=${n}: mismatches ${r.mismatches}, inversions ${r.inversions}, dispatches ${r.dispatches}, `
                    + `wall ${r.wallMs.toFixed(2)} ms, CPU ${r.cpuMs.toFixed(2)} ms, phases ${Object.entries(r.phases).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' ')}`);
                assert.equal(r.mismatches, 0, `${algorithm} ${dataType} N=${n}`);
                assert.equal(r.inversions, 0);
                assert.equal(r.padded, algorithm === 'radix' ? n : 2 ** Math.ceil(Math.log2(n)));
                assert.equal(r.gpuTimes.length, 2);
                assert.deepEqual(Object.keys(r.phases).sort(), ['compile', 'dispatch', 'readback', 'upload']);
            }
        }
    }
    // Given keys and a CPU result, with a shared sorter (pipelines compiled once)
    const sorter = computeSorter(gpuDevice);
    try {
        const keys = generateKeys(10000, 'f32', 9);
        const expected = new Uint32Array(keys.slice().sort().buffer);
        for (const algorithm of ['bitonic', 'bitonic-shared', 'radix']) {
            const r = await measureSort(gpuDevice, { algorithm, n: keys.length, dataType: 'f32', keys, expected, sorter });
            assert.equal(r.mismatches, 0, algorithm);
            assert.equal(r.cpuMs, null);
        }
    } finally {
        sorter.destroy();
    }
});

const fmt = (v) => (v === null || v === undefined ? '—' : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const cell = (r, k) => `${fmt(r[k])} (${fmt(r[`${k}Lo`])}–${fmt(r[`${k}Hi`])})`;

for (const [dataType, maxN] of [['u32', 1048576], ['f32', 262144]]) {
    test(`experiment scaling (${dataType}, N ≤ ${maxN.toLocaleString('en-US')}): schema-valid record, verified, interpreted`, async (t) => {
        const e = experiments[0];
        const settings = { dataType, maxN };
        const ctx = makeTestContext(gpuDevice);
        const rows = [];
        const t0 = performance.now();
        for await (const row of e.run(ctx, settings)) rows.push(row);
        const seconds = (performance.now() - t0) / 1000;
        const record = buildResultRecord({ exampleId: 'parallel-sort', experiment: e, settings, rows, outcome: { status: 'done' },
            measured: ctx.measured, gpuDevice, source: { version: 'ci', commit: 'local', url: null },
            userAgent: 'node', budgetMs: e.budgetMs });
        assert.deepEqual(validateResultRecord(record, schema).errors, [], 'record matches schema v1');

        t.diagnostic(`${dataType} ${seconds.toFixed(1)} s, timestamps ${gpuDevice.timestamps}; per N: wall ms median (q1–q3) bitonic / workgroup / radix / CPU; upload / readback (radix); dispatches`);
        for (const r of rows) {
            t.diagnostic(`  N=${r.n} ×${r.repeats}: ${cell(r, 'bitonicWall')} / ${cell(r, 'sharedWall')} / ${cell(r, 'radixWall')} / ${cell(r, 'cpuMs')}; `
                + `GPU ${fmt(r.bitonicGpu)} / ${fmt(r.sharedGpu)} / ${fmt(r.radixGpu)}; ${cell(r, 'uploadMs')} / ${cell(r, 'readbackMs')}; setup ${fmt(r.setupMs)}; `
                + `pipelines ${fmt(r.pipelineMs)}; [${r.dispatches}]; wrong ${r.bitonicWrong}/${r.sharedWrong}/${r.radixWrong}`);
        }
        for (const c of record.verification) t.diagnostic(`verify: ${c.status} ${c.metric} = ${c.value} (${c.reference})`);
        for (const c of record.interpretation) t.diagnostic(`interpret: [${c.agrees}] ${c.claim}: predicted ${c.predicted}; measured ${c.measured}`);

        assert.deepEqual(rows.map(r => r.n), COUNTS.filter(n => n <= maxN));
        assert.deepEqual([record.protocol.warmup, record.protocol.repeats, record.protocol.statistic, record.protocol.spread],
            [PROTOCOL.warmup, PROTOCOL.repeats, 'median', 'iqr']);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.ok(row.bitonicWall > 0 && row.sharedWall > 0 && row.radixWall > 0 && row.cpuMs > 0);
            for (const k of ['bitonicWall', 'sharedWall', 'radixWall', 'cpuMs', 'uploadMs', 'readbackMs']) {
                assert.ok(row[`${k}Lo`] <= row[k] && row[k] <= row[`${k}Hi`], `${k}: q1 ≤ median ≤ q3 at N = ${row.n}`);
            }
        }
        // Verification: every sort bit-identical (and, for f32, the key map round trip)
        const expected = dataType === 'f32' ? ['pass', 'pass', 'pass', 'pass', 'pass'] : ['pass', 'pass', 'pass', 'not-checked'];
        assert.deepEqual(record.verification.map(c => c.status), expected, JSON.stringify(record.verification));
        // Interpretation: four slopes, three crossovers, one ranking, each with a measured value
        assert.equal(record.interpretation.length, 8);
        for (const c of record.interpretation) {
            assert.ok(c.measured !== '—', `${c.claim} has a measured value`);
            assert.ok([true, false, null].includes(c.agrees));
        }
        for (const c of record.interpretation.slice(0, 4)) assert.match(c.measured, /95% CI/);
    });
}
