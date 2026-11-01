// Parallel sort Measure tab, CPU only: the compute-only API of js/examples/parallel-sort.js (measureSort)
// and the experiment in js/teach/experiments/parallel-sort.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: a compute-only ParallelSort creates no render pipeline or canvas context; the uniform bytes
// measureSort writes match the WGSL structs; the key buffers are sized for the padded length; the caps
// are enforced; the sweep respects its caps and the "Largest N" setting, and every row carries the
// declared columns. The real sorts are checked on a software GPU in test/gpu/parallel-sort-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureSort, MEASURE_MAX_KEYS, ParallelSort, sortInternals as S, wgslModules } from '../js/examples/parallel-sort.js';
import experiments, { COUNTS, MAX_N, ALGORITHMS } from '../js/teach/experiments/parallel-sort.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const L = {
    bitonic: structLayout(wgslModules().find(m => m.name === 'bitonic').code, 'BitonicParams'),
    radix: structLayout(wgslModules().find(m => m.name === 'radix').code, 'RadixParams'),
};
const isUniform = b => (b.usage & GPUBufferUsage.UNIFORM) !== 0;

test('compute-only ParallelSort: no canvas context, no render pipeline', () => {
    const device = mockDevice();
    let rendered = 0;
    const counting = new Proxy(device, { get: (t, p) => (p === 'createRenderPipeline' ? () => { rendered++; } : t[p]) });
    const sorter = new ParallelSort(counting, null);
    assert.equal(rendered, 0);
    assert.equal(sorter.context, undefined);
    assert.ok(sorter.bitonicPipelines && sorter.radixPipelines, 'compute pipelines exist');
    sorter.destroy();
});

test('measureSort on a mock device: RadixParams / BitonicParams bytes, padded buffers, result shape', async () => {
    let device = mockDevice();
    const n = 5000;
    const r = await measureSort({ device }, { algorithm: 'radix', n, dataType: 'u32' });
    const radix = device.writes.find(w => isUniform(w.buffer) && w.buffer.size === S.RADIX_PASSES * 256);
    for (let p = 0; p < S.RADIX_PASSES; p++) {
        assert.deepEqual(decodeStruct(radix.bytes, L.radix, 256 * p), { shift: 4 * p, count: n, numTiles: Math.ceil(n / S.RADIX_TILE), pad0: 0 });
    }
    assert.ok(device.buffers.filter(b => b.size === 4 * n && (b.usage & GPUBufferUsage.STORAGE)).length >= 2, 'keysA/keysB of n keys');
    assert.deepEqual({ algorithm: r.algorithm, n: r.n, padded: r.padded, dataType: r.dataType, dispatches: r.dispatches },
        { algorithm: 'radix', n, padded: n, dataType: 'u32', dispatches: 24 });
    assert.equal(r.gpuTimes.length, 3, 'three timed sorts');
    assert.equal(r.bestKernelMs, null, 'no timestamp-query on the mock');
    assert.ok(r.cpuMs >= 0 && r.bestMs >= 0);

    device = mockDevice();
    const b = await measureSort({ device }, { algorithm: 'bitonic', n, dataType: 'f32' });
    const stages = S.bitonicSchedule(8192, false);
    const w = device.writes.find(x => isUniform(x.buffer) && x.buffer.size === stages.length * 256);
    stages.forEach((stage, s) => assert.deepEqual(decodeStruct(w.bytes, L.bitonic, 256 * s), { k: stage.k, j: stage.j, threads: 4096, pad0: 0 }));
    assert.ok(device.buffers.some(x => x.size === 4 * 8192 && (x.usage & GPUBufferUsage.STORAGE)), 'keys padded to 8,192');
    assert.equal(b.padded, 8192);
    assert.equal(b.dispatches, stages.length + 2, 'f32 adds encode + decode');
});

test('measureSort enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measureSort({ device }, { algorithm: 'quick', n: 1024 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, dataType: 'f64' }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 0 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'bitonic', n: MEASURE_MAX_KEYS + 1 }), RangeError);
    assert.equal(MEASURE_MAX_KEYS, S.RADIX_MAX_KEYS);
});

test('experiment sweep stays within the caps', () => {
    assert.ok(COUNTS.every(n => n <= MAX_N) && MAX_N <= MEASURE_MAX_KEYS);
    assert.deepEqual(ALGORITHMS, ['bitonic', 'bitonic-shared', 'radix']);
    for (const e of experiments) {
        assert.ok(e.budgetMs <= 20000);
        for (const s of e.settings ?? []) for (const o of s.options) if (typeof o.value === 'number') assert.ok(o.value <= MAX_N);
    }
});

test('experiment runs on a mock device: one row per N up to "Largest N", declared columns', async (t) => {
    for (const settings of [{ dataType: 'u32', maxN: 262144 }, { dataType: 'f32', maxN: 262144 }]) {
        const experiment = experiments[0];
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.deepEqual(rows.map(r => r.n), COUNTS.filter(n => n <= 262144));
        for (const row of rows) {
            assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort());
            assert.match(row.dispatches, /^\d+ \/ \d+ \/ \d+$/);
            assert.match(row.verified, /^NO/, 'the mock reads back zeros, which must not verify');
        }
        const counts = device.writes.filter(w => isUniform(w.buffer) && w.buffer.size === S.RADIX_PASSES * 256)
            .map(w => decodeStruct(w.bytes, L.radix).count);
        assert.ok(counts.length === rows.length && Math.max(...counts) <= 262144);
    }
});
