// K-means Measure tab, CPU only: the compute-only API of js/examples/kmeans.js (measureKMeans) and the
// experiments in js/teach/experiments/kmeans.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing (its Stats
// readback is all zeros, i.e. "no label changed").
// Checked: a compute-only KMeans creates no render pipeline; the Params bytes match the WGSL struct;
// points, padded centroids and unassigned labels are uploaded; the caps are enforced; between() runs
// after every iteration; the float64 convergence helper agrees with step-by-step lloyd64Step; every
// sweep stays within its caps and every row carries the declared columns. The real results are checked
// on a software GPU in test/gpu/kmeans-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as mod from '../js/examples/kmeans.js';
import { measureKMeans, generatePoints, initCentroids, mulberry32, KMeans, MAX_K, MAX_N, MAX_ITERATIONS, TILE } from '../js/examples/kmeans.js';
import { lloyd64Step } from '../js/refs/kmeans.js';
import experiments, {
    POINT_COUNTS, CLUSTER_COUNTS, TRIALS, TIMED_ITERATIONS, MAX_CONVERGE_ITERATIONS, CONVERGE_N, CONVERGE_K, lloyd64Converge,
} from '../js/teach/experiments/kmeans.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const P = structLayout(moduleCode(mod, 'compute'), 'Params');
const paramsWrites = (device) => device.writes.filter(w => w.buffer.size === 16 && (w.buffer.usage & GPUBufferUsage.UNIFORM))
    .map(w => decodeStruct(w.bytes, P));

test('compute-only KMeans: no canvas context, no render pipeline', () => {
    let rendered = 0;
    const device = mockDevice();
    const counting = new Proxy(device, { get: (t, p) => (p === 'createRenderPipeline' ? () => { rendered++; } : t[p]) });
    const km = new KMeans(counting, null);
    assert.equal(rendered, 0);
    assert.equal(km.context, undefined);
    km.destroy();
});

test('measureKMeans on a mock device: Params bytes, uploads, iterations, result shape', async () => {
    const device = mockDevice();
    const n = 5000, k = 7;
    const points = generatePoints(n, 4, 0.05, 3);
    const centroids = initCentroids(points, n, k, 'random', mulberry32(3));
    let calls = 0;
    const r = await measureKMeans({ device }, { points, centroids, iterations: 4, untilConverged: false, readLabels: true,
        between: async () => { calls++; } });
    const params = paramsWrites(device);
    assert.equal(params.length, 4, 'one Params write per iteration');
    for (const p of params) assert.deepEqual(p, { n, k, numGroups: Math.ceil(n / TILE), pad0: 0 });
    assert.equal(calls, 4);
    const upload = (bytes) => device.writes.find(w => w.bytes.byteLength === bytes);
    assert.deepEqual(new Float32Array(upload(8 * n).bytes.buffer), points, 'points uploaded unchanged');
    const cent = new Float32Array(upload(8 * MAX_K).bytes.buffer);
    assert.deepEqual(cent.subarray(0, 2 * k), centroids);
    assert.ok(cent.subarray(2 * k).every(v => v === 0), 'padded to MAX_K');
    assert.ok(new Uint32Array(upload(4 * n).bytes.buffer).every(v => v === 0xffffffff), 'labels start unassigned');
    assert.equal(r.iterations, 4);
    assert.deepEqual(r.history.map(h => h.iteration), [1, 2, 3, 4]);
    assert.equal(r.centroids.length, 2 * k);
    assert.equal(r.labels.length, n);
    assert.equal(r.gpuMs, null, 'no timestamp-query on the mock');
    assert.ok(r.wallMs >= 0);

    // The mock's Stats read as zero changed labels: untilConverged stops after one iteration
    const c = await measureKMeans({ device: mockDevice() }, { points, centroids, iterations: 50 });
    assert.equal(c.iterations, 1);
    assert.equal(c.converged, true);
});

test('measureKMeans enforces the caps', async () => {
    const device = mockDevice();
    const cent = (k) => new Float32Array(2 * k);
    await assert.rejects(measureKMeans({ device }, { points: new Float32Array(2), centroids: cent(2) }), RangeError);
    await assert.rejects(measureKMeans({ device }, { points: new Float32Array(2 * (MAX_N + 1)), centroids: cent(2) }), RangeError);
    await assert.rejects(measureKMeans({ device }, { points: new Float32Array(2000), centroids: cent(1) }), RangeError);
    await assert.rejects(measureKMeans({ device }, { points: new Float32Array(2000), centroids: cent(MAX_K + 1) }), RangeError);
    await assert.rejects(measureKMeans({ device }, { points: new Float32Array(2000), centroids: cent(4), iterations: MAX_ITERATIONS + 1 }), RangeError);
    await assert.rejects(measureKMeans({ device }, { points: new Float32Array(2000), centroids: cent(4), iterations: 0 }), RangeError);
});

test('lloyd64Converge equals repeated lloyd64Step and stops when no label changes', () => {
    const n = 3000, k = 4;
    const points = generatePoints(n, 4, 0.05, 11);
    const init = initCentroids(points, n, k, 'kmeans++', mulberry32(11));
    const r = lloyd64Converge(points, n, k, init);
    assert.ok(r.converged && r.iterations > 1 && r.iterations < MAX_CONVERGE_ITERATIONS);
    let cent = init, prev = null, steps = 0, s;
    for (;;) {
        s = lloyd64Step(points, n, k, cent);
        steps++;
        if (prev && s.labels.every((l, i) => l === prev[i])) break;
        prev = s.labels;
        cent = s.next;
    }
    assert.equal(r.iterations, steps);
    assert.equal(r.inertia, s.inertia);
    assert.deepEqual(r.next, s.next);
    const capped = lloyd64Converge(points, n, k, init, 1);
    assert.equal(capped.converged, false);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(POINT_COUNTS.every(n => n <= MAX_N) && CLUSTER_COUNTS.every(k => k >= 2 && k <= MAX_K));
    assert.ok(TIMED_ITERATIONS <= MAX_ITERATIONS && MAX_CONVERGE_ITERATIONS <= MAX_ITERATIONS);
    assert.ok(CONVERGE_N <= MAX_N && CONVERGE_K <= MAX_K && TRIALS.length <= 5);
    for (const e of experiments) {
        assert.ok(e.budgetMs <= 20000);
        for (const s of e.settings ?? []) for (const o of s.options) if (typeof o.value === 'number') assert.ok(o.value <= MAX_N);
    }
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [
        ['points', { k: 8 }, POINT_COUNTS.length],
        ['clusters', { n: 100000 }, CLUSTER_COUNTS.length],
        ['convergence', { init: 'random' }, TRIALS.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys`);
        const params = paramsWrites(device);
        assert.ok(params.length > 0 && params.every(p => p.n <= MAX_N && p.k <= MAX_K));
    }
});
