// K-means compute-only API (measureKMeans in js/examples/kmeans.js) and the Measure experiments built on
// it (js/teach/experiments/kmeans.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/kmeans.gpu.test.mjs, which rebuilds the dispatch sequence with explicit layouts, this
// runs the example's own pipelines, bind group and encodeIteration (with its Stats copy) through a
// KMeans without a canvas, exactly as the Measure tab does in a browser. Checked against an independent
// float64 Lloyd run from the same initial centroids (js/refs/kmeans.js lloyd64Step).
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureKMeans, generatePoints, initCentroids, mulberry32 } from '../../js/examples/kmeans.js';
import { lloyd64Step } from '../../js/refs/kmeans.js';
import experiments, { lloyd64Converge } from '../../js/teach/experiments/kmeans.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureKMeans: converges to the float64 Lloyd result; Stats per iteration are consistent', async (t) => {
    for (const [n, k, blobs, seed] of [[20000, 4, 4, 3], [50001, 9, 6, 8]]) {
        const points = generatePoints(n, blobs, 0.05, seed);
        const centroids = initCentroids(points, n, k, 'kmeans++', mulberry32(seed));
        const r = await measureKMeans(gpuDevice, { points, centroids, iterations: 100, readLabels: true });
        const cpu = lloyd64Converge(points, n, k, centroids);
        let differ = 0;
        for (let i = 0; i < n; i++) if (r.labels[i] !== cpu.labels[i]) differ++;
        let shift = 0;
        for (let j = 0; j < k; j++) shift = Math.max(shift, Math.hypot(r.centroids[2 * j] - cpu.next[2 * j], r.centroids[2 * j + 1] - cpu.next[2 * j + 1]));
        const atGpu = lloyd64Step(points, n, k, r.centroids);
        const sumErr = Math.abs(r.inertia - atGpu.inertia) / atGpu.inertia;
        t.diagnostic(`n=${n} k=${k}: GPU ${r.iterations} iterations (converged ${r.converged}), CPU ${cpu.iterations}; labels differ ${differ}; `
            + `max centroid shift ${shift.toExponential(2)}; inertia GPU ${r.inertia.toFixed(5)} CPU ${cpu.inertia.toFixed(5)}; `
            + `f32 sum error ${sumErr.toExponential(2)}; changed ${r.history.map(h => h.changed).join(',')}; wall ${r.wallMs.toFixed(2)} ms/iteration`);
        assert.ok(r.converged, 'converged within 100 iterations');
        assert.equal(r.history[0].changed, n, 'every label changes in the first iteration');
        assert.equal(r.history.at(-1).changed, 0);
        assert.ok(r.history.every(h => h.empty === 0));
        assert.ok(differ <= n * 1e-3, `${differ} labels differ`);
        assert.ok(shift < 1e-3, `centroid shift ${shift}`);
        assert.ok(sumErr < 1e-4, `inertia sum error ${sumErr}`);
        assert.ok(Math.abs(atGpu.inertia - cpu.inertia) / cpu.inertia < 1e-4);
        // inertia never increases (Lloyd's monotonicity), up to f32 rounding
        for (let i = 2; i < r.history.length; i++) assert.ok(r.history[i].inertia <= r.history[i - 1].inertia * (1 + 1e-5));
    }
});

const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(makeTestContext(gpuDevice), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toPrecision(3)) : String(v);
}).join(' ')).join('; ');

test('experiment convergence (k-means++ and random): same clustering as float64', async (t) => {
    const e = byId.convergence;
    for (const init of ['kmeans++', 'random']) {
        const { rows, seconds } = await runAll(e, { init });
        t.diagnostic(`${init} ${seconds.toFixed(1)} s: trial gpuIt cpuIt gpuInertia cpuInertia gap sumError differ shift: `
            + show(rows, ['trial', 'gpuIterations', 'cpuIterations', 'gpuInertia', 'cpuInertia', 'inertiaGap', 'sumError', 'labelsDiffer', 'centroidShift']));
        assert.equal(rows.length, 5);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.ok(row.gpuIterations > 1, 'GPU converged');
            assert.ok(Math.abs(row.inertiaGap) < 1e-3 && row.sumError < 1e-4, `trial ${row.trial}`);
        }
    }
});

test('experiments points and clusters: rows for every size, CPU cost grows with N·K', async (t) => {
    const p = await runAll(byId.points, { k: 8 });
    t.diagnostic(`points k=8 ${p.seconds.toFixed(1)} s: n wallMs cpuMs: ${show(p.rows, ['n', 'wallMs', 'cpuMs', 'speedup'])}`);
    assert.deepEqual(p.rows.map(r => r.n), [10000, 30000, 100000, 300000, 500000]);
    const c = await runAll(byId.clusters, { n: 100000 });
    t.diagnostic(`clusters n=100k ${c.seconds.toFixed(1)} s: k wallMs cpuMs: ${show(c.rows, ['k', 'wallMs', 'cpuMs', 'speedup'])}`);
    assert.deepEqual(c.rows.map(r => r.k), [2, 4, 8, 12, 16]);
    for (const row of [...p.rows, ...c.rows]) {
        for (const col of byId.points.columns) assert.ok(col.key in row, `column ${col.key}`);
        assert.ok(row.wallMs > 0 && row.cpuMs > 0);
    }
    assert.ok(p.rows.at(-1).cpuMs > 5 * p.rows[0].cpuMs, 'CPU step grows with N');
});
