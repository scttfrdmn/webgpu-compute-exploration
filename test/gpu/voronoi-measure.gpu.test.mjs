// Voronoi / JFA compute-only API (measureJFA in js/examples/voronoi-jfa.js) and the Measure experiments
// built on it (js/teach/experiments/voronoi.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/voronoi-jfa.gpu.test.mjs, which rebuilds the dispatch sequence with explicit
// layouts, this runs the example's own pipelines, dynamic-offset bind groups, encodeCompute and
// exactCheck through a VoronoiJFA without a canvas, exactly as the Measure tab does in a browser.
// Checked against the float64 brute force of js/refs/voronoi.js: the GPU exact check and the CPU check
// count the same wrong pixels (up to f32 near-ties), and the error rates stay within the suites' bounds.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureJFA, jfaSteps } from '../../js/examples/voronoi-jfa.js';
import { errorsVsBrute, MAX_ERROR_RATE, MAX_ERROR_RATE_PLUS_ONE } from '../../js/refs/voronoi.js';
import experiments, { makeSeeds } from '../../js/teach/experiments/voronoi.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureJFA: GPU exact check agrees with the float64 brute force (3 metrics, JFA and JFA+1)', async (t) => {
    const w = 320, h = 240;
    for (const layout of ['random', 'clusters']) {
        const seeds = makeSeeds(200, w, h, layout, 5);
        for (const metric of [0, 1, 2]) {
            for (const plusOne of [false, true]) {
                const r = await measureJFA(gpuDevice, { width: w, height: h, seeds, metric, plusOne, repeats: 1, check: true, readIds: true });
                assert.equal(r.passes, jfaSteps(w, h, plusOne).length);
                const c = errorsVsBrute(r.ids, w, h, seeds, metric);
                t.diagnostic(`${layout} metric ${metric}${plusOne ? ' +1' : ''}: GPU check wrong ${r.wrong}, differ ${r.differ}; `
                    + `float64 wrong ${c.wrong} (max excess ${c.maxExcess.toFixed(3)} px)`);
                assert.ok(r.ids.every(id => id < 200), 'every pixel has a seed');
                assert.ok(Math.abs(r.wrong - c.wrong) <= 2, 'GPU and CPU checks agree up to f32 near-ties');
                assert.ok(c.wrong / (w * h) < (plusOne ? MAX_ERROR_RATE_PLUS_ONE : MAX_ERROR_RATE));
            }
        }
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
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');

test('experiment error-rate (3 metrics, random; Euclidean clusters): JFA+1 ≤ JFA, rates within bounds', async (t) => {
    const e = byId['error-rate'];
    for (const [metric, layout] of [[0, 'random'], [1, 'random'], [2, 'random'], [0, 'clusters']]) {
        const { rows, seconds } = await runAll(e, { metric, layout });
        t.diagnostic(`metric ${metric} ${layout} ${seconds.toFixed(1)} s: seeds jfaWrong plusOneWrong cpuWrong jfaRate plusOneRate maxExcess: `
            + show(rows, ['seeds', 'jfaWrong', 'plusOneWrong', 'cpuWrong', 'jfaRate', 'plusOneRate', 'maxExcess']));
        assert.deepEqual(rows.map(r => r.seeds), [16, 64, 256, 1024]);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.ok(row.plusOneWrong <= row.jfaWrong, `seeds ${row.seeds}: JFA+1 ${row.plusOneWrong} > JFA ${row.jfaWrong}`);
            assert.ok(row.jfaRate < MAX_ERROR_RATE && row.plusOneRate < MAX_ERROR_RATE_PLUS_ONE);
            if (row.cpuWrong !== null) assert.ok(Math.abs(row.cpuWrong - row.jfaWrong) <= 3, 'GPU and float64 checks agree');
        }
    }
});

test('experiment resolution: one more pass per doubling of the side', async (t) => {
    const e = byId.resolution;
    const { rows, seconds } = await runAll(e, { plusOne: false });
    t.diagnostic(`${seconds.toFixed(1)} s: side passes wallMs: ${show(rows, ['side', 'passes', 'wallMs'])}`);
    assert.deepEqual(rows.map(r => r.side), [256, 512, 1024, 2048]);
    assert.deepEqual(rows.map(r => r.passes), [8, 9, 10, 11]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.ok(rows[3].wallMs > rows[0].wallMs, 'the 2048² grid takes longer than 256²');
});
