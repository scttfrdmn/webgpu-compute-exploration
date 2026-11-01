// Boids compute-only API (measureBoids in js/examples/boids.js) and its Measure experiments, on a software
// GPU — CI only, see harness.mjs. Checked against the float64 step of js/refs/boids.js for every boid and
// behaviour mode, and the emergence of order. The printed numbers are the CI baselines.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureBoids } from '../../js/examples/boids.js';
import experiments, { randomBoids, stepError, sampleTargets } from '../../js/teach/experiments/boids.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('one GPU step vs the float64 step, every boid, all behaviour modes', async (t) => {
    const n = 1500;
    const boids = randomBoids(n, 5);
    for (const behaviorMode of [0, 1, 2, 3]) {
        const r = await measureBoids(gpuDevice, { boids, steps: 1, samples: 1, behaviorMode });
        const err = stepError(boids, r.snapshots[1].boids, n, r.params, sampleTargets(n, n));
        t.diagnostic(`mode ${behaviorMode}: max position difference ${err.toExponential(2)} px`);
        assert.ok(err < 1e-3, `mode ${behaviorMode}: ${err}`);
    }
});

test('experiment polarization (N = 500): order emerges from random headings', async (t) => {
    const e = experiments.find(x => x.id === 'polarization');
    const rows = [];
    for await (const row of e.run(makeTestContext(gpuDevice), { n: 500 })) rows.push(row);
    t.diagnostic('step polarization: ' + rows.filter((_, i) => i % 5 === 0).map(r => `${r.step} ${r.polarization.toFixed(3)}`).join('; '));
    assert.equal(rows.length, 41);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, c.key);
    assert.ok(rows[0].polarization < 0.15);
    assert.ok(rows.at(-1).polarization > 2 * rows[0].polarization);
});

test('experiment scaling: rows up to 10,000 boids, one-step error small', async (t) => {
    const e = experiments.find(x => x.id === 'scaling');
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(makeTestContext(gpuDevice), {})) rows.push(row);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: N wallMs/step Gtests/s(wall) stepError: ` + rows.map(r => [r.n,
        r.wallMsPerStep.toFixed(3), r.testsWall.toFixed(3), r.stepError.toExponential(1)].join(' ')).join('; '));
    assert.deepEqual(rows.map(r => r.n), [250, 500, 1000, 2000, 4000, 8000, 10000]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, c.key);
        assert.ok(row.stepError < 1e-3);
    }
});
