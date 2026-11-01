// SPH fluid compute-only API (measureSPHFrame / runSPH in js/examples/sph-fluid.js) and its Measure
// experiments, on a software GPU — CI only, see harness.mjs. Checked against the brute-force float64
// density sum (js/refs/sph.js) and the stiffness trend the Understand page describes. The printed numbers
// are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { runSPH } from '../../js/examples/sph-fluid.js';
import experiments, { densityError } from '../../js/teach/experiments/sph-fluid.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('runSPH: densities of the last density pass vs the float64 sum, at the start and mid-splash', async (t) => {
    const r = await runSPH(gpuDevice, { n: 500, frames: 90, samples: 3 });
    const errs = r.snapshots.slice(1).map(s => densityError(s.particles, 500, r.smoothingRadius, r.mass));
    t.diagnostic(`frames ${r.snapshots.slice(1).map(s => s.frame).join(', ')}: max rel density error ${errs.map(e => e.toExponential(2)).join(', ')}`);
    assert.ok(errs.every(e => e < 1e-5));
    assert.ok(r.snapshots.at(-1).particles.every(Number.isFinite));
});

test('experiment stiffness (N = 500): compression falls with k in the stable range', async (t) => {
    const e = experiments.find(x => x.id === 'stiffness');
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(makeTestContext(gpuDevice), { n: 500 })) rows.push(row);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: k cfl peak(ρ/ρ0−1) maxSpeed densityErr: ` + rows.map(r => [r.k.toExponential(1),
        r.cfl.toFixed(2), r.peakCompression.toFixed(3), r.maxSpeed.toFixed(0), r.densityError.toExponential(1)].join(' ')).join('; '));
    assert.equal(rows.length, 5);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, c.key);
    for (let i = 1; i < 4; i++) assert.ok(rows[i].peakCompression < rows[i - 1].peakCompression, `k = ${rows[i].k}`);
    for (const row of rows.slice(0, 4)) assert.ok(row.densityError < 1e-4, `k = ${row.k}: density error ${row.densityError}`);
});

test('experiment scaling: rows up to 4,000 particles', async (t) => {
    const e = experiments.find(x => x.id === 'scaling');
    const rows = [];
    for await (const row of e.run(makeTestContext(gpuDevice), {})) rows.push(row);
    t.diagnostic('N wallMs/frame density/forces/integrate GPU ms: ' + rows.map(r => `${r.n} ${r.wallMsPerFrame.toFixed(2)} `
        + `${[r.density, r.forces, r.integrate].map(v => v?.toFixed(3) ?? '—').join('/')}`).join('; '));
    assert.deepEqual(rows.map(r => r.n), [250, 500, 1000, 2000, 4000]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, c.key);
});
