// Molecular dynamics compute-only API (measureMD in js/examples/molecular-dynamics.js, shipped kernel and
// NVE variant) and its Measure experiments, on a software GPU — CI only, see harness.mjs. Checked against
// js/refs/molecular-dynamics.js: one step vs the float64 integrator (with and without the thermostat), and
// the energy behaviour the Understand page describes. The printed numbers are the CI baselines.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureMD, generateAtoms } from '../../js/examples/molecular-dynamics.js';
import { mdStep, ljEnergy } from '../../js/refs/molecular-dynamics.js';
import { mulberry32 } from '../../js/refs/random.js';
import experiments, { boxFor, SEED, TEMPERATURE } from '../../js/teach/experiments/molecular-dynamics.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('one step of the shipped kernel and of the NVE variant vs the float64 step', async (t) => {
    const n = 343, L = boxFor(n);
    const atoms = generateAtoms(n, L, TEMPERATURE, mulberry32(SEED));
    for (const thermostat of [true, false]) {
        const r = await measureMD(gpuDevice, { atoms, boxSize: L, dt: 0.004, temperature: TEMPERATURE, thermostat, steps: 1, samples: 1 });
        const ref = mdStep(atoms, n, L, 0.004, { thermostat, temperature: TEMPERATURE });
        let dx = 0, dv = 0;
        const got = r.snapshots[1].atoms;
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
            dx = Math.max(dx, Math.abs(got[8 * i + c] - ref[8 * i + c]));
            dv = Math.max(dv, Math.abs(got[8 * i + 4 + c] - ref[8 * i + 4 + c]));
        }
        t.diagnostic(`thermostat ${thermostat}: max |Δx| ${dx.toExponential(2)}, max |Δv| ${dv.toExponential(2)}`);
        assert.ok(dx < 1e-5 && dv < 1e-4, `thermostat ${thermostat}: ${dx} ${dv}`);
    }
});

test('experiment energy-drift (NVE, N = 512): first-order energy error, falling with dt', async (t) => {
    const e = experiments.find(x => x.id === 'energy-drift');
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(makeTestContext(gpuDevice), { n: 512, thermostat: false })) rows.push(row);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: dt steps max|dE/E0| final K(end)/K(0): ` + rows.map(r => [r.dt, r.steps,
        r.maxDrift.toExponential(2), r.finalDrift.toExponential(2), r.kineticRatio.toFixed(1)].join(' ')).join('; '));
    assert.deepEqual(rows.map(r => r.steps), [250, 500, 1000, 2000]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, c.key);
    for (let i = 1; i < rows.length; i++) {
        const ratio = rows[i - 1].maxDrift / rows[i].maxDrift;
        assert.ok(ratio > 1.3 && ratio < 3.5, `dt ${rows[i - 1].dt} → ${rows[i].dt}: ratio ${ratio} (first order: 2)`);
    }
    assert.ok(rows[0].maxDrift < 2e-2);
    assert.ok(rows.every(r => r.kineticRatio > 5), 'the simple cubic lattice melts');
});

test('experiment energy-drift with the thermostat: the energy follows the thermostat, not dt', async (t) => {
    const n = 512, L = boxFor(n);
    const atoms = generateAtoms(n, L, TEMPERATURE, mulberry32(SEED));
    const out = [];
    for (const dt of [0.016, 0.008]) {
        const r = await measureMD(gpuDevice, { atoms, boxSize: L, dt, temperature: TEMPERATURE, thermostat: true, steps: Math.round(2 / dt), samples: 4 });
        const E = r.snapshots.map(s => ljEnergy(s.atoms, n, L).E);
        out.push({ dt, drift: Math.max(...E.map(x => Math.abs(x - E[0]) / Math.abs(E[0]))) });
    }
    t.diagnostic(`thermostat on, t = 2: ${out.map(o => `dt ${o.dt}: max |ΔE/E₀| ${o.drift.toExponential(2)}`).join('; ')}`);
    assert.ok(out[1].drift > 0.5 * out[0].drift, 'halving dt does not halve the change');
});

test('experiment scaling: rows up to 5,000 atoms', async (t) => {
    const e = experiments.find(x => x.id === 'scaling');
    const rows = [];
    for await (const row of e.run(makeTestContext(gpuDevice), {})) rows.push(row);
    t.diagnostic('N wallMs/step Gpairs/s(wall): ' + rows.map(r => `${r.n} ${r.wallMsPerStep.toFixed(3)} ${r.pairsWall.toFixed(3)}`).join('; '));
    assert.deepEqual(rows.map(r => r.n), [250, 500, 1000, 2000, 4000, 5000]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, c.key);
});
