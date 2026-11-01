// Molecular dynamics Measure tab, CPU only: compute-only MolecularDynamics (canvas = null), the NVE shader
// variant, generateAtoms() and measureMD() in js/examples/molecular-dynamics.js, the experiments in
// js/teach/experiments/molecular-dynamics.js, and the float64 references in js/refs/molecular-dynamics.js,
// on the mock device. Checked: the NVE variant only drops the thermostat; Params bytes carry the
// requested dt; snapshots at even steps; the caps; rows carry the declared columns; ljEnergy uses the
// shifted potential the truncated force conserves, and mdStep matches the integrator by hand.
// Real dynamics: test/gpu/molecular-dynamics-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    MolecularDynamics, measureMD, generateAtoms, computeShaderNVE, wgslModules, MAX_MD_ATOMS, MAX_MEASURE_STEPS, MAX_STEPS_PER_SUBMIT,
    MAX_PAIRS_PER_SUBMIT, stepsPerSubmit,
} from '../js/examples/molecular-dynamics.js';
import experiments, { COUNTS, DTS, DRIFT_TIME, DRIFT_COUNTS, boxFor } from '../js/teach/experiments/molecular-dynamics.js';
import { ljEnergy, mdStep, CUTOFF } from '../js/refs/molecular-dynamics.js';
import { mulberry32 } from '../js/refs/random.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const f32 = Math.fround;
const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });
const paramsWrites = (device) => {
    const layout = structLayout(moduleCode(mod, 'compute'), 'Params');
    return device.writes.filter(w => w.bytes.byteLength === layout.size && (w.buffer.usage & GPUBufferUsage.UNIFORM))
        .map(w => decodeStruct(w.bytes, layout));
};

test('NVE variant: the shipped kernel minus the thermostat block, same structs', () => {
    const shipped = moduleCode(mod, 'compute');
    assert.equal(moduleCode(mod, 'compute-nve'), computeShaderNVE);
    assert.notEqual(computeShaderNVE, shipped, 'the thermostat block was found and removed');
    assert.doesNotMatch(computeShaderNVE, /targetSpeed/);
    assert.match(shipped, /targetSpeed/);
    assert.deepEqual(structLayout(computeShaderNVE, 'Params'), structLayout(shipped, 'Params'));
    const lines = (c) => c.split('\n');
    assert.equal(lines(shipped).length - lines(computeShaderNVE).length, 5, 'six lines removed, one comment added');
});

test('compute-only MolecularDynamics; measureMD on a mock: atoms uploaded, dt in Params, snapshots, chunks', async () => {
    const device = mockDevice();
    const sim = new MolecularDynamics(device, null, 64, 10, { thermostat: false });
    assert.equal(sim.renderPipeline, undefined);
    for (const b of sim.atomBuffers) assert.ok(b.usage & GPUBufferUsage.COPY_SRC);
    sim.destroy();

    const n = 343;
    const atoms = generateAtoms(n, boxFor(n), 30, mulberry32(1));
    const r = await measureMD({ device }, { atoms, boxSize: boxFor(n), dt: 0.004, temperature: 30, thermostat: false, steps: 250, samples: 4 });
    assert.deepEqual(r.snapshots.map(s => s.step), [0, 63, 125, 188, 250]);
    assert.ok(Math.abs(r.snapshots[4].time - 1) < 1e-12);
    const upload = device.writes.find(w => w.bytes.byteLength === 32 * n);
    assert.deepEqual(new Float32Array(upload.bytes.buffer), atoms);
    const p = paramsWrites(device).at(-1);
    assert.deepEqual(p, { numAtoms: n, deltaTime: f32(0.004), boxSize: f32(boxFor(n)), temperature: 30 });
    assert.equal(r.gpuMs, null);
    assert.ok(MAX_STEPS_PER_SUBMIT === 100 && r.wallMsPerStep >= 0);
});

test('measureMD enforces the caps', async () => {
    const device = mockDevice();
    const atoms = (n) => new Float32Array(8 * n);
    await assert.rejects(measureMD({ device }, { atoms: atoms(MAX_MD_ATOMS + 1), boxSize: 30 }), RangeError);
    await assert.rejects(measureMD({ device }, { atoms: atoms(1), boxSize: 30 }), RangeError);
    await assert.rejects(measureMD({ device }, { atoms: atoms(100), boxSize: 30, steps: MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureMD({ device }, { atoms: atoms(100), boxSize: 30, dt: 0.05 }), RangeError);
    await assert.rejects(measureMD({ device }, { atoms: atoms(100), boxSize: 4 }), RangeError);
    // bounded GPU work per submit
    assert.deepEqual([512, 1000, 2000, 5000].map(stepsPerSubmit), [100, 100, 62, 10]);
    for (const n of [2, 100, 1000, 3000, MAX_MD_ATOMS]) assert.ok(stepsPerSubmit(n) * n * n <= Math.max(MAX_PAIRS_PER_SUBMIT, n * n));
});

test('measureMD splits a long run into submits of stepsPerSubmit(n) steps', async () => {
    const device = mockDevice();
    let submits = 0;
    const queue = device.queue;
    const submit = queue.submit;
    queue.submit = (...args) => { submits++; return submit(...args); };
    const n = 5000;
    await measureMD({ device }, { atoms: new Float32Array(8 * n), boxSize: 30, steps: 25 });
    assert.equal(submits, 3, '10 + 10 + 5 steps');
});

test('generateAtoms is initializeAtoms with a seeded generator', () => {
    const atoms = generateAtoms(1000, 30, 30, mulberry32(2));
    assert.equal(atoms.length, 8000);
    assert.deepEqual([...atoms.subarray(0, 3)], [-13.5, -13.5, -13.5], 'first lattice site, spacing 3');
    for (let i = 0; i < 1000; i++) assert.ok([4, 5, 6].every(c => Math.abs(atoms[8 * i + c]) <= 0.15 + 1e-7));
    assert.deepEqual(generateAtoms(10, 30, 30, mulberry32(2)), generateAtoms(10, 30, 30, mulberry32(2)));
});

test('ljEnergy: shifted LJ pair energy with minimum image; mdStep: semi-implicit Euler by hand', () => {
    const U = (r) => 4 * (r ** -12 - r ** -6) - 4 * (CUTOFF ** -12 - CUTOFF ** -6);
    const L = 10;
    const atoms = new Float64Array(16);
    atoms.set([4.5, 0, 0, 0, 0.1, 0, 0, 0]);
    atoms.set([-4.4, 0, 0, 0, 0, -0.2, 0, 0], 8);   // 1.1 apart through the periodic boundary
    const e = ljEnergy(atoms, 2, L);
    assert.ok(Math.abs(e.U - U(1.1)) < 1e-12);
    assert.ok(Math.abs(e.K - 0.5 * (0.01 + 0.04)) < 1e-15);
    assert.equal(ljEnergy(Float64Array.from([0, 0, 0, 0, 0, 0, 0, 0, 2.6, 0, 0, 0, 0, 0, 0, 0]), 2, L).U, 0, 'beyond the cutoff');
    // continuity at the cutoff: the shifted potential → 0 as r → r_c
    assert.ok(Math.abs(U(CUTOFF - 1e-9)) < 1e-9);

    const dt = 0.01;
    const next = mdStep(atoms, 2, L, dt);
    // Atom 1 sits 1.1 to the +x side of atom 0 through the boundary; LJ is repulsive at 1.1 < 2^{1/6},
    // so atom 0 is pushed toward −x and atom 1 toward +x
    const F = 24 * (2 * 1.1 ** -12 - 1.1 ** -6) / 1.1;
    assert.ok(F > 0);
    assert.ok(Math.abs(next[4] - (0.1 - F * dt)) < 1e-12);
    assert.ok(Math.abs(next[0] - (4.5 + next[4] * dt)) < 1e-12, 'x advances with the new velocity');
    assert.ok(Math.abs(next[8 + 4] - F * dt) < 1e-12, 'Newton\'s third law');
    const far = mdStep(Float64Array.from([4.99, 0, 0, 0, 2, 0, 0, 0]), 1, L, 0.01);
    assert.ok(Math.abs(far[0] - (4.99 + 0.02 - 10)) < 1e-12, 'leaving at +L/2 re-enters at −L/2');
});

test('experiments on a mock device: rows, declared columns, caps', async () => {
    assert.ok(COUNTS.every(n => n <= MAX_MD_ATOMS) && DRIFT_COUNTS.every(n => n <= MAX_MD_ATOMS));
    assert.ok(DTS.every(dt => Math.round(DRIFT_TIME / dt) <= 2000));
    assert.ok(DRIFT_COUNTS.every(n => boxFor(n) > 2 * CUTOFF), 'minimum image needs L > 2 r_c');
    for (const [id, settings, expected] of [
        ['scaling', {}, COUNTS.length],
        ['energy-drift', { n: 512, thermostat: false }, DTS.length],
        ['energy-drift', { n: 1000, thermostat: true }, DTS.length],
    ]) {
        const e = experiments.find(x => x.id === id);
        const device = mockDevice();
        const rows = await runExperiment(e, settings, mockCtx(device, e.budgetMs));
        assert.equal(rows.length, expected, id);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort(), id);
        const counts = paramsWrites(device).map(p => p.numAtoms);
        assert.ok(counts.length > 0 && Math.max(...counts) <= MAX_MD_ATOMS);
        if (id === 'energy-drift') assert.deepEqual(rows.map(r => r.steps), DTS.map(dt => Math.round(DRIFT_TIME / dt)));
    }
});
