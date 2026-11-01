// SPH fluid Measure tab, CPU only: compute-only SPHFluidSimulation (canvas = null), measureSPHFrame() and
// runSPH() in js/examples/sph-fluid.js, the experiments in js/teach/experiments/sph-fluid.js, and the
// float64 density sum in js/refs/sph.js, on the mock device. Checked: bounds of the compute-only world,
// readable particle buffers, Params bytes with the requested gas constant, 18 timed passes per frame,
// snapshots at even frames, the caps, rows carry the declared columns, and sphDensities against the
// shipped latticeDensity(). Real runs: test/gpu/sph-fluid-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SPHFluidSimulation, measureSPHFrame, runSPH, wgslModules, MAX_SPH_PARTICLES, MAX_SPH_FRAMES } from '../js/examples/sph-fluid.js';
import experiments, { COUNTS, GAS_CONSTANTS, STIFFNESS_COUNTS, FRAMES, densityError } from '../js/teach/experiments/sph-fluid.js';
import { sphDensities } from '../js/refs/sph.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const f32 = Math.fround;
const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });
const paramsWrites = (device) => {
    const layout = structLayout(moduleCode(mod, 'integrate'), 'Params');
    return device.writes.filter(w => w.buffer.size === 256 && (w.buffer.usage & GPUBufferUsage.UNIFORM)).map(w => decodeStruct(w.bytes, layout));
};

test('compute-only SPHFluidSimulation: 1000 × 700 world, no render pipeline, readable particle buffers', () => {
    const sim = new SPHFluidSimulation(mockDevice(), null, 500);
    assert.deepEqual([sim.boundsMin, sim.boundsMax], [[10, 10], [990, 690]], 'as index.html\'s 1000 × 700 canvas');
    assert.equal(sim.renderPipeline, undefined);
    for (const b of sim.particleBuffers) assert.ok(b.usage & GPUBufferUsage.COPY_SRC);
    sim.destroy();
});

test('runSPH on a mock device: gas constant in Params, snapshots at even frames, caps', async () => {
    const device = mockDevice();
    let calls = 0;
    const r = await runSPH({ device }, { n: 300, gasConstant: 4e6, frames: 10, samples: 4, between: async () => { calls++; } });
    assert.deepEqual(r.snapshots.map(s => s.frame), [0, 3, 5, 8, 10]);
    assert.equal(calls, 4);
    assert.equal(r.snapshots[0].particles.length, 8 * 300);
    const p = paramsWrites(device).at(-1);
    assert.equal(p.numParticles, 300);
    assert.equal(p.gasConstant, 4e6);
    assert.equal(p.deltaTime, f32(1 / 360));
    assert.deepEqual([p.boundsMin, p.boundsMax], [[10, 10], [990, 690]]);
    assert.deepEqual({ dt: r.dt, h: r.smoothingRadius, k: r.gasConstant }, { dt: 1 / 360, h: 16, k: 4e6 });
    await assert.rejects(runSPH({ device }, { n: MAX_SPH_PARTICLES + 1 }), RangeError);
    await assert.rejects(runSPH({ device }, { n: 100, frames: MAX_SPH_FRAMES + 1 }), RangeError);
    await assert.rejects(runSPH({ device }, { n: 100, gasConstant: 0 }), RangeError);
});

test('measureSPHFrame on a mock device: result shape per pass', async () => {
    const r = await measureSPHFrame({ device: mockDevice() }, { n: 400, repeats: 2 });
    assert.deepEqual(Object.keys(r.passes), ['density', 'forces', 'integrate']);
    assert.equal(r.substeps, 6);
    assert.equal(r.gpuMs, null);
    assert.ok(r.wallMs >= 0);
});

test('sphDensities: the shipped lattice mass gives rest density 1 inside the dam-break block', (t) => {
    const sim = new SPHFluidSimulation(mockDevice(), null, 900);   // 30 × 30 block
    const rho = sphDensities(sim.particles, 900, sim.smoothingRadius, sim.mass);
    const interior = 15 * 30 + 15;
    t.diagnostic(`interior ρ = ${rho[interior].toFixed(9)}, corner ρ = ${rho[0].toFixed(3)}`);
    assert.ok(Math.abs(rho[interior] - 1) < 1e-12, 'mass = 1 / latticeDensity()');
    assert.ok(rho[0] > 0.5 && rho[0] < 0.7, 'a corner particle misses three quarters of its neighbours (≈ 0.62)');
    const packed = Float32Array.from(sim.particles);
    for (let i = 0; i < 900; i++) packed[8 * i + 4] = rho[i];
    assert.ok(densityError(packed, 900, sim.smoothingRadius, sim.mass) < 1e-7);
});

test('experiments on a mock device: rows, declared columns, caps', async () => {
    assert.ok([...COUNTS, ...STIFFNESS_COUNTS].every(n => n <= MAX_SPH_PARTICLES) && FRAMES <= MAX_SPH_FRAMES);
    assert.ok(GAS_CONSTANTS.includes(1e6), 'the shipped stiffness is one of the points');
    for (const [id, settings, expected] of [['scaling', {}, COUNTS.length], ['stiffness', { n: 500 }, GAS_CONSTANTS.length]]) {
        const e = experiments.find(x => x.id === id);
        const device = mockDevice();
        const rows = await runExperiment(e, settings, mockCtx(device, e.budgetMs));
        assert.equal(rows.length, expected, id);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort(), id);
        const counts = paramsWrites(device).map(p => p.numParticles);
        assert.ok(counts.length > 0 && Math.max(...counts) <= MAX_SPH_PARTICLES);
    }
});
