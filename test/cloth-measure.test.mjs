// Cloth Measure tab, CPU only: the compute-only API of js/examples/cloth.js (measureCloth, ClothSimulation
// with canvas = null) and the experiments in js/teach/experiments/cloth.js, driven through the mock
// device (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: the SimParams bytes the compute-only path writes (one write per command buffer, with the
// simulated time advancing); the caps; the stretch statistics of js/refs/cloth.js; every sweep within
// its caps and every row with the declared columns. Real results: test/gpu/cloth-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    measureCloth, buildCloth, iterationStiffness, wgslModules, GRID_SIZES, MAX_ITERATIONS, MAX_MEASURE_SUBSTEPS,
    MAX_MEASURE_TIMED, MAX_SUBSTEPS, SUBSTEP_DT, NUM_COLORS,
} from '../js/examples/cloth.js';
import { stretchStats } from '../js/refs/cloth.js';
import experiments, { ITERATIONS, STRETCH_GRIDS, SETTLE_SUBSTEPS, TIMED_SUBSTEPS, SCALING_SETTLE } from '../js/teach/experiments/cloth.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const SIM = structLayout(wgslModules().find(m => m.name === 'compute').code, 'SimParams');
const simWrites = (device) => device.writes
    .filter(w => w.bytes.byteLength === SIM.size && (w.buffer.usage & GPUBufferUsage.UNIFORM))
    .map(w => decodeStruct(w.bytes, SIM));
const f = Math.fround;

test('measureCloth on a mock device: SimParams per command buffer, time advancing, positions read back', async () => {
    const device = mockDevice();
    let calls = 0;
    const r = await measureCloth({ device }, { n: 48, preset: 'edge', iterations: 5, stiffness: 0.8, substeps: 70, timed: 16,
        between: async () => { calls++; } });
    const sims = simWrites(device);
    // reset()'s normals pass, ceil(70 / 8) settle chunks, 16 / 8 timed chunks
    assert.equal(sims.length, 1 + Math.ceil(70 / MAX_SUBSTEPS) + 2);
    // Simulated time at the start of each command buffer: normals at 0, settle chunks of 8 (the last
    // one 6), then the timed chunks
    const starts = [0, 0, 8, 16, 24, 32, 40, 48, 56, 64, 70, 78];
    sims.forEach((s, i) => {
        assert.equal(s.gridN, 48);
        assert.equal(s.tearOn, 0);
        assert.equal(s.dt, f(SUBSTEP_DT));
        // the first write is the constructor's normals pass, before the solver settings are applied
        if (i > 0) assert.deepEqual(s.stiffness.slice(0, 3), [0.8, 0.4, 0.16].map(k => f(iterationStiffness(k, 5))));
        assert.equal(s.time, f(starts[i] * SUBSTEP_DT), `chunk ${i} time`);
    });
    assert.equal(calls, 2, 'between() every 64 settle substeps and at the end of the settle phase');
    assert.equal(r.count, 48 * 48);
    assert.equal(r.iterations, 5);
    assert.equal(r.dispatchesPerSubstep, 3 + 5 * NUM_COLORS);
    assert.equal(r.gpuMsPerSubstep, null, 'no timestamp-query on the mock');
    assert.ok(r.wallMsPerSubstep >= 0);
    assert.equal(r.positions.length, 4 * 48 * 48);
    assert.ok(Math.abs(r.simulatedSeconds - 86 * SUBSTEP_DT) < 1e-12);
    assert.ok(device.buffers.some(b => b.size === 16 * 48 * 48 && (b.usage & GPUBufferUsage.MAP_READ)), 'positions readback');
    assert.equal(r.constraintData.byteLength, buildCloth(48, 'edge').constraintData.byteLength);
});

test('measureCloth enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measureCloth({ device }, { n: 40 }), RangeError);
    await assert.rejects(measureCloth({ device }, { n: 32, preset: 'nope' }), RangeError);
    await assert.rejects(measureCloth({ device }, { n: 32, iterations: 0 }), RangeError);
    await assert.rejects(measureCloth({ device }, { n: 32, iterations: MAX_ITERATIONS + 1 }), RangeError);
    await assert.rejects(measureCloth({ device }, { n: 32, substeps: MAX_MEASURE_SUBSTEPS + 1 }), RangeError);
    await assert.rejects(measureCloth({ device }, { n: 32, timed: MAX_MEASURE_TIMED + 8 }), RangeError);
    await assert.rejects(measureCloth({ device }, { n: 32, timed: 4 }), RangeError);
});

test('stretchStats: zero at rest, exact for a uniform stretch, kinds filter', () => {
    const n = 12;
    const rest = buildCloth(n, 'edge');       // edge: placed == rest layout
    const s0 = stretchStats(rest.positions, rest.constraintData, [0, 1, 2]);
    assert.ok(s0.max < 1e-6 && Math.abs(s0.mean) < 1e-6 && s0.count === rest.numConstraints);
    const scaled = Float32Array.from(rest.positions, (v, i) => (i % 4 === 3 ? v : 1.25 * v));
    const s1 = stretchStats(scaled, rest.constraintData);
    assert.equal(s1.count, rest.kindCounts[0]);
    assert.ok(Math.abs(s1.max - 0.25) < 1e-5 && Math.abs(s1.mean - 0.25) < 1e-5 && Math.abs(s1.p99 - 0.25) < 1e-5 && Math.abs(s1.rms - 0.25) < 1e-5);
    // Curtain: hung from a rod 80% as wide as the fabric, so the horizontal links start compressed
    const curtain = buildCloth(n, 'curtain');
    assert.ok(stretchStats(curtain.positions, curtain.constraintData).mean < -0.01);
    assert.deepEqual(stretchStats(rest.positions, rest.constraintData, [7]), { count: 0, max: 0, mean: 0, p99: 0, rms: 0 });
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(ITERATIONS.every(k => Number.isInteger(k) && k >= 1 && k <= MAX_ITERATIONS));
    assert.ok(STRETCH_GRIDS.every(n => GRID_SIZES.includes(n)));
    assert.ok(SETTLE_SUBSTEPS <= MAX_MEASURE_SUBSTEPS && SCALING_SETTLE <= MAX_MEASURE_SUBSTEPS);
    assert.ok(TIMED_SUBSTEPS <= MAX_MEASURE_TIMED && TIMED_SUBSTEPS % MAX_SUBSTEPS === 0);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per sweep value, declared columns, caps respected', async (t) => {
    const cases = [
        ['iterations-stretch', { n: 64, preset: 'corners' }, ITERATIONS.length],
        ['iterations-stretch', { n: 128, preset: 'edge' }, ITERATIONS.length],
        ['grid-scaling', { iterations: 16 }, GRID_SIZES.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort());
        const sims = simWrites(device);
        assert.ok(sims.every(s => GRID_SIZES.includes(s.gridN)));
        if (id === 'iterations-stretch') assert.deepEqual(rows.map(r => r.iterations), ITERATIONS);
        if (id === 'grid-scaling') assert.deepEqual(rows.map(r => r.n), GRID_SIZES);
    }
});
