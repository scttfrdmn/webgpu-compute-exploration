// Boids Measure tab, CPU only: compute-only BoidsSimulation (canvas = null), measureBoids() in
// js/examples/boids.js, the experiments in js/teach/experiments/boids.js, and the float64 step and
// polarization in js/refs/boids.js, on the mock device. Checked: Params bytes for the compute-only world,
// readable boid buffers, the upload, snapshots at even steps, the caps, rows carry the declared columns;
// the reference step on hand-computed cases and its invariants. Real steps: test/gpu/boids-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    BoidsSimulation, measureBoids, wgslModules, MAX_BOIDS, MAX_MEASURE_STEPS, WORLD, stepsPerSubmit, MAX_TESTS_PER_SUBMIT,
} from '../js/examples/boids.js';
import experiments, { COUNTS, ORDER_COUNTS, ORDER_STEPS, ORDER_SAMPLES, randomBoids, sampleTargets, stepError } from '../js/teach/experiments/boids.js';
import { boidsStepAt, polarization } from '../js/refs/boids.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });
const paramsWrites = (device) => {
    const layout = structLayout(moduleCode(mod, 'compute'), 'Params');
    return device.writes.filter(w => w.buffer.size === 128 && (w.buffer.usage & GPUBufferUsage.UNIFORM)).map(w => decodeStruct(w.bytes, layout));
};

test('compute-only BoidsSimulation and measureBoids on a mock: Params, upload, snapshots, caps', async () => {
    const sim = new BoidsSimulation(mockDevice(), null, 10);
    assert.equal(sim.renderPipeline, undefined);
    for (const b of sim.boidBuffers) assert.ok(b.usage & GPUBufferUsage.COPY_SRC);
    sim.destroy();

    const device = mockDevice();
    const boids = randomBoids(300);
    const r = await measureBoids({ device }, { boids, steps: 130, samples: 3 });
    assert.deepEqual(r.snapshots.map(s => s.step), [0, 43, 87, 130]);
    assert.deepEqual(r.snapshots[0].boids, boids);
    const upload = device.writes.find(w => w.bytes.byteLength === 16 * 300);
    assert.deepEqual(new Float32Array(upload.bytes.buffer), boids);
    const p = paramsWrites(device).at(-1);
    assert.deepEqual({ n: p.numBoids, min: p.boundsMin, max: p.boundsMax, mouse: p.mousePos, force: p.mouseForce, mode: p.behaviorMode },
        { n: 300, min: [0, 0], max: [WORLD.width, WORLD.height], mouse: [500, 350], force: 0, mode: 0 });
    assert.deepEqual(r.params.boundsMax, [1000, 700]);
    assert.equal(r.gpuMsPerStep, null);
    await assert.rejects(measureBoids({ device }, { boids: new Float32Array(4 * (MAX_BOIDS + 1)) }), RangeError);
    await assert.rejects(measureBoids({ device }, { boids, steps: MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureBoids({ device }, { boids, behaviorMode: 4 }), RangeError);
    // bounded GPU work per submit
    assert.deepEqual([500, 2000, 4000, 10000].map(stepsPerSubmit), [60, 60, 20, 3]);
    for (const n of [1, 100, 3000, MAX_BOIDS]) assert.ok(stepsPerSubmit(n) * 3 * n * n <= MAX_TESTS_PER_SUBMIT);
});

const params = {
    separationRadius: 25, alignmentRadius: 50, cohesionRadius: 50, separationWeight: 1.5, alignmentWeight: 1, cohesionWeight: 1,
    maxSpeed: 4, maxForce: 0.1, boundsMin: [0, 0], boundsMax: [1000, 700], mousePos: [500, 350], mouseRadius: 150, mouseForce: 0, behaviorMode: 0,
};

test('reference step: a lone boid coasts; a pair 10 px apart separates; speed limit and wrap hold', () => {
    const lone = boidsStepAt(Float64Array.from([100, 100, 1, 2]), 1, params, [0]);
    assert.deepEqual([...lone], [101, 102, 1, 2]);
    const pair = Float64Array.from([100, 100, 0, 0, 110, 100, 0, 0]);
    const out = boidsStepAt(pair, 2, params, [0, 1]);
    // separation (inside 25 px) and cohesion (inside 50 px) on the same axis: 1.5·0.1 away, 0.1 toward
    assert.ok(Math.abs(out[2] + 0.05) < 1e-12 && Math.abs(out[6] - 0.05) < 1e-12, `vx ${out[2]}, ${out[6]}`);
    const boids = randomBoids(500, 4);
    const next = boidsStepAt(boids, 500, params, sampleTargets(500, 500));
    for (let i = 0; i < 500; i++) {
        assert.ok(Math.hypot(next[4 * i + 2], next[4 * i + 3]) <= 4 + 1e-12);
        assert.ok(next[4 * i] >= 0 && next[4 * i] <= 1000 && next[4 * i + 1] >= 0 && next[4 * i + 1] <= 700);
    }
    assert.equal(stepError(boids, Float32Array.from(next), 500, params, sampleTargets(500)) < 1e-4, true);
});

test('polarization: 1 for a uniform heading, about 1/√N for random headings', () => {
    const aligned = Float32Array.from({ length: 400 }, (_, i) => (i % 4 === 2 ? 3 : i % 4 === 3 ? -1 : 0));
    assert.ok(Math.abs(polarization(aligned, 100) - 1) < 1e-12);
    const p = polarization(randomBoids(10000, 8), 10000);
    assert.ok(p < 5 / Math.sqrt(10000), `random headings: ${p}`);
});

test('experiments on a mock device: rows, declared columns, caps', async () => {
    assert.ok([...COUNTS, ...ORDER_COUNTS].every(n => n <= MAX_BOIDS) && ORDER_STEPS <= MAX_MEASURE_STEPS && Number.isInteger(ORDER_STEPS / ORDER_SAMPLES));
    for (const [id, settings, expected] of [['scaling', {}, COUNTS.length], ['polarization', { n: 500 }, ORDER_SAMPLES + 1]]) {
        const e = experiments.find(x => x.id === id);
        const device = mockDevice();
        const rows = await runExperiment(e, settings, mockCtx(device, e.budgetMs));
        assert.equal(rows.length, expected, id);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort(), id);
        const counts = paramsWrites(device).map(p => p.numBoids);
        assert.ok(counts.length > 0 && Math.max(...counts) <= MAX_BOIDS);
    }
});
