// Wave-equation Measure tab, CPU only: the compute-only API of js/examples/wave-equation.js
// (measureWave) and the experiments in js/teach/experiments/wave-equation.js, driven through the mock
// device (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing. Also the
// von Neumann growth factor of js/refs/wave.js against the CPU replica of cs_step. Checked: Params bytes
// in every dynamic-offset slot match the WGSL struct; the preset's state is uploaded; reads happen every
// readEvery steps and onSample can stop the run; caps are enforced; sweeps stay within their caps; every
// row carries the declared columns. The real results are checked on a software GPU in
// test/gpu/wave-equation-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as wave from '../js/examples/wave-equation.js';
import { growthFactor } from '../js/refs/wave.js';
import experiments, { COURANTS, CFL_STEPS, CFL_W, CFL_H, READ_EVERY, SIDES, MAX_SIDE, stepsFor } from '../js/teach/experiments/wave-equation.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { stepReplica } from './helpers/wave-equation-refs.mjs';

const { measureWave, PARAMS_SIZE, MAX_SUBSTEPS } = wave;
const layout = structLayout(moduleCode(wave, 'compute'), 'Params');
const STRIDE = 256;
// Every Params slot of every write to the dynamic-offset uniform buffer
const paramSlots = (device) => device.writes
    .filter(w => (w.buffer.usage & GPUBufferUsage.UNIFORM) && w.buffer.size === STRIDE * MAX_SUBSTEPS)
    .flatMap(w => Array.from({ length: w.bytes.byteLength / STRIDE }, (_, k) => decodeStruct(w.bytes, layout, k * STRIDE)));

test('measureWave on a mock device: Params slots, preset upload, readEvery, onSample stop, between()', async () => {
    assert.equal(layout.size, PARAMS_SIZE);
    const device = mockDevice();
    const w = 48, h = 32;
    const seen = [];
    let calls = 0;
    const r = await measureWave({ device }, { width: w, height: h, preset: 'open', courant: 0.72, damping: 0, steps: 100,
        warmup: 3, readEvery: 20, between: async () => { calls++; },
        onSample: ({ step, cur, prev }) => { seen.push(step); assert.equal(cur.length, w * h); assert.equal(prev.length, w * h); return step < 60; } });
    assert.deepEqual(seen, [0, 20, 40, 60], 'reads at 0 and every 20 steps; onSample returning false stops the run');
    assert.equal(r.steps, 60);
    assert.equal(r.stopped, true);
    assert.equal(calls, 6, 'chunks of ≤ 16 split at the read points: 16 + 4 per 20 steps');
    assert.equal(r.C2, Math.fround(0.72 * 0.72));
    const slots = paramSlots(device);
    // the loadPreset render() is skipped without a canvas: only the warm-up and step slots
    assert.equal(slots.length, 3 + 60);
    slots.forEach((p, k) => {
        assert.deepEqual(p.size, [w, h]);
        assert.equal(p.courant2, Math.fround(0.72 * 0.72));
        assert.equal(p.damping, 0);
        assert.deepEqual([p.dropOn, p.wallOn, p.rainOn, p.planeWave, p.absorb], [0, 0, 0, 0, 0]);
        assert.ok(p.step >= 0 && p.step < 63);
        if (k < 3) assert.equal(p.step, k, 'warm-up slots');
    });
    const heights = device.buffers.filter(b => b.size === 4 * w * h && (b.usage & GPUBufferUsage.STORAGE) && (b.usage & GPUBufferUsage.COPY_SRC));
    assert.equal(heights.length, 2, 'both time levels can be read back');
    const uploads = device.writes.filter(wr => wr.bytes.byteLength === 4 * w * h);
    assert.equal(uploads.length, 3, 'walls and two equal time levels');
    assert.deepEqual(uploads[1].bytes, uploads[2].bytes, 'the drop starts at rest');
    assert.equal(r.gpuMs, null);
});

test('measureWave enforces the caps', async () => {
    const device = mockDevice();
    const ok = { width: 64, height: 48, steps: 10 };
    await assert.rejects(measureWave({ device }, { ...ok, width: 8 }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, height: wave.MAX_MEASURE_SIDE + 1 }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, steps: wave.MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, courant: 1.2 }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, courant: 0 }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, damping: 0.1 }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, preset: 'lake' }), RangeError);
    await assert.rejects(measureWave({ device }, { ...ok, warmup: 17 }), RangeError);
});

test('growthFactor: 1 up to C = 1/√2, then the checkerboard mode grows as the CPU replica shows', (t) => {
    assert.equal(growthFactor(0.5), 1);
    assert.equal(growthFactor(0.7071), 1);
    assert.ok(growthFactor(wave.CFL_LIMIT) - 1 < 1e-6, "rounding at the limit itself");
    assert.ok(growthFactor(0.7072) > 1);
    // brute force over the Fourier symbols λ ∈ [−8, 0]: the largest root magnitude of g² − (2 + C²λ)g + 1
    for (const C of [0.3, 0.7, 0.71, 0.75, 0.9]) {
        let worst = 0;
        for (let i = 0; i <= 400; i++) {
            const a = 2 + C * C * (-8 * i / 400), disc = a * a - 4;
            worst = Math.max(worst, disc <= 0 ? 1 : (Math.abs(a) + Math.sqrt(disc)) / 2);
        }
        assert.ok(Math.abs(worst - growthFactor(C)) < 1e-12, `C = ${C}`);
    }
    // replica of cs_step on a 32 × 32 grid seeded with a checkerboard: per-step growth → growthFactor
    for (const C of [0.72, 0.75]) {
        const W = 32, H = 32, n = W * H;
        const s = { W, H, cur: new Float32Array(n), prev: new Float32Array(n), walls: null };
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) s.cur[y * W + x] = s.prev[y * W + x] = ((x + y) & 1 ? 1e-6 : -1e-6) * Math.sin(Math.PI * (x + 1) / (W + 1)) * Math.sin(Math.PI * (y + 1) / (H + 1));
        const amp = () => s.cur.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
        const o = { courant2: Math.fround(C * C), damping: 0, absorb: false, planeWave: false, step: 0, period: 20 };
        for (let k = 0; k < 40; k++) stepReplica(s, o);
        const a0 = amp();
        for (let k = 0; k < 20; k++) stepReplica(s, o);
        const g = (amp() / a0) ** (1 / 20);
        t.diagnostic(`C = ${C}: replica growth per step ${g.toFixed(4)}, von Neumann ${growthFactor(C).toFixed(4)}`);
        // the boundary mode is slightly below the infinite-grid checkerboard
        assert.ok(g > 1 && g <= growthFactor(C) * 1.001 && g > 0.95 * growthFactor(C), `C = ${C}: ${g}`);
    }
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(COURANTS.every(c => c > 0 && c <= wave.MAX_MEASURE_COURANT));
    assert.ok(COURANTS.some(c => c < wave.CFL_LIMIT) && COURANTS.some(c => c > wave.CFL_LIMIT), 'the sweep straddles the limit');
    assert.ok(CFL_STEPS <= wave.MAX_MEASURE_STEPS && CFL_STEPS % READ_EVERY === 0);
    assert.ok(CFL_W * CFL_H <= 128 * 96);
    assert.ok(SIDES.every(s => s <= MAX_SIDE && s <= wave.MAX_MEASURE_SIDE && stepsFor(s) <= 1024));
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [['cfl', {}, COURANTS.length], ['scaling', { maxSide: 2048 }, SIDES.length], ['scaling', { maxSide: 1024 }, 4]];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys = columns`);
        const sides = paramSlots(device).map(p => Math.max(...p.size));
        assert.ok(Math.max(...sides) <= MAX_SIDE);
        if (id === 'cfl') {
            assert.deepEqual(rows.map(r => r.courant), COURANTS);
            assert.ok(rows.every(r => r.steps === CFL_STEPS), 'the mock reads zeros: nothing diverges');
            const c2 = new Set(paramSlots(device).map(p => p.courant2));
            assert.equal(c2.size, COURANTS.length);
        }
    }
    // the CFL chart's reference line is vertical at the limit and spans the plotted values
    const ref = experiments[0].charts[0].reference([{ growth: 0.5 }, { growth: 4e3 }, { growth: null }]);
    assert.deepEqual(ref.points, [[wave.CFL_LIMIT, 0.5], [wave.CFL_LIMIT, 4e3]]);
});
