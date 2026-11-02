// N-body Measure tab, CPU only: the compute-only API of js/examples/nbody.js (measureForces,
// measureEnergyDrift) and the experiments in js/teach/experiments/nbody.js, driven through the mock
// device (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: the uniform bytes the compute-only path writes match the WGSL structs; buffers are sized as
// allocate() documents; the caps are enforced; every sweep stays within its caps; every row carries the
// declared columns. The real results are checked on a software GPU in test/gpu/nbody-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    generateBodies, measureForces, measureEnergyDrift, wgslModules, WG, REDUCE_TILE, RADIX_TILE, RADIX_PASSES,
    MAX_BH, MAX_BRUTE, MAX_MEASURE_STEPS,
} from '../js/examples/nbody.js';
import { directForces, directForcesAt } from '../js/refs/nbody.js';
import experiments, {
    BH_MAX_N, BRUTE_MAX_N, THETAS, ERROR_COUNTS, SCALING_COUNTS, DTS, DRIFT_TIME, DRIFT_COUNTS, SAMPLE, sampleTargets,
    PROTOCOL, REPLICA_MEDIANS, REPLICA_FACTOR, SOFTENING, SEED, DRIFT_BOUNDS,
} from '../js/teach/experiments/nbody.js';
import { buildBH, forceReplica, relErrors } from './helpers/nbody-refs.mjs';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const code = name => wgslModules().find(m => m.name === name).code;
const f32 = Math.fround;
const simLayout = structLayout(code('force'), 'SimParams');
const radixLayout = structLayout(code('radix'), 'RadixParams');

// SimParams writes on a mock device: 32-byte writes to a UNIFORM buffer of that size
function simParamsWrites(device) {
    return device.writes
        .filter(w => w.buffer.size === simLayout.size && (w.buffer.usage & GPUBufferUsage.UNIFORM))
        .map(w => decodeStruct(w.bytes, simLayout));
}

test('measureForces on a mock device: SimParams / RadixParams bytes, buffer sizes, bodies uploaded, result shape', async () => {
    const device = mockDevice();
    const n = 5000;   // not a multiple of 256 or 1024
    const { positions } = generateBodies('plummer', n, 0.05, 3);
    const r = await measureForces({ device }, { positions, mode: 'bh', theta: 0.7, softening: 0.04, warmup: 1, repeats: 3 });

    const sims = simParamsWrites(device);
    assert.equal(sims.length, 4, 'one SimParams write per evaluation (1 warm-up + 3 timed)');
    for (const s of sims) {
        assert.deepEqual({ count: s.count, numTiles: s.numTiles, numPartials: s.numPartials, trailSlot: s.trailSlot, pad0: s.pad0 },
            { count: n, numTiles: Math.ceil(n / WG), numPartials: Math.ceil(n / REDUCE_TILE), trailSlot: 0, pad0: 0 });
        assert.equal(s.theta, f32(0.7));
        assert.equal(s.eps2, f32(0.04 * 0.04));
    }
    const radix = device.writes.find(w => w.bytes.byteLength === 256 * RADIX_PASSES);
    for (let p = 0; p < RADIX_PASSES; p++) {
        assert.deepEqual(decodeStruct(radix.bytes, radixLayout, p * 256), { shift: 4 * p, count: n, numTiles: Math.ceil(n / RADIX_TILE), pad0: 0 });
    }
    const upload = device.writes.find(w => w.bytes.byteLength === 16 * n);
    assert.deepEqual(new Float32Array(upload.bytes.buffer), positions, 'positions uploaded unchanged');

    // allocate()'s sizes, plus the readback copies of the accelerations and the interaction counts
    const sizes = device.buffers.map(b => b.size);
    for (const size of [16 * n, 4 * n, 8 * (n - 1), 4 * (2 * n - 1), 16 * (2 * n - 1), 32 * (2 * n - 1)]) {
        assert.ok(sizes.includes(size), `a buffer of ${size} bytes`);
    }
    const accel = device.buffers.find(b => b.size === 16 * n && (b.usage & GPUBufferUsage.STORAGE) && (b.usage & GPUBufferUsage.COPY_SRC)
        && !(b.usage & GPUBufferUsage.COPY_DST));
    assert.ok(accel, 'accel is a COPY_SRC storage buffer');
    assert.ok(device.buffers.some(b => b.size === 16 * n && (b.usage & GPUBufferUsage.MAP_READ)), 'accel readback');
    assert.ok(device.buffers.some(b => b.size === 4 * n && (b.usage & GPUBufferUsage.MAP_READ)), 'interactions readback');

    assert.equal(r.n, n);
    assert.equal(r.dispatches, 31);
    assert.equal(r.accel.length, 4 * n);
    assert.equal(r.gpuMs, null, 'no timestamp-query on the mock device');
    assert.ok(r.wallMs >= 0);
    assert.deepEqual(Object.keys(r.passes), ['tree', 'force']);

    const brute = await measureForces({ device: mockDevice() }, { positions, mode: 'brute', accel: false });
    assert.equal(brute.dispatches, 1);
    assert.equal(brute.interactionsPerBody, n - 1);
    assert.equal(brute.accel, undefined);
    assert.deepEqual(Object.keys(brute.passes), ['force']);

    // Wall-clock phases (js/phase-clock.js): all four, non-negative, dispatch = the evaluation's wall time
    for (const res of [r, brute]) {
        assert.deepEqual(Object.keys(res.phases).sort(), ['compile', 'dispatch', 'readback', 'upload']);
        for (const v of Object.values(res.phases)) assert.ok(v >= 0);
        assert.equal(res.phases.dispatch, res.wallMs);
    }
});

test('REPLICA_MEDIANS are the CPU replica of cs_bh_force on the experiment\'s bodies and targets', (t) => {
    for (const [config, medians] of Object.entries(REPLICA_MEDIANS)) {
        const [preset, nText] = config.split('/');
        const n = Number(nText);
        assert.ok(ERROR_COUNTS.includes(n));
        const { positions } = generateBodies(preset, n, SOFTENING, SEED);
        const bh = buildBH(positions, n);
        const targets = sampleTargets(n);
        const ref = directForcesAt(positions, n, SOFTENING * SOFTENING, targets);
        const got = THETAS.map(theta => {
            const r = forceReplica(positions, bh.vals, bh, theta, SOFTENING * SOFTENING);
            const acc = new Float64Array(3 * targets.length);
            targets.forEach((i, k) => { for (let c = 0; c < 3; c++) acc[3 * k + c] = r.acc[3 * i + c]; });
            return relErrors(acc, ref.acc, targets.length).median;
        });
        t.diagnostic(`${config}: ${got.map(v => v.toExponential(2)).join(' ')}`);
        got.forEach((v, i) => assert.ok(Math.abs(v / medians[i] - 1) < 0.01, `${config} θ = ${THETAS[i]}: ${v} vs ${medians[i]}`));
    }
    assert.ok(REPLICA_FACTOR > 1 && REPLICA_FACTOR <= 2);
});

test('N-body verify(): pass, fail and not-checked on synthetic rows', () => {
    const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
    const status = (checks) => checks.map(c => c.status);

    // theta-error: rows at exactly the replica medians pass; a 3× error fails that θ; a non-growing step fails
    const settings = { n: 4096, preset: 'plummer' };
    const good = THETAS.map((theta, i) => ({ theta, median: REPLICA_MEDIANS['plummer/4096'][i] }));
    let checks = byId['theta-error'].verify(good, settings);
    assert.equal(checks.length, THETAS.length + 1);
    assert.ok(checks.every(c => c.status === 'pass'), JSON.stringify(checks));
    const bad = good.map((r, i) => (i === 2 ? { ...r, median: 3 * r.median } : r));
    checks = byId['theta-error'].verify(bad, settings);
    assert.equal(checks[2].status, 'fail');
    assert.equal(checks[THETAS.length].status, 'fail', 'median falls from θ = 0.5 to 0.7');
    assert.equal(checks[THETAS.length].value, 1);
    // a partial run: missing θ are not checked, not failed
    checks = byId['theta-error'].verify(good.slice(0, 1), settings);
    assert.deepEqual(status(checks), ['pass', ...Array(THETAS.length - 1).fill('not-checked'), 'not-checked']);
    // a config without replica values
    assert.ok(byId['theta-error'].verify(good, { n: 1234, preset: 'plummer' }).slice(0, THETAS.length).every(c => c.status === 'not-checked'));

    // energy-drift, brute force: CI values pass; a rising error and an out-of-bound value fail
    const ci = [3.4e-5, 5.5e-6, 1.3e-6, 3.0e-7].map((maxDrift, i) => ({ dt: DTS[i], maxDrift }));
    checks = byId['energy-drift'].verify(ci, { mode: 'brute' });
    assert.equal(checks.length, DTS.length + 2);
    assert.ok(checks.every(c => c.status === 'pass'), JSON.stringify(checks));
    assert.ok(Math.abs(checks[DTS.length + 1].value - Math.log2(5.5 / 1.3)) < 1e-12);
    const rising = ci.map((r, i) => (i === 3 ? { ...r, maxDrift: 2 * DRIFT_BOUNDS.brute } : r));
    checks = byId['energy-drift'].verify(rising, { mode: 'brute' });
    assert.equal(checks[3].status, 'fail');
    assert.equal(checks[DTS.length].status, 'fail');
    // Barnes-Hut: a flat floor is fine (no monotonicity or order check), only the bound
    const floor = DTS.map(dt => ({ dt, maxDrift: 6.6e-4 }));
    checks = byId['energy-drift'].verify(floor, { mode: 'bh' });
    assert.deepEqual(status(checks), Array(DTS.length).fill('pass'));
    assert.equal(byId['energy-drift'].verify([{ dt: 0.04, maxDrift: 0.02 }], { mode: 'bh' })[0].status, 'fail');

    // scaling: no verify (the record says not-checked)
    assert.equal(byId.scaling.verify, undefined);
});

test('compute-only API enforces the caps', async () => {
    const device = mockDevice();
    const bodies = (n) => new Float32Array(4 * n);
    await assert.rejects(measureForces({ device }, { positions: bodies(1) }), RangeError);
    await assert.rejects(measureForces({ device }, { positions: bodies(MAX_BH + 1) }), RangeError);
    await assert.rejects(measureForces({ device }, { positions: bodies(MAX_BRUTE + 1), mode: 'brute' }), RangeError);
    await assert.rejects(measureForces({ device }, { positions: bodies(1024), mode: 'direct' }), RangeError);
    await assert.rejects(measureForces({ device }, { positions: bodies(1024), theta: 0 }), RangeError);
    await assert.rejects(measureEnergyDrift({ device }, { positions: bodies(1024), velocities: bodies(1024), steps: MAX_MEASURE_STEPS + 1 }), RangeError);
    await assert.rejects(measureEnergyDrift({ device }, { positions: bodies(1024), velocities: bodies(1024), steps: 0 }), RangeError);
    // repeats are capped too: at most 10 timed + 10 warm-up evaluations
    const r = await measureForces({ device }, { positions: bodies(1024), warmup: 50, repeats: 50, accel: false });
    assert.equal(simParamsWrites(device).length, 20);
    assert.equal(r.n, 1024);
});

test('measureEnergyDrift on a mock device: dt in SimParams, samples at even steps, between() after each, cancel', async () => {
    const device = mockDevice();
    const n = 2048;
    const { positions, velocities } = generateBodies('plummer', n, 0.05, 5);
    let calls = 0;
    const r = await measureEnergyDrift({ device }, { positions, velocities, mode: 'brute', dt: 0.02, steps: 50, samples: 4,
        between: async () => { calls++; } });
    assert.deepEqual(r.samples.map(s => s.step), [13, 25, 38, 50]);
    assert.equal(calls, 4);
    assert.ok(Math.abs(r.samples[3].time - 1) < 1e-12);
    const sims = simParamsWrites(device);
    assert.equal(sims.length, 5, 'E₀ plus one write per sample');
    for (const s of sims) { assert.equal(s.dt, f32(0.02)); assert.equal(s.count, n); }
    // velocities uploaded (the second 16·n write after the positions)
    const uploads = device.writes.filter(w => w.bytes.byteLength === 16 * n);
    assert.deepEqual(new Float32Array(uploads[1].bytes.buffer), velocities);

    // A throwing between() (the runner's ctx.step on Cancel) stops the run
    let n2 = 0;
    await assert.rejects(measureEnergyDrift({ device: mockDevice() }, { positions, velocities, steps: 100, samples: 10,
        between: async () => { if (++n2 === 2) throw new Error('Cancelled'); } }), /Cancelled/);
    assert.equal(n2, 2);
});

test('directForcesAt equals the rows of directForces', () => {
    const n = 600;
    const { positions } = generateBodies('collision', n, 0.05, 7);
    const all = directForces(positions, n, 0.0025);
    const targets = sampleTargets(n, 37);
    const some = directForcesAt(positions, n, 0.0025, targets);
    targets.forEach((i, t) => {
        for (let c = 0; c < 3; c++) assert.equal(some.acc[3 * t + c], all.acc[3 * i + c]);
        assert.equal(some.phi[t], all.phi[i]);
    });
    assert.deepEqual(sampleTargets(16384).slice(0, 3), [0, 32, 64]);
    assert.equal(new Set(sampleTargets(4096)).size, SAMPLE);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(BH_MAX_N <= 65536 && BRUTE_MAX_N <= 16384 && BH_MAX_N <= MAX_BH && BRUTE_MAX_N <= MAX_BRUTE);
    assert.ok([...ERROR_COUNTS, ...SCALING_COUNTS].every(n => n <= BH_MAX_N));
    assert.ok(DRIFT_COUNTS.every(n => n <= BRUTE_MAX_N));
    assert.ok(THETAS.every(t => t >= 0.3 && t <= 1.2));
    assert.ok(DTS.every(dt => Math.round(DRIFT_TIME / dt) <= 400), 'at most 400 leapfrog steps per run');
    assert.ok(DTS.reduce((s, dt) => s + Math.round(DRIFT_TIME / dt), 0) <= MAX_MEASURE_STEPS);
    for (const e of experiments) {
        assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
        for (const s of e.settings ?? []) {
            for (const o of s.options) if (typeof o.value === 'number') assert.ok(o.value <= BH_MAX_N, `${e.id}/${s.key}`);
        }
    }
});

// Every experiment, every setting combination that matters, on the mock device through the real runner
test('experiments run on a mock device: one row per sweep value, declared columns, caps respected', async (t) => {
    const cases = [
        ['theta-error', { n: 16384, preset: 'spiral' }, THETAS.length],
        ['theta-error', { n: 4096, preset: 'plummer' }, THETAS.length],
        ['scaling', { preset: 'plummer' }, SCALING_COUNTS.length],
        ['energy-drift', { n: 4096, mode: 'brute', preset: 'plummer' }, DTS.length],
        ['energy-drift', { n: 2048, mode: 'bh', preset: 'spiral' }, DTS.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected, `${id}: rows`);
        // row keys = the columns, plus <key>Lo / <key>Hi (the protocol's spread) for some of them
        const allowed = new Set(experiment.columns.flatMap(c => [c.key, `${c.key}Lo`, `${c.key}Hi`]));
        for (const row of rows) {
            for (const c of experiment.columns) assert.ok(c.key in row, `${id}: row has column ${c.key}`);
            for (const k of Object.keys(row)) assert.ok(allowed.has(k), `${id}: unexpected row key ${k}`);
        }
        if (id !== 'energy-drift') {
            assert.equal(ctx.measured.calls > 0 && ctx.measured.repeats, PROTOCOL.repeats, `${id}: uses ctx.measure`);
            assert.equal(ctx.measured.warmup, PROTOCOL.warmup);
            const key = id === 'scaling' ? 'bhWall' : 'wallMs';
            for (const row of rows) assert.ok(row[`${key}Lo`] <= row[key] && row[key] <= row[`${key}Hi`], `${id}: q1 ≤ median ≤ q3`);
        } else {
            assert.equal(ctx.measured.calls, 0, 'energy-drift reports single runs');
        }
        const counts = simParamsWrites(device).map(s => s.count);
        assert.ok(counts.length > 0 && Math.max(...counts) <= BH_MAX_N, `${id}: N ≤ ${BH_MAX_N}`);
        if (id === 'scaling') {
            assert.deepEqual(rows.map(r => r.n), SCALING_COUNTS);
            for (const r of rows) assert.equal(r.bruteWall === null, r.n > BRUTE_MAX_N, `brute force only up to ${BRUTE_MAX_N}`);
            for (const r of rows) for (const k of ['compileMs', 'uploadMs', 'readbackMs']) assert.ok(r[k] >= 0, `phase ${k}`);
            // 1 warm-up + 5 repeats per mode and N, one SimParams write each
            const evaluations = SCALING_COUNTS.length * 6 + SCALING_COUNTS.filter(n => n <= BRUTE_MAX_N).length * 6;
            assert.equal(simParamsWrites(device).length, evaluations);
        }
        if (id === 'energy-drift') {
            assert.deepEqual(rows.map(r => r.steps), DTS.map(dt => Math.round(DRIFT_TIME / dt)));
        }
    }
});
