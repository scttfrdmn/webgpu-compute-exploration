// N-body compute-only API (measureForces, measureEnergyDrift in js/examples/nbody.js) and the Measure
// experiments built on it (js/teach/experiments/nbody.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/nbody.gpu.test.mjs, which rebuilds the dispatch sequence with explicit layouts, this
// runs the example's own pipelines, bind groups and encodeForces / encodeCompute through a
// NBodySimulation without a canvas, exactly as the Measure tab does in a browser. Checked against:
//   forces: the float64 direct sum (js/refs/nbody.js) with the bounds of nbody.gpu.test.mjs;
//   energy: E₀ = Σ ½ m v² + ½ Σ m φ in float64 from the same bodies; the drift stays small and shrinks with dt;
//   experiments: every row has the declared columns, and the trends the Understand page describes hold.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { generateBodies, measureForces, measureEnergyDrift } from '../../js/examples/nbody.js';
import { directForces, relErrors } from '../../js/refs/nbody.js';
import experiments from '../../js/teach/experiments/nbody.js';
import { makeTestContext } from '../../js/teach/measure.js';
import { buildResultRecord, validateResultRecord, SCHEMA_PATH } from '../../js/teach/result-record.js';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const EPS = 0.05;

const accel3 = (accel, n) => Float64Array.from({ length: 3 * n }, (_, i) => accel[4 * Math.floor(i / 3) + (i % 3)]);

test('measureForces: Barnes-Hut (θ = 0.5, θ → 0) and brute force vs float64 direct sum', async (t) => {
    const n = 4096;
    const { positions } = generateBodies('plummer', n, EPS, 5);
    const ref = directForces(positions, n, EPS * EPS);
    for (const theta of [0.5, 1e-6]) {
        const r = await measureForces(gpuDevice, { positions, mode: 'bh', theta, softening: EPS, repeats: 2 });
        assert.equal(r.dispatches, 31);
        assert.equal(r.accel.length, 4 * n);
        assert.ok(r.accel.every(Number.isFinite));
        const e = relErrors(accel3(r.accel, n), ref.acc, n);
        let phiErr = 0;
        for (let i = 0; i < n; i++) phiErr = Math.max(phiErr, Math.abs(r.accel[4 * i + 3] - ref.phi[i]) / Math.abs(ref.phi[i]));
        t.diagnostic(`BH θ=${theta}: median ${e.median.toExponential(2)}, p99 ${e.p99.toExponential(2)}, max ${e.max.toExponential(2)}; `
            + `potential max ${phiErr.toExponential(2)}; interactions/body ${r.interactionsPerBody.toFixed(1)}; wall ${r.wallMs.toFixed(1)} ms`);
        if (theta === 0.5) {
            // the bounds of nbody.gpu.test.mjs (same bodies, same shader)
            assert.ok(e.median < 7e-3 && e.p99 < 7e-2, `θ=0.5 median ${e.median} p99 ${e.p99}`);
            assert.ok(r.interactionsPerBody > 100 && r.interactionsPerBody < n - 1);
        } else {
            assert.equal(r.interactionsPerBody, n - 1, 'every leaf summed');
            assert.ok(e.p99 < 1e-4 && phiErr < 1e-5, `θ→0 p99 ${e.p99} potential ${phiErr}`);
        }
    }

    const m = 3000;
    const bodies = generateBodies('collision', m, EPS, 2).positions;
    const brute = await measureForces(gpuDevice, { positions: bodies, mode: 'brute', softening: EPS });
    const e = relErrors(accel3(brute.accel, m), directForces(bodies, m, EPS * EPS).acc, m);
    t.diagnostic(`brute N=${m}: median ${e.median.toExponential(2)}, p99 ${e.p99.toExponential(2)}, max ${e.max.toExponential(2)}`);
    assert.equal(brute.dispatches, 1);
    assert.equal(brute.interactionsPerBody, m - 1);
    assert.ok(e.p99 < 1e-4 && e.max < 1e-3, `brute p99 ${e.p99} max ${e.max}`);
});

test('measureEnergyDrift: E₀ matches the float64 energy; drift stays small (brute force and Barnes-Hut)', async (t) => {
    const n = 1024;
    const { positions, velocities } = generateBodies('plummer', n, EPS, 5);
    const { phi } = directForces(positions, n, EPS * EPS);
    let K = 0, W = 0;
    for (let i = 0; i < n; i++) {
        const m = positions[4 * i + 3];
        K += 0.5 * m * (velocities[4 * i] ** 2 + velocities[4 * i + 1] ** 2 + velocities[4 * i + 2] ** 2);
        W += 0.5 * m * phi[i];
    }
    const E0 = K + W;
    for (const mode of ['brute', 'bh']) {
        let calls = 0;
        const r = await measureEnergyDrift(gpuDevice, { positions, velocities, mode, theta: 0.5, softening: EPS,
            dt: 0.01, steps: 50, samples: 5, between: async () => { calls++; } });
        const e0Err = Math.abs(r.E0 - E0) / Math.abs(E0);
        t.diagnostic(`${mode}: E0 ${r.E0.toFixed(6)} (float64 ${E0.toFixed(6)}, rel ${e0Err.toExponential(2)}); `
            + `drift ${r.samples.map(s => s.drift.toExponential(1)).join(' ')}; max ${r.maxDrift.toExponential(2)}; wall ${r.wallMs.toFixed(0)} ms`);
        assert.deepEqual(r.samples.map(s => s.step), [10, 20, 30, 40, 50]);
        assert.equal(calls, 5);
        assert.ok(Math.abs(r.samples[4].time - 0.5) < 1e-9);
        // brute force: exact pair sum in f32; Barnes-Hut: W from the tree potential (≲ 4e-3 per body at θ = 0.5)
        assert.ok(e0Err < (mode === 'brute' ? 1e-5 : 5e-3), `${mode} E0 rel error ${e0Err}`);
        assert.ok(r.maxDrift < 1e-2, `${mode} drift ${r.maxDrift}`);
    }
});

// The Measure tab's ctx without the runner's wall budget (a software GPU is far slower than a visitor's)
const testContext = () => makeTestContext(gpuDevice);

const schema = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../..', SCHEMA_PATH), 'utf8'));

// Runs an experiment and checks what the Measure tab would record: a schema-valid result record whose
// verification is `expected` ('pass' for every check, or 'not-checked'), and q1 ≤ median ≤ q3 for every
// column with a spread
async function runAll(experiment, settings, t, expected = 'pass') {
    const rows = [];
    const ctx = testContext();
    const t0 = performance.now();
    for await (const row of experiment.run(ctx, settings)) rows.push(row);
    const seconds = (performance.now() - t0) / 1000;
    const record = buildResultRecord({ exampleId: 'nbody', experiment, settings, rows, outcome: { status: 'done' },
        measured: ctx.measured, gpuDevice, source: { version: 'ci', commit: 'local', url: null },
        userAgent: 'node', budgetMs: experiment.budgetMs });
    assert.deepEqual(validateResultRecord(record, schema).errors, [], 'record matches schema v1');
    t?.diagnostic(`verify: ${record.verification.map(c => `${c.status} ${c.metric}${c.value === null ? '' : ` = ${c.value.toExponential(2)}`}`).join('; ')}`);
    for (const c of record.verification) assert.equal(c.status, expected, `${c.metric}: ${c.value} (required ${c.tolerance})`);
    for (const col of record.columns.filter(c => c.range)) {
        for (const row of rows) {
            if (row[col.key] === null) continue;
            assert.ok(row[`${col.key}Lo`] <= row[col.key] && row[col.key] <= row[`${col.key}Hi`], `${col.key}: q1 ≤ median ≤ q3`);
        }
    }
    return { rows, seconds, record, ctx };
}

const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');

test('experiment theta-error (Plummer N = 4,096): error grows ~θ², interactions fall', async (t) => {
    const e = byId['theta-error'];
    const { rows, seconds, record } = await runAll(e, { n: 4096, preset: 'plummer' }, t);
    t.diagnostic(`${seconds.toFixed(1)} s: θ median p99 interactions: ${show(rows, ['theta', 'median', 'p99', 'interactions'])}; `
        + `wall ms median (q1–q3): ${rows.map(r => `${r.wallMs.toFixed(1)} (${r.wallMsLo.toFixed(1)}–${r.wallMsHi.toFixed(1)})`).join(' ')}`);
    assert.deepEqual([record.protocol.warmup, record.protocol.repeats, record.protocol.statistic, record.protocol.spread], [1, 5, 'median', 'iqr']);
    assert.equal(record.verification.length, 7);
    assert.equal(rows.length, 6);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    for (let i = 1; i < rows.length; i++) {
        assert.ok(rows[i].median > rows[i - 1].median, 'median error grows with θ');
        assert.ok(rows[i].interactions < rows[i - 1].interactions, 'fewer interactions with larger θ');
    }
    const half = rows.find(r => r.theta === 0.5);
    assert.ok(half.median > 2e-3 && half.median < 7e-3, `median at θ = 0.5: ${half.median}`);
    // θ² from 0.3 to 1.2 would be 16×; allow a generous range around that
    const ratio = rows[5].median / rows[0].median;
    assert.ok(ratio > 6 && ratio < 40, `median(1.2)/median(0.3) = ${ratio}`);
});

test('experiment energy-drift (brute force, Plummer N = 2,048): error shrinks with dt', async (t) => {
    const e = byId['energy-drift'];
    const { rows, seconds, record } = await runAll(e, { n: 2048, mode: 'brute', preset: 'plummer' }, t);
    assert.equal(record.verification.length, 6);
    assert.equal(record.protocol.statistic, 'as-reported');
    t.diagnostic(`${seconds.toFixed(1)} s: dt steps max|dE/E0| final: ${show(rows, ['dt', 'steps', 'maxDrift', 'finalDrift'])}`);
    assert.equal(rows.length, 4);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.deepEqual(rows.map(r => r.steps), [50, 100, 200, 400]);
    assert.ok(rows.every(r => Number.isFinite(r.maxDrift) && r.maxDrift < 5e-2));
    assert.ok(rows[0].maxDrift > rows[1].maxDrift && rows[1].maxDrift > rows[2].maxDrift, 'drift falls from dt = 0.04 to 0.01');
    // second order: about 4× per halving of dt from 0.02 down (dt = 0.04 is outside the asymptotic range)
    for (let i = 2; i < rows.length; i++) {
        const r = rows[i - 1].maxDrift / rows[i].maxDrift;
        assert.ok(r > 2 && r < 8, `dt ${rows[i - 1].dt} → ${rows[i].dt}: ratio ${r}`);
    }

    // Barnes-Hut at θ = 0.5: the tree's force/potential error sets a floor that a smaller dt cannot remove
    const bh = await runAll(e, { n: 2048, mode: 'bh', preset: 'plummer' }, t);
    t.diagnostic(`BH ${bh.seconds.toFixed(1)} s: dt steps max|dE/E0| final: ${show(bh.rows, ['dt', 'steps', 'maxDrift', 'finalDrift'])}`);
    assert.equal(bh.rows.length, 4);
    assert.ok(bh.rows.every(r => r.maxDrift < 1e-2));
    assert.ok(bh.rows[3].maxDrift > 10 * rows[3].maxDrift, 'Barnes-Hut floor above the brute-force error at dt = 0.005');
});

test('experiment scaling (Plummer): rows up to the caps, brute force stops at 16,384', async (t) => {
    const e = byId['scaling'];
    const { rows, seconds, record } = await runAll(e, { preset: 'plummer' }, t, 'not-checked');
    t.diagnostic(`${seconds.toFixed(1)} s: N bhWall bruteWall interactions: ${show(rows, ['n', 'bhWall', 'bruteWall', 'interactions'])}`);
    t.diagnostic(`BH wall (q1–q3): ${rows.map(r => `${r.bhWallLo.toFixed(1)}–${r.bhWallHi.toFixed(1)}`).join(' ')}; `
        + `phases compile/upload/readback: ${show(rows, ['compileMs', 'uploadMs', 'readbackMs'])}`);
    assert.deepEqual(record.columns.filter(c => c.range).map(c => c.key),
        ['bhGpu', 'bruteGpu', 'bhWall', 'bruteWall', 'compileMs', 'uploadMs', 'readbackMs'].filter(k => rows.some(r => r[`${k}Lo`] !== null)));
    for (const row of rows) for (const k of ['compileMs', 'uploadMs', 'readbackMs']) assert.ok(row[k] >= 0, `${k} at N = ${row.n}`);
    assert.deepEqual(rows.map(r => r.n), [1024, 2048, 4096, 8192, 16384, 32768, 65536]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.bhWall > 0 && row.interactions > 0 && row.interactions < row.n - 1);
        assert.equal(row.bruteWall === null, row.n > 16384);
    }
});
