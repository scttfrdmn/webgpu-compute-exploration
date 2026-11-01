// Wave-equation compute-only API (measureWave in js/examples/wave-equation.js) and the Measure
// experiments built on it (js/teach/experiments/wave-equation.js), on a software GPU — CI only, see
// harness.mjs. This runs the example's own pipelines and dynamic-offset bind groups through a
// WaveEquation without a canvas, exactly as the Measure tab does in a browser. Checked against the
// cs_step replica (test/helpers/wave-equation-refs.mjs), the discrete energy and the von Neumann growth
// factor of js/refs/wave.js. The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureWave, CFL_LIMIT } from '../../js/examples/wave-equation.js';
import { growthFactor } from '../../js/refs/wave.js';
import experiments from '../../js/teach/experiments/wave-equation.js';
import { makeTestContext } from '../../js/teach/measure.js';
import { presetState, stepReplica, energy } from '../helpers/wave-equation-refs.mjs';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const byId = Object.fromEntries(experiments.map(e => [e.id, e]));

async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(makeTestContext(gpuDevice), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');
const maxAbs = u => u.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

test('measureWave: both time levels vs the cs_step replica (open water, C = 0.5, 120 steps)', async (t) => {
    const W = 96, H = 72;
    const st = presetState('open', W, H);
    const rep = { W, H, cur: Float32Array.from(st.cur), prev: Float32Array.from(st.prev), walls: st.walls };
    const o = { courant2: Math.fround(0.25), damping: 0, absorb: false, planeWave: false, step: 0, period: 20 };
    const u0 = maxAbs(st.cur);
    const errs = [], energies = [];
    let done = 0;
    const r = await measureWave(gpuDevice, { width: W, height: H, preset: 'open', courant: 0.5, damping: 0, steps: 120, readEvery: 40,
        onSample: ({ step, cur, prev }) => {
            while (done < step) { stepReplica(rep, o); done++; }
            let e = 0;
            for (let i = 0; i < W * H; i++) e = Math.max(e, Math.abs(cur[i] - rep.cur[i]), Math.abs(prev[i] - rep.prev[i]));
            errs.push(e);
            energies.push(energy(prev, cur, W, H, Math.fround(0.25)));
        } });
    t.diagnostic(`max |gpu − replica| at 0 / 40 / 80 / 120: ${errs.map(e => e.toExponential(2)).join(' / ')} (max|u₀| ${u0.toFixed(3)}); `
        + `energy ${energies.map(e => e.toExponential(6)).join(' ')}`);
    assert.equal(r.steps, 120);
    assert.equal(errs.length, 4, 'samples at 0, 40, 80, 120');
    assert.equal(errs[0], 0, 'step 0 is the uploaded preset');
    assert.ok(errs.every(e => e < 1e-4 * u0), 'the replica bound of wave-equation.gpu.test.mjs');
    assert.ok(energies.every(e => Math.abs(e - energies[0]) / energies[0] < 1e-5), 'discrete energy conserved');
});

test('experiment cfl: bounded below 1/√2, blow-up above it, faster further above', async (t) => {
    const e = byId.cfl;
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: C |g| steps growth drift blewUpAt: ${show(rows, ['courant', 'theory', 'steps', 'growth', 'drift', 'divergedAt'])}`);
    assert.equal(rows.length, 9);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        if (row.courant < CFL_LIMIT) {
            assert.equal(row.divergedAt, null, `C = ${row.courant} must stay stable`);
            assert.equal(row.steps, 1000);
            assert.ok(row.growth <= 1.5 && row.drift < 1e-4, `C = ${row.courant}: growth ${row.growth}, drift ${row.drift}`);
        } else {
            assert.ok(row.divergedAt > 0 && row.steps < 1000, `C = ${row.courant} must blow up`);
            assert.ok(row.theory > 1);
        }
    }
    const unstable = rows.filter(r => r.courant > CFL_LIMIT);
    for (let i = 1; i < unstable.length; i++) {
        assert.ok(unstable[i].divergedAt <= unstable[i - 1].divergedAt, 'larger C blows up no later');
    }
    // the number of steps to grow by DIVERGED from round-off follows log(growth) / log|g| roughly
    const ref = unstable.map(r => `${r.courant}: ${r.divergedAt} steps, |g|^steps ${Math.log10(growthFactor(r.courant) ** r.divergedAt).toFixed(1)} decades`);
    t.diagnostic(ref.join('; '));
});

test('experiment scaling (up to 512²): rows, positive throughput', async (t) => {
    const e = byId.scaling;
    const { rows, seconds } = await runAll(e, { maxSide: 512 });
    t.diagnostic(`${seconds.toFixed(1)} s: side steps wallMs/step Mcells/s: ${show(rows, ['side', 'steps', 'wallMsPerStep', 'wallRate'])}`);
    assert.deepEqual(rows.map(r => r.side), [128, 256, 512]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.wallMsPerStep > 0 && row.wallRate > 0);
    }
});
