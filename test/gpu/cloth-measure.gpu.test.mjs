// Cloth compute-only API (measureCloth in js/examples/cloth.js) and the Measure experiments built on it
// (js/teach/experiments/cloth.js), on a software GPU — CI only, see harness.mjs.
//
// measureCloth runs the example's own pipelines, bind groups and encodeCompute through a ClothSimulation
// without a canvas, exactly as the Measure tab does in a browser. Checked against:
//   the CPU replica of the compute shader (test/helpers/cloth-refs.mjs), driven by the same SimParams
//   bytes, chunk by chunk: positions and the stretch statistics of js/refs/cloth.js agree;
//   experiments: every row has the declared columns, and the trends the Understand page describes hold.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureCloth, buildCloth, MAX_SUBSTEPS, SUBSTEP_DT } from '../../js/examples/cloth.js';
import { stretchStats } from '../../js/refs/cloth.js';
import experiments from '../../js/teach/experiments/cloth.js';
import { makeTestContext } from '../../js/teach/measure.js';
import { makeState, frame, simParamsFor } from '../helpers/cloth-refs.mjs';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureCloth (32 × 32, two corners, 8 iterations, 1 s) vs the CPU replica', async (t) => {
    const n = 32, iterations = 8, substeps = 240, timed = 16;
    const t0 = performance.now();
    const r = await measureCloth(gpuDevice, { n, preset: 'corners', iterations, substeps, timed });
    const gpuSeconds = (performance.now() - t0) / 1000;
    assert.equal(r.count, n * n);
    assert.equal(r.dispatchesPerSubstep, 3 + 12 * iterations);
    assert.ok(Math.abs(r.simulatedSeconds - (substeps + timed) * SUBSTEP_DT) < 1e-9);
    assert.ok(r.positions.every(Number.isFinite));
    assert.ok(r.wallMsPerSubstep > 0);

    // Replica: the normals pass of reset(), then the same chunks with the same simulated time
    const cloth = buildCloth(n, 'corners');
    const P = simParamsFor({ gridN: n, presetKey: 'corners', tearOn: false, windStrength: 0, stiffness: 1, iterations, time: 0 });
    const S = makeState(cloth, P);
    frame(S, 0, iterations);
    for (let done = 0; done < substeps + timed; done += MAX_SUBSTEPS) {
        P.time = done * SUBSTEP_DT;
        frame(S, MAX_SUBSTEPS, iterations);
    }
    let maxDiff = 0;
    for (let i = 0; i < r.positions.length; i++) maxDiff = Math.max(maxDiff, Math.abs(r.positions[i] - S.pos[i]));
    const g = stretchStats(r.positions, r.constraintData), c = stretchStats(S.pos, cloth.constraintData);
    t.diagnostic(`GPU ${gpuSeconds.toFixed(1)} s, ${r.wallMsPerSubstep.toFixed(2)} ms/substep wall; max |x_gpu − x_replica| `
        + `${maxDiff.toExponential(2)} m; stretch max ${g.max.toExponential(3)} (replica ${c.max.toExponential(3)}), `
        + `p99 ${g.p99.toExponential(3)} (${c.p99.toExponential(3)}), mean ${g.mean.toExponential(3)} (${c.mean.toExponential(3)})`);
    // Pins never move. f32 (and fused multiply-adds) vs the replica's float64 arithmetic: the swinging
    // cloth amplifies rounding differences to millimetres in 256 substeps, but the stretch statistics, which
    // the experiments report, agree to a fraction of a percent
    for (let i = 0; i < r.count; i++) if (cloth.positions[4 * i + 3] === 0) {
        for (let k = 0; k < 3; k++) assert.equal(r.positions[4 * i + k], cloth.positions[4 * i + k]);
    }
    assert.ok(maxDiff < 0.03, `positions differ by ${maxDiff}`);
    assert.ok(Math.abs(g.max - c.max) < 0.05 * c.max + 1e-5, `max stretch ${g.max} vs ${c.max}`);
    assert.ok(Math.abs(g.p99 - c.p99) < 0.05 * c.p99 + 1e-5, `p99 stretch ${g.p99} vs ${c.p99}`);
});

const testContext = () => makeTestContext(gpuDevice);
async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(testContext(), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}
const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');

test('experiment iterations-stretch (32 × 32, two corners): stretch falls with iterations, cost grows', async (t) => {
    const e = byId['iterations-stretch'];
    const { rows, seconds } = await runAll(e, { n: 32, preset: 'corners' });
    t.diagnostic(`${seconds.toFixed(1)} s: iterations maxStretch p99 mean wallMs: ${show(rows, ['iterations', 'maxStretch', 'p99Stretch', 'meanStretch', 'wallMs'])}`);
    assert.deepEqual(rows.map(r => r.iterations), [1, 2, 4, 8, 16, 30]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.ok(rows.every(r => r.maxStretch > 0 && r.maxStretch < 0.1 && r.p99Stretch <= r.maxStretch));
    // The 99th percentile falls steadily; the maximum is one link and can jump with the swing of the cloth
    for (let i = 1; i < rows.length; i++) assert.ok(rows[i].p99Stretch < rows[i - 1].p99Stretch, 'p99 stretch falls with iterations');
    assert.ok(rows[5].p99Stretch < rows[0].p99Stretch / 4, '30 iterations at least 4× tighter than 1');
    assert.ok(rows[5].maxStretch < rows[0].maxStretch / 2);
    assert.ok(rows[5].wallMs > rows[0].wallMs, 'more iterations cost more');
});

test('experiment grid-scaling (4 iterations): every grid size, cost grows with N', async (t) => {
    const e = byId['grid-scaling'];
    const { rows, seconds } = await runAll(e, { iterations: 4 });
    t.diagnostic(`${seconds.toFixed(1)} s: n wallMs Mproj/s: ${show(rows, ['n', 'wallMs', 'mProjPerS'])}`);
    assert.deepEqual(rows.map(r => r.n), [32, 48, 64, 96, 128]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.ok(rows.every(r => r.wallMs > 0 && r.mProjPerS > 0));
    assert.ok(rows[4].wallMs > rows[0].wallMs);
});
