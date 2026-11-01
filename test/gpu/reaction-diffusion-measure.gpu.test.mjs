// Reaction-diffusion compute-only API (measureSteps in js/examples/reaction-diffusion.js) and the Measure
// experiments built on it (js/teach/experiments/reaction-diffusion.js), on a software GPU — CI only, see
// harness.mjs. Unlike reaction-diffusion.gpu.test.mjs, which binds the shader itself, this runs the
// example's own pipeline and bind groups through a ReactionDiffusion without a canvas, exactly as the
// Measure tab does in a browser. Checked against the float64 step of js/refs/reaction-diffusion.js with
// the example's exported constants. The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureSteps, seedCells, seededRandom, PRESETS } from '../../js/examples/reaction-diffusion.js';
import { stepReference } from '../../js/refs/reaction-diffusion.js';
import experiments, { MODEL, compare } from '../../js/teach/experiments/reaction-diffusion.js';
import { makeTestContext } from '../../js/teach/measure.js';

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

test('measureSteps: 300 steps on a ragged 60 x 44 torus vs the float64 reference, read back at 1 / 100 / 300', async (t) => {
    const w = 60, h = 44;
    const init = seedCells(w, h, seededRandom(3));
    for (const key of ['coral', 'maze']) {
        const { feed, kill } = PRESETS[key];
        const r = await measureSteps(gpuDevice, { width: w, height: h, cells: init, feed, kill, steps: 300, readAt: [1, 100, 300] });
        assert.deepEqual(r.samples.map(s => s.step), [1, 100, 300]);
        let U = Float64Array.from({ length: w * h }, (_, i) => init[2 * i]);
        let V = Float64Array.from({ length: w * h }, (_, i) => init[2 * i + 1]);
        let done = 0;
        const errs = [];
        for (const s of r.samples) {
            while (done < s.step) { [U, V] = stepReference(U, V, w, h, feed, kill, MODEL); done++; }
            assert.ok(s.cells.every(Number.isFinite));
            errs.push(compare(s.cells, U, V).max);
        }
        t.diagnostic(`${key}: max |gpu − f64| after 1 / 100 / 300 steps: ${errs.map(e => e.toExponential(2)).join(' / ')}; `
            + `wall ${r.wallMsPerStep.toFixed(3)} ms per step`);
        assert.ok(errs[0] < 1e-6, 'one step: f32 rounding only');
        assert.ok(errs[2] < 1e-4, 'the bound of reaction-diffusion.gpu.test.mjs');
    }
});

test('experiment precision (Coral, Chaos): f32 vs float64 over 4,000 steps', async (t) => {
    const e = byId.precision;
    const out = {};
    for (const preset of ['coral', 'chaos']) {
        const { rows, seconds } = await runAll(e, { preset });
        t.diagnostic(`${preset} ${seconds.toFixed(1)} s: steps max rms coverage: ${show(rows, ['steps', 'maxError', 'rmsError', 'coverage'])}`);
        assert.equal(rows.length, 8);
        for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(rows[0].maxError < 1e-6, `one step: ${rows[0].maxError}`);
        out[preset] = rows;
    }
    // Coral stays close to the float64 solution; Chaos amplifies the rounding differences
    assert.ok(out.coral[7].maxError < 1e-2, `coral after 4,000 steps: ${out.coral[7].maxError}`);
    assert.ok(out.chaos[7].maxError > out.chaos[3].maxError, 'chaos: the difference grows');
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
