// Ray marching compute-only API (measureRayMarch and MARCH_STATS_SHADER in js/examples/ray-marching.js)
// and its Measure experiment, on a software GPU — CI only, see harness.mjs. Checked against the float64
// sphere tracer (js/refs/ray-marching.js) pixel by pixel at a reduced resolution. The printed numbers are
// the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureRayMarch, FRACTALS } from '../../js/examples/ray-marching.js';
import experiments, { agreement } from '../../js/teach/experiments/ray-marching.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('statistics kernel vs float64 tracer: every pixel of a 160 × 112 view, all four fractals', async (t) => {
    const [w, h] = [160, 112];
    const all = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) all.push([x, y]);
    for (const fractal of FRACTALS) {
        const r = await measureRayMarch(gpuDevice, { width: w, height: h, fractal, maxSteps: 128, repeats: 1, readback: true });
        const agree = agreement(r.pixels, w, h, FRACTALS.indexOf(fractal), 128, all);
        t.diagnostic(`${fractal}: hit ${(100 * r.hit).toFixed(1)}% escaped ${(100 * r.escaped).toFixed(1)}% out of steps `
            + `${(100 * r.exhausted).toFixed(1)}%, mean steps ${r.meanSteps.toFixed(1)}, agreement with f64 ${(100 * agree).toFixed(2)}%, frame wall ${r.wallMs.toFixed(1)} ms`);
        assert.ok(r.hit > 0.05, `${fractal} is visible`);
        assert.ok(agree > 0.97, `${fractal}: ${agree}`);
    }
});

test('experiment max-steps (Mandelbulb, 500 × 350): fewer pixels run out of steps as the budget grows', async (t) => {
    const [e] = experiments;
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(makeTestContext(gpuDevice), { fractal: 'mandelbulb', width: 500 })) rows.push(row);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: maxSteps wallMs hit% esc% exh% meanSteps agree%: ` + rows.map(r => [r.maxSteps,
        r.wallMs.toFixed(1), (100 * r.hit).toFixed(1), (100 * r.escaped).toFixed(1), (100 * r.exhausted).toFixed(1), r.meanSteps.toFixed(1),
        (100 * r.agreement).toFixed(1)].join(' ')).join('; '));
    assert.equal(rows.length, 8);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, c.key);
    for (let i = 1; i < rows.length; i++) assert.ok(rows[i].exhausted <= rows[i - 1].exhausted + 1e-9);
    assert.ok(rows[0].exhausted > 0.2 && rows.at(-1).exhausted < 0.01);
    assert.ok(rows.every(r => r.agreement > 0.95));
});
