// Path tracer compute-only API (measurePathTracer in js/examples/path-tracer.js) and the Measure
// experiments built on it (js/teach/experiments/path-tracer.js), on a software GPU — CI only, see harness.mjs.
//
// measurePathTracer runs the example's own trace pipeline, bind group and Params writer through a
// PathTracer without a canvas, one sample per pixel per dispatch, as the Measure tab does in a browser.
// Checked against:
//   the closed-form furnace box (js/refs/path-tracer.js furnaceRadiance): the image mean is unbiased
//   (within its standard error) for BSDF sampling and for NEE + MIS, at several path lengths;
//   experiments: every row has the declared columns, the noise falls as 1/√spp, the bounce sums match.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measurePathTracer, FURNACE } from '../../js/examples/path-tracer.js';
import { furnaceRadiance, imageRmse } from '../../js/refs/path-tracer.js';
import experiments from '../../js/teach/experiments/path-tracer.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

// Mean and standard error over pixels of the red channel (the furnace is gray: r = g = b per sample)
function meanAndError(image) {
    let s = 0, s2 = 0;
    for (let i = 0; i < image.length; i += 3) { s += image[i]; s2 += image[i] * image[i]; }
    const n = image.length / 3, mean = s / n;
    return { mean, se: Math.sqrt(Math.max(s2 / n - mean * mean, 0) / n) };
}

test('measurePathTracer: furnace box mean equals Le·Σρᵏ (BSDF only and NEE + MIS, 2 and 5 bounces)', async (t) => {
    for (const useNEE of [false, true]) {
        for (const bounces of [2, 5]) {
            const t0 = performance.now();
            const r = await measurePathTracer(gpuDevice, { scene: 'furnace', width: 64, height: 48, checkpoints: [4, 32],
                bounces, useNEE, seed: 3 });
            const exact = furnaceRadiance(FURNACE.rho, FURNACE.le, bounces);
            assert.equal(r.bounces, bounces);
            assert.deepEqual(r.checkpoints.map(c => c.spp), [4, 32]);
            const c = r.checkpoints[1];
            assert.equal(c.image.length, 64 * 48 * 3);
            assert.ok(c.image.every(Number.isFinite));
            // The pixel means at 32 spp are independent estimates of `exact`: their mean has standard error se
            const { mean, se } = meanAndError(c.image);
            t.diagnostic(`NEE ${useNEE ? 'on ' : 'off'}, ${bounces} bounces: mean ${mean.toFixed(5)} ± ${se.toFixed(5)} `
                + `(exact ${exact.toFixed(5)}, ${((mean - exact) / se).toFixed(2)} σ); rays/pixel ${c.raysPerPixel.toFixed(2)}; `
                + `relative RMSE ${imageRmse(r.checkpoints[0].image, exact).relative.toExponential(2)} at 4 spp, `
                + `${imageRmse(c.image, exact).relative.toExponential(2)} at 32; ${((performance.now() - t0) / 1000).toFixed(1)} s`);
            assert.ok(Math.abs(mean - exact) < 5 * se + 1e-4, `mean ${mean} vs ${exact} (se ${se})`);
            assert.ok(Math.abs(mean - exact) / exact < 0.01);
            assert.ok(c.raysPerPixel >= 1 && c.raysPerPixel <= 2 * bounces + 1);
        }
    }
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

// Least-squares slope of log(y) against log(x)
function logSlope(rows, x, y) {
    const pts = rows.map(r => [Math.log(r[x]), Math.log(r[y])]);
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    let num = 0, den = 0;
    for (const [a, b] of pts) { num += (a - mx) * (b - my); den += (a - mx) ** 2; }
    return num / den;
}

test('experiment spp-noise (furnace, NEE on and off): relative RMSE falls as 1/√spp, mean unbiased', async (t) => {
    const e = byId['spp-noise'];
    for (const nee of [true, false]) {
        const { rows, seconds } = await runAll(e, { scene: 'furnace', nee });
        const slope = logSlope(rows, 'spp', 'relRmse');
        t.diagnostic(`NEE ${nee}: ${seconds.toFixed(1)} s, slope ${slope.toFixed(3)}: spp relRmse bias: ${show(rows, ['spp', 'relRmse', 'bias'])}`);
        assert.deepEqual(rows.map(r => r.spp), [1, 2, 4, 8, 16, 32, 64, 128, 256]);
        for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(slope > -0.6 && slope < -0.4, `log-log slope ${slope}`);
        assert.ok(Math.abs(rows[8].bias) < 0.01 * rows[8].mean);
    }
});

test('experiment spp-noise (Cornell box, 2,048-spp reference): noise falls with spp', async (t) => {
    const e = byId['spp-noise'];
    const { rows, seconds } = await runAll(e, { scene: 'cornell', nee: true });
    const slope = logSlope(rows, 'spp', 'relRmse');
    t.diagnostic(`${seconds.toFixed(1)} s, slope ${slope.toFixed(3)}: spp relRmse wallMs: ${show(rows, ['spp', 'relRmse', 'wallMs'])}`);
    assert.equal(rows.length, 9);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.ok(rows.every(r => r.bias === null));
    assert.ok(rows[8].relRmse < rows[0].relRmse / 4, 'at least 4× less noise at 256 spp than at 1');
    assert.ok(slope > -0.65 && slope < -0.3, `log-log slope ${slope}`);
});

test('experiment bounces (furnace and Cornell box): means follow the geometric series, rays grow', async (t) => {
    const e = byId['bounces'];
    const furnace = await runAll(e, { scene: 'furnace' });
    t.diagnostic(`furnace ${furnace.seconds.toFixed(1)} s: bounces rays mean exact wallMs: ${show(furnace.rows, ['bounces', 'raysPerPixel', 'mean', 'exact', 'wallMs'])}`);
    assert.deepEqual(furnace.rows.map(r => r.bounces), [1, 2, 3, 4, 5, 6, 7, 8]);
    for (const row of furnace.rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(Math.abs(row.mean - row.exact) / row.exact < 0.01, `bounces ${row.bounces}: ${row.mean} vs ${row.exact}`);
    }
    for (let i = 1; i < furnace.rows.length; i++) assert.ok(furnace.rows[i].raysPerPixel >= furnace.rows[i - 1].raysPerPixel - 1e-9);
    const cornell = await runAll(e, { scene: 'cornell' });
    t.diagnostic(`Cornell ${cornell.seconds.toFixed(1)} s: bounces rays mean wallMs: ${show(cornell.rows, ['bounces', 'raysPerPixel', 'mean', 'wallMs'])}`);
    assert.ok(cornell.rows.every(r => r.exact === null && r.mean > 0));
    assert.ok(cornell.rows[7].mean > cornell.rows[0].mean, 'indirect light adds energy');
});
