// Gaussian blur compute-only API (measureBlur / blurShaderCode in js/examples/image-blur.js) and its
// Measure experiments, on a software GPU — CI only, see harness.mjs. Checked against the separable
// float64 blur (js/refs/basic.js). The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device, rng } from './harness.mjs';
import { measureBlur, createTestImage, BLUR_SIGMAS, SHIPPED_SIGMA } from '../../js/examples/image-blur.js';
import experiments, { samplePixels, maxErrorVsSeparable } from '../../js/teach/experiments/gaussian-blur.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureBlur: every σ variant on a random 61 × 37 image, every pixel vs float64', async (t) => {
    const [w, h] = [61, 37];
    const rand = rng(3);
    const image = Float32Array.from({ length: 4 * w * h }, () => rand());
    const all = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) all.push([x, y]);
    for (const sigma of BLUR_SIGMAS) {
        const r = await measureBlur(gpuDevice, { image, width: w, height: h, sigma, repeats: 1 });
        const err = maxErrorVsSeparable(image, r.output, w, h, sigma, all);
        t.diagnostic(`σ = ${sigma} (${r.kernelSize}²): max |GPU − f64| = ${err.toExponential(2)}`);
        assert.ok(err < 1e-5);
    }
});

test('experiment sigma-cost (256 × 256): f32-level error for every σ, cost grows with the taps', async (t) => {
    const e = experiments.find(x => x.id === 'sigma-cost');
    const rows = [];
    for await (const row of e.run(makeTestContext(gpuDevice), { side: 256 })) rows.push(row);
    t.diagnostic('σ taps wallMs Gtaps/s(wall) maxError: ' + rows.map(r => [r.sigma, r.taps, r.wallMs.toFixed(2),
        (256 * 256 * r.taps / r.wallMs / 1e6).toFixed(3), r.maxError.toExponential(2)].join(' ')).join('; '));
    assert.deepEqual(rows.map(r => r.sigma), BLUR_SIGMAS);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, c.key);
        assert.ok(row.maxError < 1e-5, `σ = ${row.sigma}: ${row.maxError}`);
    }
    assert.ok(rows.at(-1).wallMs > 5 * rows[0].wallMs, '2,401 taps cost more than 49');
});

test('size scaling at σ = 4 (128² … 512², direct calls; the experiment goes to 2,048²)', async (t) => {
    const out = [];
    for (const side of [128, 256, 512]) {
        const image = createTestImage(side, side);
        const r = await measureBlur(gpuDevice, { image, width: side, height: side, sigma: SHIPPED_SIGMA, repeats: 2 });
        const err = maxErrorVsSeparable(image, r.output, side, side, SHIPPED_SIGMA, samplePixels(side, side));
        out.push(`${side}²: ${r.wallMs.toFixed(2)} ms wall, ${(side * side / r.wallMs / 1e3).toFixed(2)} Mpixel/s, err ${err.toExponential(2)}`);
        assert.ok(err < 1e-5);
    }
    t.diagnostic(out.join('; '));
    // the experiment itself: declared columns on its first row
    const e = experiments.find(x => x.id === 'size-scaling');
    const first = (await e.run(makeTestContext(gpuDevice), {}).next()).value;
    for (const c of e.columns) assert.ok(c.key in first, c.key);
});
