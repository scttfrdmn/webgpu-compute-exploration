// Gaussian blur Measure tab, CPU only: blurShaderCode() / measureBlur() in js/examples/image-blur.js and
// the experiments in js/teach/experiments/gaussian-blur.js, on the mock device. Checked: the σ variants
// differ from the shipped shader only in its two constants (and σ = 4 is the shipped source), the
// dimensions uniform (vec2<u32>), buffer sizes, the caps, rows carry the declared columns. A CPU replica of
// the direct 2D sum agrees with the separable reference for every σ. Real blurs:
// test/gpu/gaussian-blur-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    blurShaderCode, measureBlur, wgslModules, createTestImage, BLUR_SIGMAS, SHIPPED_SIGMA, MAX_BLUR_SIDE, kernelSizeFor,
} from '../js/examples/image-blur.js';
import experiments, { SIGMA_SIDES, SCALING_SIDES, samplePixels, maxErrorVsSeparable } from '../js/teach/experiments/gaussian-blur.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { mulberry32 } from '../js/refs/random.js';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });

test('σ variants: only the two constants change; σ = 4 is the shipped module; all variants exported', () => {
    const shipped = wgslModules()[0].code;
    assert.equal(blurShaderCode(SHIPPED_SIGMA), shipped);
    assert.deepEqual(BLUR_SIGMAS.map(kernelSizeFor), [7, 13, 25, 49]);
    for (const s of BLUR_SIGMAS) {
        const code = blurShaderCode(s);
        assert.match(code, new RegExp(`const sigma = ${s}\\.0;`));
        assert.match(code, new RegExp(`const kernelSize = ${kernelSizeFor(s)};`));
        const strip = (c) => c.replace(/const sigma = [\d.]+;/, '').replace(/const kernelSize = \d+;/, '');
        assert.equal(strip(code), strip(shipped));
    }
    assert.deepEqual(wgslModules().map(m => m.name), ['compute', 'compute-sigma1', 'compute-sigma2', 'compute-sigma8']);
    assert.throws(() => blurShaderCode(3), RangeError);
});

test('measureBlur on a mock device: dimensions bytes, buffers, result shape, caps', async () => {
    const device = mockDevice();
    const [w, h] = [61, 37];
    const image = createTestImage(w, h);
    const r = await measureBlur({ device }, { image, width: w, height: h, sigma: 2, repeats: 2 });
    const dims = device.writes.find(x => x.buffer.usage & GPUBufferUsage.UNIFORM);
    assert.equal(dims.buffer.size, 8);
    assert.deepEqual([...new Uint32Array(dims.bytes.buffer)], [w, h]);
    assert.equal(device.buffers.filter(b => b.size === 16 * w * h).length, 3, 'input, output, readback');
    assert.deepEqual({ kernelSize: r.kernelSize, taps: r.taps, gpuMs: r.gpuMs }, { kernelSize: 13, taps: 169, gpuMs: null });
    assert.equal(r.output.length, 4 * w * h);
    await assert.rejects(measureBlur({ device }, { image, width: w, height: h, sigma: 3 }), RangeError);
    await assert.rejects(measureBlur({ device }, { image, width: w + 1, height: h }), RangeError);
    await assert.rejects(measureBlur({ device }, { image: { length: 4 * (MAX_BLUR_SIDE + 1) }, width: MAX_BLUR_SIDE + 1, height: 1 }), RangeError);
});

// The shader's direct sum: Σ img·g / Σ g over the taps inside the image, g = exp(−(x² + y²)/2σ²)/(2πσ²)
function directBlurPixel(img, w, h, x, y, sigma) {
    const half = Math.floor(kernelSizeFor(sigma) / 2);
    const out = [0, 0, 0, 0];
    let ws = 0;
    for (let ky = -half; ky <= half; ky++) for (let kx = -half; kx <= half; kx++) {
        const sx = x + kx, sy = y + ky;
        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
        const g = Math.exp(-(kx * kx + ky * ky) / (2 * sigma * sigma)) / (2 * 3.14159265 * sigma * sigma);
        for (let c = 0; c < 4; c++) out[c] += img[(sy * w + sx) * 4 + c] * g;
        ws += g;
    }
    return out.map(v => v / ws);
}

test('the direct 2D sum (shader replica) equals the separable float64 reference for every σ', (t) => {
    const rand = mulberry32(9);
    const [w, h] = [53, 41];
    const img = Float32Array.from({ length: 4 * w * h }, () => rand());
    const pts = samplePixels(w, h, 60);
    for (const sigma of BLUR_SIGMAS) {
        const out = new Float32Array(4 * w * h);
        for (const [x, y] of pts) out.set(directBlurPixel(img, w, h, x, y, sigma), 4 * (y * w + x));
        const err = maxErrorVsSeparable(img, out, w, h, sigma, pts);
        t.diagnostic(`σ = ${sigma}: max |direct (f32-rounded) − separable f64| = ${err.toExponential(2)}`);
        assert.ok(err < 1e-6, 'the two sums agree to f32 rounding of the output');
    }
});

test('experiments on a mock device: one row per sweep value, declared columns, caps', async () => {
    assert.ok([...SIGMA_SIDES, ...SCALING_SIDES].every(s => s <= MAX_BLUR_SIDE) && Math.max(...SIGMA_SIDES) <= 1024);
    for (const [id, settings, expected] of [['sigma-cost', { side: 256 }, BLUR_SIGMAS.length], ['size-scaling', {}, SCALING_SIDES.length]]) {
        const e = experiments.find(x => x.id === id);
        const device = mockDevice();
        const rows = await runExperiment(e, settings, mockCtx(device, e.budgetMs));
        assert.equal(rows.length, expected, id);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort(), id);
        assert.ok(Math.max(...device.buffers.map(b => b.size)) <= 16 * MAX_BLUR_SIDE * MAX_BLUR_SIDE);
    }
});
