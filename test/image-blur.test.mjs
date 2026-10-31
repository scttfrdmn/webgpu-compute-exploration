// Gaussian image blur (example #3).
// CPU replica of the `compute` shader; validates the algorithm and index math, not the WGSL itself — see #51.
// The shipped runImageBlur() is driven with a mock device (and a stub `document` for its two 2D
// canvases) to capture the test image and `dimensions: vec2<u32>` uniform it uploads.

import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { runImageBlur, wgslModules } from '../js/examples/image-blur.js';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { typeLayout } from './helpers/wgsl-layout.mjs';

const SIGMA = 4.0;
const KERNEL = 25;
const HALF = Math.floor(KERNEL / 2);

// Replica of main() for one pixel: 25x25 window, out-of-image taps skipped, result divided by the
// sum of the weights actually used (so edges are renormalized).
function gaussian(x, y) {
    return Math.fround(Math.exp(-(x * x + y * y) / (2 * SIGMA * SIGMA)) / (2 * 3.14159265 * SIGMA * SIGMA));
}
function replicaPixel(img, w, h, x, y) {
    const sum = [0, 0, 0, 0];
    let wsum = 0;
    for (let ky = -HALF; ky <= HALF; ky++)
        for (let kx = -HALF; kx <= HALF; kx++) {
            const sx = x + kx, sy = y + ky;
            if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                const idx = sy * w + sx;
                const wt = gaussian(kx, ky);
                for (let c = 0; c < 4; c++) sum[c] = Math.fround(sum[c] + Math.fround(img[idx * 4 + c] * wt));
                wsum = Math.fround(wsum + wt);
            }
        }
    return sum.map(s => Math.fround(s / wsum));
}

// Independent reference: separable float64 blur (horizontal 1-D pass, then vertical) with per-axis
// renormalization. Because the clipped window is a rectangle, the 2-D renormalized Gaussian factors
// exactly into two 1-D renormalized passes.
function separablePixel(img, w, h, x, y) {
    const g = k => Math.exp(-(k * k) / (2 * SIGMA * SIGMA));
    const out = [0, 0, 0, 0];
    let wy = 0;
    for (let ky = -HALF; ky <= HALF; ky++) {
        const sy = y + ky;
        if (sy < 0 || sy >= h) continue;
        const row = [0, 0, 0, 0];
        let wx = 0;
        for (let kx = -HALF; kx <= HALF; kx++) {
            const sx = x + kx;
            if (sx < 0 || sx >= w) continue;
            for (let c = 0; c < 4; c++) row[c] += img[(sy * w + sx) * 4 + c] * g(kx);
            wx += g(kx);
        }
        for (let c = 0; c < 4; c++) out[c] += (row[c] / wx) * g(ky);
        wy += g(ky);
    }
    return out.map(v => v / wy);
}

let captured;
before(async () => {
    const canvas = () => ({
        width: 0, height: 0,
        getContext: () => ({
            createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
            putImageData() {},
        }),
    });
    globalThis.document = { getElementById: canvas };
    const device = mockDevice();
    const html = await runImageBlur({ device });
    delete globalThis.document;
    const [input, , dims] = device.buffers;
    captured = {
        html,
        img: new Float32Array(input.getMappedRange()),
        dims: new Uint32Array(dims.getMappedRange()),
        dimsSize: dims.size,
    };
});

test('dimensions uniform is vec2<u32> (width, height)', (t) => {
    assert.match(wgslModules()[0].code, /var<uniform> dimensions: vec2<u32>/);
    assert.equal(captured.dimsSize, typeLayout('vec2<u32>').size);
    t.diagnostic(`dimensions = [${[...captured.dims]}]`);
    assert.deepEqual([...captured.dims], [512, 512]);
    assert.equal(captured.img.length, 512 * 512 * 4);
});

test('shader constants match the documented σ=4, 25x25 (±3σ) kernel', () => {
    const code = wgslModules()[0].code;
    assert.match(code, /const sigma = 4\.0;/);
    assert.match(code, /const kernelSize = 25;/);
    assert.ok(HALF >= 3 * SIGMA, 'window covers ±3σ');
    assert.match(captured.html, /Kernel size: 25 × 25/);
});

test('kernel normalization: truncated mass and constant-image invariance', (t) => {
    let mass = 0;
    for (let ky = -HALF; ky <= HALF; ky++) for (let kx = -HALF; kx <= HALF; kx++) mass += gaussian(kx, ky);
    t.diagnostic(`unnormalized 25x25 kernel mass = ${mass.toFixed(6)} (continuous Gaussian = 1)`);
    // Integer taps |k| <= 12 approximate the integral over [-12.5, 12.5] per axis:
    // erf(12.5 / (4·√2))^2 = 0.99651. Anything far from that means the kernel/σ changed.
    assert.ok(Math.abs(mass - 0.99651) < 2e-4);

    const w = 20, h = 15;
    const flat = new Float32Array(w * h * 4).fill(0.375);
    let maxDev = 0;
    for (const [x, y] of [[0, 0], [w - 1, h - 1], [5, 0], [10, 7], [0, 14]]) {
        for (const v of replicaPixel(flat, w, h, x, y)) maxDev = Math.max(maxDev, Math.abs(v - 0.375));
    }
    t.diagnostic(`constant image max deviation after blur (incl. corners) = ${maxDev.toExponential(2)}`);
    assert.ok(maxDev < 1e-6, 'renormalization by weightSum keeps flat regions flat at edges too');
});

test('replica vs separable float64 reference on the shipped 512x512 test image', (t) => {
    const { img } = captured;
    const w = 512, h = 512;
    // Sample corners, edges, and a deterministic spread of interior pixels.
    const pts = [[0, 0], [511, 0], [0, 511], [511, 511], [256, 0], [0, 256], [3, 9], [255, 255]];
    for (let i = 0; i < 120; i++) pts.push([(i * 197 + 13) % w, (i * 331 + 71) % h]);
    let maxErr = 0, maxIn = 0;
    for (const [x, y] of pts) {
        const r = replicaPixel(img, w, h, x, y), s = separablePixel(img, w, h, x, y);
        for (let c = 0; c < 4; c++) {
            maxErr = Math.max(maxErr, Math.abs(r[c] - s[c]));
            maxIn = Math.max(maxIn, img[(y * w + x) * 4 + c]);
        }
    }
    t.diagnostic(`${pts.length} pixels, max |replica - separable f64| = ${maxErr.toExponential(2)}`);
    assert.ok(maxIn <= 1);
    assert.ok(maxErr < 1e-5, 'f32 accumulation of 625 taps of values in [0,1]');
    for (let i = 3; i < img.length; i += 4) assert.equal(img[i], 1, 'alpha channel is 1');
});
