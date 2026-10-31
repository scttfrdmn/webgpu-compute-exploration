// Tests for js/examples/mnist-recognize.js (example 22) — shipped JS, no GPU:
//  - preprocessDrawing(): empty input, output range, 20-px bounding box with aspect ratio kept,
//    center of mass moved to (14, 14), invariance to where/how large the digit is drawn
//  - end to end with the committed weights: real MNIST test digits upscaled 10× to the 280×280
//    drawing canvas, preprocessed, and classified by the CPU forward pass (tools/train-mnist.mjs),
//    compared with classifying the original 28×28 digits
//  - infer(): writes the input at the batch-1 work-buffer X offset and copies PROB back, and the
//    maps shader's work-buffer constants match workLayout(1) and the compute module.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { preprocessDrawing, MNISTRecognizer, wgslModules } from '../js/examples/mnist-recognize.js';
import { MNISTGPU, workLayout, parseWeights, buildComputeShader, NUM_CLASSES } from '../js/examples/mnist-model.js';
import { loadDatasetNode, createScratch, forward } from '../tools/train-mnist.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const DRAW = 280;

function moments(img) {
    let mass = 0, mx = 0, my = 0, minX = 28, minY = 28, maxX = -1, maxY = -1;
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
        const v = img[y * 28 + x];
        if (v > 0) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
        mass += v; mx += v * (x + 0.5); my += v * (y + 0.5);
    }
    return { mass, cx: mx / mass, cy: my / mass, w: maxX - minX + 1, h: maxY - minY + 1 };
}

// Antialiased-free filled shapes on a w×h canvas
function canvas(w = DRAW, h = DRAW) { return new Float32Array(w * h); }
function rect(c, w, x0, y0, x1, y1, v = 1) {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) c[y * w + x] = v;
    return c;
}
function ring(c, w, cx, cy, r0, r1) {
    for (let y = 0; y < c.length / w; y++) for (let x = 0; x < w; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        if (d >= r0 && d <= r1) c[y * w + x] = 1;
    }
    return c;
}

test('preprocessDrawing: empty and faint drawings return null', () => {
    assert.equal(preprocessDrawing(canvas(), DRAW, DRAW), null);
    assert.equal(preprocessDrawing(rect(canvas(), DRAW, 10, 10, 50, 50, 0.05), DRAW, DRAW), null, 'below the ink threshold');
});

test('preprocessDrawing: 20-px box, aspect kept, center of mass at (14, 14), values in [0, 1]', (t) => {
    const cases = {
        'tall bar 30×180 at the left edge': rect(canvas(), DRAW, 0, 20, 30, 200),
        'wide bar 200×40 near the bottom': rect(canvas(), DRAW, 60, 230, 260, 270),
        'ring r 40..70 off-center': ring(canvas(), DRAW, 90, 180, 40, 70),
        'L shape': rect(rect(canvas(), DRAW, 100, 40, 130, 220), DRAW, 100, 190, 220, 220),
        'single dot': rect(canvas(), DRAW, 140, 140, 141, 141),
    };
    for (const [name, c] of Object.entries(cases)) {
        const out = preprocessDrawing(c, DRAW, DRAW);
        assert.equal(out.length, 784);
        assert.ok(out.every(v => v >= 0 && v <= 1), `${name}: range`);
        const m = moments(out);
        const err = Math.hypot(m.cx - 14, m.cy - 14);
        t.diagnostic(`${name}: box ${m.w}×${m.h}, center of mass (${m.cx.toFixed(2)}, ${m.cy.toFixed(2)}), |Δ| ${err.toFixed(2)} px`);
        assert.ok(Math.max(m.w, m.h) <= 20, `${name}: fits 20×20`);
        if (name !== 'single dot') {
            assert.ok(Math.max(m.w, m.h) >= 19, `${name}: scaled up to ~20`);
            // integer shift: within half a pixel per axis, unless clamped to keep the box in frame
            assert.ok(Math.abs(m.cx - 14) <= 0.5 + 1e-9 && Math.abs(m.cy - 14) <= 0.5 + 1e-9, `${name}: centered`);
        }
    }
    const tall = moments(preprocessDrawing(cases['tall bar 30×180 at the left edge'], DRAW, DRAW));
    assert.equal(tall.h, 20);
    assert.ok(Math.abs(tall.w - 20 * 30 / 180) <= 1, `aspect ratio kept: width ${tall.w}`);
    // a filled area stays fully inked in the interior (area averaging keeps mass)
    const bar = preprocessDrawing(cases['wide bar 200×40 near the bottom'], DRAW, DRAW);
    assert.ok(Math.max(...bar) === 1);
});

test('preprocessDrawing is invariant to the position and size of the drawing', (t) => {
    const a = preprocessDrawing(ring(canvas(), DRAW, 70, 70, 20, 35), DRAW, DRAW);
    const b = preprocessDrawing(ring(canvas(), DRAW, 180, 160, 60, 105), DRAW, DRAW);   // 3× larger, moved
    let diff = 0;
    for (let i = 0; i < 784; i++) diff = Math.max(diff, Math.abs(a[i] - b[i]));
    let mean = 0;
    for (let i = 0; i < 784; i++) mean += Math.abs(a[i] - b[i]) / 784;
    t.diagnostic(`ring drawn at 1× and 3×: max |Δ| ${diff.toFixed(3)}, mean |Δ| ${mean.toFixed(4)}`);
    // the rasterized rings differ only at their edges (pixel quantization of the source)
    assert.ok(mean < 0.03);
});

test('real MNIST digits redrawn at 10× and preprocessed are classified like the originals', (t) => {
    const test = loadDatasetNode('test');
    const w = parseWeights(readFileSync(fileURLToPath(new URL('../data/mnist/weights.bin', import.meta.url))));
    const s = createScratch();
    const n = 300;
    let agree = 0, correctOrig = 0, correctPre = 0, meanAbs = 0;
    for (let i = 0; i < n; i++) {
        const px = test.images.subarray(i * 784, (i + 1) * 784);
        for (let k = 0; k < 784; k++) s.x[k] = px[k] / 255;
        const orig = Float32Array.from(s.x);
        const p0 = forward(w, s);
        // 10× nearest-neighbor upscale, as randomTestDigit() draws it on the canvas
        const big = new Float32Array(DRAW * DRAW);
        for (let y = 0; y < DRAW; y++) for (let x = 0; x < DRAW; x++) big[y * DRAW + x] = px[((y / 10) | 0) * 28 + ((x / 10) | 0)] / 255;
        const pre = preprocessDrawing(big, DRAW, DRAW);
        s.x.set(pre);
        const p1 = forward(w, s);
        for (let k = 0; k < 784; k++) meanAbs += Math.abs(pre[k] - orig[k]) / (784 * n);
        if (p0 === p1) agree++;
        if (p0 === test.labels[i]) correctOrig++;
        if (p1 === test.labels[i]) correctPre++;
    }
    t.diagnostic(`${n} test digits: accuracy original ${(100 * correctOrig / n).toFixed(1)}%, ` +
        `redrawn+preprocessed ${(100 * correctPre / n).toFixed(1)}%, predictions agree ${(100 * agree / n).toFixed(1)}%, ` +
        `mean |pixel Δ| ${meanAbs.toFixed(4)}`);
    // MNIST digits are already 20×20-boxed and mass-centered, so preprocessing should nearly reproduce them
    assert.ok(agree / n >= 0.95);
    assert.ok(correctPre / n >= 0.93);
});

test('infer(): input written at the batch-1 X offset, PROB copied back; maps shader constants', async (t) => {
    const L = workLayout(1);
    const device = mockDevice();
    const copies = [];
    device.createCommandEncoder = () => ({
        beginComputePass: () => ({ setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} }),
        copyBufferToBuffer: (src, srcOffset, dst, dstOffset, size) => copies.push({ src, srcOffset, dst, dstOffset, size }),
        finish: () => ({}),
    });
    const r = Object.create(MNISTRecognizer.prototype);
    const probs = Float32Array.from({ length: NUM_CLASSES }, (_, k) => (k + 1) / 55);
    Object.assign(r, {
        device, gpu: new MNISTGPU(device, 1, 1), weightsReady: true, destroyed: false, busy: false, pending: null,
        probsBuffer: { mapAsync: () => Promise.resolve(), getMappedRange: () => probs.buffer.slice(0), unmap() {} },
        encodeMaps() {},
    });
    const input = Float32Array.from({ length: 784 }, (_, i) => (i % 29) / 28);
    r.infer(input);
    const wr = device.lastWrite(r.gpu.workBuffer);
    assert.equal(wr.offset, L.X * 4);
    assert.deepEqual(new Float32Array(wr.bytes.buffer), input);
    assert.deepEqual(copies, [{ src: r.gpu.workBuffer, srcOffset: L.PROB * 4, dst: r.probsBuffer, dstOffset: 0, size: NUM_CLASSES * 4 }]);
    await new Promise(res => setImmediate(res));
    assert.deepEqual(r.probs, Array.from(probs));

    // maps shader reads the same regions the batch-1 compute module writes
    const consts = (code) => Object.fromEntries([...code.matchAll(/const\s+(W_\w+)\s*:\s*u32\s*=\s*(\d+)u/g)].map(m => [m[1], Number(m[2])]));
    const maps = consts(wgslModules()[0].code);
    const compute = consts(buildComputeShader(1));
    for (const [k, v] of Object.entries(maps)) {
        assert.equal(v, L[k.slice(2)], k);
        assert.equal(v, compute[k], `${k} vs compute module`);
    }
    assert.deepEqual(Object.keys(maps).sort(), ['W_A1', 'W_A2', 'W_MAPMAX', 'W_X']);
    t.diagnostic(`batch-1 work buffer: X at ${L.X}, PROB at ${L.PROB}, MAPMAX at ${L.MAPMAX} floats`);
});
