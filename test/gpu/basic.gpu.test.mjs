// GPU execution of the one-shot benchmark kernels: vector-addition, matrix-multiplication, image-blur.
// CI-only on a software GPU — see test/gpu/harness.mjs. Exact shipped WGSL (wgslModules()).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, buffer, dispatch, read, rng, compare } from './harness.mjs';
import { wgslModules as vecModules } from '../../js/examples/vector-addition.js';
import { wgslModules as matModules } from '../../js/examples/matrix-multiplication.js';
import { wgslModules as blurModules, runImageBlur } from '../../js/examples/image-blur.js';
import { matmulF64, separableBlurPixel } from '../helpers/basic-refs.mjs';
import { mockDevice } from '../helpers/mock-gpu.mjs';

test('vector-addition: shipped 1M inputs and a masked 1000-element tail', async (t) => {
    const p = await computePipeline(vecModules()[0].code, 'main');
    for (const [n, make] of [
        [1_000_000, i => [i, i * 2]],                    // exactly what runVectorAddition uploads
        [1000, i => [Math.sin(i), Math.cos(i)]],          // 16 workgroups, 24 masked invocations
    ]) {
        const a = new Float32Array(n), b = new Float32Array(n);
        for (let i = 0; i < n; i++) [a[i], b[i]] = make(i);
        // The output is longer than the inputs and pre-filled with a sentinel: the bounds guard
        // (index < arrayLength(&input1)) must leave the tail untouched.
        const out = buffer(new Float32Array(n + 4).fill(-7));
        await dispatch(p, { input1: buffer(a), input2: buffer(b), output: out }, [Math.ceil(n / 64)]);
        const got = await read(out, Float32Array, (n + 4) * 4);
        let maxRel = 0;
        for (let i = 0; i < n; i++) {
            const exact = a[i] + b[i];
            maxRel = Math.max(maxRel, Math.abs(got[i] - exact) / Math.max(Math.abs(exact), 1e-30));
        }
        t.diagnostic(`n=${n} max relative error vs float64 = ${maxRel.toExponential(2)}`);
        assert.ok(maxRel <= 2 ** -24, 'one correctly rounded f32 add (half ulp)');
        assert.deepEqual([...got.subarray(n, n + 4)], [-7, -7, -7, -7], 'no writes past arrayLength');
    }
});

test('matrix-multiplication: 512^3 (shipped size) and 37x23*23x19 vs float64', async (t) => {
    const p = await computePipeline(matModules()[0].code, 'main');
    for (const [M, N, K, lo, tol] of [
        // Sequential f32 accumulation of 512 positive terms: relative error <= ~N·eps/2 (CPU-suite bound).
        [512, 512, 512, 0, { rel: 512 * 2 ** -24 }],
        [37, 23, 19, -1, { abs: 1e-5 }],
    ]) {
        const rand = rng(M * 1000 + K);
        const A = Float32Array.from({ length: M * N }, () => lo + rand() * (1 - lo));
        const B = Float32Array.from({ length: N * K }, () => lo + rand() * (1 - lo));
        const C = buffer(new Float32Array(M * K + 4).fill(NaN));
        await dispatch(p, { matrixA: buffer(A), matrixB: buffer(B), result: C, dimensions: buffer(new Uint32Array([M, N, K])) },
            [Math.ceil(M / 8), Math.ceil(K / 8)]);
        const got = await read(C, Float32Array, (M * K + 4) * 4);
        const ref = matmulF64(A, B, M, N, K);
        const e = compare(got, ref, 1e-3);
        t.diagnostic(`${M}x${N}*${N}x${K}: max|err|=${e.maxAbs.toExponential(2)} max rel=${e.maxRel.toExponential(2)}`);
        assert.equal(e.nonFinite, 0, 'every element written');
        if (tol.rel) assert.ok(e.maxRel < tol.rel, `max rel ${e.maxRel} >= ${tol.rel}`);
        if (tol.abs) assert.ok(e.maxAbs < tol.abs, `max abs ${e.maxAbs} >= ${tol.abs}`);
        assert.ok(got.subarray(M * K).every(Number.isNaN), 'no writes past M·K');
    }
});

const SIGMA = 4, HALF = 12;

async function shippedBlurImage() {
    const canvas = () => ({ width: 0, height: 0, getContext: () => ({
        createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {} }) });
    globalThis.document = { getElementById: canvas };
    try {
        const device = mockDevice();
        await runImageBlur({ device });
        return new Float32Array(device.buffers[0].getMappedRange().slice(0));
    } finally { delete globalThis.document; }
}

test('image-blur: shipped 512x512 test image and a random 61x37 image vs separable float64', async (t) => {
    const p = await computePipeline(blurModules()[0].code, 'main');
    const rand = rng(3);
    for (const [w, h, img, sampled] of [
        [512, 512, await shippedBlurImage(), true],
        [61, 37, Float32Array.from({ length: 61 * 37 * 4 }, () => rand()), false],
    ]) {
        const out = buffer(w * h * 16);
        await dispatch(p, { input: buffer(img), output: out, dimensions: buffer(new Uint32Array([w, h])) },
            [Math.ceil(w / 8), Math.ceil(h / 8)]);
        const got = await read(out);
        const pts = [];
        if (sampled) {
            pts.push([0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [256, 0], [0, 256], [255, 255]);
            for (let i = 0; i < 2000; i++) pts.push([(i * 197 + 13) % w, (i * 331 + 71) % h]);
        } else {
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pts.push([x, y]);
        }
        let maxErr = 0;
        for (const [x, y] of pts) {
            const ref = separableBlurPixel(img, w, h, x, y, SIGMA, HALF);
            for (let c = 0; c < 4; c++) maxErr = Math.max(maxErr, Math.abs(got[(y * w + x) * 4 + c] - ref[c]));
        }
        t.diagnostic(`${w}x${h}: ${pts.length} pixels, max |gpu - separable f64| = ${maxErr.toExponential(2)}`);
        assert.ok(maxErr < 1e-5, 'f32 accumulation of 625 taps of values in [0,1] (CPU-suite bound)');
    }
});
