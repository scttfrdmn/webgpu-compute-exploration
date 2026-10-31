// Tests for tools/train-mnist.mjs, the CPU trainer that produced data/mnist/weights.bin (and whose
// math the GPU kernels of example 21 follow). Shipped JS throughout:
//  - backward() checked against central finite differences of forward() + softmaxLoss() in float64
//  - softmaxLoss() against a direct softmax / cross-entropy formula
//  - loadImage() translation, epochLearningRate(), trainBatch() reducing the loss on real MNIST,
//    train() determinism, and the committed weights reproducing the documented 98.00% test accuracy.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    loadDatasetNode, createScratch, forward, softmaxLoss, backward, loadImage, trainBatch, evaluate,
    epochLearningRate, train,
} from '../tools/train-mnist.mjs';
import { NUM_PARAMS, OFF, TENSORS, DEFAULTS, initWeights, parseWeights, epochOrder, augmentSeed } from '../js/examples/mnist-model.js';

const trainData = loadDatasetNode('train');
const testData = loadDatasetNode('test');

// Same shape as createScratch(), but float64 everywhere, for finite differences
function scratch64() {
    const s = createScratch();
    for (const k of Object.keys(s)) s[k] = new Float64Array(s[k].length);
    return s;
}

test('importing the trainer does not start training; datasets load with the documented sizes', () => {
    assert.equal(trainData.count, 10000);
    assert.equal(trainData.images.length, 10000 * 784);
    assert.equal(testData.count, 2000);
    assert.ok(trainData.labels.every(l => l < 10));
});

test('softmaxLoss matches a direct softmax cross-entropy, and is stable for huge logits', (t) => {
    const s = createScratch();
    const logits = [1.5, -2, 0.25, 3, 0, -1, 2.5, 0.75, -0.5, 1];
    s.logit.set(logits);
    const label = 3;
    const loss = softmaxLoss(s, label);
    const z = logits.reduce((a, l) => a + Math.exp(l), 0);
    const ref = -Math.log(Math.exp(logits[label]) / z);
    let maxErr = Math.abs(loss - ref);
    for (let k = 0; k < 10; k++) {
        const p = Math.exp(logits[k]) / z;
        maxErr = Math.max(maxErr, Math.abs(s.prob[k] - p), Math.abs(s.dlog[k] - (p - (k === label ? 1 : 0))));
    }
    t.diagnostic(`loss ${loss.toFixed(6)} vs direct ${ref.toFixed(6)}; max |error| over loss/prob/dlog ${maxErr.toExponential(2)}`);
    assert.ok(maxErr < 1e-6, 'prob/dlog are stored as f32: ~1e-7');
    // log-sum-exp: no overflow at logits ±1e30 (clamped to ±1e4)
    s.logit.fill(0); s.logit[0] = 1e30; s.logit[1] = -1e30;
    assert.ok(Number.isFinite(softmaxLoss(s, 1)));
    assert.equal(softmaxLoss(s, 0), 0);
});

test('backward() matches central finite differences of the loss (float64)', (t) => {
    const s = scratch64();
    const w = Float64Array.from(initWeights(5));
    // Nonzero biases: with b1 = 0 (the He init) every background pixel of conv1 sits exactly on
    // the ReLU kink, where backward() uses the subgradient 0 but a central difference sees 1/2.
    for (const [o, n] of [[OFF.B1, 8], [OFF.B2, 16], [OFF.B3, 10]]) {
        for (let i = 0; i < n; i++) w[o + i] = (i % 2 ? 0.02 : -0.01) * (1 + i / n);
    }
    const lossAt = (img, label) => { forward(w, s); return softmaxLoss(s, label); };
    const h = 1e-6;
    const rows = [];
    let worst = 0, checked = 0;
    for (const index of [0, 7]) {
        const label = testData.labels[index];
        loadImage(s, testData.images, index);
        lossAt(index, label);
        backward(w, s);
        const g = Float64Array.from(s.g);
        // every bias, and every 13th weight of each weight tensor (spread over all filters)
        const params = [];
        for (const tensor of TENSORS) {
            const n = tensor.shape.reduce((a, b) => a * b, 1);
            const step = tensor.name.startsWith('b') ? 1 : 13;
            for (let i = 0; i < n; i += step) params.push([tensor.name, tensor.offset + i]);
        }
        const perTensor = {};
        for (const [name, p] of params) {
            const w0 = w[p];
            w[p] = w0 + h; const lp = lossAt(index, label);
            w[p] = w0 - h; const lm = lossAt(index, label);
            w[p] = w0;
            const numeric = (lp - lm) / (2 * h);
            const rel = Math.abs(numeric - g[p]) / Math.max(Math.abs(numeric) + Math.abs(g[p]), 1e-8);
            perTensor[name] = Math.max(perTensor[name] || 0, rel);
            worst = Math.max(worst, rel);
            checked++;
        }
        rows.push(`image ${index} (label ${label}): max relative error per tensor ` +
            Object.entries(perTensor).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(', '));
    }
    rows.forEach(r => t.diagnostic(r));
    t.diagnostic(`${checked} parameters checked, worst relative error ${worst.toExponential(2)} (h = ${h})`);
    // float64 central differences: truncation O(h²) and round-off ~1e-16/h ≈ 1e-10; a ReLU or
    // max-pool switching inside ±h would show up as O(1) error
    assert.ok(worst < 1e-5, `worst relative error ${worst}`);
});

test('loadImage scales bytes to [0,1] and translates with zeros shifted in', () => {
    const s = createScratch();
    const images = new Uint8Array(784);
    images[5 * 28 + 7] = 255;
    images[0] = 51;
    loadImage(s, images, 0, 2, -1);
    assert.equal(s.x[4 * 28 + 9], 1);
    assert.equal(s.x.reduce((a, b) => a + b, 0), 1, 'pixel (0,0) shifted out of the frame');
    loadImage(s, images, 0);
    assert.equal(s.x[0], Math.fround(51 / 255));
});

test('epochLearningRate: η0 · decay^epoch', () => {
    assert.equal(epochLearningRate(0.04, 0.85, 0), 0.04);
    assert.ok(Math.abs(epochLearningRate(0.04, 0.85, 3) - 0.04 * 0.85 ** 3) < 1e-15);
    assert.equal(epochLearningRate(0.1, 1, 10), 0.1);
});

test('trainBatch reduces the loss on real MNIST (first 1,024 images, batch 32)', (t) => {
    const n = 1024, batch = 32;
    const data = { images: trainData.images.subarray(0, n * 784), labels: trainData.labels.subarray(0, n), count: n };
    const w = initWeights(DEFAULTS.seed);
    const v = new Float32Array(NUM_PARAMS);
    const s = createScratch();
    const order = epochOrder(n, DEFAULTS.seed, 0);
    const o = { lr: DEFAULTS.lr, momentum: DEFAULTS.momentum, weightDecay: DEFAULTS.weightDecay,
        augSeed: augmentSeed(DEFAULTS.seed, 0), maxShift: DEFAULTS.maxShift };
    const losses = [];
    let correctLast = 0;
    for (let pos = 0; pos < n; pos += batch) {
        const r = trainBatch(w, v, data, order, pos, batch, o, s);
        losses.push(r.loss / batch);
        if (pos >= n - 8 * batch) correctLast += r.correct;
    }
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const first = mean(losses.slice(0, 4)), last = mean(losses.slice(-8));
    const accLast = correctLast / (8 * batch);
    t.diagnostic(`mean loss first 4 batches ${first.toFixed(3)}, last 8 batches ${last.toFixed(3)}; ` +
        `train accuracy over last 8 batches ${(100 * accLast).toFixed(1)}%`);
    // chance is ln 10 = 2.30; after one pass over 1,024 shifted images the net is well past chance
    assert.ok(first > 1.8, 'starts near chance');
    assert.ok(last < 0.6 * first, 'loss falls by more than 40%');
    assert.ok(accLast > 0.45, 'chance is 10%');
    assert.ok(w.every(Number.isFinite) && v.every(Number.isFinite));
});

test('train() is deterministic for a seed', () => {
    const n = 96;
    const data = { images: trainData.images.subarray(0, n * 784), labels: trainData.labels.subarray(0, n), count: n };
    const run = () => train({ ...DEFAULTS, batch: 32, epochs: 2, train: data, test: null });
    const a = run(), b = run();
    assert.deepEqual(a.weights, b.weights);
    assert.deepEqual(a.momentum, b.momentum);
    assert.notDeepEqual(a.weights, initWeights(DEFAULTS.seed));
});

test('committed data/mnist/weights.bin reproduces the documented 98.00% (1,960 / 2,000) test accuracy', (t) => {
    const w = parseWeights(readFileSync(fileURLToPath(new URL('../data/mnist/weights.bin', import.meta.url))));
    const r = evaluate(w, testData, createScratch());
    t.diagnostic(`test accuracy ${(100 * r.accuracy).toFixed(2)}% (${r.correct}/2000), mean loss ${r.loss.toFixed(4)}`);
    // forward() uses no transcendental functions before the argmax, so this is exact
    assert.equal(r.correct, 1960);
});
