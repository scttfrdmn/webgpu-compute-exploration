// MNIST Measure tabs (21 Train, 22 Recognize), CPU only: the compute-only API of js/examples/mnist-model.js
// (measureTraining, evaluateWeights, measureInference), the float64 reference of js/refs/mnist.js, and the
// experiments in js/teach/experiments/mnist-train.js and mnist-recognize.js, driven through the mock
// device (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: the Params slots the compute-only paths write (mode, counts, learning-rate schedule, epoch
// rollover); the caps; the reference forward pass against the CPU trainer on the cached weights (1,960 /
// 2,000); every sweep within its caps and every row with the declared columns. Real results:
// test/gpu/mnist-train-measure.gpu.test.mjs and test/gpu/mnist-recognize-measure.gpu.test.mjs.
// The experiments load data with loadDataset() (fetch); here fetch is given file: URLs of data/mnist/.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    measureTraining, evaluateWeights, measureInference, parseWeights, initWeights, DEFAULTS, MAX_BATCH,
    MAX_MEASURE_BATCHES, MEASURE_SLOTS, MAX_MEASURE_REPEATS, PARAMS_SIZE, wgslModules,
} from '../js/examples/mnist-model.js';
import { loadDatasetNode, evaluate, createScratch } from '../tools/train-mnist.mjs';
import { mnistForward, mnistEvaluate, argmax } from '../js/refs/mnist.js';
import trainExperiments, { LEARNING_RATES, LR_BATCHES, BATCH_SIZES, SPEED_BATCHES, CPU_CHECK } from '../js/teach/experiments/mnist-train.js';
import recognizeExperiments, { PREFIXES, LATENCY_BATCHES, LATENCY_REPEATS } from '../js/teach/experiments/mnist-recognize.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const PARAMS = structLayout(wgslModules()[1].code, 'Params');
const trainSet = loadDatasetNode('train');
const testSet = loadDatasetNode('test');
const cached = parseWeights(readFileSync(new URL('../data/mnist/weights.bin', import.meta.url)));

const nodeFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
    const u = new URL(url);
    return u.protocol === 'file:' ? new Response(readFileSync(fileURLToPath(u))) : nodeFetch(url);
};

// Every Params slot written (slots are 256 bytes apart on the mock device)
function slots(device) {
    const out = [];
    for (const w of device.writes) {
        if (!(w.buffer.usage & GPUBufferUsage.UNIFORM)) continue;
        for (let o = 0; o + PARAMS_SIZE <= w.bytes.byteLength; o += 256) out.push(decodeStruct(w.bytes, PARAMS, o));
    }
    return out;
}

test('measureTraining on a mock device: Params per mini-batch, learning-rate schedule across an epoch, evaluation', async () => {
    const device = mockDevice();
    let calls = 0;
    // 10,000 images / 128 = 79 batches per epoch (the last one 16): 100 batches cross into epoch 1
    const r = await measureTraining({ device }, { train: trainSet, test: testSet, batch: 128, lr: 0.05, batches: 100, slots: 8,
        between: async () => { calls++; } });
    const all = slots(device);
    const train = all.filter(s => s.mode === 0), evalSlots = all.filter(s => s.mode === 1);
    assert.equal(train.length, 100);
    assert.equal(train.filter(s => s.count === 16).length, 1, 'the last batch of epoch 0 is the remainder');
    const f = Math.fround;
    train.forEach((s, i) => {
        const epoch = i < 79 ? 0 : 1;
        assert.equal(s.lr, f(0.05 * DEFAULTS.lrDecay ** epoch), `batch ${i} learning rate`);
        assert.equal(s.momentum, f(DEFAULTS.momentum));
        assert.equal(s.maxShift, DEFAULTS.maxShift);
        assert.equal(s.dataLen, 10000);
        assert.equal(s.start, i < 79 ? 128 * i : 128 * (i - 79));
    });
    // submits never cross an epoch: 9 full submits of 8, one of 7, then 8, 8, 5
    assert.equal(r.curve.length, 13);
    assert.equal(calls, 13);
    assert.equal(evalSlots.length, Math.ceil(2000 / MAX_BATCH));
    assert.deepEqual(evalSlots.map(s => s.start), Array.from({ length: 16 }, (_, i) => 128 * i));
    assert.equal(r.batches, 100);
    assert.equal(r.gpuMsPerBatch, null);
    assert.equal(r.weights.length, 5994);
    assert.equal(r.test.probs.length, 20000);
    // He initialization uploaded (weights and zero momentum)
    const upload = device.writes.find(w => w.bytes.byteLength === 5994 * 8);
    assert.deepEqual(new Float32Array(upload.bytes.buffer).slice(0, 5994), initWeights(DEFAULTS.seed));
});

test('evaluateWeights / measureInference on a mock device: mode 1 slots, caps', async () => {
    const device = mockDevice();
    const r = await evaluateWeights({ device }, { weights: cached, data: testSet, count: 300, batch: 100 });
    assert.deepEqual(slots(device).map(s => [s.mode, s.start, s.count]), [[1, 0, 100], [1, 100, 100], [1, 200, 100]]);
    assert.equal(r.probs.length, 3000);
    const d2 = mockDevice();
    const inf = await measureInference({ device: d2 }, { weights: cached, data: testSet, batch: 32, warmup: 50, repeats: 50 });
    assert.equal(inf.batch, 32);
    assert.ok(inf.wallMs >= 0 && inf.gpuMs === null);
    const copies = d2.buffers.filter(b => b.size === 32 * 40 && (b.usage & GPUBufferUsage.MAP_READ));
    assert.equal(copies.length, 1, 'one probabilities readback buffer');
    await assert.rejects(measureTraining({ device }, { train: trainSet, batches: MAX_MEASURE_BATCHES + 1 }), RangeError);
    await assert.rejects(measureTraining({ device }, { train: trainSet, batch: MAX_BATCH + 1 }), RangeError);
    await assert.rejects(measureTraining({ device }, { train: trainSet, lr: 0 }), RangeError);
    await assert.rejects(evaluateWeights({ device }, { weights: cached, data: testSet, count: 2001 }), RangeError);
    await assert.rejects(measureInference({ device }, { weights: cached, data: testSet, batch: 0 }), RangeError);
    assert.ok(MEASURE_SLOTS <= 16 && MAX_MEASURE_REPEATS <= 20);
});

test('js/refs/mnist.js: the float64 forward pass reproduces the trainer on the cached weights (1,960 / 2,000)', (t) => {
    const ref = mnistEvaluate(cached, testSet);
    const tool = evaluate(cached, testSet, createScratch());
    t.diagnostic(`refs: ${ref.correct}/2000 loss ${ref.loss.toFixed(6)}; tools/train-mnist.mjs: ${tool.correct}/2000 loss ${tool.loss.toFixed(6)}`);
    assert.equal(ref.correct, 1960);
    assert.equal(ref.correct, tool.correct);
    assert.ok(Math.abs(ref.loss - tool.loss) < 1e-6);
    const p = mnistForward(cached, testSet.images, 0);
    assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    assert.equal(argmax(p), testSet.labels[0]);
    assert.equal(argmax([0, 3, 3, 1]), 1, 'first maximum wins');
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(LR_BATCHES <= 100 && LEARNING_RATES.length * LR_BATCHES <= 500, 'short training runs');
    assert.ok(BATCH_SIZES.every(b => b <= MAX_BATCH) && BATCH_SIZES.length * (8 + SPEED_BATCHES) <= 500);
    assert.ok(CPU_CHECK <= 2000 && PREFIXES.every(n => n <= 2000));
    assert.ok(LATENCY_BATCHES.every(b => b <= MAX_BATCH) && LATENCY_REPEATS <= MAX_MEASURE_REPEATS);
    for (const e of [...trainExperiments, ...recognizeExperiments]) assert.ok(e.budgetMs <= 20000);
});

test('experiments run on a mock device: one row per sweep value, declared columns, mini-batch caps', async (t) => {
    const cases = [
        [trainExperiments, 'learning-rate', LEARNING_RATES.length],
        [trainExperiments, 'batch-speed', BATCH_SIZES.length],
        [recognizeExperiments, 'test-accuracy', PREFIXES.length],
        [recognizeExperiments, 'latency', LATENCY_BATCHES.length],
    ];
    for (const [list, id, expected] of cases) {
        const experiment = list.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, {}, ctx);
        const train = slots(device).filter(s => s.mode === 0);
        t.diagnostic(`${id}: ${rows.length} rows, ${train.length} training mini-batches, ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort());
        assert.ok(train.length <= 500, 'at most 500 mini-batches per experiment');
        assert.ok(slots(device).every(s => s.count <= MAX_BATCH));
    }
});
