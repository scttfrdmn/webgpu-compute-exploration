// MNIST training compute-only API (measureTraining in js/examples/mnist-model.js) and the Measure
// experiments built on it (js/teach/experiments/mnist-train.js), on a software GPU — CI only, see harness.mjs.
//
// measureTraining drives the shipped MNISTGPU (its pipelines, bind groups, writeSlots and encodeBatch)
// on the real device, as the Measure tab does in a browser. Checked against:
//   the CPU trainer tools/train-mnist.mjs (the program that produced data/mnist/weights.bin), on the
//   identical short run: same seed, shuffle, augmentation and learning rate, so test accuracy and loss
//   must agree closely; and the float64 forward pass of js/refs/mnist.js on the GPU-trained weights;
//   experiments: every row has the declared columns; the learning-rate curve has its interior optimum.
// The experiments load data with loadDataset() (fetch); here fetch is given file: URLs of data/mnist/.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { device } from './harness.mjs';
import { measureTraining, DEFAULTS, NUM_PARAMS } from '../../js/examples/mnist-model.js';
import { loadDatasetNode, train as cpuTrain, evaluate, createScratch } from '../../tools/train-mnist.mjs';
import { mnistEvaluate, argmax } from '../../js/refs/mnist.js';
import experiments from '../../js/teach/experiments/mnist-train.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const trainSet = loadDatasetNode('train');
const testSet = loadDatasetNode('test');

const nodeFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
    const u = new URL(url);
    return u.protocol === 'file:' ? new Response(readFileSync(fileURLToPath(u))) : nodeFetch(url);
};

test('measureTraining (seed 1, batch 64, η = 0.04, 40 mini-batches) vs the CPU trainer on the same run', async (t) => {
    const batches = 40;
    let t0 = performance.now();
    const r = await measureTraining(gpuDevice, { train: trainSet, test: testSet, batch: 64, lr: 0.04, batches, slots: 8 });
    const gpuSeconds = (performance.now() - t0) / 1000;
    assert.equal(r.batches, batches);
    assert.equal(r.curve.length, batches / 8);
    assert.equal(r.weights.length, NUM_PARAMS);
    assert.ok(r.weights.every(Number.isFinite));
    assert.equal(r.test.count, testSet.count);

    t0 = performance.now();
    const cpu = cpuTrain({ ...DEFAULTS, lr: 0.04, batch: 64, epochs: 1, train: trainSet, maxBatches: batches });
    const ev = evaluate(cpu.weights, testSet, createScratch());
    const cpuSeconds = (performance.now() - t0) / 1000;
    let wDiff = 0, wScale = 0;
    for (let i = 0; i < NUM_PARAMS; i++) {
        wDiff = Math.max(wDiff, Math.abs(r.weights[i] - cpu.weights[i]));
        wScale = Math.max(wScale, Math.abs(cpu.weights[i]));
    }
    // The GPU's own predictions vs the float64 reference forward pass with the GPU's weights
    const ref = mnistEvaluate(r.weights, testSet);
    let agree = 0;
    for (let i = 0; i < testSet.count; i++) if (argmax(ref.probs, 10 * i) === argmax(r.test.probs, 10 * i)) agree++;
    t.diagnostic(`GPU ${gpuSeconds.toFixed(1)} s (${r.wallMsPerBatch.toFixed(1)} ms/batch wall): test ${r.test.correct}/${r.test.count} `
        + `loss ${r.test.loss.toFixed(4)}; CPU trainer ${cpuSeconds.toFixed(1)} s: ${ev.correct}/${testSet.count} loss ${ev.loss.toFixed(4)}; `
        + `max |w_gpu − w_cpu| ${wDiff.toExponential(2)} (of max |w| ${wScale.toFixed(3)}); float64 forward with GPU weights `
        + `${ref.correct} correct, predictions agree on ${agree}/${testSet.count}; train curve `
        + r.curve.map(c => c.loss.toFixed(3)).join(' '));
    assert.ok(Math.abs(r.test.correct - ev.correct) <= 10, `GPU ${r.test.correct} vs CPU ${ev.correct} correct`);
    assert.ok(Math.abs(r.test.loss - ev.loss) < 0.02);
    assert.ok(wDiff < 1e-2 * wScale, `weights differ by ${wDiff}`);
    assert.ok(Math.abs(ref.correct - r.test.correct) <= 2 && agree >= testSet.count - 4);
    assert.ok(r.curve[r.curve.length - 1].loss < r.curve[0].loss, 'training loss falls');
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
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toPrecision(4)) : String(v);
}).join(' ')).join('; ');

test('experiment learning-rate: interior optimum, CPU agreement', async (t) => {
    const e = byId['learning-rate'];
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: lr testAccuracy testLoss trainLoss cpuAgree wallMs: `
        + show(rows, ['lr', 'testAccuracy', 'testLoss', 'trainLoss', 'cpuAgree', 'wallMs']));
    assert.deepEqual(rows.map(r => r.lr), [0.005, 0.01, 0.02, 0.04, 0.08, 0.15]);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    const best = rows.reduce((a, b) => (b.testAccuracy > a.testAccuracy ? b : a));
    assert.ok(best.lr >= 0.02 && best.lr <= 0.08, `best η ${best.lr}`);
    assert.ok(best.testAccuracy > 0.85);
    assert.ok(rows[0].testAccuracy < best.testAccuracy && rows[5].testAccuracy < best.testAccuracy);
    assert.ok(rows.every(r => r.cpuAgree >= 0.98));
});

test('experiment batch-speed: every batch size, finite times', async (t) => {
    const e = byId['batch-speed'];
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: batch wallMs images/s: ${show(rows, ['batch', 'wallMs', 'imagesPerSecond'])}`);
    assert.deepEqual(rows.map(r => r.batch), [8, 16, 32, 64, 128]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.wallMs > 0 && Number.isFinite(row.trainLoss));
    }
});
