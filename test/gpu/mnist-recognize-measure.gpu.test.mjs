// MNIST inference compute-only API (evaluateWeights, measureInference in js/examples/mnist-model.js) and
// the Measure experiments built on it (js/teach/experiments/mnist-recognize.js), on a software GPU —
// CI only, see harness.mjs.
//
// Both drive the shipped MNISTGPU on the real device. Checked against: the trainer's recorded result
// for data/mnist/weights.bin (1,960 / 2,000, data/mnist/README.md) and the float64 forward pass of
// js/refs/mnist.js, image by image. The experiments load data with loadDataset() / loadCachedWeights()
// (fetch); here fetch is given file: URLs of data/mnist/. Printed numbers are the CI baselines.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { device } from './harness.mjs';
import { evaluateWeights, measureInference, parseWeights } from '../../js/examples/mnist-model.js';
import { loadDatasetNode } from '../../tools/train-mnist.mjs';
import { mnistEvaluate, argmax } from '../../js/refs/mnist.js';
import experiments from '../../js/teach/experiments/mnist-recognize.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const testSet = loadDatasetNode('test');
const weights = parseWeights(readFileSync(new URL('../../data/mnist/weights.bin', import.meta.url)));

const nodeFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
    const u = new URL(url);
    return u.protocol === 'file:' ? new Response(readFileSync(fileURLToPath(u))) : nodeFetch(url);
};

test('evaluateWeights: cached weights reach 1,960 / 2,000, image by image equal to the float64 forward pass', async (t) => {
    for (const batch of [128, 37]) {
        const r = await evaluateWeights(gpuDevice, { weights, data: testSet, batch });
        const ref = mnistEvaluate(weights, testSet);
        let agree = 0, maxDiff = 0;
        for (let i = 0; i < testSet.count; i++) {
            if (argmax(r.probs, 10 * i) === argmax(ref.probs, 10 * i)) agree++;
            for (let k = 0; k < 10; k++) maxDiff = Math.max(maxDiff, Math.abs(r.probs[10 * i + k] - ref.probs[10 * i + k]));
        }
        t.diagnostic(`batch ${batch}: ${r.correct}/${r.count} (loss ${r.loss.toFixed(5)}, float64 ${ref.loss.toFixed(5)}), `
            + `predictions agree ${agree}/${testSet.count}, max |p_gpu − p_f64| ${maxDiff.toExponential(2)}, ${r.wallMs.toFixed(0)} ms`);
        assert.equal(r.count, 2000);
        assert.equal(r.correct, 1960);
        assert.equal(ref.correct, 1960);
        assert.equal(agree, testSet.count);
        assert.ok(maxDiff < 1e-4 && Math.abs(r.loss - ref.loss) < 1e-4);
    }
});

test('measureInference: batch 1 and 128 give finite round trips', async (t) => {
    for (const batch of [1, 128]) {
        const r = await measureInference(gpuDevice, { weights, data: testSet, batch, warmup: 1, repeats: 3 });
        t.diagnostic(`batch ${batch}: round trip ${r.wallMs.toFixed(2)} ms, ${(1000 * r.perImageMs).toFixed(1)} µs per image`);
        assert.equal(r.batch, batch);
        assert.ok(r.wallMs > 0 && r.perImageMs > 0);
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
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toPrecision(4)) : String(v);
}).join(' ')).join('; ');

test('experiment test-accuracy: 98.00% on 2,000 images, GPU = CPU on every prefix', async (t) => {
    const e = byId['test-accuracy'];
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: images accuracy low high agree maxDiff: ${show(rows, ['images', 'accuracy', 'low', 'high', 'agree', 'maxDiff'])}`);
    assert.deepEqual(rows.map(r => r.images), [125, 250, 500, 1000, 2000]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.equal(row.agree, 1);
        assert.equal(row.accuracy, row.cpuAccuracy);
        assert.ok(row.low <= row.accuracy && row.accuracy <= row.high);
    }
    assert.equal(rows[4].accuracy, 0.98);
});

test('experiment latency: every batch size', async (t) => {
    const e = byId['latency'];
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: batch wallMs µs/image: ${show(rows, ['batch', 'wallMs', 'perImageUs'])}`);
    assert.deepEqual(rows.map(r => r.batch), [1, 2, 4, 8, 16, 32, 64, 128]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.wallMs > 0);
    }
    assert.ok(rows[7].perImageUs < rows[0].perImageUs, 'batching lowers the cost per image');
});
