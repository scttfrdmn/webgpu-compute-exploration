// GPU execution of the shared MNIST CNN kernels (js/examples/mnist-model.js buildComputeShader(),
// exactly the two variants wgslModules() exports) vs the CPU trainer tools/train-mnist.mjs, which
// produced data/mnist/weights.bin. CI-only on a software GPU — see test/gpu/harness.mjs.
//
// The dispatch sequence and the Params bytes are taken from the shipped MNISTGPU driver itself:
// MNISTGPU is constructed on the recording mock device, writeSlots() supplies the uniform bytes and
// encodeBatch() is replayed into a recording pass to get (entry point, workgroups) in order. The
// only difference from the page is that Params is bound without a dynamic offset (one slot, rewritten
// between batches with queue.writeBuffer, which is ordered with the submitted dispatches).
//
// Data: the first 256 training images and 40 test images of the MNIST files tracked in data/mnist/.
// Tolerances are the CPU-replica ones (test/mnist-kernels.test.mjs): the trainer accumulates in f64,
// the GPU in f32 (possibly with fused multiply-adds), so agreement is ~1e-6 relative, not bitwise.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { computePipeline, bindGroup, buffer, read, run } from './harness.mjs';
import {
    MNISTGPU, NUM_PARAMS, OFF, DEFAULTS, MAX_BATCH, STATS_SIZE, PARAMS_SIZE, wgslModules,
    initWeights, epochOrder, augmentSeed, sampleShift, parseWeights,
} from '../../js/examples/mnist-model.js';
import {
    loadDatasetNode, createScratch, forward, softmaxLoss, backward, loadImage, trainBatch, evaluate,
} from '../../tools/train-mnist.mjs';
import { mockDevice } from '../helpers/mock-gpu.mjs';

const IMG = 784, NPARAM = NUM_PARAMS;
const TENSORS = [[OFF.W1, 200], [OFF.B1, 8], [OFF.W2, 3200], [OFF.B2, 16], [OFF.W3, 2560], [OFF.B3, 10]];

const maxRel = (a, b) => {
    let scale = 0, err = 0;
    for (let i = 0; i < b.length; i++) {
        scale = Math.max(scale, Math.abs(b[i]));
        err = Math.max(err, Number.isFinite(a[i]) ? Math.abs(a[i] - b[i]) : Infinity);
    }
    return { err, scale, rel: err / Math.max(scale, 1e-30) };
};

// A GPU-side MNISTGPU(maxBatch) equivalent built from the shipped module and driver.
class GPUNet {
    static async create(maxBatch, dataset, order) {
        const net = new GPUNet();
        const code = wgslModules().find(m => m.name === `compute-batch${maxBatch}`).code;
        net.mock = mockDevice();
        net.driver = new MNISTGPU(net.mock, maxBatch, 1);   // layout, writeSlots, encodeBatch
        net.L = net.driver.layout;
        const names = Object.keys(net.driver.pipelines);
        net.pipelines = {};
        for (const name of names) net.pipelines[name] = await computePipeline(code, name);
        net.params = buffer(PARAMS_SIZE);
        net.weights = buffer(NPARAM * 2 * 4);
        net.work = buffer(net.L.total * 4);
        net.grads = buffer(maxBatch * NPARAM * 4);
        net.stats = buffer(STATS_SIZE);
        const dummy = buffer(16);
        const res = {
            params: net.params,
            images: dataset ? buffer(dataset.images) : dummy,
            labels: dataset ? buffer(dataset.labels) : dummy,
            order: order ? buffer(order) : dummy,
            weights: net.weights, work: net.work, grads: net.grads, stats: net.stats,
        };
        net.groups = Object.fromEntries(names.map(n => [n, bindGroup(net.pipelines[n], 0, res)]));
        return net;
    }

    // Params bytes exactly as MNISTGPU.writeSlots() produces them for slot 0
    paramsBytes(step) {
        this.driver.writeSlots([step]);
        return this.mock.lastWrite(this.driver.paramsBuffer).bytes.slice(0, PARAMS_SIZE);
    }

    // [(entryPoint, workgroups)] in the order MNISTGPU.encodeBatch() records them
    sequence(step) {
        const byPipeline = new Map(Object.entries(this.driver.pipelines).map(([n, p]) => [p, n]));
        const seq = [];
        let cur;
        const pass = {
            setBindGroup() {},
            setPipeline(p) { cur = byPipeline.get(p); },
            dispatchWorkgroups(x, y = 1, z = 1) { seq.push([cur, [x, y, z]]); },
        };
        this.driver.encodeBatch(pass, null, 0, step);
        assert.ok(seq.every(([n]) => n), 'every dispatch maps to a named pipeline');
        return seq;
    }

    // Steps for one batch, to be concatenated and submitted with run()
    batchSteps(step) {
        return [{ write: [this.params, this.paramsBytes(step), 0] },
            ...this.sequence(step).map(([n, wg]) => ({ p: this.pipelines[n], groups: [this.groups[n]], workgroups: wg }))];
    }

    async readStats() {
        const bytes = await read(this.stats, Uint8Array, STATS_SIZE);
        const f = new Float32Array(bytes.buffer), u = new Uint32Array(bytes.buffer);
        return { trainLoss: f[0], trainCorrect: u[1], trainCount: u[2], testLoss: f[4], testCorrect: u[5], testCount: u[6] };
    }
}

const trainData = loadDatasetNode('train');
const testData = loadDatasetNode('test');
const SUBSET = 256;
const subset = { images: trainData.images.subarray(0, SUBSET * IMG), labels: trainData.labels.subarray(0, SUBSET), count: SUBSET };

test('mnist-model: forward + backward per sample on a shifted training batch vs the CPU trainer', async (t) => {
    const B = 6;
    const seed = DEFAULTS.seed, epoch = 1;
    const order = epochOrder(SUBSET, seed, epoch);
    const net = await GPUNet.create(MAX_BATCH, subset, order);
    const w = initWeights(seed);
    const wv = new Float32Array(2 * NPARAM);
    wv.set(w);
    // lr = 0, momentum = 0, weightDecay = 0: cs_update then leaves the weights unchanged
    const step = { start: 40, count: B, mode: 0, dataLen: SUBSET, lr: 0, momentum: 0, weightDecay: 0,
        augSeed: augmentSeed(seed, epoch), maxShift: DEFAULTS.maxShift };
    await run([{ write: [net.weights, wv, 0] }, ...net.batchSteps(step)], 'mnist train batch');
    const work = await read(net.work);
    const grads = await read(net.grads);
    const weightsAfter = await read(net.weights, Float32Array, NPARAM * 4);
    const stats = await net.readStats();
    const L = net.L;

    const s = createScratch();
    const worst = { x: 0, logit: 0, prob: 0, loss: 0, grad: 0, dz: 0 };
    let cpuLoss = 0, cpuCorrect = 0;
    for (let b = 0; b < B; b++) {
        const index = order[step.start + b];
        const [dx, dy] = sampleShift(step.augSeed, step.start + b, step.maxShift);
        loadImage(s, subset.images, index, dx, dy);
        if (forward(w, s) === subset.labels[index]) cpuCorrect++;
        const loss = softmaxLoss(s, subset.labels[index]);
        cpuLoss += loss;
        backward(w, s);
        worst.x = Math.max(worst.x, maxRel(work.subarray(L.X + b * IMG, L.X + (b + 1) * IMG), s.x).err);
        worst.logit = Math.max(worst.logit, maxRel(work.subarray(L.LOGIT + b * 10, L.LOGIT + b * 10 + 10), s.logit).rel);
        worst.prob = Math.max(worst.prob, maxRel(work.subarray(L.PROB + b * 10, L.PROB + b * 10 + 10), s.prob).err);
        worst.loss = Math.max(worst.loss, Math.abs(work[L.LOSS + b] - loss));
        worst.dz = Math.max(worst.dz, maxRel(work.subarray(L.DZ1 + b * 4608, L.DZ1 + (b + 1) * 4608), s.dz1).rel);
        // each tensor of the gradient row, relative to that tensor's largest |g|
        for (const [o, n] of TENSORS) {
            worst.grad = Math.max(worst.grad, maxRel(grads.subarray(b * NPARAM + o, b * NPARAM + o + n), s.g.subarray(o, o + n)).rel);
        }
    }
    t.diagnostic(`B=${B}: input max |Δ| ${worst.x.toExponential(1)}, logits max rel ${worst.logit.toExponential(2)}, ` +
        `probs max |Δ| ${worst.prob.toExponential(2)}, loss max |Δ| ${worst.loss.toExponential(2)}, ` +
        `dL/dz1 max rel ${worst.dz.toExponential(2)}, per-tensor gradient max rel ${worst.grad.toExponential(2)}; ` +
        `Stats trainLoss ${stats.trainLoss.toFixed(5)} vs ${cpuLoss.toFixed(5)}, correct ${stats.trainCorrect} vs ${cpuCorrect}`);
    assert.equal(worst.x, 0, 'cs_gather (packed bytes + hashed shift) is bit-exact with loadImage + sampleShift');
    assert.ok(worst.logit < 1e-5, `logits ${worst.logit}`);   // f32 accumulation over ≤ 200 terms vs f64
    assert.ok(worst.prob < 1e-6, `probs ${worst.prob}`);
    assert.ok(worst.loss < 1e-5, `loss ${worst.loss}`);
    assert.ok(worst.grad < 1e-5, `gradients ${worst.grad} (f32 sums of up to 576 products vs f64)`);
    assert.equal(stats.trainCount, B);
    assert.equal(stats.trainCorrect, cpuCorrect);
    assert.ok(Math.abs(stats.trainLoss - cpuLoss) < 1e-5 * B);
    assert.ok(maxRel(weightsAfter, w).err === 0, 'lr = 0 leaves the weights unchanged');
});

test('mnist-model: 4 SGD steps (cs_grads + cs_update) track trainBatch()', async (t) => {
    const B = 8, steps = 4;
    const seed = 3, epoch = 0;
    const order = epochOrder(SUBSET, seed, epoch);
    const net = await GPUNet.create(MAX_BATCH, subset, order);
    const w = initWeights(seed);
    const wv = new Float32Array(2 * NPARAM);
    wv.set(w);
    const v = new Float32Array(NPARAM);
    const s = createScratch();
    const o = { lr: 0.04, momentum: 0.9, weightDecay: 0.0005, augSeed: augmentSeed(seed, epoch), maxShift: 2 };
    const seq = [{ write: [net.weights, wv, 0] }];
    let cpuCorrect = 0;
    for (let k = 0; k < steps; k++) {
        seq.push(...net.batchSteps({ start: k * B, count: B, mode: 0, dataLen: SUBSET, ...o }));
        cpuCorrect += trainBatch(w, v, subset, order, k * B, B, o, s).correct;
    }
    await run(seq, 'mnist SGD steps');
    const got = await read(net.weights, Float32Array, 2 * NPARAM * 4);
    const stats = await net.readStats();
    const dw = maxRel(got.subarray(0, NPARAM), w);
    const dv = maxRel(got.subarray(NPARAM), v);
    const w0 = initWeights(seed);
    let moved = 0;
    for (let i = 0; i < NPARAM; i++) moved = Math.max(moved, Math.abs(w[i] - w0[i]));
    t.diagnostic(`after ${steps} steps of B=${B}: weights max |Δ| ${dw.err.toExponential(2)} (largest weight move ` +
        `${moved.toExponential(2)}), momentum max |Δ| ${dv.err.toExponential(2)} rel ${dv.rel.toExponential(2)} ` +
        `(max |v| ${dv.scale.toExponential(2)}); correct ${stats.trainCorrect} vs ${cpuCorrect}`);
    // the documented agreement is "about 1e-7" (f32 vs f64 local sums); 1e-6 absolute on O(0.1-1) weights
    assert.ok(dw.err < 1e-6, `weights differ by ${dw.err}`);
    assert.ok(dv.rel < 1e-4, `momentum differs by ${dv.rel} relative`);
    assert.ok(moved > 1e-3, 'the steps actually moved the weights');
    assert.equal(stats.trainCount, B * steps);
    assert.equal(stats.trainCorrect, cpuCorrect);
});

const shippedWeights = () =>
    parseWeights(readFileSync(fileURLToPath(new URL('../../data/mnist/weights.bin', import.meta.url))));

test('mnist-model: evaluation batches (mode 1) with the shipped weights vs evaluate()', async (t) => {
    const w = shippedWeights();
    const n = 40, B = 16;
    const data = { images: testData.images.subarray(0, n * IMG), labels: testData.labels.subarray(0, n), count: n };
    const net = await GPUNet.create(MAX_BATCH, data, null);
    const wv = new Float32Array(2 * NPARAM);
    wv.set(w);
    const seq = [{ write: [net.weights, wv, 0] }];
    for (let start = 0; start < n; start += B) seq.push(...net.batchSteps({ start, count: Math.min(B, n - start), mode: 1, dataLen: n }));
    await run(seq, 'mnist evaluation');
    const stats = await net.readStats();
    const ref = evaluate(w, data, createScratch());
    t.diagnostic(`${n} test images: GPU correct ${stats.testCorrect}, mean loss ${(stats.testLoss / n).toFixed(6)}; ` +
        `evaluate() correct ${ref.correct}, mean loss ${ref.loss.toFixed(6)}; |Δ loss| ${Math.abs(stats.testLoss / n - ref.loss).toExponential(2)}`);
    assert.equal(stats.testCount, n);
    assert.equal(stats.trainCount, 0);
    assert.equal(stats.testCorrect, ref.correct);
    assert.ok(Math.abs(stats.testLoss / n - ref.loss) < 1e-5);
});

test('mnist-model: Recognize path (compute-batch1, mode 2, CPU-written input) probabilities', async (t) => {
    const w = shippedWeights();
    const net = await GPUNet.create(1, null, null);
    const s = createScratch();
    let worst = 0, agree = 0;
    const N = 8;
    for (let i = 0; i < N; i++) {
        loadImage(s, testData.images, i);
        const wv = new Float32Array(2 * NPARAM);
        wv.set(w);
        await run([{ write: [net.weights, wv, 0] }, { write: [net.work, s.x, net.L.X * 4] },
            ...net.batchSteps({ start: 0, count: 1, mode: 2, dataLen: 1 })], 'mnist recognize');
        const prob = await read(net.work, Float32Array, 40, net.L.PROB * 4);
        const pred = forward(w, s);
        softmaxLoss(s, 0);
        worst = Math.max(worst, maxRel(prob, s.prob).err);
        if (prob.indexOf(Math.max(...prob)) === pred) agree++;
    }
    const stats = await net.readStats();
    t.diagnostic(`${N} test images: probabilities max |Δ| ${worst.toExponential(2)}, argmax agrees ${agree}/${N}`);
    assert.ok(worst < 1e-6);
    assert.equal(agree, N);
    assert.equal(stats.testCount + stats.trainCount, 0, 'mode 2 records no stats');
});
