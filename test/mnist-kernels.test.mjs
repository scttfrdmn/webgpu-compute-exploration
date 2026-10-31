// CPU replica of the MNIST compute kernels in js/examples/mnist-model.js buildComputeShader()
// (cs_gather … cs_update); validates the algorithm and index math, not the WGSL itself — see #51.
//
// Each kernel is replicated as a loop over its global invocation id, decomposing the id exactly as
// the WGSL does (b = id / N, r = id % N, …), reading and writing one flat f32 work buffer laid out by
// the shipped workLayout(), with packed-u32 image/label buffers and f32 arithmetic (Math.fround after
// every operation). The independent reference is the shipped CPU trainer tools/train-mnist.mjs
// (straight nested loops over tensors, f64 accumulation), which produced data/mnist/weights.bin:
//  - forward activations, probabilities and loss per sample vs forward()/softmaxLoss()
//  - per-sample gradient rows (cs_grads) vs backward()
//  - weights and momentum after several SGD steps (cs_update) vs trainBatch()
//  - mode-1 evaluation batches (sequential, index clamp) and the Stats sums vs evaluate().

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    NUM_PARAMS, OFF, GRAD_CLIP, DEFAULTS, workLayout, initWeights, epochOrder, augmentSeed, sampleShift, parseWeights,
} from '../js/examples/mnist-model.js';
import {
    loadDatasetNode, createScratch, forward, softmaxLoss, backward, loadImage, trainBatch, evaluate,
} from '../tools/train-mnist.mjs';

const f = Math.fround;
const IMG = 784, N1 = 4608, NP1 = 1152, N2 = 1024, NP2 = 256, NCLS = 10, NPARAM = NUM_PARAMS;
const LOGIT_CLAMP = 1e4;

function pack(bytes) {
    const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
    padded.set(bytes);
    return new Uint32Array(padded.buffer);
}

const u32 = (x) => x >>> 0;
function pcg(v) {
    const state = u32(Math.imul(v, 747796405) + 2891336453);
    const word = u32(Math.imul(u32((state >>> ((state >>> 28) + 4)) ^ state), 277803737));
    return u32((word >>> 22) ^ word);
}
const unpackByte = (word, byteIndex) => (word >>> ((byteIndex & 3) * 8)) & 0xff;
const clampF = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

// The GPU state: buffers bound to the compute module, and the kernels as JS functions.
class Replica {
    constructor(maxBatch, dataset, order) {
        this.MAX_BATCH = maxBatch;
        this.L = workLayout(maxBatch);
        this.images = pack(dataset.images);
        this.labels = pack(dataset.labels);
        this.order = order;
        this.weights = new Float32Array(2 * NPARAM);
        this.work = new Float32Array(this.L.total);
        this.grads = new Float32Array(maxBatch * NPARAM);
        this.stats = { trainLoss: 0, trainCorrect: 0, trainCount: 0, testLoss: 0, testCorrect: 0, testCount: 0 };
    }
    run(kernel, invocations) { for (let id = 0; id < invocations; id++) kernel.call(this, id); }

    imageIndex(b) {
        const p = this.params;
        let i = p.start + b;
        if (p.mode === 0) i = this.order[p.start + b];
        return Math.min(i, p.dataLen - 1);
    }
    labelOf(b) { const i = this.imageIndex(b); return unpackByte(this.labels[i >>> 2], i); }

    cs_gather(id) {
        const p = this.params, W = this.work, L = this.L;
        if (id >= p.count * IMG) return;
        const b = Math.floor(id / IMG), i = id % IMG;
        let dx = 0, dy = 0;
        if (p.mode === 0 && p.maxShift > 0) {
            const h = pcg(u32(p.augSeed ^ pcg(p.start + b)));
            const span = 2 * p.maxShift + 1;
            dx = (h % span) - p.maxShift;
            dy = ((h >>> 8) % span) - p.maxShift;
        }
        const sx = (i % 28) - dx, sy = Math.floor(i / 28) - dy;
        let v = 0;
        if (sx >= 0 && sy >= 0 && sx < 28 && sy < 28) {
            const byteIndex = this.imageIndex(b) * IMG + sy * 28 + sx;
            v = unpackByte(this.images[byteIndex >>> 2], byteIndex);
        }
        W[L.X + b * IMG + i] = f(v / 255);
    }
    cs_conv1(id) {
        const p = this.params, W = this.work, L = this.L, w = this.weights;
        if (id >= p.count * N1) return;
        const b = Math.floor(id / N1), r = id % N1, c = Math.floor(r / 576), q = r % 576;
        const oy = Math.floor(q / 24), ox = q % 24, xBase = L.X + b * IMG;
        let acc = w[OFF.B1 + c];
        for (let ky = 0; ky < 5; ky++) for (let kx = 0; kx < 5; kx++) {
            acc = f(acc + f(w[OFF.W1 + c * 25 + ky * 5 + kx] * W[xBase + (oy + ky) * 28 + ox + kx]));
        }
        W[L.A1 + b * N1 + r] = Math.max(acc, 0);
    }
    argmax4(base, rowStride) {
        const W = this.work;
        let best = W[base], bestK = 0;
        for (let k = 1; k < 4; k++) {
            const v = W[base + Math.floor(k / 2) * rowStride + (k % 2)];
            if (v > best) { best = v; bestK = k; }
        }
        return bestK;
    }
    cs_pool1(id) {
        const p = this.params, W = this.work, L = this.L;
        if (id >= p.count * NP1) return;
        const b = Math.floor(id / NP1), r = id % NP1, c = Math.floor(r / 144), q = r % 144;
        const base = L.A1 + b * N1 + c * 576 + Math.floor(q / 12) * 48 + (q % 12) * 2;
        const k = this.argmax4(base, 24);
        W[L.P1 + b * NP1 + r] = W[base + Math.floor(k / 2) * 24 + (k % 2)];
    }
    cs_conv2(id) {
        const p = this.params, W = this.work, L = this.L, w = this.weights;
        if (id >= p.count * N2) return;
        const b = Math.floor(id / N2), r = id % N2, co = Math.floor(r / 64), q = r % 64;
        const oy = Math.floor(q / 8), ox = q % 8, inBase = L.P1 + b * NP1;
        let acc = w[OFF.B2 + co];
        for (let ci = 0; ci < 8; ci++) for (let ky = 0; ky < 5; ky++) for (let kx = 0; kx < 5; kx++) {
            acc = f(acc + f(w[OFF.W2 + ((co * 8 + ci) * 5 + ky) * 5 + kx] * W[inBase + ci * 144 + (oy + ky) * 12 + ox + kx]));
        }
        W[L.A2 + b * N2 + r] = Math.max(acc, 0);
    }
    cs_pool2(id) {
        const p = this.params, W = this.work, L = this.L;
        if (id >= p.count * NP2) return;
        const b = Math.floor(id / NP2), r = id % NP2, c = Math.floor(r / 16), q = r % 16;
        const base = L.A2 + b * N2 + c * 64 + Math.floor(q / 4) * 16 + (q % 4) * 2;
        const k = this.argmax4(base, 8);
        W[L.P2 + b * NP2 + r] = W[base + Math.floor(k / 2) * 8 + (k % 2)];
    }
    cs_dense(id) {
        const p = this.params, W = this.work, L = this.L, w = this.weights;
        if (id >= p.count * NCLS) return;
        const b = Math.floor(id / NCLS), k = id % NCLS, inBase = L.P2 + b * NP2;
        let acc = w[OFF.B3 + k];
        for (let j = 0; j < NP2; j++) acc = f(acc + f(w[OFF.W3 + k * NP2 + j] * W[inBase + j]));
        W[L.LOGIT + b * NCLS + k] = acc;
    }
    cs_softmax(b) {
        const p = this.params, W = this.work, L = this.L;
        if (b >= p.count) return;
        const base = b * NCLS;
        let m = -LOGIT_CLAMP, best = 0;
        for (let k = 0; k < NCLS; k++) {
            const l = clampF(W[L.LOGIT + base + k], -LOGIT_CLAMP, LOGIT_CLAMP);
            if (l > m) { m = l; best = k; }
        }
        let s = 0;
        for (let k = 0; k < NCLS; k++) s = f(s + f(Math.exp(f(clampF(W[L.LOGIT + base + k], -LOGIT_CLAMP, LOGIT_CLAMP) - m))));
        let label = NCLS;
        if (p.mode !== 2) label = this.labelOf(b);
        let labelLogit = m;
        for (let k = 0; k < NCLS; k++) {
            const l = clampF(W[L.LOGIT + base + k], -LOGIT_CLAMP, LOGIT_CLAMP);
            const prob = f(f(Math.exp(f(l - m))) / s);
            W[L.PROB + base + k] = prob;
            let onehot = 0;
            if (k === label) { onehot = 1; labelLogit = l; }
            W[L.DLOG + base + k] = f(prob - onehot);
        }
        W[L.LOSS + b] = Math.max(f(f(m + f(Math.log(s))) - labelLogit), 0);
        W[L.CORR + b] = best === label ? 1 : 0;
    }
    cs_stats() {
        const p = this.params, W = this.work, L = this.L, st = this.stats;
        let loss = 0, correct = 0;
        for (let b = 0; b < this.MAX_BATCH; b++) {
            if (b >= p.count) break;
            loss = f(loss + W[L.LOSS + b]);
            correct += W[L.CORR + b];
        }
        if (p.mode === 0) { st.trainLoss = f(st.trainLoss + loss); st.trainCorrect += correct; st.trainCount += p.count; }
        else { st.testLoss = f(st.testLoss + loss); st.testCorrect += correct; st.testCount += p.count; }
    }
    cs_dense_back(id) {
        const p = this.params, W = this.work, L = this.L, w = this.weights;
        if (id >= p.count * NP2) return;
        const b = Math.floor(id / NP2), j = id % NP2;
        let acc = 0;
        for (let k = 0; k < NCLS; k++) acc = f(acc + f(w[OFF.W3 + k * NP2 + j] * W[L.DLOG + b * NCLS + k]));
        W[L.DP2 + b * NP2 + j] = acc;
    }
    poolBack(aBase, size, c, y, x, dpBase) {
        const W = this.work, half = size / 2;
        const a = W[aBase + c * size * size + y * size + x];
        const winBase = aBase + c * size * size + (y & ~1) * size + (x & ~1);
        const k = this.argmax4(winBase, size);
        if (k !== (y % 2) * 2 + (x % 2) || a <= 0) return 0;
        return W[dpBase + c * half * half + Math.floor(y / 2) * half + Math.floor(x / 2)];
    }
    cs_pool2_back(id) {
        const p = this.params, L = this.L;
        if (id >= p.count * N2) return;
        const b = Math.floor(id / N2), r = id % N2, q = r % 64;
        this.work[L.DZ2 + b * N2 + r] = this.poolBack(L.A2 + b * N2, 8, Math.floor(r / 64), Math.floor(q / 8), q % 8, L.DP2 + b * NP2);
    }
    cs_conv2_back(id) {
        const p = this.params, W = this.work, L = this.L, w = this.weights;
        if (id >= p.count * NP1) return;
        const b = Math.floor(id / NP1), r = id % NP1, ci = Math.floor(r / 144), q = r % 144;
        const y = Math.floor(q / 12), x = q % 12, dzBase = L.DZ2 + b * N2;
        let acc = 0;
        for (let co = 0; co < 16; co++) for (let ky = 0; ky < 5; ky++) {
            const oy = y - ky;
            if (oy < 0 || oy >= 8) continue;
            for (let kx = 0; kx < 5; kx++) {
                const ox = x - kx;
                if (ox < 0 || ox >= 8) continue;
                acc = f(acc + f(w[OFF.W2 + ((co * 8 + ci) * 5 + ky) * 5 + kx] * W[dzBase + co * 64 + oy * 8 + ox]));
            }
        }
        W[L.DP1 + b * NP1 + r] = acc;
    }
    cs_pool1_back(id) {
        const p = this.params, L = this.L;
        if (id >= p.count * N1) return;
        const b = Math.floor(id / N1), r = id % N1, q = r % 576;
        this.work[L.DZ1 + b * N1 + r] = this.poolBack(L.A1 + b * N1, 24, Math.floor(r / 576), Math.floor(q / 24), q % 24, L.DP1 + b * NP1);
    }
    cs_grads(id) {
        const p = this.params, W = this.work, L = this.L;
        if (id >= p.count * NPARAM) return;
        const b = Math.floor(id / NPARAM), q0 = id % NPARAM;
        let g = 0;
        if (q0 < OFF.B1) {
            const c = Math.floor(q0 / 25), ky = Math.floor((q0 % 25) / 5), kx = q0 % 5;
            const dzBase = L.DZ1 + b * N1 + c * 576, xBase = L.X + b * IMG;
            for (let oy = 0; oy < 24; oy++) for (let ox = 0; ox < 24; ox++) {
                g = f(g + f(W[dzBase + oy * 24 + ox] * W[xBase + (oy + ky) * 28 + ox + kx]));
            }
        } else if (q0 < OFF.W2) {
            const dzBase = L.DZ1 + b * N1 + (q0 - OFF.B1) * 576;
            for (let i = 0; i < 576; i++) g = f(g + W[dzBase + i]);
        } else if (q0 < OFF.B2) {
            const q = q0 - OFF.W2;
            const co = Math.floor(q / 200), ci = Math.floor(q / 25) % 8, ky = Math.floor((q % 25) / 5), kx = q % 5;
            const dzBase = L.DZ2 + b * N2 + co * 64, inBase = L.P1 + b * NP1 + ci * 144;
            for (let oy = 0; oy < 8; oy++) for (let ox = 0; ox < 8; ox++) {
                g = f(g + f(W[dzBase + oy * 8 + ox] * W[inBase + (oy + ky) * 12 + ox + kx]));
            }
        } else if (q0 < OFF.W3) {
            const dzBase = L.DZ2 + b * N2 + (q0 - OFF.B2) * 64;
            for (let i = 0; i < 64; i++) g = f(g + W[dzBase + i]);
        } else if (q0 < OFF.B3) {
            const q = q0 - OFF.W3;
            g = f(W[L.DLOG + b * NCLS + Math.floor(q / NP2)] * W[L.P2 + b * NP2 + q % NP2]);
        } else {
            g = W[L.DLOG + b * NCLS + (q0 - OFF.B3)];
        }
        this.grads[b * NPARAM + q0] = g;
    }
    cs_update(q) {
        const p = this.params, w = this.weights;
        if (q >= NPARAM) return;
        let g = 0;
        for (let b = 0; b < this.MAX_BATCH; b++) {
            if (b >= p.count) break;
            g = f(g + this.grads[b * NPARAM + q]);
        }
        g = clampF(f(f(g / p.count) + f(f(p.weightDecay) * w[q])), -GRAD_CLIP, GRAD_CLIP);
        const v = f(f(f(p.momentum) * w[NPARAM + q]) + g);
        w[NPARAM + q] = v;
        w[q] = f(w[q] - f(f(p.lr) * v));
    }

    // MNISTGPU.encodeBatch: the dispatch sequence of one mini-batch
    encodeBatch(params, { backwardOnly = false } = {}) {
        this.params = params;
        const n = params.count;
        if (!backwardOnly) {
            if (params.mode !== 2) this.run(this.cs_gather, n * IMG);
            this.run(this.cs_conv1, n * N1);
            this.run(this.cs_pool1, n * NP1);
            this.run(this.cs_conv2, n * N2);
            this.run(this.cs_pool2, n * NP2);
            this.run(this.cs_dense, n * NCLS);
            this.run(this.cs_softmax, n);
            if (params.mode === 2) return;
            this.cs_stats();
            if (params.mode !== 0) return;
        }
        this.run(this.cs_dense_back, n * NP2);
        this.run(this.cs_pool2_back, n * N2);
        this.run(this.cs_conv2_back, n * NP1);
        this.run(this.cs_pool1_back, n * N1);
        this.run(this.cs_grads, n * NPARAM);
        this.run(this.cs_update, NPARAM);
    }
}

const trainData = loadDatasetNode('train');
const testData = loadDatasetNode('test');
const SUBSET = 256;
const subset = { images: trainData.images.subarray(0, SUBSET * IMG), labels: trainData.labels.subarray(0, SUBSET), count: SUBSET };

const maxRel = (a, b) => {
    let scale = 0, err = 0;
    for (let i = 0; i < a.length; i++) { scale = Math.max(scale, Math.abs(b[i])); err = Math.max(err, Math.abs(a[i] - b[i])); }
    return { err, scale, rel: err / Math.max(scale, 1e-30) };
};

test('replica forward + backward per sample matches the CPU trainer (shifted training batch)', (t) => {
    const B = 6;
    const seed = DEFAULTS.seed, epoch = 1;
    const order = epochOrder(SUBSET, seed, epoch);
    const gpu = new Replica(B, subset, order);
    gpu.weights.set(initWeights(seed));
    const params = { start: 40, count: B, mode: 0, dataLen: SUBSET, lr: 0, momentum: 0, weightDecay: 0,
        augSeed: augmentSeed(seed, epoch), maxShift: DEFAULTS.maxShift };
    gpu.encodeBatch(params);

    const s = createScratch();
    const w = initWeights(seed);
    const L = gpu.L;
    let worst = { x: 0, logit: 0, prob: 0, loss: 0, grad: 0 };
    let cpuLoss = 0;
    for (let b = 0; b < B; b++) {
        const index = order[params.start + b];
        const [dx, dy] = sampleShift(params.augSeed, params.start + b, params.maxShift);
        loadImage(s, subset.images, index, dx, dy);
        forward(w, s);
        const loss = softmaxLoss(s, subset.labels[index]);
        cpuLoss += loss;
        backward(w, s);
        worst.x = Math.max(worst.x, maxRel(gpu.work.subarray(L.X + b * IMG, L.X + (b + 1) * IMG), s.x).err);
        worst.logit = Math.max(worst.logit, maxRel(gpu.work.subarray(L.LOGIT + b * 10, L.LOGIT + b * 10 + 10), s.logit).rel);
        worst.prob = Math.max(worst.prob, maxRel(gpu.work.subarray(L.PROB + b * 10, L.PROB + b * 10 + 10), s.prob).err);
        worst.loss = Math.max(worst.loss, Math.abs(gpu.work[L.LOSS + b] - loss));
        // each tensor of the gradient row, relative to that tensor's largest |g|
        for (const [o, n] of [[OFF.W1, 200], [OFF.B1, 8], [OFF.W2, 3200], [OFF.B2, 16], [OFF.W3, 2560], [OFF.B3, 10]]) {
            const r = maxRel(gpu.grads.subarray(b * NPARAM + o, b * NPARAM + o + n), s.g.subarray(o, o + n));
            worst.grad = Math.max(worst.grad, r.rel);
        }
    }
    t.diagnostic(`B=${B}: input max |Δ| ${worst.x.toExponential(1)}, logits max rel ${worst.logit.toExponential(2)}, ` +
        `probs max |Δ| ${worst.prob.toExponential(2)}, loss max |Δ| ${worst.loss.toExponential(2)}, ` +
        `per-tensor gradient max rel ${worst.grad.toExponential(2)}; Stats trainLoss ${gpu.stats.trainLoss.toFixed(5)} vs ${cpuLoss.toFixed(5)}`);
    assert.equal(worst.x, 0, 'gather (packed bytes + hashed shift) is bit-exact with loadImage + sampleShift');
    // f32 accumulation over ≤ 200 terms vs f64: ~1e-6 relative
    assert.ok(worst.logit < 1e-5);
    assert.ok(worst.prob < 1e-6);
    assert.ok(worst.loss < 1e-5);
    assert.ok(worst.grad < 1e-5, 'f32 sums of up to 576 products vs f64: measured ~1e-6');
    assert.equal(gpu.stats.trainCount, B);
    assert.ok(Math.abs(gpu.stats.trainLoss - cpuLoss) < 1e-5 * B);
});

test('replica SGD steps (cs_grads + cs_update) track trainBatch() over 4 mini-batches', (t) => {
    const B = 8, steps = 4;
    const seed = 3, epoch = 0;
    const order = epochOrder(SUBSET, seed, epoch);
    const gpu = new Replica(B, subset, order);
    const w = initWeights(seed);
    gpu.weights.set(w);
    const v = new Float32Array(NPARAM);
    const s = createScratch();
    const o = { lr: 0.04, momentum: 0.9, weightDecay: 0.0005, augSeed: augmentSeed(seed, epoch), maxShift: 2 };
    let cpuCorrect = 0;
    for (let k = 0; k < steps; k++) {
        gpu.encodeBatch({ start: k * B, count: B, mode: 0, dataLen: SUBSET, ...o });
        cpuCorrect += trainBatch(w, v, subset, order, k * B, B, o, s).correct;
    }
    const dw = maxRel(gpu.weights.subarray(0, NPARAM), w);
    const dv = maxRel(gpu.weights.subarray(NPARAM), v);
    let moved = 0;
    const w0 = initWeights(seed);
    for (let i = 0; i < NPARAM; i++) moved = Math.max(moved, Math.abs(w[i] - w0[i]));
    t.diagnostic(`after ${steps} steps of B=${B}: weights max |Δ| ${dw.err.toExponential(2)} (largest weight move ${moved.toExponential(2)}), ` +
        `momentum max |Δ| ${dv.err.toExponential(2)} (max |v| ${dv.scale.toExponential(2)}); correct ${gpu.stats.trainCorrect} vs ${cpuCorrect}`);
    // the documented agreement is "about 1e-7" (f32 vs f64 local sums); allow 1e-6 absolute on O(0.1-1) weights
    assert.ok(dw.err < 1e-6, `weights differ by ${dw.err}`);
    assert.ok(dv.rel < 1e-4, `momentum differs by ${dv.rel} relative`);
    assert.ok(moved > 1e-3, 'the steps actually moved the weights');
    assert.equal(gpu.stats.trainCorrect, cpuCorrect);
});

test('replica evaluation (mode 1, sequential, index clamp) and Stats match evaluate()', (t) => {
    const w = parseWeights(readFileSync(fileURLToPath(new URL('../data/mnist/weights.bin', import.meta.url))));
    const n = 40;
    const data = { images: testData.images.subarray(0, n * IMG), labels: testData.labels.subarray(0, n), count: n };
    const B = 16;
    const gpu = new Replica(B, data, null);
    gpu.weights.set(w);
    // batches of 16, 16, 8 as planSteps() would produce, plus a batch past the end (clamped)
    for (let start = 0; start < n; start += B) gpu.encodeBatch({ start, count: Math.min(B, n - start), mode: 1, dataLen: n });
    const ref = evaluate(w, data, createScratch());
    t.diagnostic(`${n} test images: replica correct ${gpu.stats.testCorrect}, loss ${(gpu.stats.testLoss / n).toFixed(5)}; ` +
        `evaluate() correct ${ref.correct}, loss ${ref.loss.toFixed(5)}`);
    assert.equal(gpu.stats.testCount, n);
    assert.equal(gpu.stats.testCorrect, ref.correct);
    assert.ok(Math.abs(gpu.stats.testLoss / n - ref.loss) < 1e-5);
    assert.equal(gpu.stats.trainCount, 0);

    // imageIndex clamps to dataLen - 1: a batch starting past the end reads the last image
    gpu.encodeBatch({ start: n + 3, count: 2, mode: 1, dataLen: n });
    const s = createScratch();
    loadImage(s, data.images, n - 1);
    assert.deepEqual(gpu.work.subarray(gpu.L.X, gpu.L.X + IMG), s.x);
    assert.deepEqual(gpu.work.subarray(gpu.L.X + IMG, gpu.L.X + 2 * IMG), s.x);

    // mode 2 (Recognize: CPU-written input, no labels): probabilities only, loss uses label = none
    const g1 = new Replica(1, data, null);
    g1.weights.set(w);
    g1.work.set(s.x, g1.L.X);
    g1.encodeBatch({ start: 0, count: 1, mode: 2, dataLen: 1 });
    forward(w, s);
    softmaxLoss(s, 0);
    const dp = maxRel(g1.work.subarray(g1.L.PROB, g1.L.PROB + 10), s.prob);
    assert.ok(dp.err < 1e-6);
    assert.equal(g1.stats.testCount + g1.stats.trainCount, 0, 'mode 2 records no stats');
});

test('max-pool ties: replica and trainer both route the gradient to the first maximum', (t) => {
    // W1 = 0, b1 = 0.5: every conv1 output is exactly 0.5, so every 2×2 window of pool1 is a 4-way
    // tie. Gradients then depend on which element wins (dW1 = Σ dz1 · x), so a different tie rule
    // on either side shows up as a W1 gradient mismatch.
    const w = initWeights(11);
    w.fill(0, OFF.W1, OFF.W1 + 200);
    w.fill(0.5, OFF.B1, OFF.B1 + 8);
    const data = { images: testData.images.subarray(0, IMG), labels: testData.labels.subarray(0, 1), count: 1 };
    const gpu = new Replica(1, data, null);
    gpu.weights.set(w);
    gpu.encodeBatch({ start: 0, count: 1, mode: 1, dataLen: 1 });
    gpu.encodeBatch({ start: 0, count: 1, mode: 0, dataLen: 1, lr: 0, momentum: 0, weightDecay: 0, augSeed: 0, maxShift: 0 },
        { backwardOnly: true });
    const s = createScratch();
    loadImage(s, data.images, 0);
    forward(w, s);
    softmaxLoss(s, data.labels[0]);
    backward(w, s);
    const r = maxRel(gpu.grads.subarray(OFF.W1, OFF.W1 + 200), s.g.subarray(OFF.W1, OFF.W1 + 200));
    t.diagnostic(`all-tie pool1: dW1 max rel difference ${r.rel.toExponential(2)} (max |dW1| ${r.scale.toExponential(2)})`);
    assert.ok(r.scale > 0);
    assert.ok(r.rel < 1e-5);
});
