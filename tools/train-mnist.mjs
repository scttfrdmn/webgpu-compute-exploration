#!/usr/bin/env node
// Trains the shared MNIST CNN (js/examples/mnist-model.js) on the CPU and writes data/mnist/weights.bin,
// the cached weights that example 22 (MNIST: Recognize) loads by default.
//
// The math is the same as the GPU kernels in mnist-model.js: seeded He init, the per-epoch
// Fisher-Yates shuffle, inputs scaled to byte / 255 and translated by the hashed per-sample shift
// (up to ±2 px), ReLU, 2×2 max-pool with the first maximum winning ties, log-sum-exp softmax
// cross-entropy, per-sample gradients summed in sample order, the batch mean plus L2 weight decay
// clamped to ±GRAD_CLIP, and SGD with momentum (v = μv + g, w = w − ηv, η = η0 · decay^epoch),
// with weights and momentum stored as f32. Local sums use f64 here and f32 on the GPU, so the two
// agree to about 1e-7 rather than bit for bit. Deterministic for a given seed.
//
// Usage: node tools/train-mnist.mjs [--seed 1] [--batch 64] [--lr 0.04] [--lrDecay 0.85]
//          [--momentum 0.9] [--weightDecay 0.0005] [--maxShift 2] [--epochs 15]
//          [--out data/mnist/weights.bin]

import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import {
    IMG_PIXELS, NUM_CLASSES, NUM_PARAMS, OFF, GRAD_CLIP, DEFAULTS, DATASETS,
    initWeights, epochOrder, serializeWeights, augmentSeed, sampleShift,
} from '../js/examples/mnist-model.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA_DIR = join(ROOT, 'data', 'mnist');
const LOGIT_CLAMP = 1e4;

export function loadDatasetNode(which, dir = DATA_DIR) {
    const info = DATASETS[which];
    const read = (name) => {
        const bytes = readFileSync(join(dir, name));
        return bytes[0] === 0x1f && bytes[1] === 0x8b ? new Uint8Array(gunzipSync(bytes)) : new Uint8Array(bytes);
    };
    const images = read(info.images);
    const labels = read(info.labels);
    if (images.length !== info.count * IMG_PIXELS || labels.length !== info.count) {
        throw new Error(`unexpected ${which} data size`);
    }
    return { images, labels, count: info.count };
}

// Per-sample activations and gradients (f32 storage, like the GPU work buffer)
export function createScratch() {
    const f = (n) => new Float32Array(n);
    return {
        x: f(784), a1: f(4608), p1: f(1152), a2: f(1024), p2: f(256), logit: f(10), prob: f(10), dlog: f(10),
        dp2: f(256), dz2: f(1024), dp1: f(1152), dz1: f(4608), g: f(NUM_PARAMS), acc: new Float64Array(NUM_PARAMS),
    };
}

// First maximum of a 2×2 window wins (same scan order as argmax4 in the WGSL)
function argmax4(a, base, stride) {
    let best = a[base];
    let bestK = 0;
    for (let k = 1; k < 4; k++) {
        const v = a[base + (k >> 1) * stride + (k & 1)];
        if (v > best) { best = v; bestK = k; }
    }
    return bestK;
}

function pool(a, size, channels, out) {
    const h = size / 2;
    for (let c = 0; c < channels; c++) {
        for (let py = 0; py < h; py++) {
            for (let px = 0; px < h; px++) {
                const base = c * size * size + py * 2 * size + px * 2;
                const k = argmax4(a, base, size);
                out[c * h * h + py * h + px] = a[base + (k >> 1) * size + (k & 1)];
            }
        }
    }
}

function poolBack(a, size, channels, dp, out) {
    const h = size / 2;
    for (let c = 0; c < channels; c++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const i = c * size * size + y * size + x;
                const k = argmax4(a, c * size * size + (y & ~1) * size + (x & ~1), size);
                out[i] = (k === (y & 1) * 2 + (x & 1) && a[i] > 0) ? dp[c * h * h + (y >> 1) * h + (x >> 1)] : 0;
            }
        }
    }
}

// Forward pass of one image (x already in s.x); returns the predicted class
export function forward(w, s) {
    const { x, a1, p1, a2, p2, logit } = s;
    for (let c = 0; c < 8; c++) {
        for (let oy = 0; oy < 24; oy++) {
            for (let ox = 0; ox < 24; ox++) {
                let acc = w[OFF.B1 + c];
                for (let ky = 0; ky < 5; ky++) {
                    for (let kx = 0; kx < 5; kx++) {
                        acc += w[OFF.W1 + c * 25 + ky * 5 + kx] * x[(oy + ky) * 28 + ox + kx];
                    }
                }
                a1[c * 576 + oy * 24 + ox] = acc > 0 ? acc : 0;
            }
        }
    }
    pool(a1, 24, 8, p1);
    for (let co = 0; co < 16; co++) {
        for (let oy = 0; oy < 8; oy++) {
            for (let ox = 0; ox < 8; ox++) {
                let acc = w[OFF.B2 + co];
                for (let ci = 0; ci < 8; ci++) {
                    const wb = OFF.W2 + (co * 8 + ci) * 25;
                    const ib = ci * 144 + oy * 12 + ox;
                    for (let ky = 0; ky < 5; ky++) {
                        for (let kx = 0; kx < 5; kx++) {
                            acc += w[wb + ky * 5 + kx] * p1[ib + ky * 12 + kx];
                        }
                    }
                }
                a2[co * 64 + oy * 8 + ox] = acc > 0 ? acc : 0;
            }
        }
    }
    pool(a2, 8, 16, p2);
    let best = 0;
    for (let k = 0; k < NUM_CLASSES; k++) {
        let acc = w[OFF.B3 + k];
        for (let j = 0; j < 256; j++) acc += w[OFF.W3 + k * 256 + j] * p2[j];
        logit[k] = acc;
        if (logit[k] > logit[best]) best = k;
    }
    return best;
}

// Softmax cross-entropy via log-sum-exp; fills prob and dlog = prob − onehot; returns the loss
export function softmaxLoss(s, label) {
    const { logit, prob, dlog } = s;
    let m = -LOGIT_CLAMP;
    for (let k = 0; k < NUM_CLASSES; k++) m = Math.max(m, Math.min(Math.max(logit[k], -LOGIT_CLAMP), LOGIT_CLAMP));
    let sum = 0;
    for (let k = 0; k < NUM_CLASSES; k++) sum += Math.exp(Math.min(Math.max(logit[k], -LOGIT_CLAMP), LOGIT_CLAMP) - m);
    for (let k = 0; k < NUM_CLASSES; k++) {
        prob[k] = Math.exp(Math.min(Math.max(logit[k], -LOGIT_CLAMP), LOGIT_CLAMP) - m) / sum;
        dlog[k] = prob[k] - (k === label ? 1 : 0);
    }
    const l = Math.min(Math.max(logit[label], -LOGIT_CLAMP), LOGIT_CLAMP);
    return Math.max(m + Math.log(sum) - l, 0);
}

// Backward pass of one image after forward + softmaxLoss; writes the sample's gradient to s.g
export function backward(w, s) {
    const { x, a1, p1, a2, p2, dlog, dp2, dz2, dp1, dz1, g } = s;
    for (let j = 0; j < 256; j++) {
        let acc = 0;
        for (let k = 0; k < NUM_CLASSES; k++) acc += w[OFF.W3 + k * 256 + j] * dlog[k];
        dp2[j] = acc;
    }
    poolBack(a2, 8, 16, dp2, dz2);
    for (let ci = 0; ci < 8; ci++) {
        for (let y = 0; y < 12; y++) {
            for (let x0 = 0; x0 < 12; x0++) {
                let acc = 0;
                for (let co = 0; co < 16; co++) {
                    for (let ky = 0; ky < 5; ky++) {
                        const oy = y - ky;
                        if (oy < 0 || oy >= 8) continue;
                        for (let kx = 0; kx < 5; kx++) {
                            const ox = x0 - kx;
                            if (ox < 0 || ox >= 8) continue;
                            acc += w[OFF.W2 + ((co * 8 + ci) * 5 + ky) * 5 + kx] * dz2[co * 64 + oy * 8 + ox];
                        }
                    }
                }
                dp1[ci * 144 + y * 12 + x0] = acc;
            }
        }
    }
    poolBack(a1, 24, 8, dp1, dz1);

    for (let c = 0; c < 8; c++) {
        for (let k = 0; k < 25; k++) {
            const ky = (k / 5) | 0;
            const kx = k % 5;
            let acc = 0;
            for (let oy = 0; oy < 24; oy++) {
                for (let ox = 0; ox < 24; ox++) acc += dz1[c * 576 + oy * 24 + ox] * x[(oy + ky) * 28 + ox + kx];
            }
            g[OFF.W1 + c * 25 + k] = acc;
        }
        let bsum = 0;
        for (let i = 0; i < 576; i++) bsum += dz1[c * 576 + i];
        g[OFF.B1 + c] = bsum;
    }
    for (let co = 0; co < 16; co++) {
        for (let ci = 0; ci < 8; ci++) {
            for (let k = 0; k < 25; k++) {
                const ky = (k / 5) | 0;
                const kx = k % 5;
                let acc = 0;
                for (let oy = 0; oy < 8; oy++) {
                    for (let ox = 0; ox < 8; ox++) {
                        acc += dz2[co * 64 + oy * 8 + ox] * p1[ci * 144 + (oy + ky) * 12 + ox + kx];
                    }
                }
                g[OFF.W2 + (co * 8 + ci) * 25 + k] = acc;
            }
        }
        let bsum = 0;
        for (let i = 0; i < 64; i++) bsum += dz2[co * 64 + i];
        g[OFF.B2 + co] = bsum;
    }
    for (let k = 0; k < NUM_CLASSES; k++) {
        for (let j = 0; j < 256; j++) g[OFF.W3 + k * 256 + j] = dlog[k] * p2[j];
        g[OFF.B3 + k] = dlog[k];
    }
}

// Loads image `index` into s.x as byte / 255, translated by (dx, dy) (zeros shifted in)
export function loadImage(s, images, index, dx = 0, dy = 0) {
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const sx = x - dx;
            const sy = y - dy;
            s.x[y * 28 + x] = (sx >= 0 && sy >= 0 && sx < 28 && sy < 28) ? images[index * IMG_PIXELS + sy * 28 + sx] / 255 : 0;
        }
    }
}

// One SGD-with-momentum step on the batch at positions [start, start + count) of `order`
// (w and v are Float32Array(NUM_PARAMS)); o = { lr, momentum, weightDecay, augSeed, maxShift }
export function trainBatch(w, v, data, order, start, count, o, s) {
    const lr32 = Math.fround(o.lr);
    const mu32 = Math.fround(o.momentum);
    const wd32 = Math.fround(o.weightDecay || 0);
    s.acc.fill(0);
    let loss = 0;
    let correct = 0;
    for (let b = 0; b < count; b++) {
        const index = order[start + b];
        const [dx, dy] = sampleShift(o.augSeed || 0, start + b, o.maxShift || 0);
        loadImage(s, data.images, index, dx, dy);
        const pred = forward(w, s);
        const label = data.labels[index];
        loss += softmaxLoss(s, label);
        if (pred === label) correct++;
        backward(w, s);
        for (let p = 0; p < NUM_PARAMS; p++) s.acc[p] += s.g[p];
    }
    for (let p = 0; p < NUM_PARAMS; p++) {
        const mean = Math.fround(s.acc[p] / count);
        const g = Math.min(Math.max(Math.fround(mean + wd32 * w[p]), -GRAD_CLIP), GRAD_CLIP);
        v[p] = mu32 * v[p] + g;
        w[p] = w[p] - lr32 * v[p];
    }
    return { loss, correct };
}

export function evaluate(w, data, s) {
    let correct = 0;
    let loss = 0;
    for (let i = 0; i < data.count; i++) {
        loadImage(s, data.images, i);
        const pred = forward(w, s);
        loss += softmaxLoss(s, data.labels[i]);
        if (pred === data.labels[i]) correct++;
    }
    return { accuracy: correct / data.count, loss: loss / data.count, correct };
}

// Runs `epochs` epochs (or stops after maxBatches mini-batches)
export function epochLearningRate(lr, lrDecay, epoch) {
    return lr * Math.pow(lrDecay, epoch);
}

export function train({ seed, batch, lr, momentum, epochs, lrDecay, weightDecay, maxShift,
                        train: data, test, log = () => {}, maxBatches = Infinity }) {
    const w = initWeights(seed);
    const v = new Float32Array(NUM_PARAMS);
    const s = createScratch();
    let batches = 0;
    for (let epoch = 0; epoch < epochs && batches < maxBatches; epoch++) {
        const t0 = performance.now();
        const order = epochOrder(data.count, seed, epoch);
        let loss = 0;
        let correct = 0;
        for (let pos = 0; pos < data.count && batches < maxBatches; pos += batch) {
            const o = { lr: epochLearningRate(lr, lrDecay, epoch), momentum, weightDecay, augSeed: augmentSeed(seed, epoch), maxShift };
            const r = trainBatch(w, v, data, order, pos, Math.min(batch, data.count - pos), o, s);
            loss += r.loss;
            correct += r.correct;
            batches++;
        }
        if (batches >= maxBatches && epoch === 0 && !test) break;
        const ev = test ? evaluate(w, test, s) : null;
        log(`epoch ${epoch + 1}/${epochs}: train loss ${(loss / data.count).toFixed(4)}, ` +
            `train acc ${(100 * correct / data.count).toFixed(2)}%` +
            (ev ? `, test loss ${ev.loss.toFixed(4)}, test acc ${(100 * ev.accuracy).toFixed(2)}% (${ev.correct}/${test.count})` : '') +
            `, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    }
    return { weights: w, momentum: v };
}

function parseArgs(argv) {
    const opts = { ...DEFAULTS, out: join(DATA_DIR, 'weights.bin') };
    for (let i = 0; i < argv.length; i += 2) {
        const key = argv[i].replace(/^--/, '');
        if (!(key in opts)) throw new Error(`unknown option ${argv[i]}`);
        opts[key] = key === 'out' ? argv[i + 1] : Number(argv[i + 1]);
    }
    return opts;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
    const opts = parseArgs(process.argv.slice(2));
    const trainData = loadDatasetNode('train');
    const testData = loadDatasetNode('test');
    console.log(`MNIST CPU trainer: ${trainData.count} train / ${testData.count} test images, seed ${opts.seed}, ` +
        `batch ${opts.batch}, lr ${opts.lr} × ${opts.lrDecay}^epoch, momentum ${opts.momentum}, ` +
        `weight decay ${opts.weightDecay}, shifts ±${opts.maxShift} px, ${opts.epochs} epochs`);
    const t0 = performance.now();
    const { weights } = train({ ...opts, train: trainData, test: testData, log: (m) => console.log(m) });
    const seconds = (performance.now() - t0) / 1000;
    writeFileSync(opts.out, serializeWeights(weights));
    console.log(`wrote ${opts.out} (${NUM_PARAMS} parameters) after ${seconds.toFixed(1)} s`);
}
