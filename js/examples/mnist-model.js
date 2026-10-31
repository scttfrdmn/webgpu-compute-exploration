// Shared small CNN for the MNIST examples (21 Train, 22 Recognize) and the Node trainer
// (tools/train-mnist.mjs): architecture constants, WGSL compute kernels, the GPU driver, seeded
// He initialization, the per-epoch shuffle, data loading and the weights file format.
//
// Network (all 'valid' convolutions, stride 1, inputs scaled to [0, 1] as byte / 255, no mean/std
// normalization):
//   input 28×28×1
//   conv1 5×5, 8 filters  -> 24×24×8  -> ReLU -> 2×2 max-pool -> 12×12×8
//   conv2 5×5, 16 filters -> 8×8×16   -> ReLU -> 2×2 max-pool -> 4×4×16 = 256
//   dense 256 -> 10 -> softmax, cross-entropy loss
// 5,994 parameters. Tensors are row-major; feature maps are channel-major ([c][y][x]), so the dense
// input index is j = c·16 + y·4 + x.
//
// Parameter vector (f32, also the order of the weights file):
//   offset    0  W1 [8][1][5][5]    200
//   offset  200  b1 [8]               8
//   offset  208  W2 [16][8][5][5]  3200
//   offset 3408  b2 [16]             16
//   offset 3424  W3 [10][256]      2560
//   offset 5984  b3 [10]             10
//   total 5994
//
// Training (identical on the GPU and in tools/train-mnist.mjs): He-normal init from a seeded
// mulberry32 + Box-Muller generator, biases 0; every epoch a Fisher-Yates shuffle (seeded by seed and
// epoch) on the CPU; mini-batches of B consecutive shuffled indices (the last one may be smaller);
// each training image translated by a random (dx, dy) ∈ [−2, 2]² (hash of seed, epoch and
// position, see sampleShift); mean softmax cross-entropy over the batch; SGD with momentum and
// L2 weight decay λ (applied to all parameters), learning rate η = η0 · decay^epoch:
//   g = clamp(mean over the batch of dL/dw + λ·w, ±GRAD_CLIP);  v = μ·v + g;  w = w − η·v
// Gradient accumulation is race-free without atomics: one invocation per (sample, parameter) writes
// that sample's gradient to its own slot of a [B][5994] buffer, then one invocation per parameter
// sums its B slots in sample order and applies the update.
//
// Weights file ("MNW1", little-endian, 24,108 bytes):
//   offset   0  char[4]  magic "MNW1"
//   offset   4  u32      version = 1
//   offset   8  u32      tensor count = 6
//   offset  12  6 × { u32 rank; u32 dims[4] (unused dims = 0) }   (20 bytes each)
//   offset 132  f32[5994] W1, b1, W2, b2, W3, b3 in the order and shapes above
// Shapes: W1 [8,1,5,5], b1 [8], W2 [16,8,5,5], b2 [16], W3 [10,256], b3 [10].

export const IMG_SIZE = 28;
export const IMG_PIXELS = 784;
export const NUM_CLASSES = 10;
export const NUM_PARAMS = 5994;
export const TENSORS = [
    { name: 'W1', shape: [8, 1, 5, 5], offset: 0 },
    { name: 'b1', shape: [8], offset: 200 },
    { name: 'W2', shape: [16, 8, 5, 5], offset: 208 },
    { name: 'b2', shape: [16], offset: 3408 },
    { name: 'W3', shape: [10, 256], offset: 3424 },
    { name: 'b3', shape: [10], offset: 5984 },
];
export const OFF = { W1: 0, B1: 200, W2: 208, B2: 3408, W3: 3424, B3: 5984 };
export const GRAD_CLIP = 5.0;
export const MAX_BATCH = 128;
// Defaults shared by the browser trainer and tools/train-mnist.mjs, so a browser run with the default
// settings and seed repeats the run that produced data/mnist/weights.bin
export const DEFAULTS = {
    seed: 1, batch: 64, lr: 0.04, lrDecay: 0.85, momentum: 0.9, weightDecay: 0.0005, maxShift: 2, epochs: 15,
};

const WEIGHTS_MAGIC = 'MNW1';
const WEIGHTS_VERSION = 1;
const WEIGHTS_HEADER = 12 + TENSORS.length * 20;

// ---------------------------------------------------------------------------------------------
// Seeded randomness (shared by the browser and the Node trainer)

export function mulberry32(seed) {
    let a = seed >>> 0;
    return function next() {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// He-normal initialization: w ~ N(0, 2 / fanIn), biases 0. Returns a Float32Array(NUM_PARAMS).
export function initWeights(seed) {
    const rng = mulberry32(seed);
    const gauss = () => {
        const u1 = Math.max(rng(), 1e-12);
        const u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    const w = new Float32Array(NUM_PARAMS);
    const fill = (offset, count, fanIn) => {
        const std = Math.sqrt(2 / fanIn);
        for (let i = 0; i < count; i++) w[offset + i] = gauss() * std;
    };
    fill(OFF.W1, 200, 25);
    fill(OFF.W2, 3200, 200);
    fill(OFF.W3, 2560, 256);
    return w;
}

// Fisher-Yates shuffle of 0..n-1 for one epoch, seeded by (seed, epoch)
export function epochOrder(n, seed, epoch) {
    const rng = mulberry32((seed + Math.imul(epoch + 1, 0x9E3779B9)) >>> 0);
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    return order;
}

// Data augmentation: each training sample is translated by (dx, dy) ∈ [−maxShift, maxShift]²
// (pixels shifted in from outside are 0). The shift is a PCG hash of a per-epoch seed and the
// sample's position in the epoch, computed identically here and in cs_gather.
export function pcgHash(v) {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

export function augmentSeed(seed, epoch) {
    return pcgHash((Math.imul(seed, 0x9E3779B9) + Math.imul(epoch + 1, 0x85EBCA6B)) >>> 0);
}

export function sampleShift(augSeed, position, maxShift) {
    const h = pcgHash((augSeed ^ pcgHash(position)) >>> 0);
    const span = 2 * maxShift + 1;
    return [(h % span) - maxShift, ((h >>> 8) % span) - maxShift];
}

// ---------------------------------------------------------------------------------------------
// Weights file

export function serializeWeights(weights) {
    if (weights.length !== NUM_PARAMS) throw new Error(`expected ${NUM_PARAMS} weights, got ${weights.length}`);
    const buffer = new ArrayBuffer(WEIGHTS_HEADER + NUM_PARAMS * 4);
    const view = new DataView(buffer);
    for (let i = 0; i < 4; i++) view.setUint8(i, WEIGHTS_MAGIC.charCodeAt(i));
    view.setUint32(4, WEIGHTS_VERSION, true);
    view.setUint32(8, TENSORS.length, true);
    TENSORS.forEach((t, k) => {
        const o = 12 + k * 20;
        view.setUint32(o, t.shape.length, true);
        for (let d = 0; d < 4; d++) view.setUint32(o + 4 + d * 4, t.shape[d] || 0, true);
    });
    for (let i = 0; i < NUM_PARAMS; i++) view.setFloat32(WEIGHTS_HEADER + i * 4, weights[i], true);
    return new Uint8Array(buffer);
}

// Parses and checks a weights file; returns a Float32Array(NUM_PARAMS) or throws
export function parseWeights(bytes) {
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    if (u8.length !== WEIGHTS_HEADER + NUM_PARAMS * 4) {
        throw new Error(`weights file has ${u8.length} bytes, expected ${WEIGHTS_HEADER + NUM_PARAMS * 4}`);
    }
    const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
    if (magic !== WEIGHTS_MAGIC) throw new Error(`bad weights magic "${magic}"`);
    const version = view.getUint32(4, true);
    if (version !== WEIGHTS_VERSION) throw new Error(`unsupported weights version ${version}`);
    if (view.getUint32(8, true) !== TENSORS.length) throw new Error('unexpected tensor count');
    TENSORS.forEach((t, k) => {
        const o = 12 + k * 20;
        const rank = view.getUint32(o, true);
        const dims = [0, 1, 2, 3].map(d => view.getUint32(o + 4 + d * 4, true));
        const ok = rank === t.shape.length && dims.every((v, d) => v === (t.shape[d] || 0));
        if (!ok) throw new Error(`tensor ${t.name} has shape [${dims.slice(0, rank)}], expected [${t.shape}]`);
    });
    const w = new Float32Array(NUM_PARAMS);
    for (let i = 0; i < NUM_PARAMS; i++) {
        const v = view.getFloat32(WEIGHTS_HEADER + i * 4, true);
        if (!Number.isFinite(v)) throw new Error(`weight ${i} is not finite`);
        w[i] = v;
    }
    return w;
}

// ---------------------------------------------------------------------------------------------
// Data loading (browser). The files are gzip-compressed raw bytes; some static servers decode gzip
// transparently, so the gzip magic bytes decide whether to decompress.

export const DATA_URL = new URL('../../data/mnist/', import.meta.url);
export const DATASETS = {
    train: { images: 'train-images.u8.gz', labels: 'train-labels.u8.gz', count: 10000 },
    test: { images: 'test-images.u8.gz', labels: 'test-labels.u8.gz', count: 2000 },
};

export async function fetchBytes(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
        return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    return bytes;
}

const datasetCache = new Map();

// { images: Uint8Array(count × 784), labels: Uint8Array(count), count }; cached per page
export function loadDataset(which) {
    if (!datasetCache.has(which)) {
        const info = DATASETS[which];
        const promise = Promise.all([
            fetchBytes(new URL(info.images, DATA_URL)),
            fetchBytes(new URL(info.labels, DATA_URL)),
        ]).then(([images, labels]) => {
            if (images.length !== info.count * IMG_PIXELS || labels.length !== info.count) {
                throw new Error(`MNIST ${which} data has unexpected size (${images.length} / ${labels.length} bytes)`);
            }
            return { images, labels, count: info.count };
        });
        promise.catch(() => datasetCache.delete(which));
        datasetCache.set(which, promise);
    }
    return datasetCache.get(which);
}

export async function loadCachedWeights() {
    const response = await fetch(new URL('weights.bin', DATA_URL));
    if (!response.ok) throw new Error(`weights.bin: HTTP ${response.status}`);
    return parseWeights(new Uint8Array(await response.arrayBuffer()));
}

// Hand-off from Train to Recognize: Train publishes a copy of its weights, Recognize subscribes
export const sharedWeights = {
    trained: null,               // { weights: Float32Array, label: string } or null
    listeners: new Set(),
    publish(weights, label) {
        this.trained = { weights: Float32Array.from(weights), label };
        this.listeners.forEach(fn => fn(this.trained));
    },
    subscribe(fn) {
        this.listeners.add(fn);
        return () => this.listeners.delete(fn);
    },
};

// ---------------------------------------------------------------------------------------------
// WGSL compute kernels

// Uniform layout (struct Params, 32 bytes; one slot per mini-batch at a dynamic offset):
//   offset  0  start     u32  first position in the order (mode 0) or first image (mode 1) -> Uint32 [0]
//   offset  4  count     u32  samples in this batch (1..MAX_BATCH)                         -> Uint32 [1]
//   offset  8  mode      u32  0 = shuffled training batch, 1 = sequential (evaluation),
//                             2 = input already written to the work buffer by the CPU      -> Uint32 [2]
//   offset 12  dataLen   u32  images in the bound dataset (index clamp)                    -> Uint32 [3]
//   offset 16  lr        f32  learning rate η (this epoch's)                               -> Float32 [4]
//   offset 20  momentum  f32  momentum μ                                                   -> Float32 [5]
//   offset 24  weightDecay f32 L2 weight decay λ                                           -> Float32 [6]
//   offset 28  augSeed   u32  this epoch's augmentation seed (augmentSeed)                 -> Uint32 [7]
//   offset 32  maxShift  u32  largest random translation in pixels (0 = off)              -> Uint32 [8]
//   offset 36  pad0, pad1, pad2 u32 (struct size rounded up to 16)                         -> Uint32 [9..11]
export const PARAMS_SIZE = 48;

// Stats layout (struct Stats, 32 bytes, read back by Train): sums since the last clear
//   offset  0  trainLoss f32 · 4 trainCorrect u32 · 8 trainCount u32 · 12 pad
//   offset 16  testLoss  f32 · 20 testCorrect  u32 · 24 testCount  u32 · 28 pad
export const STATS_SIZE = 32;

// Work buffer regions (f32 offsets, one block of MAX_BATCH samples each); see workLayout()
export function workLayout(maxBatch) {
    const sizes = [
        ['X', 784], ['A1', 4608], ['P1', 1152], ['A2', 1024], ['P2', 256], ['LOGIT', 10], ['PROB', 10],
        ['DLOG', 10], ['LOSS', 1], ['CORR', 1], ['DP2', 256], ['DZ2', 1024], ['DP1', 1152], ['DZ1', 4608],
    ];
    const layout = {};
    let offset = 0;
    for (const [name, size] of sizes) {
        layout[name] = offset;
        offset += size * maxBatch;
    }
    layout.MAPMAX = offset;     // 24 per-map maxima of sample 0 (8 conv1 + 16 conv2), for display
    layout.total = offset + 24;
    return layout;
}

export function buildComputeShader(maxBatch) {
    const L = workLayout(maxBatch);
    return /* wgsl */ `
struct Params {
    start: u32,
    count: u32,
    mode: u32,
    dataLen: u32,
    lr: f32,
    momentum: f32,
    weightDecay: f32,
    augSeed: u32,
    maxShift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Stats {
    trainLoss: f32,
    trainCorrect: u32,
    trainCount: u32,
    pad0: u32,
    testLoss: f32,
    testCorrect: u32,
    testCount: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> images: array<u32>;      // uint8 pixels, 4 per u32
@group(0) @binding(2) var<storage, read> labels: array<u32>;      // uint8 labels, 4 per u32
@group(0) @binding(3) var<storage, read> order: array<u32>;       // shuffled sample indices
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;  // [0, NPARAM) weights, [NPARAM, 2·NPARAM) momentum
@group(0) @binding(5) var<storage, read_write> work: array<f32>;  // activations and their gradients
@group(0) @binding(6) var<storage, read_write> grads: array<f32>; // per-sample gradients [MAX_BATCH][NPARAM]
@group(0) @binding(7) var<storage, read_write> stats: Stats;

const MAX_BATCH: u32 = ${maxBatch}u;
const IMG: u32 = 784u;
const N1: u32 = 4608u;     // conv1 output 8 × 24 × 24
const NP1: u32 = 1152u;    // pool1 output 8 × 12 × 12
const N2: u32 = 1024u;     // conv2 output 16 × 8 × 8
const NP2: u32 = 256u;     // pool2 output 16 × 4 × 4
const NCLS: u32 = 10u;
const NPARAM: u32 = 5994u;
const OFF_W1: u32 = 0u;
const OFF_B1: u32 = 200u;
const OFF_W2: u32 = 208u;
const OFF_B2: u32 = 3408u;
const OFF_W3: u32 = 3424u;
const OFF_B3: u32 = 5984u;
const GRAD_CLIP: f32 = ${GRAD_CLIP.toFixed(1)};
const LOGIT_CLAMP: f32 = 1.0e4;

// Work buffer regions: sample b of region R starts at R + b · (size of one sample)
const W_X: u32 = ${L.X}u;
const W_A1: u32 = ${L.A1}u;
const W_P1: u32 = ${L.P1}u;
const W_A2: u32 = ${L.A2}u;
const W_P2: u32 = ${L.P2}u;
const W_LOGIT: u32 = ${L.LOGIT}u;
const W_PROB: u32 = ${L.PROB}u;
const W_DLOG: u32 = ${L.DLOG}u;
const W_LOSS: u32 = ${L.LOSS}u;
const W_CORR: u32 = ${L.CORR}u;
const W_DP2: u32 = ${L.DP2}u;
const W_DZ2: u32 = ${L.DZ2}u;
const W_DP1: u32 = ${L.DP1}u;
const W_DZ1: u32 = ${L.DZ1}u;
const W_MAPMAX: u32 = ${L.MAPMAX}u;

// Dataset index of batch sample b: shuffled order (mode 0) or consecutive (mode 1)
fn imageIndex(b: u32) -> u32 {
    var i = params.start + b;
    if (params.mode == 0u) {
        i = order[params.start + b];
    }
    return min(i, params.dataLen - 1u);
}

// Byte i of a packed uint8 array (little-endian: byte 0 is the lowest 8 bits of word 0)
fn unpackByte(word: u32, byteIndex: u32) -> u32 {
    return (word >> ((byteIndex & 3u) * 8u)) & 0xffu;
}

fn labelOf(b: u32) -> u32 {
    let i = imageIndex(b);
    return unpackByte(labels[i >> 2u], i);
}

// PCG hash (one round), the same as pcgHash() in JS
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Batch assembly: unpack the batch's images from the uploaded dataset into X, scaled to [0, 1].
// Training batches (mode 0) are translated by a per-sample random shift hashed from the epoch's
// seed and the sample's position in the epoch (sampleShift() in JS); pixels shifted in are 0.
// One invocation per (sample, pixel).
@compute @workgroup_size(64)
fn cs_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * IMG) { return; }
    let b = id / IMG;
    let i = id % IMG;
    var dx = 0;
    var dy = 0;
    if (params.mode == 0u && params.maxShift > 0u) {
        let h = pcg(params.augSeed ^ pcg(params.start + b));
        let span = 2u * params.maxShift + 1u;
        dx = i32(h % span) - i32(params.maxShift);
        dy = i32((h >> 8u) % span) - i32(params.maxShift);
    }
    let sx = i32(i % 28u) - dx;
    let sy = i32(i / 28u) - dy;
    var v = 0u;
    if (sx >= 0 && sy >= 0 && sx < 28 && sy < 28) {
        let byteIndex = imageIndex(b) * IMG + u32(sy) * 28u + u32(sx);
        v = unpackByte(images[byteIndex >> 2u], byteIndex);
    }
    work[W_X + b * IMG + i] = f32(v) / 255.0;
}

// conv1 + ReLU: a1[c][y][x] = max(0, b1[c] + Σ_ky,kx W1[c][ky][kx] · x[y+ky][x+kx]).
// One invocation per (sample, channel, output pixel).
@compute @workgroup_size(64)
fn cs_conv1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * N1) { return; }
    let b = id / N1;
    let r = id % N1;
    let c = r / 576u;
    let p = r % 576u;
    let oy = p / 24u;
    let ox = p % 24u;
    let xBase = W_X + b * IMG;
    var acc = weights[OFF_B1 + c];
    for (var ky = 0u; ky < 5u; ky++) {
        for (var kx = 0u; kx < 5u; kx++) {
            acc += weights[OFF_W1 + c * 25u + ky * 5u + kx] * work[xBase + (oy + ky) * 28u + ox + kx];
        }
    }
    work[W_A1 + b * N1 + r] = max(acc, 0.0);
}

// Index (0..3, as dy·2 + dx) of the largest of a 2×2 window; the first one wins ties. The
// backward passes recompute it instead of storing it, with the same scan order.
fn argmax4(base: u32, rowStride: u32) -> u32 {
    var best = work[base];
    var bestK = 0u;
    for (var k = 1u; k < 4u; k++) {
        let v = work[base + (k / 2u) * rowStride + (k % 2u)];
        if (v > best) {
            best = v;
            bestK = k;
        }
    }
    return bestK;
}

// 2×2 max-pool of a1 (24×24 -> 12×12 per channel). One invocation per (sample, pooled cell).
@compute @workgroup_size(64)
fn cs_pool1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * NP1) { return; }
    let b = id / NP1;
    let r = id % NP1;
    let c = r / 144u;
    let q = r % 144u;
    let base = W_A1 + b * N1 + c * 576u + (q / 12u) * 48u + (q % 12u) * 2u;
    let k = argmax4(base, 24u);
    work[W_P1 + b * NP1 + r] = work[base + (k / 2u) * 24u + (k % 2u)];
}

// conv2 + ReLU over the 8 pooled channels: a2[co][y][x] = max(0, b2[co] + Σ_ci,ky,kx W2·p1).
@compute @workgroup_size(64)
fn cs_conv2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * N2) { return; }
    let b = id / N2;
    let r = id % N2;
    let co = r / 64u;
    let p = r % 64u;
    let oy = p / 8u;
    let ox = p % 8u;
    let inBase = W_P1 + b * NP1;
    var acc = weights[OFF_B2 + co];
    for (var ci = 0u; ci < 8u; ci++) {
        for (var ky = 0u; ky < 5u; ky++) {
            for (var kx = 0u; kx < 5u; kx++) {
                acc += weights[OFF_W2 + ((co * 8u + ci) * 5u + ky) * 5u + kx]
                     * work[inBase + ci * 144u + (oy + ky) * 12u + ox + kx];
            }
        }
    }
    work[W_A2 + b * N2 + r] = max(acc, 0.0);
}

// 2×2 max-pool of a2 (8×8 -> 4×4 per channel)
@compute @workgroup_size(64)
fn cs_pool2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * NP2) { return; }
    let b = id / NP2;
    let r = id % NP2;
    let c = r / 16u;
    let q = r % 16u;
    let base = W_A2 + b * N2 + c * 64u + (q / 4u) * 16u + (q % 4u) * 2u;
    let k = argmax4(base, 8u);
    work[W_P2 + b * NP2 + r] = work[base + (k / 2u) * 8u + (k % 2u)];
}

// Dense layer: logit[k] = b3[k] + Σ_j W3[k][j] · p2[j]. One invocation per (sample, class).
@compute @workgroup_size(64)
fn cs_dense(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * NCLS) { return; }
    let b = id / NCLS;
    let k = id % NCLS;
    let inBase = W_P2 + b * NP2;
    var acc = weights[OFF_B3 + k];
    for (var j = 0u; j < NP2; j++) {
        acc += weights[OFF_W3 + k * NP2 + j] * work[inBase + j];
    }
    work[W_LOGIT + b * NCLS + k] = acc;
}

// Softmax with log-sum-exp (no overflow for any finite logits), per-sample loss
// L = log Σ exp(l) − l[label], correctness, and dL/dlogit = softmax − onehot(label).
// One invocation per sample. Mode 2 (CPU-written input) has no label: only the probabilities.
@compute @workgroup_size(64)
fn cs_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.count) { return; }
    let base = b * NCLS;
    var m = -LOGIT_CLAMP;
    var best = 0u;
    for (var k = 0u; k < NCLS; k++) {
        let l = clamp(work[W_LOGIT + base + k], -LOGIT_CLAMP, LOGIT_CLAMP);
        if (l > m) {
            m = l;
            best = k;
        }
    }
    var s = 0.0;
    for (var k = 0u; k < NCLS; k++) {
        s += exp(clamp(work[W_LOGIT + base + k], -LOGIT_CLAMP, LOGIT_CLAMP) - m);
    }
    var label = NCLS;
    if (params.mode != 2u) {
        label = labelOf(b);
    }
    var labelLogit = m;
    for (var k = 0u; k < NCLS; k++) {
        let l = clamp(work[W_LOGIT + base + k], -LOGIT_CLAMP, LOGIT_CLAMP);
        let prob = exp(l - m) / s;
        work[W_PROB + base + k] = prob;
        var onehot = 0.0;
        if (k == label) {
            onehot = 1.0;
            labelLogit = l;
        }
        work[W_DLOG + base + k] = prob - onehot;
    }
    work[W_LOSS + b] = max(m + log(s) - labelLogit, 0.0);
    work[W_CORR + b] = select(0.0, 1.0, best == label);
}

// Adds the batch's loss and correct count to Stats (train fields in mode 0, test fields in
// mode 1). A single invocation sums in sample order, so no atomics are needed.
@compute @workgroup_size(1)
fn cs_stats() {
    var loss = 0.0;
    var correct = 0u;
    for (var b = 0u; b < MAX_BATCH; b++) {
        if (b >= params.count) { break; }
        loss += work[W_LOSS + b];
        correct += u32(work[W_CORR + b]);
    }
    if (params.mode == 0u) {
        stats.trainLoss += loss;
        stats.trainCorrect += correct;
        stats.trainCount += params.count;
    } else {
        stats.testLoss += loss;
        stats.testCorrect += correct;
        stats.testCount += params.count;
    }
}

// ---- Backward pass (all gathers: every invocation writes only its own element) ----

// dL/dp2[j] = Σ_k W3[k][j] · dL/dlogit[k]
@compute @workgroup_size(64)
fn cs_dense_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * NP2) { return; }
    let b = id / NP2;
    let j = id % NP2;
    var acc = 0.0;
    for (var k = 0u; k < NCLS; k++) {
        acc += weights[OFF_W3 + k * NP2 + j] * work[W_DLOG + b * NCLS + k];
    }
    work[W_DP2 + b * NP2 + j] = acc;
}

// Max-pool backward + ReLU backward for a 2×2 window: an element receives its pooled cell's
// gradient only if it was that window's argmax, and only where the ReLU was active (a > 0).
fn poolBack(aBase: u32, size: u32, c: u32, y: u32, x: u32, dpBase: u32) -> f32 {
    let halfSize = size / 2u;
    let a = work[aBase + c * size * size + y * size + x];
    let winBase = aBase + c * size * size + (y & ~1u) * size + (x & ~1u);
    let k = argmax4(winBase, size);
    if (k != (y % 2u) * 2u + (x % 2u) || a <= 0.0) {
        return 0.0;
    }
    return work[dpBase + c * halfSize * halfSize + (y / 2u) * halfSize + x / 2u];
}

// dL/dz2 (pre-activation of conv2) from dL/dp2
@compute @workgroup_size(64)
fn cs_pool2_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * N2) { return; }
    let b = id / N2;
    let r = id % N2;
    let p = r % 64u;
    work[W_DZ2 + b * N2 + r] = poolBack(W_A2 + b * N2, 8u, r / 64u, p / 8u, p % 8u, W_DP2 + b * NP2);
}

// conv2 input gradient (a "full" correlation with the flipped kernel, written as a gather):
// dL/dp1[ci][y][x] = Σ_co,ky,kx W2[co][ci][ky][kx] · dL/dz2[co][y−ky][x−kx] over valid outputs
@compute @workgroup_size(64)
fn cs_conv2_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * NP1) { return; }
    let b = id / NP1;
    let r = id % NP1;
    let ci = r / 144u;
    let q = r % 144u;
    let y = i32(q / 12u);
    let x = i32(q % 12u);
    let dzBase = W_DZ2 + b * N2;
    var acc = 0.0;
    for (var co = 0u; co < 16u; co++) {
        for (var ky = 0; ky < 5; ky++) {
            let oy = y - ky;
            if (oy < 0 || oy >= 8) { continue; }
            for (var kx = 0; kx < 5; kx++) {
                let ox = x - kx;
                if (ox < 0 || ox >= 8) { continue; }
                acc += weights[OFF_W2 + ((co * 8u + ci) * 5u + u32(ky)) * 5u + u32(kx)]
                     * work[dzBase + co * 64u + u32(oy) * 8u + u32(ox)];
            }
        }
    }
    work[W_DP1 + b * NP1 + r] = acc;
}

// dL/dz1 (pre-activation of conv1) from dL/dp1
@compute @workgroup_size(64)
fn cs_pool1_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * N1) { return; }
    let b = id / N1;
    let r = id % N1;
    let p = r % 576u;
    work[W_DZ1 + b * N1 + r] = poolBack(W_A1 + b * N1, 24u, r / 576u, p / 24u, p % 24u, W_DP1 + b * NP1);
}

// Per-sample parameter gradients, one invocation per (sample, parameter), written to that
// sample's own row of grads (no two invocations write the same address):
//   dW1[c][ky][kx] = Σ_oy,ox dz1[c][oy][ox] · x[oy+ky][ox+kx]       db1[c] = Σ dz1[c]
//   dW2[co][ci][ky][kx] = Σ_oy,ox dz2[co][oy][ox] · p1[ci][oy+ky][ox+kx]   db2[co] = Σ dz2[co]
//   dW3[k][j] = dlogit[k] · p2[j]                                    db3[k] = dlogit[k]
@compute @workgroup_size(64)
fn cs_grads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.count * NPARAM) { return; }
    let b = id / NPARAM;
    let p = id % NPARAM;
    var g = 0.0;
    if (p < OFF_B1) {
        let c = p / 25u;
        let ky = (p % 25u) / 5u;
        let kx = p % 5u;
        let dzBase = W_DZ1 + b * N1 + c * 576u;
        let xBase = W_X + b * IMG;
        for (var oy = 0u; oy < 24u; oy++) {
            for (var ox = 0u; ox < 24u; ox++) {
                g += work[dzBase + oy * 24u + ox] * work[xBase + (oy + ky) * 28u + ox + kx];
            }
        }
    } else if (p < OFF_W2) {
        let dzBase = W_DZ1 + b * N1 + (p - OFF_B1) * 576u;
        for (var i = 0u; i < 576u; i++) {
            g += work[dzBase + i];
        }
    } else if (p < OFF_B2) {
        let q = p - OFF_W2;
        let co = q / 200u;
        let ci = (q / 25u) % 8u;
        let ky = (q % 25u) / 5u;
        let kx = q % 5u;
        let dzBase = W_DZ2 + b * N2 + co * 64u;
        let inBase = W_P1 + b * NP1 + ci * 144u;
        for (var oy = 0u; oy < 8u; oy++) {
            for (var ox = 0u; ox < 8u; ox++) {
                g += work[dzBase + oy * 8u + ox] * work[inBase + (oy + ky) * 12u + ox + kx];
            }
        }
    } else if (p < OFF_W3) {
        let dzBase = W_DZ2 + b * N2 + (p - OFF_B2) * 64u;
        for (var i = 0u; i < 64u; i++) {
            g += work[dzBase + i];
        }
    } else if (p < OFF_B3) {
        let q = p - OFF_W3;
        g = work[W_DLOG + b * NCLS + q / NP2] * work[W_P2 + b * NP2 + q % NP2];
    } else {
        g = work[W_DLOG + b * NCLS + (p - OFF_B3)];
    }
    grads[b * NPARAM + p] = g;
}

// Reduction + SGD with momentum, one invocation per parameter: sum the per-sample gradients in
// sample order, take the batch mean, add weight decay λ·w, clamp (guards against a divergent
// step), then  v = μ·v + g,  w = w − η·v
@compute @workgroup_size(64)
fn cs_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= NPARAM) { return; }
    var g = 0.0;
    for (var b = 0u; b < MAX_BATCH; b++) {
        if (b >= params.count) { break; }
        g += grads[b * NPARAM + p];
    }
    g = clamp(g / f32(params.count) + params.weightDecay * weights[p], -GRAD_CLIP, GRAD_CLIP);
    let v = params.momentum * weights[NPARAM + p] + g;
    weights[NPARAM + p] = v;
    weights[p] = weights[p] - params.lr * v;
}

// Display helper: the maximum of each activation map of sample 0 (8 conv1 maps of 576 values,
// 16 conv2 maps of 64), so the renderer can normalize each map. 24 invocations.
@compute @workgroup_size(32)
fn cs_mapmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= 24u) { return; }
    var base = W_A1 + m * 576u;
    var n = 576u;
    if (m >= 8u) {
        base = W_A2 + (m - 8u) * 64u;
        n = 64u;
    }
    var best = 0.0;
    for (var i = 0u; i < 576u; i++) {
        if (i >= n) { break; }
        best = max(best, work[base + i]);
    }
    work[W_MAPMAX + m] = best;
}
`;
}

// ---------------------------------------------------------------------------------------------
// GPU driver

const FORWARD_KERNELS = [
    ['cs_conv1', 4608], ['cs_pool1', 1152], ['cs_conv2', 1024], ['cs_pool2', 256], ['cs_dense', 10],
];
const BACKWARD_KERNELS = [
    ['cs_dense_back', 256], ['cs_pool2_back', 1024], ['cs_conv2_back', 1152], ['cs_pool1_back', 4608],
    ['cs_grads', NUM_PARAMS],
];
const WORKGROUP = 64;

export class MNISTGPU {
    // maxBatch: largest batch (1 for inference only); maxSlots: mini-batches encoded per submit
    constructor(device, maxBatch, maxSlots) {
        this.device = device;
        this.maxBatch = maxBatch;
        this.maxSlots = maxSlots;
        this.layout = workLayout(maxBatch);

        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.paramsData = new ArrayBuffer(this.paramsStride * maxSlots);
        this.paramsU32 = new Uint32Array(this.paramsData);
        this.paramsF32 = new Float32Array(this.paramsData);

        const S = GPUBufferUsage.STORAGE;
        const D = GPUBufferUsage.COPY_DST;
        const C = GPUBufferUsage.COPY_SRC;
        this.paramsBuffer = device.createBuffer({ size: this.paramsStride * maxSlots, usage: GPUBufferUsage.UNIFORM | D });
        this.weightsBuffer = device.createBuffer({ size: NUM_PARAMS * 2 * 4, usage: S | D | C });
        this.workBuffer = device.createBuffer({ size: this.layout.total * 4, usage: S | D | C });
        this.gradsBuffer = device.createBuffer({ size: maxBatch * NUM_PARAMS * 4, usage: S });
        this.statsBuffer = device.createBuffer({ size: STATS_SIZE, usage: S | D | C });
        this.dummyBuffer = device.createBuffer({ size: 16, usage: S });

        const storage = (type) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE } },
                { binding: 1, ...storage('read-only-storage') },
                { binding: 2, ...storage('read-only-storage') },
                { binding: 3, ...storage('read-only-storage') },
                { binding: 4, ...storage('storage') },
                { binding: 5, ...storage('storage') },
                { binding: 6, ...storage('storage') },
                { binding: 7, ...storage('storage') },
            ]
        });
        const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
        const module = device.createShaderModule({ code: buildComputeShader(maxBatch) });
        const names = ['cs_gather', 'cs_softmax', 'cs_stats', 'cs_update', 'cs_mapmax',
            ...FORWARD_KERNELS.map(k => k[0]), ...BACKWARD_KERNELS.map(k => k[0])];
        this.pipelines = {};
        for (const name of names) {
            this.pipelines[name] = device.createComputePipeline({ layout, compute: { module, entryPoint: name } });
        }
        this.inferenceBindGroup = this.createBindGroup(null);
    }

    // Bind group for a dataset { imagesBuffer, labelsBuffer, orderBuffer } (null: no dataset, mode 2)
    createBindGroup(data) {
        const entries = [
            { binding: 0, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } },
            { binding: 1, resource: { buffer: data ? data.imagesBuffer : this.dummyBuffer } },
            { binding: 2, resource: { buffer: data ? data.labelsBuffer : this.dummyBuffer } },
            { binding: 3, resource: { buffer: data && data.orderBuffer ? data.orderBuffer : this.dummyBuffer } },
            { binding: 4, resource: { buffer: this.weightsBuffer } },
            { binding: 5, resource: { buffer: this.workBuffer } },
            { binding: 6, resource: { buffer: this.gradsBuffer } },
            { binding: 7, resource: { buffer: this.statsBuffer } },
        ];
        return this.device.createBindGroup({ layout: this.bindGroupLayout, entries });
    }

    // Uploads a dataset (uint8 images and labels) as packed u32 storage buffers
    uploadDataset(dataset, withOrder) {
        const pad4 = (bytes) => {
            const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
            padded.set(bytes);
            return padded;
        };
        const make = (bytes) => {
            const data = pad4(bytes);
            const buffer = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            this.device.queue.writeBuffer(buffer, 0, data);
            return buffer;
        };
        const gpu = {
            count: dataset.count,
            imagesBuffer: make(dataset.images),
            labelsBuffer: make(dataset.labels),
            orderBuffer: withOrder ? this.device.createBuffer({
                size: dataset.count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) : null,
        };
        gpu.bindGroup = this.createBindGroup(gpu);
        gpu.destroy = () => [gpu.imagesBuffer, gpu.labelsBuffer, gpu.orderBuffer].forEach(b => b && b.destroy());
        return gpu;
    }

    // Weights and zeroed momentum
    setWeights(weights) {
        const data = new Float32Array(NUM_PARAMS * 2);
        data.set(weights);
        this.device.queue.writeBuffer(this.weightsBuffer, 0, data);
    }

    // Writes Params for slots 0..steps.length-1; see the layout table above PARAMS_SIZE.
    // steps[i] = { start, count, mode, dataLen, lr, momentum, weightDecay, augSeed, maxShift }
    writeSlots(steps) {
        const words = this.paramsStride / 4;
        steps.forEach((s, i) => {
            const o = i * words;
            this.paramsU32[o + 0] = s.start;
            this.paramsU32[o + 1] = s.count;
            this.paramsU32[o + 2] = s.mode;
            this.paramsU32[o + 3] = Math.max(s.dataLen || 1, 1);
            this.paramsF32[o + 4] = s.lr || 0;
            this.paramsF32[o + 5] = s.momentum || 0;
            this.paramsF32[o + 6] = s.weightDecay || 0;
            this.paramsU32[o + 7] = s.augSeed || 0;
            this.paramsU32[o + 8] = s.maxShift || 0;
            this.paramsU32[o + 9] = 0;
            this.paramsU32[o + 10] = 0;
            this.paramsU32[o + 11] = 0;
        });
        if (steps.length) {
            this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData, 0, steps.length * this.paramsStride);
        }
    }

    dispatch(pass, name, invocations) {
        pass.setPipeline(this.pipelines[name]);
        pass.dispatchWorkgroups(Math.ceil(invocations / WORKGROUP));
    }

    // Encodes one mini-batch in slot `slot`: forward (with gather unless mode 2), softmax/loss,
    // then for training (mode 0) stats, backward, per-sample gradients and the update
    encodeBatch(pass, bindGroup, slot, step) {
        pass.setBindGroup(0, bindGroup, [slot * this.paramsStride]);
        const n = step.count;
        if (step.mode !== 2) this.dispatch(pass, 'cs_gather', n * IMG_PIXELS);
        for (const [name, size] of FORWARD_KERNELS) this.dispatch(pass, name, n * size);
        this.dispatch(pass, 'cs_softmax', n);
        if (step.mode === 2) return;
        pass.setPipeline(this.pipelines.cs_stats);
        pass.dispatchWorkgroups(1);
        if (step.mode !== 0) return;
        for (const [name, size] of BACKWARD_KERNELS) this.dispatch(pass, name, n * size);
        this.dispatch(pass, 'cs_update', NUM_PARAMS);
    }

    encodeMapMax(pass, bindGroup, slot) {
        pass.setBindGroup(0, bindGroup, [slot * this.paramsStride]);
        pass.setPipeline(this.pipelines.cs_mapmax);
        pass.dispatchWorkgroups(1);
    }

    destroy() {
        [this.paramsBuffer, this.weightsBuffer, this.workBuffer, this.gradsBuffer, this.statsBuffer, this.dummyBuffer]
            .forEach(b => b.destroy());
    }
}
