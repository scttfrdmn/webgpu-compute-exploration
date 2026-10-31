// Tests for js/examples/mnist-model.js (shared by examples 21 MNIST: Train and 22 MNIST: Recognize):
// the shipped JS — architecture constants, seeded He init, the per-epoch shuffle, the PCG
// augmentation hash, the MNW1 weights format, the work-buffer layout — and the MNISTGPU uniform
// writer (writeSlots) checked byte for byte against the WGSL Params / Stats structs.
// The WGSL kernels themselves are replicated in test/mnist-kernels.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as M from '../js/examples/mnist-model.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const WEIGHTS_FILE = fileURLToPath(new URL('../data/mnist/weights.bin', import.meta.url));

// `const NAME: u32 = 123u;` declarations of a WGSL module
function wgslU32Consts(code) {
    const out = {};
    for (const m of code.matchAll(/const\s+(\w+)\s*:\s*u32\s*=\s*(\d+)u\s*;/g)) out[m[1]] = Number(m[2]);
    return out;
}

test('architecture constants: TENSORS, OFF and NUM_PARAMS agree', (t) => {
    let offset = 0;
    for (const tensor of M.TENSORS) {
        assert.equal(tensor.offset, offset, `${tensor.name} offset`);
        offset += tensor.shape.reduce((a, b) => a * b, 1);
    }
    assert.equal(offset, M.NUM_PARAMS);
    assert.equal(M.NUM_PARAMS, 5994);
    for (const tensor of M.TENSORS) assert.equal(M.OFF[tensor.name.toUpperCase()], tensor.offset);
    // conv1 5×5 on 28×28 -> 24 -> pool 12; conv2 -> 8 -> pool 4; 16·4·4 = 256 dense inputs
    assert.equal(M.TENSORS[4].shape[1], 16 * ((((M.IMG_SIZE - 4) / 2) - 4) / 2) ** 2);
    t.diagnostic(`NUM_PARAMS = ${offset}`);
});

test('mulberry32 is deterministic and roughly uniform', (t) => {
    const a = M.mulberry32(42), b = M.mulberry32(42);
    let sum = 0, sum2 = 0;
    const n = 100000;
    for (let i = 0; i < n; i++) {
        const x = a();
        assert.equal(x, b());
        assert.ok(x >= 0 && x < 1);
        sum += x; sum2 += x * x;
    }
    const mean = sum / n, variance = sum2 / n - mean * mean;
    t.diagnostic(`mean ${mean.toFixed(5)} (1/2), variance ${variance.toFixed(5)} (1/12 = 0.08333)`);
    // standard error of the mean is 0.29/sqrt(1e5) ≈ 0.0009: 5 SE
    assert.ok(Math.abs(mean - 0.5) < 0.005);
    assert.ok(Math.abs(variance - 1 / 12) < 0.003);
});

test('initWeights: He-normal per tensor, zero biases, deterministic per seed', (t) => {
    const w = M.initWeights(1);
    assert.equal(w.length, M.NUM_PARAMS);
    assert.deepEqual(w, M.initWeights(1));
    assert.notDeepEqual(w, M.initWeights(2));
    const fanIn = { W1: 25, W2: 200, W3: 256 };
    for (const tensor of M.TENSORS) {
        const n = tensor.shape.reduce((a, b) => a * b, 1);
        const v = w.subarray(tensor.offset, tensor.offset + n);
        if (tensor.name.startsWith('b')) {
            assert.ok(v.every(x => x === 0), `${tensor.name} must be zero`);
            continue;
        }
        let s = 0, s2 = 0;
        for (const x of v) { s += x; s2 += x * x; }
        const mean = s / n, std = Math.sqrt(s2 / n - mean * mean);
        const expected = Math.sqrt(2 / fanIn[tensor.name]);
        const ratio = std / expected;
        t.diagnostic(`${tensor.name}: n=${n} mean ${mean.toFixed(4)} std ${std.toFixed(4)} expected ${expected.toFixed(4)} ratio ${ratio.toFixed(3)}`);
        // relative SE of a sample std is ~1/sqrt(2n): 5% for W1 (n=200), 1.3% for W2/W3; allow ~4 SE
        const tol = 4 / Math.sqrt(2 * n);
        assert.ok(Math.abs(ratio - 1) < tol, `${tensor.name} std ratio ${ratio}`);
        assert.ok(Math.abs(mean) < 4 * expected / Math.sqrt(n), `${tensor.name} mean ${mean}`);
    }
});

test('epochOrder: a permutation of 0..n-1, deterministic, different per epoch and seed', (t) => {
    const n = 10000;
    const o0 = M.epochOrder(n, 1, 0);
    assert.deepEqual(Array.from(o0).sort((a, b) => a - b), Array.from({ length: n }, (_, i) => i));
    assert.deepEqual(o0, M.epochOrder(n, 1, 0));
    const o1 = M.epochOrder(n, 1, 1);
    const o2 = M.epochOrder(n, 2, 0);
    let same01 = 0, same02 = 0, fixed = 0;
    for (let i = 0; i < n; i++) {
        if (o0[i] === o1[i]) same01++;
        if (o0[i] === o2[i]) same02++;
        if (o0[i] === i) fixed++;
    }
    t.diagnostic(`n=${n}: positions equal epoch0 vs epoch1 ${same01}, seed1 vs seed2 ${same02}, fixed points ${fixed} (expected ≈1 each)`);
    // a random permutation has Poisson(1) coincidences with any other: > 20 means correlated shuffles
    assert.ok(same01 < 20 && same02 < 20 && fixed < 20);
    // Unbiased: over 6,000 (seed, epoch) pairs every permutation of 3 elements appears ~1,000 times
    // (catches Sattolo's off-by-one `rng() * i`, which never leaves an element in place)
    const perms = new Map();
    for (let k = 0; k < 6000; k++) {
        const key = M.epochOrder(3, k, k % 7).join('');
        perms.set(key, (perms.get(key) || 0) + 1);
    }
    const dev = Math.max(...[...perms.values()].map(c => Math.abs(c - 1000) / 1000));
    t.diagnostic(`n=3 over 6000 seeds: ${perms.size}/6 permutations, worst deviation ${dev.toFixed(3)}`);
    assert.equal(perms.size, 6);
    assert.ok(dev < 0.15, 'SE ≈ 3%');
    // Edge cases
    assert.deepEqual(Array.from(M.epochOrder(1, 1, 0)), [0]);
    assert.equal(M.epochOrder(0, 1, 0).length, 0);
});

// Independent BigInt implementation of the PCG-RXS-M-XS hash (u32 arithmetic mod 2^32)
function pcgRef(v) {
    const M32 = (1n << 32n) - 1n;
    const state = (BigInt(v >>> 0) * 747796405n + 2891336453n) & M32;
    const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & M32;
    return Number((word >> 22n) ^ word);
}

test('pcgHash matches a BigInt u32 reference; sampleShift covers [-maxShift, maxShift]² uniformly', (t) => {
    const rng = M.mulberry32(7);
    const probes = [0, 1, 2, 0x7fffffff, 0x80000000, 0xffffffff];
    for (let i = 0; i < 2000; i++) probes.push(Math.floor(rng() * 4294967296));
    for (const v of probes) assert.equal(M.pcgHash(v), pcgRef(v), `pcgHash(${v})`);

    const seed = M.augmentSeed(1, 0);
    assert.notEqual(seed, M.augmentSeed(1, 1));
    assert.notEqual(seed, M.augmentSeed(2, 0));
    assert.ok(Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32);
    for (const maxShift of [0, 1, 2]) {
        const span = 2 * maxShift + 1;
        const counts = new Map();
        const n = 25000;
        for (let p = 0; p < n; p++) {
            const [dx, dy] = M.sampleShift(seed, p, maxShift);
            assert.ok(Number.isInteger(dx) && Math.abs(dx) <= maxShift && Math.abs(dy) <= maxShift);
            counts.set(`${dx},${dy}`, (counts.get(`${dx},${dy}`) || 0) + 1);
        }
        assert.equal(counts.size, span * span);
        const expected = n / (span * span);
        const worst = Math.max(...[...counts.values()].map(c => Math.abs(c - expected) / expected));
        t.diagnostic(`maxShift ${maxShift}: ${counts.size} shifts, worst relative deviation from uniform ${worst.toFixed(3)}`);
        // expected 1000 per cell for maxShift 2 (SE ≈ 3%): 15% bound
        assert.ok(worst < 0.15);
    }
});

test('MNW1 weights format: round trip, committed file, corruption rejected', (t) => {
    const w = M.initWeights(3);
    w[5] = -0; w[6] = 1e-40; w[7] = 3.4e38;
    const bytes = M.serializeWeights(w);
    assert.equal(bytes.length, 24108);
    assert.equal(String.fromCharCode(...bytes.subarray(0, 4)), 'MNW1');
    const back = M.parseWeights(bytes);
    assert.deepEqual(new Uint32Array(back.buffer), new Uint32Array(w.buffer), 'bit-exact round trip');
    // header: shapes of every tensor at 12 + 20k
    const view = new DataView(bytes.buffer);
    M.TENSORS.forEach((tensor, k) => {
        assert.equal(view.getUint32(12 + 20 * k, true), tensor.shape.length);
        tensor.shape.forEach((d, i) => assert.equal(view.getUint32(16 + 20 * k + 4 * i, true), d));
    });
    assert.equal(view.getFloat32(132, true), w[0]);

    const file = M.parseWeights(readFileSync(WEIGHTS_FILE));
    assert.equal(file.length, M.NUM_PARAMS);

    const corrupt = (fn) => { const b = bytes.slice(); fn(b, new DataView(b.buffer)); return b; };
    const cases = {
        'short file': bytes.slice(0, -4),
        'bad magic': corrupt(b => { b[0] = 0x4e; }),
        'bad version': corrupt((b, v) => v.setUint32(4, 2, true)),
        'bad tensor count': corrupt((b, v) => v.setUint32(8, 5, true)),
        'bad shape': corrupt((b, v) => v.setUint32(16, 9, true)),
        'bad rank': corrupt((b, v) => v.setUint32(12 + 20, 2, true)),
        'NaN weight': corrupt((b, v) => v.setFloat32(132 + 4 * 100, NaN, true)),
        'infinite weight': corrupt((b, v) => v.setFloat32(132 + 4 * 5993, Infinity, true)),
    };
    for (const [name, b] of Object.entries(cases)) assert.throws(() => M.parseWeights(b), undefined, name);
    assert.throws(() => M.serializeWeights(new Float32Array(10)));
    t.diagnostic(`round trip bit-exact; ${Object.keys(cases).length} corruptions rejected`);
});

test('workLayout: contiguous non-overlapping regions, matches the WGSL constants', (t) => {
    const sizes = { X: 784, A1: 4608, P1: 1152, A2: 1024, P2: 256, LOGIT: 10, PROB: 10, DLOG: 10, LOSS: 1,
        CORR: 1, DP2: 256, DZ2: 1024, DP1: 1152, DZ1: 4608 };
    for (const B of [1, 7, M.MAX_BATCH]) {
        const L = M.workLayout(B);
        let offset = 0;
        for (const [name, size] of Object.entries(sizes)) {
            assert.equal(L[name], offset, `B=${B} ${name}`);
            offset += size * B;
        }
        assert.equal(L.MAPMAX, offset);
        assert.equal(L.total, offset + 24);

        const c = wgslU32Consts(M.buildComputeShader(B));
        for (const name of [...Object.keys(sizes), 'MAPMAX']) assert.equal(c[`W_${name}`], L[name], `W_${name}`);
        assert.equal(c.MAX_BATCH, B);
        assert.equal(c.NPARAM, M.NUM_PARAMS);
        for (const [k, v] of Object.entries(M.OFF)) assert.equal(c[`OFF_${k}`], v, `OFF_${k}`);
        if (B === M.MAX_BATCH) t.diagnostic(`B=${B}: work buffer ${L.total} floats (${(L.total * 4 / 1e6).toFixed(2)} MB)`);
    }
});

test('Params / Stats: PARAMS_SIZE and STATS_SIZE match the WGSL structs', (t) => {
    for (const { name, code } of M.wgslModules()) {
        const params = structLayout(code, 'Params');
        const stats = structLayout(code, 'Stats');
        assert.equal(params.size, M.PARAMS_SIZE, `${name} Params size`);
        assert.equal(stats.size, M.STATS_SIZE, `${name} Stats size`);
        // mnist-train.js readStats() reads f32[0], u32[1], u32[2], f32[4], u32[5], u32[6]
        assert.deepEqual(
            [stats.offsets.trainLoss, stats.offsets.trainCorrect, stats.offsets.trainCount,
                stats.offsets.testLoss, stats.offsets.testCorrect, stats.offsets.testCount],
            [0, 4, 8, 16, 20, 24]);
        t.diagnostic(`${name}: Params ${params.size} B, Stats ${stats.size} B`);
    }
});

test('MNISTGPU.writeSlots writes every Params field at its WGSL offset and type (mock device)', (t) => {
    const device = mockDevice();
    const gpu = new M.MNISTGPU(device, 4, 3);
    assert.equal(gpu.paramsStride, 256);
    const layout = structLayout(moduleCode(M, `compute-batch${M.MAX_BATCH}`), 'Params');
    const steps = [
        { start: 9984, count: 16, mode: 0, dataLen: 10000, lr: 0.0347, momentum: 0.9, weightDecay: 0.0005, augSeed: 0xdeadbeef, maxShift: 2 },
        { start: 1920, count: 80, mode: 1, dataLen: 2000 },
        { start: 0, count: 1, mode: 2, dataLen: 0 },
    ];
    gpu.writeSlots(steps);
    const w = device.lastWrite(gpu.paramsBuffer);
    assert.equal(w.offset, 0);
    assert.equal(w.bytes.length, steps.length * gpu.paramsStride);
    const expect = (s) => ({
        start: s.start, count: s.count, mode: s.mode, dataLen: Math.max(s.dataLen || 1, 1),
        lr: Math.fround(s.lr || 0), momentum: Math.fround(s.momentum || 0), weightDecay: Math.fround(s.weightDecay || 0),
        augSeed: s.augSeed || 0, maxShift: s.maxShift || 0, pad0: 0, pad1: 0, pad2: 0,
    });
    steps.forEach((s, i) => {
        const got = decodeStruct(w.bytes, layout, i * gpu.paramsStride);
        assert.deepEqual(got, expect(s), `slot ${i}`);
    });
    t.diagnostic(`3 slots × ${gpu.paramsStride} B decoded against struct Params (${layout.size} B)`);

    gpu.setWeights(M.initWeights(1));
    const ww = device.lastWrite(gpu.weightsBuffer);
    const f = new Float32Array(ww.bytes.buffer);
    assert.equal(f.length, 2 * M.NUM_PARAMS);
    assert.deepEqual(f.subarray(0, M.NUM_PARAMS), M.initWeights(1));
    assert.ok(f.subarray(M.NUM_PARAMS).every(x => x === 0), 'momentum zeroed');
    // buffer sizes the kernels index into
    assert.equal(gpu.workBuffer.size, M.workLayout(4).total * 4);
    assert.equal(gpu.gradsBuffer.size, 4 * M.NUM_PARAMS * 4);
    assert.equal(gpu.statsBuffer.size, M.STATS_SIZE);
});
