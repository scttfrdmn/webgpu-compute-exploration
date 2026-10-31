// Parallel sort (js/examples/parallel-sort.js).
//
// CPU replica of the bitonic (global / local-sort / local-merge), radix (histogram / scan / scatter)
// and f32 key-conversion shaders; validates the algorithm and index math, not the WGSL itself —
// see #51. The replicas follow the shaders' thread/workgroup indexing (pairIndex, flatGroup,
// 2D dispatch grids, digit-major histogram, chunked reduce-then-scan, packed 16-bit rank counters)
// and are compared against Array.prototype.sort / TypedArray.prototype.sort.
//
// Real code tested: sortInternals.bitonicSchedule / dispatchGrid / nextPowerOfTwo / makeBars and the
// ParallelSort uniform writers (BitonicParams, ChartParams, RadixParams, ConvertParams) through a
// mock device, decoded with the WGSL struct layouts.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sortInternals as S, ParallelSort, wgslModules } from '../js/examples/parallel-sort.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { rng, encode, decode, edgeCases, cpuSort } from './helpers/parallel-sort-refs.mjs';

const WG = S.WORKGROUP_SIZE;
const BLOCK = S.BITONIC_BLOCK;
const RADIX_BITS = 32 / S.RADIX_PASSES;

// ---- Bitonic replica ----
const pairIndex = (t, j) => Math.floor(t / j) * (2 * j) + (t % j);

function compareExchange(arr, i, l, ascending) {
    const a = arr[i], b = arr[l];
    if ((ascending && a > b) || (!ascending && a < b)) { arr[i] = b; arr[l] = a; }
}

// Runs every stage of bitonicSchedule over `keys` (padded power-of-two Uint32Array), visiting
// the workgroups of the 2D dispatch grid like the shader (flatGroup = y·gridX + x).
function bitonicReplica(keys, useShared, maxPerDim = 65535) {
    const padded = keys.length;
    const threads = padded / 2;
    const stages = S.bitonicSchedule(padded, useShared);
    const [gx, gy] = S.dispatchGrid(Math.ceil(threads / WG), maxPerDim);
    const local = new Uint32Array(BLOCK);
    for (const stage of stages) {
        for (let wy = 0; wy < gy; wy++) for (let wx = 0; wx < gx; wx++) {
            const group = wy * gx + wx;
            if (stage.kind === 'global') {
                for (let lid = 0; lid < WG; lid++) {
                    const t = group * WG + lid;
                    if (t >= threads) continue;
                    const i = pairIndex(t, stage.j);
                    compareExchange(keys, i, i + stage.j, (i & stage.k) === 0);
                }
                continue;
            }
            if (group * WG >= threads) continue;
            const base = group * BLOCK;
            local.set(keys.subarray(base, base + BLOCK));
            const run = (k, j) => {
                for (let lane = 0; lane < WG; lane++) {
                    const i = pairIndex(lane, j);
                    compareExchange(local, i, i + j, ((base + i) & k) === 0);
                }
            };
            if (stage.kind === 'localSort') {
                for (let k = 2; k <= BLOCK; k <<= 1) for (let j = k >> 1; j > 0; j >>= 1) run(k, j);
            } else {
                for (let j = BLOCK >> 1; j > 0; j >>= 1) run(stage.k, j);
            }
            keys.set(local, base);
        }
    }
    return stages.length;
}

function bitonicSortU32(input, useShared, maxPerDim) {
    const padded = S.nextPowerOfTwo(input.length);
    const keys = new Uint32Array(padded).fill(S.PAD_KEY);
    keys.set(input);
    bitonicReplica(keys, useShared, maxPerDim);
    return keys.slice(0, input.length);
}

// ---- Radix replica ----
function radixPass(keysIn, keysOut, shift) {
    const count = keysIn.length;
    const numTiles = Math.ceil(count / S.RADIX_TILE);
    const B = S.RADIX_BUCKETS;
    const digitOf = key => (key >>> shift) & (B - 1);
    const offsets = new Uint32Array(B * numTiles);

    // cs_radix_histogram
    for (let tile = 0; tile < numTiles; tile++) {
        const counts = new Uint32Array(B);
        for (let r = 0; r < S.RADIX_ROUNDS; r++) for (let lid = 0; lid < WG; lid++) {
            const idx = tile * S.RADIX_TILE + r * WG + lid;
            if (idx < count) counts[digitOf(keysIn[idx])]++;
        }
        for (let d = 0; d < B; d++) offsets[d * numTiles + tile] = counts[d];
    }

    // cs_radix_scan: one workgroup, chunked reduce, Hillis-Steele over 256 partial sums, rewrite
    const total = B * numTiles;
    const chunk = Math.ceil(total / WG);
    assert.ok(chunk <= S.SCAN_MAX_CHUNK, `scan chunk ${chunk} exceeds MAX_CHUNK`);
    const sums = new Uint32Array(WG);
    for (let lid = 0; lid < WG; lid++) {
        let s = 0;
        for (let i = 0; i < chunk; i++) { const idx = lid * chunk + i; if (idx < total) s += offsets[idx]; }
        sums[lid] = s;
    }
    let partial = sums.slice();
    for (let o = 1; o < WG; o <<= 1) {
        const next = partial.slice();
        for (let lid = o; lid < WG; lid++) next[lid] = partial[lid] + partial[lid - o];
        partial = next;
    }
    for (let lid = 0; lid < WG; lid++) {
        let running = partial[lid] - sums[lid];
        for (let i = 0; i < chunk; i++) {
            const idx = lid * chunk + i;
            if (idx < total) { const c = offsets[idx]; offsets[idx] = running; running += c; }
        }
    }

    // cs_radix_scatter: 16 counters of 16 bits packed in 8 words per lane (word = digit >> 1,
    // half = digit & 1), Hillis-Steele scan across lanes gives the rank among lower lanes.
    for (let tile = 0; tile < numTiles; tile++) {
        const tileBase = new Uint32Array(B);
        for (let d = 0; d < B; d++) tileBase[d] = offsets[d * numTiles + tile];
        for (let r = 0; r < S.RADIX_ROUNDS; r++) {
            const own = new Uint32Array(WG * 8);
            const keys = new Uint32Array(WG), digits = new Uint32Array(WG), valid = new Uint8Array(WG);
            for (let lid = 0; lid < WG; lid++) {
                const idx = tile * S.RADIX_TILE + r * WG + lid;
                if (idx >= count) continue;
                valid[lid] = 1;
                keys[lid] = keysIn[idx];
                const d = digitOf(keys[lid]);
                digits[lid] = d;
                own[lid * 8 + (d >> 1)] = (1 << ((d & 1) * 16)) >>> 0;
            }
            let scan = own.slice();
            for (let o = 1; o < WG; o <<= 1) {
                const next = scan.slice();
                for (let lid = o; lid < WG; lid++) {
                    for (let w = 0; w < 8; w++) next[lid * 8 + w] = (scan[lid * 8 + w] + scan[(lid - o) * 8 + w]) >>> 0;
                }
                scan = next;
            }
            for (let lid = 0; lid < WG; lid++) {
                if (!valid[lid]) continue;
                const d = digits[lid];
                const word = (scan[lid * 8 + (d >> 1)] - own[lid * 8 + (d >> 1)]) >>> 0;
                const rank = (word >>> ((d & 1) * 16)) & 0xffff;
                keysOut[tileBase[d] + rank] = keys[lid];
            }
            for (let d = 0; d < B; d++) {
                tileBase[d] += (scan[(WG - 1) * 8 + (d >> 1)] >>> ((d & 1) * 16)) & 0xffff;
            }
        }
    }
}

function radixSortU32(input) {
    let a = Uint32Array.from(input), b = new Uint32Array(input.length);
    for (let p = 0; p < S.RADIX_PASSES; p++) {
        radixPass(a, b, p * RADIX_BITS);
        [a, b] = [b, a];
    }
    return a;   // 8 passes: back in the first buffer, as the shader's keysA
}

// rng, encode/decode (f32 key conversion replica), edgeCases and cpuSort: test/helpers/parallel-sort-refs.mjs

test('bitonicSchedule: stage counts and order (real code)', (t) => {
    for (const padded of [2, 8, 256, 512, 1024, 8192, 1 << 20]) {
        const L = Math.log2(padded);
        const global = S.bitonicSchedule(padded, false);
        assert.equal(global.length, L * (L + 1) / 2, `global stages for ${padded}`);
        assert.deepEqual(global[0], { kind: 'global', k: 2, j: 1 });
        const shared = S.bitonicSchedule(padded, true);
        if (padded < BLOCK) {
            assert.deepEqual(shared, global);
        } else {
            let expected = 1;
            for (let k = BLOCK * 2; k <= padded; k *= 2) expected += Math.log2(k) - Math.log2(BLOCK) + 1;
            assert.equal(shared.length, expected, `shared stages for ${padded}`);
        }
        if (padded === 1 << 20) t.diagnostic(`n=2^20: ${global.length} global stages, ${shared.length} with workgroup memory`);
    }
    assert.equal(S.nextPowerOfTwo(1000), 1024);
    assert.equal(S.nextPowerOfTwo(1024), 1024);
    assert.deepEqual(S.dispatchGrid(10, 65535), [10, 1]);
    const [x, y] = S.dispatchGrid(100000, 65535);
    assert.ok(x * y >= 100000 && x <= 65535 && y <= 65535);
});

test('bitonic replica (global and workgroup-memory variants) matches CPU sort', (t) => {
    let total = 0;
    for (const [name, input] of Object.entries(edgeCases(1))) {
        const expected = cpuSort(input);
        for (const shared of [false, true]) {
            // maxPerDim = 5 forces 2D dispatch grids (flatGroup, with excess workgroups) for n > 2560
            const got = bitonicSortU32(input, shared, 5);
            assert.deepEqual(got, expected, `${name} shared=${shared}`);
            total++;
        }
    }
    t.diagnostic(`${total} bitonic runs over sorted/reversed/all-equal/PAD_KEY/non-power-of-2 inputs: 0 mismatches`);
});

test('pairIndex visits every index exactly once per stage', () => {
    for (const n of [16, 1024]) {
        for (let j = 1; j < n; j *= 2) {
            const seen = new Uint8Array(n);
            for (let t = 0; t < n / 2; t++) { const i = pairIndex(t, j); seen[i]++; seen[i + j]++; assert.equal(i & j, 0); }
            assert.ok(seen.every(v => v === 1), `n=${n} j=${j}`);
        }
    }
});

test('radix replica matches CPU sort (u32 edge cases)', (t) => {
    let n = 0;
    for (const [name, input] of Object.entries(edgeCases(2))) {
        assert.deepEqual(radixSortU32(input), cpuSort(input), name);
        n++;
    }
    const r = rng(3);
    const big = Uint32Array.from({ length: 10000 }, () => (r() * 0x100000000) >>> 0);
    assert.deepEqual(radixSortU32(big), cpuSort(big), 'random 10000');
    t.diagnostic(`${n + 1} radix inputs (up to 10,000 keys, 10 tiles): 0 mismatches`);
});

test('radix scan capacity: RADIX_MAX_KEYS fits MAX_CHUNK exactly', (t) => {
    const tiles = S.RADIX_MAX_KEYS / S.RADIX_TILE;
    const chunk = Math.ceil(S.RADIX_BUCKETS * tiles / WG);
    t.diagnostic(`RADIX_MAX_KEYS=${S.RADIX_MAX_KEYS}: ${tiles} tiles, scan chunk ${chunk} of ${S.SCAN_MAX_CHUNK}`);
    assert.equal(chunk, S.SCAN_MAX_CHUNK);
    // One more tile would overflow the chunk loop bound and silently drop histogram entries
    assert.ok(Math.ceil(S.RADIX_BUCKETS * (tiles + 1) / WG) > S.SCAN_MAX_CHUNK);
    // Packed 16-bit rank counters: at most WG lanes per round can share a digit
    assert.ok(WG < 0x10000);
});

test('f32 key encoding is order preserving; float sorts match Float32Array.sort', (t) => {
    const r = rng(4);
    const specials = [0, -0, 1, -1, Infinity, -Infinity, 3.4028234663852886e38, -3.4028234663852886e38,
        1.401298464324817e-45, -1.401298464324817e-45, 1.1754943508222875e-38, -1.1754943508222875e-38, 0.5, -0.5];
    const f = new Float32Array(3000);
    for (let i = 0; i < f.length; i++) f[i] = i < specials.length * 4 ? specials[i % specials.length] : (r() * 2 - 1) * 1e6;
    const bits = new Uint32Array(f.buffer);
    const expectedBits = new Uint32Array(f.slice().sort().buffer);   // TypedArray sort: -0 before +0

    // Monotonic: for random float pairs, a < b ⇔ enc(a) < enc(b) (bitwise distinct values)
    let violations = 0;
    for (let i = 0; i + 1 < f.length; i++) {
        const a = f[i], b = f[i + 1];
        if (Object.is(a, b)) continue;
        const less = a < b || (a === 0 && b === 0 && Object.is(a, -0));
        if (less !== (encode(bits[i]) < encode(bits[i + 1]))) violations++;
    }
    assert.equal(violations, 0);
    for (const u of bits) assert.equal(decode(encode(u)), u);

    const enc = bits.map(encode);
    const viaRadix = radixSortU32(enc).map(decode);
    const viaBitonic = bitonicSortU32(enc, true).map(decode);
    assert.deepEqual(viaRadix, expectedBits);
    assert.deepEqual(viaBitonic, expectedBits);
    t.diagnostic(`3000 f32 keys incl. ±0, ±Inf, ±denormal, ±FLT_MAX: radix and bitonic bit-identical to Float32Array.sort`);
});

test('makeBars patterns are permutations / bounded (real code)', () => {
    for (const pattern of ['random', 'reversed', 'nearly']) {
        const v = S.makeBars(1000, pattern);
        assert.deepEqual(Uint32Array.from(v).sort(), Uint32Array.from({ length: 1000 }, (_, i) => i + 1), pattern);
    }
    const few = S.makeBars(1000, 'few');
    assert.ok(new Set(few).size <= 8);
});

// ---- Uniform layouts (real writers through a mock device) ----
const L = {
    bitonic: structLayout(wgslModules().find(m => m.name === 'bitonic').code, 'BitonicParams'),
    radix: structLayout(wgslModules().find(m => m.name === 'radix').code, 'RadixParams'),
    convert: structLayout(wgslModules().find(m => m.name === 'convert').code, 'ConvertParams'),
    chart: structLayout(wgslModules().find(m => m.name === 'render').code, 'ChartParams'),
};

function makeSorter() {
    const device = mockDevice();
    const canvas = { width: 813, height: 427, getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) }) };
    return { device, sorter: new ParallelSort(device, canvas) };
}

test('uniform layouts: BitonicParams and ChartParams writers match WGSL', (t) => {
    assert.equal(L.bitonic.size, 16);
    assert.equal(L.chart.size, 32);
    const { device, sorter } = makeSorter();
    sorter.setBarCount(1000);   // writes per-stage BitonicParams, then ChartParams via render()
    const stageWrite = device.lastWrite(sorter.stageParamsBuffer);
    assert.equal(stageWrite.bytes.byteLength, sorter.stages.length * sorter.paramsStride);
    sorter.stages.forEach((stage, s) => {
        const p = decodeStruct(stageWrite.bytes, L.bitonic, s * sorter.paramsStride);
        assert.deepEqual(p, { k: stage.k, j: stage.j, threads: 512, pad0: 0 }, `stage ${s}`);
    });
    const chart = decodeStruct(device.lastWrite(sorter.chartParamsBuffer).bytes, L.chart);
    assert.equal(chart.bars, 1024);
    assert.equal(chart.k, 2);
    assert.equal(chart.j, 1);
    assert.equal(chart.maxValue, 1000);
    assert.equal(chart.width, 813);
    assert.equal(chart.height, 427);
    assert.equal(chart.done, 0);
    t.diagnostic(`${sorter.stages.length} BitonicParams slots at stride ${sorter.paramsStride}; ChartParams decoded OK`);
});

test('uniform layouts: RadixParams / ConvertParams / shared-bitonic writers in runBenchmark', async (t) => {
    const { device, sorter } = makeSorter();
    const isUniform = b => (b.usage & GPUBufferUsage.UNIFORM) !== 0;
    const n = 3000;
    device.writes.length = 0;
    await sorter.runBenchmark('radix', n, 'f32');
    const radixWrite = device.writes.find(w => isUniform(w.buffer) && w.buffer.size === S.RADIX_PASSES * sorter.paramsStride);
    for (let p = 0; p < S.RADIX_PASSES; p++) {
        const v = decodeStruct(radixWrite.bytes, L.radix, p * sorter.paramsStride);
        assert.deepEqual(v, { shift: p * RADIX_BITS, count: n, numTiles: Math.ceil(n / S.RADIX_TILE), pad0: 0 }, `pass ${p}`);
    }
    const convertWrite = device.writes.find(w => isUniform(w.buffer) && w.buffer.size === 16);
    assert.deepEqual(decodeStruct(convertWrite.bytes, L.convert), { count: n, pad0: 0, pad1: 0, pad2: 0 });

    device.writes.length = 0;
    await sorter.runBenchmark('bitonic-shared', 5000, 'u32');
    const stages = S.bitonicSchedule(8192, true);
    const w = device.writes.find(x => isUniform(x.buffer) && x.buffer.size === stages.length * sorter.paramsStride);
    stages.forEach((stage, s) => {
        assert.deepEqual(decodeStruct(w.bytes, L.bitonic, s * sorter.paramsStride),
            { k: stage.k, j: stage.j, threads: 4096, pad0: 0 });
    });
    t.diagnostic(`RadixParams x${S.RADIX_PASSES}, ConvertParams, ${stages.length} shared-bitonic slots decoded OK`);
});
