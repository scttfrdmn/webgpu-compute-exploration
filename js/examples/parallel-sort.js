// Parallel sorting on the GPU: bitonic sort and LSD radix sort, plus an animated bitonic bar chart.
//
// Uniform layouts (every field is u32, written through a Uint32Array view):
//
// struct BitonicParams (16 bytes; one copy per (k, j) stage at a dynamic offset of paramsStride):
//   offset  0  k        u32  bitonic block size of this stage               -> Uint32 view [0]
//   offset  4  j        u32  compare distance of this stage                 -> Uint32 view [1]
//   offset  8  threads  u32  padded length / 2 (one thread per pair)        -> Uint32 view [2]
//   offset 12  pad0     u32  (unused, keeps the struct 16 bytes)            -> Uint32 view [3]
//
// struct RadixParams (16 bytes; one copy per digit pass at a dynamic offset of paramsStride):
//   offset  0  shift     u32  bit position of this pass's 4-bit digit (0, 4, ..., 28) -> Uint32 view [0]
//   offset  4  count     u32  number of keys                                         -> Uint32 view [1]
//   offset  8  numTiles  u32  ceil(count / RADIX_TILE)                               -> Uint32 view [2]
//   offset 12  pad0      u32                                                         -> Uint32 view [3]
//
// struct ConvertParams (16 bytes, plain uniform):
//   offset  0  count  u32  number of keys to encode / decode                -> Uint32 view [0]
//   offset  4  pad0   u32                                                   -> Uint32 view [1]
//   offset  8  pad1   u32                                                   -> Uint32 view [2]
//   offset 12  pad2   u32                                                   -> Uint32 view [3]
//
// struct ChartParams (32 bytes, render shader):
//   offset  0  bars      u32  bars drawn (the padded length)                -> Uint32 view [0]
//   offset  4  k         u32  block size of the next stage                  -> Uint32 view [1]
//   offset  8  j         u32  compare distance of the next stage            -> Uint32 view [2]
//   offset 12  maxValue  u32  tallest real bar                              -> Uint32 view [3]
//   offset 16  width     u32  canvas width in pixels                        -> Uint32 view [4]
//   offset 20  height    u32  canvas height in pixels                       -> Uint32 view [5]
//   offset 24  done      u32  1 = all stages ran                            -> Uint32 view [6]
//   offset 28  pad0      u32                                                -> Uint32 view [7]
//
// Per-dispatch values (k, j, shift) never share one uniform slot: writeBuffer runs before the
// whole submit, so each dispatch gets its own 256-byte-aligned slot and a dynamic offset.
//
// Compute-only use (the Measure tab, js/teach/experiments/parallel-sort.js, and test/gpu/): a
// ParallelSort with canvas = null creates no render pipeline or canvas context; measureSort() runs
// the shipped runBenchmark() on it (sort, timing, readback and the check against the CPU sort).

import { GpuTimer, GPU_TIME_NOTE, GPU_TIME_UNAVAILABLE } from '../gpu-timer.js';
import { PhaseClock } from '../phase-clock.js';
import { mulberry32 } from '../refs/random.js';

const PARAMS_SIZE = 16;
const CHART_PARAMS_SIZE = 32;

const WORKGROUP_SIZE = 256;
const BITONIC_BLOCK = 512;          // elements sorted in workgroup memory by one workgroup
const RADIX_BITS = 4;
const RADIX_BUCKETS = 16;
const RADIX_PASSES = 8;             // 32 bits / 4 bits per digit
const RADIX_ROUNDS = 4;             // 256-key rounds per tile
const RADIX_TILE = WORKGROUP_SIZE * RADIX_ROUNDS;   // 1024 keys per workgroup
const SCAN_MAX_CHUNK = 256;         // scan: 256 threads × 256 entries = 65,536 histogram entries
const RADIX_MAX_KEYS = (SCAN_MAX_CHUNK * WORKGROUP_SIZE / RADIX_BUCKETS) * RADIX_TILE;  // 2^22
const PAD_KEY = 0xFFFFFFFF;

const bitonicShader = `
struct BitonicParams {
    k: u32,
    j: u32,
    threads: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<uniform> params: BitonicParams;

const WG: u32 = 256u;
const BLOCK: u32 = 512u;

var<workgroup> localKeys: array<u32, 512>;

// Bitonic sort of a power-of-two array (non-power-of-two inputs are padded with 0xFFFFFFFF).
// For block sizes k = 2, 4, ..., n and compare distances j = k/2, k/4, ..., 1, every element i
// with bit j clear is compared with i + j. Blocks whose index bit k is clear sort ascending, the
// others descending, so each finished k-stage leaves bitonic sequences of length 2k for the next.
// That is log2(n)·(log2(n)+1)/2 stages of n/2 independent compare-exchanges: O(n log² n) work.
//
// Thread t of a stage owns one pair. Inserting a 0 bit at position log2(j) of t gives the lower
// index: i = (t / j)·2j + (t mod j), partner l = i + j. Every index appears in exactly one pair.
fn pairIndex(t: u32, j: u32) -> u32 {
    return (t / j) * (2u * j) + (t % j);
}

// Workgroups are laid out on a 2D grid when n/2 / 256 exceeds maxComputeWorkgroupsPerDimension
fn flatGroup(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}

// One (k, j) stage over the whole array in global memory
@compute @workgroup_size(256)
fn cs_bitonic_global(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(num_workgroups) nwg: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = flatGroup(wid, nwg) * WG + lid.x;
    if (t >= params.threads) {
        return;
    }
    let i = pairIndex(t, params.j);
    let l = i + params.j;
    let a = keys[i];
    let b = keys[l];
    let ascending = (i & params.k) == 0u;
    if ((ascending && a > b) || (!ascending && a < b)) {
        keys[i] = b;
        keys[l] = a;
    }
}

// Compare-exchange inside the workgroup's 512-element block; globalBase decides the direction
fn localCompareExchange(lane: u32, globalBase: u32, k: u32, j: u32) {
    let i = pairIndex(lane, j);
    let l = i + j;
    let a = localKeys[i];
    let b = localKeys[l];
    let ascending = ((globalBase + i) & k) == 0u;
    if ((ascending && a > b) || (!ascending && a < b)) {
        localKeys[i] = b;
        localKeys[l] = a;
    }
}

// Shared-memory variant, first dispatch: all stages with k <= 512 for one 512-key block, with a
// barrier between stages instead of a dispatch. 45 stages in one dispatch.
@compute @workgroup_size(256)
fn cs_bitonic_local_sort(@builtin(workgroup_id) wid: vec3<u32>,
                         @builtin(num_workgroups) nwg: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>) {
    let group = flatGroup(wid, nwg);
    if (group * WG >= params.threads) {
        return;   // whole workgroup leaves: the condition is workgroup-uniform
    }
    let base = group * BLOCK;
    localKeys[lid.x] = keys[base + lid.x];
    localKeys[lid.x + WG] = keys[base + lid.x + WG];
    workgroupBarrier();
    for (var k = 2u; k <= BLOCK; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            localCompareExchange(lid.x, base, k, j);
            workgroupBarrier();
        }
    }
    keys[base + lid.x] = localKeys[lid.x];
    keys[base + lid.x + WG] = localKeys[lid.x + WG];
}

// Shared-memory variant, end of each k > 512: the stages j = 256, ..., 1 stay inside one block
@compute @workgroup_size(256)
fn cs_bitonic_local_merge(@builtin(workgroup_id) wid: vec3<u32>,
                          @builtin(num_workgroups) nwg: vec3<u32>,
                          @builtin(local_invocation_id) lid: vec3<u32>) {
    let group = flatGroup(wid, nwg);
    if (group * WG >= params.threads) {
        return;
    }
    let base = group * BLOCK;
    localKeys[lid.x] = keys[base + lid.x];
    localKeys[lid.x + WG] = keys[base + lid.x + WG];
    workgroupBarrier();
    for (var j = BLOCK >> 1u; j > 0u; j = j >> 1u) {
        localCompareExchange(lid.x, base, params.k, j);
        workgroupBarrier();
    }
    keys[base + lid.x] = localKeys[lid.x];
    keys[base + lid.x + WG] = localKeys[lid.x + WG];
}
`;

const radixShader = `
struct RadixParams {
    shift: u32,
    count: u32,
    numTiles: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> keysIn: array<u32>;
@group(0) @binding(1) var<storage, read_write> keysOut: array<u32>;
// digit-major: entry d·numTiles + tile. Histogram counts, then (after the scan) global offsets.
@group(0) @binding(2) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(3) var<uniform> params: RadixParams;

const WG: u32 = 256u;
const BUCKETS: u32 = 16u;
const ROUNDS: u32 = 4u;
const TILE: u32 = 1024u;           // WG · ROUNDS keys per workgroup
const MAX_CHUNK: u32 = 256u;       // scan entries per thread: supports 16 · 4096 tiles = 2^22 keys

// LSD radix sort, 4-bit digits, 8 passes from the least significant digit up. Each pass is a
// stable counting sort by one digit, so after all passes the keys are sorted. Per pass:
//   1. cs_radix_histogram: each workgroup counts the digits of its 1024-key tile
//   2. cs_radix_scan: exclusive prefix sum over the digit-major histogram. Entry (d, tile)
//      becomes "keys with a smaller digit anywhere + keys with digit d in earlier tiles"
//   3. cs_radix_scatter: each key goes to offset(d, tile) + its rank among the equal digits
//      before it in the tile, which keeps equal digits in input order (stable)

var<workgroup> digitCounts: array<atomic<u32>, 16>;
var<workgroup> partialSums: array<u32, 256>;
// Scatter: 16 digit counters of 16 bits packed in 8 words per thread (lo = digits 0-7)
var<workgroup> scanLo: array<vec4<u32>, 256>;
var<workgroup> scanHi: array<vec4<u32>, 256>;
var<workgroup> tileBase: array<u32, 16>;

fn flatGroup(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}

fn digitOf(key: u32) -> u32 {
    return (key >> params.shift) & (BUCKETS - 1u);
}

@compute @workgroup_size(256)
fn cs_radix_histogram(@builtin(workgroup_id) wid: vec3<u32>,
                      @builtin(num_workgroups) nwg: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>) {
    let tileIndex = flatGroup(wid, nwg);
    if (tileIndex >= params.numTiles) {
        return;
    }
    if (lid.x < BUCKETS) {
        atomicStore(&digitCounts[lid.x], 0u);
    }
    workgroupBarrier();
    for (var r = 0u; r < ROUNDS; r++) {
        let idx = tileIndex * TILE + r * WG + lid.x;
        if (idx < params.count) {
            atomicAdd(&digitCounts[digitOf(keysIn[idx])], 1u);
        }
    }
    workgroupBarrier();
    if (lid.x < BUCKETS) {
        offsets[lid.x * params.numTiles + tileIndex] = atomicLoad(&digitCounts[lid.x]);
    }
}

// Exclusive scan of the 16·numTiles histogram entries by a single workgroup (reduce-then-scan):
// each thread sums a contiguous chunk serially, the 256 chunk sums get a Hillis-Steele scan in
// workgroup memory (8 steps), and each thread rewrites its chunk as running offsets. At most
// 65,536 entries, so one workgroup is enough and needs no inter-workgroup synchronization.
@compute @workgroup_size(256)
fn cs_radix_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let total = BUCKETS * params.numTiles;
    let chunk = (total + WG - 1u) / WG;
    let start = lid.x * chunk;

    var sum = 0u;
    for (var i = 0u; i < MAX_CHUNK; i++) {
        if (i >= chunk) {
            break;
        }
        let idx = start + i;
        if (idx < total) {
            sum += offsets[idx];
        }
    }
    partialSums[lid.x] = sum;
    workgroupBarrier();

    for (var o = 1u; o < WG; o = o << 1u) {
        var v = partialSums[lid.x];
        if (lid.x >= o) {
            v += partialSums[lid.x - o];
        }
        workgroupBarrier();
        partialSums[lid.x] = v;
        workgroupBarrier();
    }

    var running = partialSums[lid.x] - sum;   // inclusive -> exclusive
    for (var i = 0u; i < MAX_CHUNK; i++) {
        if (i >= chunk) {
            break;
        }
        let idx = start + i;
        if (idx < total) {
            let c = offsets[idx];
            offsets[idx] = running;
            running += c;
        }
    }
}

fn wordOf(lo: vec4<u32>, hi: vec4<u32>, w: u32) -> u32 {
    if (w < 4u) {
        return lo[w];
    }
    return hi[w - 4u];
}

// Stable scatter. The tile is processed in 4 rounds of 256 keys in index order. In a round each
// thread one-hot encodes its digit as a 16-bit counter (+1 in counter d), and a Hillis-Steele scan
// of these packed counters across the workgroup gives every thread the number of equal digits
// held by lower threads. No counter can exceed 256, so the 16-bit fields never carry.
@compute @workgroup_size(256)
fn cs_radix_scatter(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(num_workgroups) nwg: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let tileIndex = flatGroup(wid, nwg);
    if (tileIndex >= params.numTiles) {
        return;
    }
    if (lid.x < BUCKETS) {
        tileBase[lid.x] = offsets[lid.x * params.numTiles + tileIndex];
    }
    workgroupBarrier();

    let lanes = vec4<u32>(0u, 1u, 2u, 3u);
    for (var r = 0u; r < ROUNDS; r++) {
        let idx = tileIndex * TILE + r * WG + lid.x;
        let valid = idx < params.count;
        var key = 0u;
        var digit = 0u;
        var ownLo = vec4<u32>(0u);
        var ownHi = vec4<u32>(0u);
        if (valid) {
            key = keysIn[idx];
            digit = digitOf(key);
            let bit = 1u << ((digit & 1u) * 16u);
            let w = digit >> 1u;
            ownLo = select(vec4<u32>(0u), vec4<u32>(bit), lanes == vec4<u32>(w));
            ownHi = select(vec4<u32>(0u), vec4<u32>(bit), lanes + vec4<u32>(4u) == vec4<u32>(w));
        }

        var lo = ownLo;
        var hi = ownHi;
        scanLo[lid.x] = lo;
        scanHi[lid.x] = hi;
        workgroupBarrier();
        for (var o = 1u; o < WG; o = o << 1u) {
            if (lid.x >= o) {
                lo += scanLo[lid.x - o];
                hi += scanHi[lid.x - o];
            }
            workgroupBarrier();
            scanLo[lid.x] = lo;
            scanHi[lid.x] = hi;
            workgroupBarrier();
        }

        if (valid) {
            let word = wordOf(lo - ownLo, hi - ownHi, digit >> 1u);
            let rank = (word >> ((digit & 1u) * 16u)) & 0xFFFFu;
            keysOut[tileBase[digit] + rank] = key;
        }
        workgroupBarrier();   // everyone has read tileBase for this round

        // Thread 255's inclusive counters are the round's digit totals
        if (lid.x < BUCKETS) {
            let totalWord = wordOf(scanLo[WG - 1u], scanHi[WG - 1u], lid.x >> 1u);
            tileBase[lid.x] += (totalWord >> ((lid.x & 1u) * 16u)) & 0xFFFFu;
        }
        workgroupBarrier();
    }
}
`;

const convertShader = `
struct ConvertParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<uniform> params: ConvertParams;

const WG: u32 = 256u;

// Order-preserving f32 <-> u32 map: negative floats get all bits flipped (so more negative means
// smaller), non-negative floats get the sign bit set (so they sort above every negative).
// Unsigned comparison of the keys then matches float comparison (NaN aside).
@compute @workgroup_size(256)
fn cs_encode_f32(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * WG + lid.x;
    if (i >= params.count) {
        return;
    }
    let u = keys[i];
    keys[i] = u ^ select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn cs_decode_f32(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * WG + lid.x;
    if (i >= params.count) {
        return;
    }
    let u = keys[i];
    keys[i] = u ^ select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) == 0u);
}
`;

const renderShader = `
struct ChartParams {
    bars: u32,
    k: u32,
    j: u32,
    maxValue: u32,
    width: u32,
    height: u32,
    done: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<uniform> chart: ChartParams;

const PAD_KEY: u32 = 0xFFFFFFFFu;
const CHART_TOP: f32 = 0.04;      // fractions of the canvas height
const CHART_BOTTOM: f32 = 0.90;
const STRIP_TOP: f32 = 0.93;      // compare-pair strip below the bars

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(corners[vi], 0.0, 1.0);
    return out;
}

// Dark blue -> teal -> yellow by bar height
fn heightRamp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.20, 0.25, 0.75);
    let c1 = vec3<f32>(0.10, 0.70, 0.70);
    let c2 = vec3<f32>(0.98, 0.85, 0.25);
    if (t < 0.5) { return mix(c0, c1, t / 0.5); }
    return mix(c1, c2, (t - 0.5) / 0.5);
}

// One bar per array element, read straight from the storage buffer the compute pass sorts.
// The background tints each k-block by its sort direction (ascending / descending) for the next
// stage; the strip under the bars marks the lower (orange) and upper (cyan) element of each
// compare pair at distance j, darker for every other group of 2j.
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let w = f32(chart.width);
    let y = pos.y / f32(chart.height);
    let barWidth = w / f32(chart.bars);
    let bar = min(u32(pos.x / barWidth), chart.bars - 1u);
    let inGap = barWidth >= 4.0 && pos.x - f32(bar) * barWidth > barWidth - 1.0;

    var background = vec3<f32>(0.05, 0.06, 0.10);
    if (chart.done == 0u) {
        if ((bar & chart.k) == 0u) {
            background = vec3<f32>(0.06, 0.08, 0.15);   // ascending block
        } else {
            background = vec3<f32>(0.13, 0.06, 0.13);   // descending block
        }
    }

    if (y >= STRIP_TOP) {
        if (inGap) { return vec4<f32>(background, 1.0); }
        if (chart.done != 0u) { return vec4<f32>(0.25, 0.75, 0.40, 1.0); }
        var c = vec3<f32>(0.10, 0.75, 0.90);            // upper element of the pair
        if ((bar & chart.j) == 0u) {
            c = vec3<f32>(0.95, 0.55, 0.15);            // lower element of the pair
        }
        if (((bar / (2u * chart.j)) & 1u) == 1u) {
            c *= 0.6;
        }
        return vec4<f32>(c, 1.0);
    }

    let key = keys[bar];
    var h = 1.0;
    if (key != PAD_KEY) {
        h = f32(key) / f32(max(chart.maxValue, 1u));
    }
    let barTop = CHART_BOTTOM - h * (CHART_BOTTOM - CHART_TOP);
    if (y < barTop || y > CHART_BOTTOM || inGap) {
        return vec4<f32>(background, 1.0);
    }
    if (key == PAD_KEY) {
        return vec4<f32>(0.32, 0.32, 0.36, 1.0);        // padding up to the next power of two
    }
    return vec4<f32>(heightRamp(h), 1.0);
}
`;

const BAR_COUNTS = [512, 1000, 1024, 2048];

const PATTERNS = {
    random:   { name: 'Random' },
    reversed: { name: 'Reversed' },
    nearly:   { name: 'Nearly sorted' },
    few:      { name: 'Few unique' },
};

const ALGORITHMS = {
    'bitonic':        { name: 'Bitonic sort (global passes)' },
    'bitonic-shared': { name: 'Bitonic sort (workgroup memory for j ≤ 256)' },
    'radix':          { name: 'LSD radix sort (4-bit digits)' },
};

const BENCH_RUNS = 3;               // sorts of the same input per Explore benchmark (best of 3)
const MAX_BENCH_RUNS = 10;

// Benchmark keys from a seeded generator (mulberry32), so a run can be repeated exactly:
// full-range random u32, or f32 uniform in ±1e6 (finite; the bit patterns are uploaded unchanged)
export function generateKeys(n, dataType, seed) {
    const r = mulberry32(seed);
    if (dataType === 'f32') {
        const keys = new Float32Array(n);
        for (let i = 0; i < n; i++) keys[i] = (r() * 2 - 1) * 1e6;
        return keys;
    }
    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) keys[i] = (r() * 0x100000000) >>> 0;
    return keys;
}

const MAX_STAGES_PER_FRAME = 16;

function nextPowerOfTwo(n) {
    let p = 1;
    while (p < n) p *= 2;
    return p;
}

// Workgroup grid for a 1D job: a single row while it fits, otherwise rows of up to maxPerDim.
// The shaders flatten (x, y) back to y·gridX + x and skip workgroups past the end.
function dispatchGrid(workgroups, maxPerDim) {
    if (workgroups <= maxPerDim) return [Math.max(workgroups, 1), 1];
    const y = Math.ceil(workgroups / maxPerDim);
    const x = Math.ceil(workgroups / y);
    if (y > maxPerDim) throw new Error(`${workgroups} workgroups exceed the dispatch limit`);
    return [x, y];
}

// Stage list for a padded (power-of-two) length. Global-only: one stage per (k, j). Shared:
// one local sort for k ≤ 512, then for each larger k the global stages with j ≥ 512 followed by
// one local merge for j = 256 ... 1.
function bitonicSchedule(padded, useShared) {
    const stages = [];
    if (!useShared || padded < BITONIC_BLOCK) {
        for (let k = 2; k <= padded; k *= 2) {
            for (let j = k / 2; j >= 1; j /= 2) stages.push({ kind: 'global', k, j });
        }
        return stages;
    }
    stages.push({ kind: 'localSort', k: BITONIC_BLOCK, j: BITONIC_BLOCK / 2 });
    for (let k = BITONIC_BLOCK * 2; k <= padded; k *= 2) {
        for (let j = k / 2; j >= BITONIC_BLOCK; j /= 2) stages.push({ kind: 'global', k, j });
        stages.push({ kind: 'localMerge', k, j: BITONIC_BLOCK / 2 });
    }
    return stages;
}

// Test values for the visualization: heights 1..n (or 8 levels for "Few unique")
function makeBars(n, pattern) {
    const values = new Uint32Array(n);
    for (let i = 0; i < n; i++) values[i] = i + 1;
    const swap = (a, b) => { const t = values[a]; values[a] = values[b]; values[b] = t; };
    if (pattern === 'reversed') {
        values.reverse();
    } else if (pattern === 'nearly') {
        for (let s = 0; s < Math.max(1, Math.floor(n / 50)); s++) {
            const a = Math.floor(Math.random() * n);
            const b = Math.min(n - 1, a + 1 + Math.floor(Math.random() * 8));
            swap(a, b);
        }
    } else {
        if (pattern === 'few') {
            for (let i = 0; i < n; i++) values[i] = Math.ceil(((i + 1) / n) * 8) * Math.ceil(n / 8);
        }
        for (let i = n - 1; i > 0; i--) swap(i, Math.floor(Math.random() * (i + 1)));
    }
    return values;
}

class ParallelSort {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;
        this.benchRunning = false;

        this.barCount = 1024;
        this.pattern = 'random';
        this.stagesPerSecond = 8;
        this.stageIndex = 0;
        this.stageBudget = 0;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;
        this.lastFrameTime = 0;
        this.benchResult = null;

        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.maxPerDim = device.limits.maxComputeWorkgroupsPerDimension || 65535;

        this.chartParams = new ArrayBuffer(CHART_PARAMS_SIZE);
        this.chartU32 = new Uint32Array(this.chartParams);
        this.chartParamsBuffer = device.createBuffer({
            size: CHART_PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.setupComputePipelines();
        if (canvas) {
            this.setupRenderPipeline();
            this.setupCanvas();
        }
    }

    setupComputePipelines() {
        const device = this.device;
        const dynamicUniform = { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE };

        const bitonicModule = device.createShaderModule({ code: bitonicShader });
        this.bitonicLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: dynamicUniform }
            ]
        });
        const bitonicPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bitonicLayout] });
        const bitonic = (entryPoint) => device.createComputePipeline({
            layout: bitonicPipelineLayout,
            compute: { module: bitonicModule, entryPoint }
        });
        this.bitonicPipelines = {
            global: bitonic('cs_bitonic_global'),
            localSort: bitonic('cs_bitonic_local_sort'),
            localMerge: bitonic('cs_bitonic_local_merge'),
        };

        const radixModule = device.createShaderModule({ code: radixShader });
        this.radixLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: dynamicUniform }
            ]
        });
        const radixPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.radixLayout] });
        const radix = (entryPoint) => device.createComputePipeline({
            layout: radixPipelineLayout,
            compute: { module: radixModule, entryPoint }
        });
        this.radixPipelines = {
            histogram: radix('cs_radix_histogram'),
            scan: radix('cs_radix_scan'),
            scatter: radix('cs_radix_scatter'),
        };

        const convertModule = device.createShaderModule({ code: convertShader });
        this.convertLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        const convertPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.convertLayout] });
        this.encodePipeline = device.createComputePipeline({
            layout: convertPipelineLayout,
            compute: { module: convertModule, entryPoint: 'cs_encode_f32' }
        });
        this.decodePipeline = device.createComputePipeline({
            layout: convertPipelineLayout,
            compute: { module: convertModule, entryPoint: 'cs_decode_f32' }
        });
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });
        this.renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });
    }

    setupCanvas() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });
    }

    // Writes one BitonicParams slot per stage; see the layout table at the top of the file
    writeBitonicParams(buffer, stages, padded) {
        const data = new ArrayBuffer(Math.max(stages.length, 1) * this.paramsStride);
        const u32 = new Uint32Array(data);
        const wordsPerSlot = this.paramsStride / 4;
        stages.forEach((stage, s) => {
            const o = s * wordsPerSlot;
            u32[o + 0] = stage.k;
            u32[o + 1] = stage.j;
            u32[o + 2] = padded / 2;
            u32[o + 3] = 0;
        });
        this.device.queue.writeBuffer(buffer, 0, data);
    }

    // ---- Visualization ----

    // (Re)creates the bar buffer for `count` bars, padded to a power of two with PAD_KEY
    setBarCount(count) {
        this.pause();
        if (this.barBuffer) this.barBuffer.destroy();
        if (this.stageParamsBuffer) this.stageParamsBuffer.destroy();

        this.barCount = count;
        this.barPadded = nextPowerOfTwo(count);
        this.stages = bitonicSchedule(this.barPadded, false);
        this.barGrid = dispatchGrid(Math.ceil(this.barPadded / 2 / WORKGROUP_SIZE), this.maxPerDim);

        this.barBuffer = this.device.createBuffer({
            size: this.barPadded * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.stageParamsBuffer = this.device.createBuffer({
            size: Math.max(this.stages.length, 1) * this.paramsStride,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.writeBitonicParams(this.stageParamsBuffer, this.stages, this.barPadded);

        this.stageBindGroup = this.device.createBindGroup({
            layout: this.bitonicLayout,
            entries: [
                { binding: 0, resource: { buffer: this.barBuffer } },
                { binding: 1, resource: { buffer: this.stageParamsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        });
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderLayout,
            entries: [
                { binding: 0, resource: { buffer: this.barBuffer } },
                { binding: 1, resource: { buffer: this.chartParamsBuffer } }
            ]
        });
        this.shuffle();
    }

    shuffle() {
        const values = makeBars(this.barCount, this.pattern);
        this.maxValue = values.reduce((m, v) => Math.max(m, v), 1);
        const data = new Uint32Array(this.barPadded).fill(PAD_KEY);
        data.set(values);
        this.device.queue.writeBuffer(this.barBuffer, 0, data);
        this.stageIndex = 0;
        this.stageBudget = 0;
        this.render();
    }

    get sorted() {
        return this.stageIndex >= this.stages.length;
    }

    // Writes ChartParams; see the layout table at the top of the file
    writeChartParams() {
        const next = this.stages[Math.min(this.stageIndex, this.stages.length - 1)];
        this.chartU32[0] = this.barPadded;
        this.chartU32[1] = next ? next.k : 1;
        this.chartU32[2] = next ? next.j : 1;
        this.chartU32[3] = this.maxValue;
        this.chartU32[4] = this.canvas.width;
        this.chartU32[5] = this.canvas.height;
        this.chartU32[6] = this.sorted ? 1 : 0;
        this.chartU32[7] = 0;
        this.device.queue.writeBuffer(this.chartParamsBuffer, 0, this.chartParams);
    }

    encodeRender(commandEncoder) {
        this.writeChartParams();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.05, g: 0.06, b: 0.10, a: 1.0 }
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();
    }

    render() {
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    // Runs the next `count` stages (each with its own params slot), then draws; no readback
    advance(count) {
        const n = Math.min(count, this.stages.length - this.stageIndex);
        if (n <= 0) return;
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.bitonicPipelines.global);
        for (let s = this.stageIndex; s < this.stageIndex + n; s++) {
            computePass.setBindGroup(0, this.stageBindGroup, [s * this.paramsStride]);
            computePass.dispatchWorkgroups(this.barGrid[0], this.barGrid[1]);
        }
        computePass.end();
        this.stageIndex += n;
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        if (this.onStage) this.onStage();
    }

    step() {
        this.pause();
        this.advance(1);
    }

    start() {
        if (this.running) return;
        if (this.sorted) this.shuffle();
        this.running = true;
        this.stageBudget = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.lastFrameTime = this.lastSampleTime;
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
    }

    animate() {
        if (!this.running || this.destroyed) return;

        const now = performance.now();
        const dt = Math.min(now - this.lastFrameTime, 100);
        this.lastFrameTime = now;
        this.stageBudget += dt * this.stagesPerSecond / 1000;
        const due = Math.min(Math.floor(this.stageBudget), MAX_STAGES_PER_FRAME);
        if (due > 0) {
            this.stageBudget -= due;
            this.advance(due);
        }

        this.framesSinceSample++;
        if (now - this.lastSampleTime >= 500) {
            this.fps = this.framesSinceSample * 1000 / (now - this.lastSampleTime);
            this.framesSinceSample = 0;
            this.lastSampleTime = now;
            if (this.onStats) this.onStats();
        }

        if (this.sorted) {
            this.pause();
            if (this.onStats) this.onStats();
            return;
        }
        requestAnimationFrame(() => this.animate());
    }

    // ---- Benchmark ----

    // Sorts n random keys on the GPU `runs` times (default BENCH_RUNS), reads the result back once and
    // checks it against Uint32Array/Float32Array.prototype.sort on a copy of the same input.
    // Options: input (n keys of dataType; default: generateKeys with a random seed), expected (the
    // CPU-sorted bit patterns of that input, as a Uint32Array; when given, the CPU sort is skipped and
    // cpuMs is null), clock (a PhaseClock that receives compile = buffers and bind groups, upload =
    // writeBuffer of the keys until the queue is idle, dispatch = each sort's submit -> done, readback).
    async runBenchmark(algorithm, n, dataType, { input: given = null, expected = null, runs = BENCH_RUNS, clock = new PhaseClock() } = {}) {
        if (this.benchRunning) return this.benchResult;
        this.benchRunning = true;
        const device = this.device;
        const buffers = [];
        // Times the sort pass of each run with timestamp queries (no-op without 'timestamp-query')
        const timer = new GpuTimer(device, { maxPasses: 1, poolSize: 1 });
        const makeBuffer = (size, usage) => {
            const buffer = device.createBuffer({ size, usage });
            buffers.push(buffer);
            return buffer;
        };

        try {
            const isRadix = algorithm === 'radix';
            const padded = isRadix ? n : nextPowerOfTwo(n);
            if (isRadix && n > RADIX_MAX_KEYS) {
                throw new Error(`radix sort here supports up to ${RADIX_MAX_KEYS.toLocaleString()} keys`);
            }
            const bytes = padded * 4;
            const maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
            if (bytes > maxBinding) {
                throw new Error(`${bytes.toLocaleString()} bytes exceed this device's storage buffer limit (${maxBinding.toLocaleString()})`);
            }

            const Keys = dataType === 'f32' ? Float32Array : Uint32Array;
            if (given && !(given instanceof Keys && given.length === n)) {
                throw new TypeError(`input must be a ${Keys.name} of ${n} keys`);
            }
            if (expected && !(expected instanceof Uint32Array && expected.length === n)) {
                throw new TypeError(`expected must be a Uint32Array of ${n} bit patterns`);
            }
            const input = given ?? generateKeys(n, dataType, (Math.random() * 0x100000000) >>> 0);
            const inputBits = new Uint32Array(input.buffer, input.byteOffset, n);

            let cpuMs = null;
            let cpuBits = expected;
            if (!cpuBits) {
                const cpuSorted = input.slice();
                const cpuStart = performance.now();
                cpuSorted.sort();
                cpuMs = performance.now() - cpuStart;
                cpuBits = new Uint32Array(cpuSorted.buffer);
            }
            const sortRuns = Math.min(Math.max(Math.round(runs) || 1, 1), MAX_BENCH_RUNS);

            const setupStart = performance.now();
            const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            const keysA = makeBuffer(bytes, storage);
            const convertParams = makeBuffer(PARAMS_SIZE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            device.queue.writeBuffer(convertParams, 0, new Uint32Array([n, 0, 0, 0]));
            const convertBindGroup = device.createBindGroup({
                layout: this.convertLayout,
                entries: [
                    { binding: 0, resource: { buffer: keysA } },
                    { binding: 1, resource: { buffer: convertParams } }
                ]
            });
            const convertGrid = dispatchGrid(Math.ceil(n / WORKGROUP_SIZE), this.maxPerDim);

            // Encodes the whole sort (all dispatches) into one compute pass
            let encodeSort;
            let dispatches;
            if (isRadix) {
                const keysB = makeBuffer(bytes, storage);
                const numTiles = Math.ceil(n / RADIX_TILE);
                const histogram = makeBuffer(RADIX_BUCKETS * numTiles * 4, GPUBufferUsage.STORAGE);
                const radixParams = makeBuffer(RADIX_PASSES * this.paramsStride, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

                // RadixParams per pass; see the layout table at the top of the file
                const data = new ArrayBuffer(RADIX_PASSES * this.paramsStride);
                const u32 = new Uint32Array(data);
                for (let p = 0; p < RADIX_PASSES; p++) {
                    const o = p * this.paramsStride / 4;
                    u32[o + 0] = p * RADIX_BITS;
                    u32[o + 1] = n;
                    u32[o + 2] = numTiles;
                    u32[o + 3] = 0;
                }
                device.queue.writeBuffer(radixParams, 0, data);

                // Pass p reads keysA and writes keysB when p is even, the reverse when odd;
                // after the 8th pass the sorted keys are back in keysA
                const bindGroups = [[keysA, keysB], [keysB, keysA]].map(([src, dst]) => device.createBindGroup({
                    layout: this.radixLayout,
                    entries: [
                        { binding: 0, resource: { buffer: src } },
                        { binding: 1, resource: { buffer: dst } },
                        { binding: 2, resource: { buffer: histogram } },
                        { binding: 3, resource: { buffer: radixParams, offset: 0, size: PARAMS_SIZE } }
                    ]
                }));
                const tileGrid = dispatchGrid(numTiles, this.maxPerDim);
                dispatches = RADIX_PASSES * 3;
                encodeSort = (pass) => {
                    for (let p = 0; p < RADIX_PASSES; p++) {
                        const offset = [p * this.paramsStride];
                        pass.setBindGroup(0, bindGroups[p % 2], offset);
                        pass.setPipeline(this.radixPipelines.histogram);
                        pass.dispatchWorkgroups(tileGrid[0], tileGrid[1]);
                        pass.setPipeline(this.radixPipelines.scan);
                        pass.dispatchWorkgroups(1);
                        pass.setPipeline(this.radixPipelines.scatter);
                        pass.dispatchWorkgroups(tileGrid[0], tileGrid[1]);
                    }
                };
            } else {
                const stages = bitonicSchedule(padded, algorithm === 'bitonic-shared');
                const stageParams = makeBuffer(Math.max(stages.length, 1) * this.paramsStride,
                    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
                this.writeBitonicParams(stageParams, stages, padded);
                const bindGroup = device.createBindGroup({
                    layout: this.bitonicLayout,
                    entries: [
                        { binding: 0, resource: { buffer: keysA } },
                        { binding: 1, resource: { buffer: stageParams, offset: 0, size: PARAMS_SIZE } }
                    ]
                });
                // One thread per pair; the local kernels use one 256-thread workgroup per 512 keys
                const grid = dispatchGrid(Math.ceil(padded / 2 / WORKGROUP_SIZE), this.maxPerDim);
                dispatches = stages.length;
                encodeSort = (pass) => {
                    stages.forEach((stage, s) => {
                        pass.setPipeline(this.bitonicPipelines[stage.kind]);
                        pass.setBindGroup(0, bindGroup, [s * this.paramsStride]);
                        pass.dispatchWorkgroups(grid[0], grid[1]);
                    });
                };
            }
            if (dataType === 'f32') dispatches += 2;

            const padding = padded > n ? new Uint32Array(padded - n).fill(PAD_KEY) : null;
            clock.add('compile', performance.now() - setupStart);
            const gpuTimes = [];
            const kernelTimes = [];
            for (let run = 0; run < sortRuns; run++) {
                await clock.time('upload', async () => {
                    device.queue.writeBuffer(keysA, 0, inputBits);
                    if (padding) device.queue.writeBuffer(keysA, n * 4, padding);
                    await device.queue.onSubmittedWorkDone();   // uploads finish before the clock starts
                });

                const start = performance.now();
                const commandEncoder = device.createCommandEncoder();
                const pass = commandEncoder.beginComputePass(timer.computePassDescriptor('sort'));
                if (dataType === 'f32') {
                    pass.setPipeline(this.encodePipeline);
                    pass.setBindGroup(0, convertBindGroup);
                    pass.dispatchWorkgroups(convertGrid[0], convertGrid[1]);
                }
                encodeSort(pass);
                if (dataType === 'f32') {
                    pass.setPipeline(this.decodePipeline);
                    pass.setBindGroup(0, convertBindGroup);
                    pass.dispatchWorkgroups(convertGrid[0], convertGrid[1]);
                }
                pass.end();
                timer.resolve(commandEncoder);
                device.queue.submit([commandEncoder.finish()]);
                const timing = timer.read();
                await device.queue.onSubmittedWorkDone();
                gpuTimes.push(performance.now() - start);
                clock.add('dispatch', gpuTimes[gpuTimes.length - 1]);
                const result = await timing;
                if (result) kernelTimes.push(result.total);
            }

            // One readback of the last run
            const readStart = performance.now();
            const staging = makeBuffer(n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            const copyEncoder = device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(keysA, 0, staging, 0, n * 4);
            device.queue.submit([copyEncoder.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const gpuBits = new Uint32Array(staging.getMappedRange().slice(0));
            staging.unmap();
            const readMs = performance.now() - readStart;
            clock.add('readback', readMs);

            // Same bits as the CPU sort <=> sorted and a permutation of the input
            const gpuValues = dataType === 'f32' ? new Float32Array(gpuBits.buffer) : gpuBits;
            let mismatches = 0;
            let inversions = 0;
            for (let i = 0; i < n; i++) {
                if (gpuBits[i] !== cpuBits[i]) mismatches++;
                if (i > 0 && gpuValues[i - 1] > gpuValues[i]) inversions++;
            }

            this.benchResult = {
                algorithm, n, padded, dataType, dispatches, cpuMs, readMs, mismatches, inversions,
                gpuTimes, bestMs: Math.min(...gpuTimes),
                timestamps: timer.supported, kernelTimes,
                bestKernelMs: kernelTimes.length ? Math.min(...kernelTimes) : null,
                phases: clock.phases,
            };
            return this.benchResult;
        } finally {
            buffers.forEach(b => b.destroy());
            timer.destroy();
            this.benchRunning = false;
        }
    }

    destroy() {
        this.running = false;
        this.destroyed = true;
        if (this.barBuffer) this.barBuffer.destroy();
        if (this.stageParamsBuffer) this.stageParamsBuffer.destroy();
        this.chartParamsBuffer.destroy();
    }
}

// ---- Compute-only API for the Measure tab (js/teach/experiments/parallel-sort.js) and test/gpu/ ----

export const MEASURE_MAX_KEYS = RADIX_MAX_KEYS;   // 4,194,304 = the benchmark's largest N

// One runBenchmark() of `algorithm` ('bitonic' | 'bitonic-shared' | 'radix') on n keys of dataType
// ('u32' | 'f32') on a compute-only ParallelSort: `runs` sorts of the same input (default 1: one
// independent measurement for the Measure protocol), one readback, and a bit-for-bit comparison with
// TypedArray.prototype.sort.
//   keys      the input (Uint32Array for u32, Float32Array for f32); default: random keys
//   expected  the CPU-sorted bit patterns of keys (Uint32Array); default: sorted here (and timed: cpuMs)
//   sorter    a compute-only ParallelSort to reuse (its pipelines are compiled once, see computeSorter);
//             default: a new one, whose construction counts as compile
// Resolves to runBenchmark's result: { algorithm, n, padded, dataType, dispatches, cpuMs (null when
// `expected` was given), readMs, mismatches, inversions, gpuTimes, bestMs (wall, submit → done),
// timestamps, kernelTimes, bestKernelMs (GPU, null without timestamps) } plus wallMs and gpuMs (medians
// over the runs) and phases { compile, upload, dispatch, readback } in wall ms (js/phase-clock.js):
// compile = pipeline creation (only when no sorter was given) + this sort's buffers and bind groups;
// upload = writeBuffer of the keys until the queue is idle; dispatch = submit → onSubmittedWorkDone
// of each sort (the wall time); readback = copy, mapAsync and the copy out.
export async function measureSort(gpuDevice, { algorithm, n, dataType = 'u32', keys = null, expected = null, runs = 1, sorter = null }) {
    if (!ALGORITHMS[algorithm]) throw new RangeError(`unknown algorithm ${algorithm}`);
    if (dataType !== 'u32' && dataType !== 'f32') throw new RangeError(`unknown key type ${dataType}`);
    if (!Number.isInteger(n) || n < 1 || n > MEASURE_MAX_KEYS) throw new RangeError(`N = ${n} is outside 1 … ${MEASURE_MAX_KEYS}`);
    if (!Number.isInteger(runs) || runs < 1 || runs > MAX_BENCH_RUNS) throw new RangeError(`runs = ${runs} is outside 1 … ${MAX_BENCH_RUNS}`);
    const clock = new PhaseClock();
    const own = sorter === null;
    const s = own ? clock.time('compile', () => computeSorter(gpuDevice)) : sorter;
    try {
        const r = await s.runBenchmark(algorithm, n, dataType, { input: keys, expected, runs, clock });
        return { ...r, wallMs: medianOf(r.gpuTimes), gpuMs: r.kernelTimes.length ? medianOf(r.kernelTimes) : null, phases: clock.phases };
    } finally {
        if (own) s.destroy();
    }
}

// A ParallelSort without a canvas (compute pipelines only) for repeated measureSort calls; destroy() it
export function computeSorter(gpuDevice) {
    return new ParallelSort(gpuDevice.device ?? gpuDevice, null);
}

function medianOf(values) {
    const v = [...values].sort((a, b) => a - b);
    const h = (v.length - 1) / 2;
    return (v[Math.floor(h)] + v[Math.ceil(h)]) / 2;
}

export async function initParallelSort(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-sort');
    const output = document.getElementById('output-sort');

    const sorter = new ParallelSort(device, canvas);
    sorter.setBarCount(sorter.barCount);
    sorter.onStage = () => updateOutput();
    sorter.onStats = () => updateOutput();

    document.getElementById('btn-sort-play').addEventListener('click', () => {
        sorter.start();
        updateOutput();
    });

    document.getElementById('btn-sort-pause').addEventListener('click', () => {
        sorter.pause();
        updateOutput();
    });

    document.getElementById('btn-sort-step').addEventListener('click', () => {
        sorter.step();
        updateOutput();
    });

    document.getElementById('btn-sort-shuffle').addEventListener('click', () => {
        sorter.pause();
        sorter.shuffle();
        updateOutput();
    });

    document.getElementById('sort-bars').addEventListener('change', (e) => {
        const count = parseInt(e.target.value);
        if (!BAR_COUNTS.includes(count)) return;
        sorter.setBarCount(count);
        updateOutput();
    });

    document.getElementById('sort-pattern').addEventListener('change', (e) => {
        if (!PATTERNS[e.target.value]) return;
        sorter.pattern = e.target.value;
        sorter.pause();
        sorter.shuffle();
        updateOutput();
    });

    const speedSlider = document.getElementById('sort-speed');
    const speedValue = document.getElementById('sort-speed-value');
    speedSlider.addEventListener('input', (e) => {
        sorter.stagesPerSecond = Math.min(Math.max(parseInt(e.target.value), 1), 60);
        speedValue.textContent = sorter.stagesPerSecond;
        updateOutput();
    });

    let benchStatus = '';   // progress or error text shown instead of the last result
    const benchButton = document.getElementById('btn-sort-bench');
    benchButton.addEventListener('click', async () => {
        const algorithm = document.getElementById('sort-algorithm').value;
        const n = parseInt(document.getElementById('sort-n').value);
        const dataType = document.getElementById('sort-type').value === 'f32' ? 'f32' : 'u32';
        if (!ALGORITHMS[algorithm] || !(n > 0)) return;
        benchButton.disabled = true;
        benchStatus = `<span class="info">Running ${ALGORITHMS[algorithm].name} on ${n.toLocaleString()} ${dataType} keys…</span>`;
        updateOutput();
        try {
            await sorter.runBenchmark(algorithm, n, dataType);
            benchStatus = '';
        } catch (error) {
            benchStatus = `<span class="error">Benchmark error: ${error.message}</span>`;
        }
        benchButton.disabled = false;
        updateOutput();
    });

    function benchmarkText() {
        if (benchStatus) return benchStatus;
        const r = sorter.benchResult;
        if (!r) return '<span class="info">Benchmark:</span> choose an algorithm, key type and size, then press Run benchmark';
        const ok = r.mismatches === 0 && r.inversions === 0;
        const verdict = ok
            ? '<span class="success">✓ Sorted, and identical to the CPU sort (same keys, same order)</span>'
            : `<span class="error">✗ ${r.mismatches.toLocaleString()} positions differ from the CPU sort, ${r.inversions.toLocaleString()} out-of-order neighbors</span>`;
        const speedup = r.cpuMs / r.bestMs;
        const faster = (x) => x >= 1 ? `${x.toFixed(1)}× faster` : `${(1 / x).toFixed(1)}× slower`;
        const kernelLine = !r.timestamps
            ? `• ${GPU_TIME_UNAVAILABLE}`
            : r.bestKernelMs === null
                ? '• GPU (timestamp queries): no result'
                : `• GPU (timestamp queries): ${r.bestKernelMs.toFixed(2)} ms best of ${r.kernelTimes.length} (${r.kernelTimes.map(t => t.toFixed(2)).join(', ')} ms)${r.bestKernelMs > 0 ? ` · ${(r.n / r.bestKernelMs / 1000).toFixed(1)} M keys/s · ${faster(r.cpuMs / r.bestKernelMs)} than the CPU sort` : ''}`;
        return `<span class="info">Benchmark: ${ALGORITHMS[r.algorithm].name}, ${r.n.toLocaleString()} ${r.dataType} keys</span>
${verdict}
${kernelLine}
• Wall (submit → done): ${r.bestMs.toFixed(2)} ms best of ${r.gpuTimes.length} (${r.gpuTimes.map(t => t.toFixed(2)).join(', ')} ms) · ${(r.n / r.bestMs / 1000).toFixed(1)} M keys/s
• CPU ${r.dataType === 'f32' ? 'Float32Array' : 'Uint32Array'}.prototype.sort: ${r.cpuMs.toFixed(2)} ms (GPU wall time is ${faster(speedup)})
• Dispatches per sort: ${r.dispatches}${r.padded !== r.n ? ` · padded to ${r.padded.toLocaleString()} with 0xFFFFFFFF` : ''}${r.dataType === 'f32' ? ' · includes f32 → u32 key encode and decode' : ''}
• Readback (once, not timed above): ${r.readMs.toFixed(2)} ms
• Wall time = submit → onSubmittedWorkDone, so it includes queue latency; GPU time is the sort pass alone; the first run may include warm-up${r.timestamps ? `\n<span class="hint">${GPU_TIME_NOTE}</span>` : ''}`;
    }

    function updateOutput() {
        const total = sorter.stages.length;
        const next = sorter.stages[sorter.stageIndex];
        const last = sorter.stages[sorter.stageIndex - 1];
        const stageLine = sorter.sorted
            ? '<span class="success">Sorted</span> (Play or Shuffle for new bars)'
            : `next is stage ${sorter.stageIndex + 1}: block size k = ${next.k}, compare distance j = ${next.j}`;
        const log2 = Math.log2(sorter.barPadded);

        output.innerHTML = `<span class="success">✓ Parallel Sorting Ready</span>

<span class="info">Visualization (bitonic sort, one stage per tick):</span>
• Bars: ${sorter.barCount.toLocaleString()}${sorter.barPadded !== sorter.barCount ? ` (+${sorter.barPadded - sorter.barCount} gray padding bars, 0xFFFFFFFF, up to ${sorter.barPadded})` : ''} · Pattern: ${PATTERNS[sorter.pattern].name}
• Stage ${sorter.stageIndex} of ${total}${last ? ` (last ran k = ${last.k}, j = ${last.j})` : ''} · ${stageLine}
• Speed: ${sorter.stagesPerSecond} stages / s · Status: ${sorter.running ? '<span class="success">Playing</span>' : '<span class="info">Paused</span>'}${sorter.running ? ` · ${sorter.fps.toFixed(0)} FPS` : ''}
• Background: blue = block sorts ascending, purple = descending (k-blocks); strip: orange/cyan = lower/upper element of each pair j apart

${benchmarkText()}

<span class="info">Algorithms:</span>
• Bitonic: for k = 2…n, j = k/2…1, thread t compares i = (t / j)·2j + t mod j with i + j, ascending where bit k of i is 0; log₂n·(log₂n+1)/2 = ${log2 * (log2 + 1) / 2} stages for ${sorter.barPadded} keys, O(n log² n) work
• Bitonic with workgroup memory: one dispatch sorts each 512-key block (k ≤ 512); for larger k, global passes run j ≥ 512 and one dispatch finishes j = 256…1 in workgroup memory
• Radix (LSD, 4-bit digits, 8 passes): per-workgroup histograms of 1024-key tiles, one-workgroup exclusive scan (reduce-then-scan, Hillis-Steele over 256 chunk sums), stable scatter ranked by a packed 16-bit-counter scan; O(n) per pass
• f32 keys: negative floats have all bits flipped, others only the sign bit, so unsigned order = float order
• Large arrays use a 2D workgroup grid when a dimension would exceed ${sorter.maxPerDim.toLocaleString()} workgroups`;
    }

    updateOutput();
    return sorter;
}

// Exposed for offline (CPU) checks of the shaders and the dispatch math; not used by the page
export const sortInternals = {
    bitonicShader, radixShader, convertShader, renderShader,
    bitonicSchedule, dispatchGrid, nextPowerOfTwo, makeBars,
    WORKGROUP_SIZE, BITONIC_BLOCK, RADIX_TILE, RADIX_ROUNDS, RADIX_BUCKETS, RADIX_PASSES, RADIX_MAX_KEYS,
    PAD_KEY, SCAN_MAX_CHUNK,
};
export { ParallelSort };   // for the CPU tests in test/ (uniform writers via a mock device)

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'bitonic', code: bitonicShader },
        { name: 'radix', code: radixShader },
        { name: 'convert', code: convertShader },
        { name: 'render', code: renderShader },
    ];
}
