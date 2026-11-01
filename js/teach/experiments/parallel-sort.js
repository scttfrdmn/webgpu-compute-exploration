// Measure experiments for parallel sorting (js/examples/parallel-sort.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureSort), which runs the shipped
// benchmark (runBenchmark: upload, one sort in one compute pass, one readback, a bit-for-bit check
// against the CPU sort) on a compute-only ParallelSort whose pipelines are created once per run. The CPU
// reference and baseline is TypedArray.prototype.sort (js/refs/sort.js cpuSort is the same call).
//
// Protocol (js/teach/measure.js): one ctx.measure per N. Each call is one independent *trial*: fresh keys
// from the seeded sequence keySeed(N, trial) (reproducible, different in every trial), the CPU sort of a
// copy (timed: the CPU baseline's sample), then each GPU algorithm sorting the same keys and checked
// against that CPU result. One discarded warm-up trial, then PROTOCOL.repeats trials (LARGE_PROTOCOL above
// LARGE_N, to fit the budget). Interleaving the algorithms within a trial keeps slow drifts (clocks,
// thermal state) from favouring one of them. Timed columns are medians with their interquartile range;
// the key-mismatch counts cover every sort, warm-up included.
//
// Caps (exported so test/parallel-sort-measure.test.mjs can check them): N ≤ 4,194,304 (the radix sort's
// limit), three algorithms per trial, at most 1 + 5 trials per N (1 + 3 from LARGE_N).

import { measureSort, computeSorter, generateKeys, MEASURE_MAX_KEYS, sortInternals } from '../../examples/parallel-sort.js';
import { encodeF32Key, decodeF32Key } from '../../refs/sort.js';
import { PhaseClock } from '../../phase-clock.js';
import { summarize, withSpread, fitPowerLaw, slopeAgrees } from '../stats.js';
import { check } from '../result-record.js';

export const COUNTS = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304];
export const MAX_N = 4194304;
export const ALGORITHMS = ['bitonic', 'bitonic-shared', 'radix'];
export const KEY = { bitonic: 'bitonic', 'bitonic-shared': 'shared', radix: 'radix' };
export const NAME = { bitonic: 'bitonic (global)', 'bitonic-shared': 'bitonic (workgroup memory)', radix: 'radix' };
export const PROTOCOL = { warmup: 1, repeats: 5 };
// From 2,097,152 keys a trial is three GPU sorts plus a CPU sort of 8–16 MB of keys (about 0.3–0.7 s on a
// laptop GPU), so these sizes use 3 repeats: the whole sweep is then about 5–10 s on typical hardware.
export const LARGE_N = 2097152;
export const LARGE_PROTOCOL = { warmup: 1, repeats: 3 };
export const protocolFor = (n) => (n >= LARGE_N ? LARGE_PROTOCOL : PROTOCOL);
export const SEED = 20261101;
// Trial 0 … warmup − 1 are the warm-ups (seeds 100 + i), trial 1 … repeats the measured ones
export const keySeed = (n, trial) => (SEED + Math.imul(n, 0x9E3779B1) + Math.imul(trial, 0x85EBCA6B)) >>> 0;

// ---- Predictions (interpret) ----------------------------------------------------------------------------
// An asymptotic form is not a slope. To test one over a finite range, evaluate the model's operation
// count at the measured N and fit the same log-log line through it: that slope is what a perfect
// measurement of a machine whose time is proportional to the count would show over these N.
//   bitonic (global)      N·S(N), S = log₂N(log₂N + 1)/2 stages, each a full pass over the keys:
//                         compare-exchanges and global memory traffic alike. Local slope 1 + 2/ln N ≈ 1.14.
//   bitonic (workgroup)   the same compare-exchanges, N·S(N) (compute-bound), but global traffic
//                         N·D(N) with D(N) = dispatches of bitonicSchedule(N, true) (traffic-bound); D grows
//                         faster than S, so the prediction is the interval between the two slopes.
//   radix                 N·8 passes: slope 1.
//   CPU sort              N log N comparisons: local slope 1 + 1/ln N ≈ 1.07.
// Only the largest FIT_POINTS sizes of a run are fitted: at small N fixed per-dispatch and submit costs
// dominate and flatten the curve (a slope below the prediction there means overhead, not a better
// algorithm). SLOPE_TOLERANCE allows for the remaining overhead and cache effects. An interval wider than
// ± MAX_HALF_WIDTH cannot tell N from N log² N (they differ by about 0.15 here), so it decides nothing.
export const FIT_POINTS = 4;
export const SLOPE_TOLERANCE = 0.05;
export const MAX_HALF_WIDTH = 0.2;
const stages = (n) => { const L = Math.log2(n); return L * (L + 1) / 2; };
export const MODELS = {
    bitonic: (n) => n * stages(n),
    sharedCompute: (n) => n * stages(n),
    sharedTraffic: (n) => n * sortInternals.bitonicSchedule(n, true).length,
    radix: (n) => n * sortInternals.RADIX_PASSES,
    cpu: (n) => n * Math.log2(n),
};
// Slope of the model's log-log line over the given N (null for fewer than 2 sizes)
export const predictedSlope = (model, ns) => fitPowerLaw(ns, ns.map(model)).slope;
// Expected order at the largest N, fastest first, on a GPU with fast workgroup memory
export const EXPECTED_RANKING = ['radix', 'bitonic-shared', 'bitonic'];

// f32 key map: edge values (±0, ±denormal min, ±normal min, ±FLT_MAX, ±∞, ±1) plus seeded random keys
export const F32_EDGES = [0, -0, 1.401298464324817e-45, -1.401298464324817e-45, 1.1754943508222875e-38,
    -1.1754943508222875e-38, 3.4028234663852886e38, -3.4028234663852886e38, Infinity, -Infinity, 1, -1, 1e6, -1e6];
export function f32KeyMapFailures(count = 65536, seed = SEED) {
    const values = Float32Array.from([...F32_EDGES, ...generateKeys(count, 'f32', seed)]);
    const bits = new Uint32Array(values.buffer);
    let roundTrip = 0;
    for (const u of bits) if (decodeF32Key(encodeF32Key(u)) !== u) roundTrip++;
    const sorted = new Uint32Array(values.slice().sort().buffer);   // Float32Array order: −0 before +0
    let order = 0;
    for (let i = 1; i < sorted.length; i++) if (encodeF32Key(sorted[i - 1]) > encodeF32Key(sorted[i])) order++;
    return { values: values.length, roundTrip, order };
}

const ms = (v) => v.toFixed(2);
const int = (v) => v.toFixed(0);
const num = (v) => typeof v === 'number' && Number.isFinite(v);
const fmtN = (n) => n.toLocaleString('en-US');
const fmtMs = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const withIqr = (row, key) => `${fmtMs(row[key])} ms (${fmtMs(row[`${key}Lo`] ?? row[key])}–${fmtMs(row[`${key}Hi`] ?? row[key])})`;
const fmtFit = (fit) => `slope ${fit.slope.toFixed(2)}${fit.ci95 ? ` (95% CI ${fit.ci95[0].toFixed(2)} … ${fit.ci95[1].toFixed(2)}, R² ${fit.r2.toFixed(3)})` : ' (no interval: 2 sizes)'}`;

// GPU-timestamp medians when every row that has a wall time also has a GPU time, else wall medians
function timeKey(rows, k) {
    const walls = rows.filter(r => num(r[`${k}Wall`]));
    return walls.length && walls.every(r => num(r[`${k}Gpu`])) ? `${k}Gpu` : `${k}Wall`;
}

// Log-log fit over the largest FIT_POINTS sizes that have `key`
export function fitTail(rows, key) {
    const pts = rows.filter(r => num(r[key]) && r[key] > 0 && num(r.n)).sort((a, b) => a.n - b.n).slice(-FIT_POINTS);
    const ns = pts.map(r => r.n);
    return { ns, fit: fitPowerLaw(ns, pts.map(r => r[key])) };
}

// Does the fit's 95% interval overlap [lo, hi] widened by tol? null without an interval
function intervalAgrees(fit, lo, hi, tol) {
    if (!fit.ci95) return null;
    return fit.ci95[1] >= lo - tol && fit.ci95[0] <= hi + tol;
}

const timing = (key) => (key.endsWith('Gpu') ? 'GPU-timestamp' : 'wall-clock');
const rangeText = (ns) => `N = ${fmtN(ns[0])} … ${fmtN(ns[ns.length - 1])}`;

function slopeNote(agrees, fit, lo, hi) {
    if (agrees === null && fit?.ci95) return `Too noisy to decide: the 95% interval is wider than ± ${MAX_HALF_WIDTH}, enough to fit both O(N) and O(N log² N). Rerun, or run to a larger N.`;
    if (agrees === null) return `Needs at least 3 sizes with a time to estimate an interval (largest ${FIT_POINTS} are fitted).`;
    if (agrees) return 'Within the interval: time grows like the operation count over this range.';
    return fit.slope < lo
        ? 'Flatter than predicted: fixed per-dispatch and submit costs are still a large share at these N, so the GPU is not saturated; a run to a larger N (or GPU timestamps) moves the slope up.'
        : `Steeper than predicted (above ${hi.toFixed(2)}): something grows faster than the work at these N, e.g. the keys no longer fitting in a cache.`;
}

function slopeClaims(rows) {
    const claims = [];
    const one = (claim, key, lo, hi, form, extra = '') => {
        const { ns, fit } = fitTail(rows, key);
        if (ns.length < 2) {
            claims.push({ claim, predicted: form, measured: '—', agrees: null, note: slopeNote(null) });
            return;
        }
        const pLo = lo(ns), pHi = hi(ns);
        const tooWide = fit.ci95 && (fit.ci95[1] - fit.ci95[0]) / 2 > MAX_HALF_WIDTH;
        const agrees = tooWide ? null
            : pLo === pHi ? slopeAgrees(fit, pLo, SLOPE_TOLERANCE) : intervalAgrees(fit, pLo, pHi, SLOPE_TOLERANCE);
        claims.push({
            claim,
            predicted: `${form}: slope ${pLo === pHi ? pLo.toFixed(2) : `${pLo.toFixed(2)} … ${pHi.toFixed(2)}`} over ${rangeText(ns)}${extra} (± ${SLOPE_TOLERANCE})`,
            measured: `${fmtFit(fit)} from ${timing(key)} medians at ${ns.length} sizes`,
            agrees,
            note: slopeNote(agrees, fit, pLo, pHi),
        });
    };
    const fixed = (model) => (ns) => predictedSlope(model, ns);
    one('Bitonic sort (global): time vs N', timeKey(rows, 'bitonic'), fixed(MODELS.bitonic), fixed(MODELS.bitonic),
        'O(N log² N), the slope of N·log₂N(log₂N+1)/2');
    one('Bitonic sort (workgroup memory): time vs N', timeKey(rows, 'shared'), fixed(MODELS.sharedCompute), fixed(MODELS.sharedTraffic),
        'O(N log² N) compare-exchanges, global traffic N × dispatches', ' (compute-bound … traffic-bound)');
    one('Radix sort: time vs N', timeKey(rows, 'radix'), fixed(MODELS.radix), fixed(MODELS.radix), 'O(N · 8 passes)');
    one('CPU sort (TypedArray.prototype.sort): time vs N', 'cpuMs', fixed(MODELS.cpu), fixed(MODELS.cpu), 'O(N log N), the slope of N·log₂N');
    return claims;
}

// −1: GPU clearly faster (its q3 below the CPU's q1), +1: clearly slower, 0: the interquartile ranges overlap
function versusCpu(row, key) {
    if (!num(row[key]) || !num(row.cpuMs)) return null;
    const g = [row[`${key}Lo`] ?? row[key], row[`${key}Hi`] ?? row[key]];
    const c = [row.cpuMsLo ?? row.cpuMs, row.cpuMsHi ?? row.cpuMs];
    return g[1] < c[0] ? -1 : g[0] > c[1] ? 1 : 0;
}

// The crossover of one GPU algorithm with the CPU sort, bracketed between sweep points
export function crossover(rows, k) {
    const key = `${k}Wall`;
    const pts = rows.filter(r => versusCpu(r, key) !== null).sort((a, b) => a.n - b.n);
    if (!pts.length) return { kind: 'no-data' };
    const states = pts.map(r => versusCpu(r, key));
    const last = states.length - 1;
    if (states.every(s => s === 1)) return { kind: 'none', at: pts[last] };
    if (states.every(s => s === -1)) return { kind: 'below', at: pts[0] };
    if (states[last] !== -1) return { kind: states[last] === 0 ? 'unresolved' : 'reversed', at: pts[last] };
    let i = last;
    while (i > 0 && states[i - 1] === -1) i--;
    return { kind: 'between', below: pts[i - 1], above: pts[i], overlap: states[i - 1] === 0 };
}

function crossoverClaims(rows) {
    return ALGORITHMS.map(a => {
        const k = KEY[a], key = `${k}Wall`;
        const c = crossover(rows, k);
        const both = (r) => `${NAME[a]} ${withIqr(r, key)} vs CPU ${withIqr(r, 'cpuMs')} at N = ${fmtN(r.n)}`;
        const measured = {
            'no-data': () => '—',
            none: () => `none in range: the CPU sort is faster at every N (${both(c.at)})`,
            below: () => `below the range: already faster at the smallest N (${both(c.at)})`,
            unresolved: () => `not resolved: the spreads overlap at the largest N (${both(c.at)})`,
            reversed: () => `not stable: faster at some N but slower at the largest (${both(c.at)})`,
            between: () => `between N = ${fmtN(c.below.n)} and ${fmtN(c.above.n)}${c.overlap ? ' (spreads overlap at the lower one)' : ''}: ${both(c.below)}; ${both(c.above)}`,
        }[c.kind]();
        return {
            claim: `Crossover: ${NAME[a]} (wall) becomes faster than the CPU sort`,
            predicted: 'device-dependent, no universal N; "none in range" is a valid outcome',
            measured,
            agrees: null,
            note: 'GPU wall time is submit → done of the sort alone (no upload or readback); a GPU is "faster" at an N only when its q3 is below the CPU sort\'s q1. '
                + 'In CI, on a CPU-emulated GPU, there is no crossover. Your GPU, driver, browser and CPU all move this point.',
        };
    });
}

function rankingClaim(rows) {
    const claim = 'Ranking at the largest N';
    const predicted = 'fastest first: radix < bitonic (workgroup memory) < bitonic (global), on a GPU with fast workgroup memory';
    const note = 'Each adjacent pair counts as decided only when the interquartile ranges do not overlap. On the CPU-emulated GPU in CI '
        + 'the order is inverted (global bitonic fastest, radix slowest): there barriers are expensive and workgroup memory is ordinary cache.';
    const full = rows.filter(r => ALGORITHMS.every(a => num(r[`${KEY[a]}Wall`]))).sort((a, b) => a.n - b.n);
    if (!full.length) return { claim, predicted, measured: '—', agrees: null, note };
    const row = full[full.length - 1];
    const gpu = ALGORITHMS.every(a => num(row[`${KEY[a]}Gpu`]));
    const keyOf = (a) => `${KEY[a]}${gpu ? 'Gpu' : 'Wall'}`;
    const lo = (a) => row[`${keyOf(a)}Lo`] ?? row[keyOf(a)], hi = (a) => row[`${keyOf(a)}Hi`] ?? row[keyOf(a)];
    const order = [...ALGORITHMS].sort((a, b) => row[keyOf(a)] - row[keyOf(b)]);
    let contradicted = false, decided = true;
    for (let i = 0; i + 1 < EXPECTED_RANKING.length; i++) {
        const fast = EXPECTED_RANKING[i], slow = EXPECTED_RANKING[i + 1];
        if (hi(fast) < lo(slow)) continue;                 // decided, as predicted
        if (hi(slow) < lo(fast)) contradicted = true;      // decided, the other way round
        else decided = false;                              // spreads overlap
    }
    return {
        claim: `${claim} (N = ${fmtN(row.n)}, ${gpu ? 'GPU timestamps' : 'wall clock'})`,
        predicted,
        measured: `${order.map(a => `${NAME[a]} ${withIqr(row, keyOf(a))}`).join(' < ')}`,
        agrees: contradicted ? false : decided ? true : null,
        note,
    };
}

const scaling = {
    id: 'scaling',
    title: 'Sorting N keys: bitonic, bitonic with workgroup memory, radix, CPU',
    question: 'How do O(n log² n) bitonic sort and O(n) radix sort scale on your GPU, and at what N do they overtake the CPU\'s built-in sort?',
    settings: [
        { key: 'dataType', label: 'Keys', options: [{ value: 'u32', label: 'u32' }, { value: 'f32', label: 'f32 (encode → sort → decode)' }], default: 'u32' },
        { key: 'maxN', label: 'Largest N', options: [262144, 1048576, 4194304].map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 4194304 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'N' },
        { key: 'repeats', label: 'repeats' },
        { key: 'bitonicGpu', label: 'bitonic GPU ms', format: ms, unit: 'ms' },
        { key: 'sharedGpu', label: 'bitonic (workgroup) GPU ms', format: ms, unit: 'ms' },
        { key: 'radixGpu', label: 'radix GPU ms', format: ms, unit: 'ms' },
        { key: 'cpuMs', label: 'CPU sort ms', format: ms, unit: 'ms' },
        { key: 'bitonicWall', label: 'bitonic wall ms', format: ms, unit: 'ms' },
        { key: 'sharedWall', label: 'bitonic (workgroup) wall ms', format: ms, unit: 'ms' },
        { key: 'radixWall', label: 'radix wall ms', format: ms, unit: 'ms' },
        { key: 'uploadMs', label: 'upload ms (radix)', format: ms, unit: 'ms' },
        { key: 'readbackMs', label: 'readback ms (radix)', format: ms, unit: 'ms' },
        { key: 'setupMs', label: 'buffers + bind groups ms (radix)', format: ms, unit: 'ms' },
        { key: 'pipelineMs', label: 'pipeline compile ms (once per run)', format: ms, unit: 'ms' },
        { key: 'radixKeysPerUs', label: 'radix M keys/s (wall)', format: int },
        { key: 'dispatches', label: 'dispatches (bitonic / workgroup / radix)', format: String },
        { key: 'bitonicWrong', label: 'bitonic keys ≠ CPU' },
        { key: 'sharedWrong', label: 'workgroup keys ≠ CPU' },
        { key: 'radixWrong', label: 'radix keys ≠ CPU' },
    ],
    charts: [
        { x: 'n', xLabel: 'keys N', logX: true, y: ['bitonicGpu', 'sharedGpu', 'radixGpu', 'cpuMs'], yLabel: 'ms per sort', logY: true,
            labels: { bitonicGpu: 'bitonic (GPU)', sharedGpu: 'bitonic, workgroup memory (GPU)', radixGpu: 'radix (GPU)', cpuMs: 'CPU sort' },
            title: 'GPU time per sort vs the CPU sort (needs timestamp-query)',
            emptyNote: 'GPU times need the timestamp-query feature; the wall-time chart below has every run.' },
        { x: 'n', xLabel: 'keys N', logX: true, y: ['bitonicWall', 'sharedWall', 'radixWall', 'cpuMs'], yLabel: 'wall ms per sort', logY: true,
            labels: { bitonicWall: 'bitonic', sharedWall: 'bitonic, workgroup memory', radixWall: 'radix', cpuMs: 'CPU sort' },
            // Slope 1 through the radix median at the largest N: where the curve bends away, overhead dominates
            reference: (rows) => {
                const pts = rows.filter(r => num(r.radixWall) && r.radixWall > 0);
                if (!pts.length) return { label: '∝ N', points: [] };
                const top = pts[pts.length - 1];
                return { label: '∝ N (through radix)', points: [pts[0].n, top.n].map(n => [n, top.radixWall * n / top.n]) };
            } },
        { x: 'n', xLabel: 'keys N', logX: true, y: ['uploadMs', 'radixWall', 'readbackMs', 'cpuMs'], yLabel: 'wall ms', logY: true,
            labels: { uploadMs: 'upload', radixWall: 'radix sort', readbackMs: 'readback', cpuMs: 'CPU sort' },
            title: 'Radix sort end to end: upload + sort + readback vs the CPU sort' },
    ],
    // CI numbers: test/gpu/parallel-sort-measure.gpu.test.mjs, runs 41356775335 and 41357331745 (software GPU, no timestamp-query)
    baseline: 'CI (shipped shaders on a CPU-emulated GPU, u32 to N = 1,048,576, f32 to 262,144, 1 warm-up + 5 trials): '
        + 'all 42 + 30 sorts per algorithm bit-identical to the CPU sort. Work per sort at N = 2²⁰: global bitonic 210 '
        + 'dispatches of 524,288 compare-exchanges (110 M in all), with workgroup memory 78 dispatches, radix 24 dispatches in '
        + '8 passes. Median wall ms (u32, bitonic / workgroup / radix / CPU sort): 14 / 15 / 31 / 0.69 at N = 16,384; '
        + '168 / 154 / 245 / 15.2 at 262,144; 541 / 660 / 887 / 67.6 at 1,048,576 (IQR within ±10% there, up to ×2.5 at '
        + 'the smallest N); a second CI run measured 714 / 837 / 1,378 / 75.8 at 1,048,576: shared machines vary by about 30% '
        + 'between runs, more than the IQR within one. Slopes over 131,072 … 1,048,576: bitonic 0.85 (0.79 … 0.91; predicted 1.15, flatter: per-dispatch '
        + 'cost), workgroup 1.00 (0.85 … 1.15), radix 0.89 (0.75 … 1.02; predicted 1.00), CPU sort 1.08 (1.06 … 1.10; predicted '
        + '1.08). No crossover: the CPU sort is about 20× faster than the fastest GPU sort at 16,384 and 8× at 1,048,576, and at 1,048,576 the ranking is inverted (global '
        + 'bitonic fastest, radix slowest) because the "GPU" is CPU cores, where barriers are expensive and workgroup memory '
        + 'is ordinary cache. Upload is 0.03–1.7 ms (the copy into staging memory), readback 0.3–17 ms. On a real GPU expect '
        + 'the workgroup-memory variant to beat the global one; whether and where radix sort beats the CPU depends on the GPU.',
    scope: {
        shows: [
            'How sort time grows with N on this device, browser and run, for three GPU algorithms and the CPU sort: medians and interquartile ranges over independent repeats, each on fresh seeded keys',
            'Whether, and between which sweep sizes, a GPU sort becomes faster than the browser\'s built-in sort here',
            'That every sort (warm-ups included) returned exactly the CPU sort\'s bits',
            'What the transfers cost: upload and readback of the keys are separate columns, not part of the sort times',
        ],
        doesNotShow: [
            'A ranking of GPUs or browsers: one device, one browser, one run',
            'Other inputs: only uniform random u32 keys or f32 keys in ±1e6; sorted, reversed, few-unique or adversarial inputs, key-value pairs and NaNs are not timed (the CI suites check them for correctness only)',
            'The best a CPU can do: the baseline is the browser\'s built-in TypedArray sort on one thread, not a parallel or radix CPU sort',
            'End-to-end GPU cost in the sort columns: they exclude upload and readback (4 bytes per key each way), which the CPU sort does not pay; the phase chart shows them next to the radix sort time. Upload is the copy into the queue\'s staging memory: the transfer to GPU memory itself may land in the sort\'s wall time',
            'Production GPU sorting: these are teaching implementations (4-bit digits, a one-workgroup scan)',
        ],
    },
    async *run(ctx, { dataType, maxN }) {
        const clock = new PhaseClock();
        const sorter = clock.time('compile', () => computeSorter(ctx.gpuDevice));
        let pipelineMs = clock.phases.compile;
        try {
            for (const n of COUNTS) {
                if (n > Math.min(maxN, MAX_N, MEASURE_MAX_KEYS)) break;
                await ctx.step();
                const wrong = { bitonic: 0, 'bitonic-shared': 0, radix: 0 };
                let dispatches = {};
                const m = await ctx.measure(async ({ warmup, index }) => {
                    const keys = generateKeys(n, dataType, keySeed(n, warmup ? 100 + index : index + 1));
                    const sorted = keys.slice();
                    const t0 = performance.now();
                    sorted.sort();
                    const trial = { cpuMs: performance.now() - t0 };
                    const expected = new Uint32Array(sorted.buffer);
                    for (const algorithm of ALGORITHMS) {
                        await ctx.step();
                        const r = await measureSort(ctx.gpuDevice, { algorithm, n, dataType, keys, expected, sorter });
                        wrong[algorithm] += r.mismatches;
                        dispatches[algorithm] = r.dispatches;
                        trial[algorithm] = { wallMs: r.wallMs, gpuMs: r.gpuMs, phases: r.phases };
                    }
                    return trial;
                }, protocolFor(n));
                const stat = (f) => summarize(m.samples.map(f));
                const row = { n, repeats: m.repeats, ...withSpread('cpuMs', stat(s => s.cpuMs)) };
                for (const a of ALGORITHMS) {
                    Object.assign(row, withSpread(`${KEY[a]}Gpu`, stat(s => s[a].gpuMs)), withSpread(`${KEY[a]}Wall`, stat(s => s[a].wallMs)));
                }
                Object.assign(row,
                    withSpread('uploadMs', stat(s => s.radix.phases.upload)),
                    withSpread('readbackMs', stat(s => s.radix.phases.readback)),
                    withSpread('setupMs', stat(s => s.radix.phases.compile)),
                    {
                        pipelineMs,
                        radixKeysPerUs: row.radixWall > 0 ? n / row.radixWall / 1000 : null,
                        dispatches: ALGORITHMS.map(a => dispatches[a]).join(' / '),
                        bitonicWrong: wrong.bitonic, sharedWrong: wrong['bitonic-shared'], radixWrong: wrong.radix,
                    });
                pipelineMs = null;   // reported once, on the first row
                yield row;
            }
        } finally {
            sorter.destroy();
        }
    },
    // Every sort of every trial bit-identical to the CPU sort, per algorithm; for f32 also the key map
    verify(rows, { dataType } = {}) {
        const typeName = dataType === 'f32' ? 'Float32Array' : 'Uint32Array';
        const sorts = rows.reduce((s, r) => s + (num(r.repeats) ? r.repeats + protocolFor(r.n).warmup : 0), 0);
        const sizes = rows.length ? `N = ${fmtN(rows[0].n)} … ${fmtN(rows[rows.length - 1].n)}` : '';
        const checks = ALGORITHMS.map(a => {
            const metric = `${NAME[a]}: keys that differ from the CPU sort`;
            if (!rows.length) return { metric, value: null, tolerance: '0', reference: 'needs 1 row', status: 'not-checked' };
            const total = rows.reduce((s, r) => s + r[`${KEY[a]}Wrong`], 0);
            return check(metric, total, { max: 0, tolerance: '0',
                reference: `${typeName}.prototype.sort of the same keys, bit for bit: ${sorts} sorts (warm-ups included) at ${sizes}` });
        });
        if (dataType === 'f32') {
            const f = f32KeyMapFailures();
            checks.push(check('f32 key map: decode(encode(u)) ≠ u', f.roundTrip, { max: 0, tolerance: '0',
                reference: `${f.values} values: ±0, ±denormals, ±FLT_MAX, ±∞ and seeded keys (the GPU round trip is covered by the sorts above)` }));
            checks.push(check('f32 key map: encoded order ≠ float order', f.order, { max: 0, tolerance: '0',
                reference: 'neighbours in Float32Array sort order (−0 before +0) must encode to non-decreasing u32 keys' }));
        } else {
            checks.push({ metric: 'f32 key map round trip', value: null, tolerance: '0', reference: 'u32 keys: no key map in this run', status: 'not-checked' });
        }
        return checks;
    },
    interpret(rows) {
        return [...slopeClaims(rows), ...crossoverClaims(rows), rankingClaim(rows)];
    },
};

export default [scaling];
