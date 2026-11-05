// Measure experiments for parallel sorting (js/examples/parallel-sort.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureSort), which runs the shipped
// benchmark (runBenchmark: upload, one sort in one compute pass, one readback, a bit-for-bit check
// against the CPU sort) on a compute-only ParallelSort whose pipelines are created once per run. The CPU
// reference and baseline is TypedArray.prototype.sort (js/refs/sort.js cpuSort is the same call).
//
// Protocol (js/teach/measure.js): one ctx.measure per N. Each call is one independent *trial*: fresh keys
// from the seeded sequence keySeed(N, trial) (reproducible, different in every trial), the CPU sort of
// copies (timed: the CPU baseline's sample, a batch of cpuReps back-to-back sorts calibrated in the warm-up
// so that it spans many ticks of a coarsened performance.now()), then each GPU algorithm sorting the same keys and checked
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
import { doublingFactor, timesText, countText } from '../measure-levels.js';

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
// Fit range: the sizes whose median time is at least FIT_MIN_MS (as in matrix-multiplication.js) and at
// least RESOLUTION_FACTOR × its clock's resolution, up to the run's largest N; at least MIN_FIT_SIZES of
// them, else the claim is undetermined. Below 1 ms fixed per-dispatch and submit costs dominate and flatten
// the curve (a slope below the prediction there means overhead, not a better algorithm), and timers
// quantize. SLOPE_TOLERANCE allows for the remaining overhead and cache effects. An interval wider than
// ± MAX_HALF_WIDTH cannot tell N from N log² N (they differ by about 0.15 here), so it decides nothing.
export const FIT_MIN_MS = 1;
export const MIN_FIT_SIZES = 3;
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

// ---- Timer resolution -----------------------------------------------------------------------------------
// Browsers coarsen performance.now() (Chrome to 0.1 ms, Safari to 1 ms) and Chrome quantizes GPU timestamps
// (to 2¹⁶ ns ≈ 0.066 ms). A time below RESOLUTION_FACTOR × its clock's resolution is not compared or fitted:
// it is at most a few ticks, so its error is 10% or more.
export const RESOLUTION_FACTOR = 10;

// performance.now() resolution: the smallest non-zero step between successive calls, over at most `ticks`
// steps or `maxMs` (null if the clock never moved). `now` is injectable for tests; works in Node too.
export function timerResolution(now = () => performance.now(), { maxMs = 50, ticks = 20, maxCalls = 1e6 } = {}) {
    const start = now();
    let prev = start, best = Infinity, seen = 0;
    for (let i = 0; i < maxCalls && seen < ticks; i++) {
        const t = now();
        if (t > prev) { best = Math.min(best, t - prev); seen++; prev = t; }
        if (t - start > maxMs) break;
    }
    return Number.isFinite(best) ? best : null;
}

// The step of a quantized series: the smallest gap between distinct values (rounded to 1 ns to merge
// float noise); null with fewer than `minDistinct` distinct values
export function granularity(values, minDistinct = 5) {
    const u = [...new Set(values.filter(num).map(v => Math.round(v * 1e6) / 1e6))].sort((a, b) => a - b);
    if (u.length < minDistinct) return null;
    let g = Infinity;
    for (let i = 1; i < u.length; i++) g = Math.min(g, u[i] - u[i - 1]);
    return g;
}

const GPU_KEYS = ['bitonic', 'shared', 'radix'];
// { wall, gpu, wallSource }: wall = the run's measured performance.now() resolution (row.timerMs), or for
// records without it the granularity of the GPU wall medians; gpu = the granularity of the timestamp medians
export function resolutions(rows) {
    const measured = rows.map(r => r.timerMs).filter(v => num(v) && v > 0);
    return {
        wall: measured.length ? Math.max(...measured) : granularity(rows.flatMap(r => GPU_KEYS.map(k => r[`${k}Wall`]))),
        wallSource: measured.length ? 'measured' : 'inferred from the wall medians',
        gpu: granularity(rows.flatMap(r => GPU_KEYS.map(k => r[`${k}Gpu`]))),
    };
}
// Is a median of `value` ms resolvable on a clock of resolution `res` (null: unknown, assume yes)? A CPU
// sample spans `reps` back-to-back sorts, so its clock reading is reps × value.
export const resolvable = (value, res, reps = 1) => res == null || value * reps >= RESOLUTION_FACTOR * res;

// Which clock to use for one GPU algorithm: GPU timestamps when every row has them and they are consistent
// with the wall clock (a pass cannot take longer on the GPU than submit → done: q1 of the GPU time above
// 1.05 × q3 of the wall time + one wall tick is a contradiction), else the wall clock.
// → { key, gpu, res, inconsistent: [N where the timestamps exceeded the wall time] }
export function clockFor(rows, k, res = resolutions(rows)) {
    const wall = { key: `${k}Wall`, gpu: false, res: res.wall, inconsistent: [] };
    const walls = rows.filter(r => num(r[`${k}Wall`]));
    if (!walls.length || !walls.every(r => num(r[`${k}Gpu`]))) return wall;
    const lo = (r, key) => r[`${key}Lo`] ?? r[key], hi = (r, key) => r[`${key}Hi`] ?? r[key];
    const inconsistent = walls.filter(r => lo(r, `${k}Gpu`) > 1.05 * hi(r, `${k}Wall`) + (res.wall ?? 0)).map(r => r.n);
    if (inconsistent.length) return { ...wall, inconsistent };
    return { key: `${k}Gpu`, gpu: true, res: res.gpu, inconsistent: [] };
}

// ---- CPU sort timing ------------------------------------------------------------------------------------
// One sort of a small array is below the resolution of a coarsened performance.now() (Safari: 1 ms), so a
// CPU sample is a batch of `reps` back-to-back sorts of separate copies of the same keys (copied before the
// clock starts), divided by reps. The warm-up trial at each N doubles reps until a batch takes at least
// cpuBatchTarget(timerMs); the measured trials use that reps. Caps: CPU_MAX_REPS sorts and
// CPU_MAX_BATCH_KEYS keys (16 MB) per batch, so a batch is bounded on any machine.
export const CPU_BATCH_MS = 10;
export const CPU_MAX_REPS = 64;
export const CPU_MAX_BATCH_KEYS = 4194304;
export const cpuBatchTarget = (timerMs) => Math.max(CPU_BATCH_MS, 2 * RESOLUTION_FACTOR * (timerMs ?? 0));
export const cpuRepsCap = (n) => Math.max(1, Math.min(CPU_MAX_REPS, Math.floor(CPU_MAX_BATCH_KEYS / n)));

// → { cpuMs (per sort), totalMs, reps, sorted (the last sorted copy) }
export function cpuSortBatch(keys, reps, now = () => performance.now()) {
    const copies = Array.from({ length: reps }, () => keys.slice());
    const t0 = now();
    for (const c of copies) c.sort();
    const totalMs = now() - t0;
    return { cpuMs: totalMs / reps, totalMs, reps, sorted: copies[reps - 1] };
}

// The warm-up's calibration: batches of 1, 2, 4, … sorts until one takes targetMs (or the cap); returns the
// last batch, whose reps the measured trials use
export function calibrateCpuBatch(keys, targetMs, now = () => performance.now()) {
    const cap = cpuRepsCap(keys.length);
    let batch = cpuSortBatch(keys, 1, now);
    while (batch.totalMs < targetMs && batch.reps < cap) batch = cpuSortBatch(keys, Math.min(cap, batch.reps * 2), now);
    return batch;
}

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

const fmtRes = (v) => (v == null ? 'unknown' : v >= 0.1 ? `${+v.toFixed(3)} ms` : `${+(v * 1000).toFixed(1)} µs`);
const fmtNs = (ns) => ns.map(fmtN).join(', ');

// Log-log fit of `key` vs N over the sizes whose median is ≥ FIT_MIN_MS and resolvable on a clock of
// resolution `res` (reps(row): sorts per clock reading, the CPU sort's batch); every size up to maxN counts
export function fitRange(rows, key, res = null, reps = () => 1) {
    const pts = rows.filter(r => num(r.n) && num(r[key]) && r[key] >= FIT_MIN_MS && resolvable(r[key], res, reps(r)))
        .sort((a, b) => a.n - b.n);
    const ns = pts.map(r => r.n);
    return { ns, fit: fitPowerLaw(ns, pts.map(r => r[key])) };
}

// Does the fit's 95% interval overlap [lo, hi] widened by tol? null without an interval
function intervalAgrees(fit, lo, hi, tol) {
    if (!fit.ci95) return null;
    return fit.ci95[1] >= lo - tol && fit.ci95[0] <= hi + tol;
}

const rangeText = (ns) => `N = ${fmtN(ns[0])} … ${fmtN(ns[ns.length - 1])}`;
const timingOf = (clock) => (clock.gpu ? 'GPU-timestamp' : 'wall-clock');
const inconsistentNote = (clock) => (clock.inconsistent.length
    ? ` GPU timestamps not used: at N = ${fmtNs(clock.inconsistent)} they exceed the wall time of the same sorts (submit → done), which a pass cannot do, so the wall clock is used.`
    : '');
const fitWindow = `the sizes whose median is at least ${FIT_MIN_MS} ms and ${RESOLUTION_FACTOR}× the timer resolution`;

function slopeNote(agrees, fit, lo, hi) {
    if (agrees === null) return `Too noisy to decide: the 95% interval is wider than ± ${MAX_HALF_WIDTH}, enough to fit both O(N) and O(N log² N). Rerun, or run to a larger N.`;
    if (agrees) return 'Within the interval: time grows like the operation count over this range.';
    return fit.slope < lo
        ? 'Flatter than predicted: fixed per-dispatch and submit costs are still a large share at these N, so the GPU is not saturated; a run to a larger N (or GPU timestamps) moves the slope up.'
        : `Steeper than predicted (above ${hi.toFixed(2)}): something grows faster than the work at these N, e.g. the keys no longer fitting in a cache.`;
}

// Names and reasons in the plain-language claims (the Beginner and Intermediate levels, js/teach/measure-levels.js)
const PLAIN_NAME = { bitonic: 'Bitonic sort', shared: 'Bitonic sort with shared memory', radix: 'Radix sort', cpu: 'The browser\'s sort' };
const PLAIN_WHY = {
    bitonic: 'its plan gets a few more steps each time the list doubles',
    shared: 'its plan gets a few more steps each time the list doubles',
    radix: 'it does the same 8 rounds for any amount of numbers',
    cpu: 'it does a little more work per number on a longer list',
};
// "about 2.2 times" / "about 2.2 to 2.3 times" for a predicted slope (range); twice the keys take 2^slope as long
function plainPredicted(k, pLo, pHi) {
    const a = timesText(doublingFactor(pLo)), b = timesText(doublingFactor(pHi));
    const factor = a === b ? (a === '2.0' ? 'twice as long' : `about ${a} times as long`) : `about ${a} to ${b} times as long`;
    return `Twice the numbers take ${factor}: ${PLAIN_WHY[k]}`;
}

function slopeClaims(rows, res) {
    const claims = [];
    const one = (id, claim, k, lo, hi, form, extra = '') => {
        const clock = k === 'cpu' ? { key: 'cpuMs', gpu: false, res: res.wall, inconsistent: [] } : clockFor(rows, k, res);
        const reps = k === 'cpu' ? (r) => (num(r.cpuReps) ? r.cpuReps : 1) : () => 1;
        const { ns, fit } = fitRange(rows, clock.key, clock.res, reps);
        const window = `Fitted over ${fitWindow} (${timingOf(clock)} resolution ${fmtRes(clock.res)}), up to the largest N of the run.`;
        if (ns.length < MIN_FIT_SIZES) {
            claims.push({
                key: id, claim, predicted: form,
                measured: ns.length >= 2 ? `${fmtFit(fit)} from ${timingOf(clock)} medians at ${ns.length} sizes (${rangeText(ns)})`
                    : `${ns.length} size(s) with a ${timingOf(clock)} median ≥ ${FIT_MIN_MS} ms`,
                agrees: null,
                plain: {
                    claim: `${PLAIN_NAME[k]}: twice the numbers`,
                    predicted: plainPredicted(k, lo(COUNTS), hi(COUNTS)),
                    measured: 'Too few sizes took long enough to time well',
                },
                note: `Undetermined: needs at least ${MIN_FIT_SIZES} sizes with a median of at least ${FIT_MIN_MS} ms (and ${RESOLUTION_FACTOR}× the timer resolution), ${ns.length} here; run to a larger N. ${window}${inconsistentNote(clock)}`,
            });
            return;
        }
        const pLo = lo(ns), pHi = hi(ns);
        const tooWide = (fit.ci95[1] - fit.ci95[0]) / 2 > MAX_HALF_WIDTH;
        const agrees = tooWide ? null
            : pLo === pHi ? slopeAgrees(fit, pLo, SLOPE_TOLERANCE) : intervalAgrees(fit, pLo, pHi, SLOPE_TOLERANCE);
        claims.push({
            key: id,
            claim,
            predicted: `${form}: slope ${pLo === pHi ? pLo.toFixed(2) : `${pLo.toFixed(2)} … ${pHi.toFixed(2)}`} over ${rangeText(ns)}${extra} (± ${SLOPE_TOLERANCE})`,
            measured: `${fmtFit(fit)} from ${timingOf(clock)} medians at ${ns.length} sizes`,
            agrees,
            plain: {
                claim: `${PLAIN_NAME[k]}: twice the numbers`,
                predicted: plainPredicted(k, pLo, pHi),
                measured: agrees === null
                    ? `About ${timesText(doublingFactor(fit.slope))} times as long, but the times were too uneven to be sure`
                    : `${timesText(doublingFactor(fit.slope))} times as long, from ${countText(ns[0])} to ${countText(ns.at(-1))} numbers`,
            },
            note: `${slopeNote(agrees, fit, pLo, pHi)} ${window}${inconsistentNote(clock)}`,
        });
    };
    const fixed = (model) => (ns) => predictedSlope(model, ns);
    one('bitonic-slope', 'Bitonic sort (global): time vs N', 'bitonic', fixed(MODELS.bitonic), fixed(MODELS.bitonic),
        'O(N log² N), the slope of N·log₂N(log₂N+1)/2');
    one('shared-slope', 'Bitonic sort (workgroup memory): time vs N', 'shared', fixed(MODELS.sharedCompute), fixed(MODELS.sharedTraffic),
        'O(N log² N) compare-exchanges, global traffic N × dispatches', ' (compute-bound … traffic-bound)');
    one('radix-slope', 'Radix sort: time vs N', 'radix', fixed(MODELS.radix), fixed(MODELS.radix), 'O(N · 8 passes)');
    one('cpu-slope', 'CPU sort (TypedArray.prototype.sort): time vs N', 'cpu', fixed(MODELS.cpu), fixed(MODELS.cpu), 'O(N log N), the slope of N·log₂N');
    return claims;
}

// −1: GPU clearly faster (its q3 below the CPU's q1), +1: clearly slower, 0: the interquartile ranges overlap,
// 'res': a median below RESOLUTION_FACTOR × its clock's resolution (not compared), null: no data
function versusCpu(row, clock, res) {
    const key = clock.key;
    if (!num(row[key]) || !num(row.cpuMs)) return null;
    if (!resolvable(row[key], clock.res) || !resolvable(row.cpuMs, res.wall, num(row.cpuReps) ? row.cpuReps : 1)) return 'res';
    const g = [row[`${key}Lo`] ?? row[key], row[`${key}Hi`] ?? row[key]];
    const c = [row.cpuMsLo ?? row.cpuMs, row.cpuMsHi ?? row.cpuMs];
    return g[1] < c[0] ? -1 : g[0] > c[1] ? 1 : 0;
}

// The crossover of one GPU algorithm with the CPU sort, bracketed between sweep points, on GPU timestamps
// when they are available and consistent (clockFor), else wall times. Sizes where a time is below timer
// resolution are skipped (`skipped`); when every size is, the kind is 'below-resolution'.
export function crossover(rows, k) {
    const res = resolutions(rows), clock = clockFor(rows, k, res);
    const all = rows.map(r => ({ r, s: versusCpu(r, clock, res) })).filter(p => p.s !== null).sort((a, b) => a.r.n - b.r.n);
    const skipped = all.filter(p => p.s === 'res').map(p => p.r.n);
    const pts = all.filter(p => p.s !== 'res').map(p => p.r);
    const base = { clock, res, skipped };
    if (!all.length) return { kind: 'no-data', ...base };
    if (!pts.length) return { kind: 'below-resolution', at: all[all.length - 1].r, ...base };
    const states = pts.map(r => versusCpu(r, clock, res));
    const last = states.length - 1;
    if (states.every(s => s === 1)) return { kind: 'none', at: pts[last], ...base };
    if (states.every(s => s === -1)) return { kind: 'below', at: pts[0], undeterminedBelow: skipped.some(n => n < pts[0].n), ...base };
    if (states[last] !== -1) return { kind: states[last] === 0 ? 'unresolved' : 'reversed', at: pts[last], ...base };
    let i = last;
    while (i > 0 && states[i - 1] === -1) i--;
    return { kind: 'between', below: pts[i - 1], above: pts[i], overlap: states[i - 1] === 0, ...base };
}

function crossoverClaims(rows, res) {
    return ALGORITHMS.map(a => {
        const k = KEY[a];
        const c = crossover(rows, k);
        const key = c.clock.key;
        const both = (r) => `${NAME[a]} ${withIqr(r, key)} vs CPU ${withIqr(r, 'cpuMs')} at N = ${fmtN(r.n)}`;
        const measured = {
            'no-data': () => '—',
            'below-resolution': () => `undetermined: below timer resolution at every N (${both(c.at)})`,
            none: () => `none in range: the CPU sort is faster at every compared N (${both(c.at)})`,
            below: () => (c.undeterminedBelow
                ? `faster from the smallest N with resolvable times on (${both(c.at)}); below that N a time is under ${RESOLUTION_FACTOR}× the timer resolution, so whether the crossover lies lower is undetermined`
                : `below the range: already faster at the smallest N (${both(c.at)})`),
            unresolved: () => `not resolved: the spreads overlap at the largest N (${both(c.at)})`,
            reversed: () => `not stable: faster at some N but slower at the largest (${both(c.at)})`,
            between: () => `between N = ${fmtN(c.below.n)} and ${fmtN(c.above.n)}${c.overlap ? ' (spreads overlap at the lower one)' : ''}: ${both(c.below)}; ${both(c.above)}`,
        }[c.kind]();
        const skipped = c.skipped.length && c.kind !== 'below-resolution'
            ? `; not compared (a time below ${RESOLUTION_FACTOR}× the timer resolution): N = ${fmtNs(c.skipped)}` : '';
        const name = PLAIN_NAME[k];
        const plainMeasured = {
            'no-data': () => null,
            'below-resolution': () => 'Every sort was too quick to time well with this browser\'s clock',
            none: () => `The browser's sort was faster at every size tried, up to ${countText(c.at.n)} numbers`,
            below: () => `${name} was faster at every size that could be timed, from ${countText(c.at.n)} numbers on`,
            unresolved: () => `No clear winner at ${countText(c.at.n)} numbers: the times overlap`,
            reversed: () => `${name} was faster at some sizes, but not at the largest`,
            between: () => `${name} was faster from between ${countText(c.below.n)} and ${countText(c.above.n)} numbers on`,
        }[c.kind]();
        return {
            key: `crossover-${k}`,
            plain: plainMeasured ? {
                claim: `${name} vs the browser's sort`,
                predicted: 'It depends on your GPU: the browser\'s sort may even win at every size',
                measured: plainMeasured,
            } : undefined,
            claim: `Crossover: ${NAME[a]} (${c.clock.gpu ? 'GPU timestamps' : 'wall'}) becomes faster than the CPU sort`,
            predicted: 'device-dependent, no universal N; "none in range" is a valid outcome',
            measured: measured + skipped,
            agrees: null,
            note: 'The GPU time is the sort pass alone (no upload or readback): GPU timestamps when the browser has timestamp-query and they are consistent with the wall clock, else submit → done wall time. '
                + 'The CPU sort is wall-clock; each sample is a batch of back-to-back sorts long enough to resolve. A GPU is "faster" at an N only when its q3 is below the CPU sort\'s q1, '
                + `and an N is compared only when both medians are at least ${RESOLUTION_FACTOR}× their clock's resolution (performance.now(): ${fmtRes(res.wall)}, ${res.wallSource}; GPU timestamps: ${fmtRes(res.gpu)}, inferred from the medians). `
                + 'In CI, on a CPU-emulated GPU, there is no crossover. Your GPU, driver, browser and CPU all move this point.'
                + inconsistentNote(c.clock),
        };
    });
}

function rankingClaim(rows, res) {
    const claim = 'Ranking at the largest N';
    const predicted = 'fastest first: radix < bitonic (workgroup memory) < bitonic (global), on a GPU with fast workgroup memory';
    const clocks = ALGORITHMS.map(a => clockFor(rows, KEY[a], res));
    const gpu = clocks.every(c => c.gpu);
    const tick = (gpu ? res.gpu : res.wall) ?? 0;
    const note = 'A prediction for GPUs with fast workgroup memory, not a law: other GPUs and drivers may order them differently. '
        + 'Each adjacent pair counts as decided only when the interquartile ranges are more than one clock tick apart; '
        + 'GPU timestamps are used only when every algorithm\'s are consistent with the wall clock. On the CPU-emulated GPU in CI the order is inverted '
        + '(global bitonic fastest, radix slowest): there barriers are expensive and workgroup memory is ordinary cache.'
        + (clocks.map(inconsistentNote).find(Boolean) ?? '');
    const full = rows.filter(r => ALGORITHMS.every(a => num(r[`${KEY[a]}Wall`]))).sort((a, b) => a.n - b.n);
    if (!full.length) return { key: 'ranking', claim, predicted, measured: '—', agrees: null, note };
    const row = full[full.length - 1];
    const keyOf = (a) => `${KEY[a]}${gpu ? 'Gpu' : 'Wall'}`;
    const lo = (a) => row[`${keyOf(a)}Lo`] ?? row[keyOf(a)], hi = (a) => row[`${keyOf(a)}Hi`] ?? row[keyOf(a)];
    const order = [...ALGORITHMS].sort((a, b) => row[keyOf(a)] - row[keyOf(b)]);
    let contradicted = false, decided = true;
    for (let i = 0; i + 1 < EXPECTED_RANKING.length; i++) {
        const fast = EXPECTED_RANKING[i], slow = EXPECTED_RANKING[i + 1];
        if (hi(fast) + tick < lo(slow)) continue;                 // decided, as predicted
        if (hi(slow) + tick < lo(fast)) contradicted = true;      // decided, the other way round
        else decided = false;                                     // spreads overlap
    }
    const plainMs = (v) => (v >= 10 ? countText(v) : v.toFixed(1));
    return {
        key: 'ranking',
        plain: {
            claim: `Fastest to slowest, for ${countText(row.n)} numbers`,
            predicted: 'Radix sort, then bitonic sort with shared memory, then bitonic sort',
            measured: order.map((a, i) => `${i ? PLAIN_NAME[KEY[a]].toLowerCase() : PLAIN_NAME[KEY[a]]} ${plainMs(row[keyOf(a)])} ms`).join(', then '),
        },
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
        { key: 'cpuReps', label: 'CPU sorts per sample' },
        { key: 'bitonicWall', label: 'bitonic wall ms', format: ms, unit: 'ms' },
        { key: 'sharedWall', label: 'bitonic (workgroup) wall ms', format: ms, unit: 'ms' },
        { key: 'radixWall', label: 'radix wall ms', format: ms, unit: 'ms' },
        { key: 'uploadMs', label: 'upload ms (radix)', format: ms, unit: 'ms' },
        { key: 'readbackMs', label: 'readback ms (radix)', format: ms, unit: 'ms' },
        { key: 'setupMs', label: 'buffers + bind groups ms (radix)', format: ms, unit: 'ms' },
        { key: 'pipelineMs', label: 'pipeline compile ms (once per run)', format: ms, unit: 'ms' },
        { key: 'timerMs', label: 'performance.now() resolution ms', format: (v) => (v == null ? '—' : String(+v.toFixed(4))), unit: 'ms' },
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
        const timerMs = timerResolution();   // performance.now() resolution, recorded in every row
        const clock = new PhaseClock();
        const sorter = clock.time('compile', () => computeSorter(ctx.gpuDevice));
        let pipelineMs = clock.phases.compile;
        try {
            for (const n of COUNTS) {
                if (n > Math.min(maxN, MAX_N, MEASURE_MAX_KEYS)) break;
                await ctx.step();
                const wrong = { bitonic: 0, 'bitonic-shared': 0, radix: 0 };
                let dispatches = {};
                let cpuReps = 1;
                const m = await ctx.measure(async ({ warmup, index }) => {
                    const keys = generateKeys(n, dataType, keySeed(n, warmup ? 100 + index : index + 1));
                    // The warm-up calibrates the CPU batch (CPU sort timing, above); measured trials reuse it
                    const batch = warmup ? calibrateCpuBatch(keys, cpuBatchTarget(timerMs)) : cpuSortBatch(keys, cpuReps);
                    if (warmup) cpuReps = batch.reps;
                    const trial = { cpuMs: batch.cpuMs };
                    const expected = new Uint32Array(batch.sorted.buffer);
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
                const row = { n, repeats: m.repeats, ...withSpread('cpuMs', stat(s => s.cpuMs)), cpuReps, timerMs };
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
        const res = resolutions(rows);
        return [...slopeClaims(rows, res), ...crossoverClaims(rows, res), rankingClaim(rows, res)];
    },
};

export default [scaling];
