// Measure experiments for matrix multiplication (js/examples/matrix-multiplication.js). Interface: see the
// top of js/teach/measure.js.
//
// One experiment, three implementations of C = AB on the same matrices (uniform in [0, 1), seeded):
//   naive GPU  the shipped one-thread-per-element kernel (`compute`), through measureMatmul
//   tiled GPU  the 16 × 16 workgroup-memory kernel (`tiled`), through measureMatmul
//   CPU        matmulBlockedF32 from js/refs/basic.js: cache-blocked, single-threaded JavaScript, f32 storage
// and one independent reference, the float64 product of js/refs/basic.js (matmulRowsF64) on SAMPLE_ROWS evenly
// spaced rows (O(rows · n²), so n = 1,024 stays cheap).
//
// Caps (exported so test/matrix-multiplication-measure.test.mjs can check them): GPU n ≤ 1,024 (2.1 GFLOP
// per product), CPU n ≤ 512 (0.27 GFLOP: about 0.1 s at the 2.4 GFLOPS one core of a 2023 laptop reaches,
// and still under 1.5 s at 0.2 GFLOPS). Protocol for every path: 1 discarded warm-up, then 5 independent
// repeats (GPU: each a fresh measureMatmul call that creates, uploads, dispatches once and reads back).
// Timed columns are medians with their interquartile range; errors are deterministic and come from the
// last repeat.

import { measureMatmul, MAX_MATMUL_DIM, TILED_T } from '../../examples/matrix-multiplication.js';
import { matmulRowsF64, matmulBlockedF32 } from '../../refs/basic.js';
import { mulberry32 } from '../../refs/random.js';
import { summarize, withSpread, fitPowerLaw, slopeAgrees } from '../stats.js';
import { check } from '../result-record.js';

export const SIZES = [64, 128, 256, 512, 1024];
export const MAX_SIZE = 1024;            // GPU cap (also the example's MAX_MATMUL_DIM)
export const CPU_MAX_N = 512;            // CPU baseline cap: one product stays well under 2 s
export const SAMPLE_ROWS = 16;
export const SEED = 7;
export const UNIT_ROUNDOFF = 2 ** -24;   // f32
export const PROTOCOL = { warmup: 1, repeats: 5 };
// Accuracy bound for verify(): max relative error over the sampled rows ≤ ERR_FACTOR · √n · u. Justification:
// each entry is a sum of n rounded positive terms; the deterministic worst case is n·u, the typical error
// √n·u (Higham and Mary, 2019), and CI measured 0.7–0.9 √n·u for the naive kernel at every n. 4 √n·u leaves
// a factor 4–5 of headroom (a different FMA contraction or summation order) and is still below n·u for
// n > 16; a kernel that drops or repeats a 16-wide tile is wrong by ≥ 16/n ≥ 1.6% and fails by 10⁴×.
export const ERR_FACTOR = 4;
// interpret(): power laws are fitted over the sizes whose median time is at least FIT_MIN_MS. Below that a
// measurement is launch-, latency- or timer-bound (Chrome quantizes GPU timestamps and performance.now() to
// ~0.1 ms), so its time does not grow like n³ yet.
export const FIT_MIN_MS = 1;
export const SLOPE_TOLERANCE = 0.15;     // n³ is asymptotic: allow for cache effects at the fit's ends

export function makeMatrix(rows, cols, seed) {
    const rand = mulberry32(seed);
    return Float32Array.from({ length: rows * cols }, () => rand());
}

export function sampleRows(n, count = SAMPLE_ROWS) {
    const m = Math.min(count, n);
    return Array.from({ length: m }, (_, k) => Math.floor((k + 0.5) * n / m));
}

// Largest |C − C64| / |C64| over the sampled rows (all entries are positive sums, so no cancellation)
export function maxRelError(C, n, rows, ref) {
    let worst = 0;
    rows.forEach((i, r) => {
        for (let j = 0; j < n; j++) {
            const exact = ref[r * n + j];
            worst = Math.max(worst, Math.abs(C[i * n + j] - exact) / Math.abs(exact));
        }
    });
    return worst;
}

// Largest |C1 − C2| / |C64| over the sampled rows
export function maxRelDiff(C1, C2, n, rows, ref) {
    let worst = 0;
    rows.forEach((i, r) => {
        for (let j = 0; j < n; j++) worst = Math.max(worst, Math.abs(C1[i * n + j] - C2[i * n + j]) / Math.abs(ref[r * n + j]));
    });
    return worst;
}

export const sqrtNu = (n) => Math.sqrt(n) * UNIT_ROUNDOFF;

const fmt = (digits) => (v) => v.toFixed(digits);
const ms = (v) => (v < 10 ? v.toFixed(3) : v.toFixed(1));
const sci = (v) => v.toExponential(2);
const gf = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1));
const PATHS = ['cpu', 'naive', 'tiled'];
const LABEL = { cpu: 'CPU (blocked JS)', naive: 'naive GPU', tiled: 'tiled GPU' };
const PHASE_KEYS = ['Compile', 'Upload', 'Dispatch', 'Readback'];

// The kernel time of one GPU sample: timestamps when the device has them, else submit → done wall time
const kernelMs = (useGpu) => (s) => (useGpu ? s.gpuMs : s.wallMs);
const gflopsOf = (flops) => (t) => (t > 0 ? flops / t / 1e6 : null);

// Row keys for one path from its ctx.measure result: <path>Ms and <path>Gflops with spread
function timedColumns(path, n, m, timeOf) {
    const times = m.samples.map(timeOf);
    return {
        ...withSpread(`${path}Ms`, summarize(times)),
        ...withSpread(`${path}Gflops`, summarize(times.map(gflopsOf(2 * n ** 3)))),
    };
}

const emptyPath = (path) => ({ ...withSpread(`${path}Ms`, null), ...withSpread(`${path}Gflops`, null), [`${path}Err`]: null });

// ---- interpret() helpers --------------------------------------------------------------------------------
const rangeText = (row, key, f) => (row[`${key}Lo`] != null && row[`${key}Hi`] != null
    ? `${f(row[key])} (${f(row[`${key}Lo`])}–${f(row[`${key}Hi`])})` : f(row[key]));

// Fit of <path>Ms vs n over the rows with a median ≥ FIT_MIN_MS; the fit over all rows for contrast
export function fitTimes(rows, path) {
    const key = `${path}Ms`;
    const all = rows.filter(r => r[key] > 0);
    const used = all.filter(r => r[key] >= FIT_MIN_MS);
    return {
        used: used.map(r => r.n),
        fit: fitPowerLaw(used.map(r => r.n), used.map(r => r[key])),
        allFit: fitPowerLaw(all.map(r => r.n), all.map(r => r[key])),
    };
}

function slopeClaim(rows, path, timing) {
    const { used, fit, allFit } = fitTimes(rows, path);
    const window = used.length ? `n = ${used[0]} … ${used.at(-1)}` : 'no sizes';
    const measured = fit.slope === null ? `${used.length} size(s) with median ≥ ${FIT_MIN_MS} ms`
        : fit.ci95 ? `${fit.slope.toFixed(2)} (95% CI ${fit.ci95[0].toFixed(2)} … ${fit.ci95[1].toFixed(2)}) over ${window}`
            : `${fit.slope.toFixed(2)} (no interval from 2 points) over ${window}`;
    const notes = [`Fitted over the sizes whose median ${timing} is at least ${FIT_MIN_MS} ms (${window}); smaller products are `
        + 'launch-, latency- or timer-bound, so their times do not grow like n³ yet.'];
    if (allFit.slope !== null) notes.push(`Over all ${allFit.n} sizes the slope is ${allFit.slope.toFixed(2)}.`);
    if (!fit.ci95) notes.push('At least 3 sizes are needed for an uncertainty; with fewer, this claim stays undecided.');
    return {
        claim: `${LABEL[path]}: time grows like n³`,
        predicted: `log-log slope 3 (2n³ flops; ±${SLOPE_TOLERANCE} for cache effects)`,
        measured,
        agrees: slopeAgrees(fit, 3, SLOPE_TOLERANCE),
        note: notes.join(' '),
    };
}

const largest = (rows, ...keys) => rows.filter(r => keys.every(k => r[k] > 0)).at(-1);

// ---- the experiment ---------------------------------------------------------------------------------------
const threeWay = {
    id: 'cpu-naive-tiled',
    title: 'CPU vs naive GPU vs tiled GPU',
    question: 'How much do workgroup-memory tiling and the GPU itself buy over a single CPU core for C = AB as n grows, and do all three get the same answer?',
    settings: [
        { key: 'maxN', label: 'Largest n', options: [256, 512, 1024].map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 1024 },
    ],
    budgetMs: 20000,
    scope: {
        shows: [
            'How the time of C = AB grows with n on this device for a naive and a tiled WGSL kernel and a blocked single-threaded JavaScript loop, on the same matrices.',
            'How much workgroup-memory tiling buys over the naive kernel on your GPU, and where the time of a small product goes (compile, upload, dispatch, readback).',
            `That all three products agree with a float64 reference within ${ERR_FACTOR}·√n·u on ${SAMPLE_ROWS} sampled rows per size.`,
        ],
        doesNotShow: [
            'Not a GPU ranking: one device, one browser, one run.',
            'The CPU baseline is single-threaded JavaScript, not an optimized BLAS: a tuned multithreaded BLAS on the same CPU is typically 10–100× faster.',
            'Not the hardware\'s peak: the tiled kernel has no register blocking or vectorized loads, and vendor libraries go several times further.',
            'On software GPUs (as in CI) the ranking inverts: the "GPU" is CPU cores, where workgroup memory and barriers cost more than they save.',
            'Kernel times exclude transfers: whether offloading one product pays also depends on upload and readback (see the phases).',
        ],
    },
    columns: [
        { key: 'n', label: 'n (n × n · n × n)' },
        { key: 'timing', label: 'GPU time from', format: String },
        { key: 'cpuGflops', label: 'CPU GFLOPS', format: gf, unit: 'GFLOP/s' },
        { key: 'naiveGflops', label: 'naive GPU GFLOPS', format: gf, unit: 'GFLOP/s' },
        { key: 'tiledGflops', label: 'tiled GPU GFLOPS', format: gf, unit: 'GFLOP/s' },
        { key: 'cpuMs', label: 'CPU ms', format: ms, unit: 'ms' },
        { key: 'naiveMs', label: 'naive GPU ms', format: ms, unit: 'ms' },
        { key: 'tiledMs', label: 'tiled GPU ms', format: ms, unit: 'ms' },
        { key: 'speedup', label: 'tiled vs naive', format: (v) => `${v.toFixed(2)}×` },
        { key: 'cpuErr', label: 'CPU max rel. error', format: sci },
        { key: 'naiveErr', label: 'naive max rel. error', format: sci },
        { key: 'tiledErr', label: 'tiled max rel. error', format: sci },
        { key: 'naiveVsTiled', label: 'max |naive − tiled| / |C|', format: sci },
        { key: 'sqrtNu', label: '√n·u', format: sci },
        ...['naive', 'tiled'].flatMap(path => PHASE_KEYS.map(p => (
            { key: `${path}${p}`, label: `${path} ${p.toLowerCase()} ms`, format: ms, unit: 'ms' }))),
    ],
    charts: [
        { x: 'n', xLabel: 'matrix size n', logX: true, y: ['cpuGflops', 'naiveGflops', 'tiledGflops'], yLabel: 'GFLOPS (2n³ per product)', logY: true,
            labels: { cpuGflops: LABEL.cpu, naiveGflops: LABEL.naive, tiledGflops: LABEL.tiled },
            title: 'Throughput vs n (GPU: kernel time)' },
        { x: 'n', xLabel: 'matrix size n', logX: true, y: ['cpuMs', 'naiveMs', 'tiledMs'], yLabel: 'ms per product', logY: true,
            labels: { cpuMs: LABEL.cpu, naiveMs: LABEL.naive, tiledMs: LABEL.tiled },
            title: 'Time vs n, with an n³ line',
            // n³ through the tiled kernel's largest time: where a series bends away from its slope, overhead dominates
            reference: (rows) => {
                const last = largest(rows, 'tiledMs');
                return last ? { label: 'n³ (through tiled GPU, largest n)', points: [SIZES[0], last.n].map(n => [n, last.tiledMs * (n / last.n) ** 3]) }
                    : { label: 'n³', points: [] };
            } },
        { x: 'n', xLabel: 'matrix size n', logX: true, y: PHASE_KEYS.map(p => `tiled${p}`), yLabel: 'wall ms per phase', logY: true,
            labels: Object.fromEntries(PHASE_KEYS.map(p => [`tiled${p}`, p.toLowerCase()])),
            title: 'Tiled GPU: wall-clock phases vs n' },
        { x: 'n', xLabel: 'matrix size n', logX: true, y: PHASE_KEYS.map(p => `naive${p}`), yLabel: 'wall ms per phase', logY: true,
            labels: Object.fromEntries(PHASE_KEYS.map(p => [`naive${p}`, p.toLowerCase()])),
            title: 'Naive GPU: wall-clock phases vs n' },
        { x: 'n', xLabel: 'matrix size n', logX: true, y: ['cpuErr', 'naiveErr', 'tiledErr'], yLabel: 'max relative error vs float64', logY: true,
            labels: { cpuErr: LABEL.cpu, naiveErr: LABEL.naive, tiledErr: LABEL.tiled },
            reference: { label: '√n·u (typical)', points: [SIZES[0], SIZES.at(-1)].map(n => [n, sqrtNu(n)]) } },
    ],
    baseline: 'CI (software GPU, lavapipe; not representative of any real GPU): see the numbers on the Understand page. '
        + 'There the CPU baseline wins at every size, because the "GPU" is the CI machine\'s CPU cores. On a real GPU expect '
        + 'the tiled kernel to beat the naive one by a few times at n ≥ 512 and both to beat one CPU core from a few hundred on.',
    async *run(ctx, { maxN = MAX_SIZE } = {}) {
        const useGpu = ctx.timestamps === true;
        const timing = useGpu ? 'timestamps' : 'wall (submit → done)';
        for (const n of SIZES) {
            if (n > Math.min(maxN, MAX_SIZE, MAX_MATMUL_DIM)) break;
            const A = makeMatrix(n, n, SEED), B = makeMatrix(n, n, SEED + 1);
            await ctx.step();
            const rows = sampleRows(n);
            const ref = matmulRowsF64(A, B, n, n, n, rows);
            const row = { n, timing };
            const C = {};
            for (const kernel of ['naive', 'tiled']) {
                await ctx.step();
                const m = await ctx.measure(() => measureMatmul(ctx.gpuDevice, { A, B, M: n, N: n, K: n, kernel, warmup: 0, repeats: 1 }), PROTOCOL);
                C[kernel] = m.samples.at(-1).C;
                for (const s of m.samples.slice(0, -1)) delete s.C;
                Object.assign(row, timedColumns(kernel, n, m, kernelMs(useGpu)), { [`${kernel}Err`]: maxRelError(C[kernel], n, rows, ref) });
                for (const p of PHASE_KEYS) Object.assign(row, withSpread(`${kernel}${p}`, m.phases[p.toLowerCase()]));
            }
            if (n <= CPU_MAX_N) {
                await ctx.step();
                const out = new Float32Array(n * n);
                const m = await ctx.measure(async () => {
                    const t0 = performance.now();
                    matmulBlockedF32(A, B, n, n, n, { out });
                    return { wallMs: performance.now() - t0 };
                }, PROTOCOL);
                Object.assign(row, timedColumns('cpu', n, m, (s) => s.wallMs), { cpuErr: maxRelError(out, n, rows, ref) });
            } else {
                Object.assign(row, emptyPath('cpu'));
            }
            const nv = row.naiveMs, tl = row.tiledMs;
            Object.assign(row, {
                speedup: nv > 0 && tl > 0 ? nv / tl : null,
                // conservative range of the ratio of medians: naive q1 / tiled q3 … naive q3 / tiled q1
                speedupLo: row.naiveMsLo > 0 && row.tiledMsHi > 0 ? row.naiveMsLo / row.tiledMsHi : null,
                speedupHi: row.naiveMsHi > 0 && row.tiledMsLo > 0 ? row.naiveMsHi / row.tiledMsLo : null,
                naiveVsTiled: maxRelDiff(C.naive, C.tiled, n, rows, ref),
                sqrtNu: sqrtNu(n),
            });
            yield Object.fromEntries(threeWay.columns.flatMap(c => [c.key, `${c.key}Lo`, `${c.key}Hi`]).filter(k => k in row).map(k => [k, row[k]]));
        }
    },
    // Each path within ERR_FACTOR·√n·u of float64 at every measured n; naive and tiled within the same bound of
    // each other (they sum in the same order, so CI measures them identical)
    verify(rows) {
        const worst = (key) => {
            const r = rows.filter(x => typeof x[key] === 'number');
            return r.length ? Math.max(...r.map(x => x[key] / sqrtNu(x.n))) : null;
        };
        const bound = { max: ERR_FACTOR, tolerance: `<= ${ERR_FACTOR} (error ≤ ${ERR_FACTOR}·√n·u)` };
        const sizes = (key) => rows.filter(x => typeof x[key] === 'number').map(x => x.n).join(', ') || 'none';
        return [
            check('naive GPU: max over n of (max rel. error) / (√n·u)', worst('naiveErr'),
                { ...bound, reference: `float64 product of ${SAMPLE_ROWS} sampled rows at n = ${sizes('naiveErr')}` }),
            check('tiled GPU: max over n of (max rel. error) / (√n·u)', worst('tiledErr'),
                { ...bound, reference: `float64 product of ${SAMPLE_ROWS} sampled rows at n = ${sizes('tiledErr')}` }),
            check('CPU baseline: max over n of (max rel. error) / (√n·u)', worst('cpuErr'),
                { ...bound, reference: `float64 product of ${SAMPLE_ROWS} sampled rows at n = ${sizes('cpuErr')}` }),
            check('naive vs tiled: max over n of max |naive − tiled| / |C| / (√n·u)', worst('naiveVsTiled'),
                { ...bound, reference: 'the two GPU kernels on the same inputs (same summation order: expected 0 without FMA differences)' }),
        ];
    },
    interpret(rows) {
        const timing = rows[0]?.timing === 'timestamps' ? 'GPU (timestamp) time' : 'time';
        const claims = [slopeClaim(rows, 'naive', timing), slopeClaim(rows, 'tiled', timing), slopeClaim(rows, 'cpu', 'time')];

        // Tiled vs naive at the largest n
        const big = largest(rows, 'naiveMs', 'tiledMs');
        claims.push({
            claim: 'Tiling speeds the GPU kernel up',
            predicted: `tiled faster (speedup > 1): ${TILED_T}× fewer global loads (arithmetic intensity ${TILED_T / 4} vs 0.25 flop/byte); typically 2–10× on real GPUs`,
            measured: big ? `${rangeText(big, 'speedup', v => `${v.toFixed(2)}×`)} at n = ${big.n}` : 'no size with both kernels',
            agrees: !big || big.speedupLo == null ? null : big.speedupLo > 1 ? true : big.speedupHi < 1 ? false : null,
            note: 'The range is conservative (naive q1 / tiled q3 … naive q3 / tiled q1); undecided when it contains 1. The tiled kernel '
                + 'also reads A and B along rows (coalesced) where the naive one reads A across rows, so part of the gain is access '
                + 'pattern, not only reuse. On a software GPU workgroup memory is ordinary memory and barriers are expensive, so tiling can lose.',
        });

        // The naive kernel's implied load bandwidth: 2 loads of 4 bytes per multiply-add
        claims.push({
            claim: 'The naive kernel is bound by memory traffic, not arithmetic',
            predicted: 'naive GFLOPS ≤ 0.25 × the bandwidth that serves its loads (8 bytes per multiply-add); tiled ≤ 4 × bandwidth',
            measured: big ? `naive: ${(8 * big.n ** 3 / big.naiveMs / 1e6).toFixed(0)} GB/s of loads, tiled: ${(8 * big.n ** 3 / TILED_T / big.tiledMs / 1e6).toFixed(0)} GB/s, at n = ${big.n}` : 'no size with both kernels',
            agrees: null,
            note: 'Compare with your GPU\'s memory bandwidth (not known to the page). If the naive figure exceeds it, caches are serving most '
                + 'of its loads: the reuse happens, but in cache rather than in workgroup memory, and cache bandwidth is the limit.',
        });

        // GPU vs CPU
        const common = largest(rows, 'cpuMs', 'naiveMs', 'tiledMs');
        let measured = 'no size with both CPU and GPU times', agrees = null, note = '';
        if (common) {
            const best = common.tiledMs <= common.naiveMs ? 'tiled' : 'naive';
            const gpuLo = common[`${best}MsLo`], gpuHi = common[`${best}MsHi`];
            const faster = rows.filter(r => r.cpuMs > 0 && Math.min(r.naiveMs, r.tiledMs) < r.cpuMs).map(r => r.n);
            const from = rows.filter(r => r.cpuMs > 0).map(r => r.n).find(n => rows.filter(r => r.cpuMs > 0 && r.n >= n).every(r => faster.includes(r.n)));
            measured = `${from ? `GPU faster from n = ${from} on` : `CPU faster at n = ${common.n}`}; at n = ${common.n}: ${LABEL[best]} `
                + `${rangeText(common, `${best}Ms`, ms)} ms vs CPU ${rangeText(common, 'cpuMs', ms)} ms`;
            agrees = gpuHi < common.cpuMsLo ? true : gpuLo > common.cpuMsHi ? false : null;
            const total = ['Upload', 'Dispatch', 'Readback'].reduce((s, p) => s + (common[`${best}${p}`] ?? 0), 0);
            note = `Kernel time only; with upload and readback the ${LABEL[best]} takes ${ms(total)} ms (wall). `
                + 'Where the crossover falls depends on the GPU, the CPU and the browser; undecided when the spreads overlap.';
        }
        claims.push({
            claim: 'The GPU beats one CPU core from some n on',
            predicted: `yes: GPU time is ≈ a fixed launch cost until the GPU is full, CPU time grows like n³ from the start (CPU measured up to n = ${CPU_MAX_N})`,
            measured, agrees, note,
        });
        return claims;
    },
};

export default [threeWay];
