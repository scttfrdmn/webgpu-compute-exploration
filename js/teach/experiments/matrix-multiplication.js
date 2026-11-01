// Measure experiments for matrix multiplication (js/examples/matrix-multiplication.js). Interface: see the
// top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureMatmul() (the shipped naive kernel, one
// thread per output element). Inputs are uniform in [0, 1) as in the example; the reference is the float64
// product of js/refs/basic.js for SAMPLE_ROWS evenly spaced rows (O(rows · n²), so 1024² stays cheap).
// Sizes come from a fixed list capped at 1,024 (2.1 GFLOP per product; the example runs 512).

import { measureMatmul } from '../../examples/matrix-multiplication.js';
import { matmulRowsF64 } from '../../refs/basic.js';
import { mulberry32 } from '../../refs/random.js';

export const SIZES = [64, 128, 256, 512, 1024];
export const MAX_SIZE = 1024;
export const SAMPLE_ROWS = 16;
export const SEED = 7;
export const UNIT_ROUNDOFF = 2 ** -24;   // f32

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

const ms = (v) => v.toFixed(3);
const sci = (v) => v.toExponential(2);
const gflops = (flops, t) => (t > 0 ? flops / t / 1e6 : null);

const scaling = {
    id: 'scaling',
    title: 'Size vs GFLOPS and accuracy',
    question: 'How fast is a naive one-thread-per-element matrix product on your GPU as n grows, and how far is its f32 result from float64?',
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'n (n × n · n × n)' },
        { key: 'gpuMs', label: 'GPU ms', format: ms },
        { key: 'wallMs', label: 'wall ms', format: ms },
        { key: 'gflopsGpu', label: 'GFLOPS (GPU time)', format: (v) => v.toFixed(1) },
        { key: 'gflopsWall', label: 'GFLOPS (wall)', format: (v) => v.toFixed(1) },
        { key: 'maxRelError', label: 'max rel. error vs float64', format: sci },
        { key: 'bound', label: 'n·u bound', format: sci },
    ],
    charts: [
        {
            x: 'n', xLabel: 'matrix size n', logX: true, y: ['gflopsGpu', 'gflopsWall'], yLabel: 'GFLOPS (2n³ per product)', logY: true,
            labels: { gflopsGpu: 'GPU time', gflopsWall: 'wall' },
        },
        {
            x: 'n', xLabel: 'matrix size n', logX: true, y: ['gpuMs', 'wallMs'], yLabel: 'ms per product', logY: true,
            labels: { gpuMs: 'GPU', wallMs: 'wall' },
            // n³ through the largest wall time: where the measurements bend away from it, overhead dominates
            reference: (rows) => {
                const last = rows.filter(r => r.wallMs > 0).at(-1);
                return last ? { label: 'n³', points: [SIZES[0], last.n].map(n => [n, last.wallMs * (n / last.n) ** 3]) } : { label: 'n³', points: [] };
            },
        },
        {
            x: 'n', xLabel: 'matrix size n', logX: true, y: ['maxRelError'], yLabel: 'max relative error vs float64', logY: true,
            labels: { maxRelError: 'measured (sampled rows)' },
            reference: { label: 'n·u worst case', points: [SIZES[0], SIZES.at(-1)].map(n => [n, n * UNIT_ROUNDOFF]) },
        },
    ],
    baseline: 'CI (shipped shader on a CPU-emulated GPU, same inputs): max relative error 3.4e-7, 6.1e-7, 8.0e-7, 1.1e-6 '
        + 'and 1.6e-6 at n = 64 … 1,024, just under √n·u (4.8e-7 … 1.9e-6) and 40–600× below the n·u worst case. '
        + 'Throughput there is 0.3–0.8 GFLOPS (wall); a real GPU runs this naive kernel at tens to hundreds.',
    async *run(ctx) {
        for (const n of SIZES) {
            if (n > MAX_SIZE) break;
            const A = makeMatrix(n, n, SEED), B = makeMatrix(n, n, SEED + 1);
            await ctx.step();
            const r = await measureMatmul(ctx.gpuDevice, { A, B, M: n, N: n, K: n, repeats: 3 });
            await ctx.step();
            const rows = sampleRows(n);
            const ref = matmulRowsF64(A, B, n, n, n, rows);
            yield {
                n, gpuMs: r.gpuMs, wallMs: r.wallMs, gflopsGpu: gflops(r.flops, r.gpuMs), gflopsWall: gflops(r.flops, r.wallMs),
                maxRelError: maxRelError(r.C, n, rows, ref), bound: n * UNIT_ROUNDOFF,
            };
        }
    },
};

export default [scaling];
