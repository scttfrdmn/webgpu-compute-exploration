// Measure experiments for vector addition (js/examples/vector-addition.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureVectorAdd(), which runs the shipped shader
// and times the kernel (GPU timestamps), the kernel as the CPU sees it (submit → done) and a full round
// trip (upload, dispatch, read back). The reference is exact: an f32 add is correctly rounded, so every
// GPU sum must equal Math.fround(a + b) (a + b of two f32 values is exact in float64). The sweep is a fixed
// list capped at MAX_DISPATCH_N = 65,535 × 64 elements, the most one 1D dispatch of the shader covers.

import { measureVectorAdd, MAX_DISPATCH_N } from '../../examples/vector-addition.js';
import { mulberry32 } from '../../refs/random.js';

export const SIZES = [1024, 4096, 16384, 65536, 262144, 1048576, MAX_DISPATCH_N];
export const SEED = 1;
export const BYTES_PER_ELEMENT = 12;   // two f32 loads and one f32 store

// Inputs uniform in [−1, 1), rounded to f32
export function makeInputs(n, seed = SEED) {
    const rand = mulberry32(seed);
    const a = new Float32Array(n), b = new Float32Array(n);
    for (let i = 0; i < n; i++) { a[i] = 2 * rand() - 1; b[i] = 2 * rand() - 1; }
    return { a, b };
}

// Elements whose GPU sum differs from the correctly rounded f32 sum
export function roundingMismatches(a, b, sum) {
    let bad = 0;
    for (let i = 0; i < a.length; i++) if (sum[i] !== Math.fround(a[i] + b[i])) bad++;
    return bad;
}

const ms = (v) => v.toFixed(3);
const gbps = (bytes, t) => (t > 0 ? bytes / t / 1e6 : null);   // bytes per ms → GB/s

const bandwidth = {
    id: 'bandwidth',
    title: 'Size vs bandwidth and overhead',
    question: 'How large must a vector be before the GPU runs at memory bandwidth, and how much of a one-shot call is transfer and launch overhead?',
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'N' },
        { key: 'gpuMs', label: 'kernel GPU ms', format: ms },
        { key: 'kernelWallMs', label: 'kernel wall ms', format: ms },
        { key: 'roundTripMs', label: 'round trip wall ms', format: ms },
        { key: 'gbGpu', label: 'GB/s (GPU time)', format: (v) => v.toFixed(1) },
        { key: 'gbWall', label: 'GB/s (kernel wall)', format: (v) => v.toFixed(1) },
        { key: 'gbRoundTrip', label: 'GB/s (round trip)', format: (v) => v.toFixed(2) },
        { key: 'mismatches', label: 'sums ≠ fround(a + b)' },
    ],
    charts: [
        {
            x: 'n', xLabel: 'elements N', logX: true, y: ['gpuMs', 'kernelWallMs', 'roundTripMs'], yLabel: 'milliseconds', logY: true,
            labels: { gpuMs: 'kernel (GPU)', kernelWallMs: 'kernel (wall)', roundTripMs: 'upload + kernel + readback' },
            title: 'Time vs N',
        },
        {
            x: 'n', xLabel: 'elements N', logX: true, y: ['gbGpu', 'gbWall', 'gbRoundTrip'], yLabel: 'effective GB/s (12 bytes per element)', logY: true,
            labels: { gbGpu: 'GPU time', gbWall: 'kernel wall', gbRoundTrip: 'round trip' },
            title: 'Effective bandwidth vs N',
        },
    ],
    baseline: 'CI (shipped shader on a CPU-emulated GPU, no timestamps): every sum equals the correctly rounded f32 sum '
        + 'at every N. Kernel wall time 0.18 ms at N = 1,024 and 0.19 ms at 4,096 (pure overhead), 16 ms at 4,194,240 '
        + '(3.2 GB/s); the round trip 49 ms there (1.0 GB/s). A real GPU runs the kernel far faster; the shape of the '
        + 'curves, flat and then linear, is the same.',
    async *run(ctx) {
        for (const n of SIZES) {
            if (n > MAX_DISPATCH_N) break;
            const { a, b } = makeInputs(n);
            await ctx.step();
            const r = await measureVectorAdd(ctx.gpuDevice, { a, b, repeats: 5 });
            const bytes = BYTES_PER_ELEMENT * n;
            yield {
                n, gpuMs: r.gpuMs, kernelWallMs: r.kernelWallMs, roundTripMs: r.roundTripMs,
                gbGpu: gbps(bytes, r.gpuMs), gbWall: gbps(bytes, r.kernelWallMs), gbRoundTrip: gbps(bytes, r.roundTripMs),
                mismatches: roundingMismatches(a, b, r.sum),
            };
        }
    },
};

export default [bandwidth];
