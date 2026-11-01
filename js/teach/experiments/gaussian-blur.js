// Measure experiments for the Gaussian blur (js/examples/image-blur.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureBlur(), which runs the shipped 2D kernel
// (blurShaderCode(σ) only changes its σ and kernel-size constants). The test image is the example's own
// createTestImage(). The reference is the separable float64 blur of js/refs/basic.js on SAMPLE pixels
// (the clipped, renormalized 2D Gaussian factors exactly into two 1D passes). Caps: σ ≤ 8 (49 × 49 taps),
// images ≤ 1,024² for the σ sweep and ≤ 2,048² at σ = 4.

import { measureBlur, createTestImage, BLUR_SIGMAS, SHIPPED_SIGMA, kernelSizeFor } from '../../examples/image-blur.js';
import { separableBlurPixel } from '../../refs/basic.js';

export const SIGMA_SIDES = [256, 512, 1024];
export const SCALING_SIDES = [128, 256, 512, 1024, 2048];
export const SAMPLE = 256;

// Pixels for the reference check: the four corners and edge midpoints (where the window is clipped) and
// a fixed scatter over the image
export function samplePixels(w, h, count = SAMPLE) {
    const pts = [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w >> 1, 0], [0, h >> 1], [w - 1, h >> 1], [w >> 1, h - 1]];
    for (let i = pts.length; i < count; i++) pts.push([(i * 197 + 13) % w, (i * 331 + 71) % h]);
    return pts;
}

export function maxErrorVsSeparable(image, output, w, h, sigma, pts) {
    const half = Math.floor(kernelSizeFor(sigma) / 2);
    let worst = 0;
    for (const [x, y] of pts) {
        const ref = separableBlurPixel(image, w, h, x, y, sigma, half);
        for (let c = 0; c < 4; c++) worst = Math.max(worst, Math.abs(output[(y * w + x) * 4 + c] - ref[c]));
    }
    return worst;
}

const ms = (v) => v.toFixed(2);
const sci = (v) => v.toExponential(2);
const rate = (count, t) => (t > 0 ? count / t / 1e6 : null);   // per ms → billions per second
const mpix = (count, t) => (t > 0 ? count / t / 1e3 : null);   // per ms → millions per second

const sigmaCost = {
    id: 'sigma-cost',
    title: 'Blur radius vs cost',
    question: 'How does the cost of the direct 2D blur grow with σ, and is the f32 result still accurate at large kernels?',
    settings: [{ key: 'side', label: 'Image', options: SIGMA_SIDES.map(s => ({ value: s, label: `${s} × ${s}` })), default: 512 }],
    budgetMs: 20000,
    columns: [
        { key: 'sigma', label: 'σ', format: String },
        { key: 'kernel', label: 'kernel', format: (v) => `${v} × ${v}` },
        { key: 'taps', label: 'taps per pixel' },
        { key: 'gpuMs', label: 'GPU ms', format: ms },
        { key: 'wallMs', label: 'wall ms', format: ms },
        { key: 'gtapsGpu', label: 'Gtaps/s (GPU time)', format: (v) => v.toFixed(2) },
        { key: 'maxError', label: 'max |error| vs float64', format: sci },
    ],
    charts: [
        {
            x: 'taps', xLabel: 'taps per pixel ((6σ + 1)²)', logX: true, y: ['gpuMs', 'wallMs'], yLabel: 'ms per blur', logY: true,
            labels: { gpuMs: 'GPU', wallMs: 'wall' },
            reference: (rows) => {
                const last = rows.filter(r => r.wallMs > 0).at(-1);
                return last ? { label: '∝ taps', points: rows.map(r => [r.taps, last.wallMs * r.taps / last.taps]) } : { label: '∝ taps', points: [] };
            },
        },
        { x: 'sigma', xLabel: 'σ (pixels)', logX: true, y: ['maxError'], yLabel: 'max |GPU − float64| (sampled pixels)', logY: true },
    ],
    baseline: 'CI (shipped shader on a CPU-emulated GPU, 256 × 256): max error 3.3e-7, 3.9e-7, 1.4e-6 and 2.8e-6 at '
        + 'σ = 1, 2, 4, 8, growing slowly with the number of f32 terms summed. Wall time 22, 66, 180 and 704 ms: 32× from '
        + 'σ = 1 to 8 for 49× the taps. A separable blur would need 2 · (6σ + 1) taps instead (14 instead of 49 at σ = 1, '
        + '98 instead of 2,401 at σ = 8).',
    async *run(ctx, { side }) {
        const s = Math.min(side, 1024);
        const image = createTestImage(s, s);
        const pts = samplePixels(s, s);
        for (const sigma of BLUR_SIGMAS) {
            await ctx.step();
            const r = await measureBlur(ctx.gpuDevice, { image, width: s, height: s, sigma, repeats: 3 });
            await ctx.step();
            yield {
                sigma, kernel: r.kernelSize, taps: r.taps, gpuMs: r.gpuMs, wallMs: r.wallMs,
                gtapsGpu: rate(s * s * r.taps, r.gpuMs), maxError: maxErrorVsSeparable(image, r.output, s, s, sigma, pts),
            };
        }
    },
};

const sizeScaling = {
    id: 'size-scaling',
    title: 'Image size vs throughput (σ = 4)',
    question: 'At what image size does the shipped 25 × 25 blur keep your GPU busy, measured in pixels per second?',
    budgetMs: 20000,
    columns: [
        { key: 'side', label: 'image', format: (v) => `${v} × ${v}` },
        { key: 'pixels', label: 'pixels' },
        { key: 'gpuMs', label: 'GPU ms', format: ms },
        { key: 'wallMs', label: 'wall ms', format: ms },
        { key: 'mpixGpu', label: 'Mpixel/s (GPU time)', format: (v) => v.toFixed(0) },
        { key: 'mpixWall', label: 'Mpixel/s (wall)', format: (v) => v.toFixed(0) },
        { key: 'maxError', label: 'max |error| vs float64', format: sci },
    ],
    charts: [{
        x: 'pixels', xLabel: 'pixels', logX: true, y: ['mpixGpu', 'mpixWall'], yLabel: 'Mpixel/s', logY: true,
        labels: { mpixGpu: 'GPU time', mpixWall: 'wall' },
    }],
    baseline: 'Each pixel does 625 taps (one exp() and one 16-byte load each), so throughput levels off once the '
        + 'image fills the GPU; below that, the fixed submit cost dominates the wall time. CI (CPU-emulated GPU): 0.22, '
        + '0.28 and 0.29 Mpixel/s at 128², 256² and 512², max error 1.4e-6; a real GPU is hundreds of times faster.',
    async *run(ctx) {
        for (const side of SCALING_SIDES) {
            const image = createTestImage(side, side);
            await ctx.step();
            const r = await measureBlur(ctx.gpuDevice, { image, width: side, height: side, sigma: SHIPPED_SIGMA, repeats: 3 });
            await ctx.step();
            const pixels = side * side;
            yield {
                side, pixels, gpuMs: r.gpuMs, wallMs: r.wallMs,
                mpixGpu: mpix(pixels, r.gpuMs), mpixWall: mpix(pixels, r.wallMs),
                maxError: maxErrorVsSeparable(image, r.output, side, side, SHIPPED_SIGMA, samplePixels(side, side)),
            };
        }
    },
};

export default [sigmaCost, sizeScaling];
