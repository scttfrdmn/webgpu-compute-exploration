// Measure experiments for the WASM + WebGPU Mandelbrot (js/examples/wasm-mandelbrot.js). Interface: see
// the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureMandelbrot): the shipped compute
// shader, one dispatch and one readback per run. The CPU reference is js/refs/mandelbrot.js
// mandelbrotEscape, a float64 JavaScript transcription of wasm/src/lib.rs compute_mandelbrot_cpu; when
// the WASM package is built (wasm/build.sh), the Rust function itself is timed too. Every sweep is a
// fixed list with hard caps (exported so test/wasm-mandelbrot-measure.test.mjs can check them):
// ≤ 640 × 480 pixels and ≤ 1,024 iterations, so the CPU references take at most a few hundred ms per row.

import { measureMandelbrot, loadWasmModule } from '../../examples/wasm-mandelbrot.js';
import { mandelbrotEscape } from '../../refs/mandelbrot.js';

export const RESOLUTIONS = [[80, 60], [160, 120], [320, 240], [640, 480]];
export const ITERATION_OPTIONS = [64, 256, 1024];
export const ZOOMS = [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7];
export const ZOOM_CENTER = [-0.743643887, 0.131825904];   // Seahorse valley
export const ZOOM_SIZE = [256, 192];
export const ZOOM_ITERATIONS = 1024;

// Fraction of pixels whose counts differ, and the largest difference
export function compareCounts(a, b) {
    let mismatched = 0, maxDiff = 0;
    for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (d > 0) { mismatched++; maxDiff = Math.max(maxDiff, d); }
    }
    return { mismatched: mismatched / a.length, maxDiff };
}

// Times the float64 references on the CPU: the JavaScript transcription, and the Rust/WASM original
// when the package is built (lib.rs MandelbrotConfig has a fixed 256 iterations unless the build has
// set_max_iterations)
async function cpuReferences(view, wasm) {
    let t0 = performance.now();
    const counts = mandelbrotEscape(view);
    const jsMs = performance.now() - t0;
    let wasmMs = null;
    if (wasm) {
        const config = new wasm.MandelbrotConfig(view.width, view.height);
        config.set_zoom(view.zoom);
        config.set_center(view.center_x, view.center_y);
        if (typeof config.set_max_iterations === 'function') config.set_max_iterations(view.max_iterations);
        if (config.get_max_iterations() === view.max_iterations) {
            t0 = performance.now();
            wasm.compute_mandelbrot_cpu(config);
            wasmMs = performance.now() - t0;
        }
        config.free?.();
    }
    return { counts, jsMs, wasmMs };
}

const ms = (v) => v.toFixed(2);
const pct = (v) => `${(100 * v).toFixed(3)}%`;

const gpuVsCpu = {
    id: 'gpu-vs-cpu',
    title: 'Resolution vs time: WebGPU, WASM and JavaScript',
    question: 'How much faster is the GPU kernel than the same loop on one CPU core, once the readback is included, and does it agree with the float64 reference?',
    settings: [
        { key: 'maxIterations', label: 'Max iterations', options: ITERATION_OPTIONS.map(k => ({ value: k, label: String(k) })), default: 256 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'pixels', label: 'pixels' },
        { key: 'gpuMs', label: 'GPU kernel ms', format: ms },
        { key: 'gpuWallMs', label: 'GPU wall ms (with readback)', format: ms },
        { key: 'wasmMs', label: 'WASM (Rust) ms', format: ms },
        { key: 'jsMs', label: 'JavaScript ms', format: ms },
        { key: 'speedup', label: 'JS / GPU wall', format: (v) => `${v.toFixed(1)}×` },
        { key: 'mismatched', label: 'pixels ≠ float64', format: pct },
    ],
    charts: [
        { x: 'pixels', xLabel: 'pixels', logX: true, y: ['gpuWallMs', 'gpuMs', 'wasmMs', 'jsMs'], yLabel: 'ms per image', logY: true,
            labels: { gpuWallMs: 'GPU + readback', gpuMs: 'GPU kernel', wasmMs: 'WASM', jsMs: 'JavaScript' } },
    ],
    baseline: 'The WASM column needs the package built with wasm/build.sh (and, for limits other than 256, a build that has '
        + 'set_max_iterations). CI checks the WebGPU kernel only, on a software GPU: at 640 × 480 and 256 iterations '
        + 'it differs from the float64 reference on 0.085% of the pixels, all near the boundary of the set, where the '
        + 'f32 orbit diverges from the float64 one. At small sizes the GPU wall time is the fixed cost of a submit and a '
        + 'readback, so the CPU can win. (In CI the "GPU" is emulated on the CPU, so its times say nothing about a real GPU.)',
    async *run(ctx, { maxIterations }) {
        const wasm = await loadWasmModule();
        for (const [width, height] of RESOLUTIONS) {
            await ctx.step();
            const view = { width, height, max_iterations: maxIterations, zoom: 1, center_x: -0.5, center_y: 0 };
            const gpu = await measureMandelbrot(ctx.gpuDevice, { width, height, maxIterations, repeats: 3 });
            await ctx.step();
            const cpu = await cpuReferences(view, wasm);
            const c = compareCounts(gpu.counts, cpu.counts);
            yield { pixels: width * height, gpuMs: gpu.gpuMs, gpuWallMs: gpu.wallMs, wasmMs: cpu.wasmMs, jsMs: cpu.jsMs,
                speedup: cpu.jsMs / gpu.wallMs, mismatched: c.mismatched };
        }
    },
};

const deepZoom = {
    id: 'deep-zoom',
    title: 'Zoom vs agreement with float64',
    question: 'At what zoom does the shader\'s single precision stop resolving neighbouring pixels?',
    budgetMs: 20000,
    columns: [
        { key: 'zoom', label: 'zoom', format: (v) => v.toExponential(0) },
        { key: 'pixelSpacing', label: 'pixel spacing', format: (v) => v.toExponential(1) },
        { key: 'mismatched', label: 'pixels ≠ float64', format: pct },
        { key: 'maxDiff', label: 'max |Δ iterations|' },
        { key: 'distinct', label: 'distinct f32 x per row', format: (v) => v.toLocaleString('en-US') },
        { key: 'gpuWallMs', label: 'GPU wall ms', format: ms },
        { key: 'jsMs', label: 'JavaScript ms', format: ms },
    ],
    charts: [
        { x: 'zoom', xLabel: 'zoom', logX: true, y: ['mismatched'], yLabel: 'fraction of pixels ≠ float64', logY: true },
        { x: 'zoom', xLabel: 'zoom', logX: true, y: ['distinct'], yLabel: `distinct f32 x values per row (of ${ZOOM_SIZE[0]})`, logY: true },
    ],
    baseline: `Seahorse valley, ${ZOOM_SIZE.join(' × ')}, ${ZOOM_ITERATIONS} iterations. The f32 spacing of numbers near 0.74 is `
        + '6e-8 (2⁻²⁴). CI (software GPU): 0.13% of the pixels differ from float64 at zoom 1, 9% at 10³, 40% at 10⁵ and 57% at '
        + '10⁶, where the pixel spacing 4 / (zoom · height) = 2e-8 is below the f32 spacing: only 90 of the 256 x values in a row '
        + 'are distinct, and 9 at 10⁷, so the image turns into blocks. (At 10⁷ the float64 view is nearly uniform too, so fewer '
        + 'pixels differ.) Before that, the mismatches are chaotic orbits near the boundary, more of them as more of the view is boundary.',
    async *run(ctx) {
        const [width, height] = ZOOM_SIZE;
        const [cx, cy] = ZOOM_CENTER;
        for (const zoom of ZOOMS) {
            await ctx.step();
            const view = { width, height, max_iterations: ZOOM_ITERATIONS, zoom, center_x: cx, center_y: cy };
            const gpu = await measureMandelbrot(ctx.gpuDevice, { width, height, maxIterations: ZOOM_ITERATIONS, zoom,
                centerX: cx, centerY: cy, repeats: 1 });
            await ctx.step();
            const t0 = performance.now();
            const ref = mandelbrotEscape(view);
            const jsMs = performance.now() - t0;
            const c = compareCounts(gpu.counts, ref);
            yield { zoom, pixelSpacing: 4 / (zoom * Math.min(width, height)), mismatched: c.mismatched, maxDiff: c.maxDiff,
                distinct: distinctX(width, height, zoom, cx), gpuWallMs: gpu.wallMs, jsMs };
        }
    },
};

// How many different f32 values the shader's cx takes along one row (the replica of its arithmetic)
export function distinctX(width, height, zoom, centerX) {
    const f = Math.fround;
    const scale = f(4 / f(f(zoom) * Math.min(width, height)));
    const values = new Set();
    for (let x = 0; x < width; x++) values.add(f(f(centerX) + f(f(x - f(width / 2)) * scale)));
    return values.size;
}

export default [gpuVsCpu, deepZoom];
