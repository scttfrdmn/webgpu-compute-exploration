// WASM + WebGPU Mandelbrot Measure tab, CPU only: the compute-only API of js/examples/wasm-mandelbrot.js
// (measureMandelbrot, loadWasmModule) and the experiments in js/teach/experiments/wasm-mandelbrot.js,
// driven through the mock device (test/helpers/mock-gpu.mjs), which records writes but runs nothing.
// Checked: the Params bytes decode through the WGSL struct in lib.rs get_shader_params order; the caps;
// the comparison and f32-resolution helpers; without the WASM package (not built in the test job) the
// experiments still run and leave the WASM column empty. Real results: test/gpu/wasm-mandelbrot-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
    measureMandelbrot, loadWasmModule, wgslModules, MEASURE_MAX_PIXELS, MEASURE_MAX_ITERATIONS, MAX_MEASURE_REPEATS,
} from '../js/examples/wasm-mandelbrot.js';
import { mandelbrotEscape } from '../js/refs/mandelbrot.js';
import experiments, {
    compareCounts, distinctX, RESOLUTIONS, ITERATION_OPTIONS, ZOOMS, ZOOM_SIZE, ZOOM_ITERATIONS,
} from '../js/teach/experiments/wasm-mandelbrot.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const PARAMS = structLayout(wgslModules()[0].code, 'Params');

test('measureMandelbrot on a mock device: Params bytes, readback size, repeats capped', async () => {
    const device = mockDevice();
    const r = await measureMandelbrot({ device }, { width: 320, height: 240, maxIterations: 1024, zoom: 1e4, centerX: -0.74,
        centerY: 0.13, repeats: 50 });
    const w = device.writes.filter(x => x.bytes.byteLength === PARAMS.size);
    assert.equal(w.length, 1);
    const f = Math.fround;
    assert.deepEqual(decodeStruct(w[0].bytes, PARAMS), { width: 320, height: 240, max_iterations: 1024, zoom: 1e4,
        center_x: f(-0.74), center_y: f(0.13) });
    assert.equal(r.counts.length, 320 * 240);
    assert.equal(r.gpuMs, null);
    assert.ok(r.wallMs >= 0);
    assert.ok(device.buffers.some(b => b.size === 320 * 240 * 4 && (b.usage & GPUBufferUsage.MAP_READ)));
    assert.ok(MAX_MEASURE_REPEATS <= 10);
    await assert.rejects(measureMandelbrot({ device }, { width: 801, height: 600 }), RangeError);
    await assert.rejects(measureMandelbrot({ device }, { width: 0, height: 10 }), RangeError);
    await assert.rejects(measureMandelbrot({ device }, { width: 10, height: 10, maxIterations: MEASURE_MAX_ITERATIONS + 1 }), RangeError);
});

test('loadWasmModule returns null without the built package; lib.rs has set_max_iterations for the Measure tab', async () => {
    assert.equal(await loadWasmModule(), null);
    const rs = readFileSync(new URL('../wasm/src/lib.rs', import.meta.url), 'utf8');
    assert.match(rs, /pub fn set_max_iterations\(&mut self, max_iterations: u32\)/);
    assert.match(rs, /#\[wasm_bindgen\]\s*pub fn compute_mandelbrot_cpu/);
});

test('compareCounts and distinctX', () => {
    assert.deepEqual(compareCounts([1, 2, 3, 4], [1, 5, 3, 0]), { mismatched: 0.5, maxDiff: 4 });
    assert.deepEqual(compareCounts([1, 2], [1, 2]), { mismatched: 0, maxDiff: 0 });
    // Pixel spacing 4 / (zoom · height) vs the f32 spacing 2⁻²⁴ ≈ 6e-8 of numbers near 0.74
    assert.equal(distinctX(256, 192, 1, -0.743643887), 256);
    assert.equal(distinctX(256, 192, 1e5, -0.743643887), 256);
    const d6 = distinctX(256, 192, 1e6, -0.743643887), d7 = distinctX(256, 192, 1e7, -0.743643887);
    assert.ok(d6 > 60 && d6 < 120 && d7 > 4 && d7 < 16, `${d6}, ${d7}`);
    const ref = mandelbrotEscape({ width: 8, height: 6, max_iterations: 64, zoom: 1, center_x: -0.5, center_y: 0 });
    assert.equal(compareCounts(ref, ref).mismatched, 0);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(RESOLUTIONS.every(([w, h]) => w * h <= 640 * 480 && w * h <= MEASURE_MAX_PIXELS));
    assert.ok(ITERATION_OPTIONS.every(k => k <= 1024) && ZOOM_ITERATIONS <= 1024);
    assert.ok(ZOOM_SIZE[0] * ZOOM_SIZE[1] <= MEASURE_MAX_PIXELS && ZOOMS.every(z => z >= 1));
    for (const e of experiments) assert.ok(e.budgetMs <= 20000);
});

test('experiments run on a mock device: one row per sweep value, declared columns, no WASM column without the package', async (t) => {
    const cases = [
        ['gpu-vs-cpu', { maxIterations: 64 }, RESOLUTIONS.length],
        ['gpu-vs-cpu', { maxIterations: 1024 }, RESOLUTIONS.length],
        ['deep-zoom', {}, ZOOMS.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms (CPU references included)`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort());
        if (id === 'gpu-vs-cpu') {
            assert.deepEqual(rows.map(r => r.pixels), RESOLUTIONS.map(([w, h]) => w * h));
            assert.ok(rows.every(r => r.wasmMs === null && r.jsMs >= 0));
        }
    }
});
