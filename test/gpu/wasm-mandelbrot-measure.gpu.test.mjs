// WASM + WebGPU Mandelbrot compute-only API (measureMandelbrot in js/examples/wasm-mandelbrot.js) and the
// Measure experiments built on it (js/teach/experiments/wasm-mandelbrot.js), on a software GPU — CI only,
// see harness.mjs.
//
// Covers the WebGPU kernel only: the WASM package (js/wasm-pkg/) is a build artifact that this job does
// not build, so loadWasmModule() returns null here and the experiments' WASM column is empty. Checked
// against the float64 transcription of wasm/src/lib.rs compute_mandelbrot_cpu (js/refs/mandelbrot.js):
// counts agree except on the set's boundary; the deep-zoom experiment shows f32 resolution running out.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureMandelbrot, loadWasmModule } from '../../js/examples/wasm-mandelbrot.js';
import { mandelbrotEscape } from '../../js/refs/mandelbrot.js';
import experiments, { compareCounts } from '../../js/teach/experiments/wasm-mandelbrot.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureMandelbrot (200 × 150 and 640 × 480, 256 iterations) vs the float64 reference', async (t) => {
    for (const [width, height] of [[200, 150], [640, 480]]) {
        const r = await measureMandelbrot(gpuDevice, { width, height, maxIterations: 256, repeats: 2 });
        const ref = mandelbrotEscape({ width, height, max_iterations: 256, zoom: 1, center_x: -0.5, center_y: 0 });
        const c = compareCounts(r.counts, ref);
        let minAtMismatch = Infinity;
        for (let i = 0; i < ref.length; i++) if (r.counts[i] !== ref[i]) minAtMismatch = Math.min(minAtMismatch, ref[i], r.counts[i]);
        t.diagnostic(`${width}×${height}: ${(100 * c.mismatched).toFixed(3)}% of pixels differ (max |Δ| ${c.maxDiff}, `
            + `smallest count at a mismatch ${minAtMismatch}); wall ${r.wallMs.toFixed(1)} ms`);
        assert.equal(r.counts.length, width * height);
        assert.equal(r.counts[(height / 2) * width + width / 2], 256, 'the centre c = −0.5 is in the set');
        assert.ok(c.mismatched < 0.01);
        assert.ok(minAtMismatch >= 10, 'quickly escaping pixels always agree');
    }
});

test('loadWasmModule: null without the built package (this job does not build js/wasm-pkg/)', async () => {
    assert.equal(await loadWasmModule(), null);
});

const testContext = () => makeTestContext(gpuDevice);
async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(testContext(), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}
const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');

test('experiment gpu-vs-cpu (256 iterations): rows for every size, WASM column empty, agreement', async (t) => {
    const e = byId['gpu-vs-cpu'];
    const { rows, seconds } = await runAll(e, { maxIterations: 256 });
    t.diagnostic(`${seconds.toFixed(1)} s: pixels gpuWallMs jsMs mismatched: ${show(rows, ['pixels', 'gpuWallMs', 'jsMs', 'mismatched'])}`);
    assert.deepEqual(rows.map(r => r.pixels), [4800, 19200, 76800, 307200]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.equal(row.wasmMs, null);
        assert.ok(row.mismatched < 0.01 && row.gpuWallMs > 0 && row.jsMs >= 0);
    }
});

test('experiment deep-zoom: f32 runs out of distinct x values', async (t) => {
    const e = byId['deep-zoom'];
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: zoom mismatched maxDiff distinct: ${show(rows, ['zoom', 'mismatched', 'maxDiff', 'distinct'])}`);
    assert.equal(rows.length, 8);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.ok(rows[0].mismatched < 0.01, 'zoom 1 agrees almost everywhere');
    assert.ok(Math.max(...rows.map(r => r.mismatched)) > 0.1, 'deep zoom disagrees widely');
    assert.equal(rows[0].distinct, 256);
    assert.ok(rows[7].distinct < 32);
});
