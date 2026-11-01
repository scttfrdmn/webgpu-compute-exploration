// FFT audio compute-only API (measureFFT in js/examples/fft-audio.js) and the Measure experiments built
// on it (js/teach/experiments/fft-audio.js), on a software GPU — CI only, see harness.mjs.
//
// Unlike test/gpu/fft-audio.gpu.test.mjs, which rebuilds the dispatch sequence with explicit layouts,
// this runs the example's own pipelines, dynamic-offset bind groups and encodeFFT / encodeAnalysis
// through an FFTAudioVisualizer without a canvas or audio, exactly as the Measure tab does in a browser.
// Checked against the float64 naive DFT (js/refs/fft.js) with the bounds of fft-audio.gpu.test.mjs, at
// every size including 256 and 512, which the interactive example does not offer.
// The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureFFT, MEASURE_FFT_SIZES } from '../../js/examples/fft-audio.js';
import { naiveDFT, spectrumError } from '../../js/refs/fft.js';
import experiments, { synthSignal } from '../../js/teach/experiments/fft-audio.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureFFT at every size vs float64 naive DFT (window, butterflies, Parseval)', async (t) => {
    for (const kind of ['noise', 'tones']) {
        for (const n of MEASURE_FFT_SIZES) {
            const samples = synthSignal(kind, n);
            const r = await measureFFT(gpuDevice, { samples, repeats: 2, windowed: true });
            assert.equal(r.spectrum.length, 2 * n);
            assert.ok(r.spectrum.every(Number.isFinite));
            // the GPU window vs float64 Hann: f32 cos, bounded by 2^-11 absolute
            let winErr = 0;
            for (let i = 0; i < n; i++) winErr = Math.max(winErr, Math.abs(r.windowed[i] - samples[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n))));
            const ref = naiveDFT(samples, r.windowed);
            const e = spectrumError(r.spectrum, ref.out, n, ref.x);
            const end = spectrumError(r.spectrum, naiveDFT(samples).out, n);
            t.diagnostic(`${kind} N=${n}: butterflies max ${e.maxRel.toExponential(2)} rms ${e.rmsRel.toExponential(2)}, `
                + `Parseval ${e.parseval.toExponential(2)}; end to end max ${end.maxRel.toExponential(2)}; window max abs ${winErr.toExponential(2)}; `
                + `wall ${r.wallMs.toFixed(2)} ms`);
            assert.ok(winErr < 2 ** -11, `window ${winErr}`);
            assert.ok(e.maxRel < 2e-6 && e.parseval < 1e-5, `butterflies ${e.maxRel}, Parseval ${e.parseval}`);
            assert.ok(end.maxRel < 2 ** -11, `end to end ${end.maxRel}`);
        }
    }
});

const testContext = () => makeTestContext(gpuDevice);
const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');

async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(testContext(), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}

test('experiment accuracy: f32 error at the ulp level, log-like growth, window error above it', async (t) => {
    const e = byId.accuracy;
    for (const signal of ['noise', 'tones', 'chirp']) {
        const { rows, seconds } = await runAll(e, { signal });
        t.diagnostic(`${signal} ${seconds.toFixed(1)} s: N fftErr fftRms endErr parseval: ${show(rows, ['n', 'fftErr', 'fftRms', 'endErr', 'parseval'])}`);
        assert.deepEqual(rows.map(r => r.n), MEASURE_FFT_SIZES);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.ok(row.fftErr > 0 && row.fftErr < 2e-6, `N=${row.n} fftErr ${row.fftErr}`);
            assert.ok(row.fftRms <= row.fftErr && row.endErr < 2 ** -11 && row.parseval < 1e-5);
        }
    }
});

test('experiment cost: rows for every size, CPU DFT grows ~N²', async (t) => {
    const e = byId.cost;
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: N dispatches wallMs cpuFftMs cpuDftMs: ${show(rows, ['n', 'dispatches', 'wallMs', 'cpuFftMs', 'cpuDftMs'])}`);
    assert.deepEqual(rows.map(r => r.n), MEASURE_FFT_SIZES);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.equal(row.dispatches, 1 + Math.log2(row.n) + 3);
        assert.ok(row.wallMs > 0 && row.cpuFftMs > 0 && row.cpuDftMs > 0);
    }
    const ratio = rows.at(-1).cpuDftMs / rows[0].cpuDftMs;
    assert.ok(ratio > 30, `naive DFT 4096 / 256 = ${ratio.toFixed(0)} (N² gives 256)`);
});
