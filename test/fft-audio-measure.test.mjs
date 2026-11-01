// FFT audio Measure tab, CPU only: the compute-only API of js/examples/fft-audio.js (measureFFT) and the
// experiments in js/teach/experiments/fft-audio.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: the Params bytes the compute-only path writes match the WGSL struct (one slot per stage,
// canvas size 0); the samples are uploaded unchanged; the caps are enforced; the synthetic signals
// are deterministic and bounded; the float64 FFT reference agrees with the naive DFT; every sweep stays
// within its caps and every row carries the declared columns. The real results are checked on a
// software GPU in test/gpu/fft-audio-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fft from '../js/examples/fft-audio.js';
import { measureFFT, MEASURE_FFT_SIZES, MAX_MEASURE_REPEATS, bitReverse } from '../js/examples/fft-audio.js';
import { naiveDFT, fft64, spectrumError } from '../js/refs/fft.js';
import experiments, { SIZES, MAX_N, SIGNALS, SAMPLE_RATE, synthSignal } from '../js/teach/experiments/fft-audio.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const layout = structLayout(moduleCode(fft, 'compute'), 'Params');
const paramWrites = (device) => device.writes.filter(w => w.buffer.size === 256 * 12 && (w.buffer.usage & GPUBufferUsage.UNIFORM));

test('measureFFT on a mock device: Params slots per stage, samples uploaded, result shape', async () => {
    for (const n of [256, 4096]) {
        const device = mockDevice();
        const samples = synthSignal('tones', n);
        const r = await measureFFT({ device }, { samples, warmup: 1, repeats: 2, windowed: true });
        const log2n = Math.log2(n);
        const writes = paramWrites(device);
        assert.equal(writes.length, 3, 'one Params write per frame (1 warm-up + 2 timed)');
        for (const w of writes) {
            assert.equal(w.bytes.byteLength, 256 * log2n, 'one 256-byte slot per butterfly stage');
            for (let s = 0; s < log2n; s++) {
                const p = decodeStruct(w.bytes, layout, 256 * s);
                assert.deepEqual({ n: p.n, log2n: p.log2n, stage: p.stage, sampleRate: p.sampleRate, smoothing: p.smoothing, canvasSize: p.canvasSize },
                    { n, log2n, stage: s, sampleRate: SAMPLE_RATE, smoothing: 0, canvasSize: [0, 0] });
            }
        }
        // the samples buffer (and the twiddle table, n/2 complex values) get 4·n-byte writes; the last is the samples
        const uploads = device.writes.filter(w => w.bytes.byteLength === 4 * n && w.buffer.size === 4 * 4096);
        assert.ok(uploads.length >= 3);
        assert.deepEqual(new Float32Array(uploads.at(-1).bytes.buffer), samples, 'samples uploaded unchanged');
        // complex buffers can be read back
        assert.equal(device.buffers.filter(b => b.size === 8 * 4096 && (b.usage & GPUBufferUsage.COPY_SRC)).length, 2);
        assert.equal(r.n, n);
        assert.equal(r.log2n, log2n);
        assert.equal(r.dispatches, 1 + log2n + 3);
        assert.equal(r.spectrum.length, 2 * n);
        assert.equal(r.windowed.length, n);
        assert.equal(r.gpuMs, null, 'no timestamp-query on the mock device');
        assert.ok(r.wallMs >= 0);
    }
});

test('measureFFT enforces the caps', async () => {
    const device = mockDevice();
    for (const n of [0, 128, 1000, 8192]) {
        await assert.rejects(measureFFT({ device }, { samples: new Float32Array(n) }), RangeError, `N = ${n}`);
    }
    await measureFFT({ device }, { samples: new Float32Array(512), warmup: 100, repeats: 100 });
    assert.equal(paramWrites(device).length, 2 * MAX_MEASURE_REPEATS, 'warm-up and timed frames capped');
    assert.deepEqual(MEASURE_FFT_SIZES, [256, 512, 1024, 2048, 4096]);
});

test('bitReverse matches cs_window (reverseBits(i) >> (32 - log2n)) and is an involution', () => {
    for (const bits of [8, 12]) {
        const n = 1 << bits, seen = new Set();
        for (let i = 0; i < n; i++) {
            const j = bitReverse(i, bits);
            let ref = 0;
            for (let b = 0; b < 32; b++) ref = (ref | (((i >>> b) & 1) << (31 - b))) >>> 0;
            assert.equal(j, ref >>> (32 - bits));
            assert.equal(bitReverse(j, bits), i);
            seen.add(j);
        }
        assert.equal(seen.size, n);
    }
});

test('synthetic signals: deterministic, bounded, and the float64 FFT reference equals the naive DFT', (t) => {
    for (const kind of Object.keys(SIGNALS)) {
        const a = synthSignal(kind, 1024), b = synthSignal(kind, 1024);
        assert.deepEqual(a, b, `${kind} is deterministic`);
        assert.ok(a.every(v => Math.abs(v) <= 1), `${kind} within [-1, 1]`);
        assert.ok(Math.max(...a.map(Math.abs)) > 0.5, `${kind} is not silent`);
        const X = fft64(a);
        const e = spectrumError(X, naiveDFT(a, a).out, 1024, a);
        t.diagnostic(`${kind}: fft64 vs naiveDFT max ${e.maxRel.toExponential(2)}, rms ${e.rmsRel.toExponential(2)}, Parseval ${e.parseval.toExponential(2)}`);
        assert.ok(e.maxRel < 1e-12 && e.parseval < 1e-12);
    }
    // spectrumError of a reference against itself is zero; a one-bin error is seen as such
    const ref = fft64(synthSignal('tones', 256));
    assert.deepEqual(spectrumError(ref, ref, 256), { maxRel: 0, rmsRel: 0 });
    const bad = Float64Array.from(ref);
    bad[10] += 1;
    let maxRef = 0;
    for (let k = 0; k < 256; k++) maxRef = Math.max(maxRef, Math.hypot(ref[2 * k], ref[2 * k + 1]));
    assert.ok(Math.abs(spectrumError(bad, ref, 256).maxRel - 1 / maxRef) < 1e-12);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(SIZES.every(n => n <= MAX_N && MEASURE_FFT_SIZES.includes(n)));
    assert.equal(MAX_N, 4096);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000, `${e.id}: budget ≤ 20 s`);
});

test('experiments run on a mock device: one row per size, declared columns', async (t) => {
    const cases = [['accuracy', { signal: 'tones' }], ['accuracy', { signal: 'noise' }], ['accuracy', { signal: 'chirp' }], ['cost', {}]];
    for (const [id, settings] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.deepEqual(rows.map(r => r.n), SIZES);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys = columns`);
        assert.ok(Math.max(...paramWrites(device).map(w => decodeStruct(w.bytes, layout).n)) <= MAX_N);
    }
});
