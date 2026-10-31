// fft-audio on a software GPU (CI only — see test/gpu/harness.mjs).
//
// Runs the shipped compute module (wgslModules() 'compute') with the dispatch sequence of
// FFTAudioVisualizer.frame(): cs_window into complexBuffers[0], log2(N) cs_butterfly stages
// ping-ponging the two complex buffers (stage s bound with params slot s), then cs_magnitude,
// cs_bands and cs_beat reading complexBuffers[log2n % 2]. The twiddle table and the per-stage
// Params bytes are the ones the shipped setFFTSize()/writeParams() produce (captured through the
// mock device, test/helpers/fft-audio-refs.mjs). The example binds params with a dynamic offset;
// here each stage gets its own bind group over the same 56-byte range (offset s·256), which is
// what the dynamic offset selects. Each dispatch is its own pass (frame() uses one pass; WebGPU
// orders storage writes between dispatches either way).
//
// References: float64 naive DFT (test/helpers/fft-audio-refs.mjs) and the cs_magnitude/cs_bands
// replicas of the CPU suite.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, write, read, run } from './harness.mjs';
import { wgslModules } from '../../js/examples/fft-audio.js';
import { NUM_ROWS, SILENT_DB, mulberry32, makeViz, naiveDFT, magnitudes, bands } from '../helpers/fft-audio-refs.mjs';

const CODE = wgslModules().find(m => m.name === 'compute').code;
const MAX_N = 4096, SPEC_COLUMNS = 512, STRIDE = 256, PARAMS_SIZE = 56, WG = 64;

const pipelines = {};
for (const e of ['cs_window', 'cs_butterfly', 'cs_magnitude', 'cs_bands', 'cs_beat']) {
    pipelines[e] = await computePipeline(CODE, e);
}

const reverse = (i, bits) => {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b);
    return r >>> 0;
};

// GPU buffers laid out as FFTAudioVisualizer.setupBuffers()/setFFTSize() leave them.
function makeState(n, smoothing = 0.6) {
    const { viz, device: mock, tw } = makeViz(n);
    viz.smoothing = smoothing;
    viz.writeParams(viz.log2n);
    const paramBytes = mock.lastWrite(viz.paramsBuffer).bytes;
    const analysis = new Float32Array(4 + NUM_ROWS).fill(SILENT_DB);
    analysis.fill(0, 0, 4);
    const s = {
        n, log2n: viz.log2n, column: viz.column, sampleRate: viz.audio.sampleRate,
        samples: buffer(MAX_N * 4),
        complex: [buffer(MAX_N * 8), buffer(MAX_N * 8)],
        twiddles: buffer(tw),
        spectrum: buffer((MAX_N / 2 + 1) * 8),
        analysis: buffer(analysis),
        spectrogram: buffer(new Float32Array(SPEC_COLUMNS * NUM_ROWS).fill(SILENT_DB)),
        params: buffer(STRIDE * 12),
    };
    write(s.params, paramBytes);
    // computeBindGroups[i] reads complexBuffers[i] and writes complexBuffers[1 - i]; slot = stage
    s.group = (entry, i, slot) => bindGroup(pipelines[entry], 0, {
        samples: s.samples, src: s.complex[i], dst: s.complex[1 - i], twiddles: s.twiddles,
        spectrum: s.spectrum, analysis: s.analysis, spectrogram: s.spectrogram,
        params: { buffer: s.params, offset: slot * STRIDE, size: PARAMS_SIZE },
    });
    return s;
}

// The compute half of frame(); `upTo` stops after the window pass if 'window'.
async function frame(s, samples, upTo = 'all') {
    write(s.samples, samples);
    const n = s.n, P = pipelines;
    const steps = [{ p: P.cs_window, groups: [s.group('cs_window', 1, 0)], workgroups: [Math.ceil(n / WG)] }];
    if (upTo === 'all') {
        for (let st = 0; st < s.log2n; st++) {
            steps.push({ p: P.cs_butterfly, groups: [s.group('cs_butterfly', st % 2, st)], workgroups: [Math.ceil(n / 2 / WG)] });
        }
        const f = s.log2n % 2;
        steps.push({ p: P.cs_magnitude, groups: [s.group('cs_magnitude', f, 0)], workgroups: [Math.ceil((n / 2 + 1) / WG)] });
        steps.push({ p: P.cs_bands, groups: [s.group('cs_bands', f, 0)], workgroups: [Math.ceil(NUM_ROWS / WG)] });
        steps.push({ p: P.cs_beat, groups: [s.group('cs_beat', f, 0)], workgroups: [1] });
    }
    await run(steps, `fft-audio frame N=${n}`);
}

test('cs_window: periodic Hann window written to bit-reversed positions', async (t) => {
    const rand = mulberry32(42);
    for (const n of [1024, 2048, 4096]) {
        const s = makeState(n);
        const log2n = Math.round(Math.log2(n));
        const samples = Float32Array.from({ length: n }, () => rand() * 2 - 1);
        await frame(s, samples, 'window');
        const out = await read(s.complex[0], Float32Array, n * 8);
        let maxErr = 0, maxIm = 0;
        for (let i = 0; i < n; i++) {
            const j = reverse(i, log2n);
            const ref = samples[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n));
            maxErr = Math.max(maxErr, Math.abs(out[2 * j] - ref));
            maxIm = Math.max(maxIm, Math.abs(out[2 * j + 1]));
        }
        t.diagnostic(`N=${n}: max |x·w_gpu - x·w_f64| = ${maxErr.toExponential(2)}`);
        assert.equal(maxIm, 0, 'imaginary parts are zero');
        // The window uses in-shader f32 cos of arguments in [0, 2π). WGSL/Vulkan only bound cos to
        // 2^-11 absolute (on [-π, π]); an index or bit-reversal error would be O(1).
        assert.ok(maxErr < 2 ** -11, `window error ${maxErr}`);
    }
});

test('window + butterflies match a float64 naive DFT; magnitudes, bands, beat energy', async (t) => {
    const rand = mulberry32(42);
    for (const n of [1024, 2048, 4096]) {
        const s = makeState(n);
        const log2n = s.log2n;
        const samples = Float32Array.from({ length: n }, () => rand() * 2 - 1);
        await frame(s, samples, 'window');
        const win = await read(s.complex[0], Float32Array, n * 8);
        const windowed = Float64Array.from({ length: n }, (_, i) => win[2 * reverse(i, log2n)]);
        await frame(s, samples);
        const X = await read(s.complex[log2n % 2], Float32Array, n * 8);

        // (a) Butterflies alone: DFT of the GPU's own windowed input (separates the in-shader cos).
        const { out: ref, x } = naiveDFT(samples, windowed);
        let maxErr = 0, maxRef = 0, eGPU = 0, eTime = 0;
        for (let k = 0; k < n; k++) {
            maxErr = Math.max(maxErr, Math.hypot(X[2 * k] - ref[2 * k], X[2 * k + 1] - ref[2 * k + 1]));
            maxRef = Math.max(maxRef, Math.hypot(ref[2 * k], ref[2 * k + 1]));
            eGPU += X[2 * k] ** 2 + X[2 * k + 1] ** 2;
        }
        for (let i = 0; i < n; i++) eTime += x[i] * x[i];
        const rel = maxErr / maxRef;
        const parseval = Math.abs(eGPU / (n * eTime) - 1);
        // (b) End to end against the float64 Hann window.
        const { out: ref64 } = naiveDFT(samples);
        let maxErr64 = 0;
        for (let k = 0; k < n; k++) maxErr64 = Math.max(maxErr64, Math.hypot(X[2 * k] - ref64[2 * k], X[2 * k + 1] - ref64[2 * k + 1]));
        const rel64 = maxErr64 / maxRef;
        t.diagnostic(`N=${n}: FFT vs DFT(gpu window) rel ${rel.toExponential(2)}, Parseval ${parseval.toExponential(2)}; vs DFT(f64 Hann) rel ${rel64.toExponential(2)}`);
        // Same bounds as the CPU suite: f32 radix-2 error ~eps·log2(N); index/twiddle errors are O(1).
        assert.ok(rel < 2e-6, `relative error ${rel}`);
        assert.ok(parseval < 1e-5, `Parseval ${parseval}`);
        // Includes the window's in-shader cos error (bounded by 2^-11 per sample, see above).
        assert.ok(rel64 < 2 ** -11, `end-to-end relative error ${rel64}`);

        // cs_magnitude (smoothing 0.6 over a zeroed spectrum) vs the replica on the GPU spectrum.
        const spec = await read(s.spectrum, Float32Array, (n / 2 + 1) * 8);
        const specRef = magnitudes(X, n, 0.6, new Float32Array(n + 2));
        let magRel = 0, dbErr = 0;
        for (let k = 0; k <= n / 2; k++) {
            magRel = Math.max(magRel, Math.abs(spec[2 * k] - specRef[2 * k]) / Math.max(specRef[2 * k], 1e-6));
            if (specRef[2 * k] > 1e-6) dbErr = Math.max(dbErr, Math.abs(spec[2 * k + 1] - specRef[2 * k + 1]));
        }
        // cs_bands vs the replica on the GPU's own dB spectrum; GPU pow() may move a row edge by
        // one bin when it lands within an ulp of an integer, so allow a couple of such rows.
        const an = await read(s.analysis, Float32Array, (4 + NUM_ROWS) * 4);
        const bandRef = bands(spec, n, s.sampleRate);
        let rowsOff = 0, bandErr = 0;
        for (let r = 0; r < NUM_ROWS; r++) {
            const d = Math.abs(an[4 + r] - bandRef[r]);
            if (d > 0.01) rowsOff++; else bandErr = Math.max(bandErr, d);
        }
        const col = await read(s.spectrogram, Float32Array, NUM_ROWS * 4, s.column * NUM_ROWS * 4);
        // cs_beat: energy of the 40-150 Hz bins of this (unsmoothed) frame.
        const binHz = s.sampleRate / n;
        const kStart = Math.max(Math.ceil(40 / binHz), 1), kEnd = Math.max(Math.floor(150 / binHz), kStart);
        let energy = 0;
        for (let k = kStart; k <= kEnd; k++) energy += (Math.hypot(X[2 * k], X[2 * k + 1]) * 4 / n) ** 2;
        const eRel = Math.abs(an[0] - energy) / energy;
        t.diagnostic(`N=${n}: |X| rel ${magRel.toExponential(2)}, dB err ${dbErr.toExponential(2)}, bands err ${bandErr.toExponential(2)} (${rowsOff} rows off), beat energy rel ${eRel.toExponential(2)}`);
        assert.ok(magRel < 1e-5, `magnitude ${magRel}`);
        assert.ok(dbErr < 1e-3, `dB ${dbErr}`);
        assert.ok(rowsOff <= 2, `${rowsOff} band rows differ`);
        assert.ok(bandErr < 1e-3, `band dB ${bandErr}`);
        assert.deepEqual([...col], [...an.subarray(4)], 'spectrogram column = bands');
        assert.ok(eRel < 1e-4, `beat energy ${eRel}`);
        assert.ok(Math.abs(an[1] - 0.03 * an[0]) <= 1e-6 * an[0] + 1e-12, 'running average updated once');
    }
});

test('a full-scale bin-centred sine reads 0 dB on the GPU; band rows find it', async (t) => {
    const n = 2048, fs = 44100, k0 = 93;
    const s = makeState(n, 0);
    const samples = Float32Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * k0 * i / n));
    await frame(s, samples);
    const spec = await read(s.spectrum, Float32Array, (n / 2 + 1) * 8);
    const peakDb = spec[2 * k0 + 1];
    t.diagnostic(`peak bin ${k0}: ${peakDb.toFixed(5)} dB; neighbour ${spec[2 * (k0 + 1) + 1].toFixed(4)} dB; bin k0+5 ${spec[2 * (k0 + 5) + 1].toFixed(1)} dB`);
    assert.ok(Math.abs(peakDb) < 1e-3, '4/N scaling with Hann coherent gain 0.5 gives 0 dB');
    assert.ok(Math.abs(spec[2 * (k0 + 1) + 1] + 6.0206) < 0.01, 'Hann first neighbour at -6.02 dB');
    assert.ok(spec[2 * (k0 + 5) + 1] < -100, 'Hann sidelobes of a bin-centred sine vanish');
    const an = await read(s.analysis, Float32Array, (4 + NUM_ROWS) * 4);
    const b = an.subarray(4);
    const f = k0 * fs / n;
    const row = Math.floor(NUM_ROWS * Math.log(f / 20) / Math.log(20000 / 20));
    t.diagnostic(`row ${row} (${f.toFixed(0)} Hz): ${b[row].toFixed(4)} dB`);
    assert.ok(Math.abs(b[row]) < 1e-3, 'row holding the sine shows the peak');
    for (let r = 0; r < NUM_ROWS; r++) if (Math.abs(r - row) > 10) assert.ok(b[r] < -60, `row ${r} = ${b[r]}`);
});
