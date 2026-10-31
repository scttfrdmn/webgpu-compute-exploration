// fft-audio: Hann window + radix-2 FFT + magnitude/dB + log bands + beat detector.
//
// CPU replica of fft-audio.js computeShader (cs_window, cs_butterfly, cs_magnitude, cs_bands,
// cs_beat); validates the algorithm and index math, not the WGSL itself — see #51.
// The replica consumes the shipped twiddle table (captured from FFTAudioVisualizer.setFFTSize via a
// mock device) and stores every intermediate in Float32Array, like the GPU buffers. It is compared
// against an independent float64 naive DFT.
//
// Shipped JS tested directly: FFTAudioVisualizer.setFFTSize (twiddle upload) and writeParams
// (struct Params, one slot per butterfly stage), decoded with the layout parsed from the WGSL.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import * as fft from '../js/examples/fft-audio.js';
// References shared with test/gpu/fft-audio.gpu.test.mjs: shipped-writer capture (makeViz), float64
// naive DFT, and the cs_magnitude / cs_bands replicas.
import { NUM_ROWS, F_LO, F_HI, mulberry32, makeViz, naiveDFT, magnitudes, bands } from './helpers/fft-audio-refs.mjs';

const reverse = (i, bits) => {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b);
    return r;
};

// ---- replica of cs_window + log2(N) cs_butterfly dispatches (ping-pong, as frame() encodes) ----
function gpuFFT(samples, tw) {
    const n = samples.length, log2n = Math.round(Math.log2(n));
    let src = new Float32Array(2 * n), dst = new Float32Array(2 * n);
    const PI = Math.fround(3.14159265358979);
    for (let i = 0; i < n; i++) {   // cs_window
        const w = Math.fround(0.5 - Math.fround(0.5 * Math.fround(Math.cos(Math.fround(Math.fround(2 * PI * i) / n)))));
        // WGSL: reverseBits(i) >> (32 - log2n)
        const j = (reverse(i, 32) >>> 0) >>> (32 - log2n);
        dst[2 * j] = samples[i] * w;
        dst[2 * j + 1] = 0;
    }
    for (let stage = 0; stage < log2n; stage++) {   // cs_butterfly
        [src, dst] = [dst, src];
        const h = 1 << stage;
        for (let t = 0; t < n / 2; t++) {
            const j = t & (h - 1);
            const i0 = ((t >>> stage) << (stage + 1)) + j;
            const i1 = i0 + h;
            const k = j << (log2n - 1 - stage);
            const wx = tw[2 * k], wy = tw[2 * k + 1];
            const ax = src[2 * i0], ay = src[2 * i0 + 1], bx = src[2 * i1], by = src[2 * i1 + 1];
            const wbx = Math.fround(Math.fround(wx * bx) - Math.fround(wy * by));
            const wby = Math.fround(Math.fround(wx * by) + Math.fround(wy * bx));
            dst[2 * i0] = ax + wbx; dst[2 * i0 + 1] = ay + wby;
            dst[2 * i1] = ax - wbx; dst[2 * i1 + 1] = ay - wby;
        }
    }
    return dst;
}

test('twiddle table uploaded by setFFTSize is W_N^k = e^(-2πik/N), k < N/2', () => {
    for (const n of [1024, 2048, 4096]) {
        const { tw } = makeViz(n);
        assert.equal(tw.length, n, 'n/2 complex values');
        let maxErr = 0;
        for (let k = 0; k < n / 2; k++) {
            maxErr = Math.max(maxErr, Math.abs(tw[2 * k] - Math.cos(-2 * Math.PI * k / n)),
                Math.abs(tw[2 * k + 1] - Math.sin(-2 * Math.PI * k / n)));
        }
        // f32 rounding of a double-precision value: <= 2^-24 relative, i.e. < 6e-8 absolute
        assert.ok(maxErr < 6e-8, `N=${n} twiddle error ${maxErr}`);
        assert.ok(tw[2 * (n / 4) + 1] < -0.9999999, 'W_N^(N/4) = -i (forward transform sign)');
    }
});

test('replica FFT (window + bit reversal + butterflies) matches float64 naive DFT', (t) => {
    const rand = mulberry32(42);
    for (const n of [1024, 2048, 4096]) {
        const { tw } = makeViz(n);
        const samples = Float32Array.from({ length: n }, () => rand() * 2 - 1);
        const X = gpuFFT(samples, tw);
        const { out: ref, x } = naiveDFT(samples);
        let maxErr = 0, maxRef = 0, eGPU = 0, eTime = 0;
        for (let k = 0; k < n; k++) {
            maxErr = Math.max(maxErr, Math.hypot(X[2 * k] - ref[2 * k], X[2 * k + 1] - ref[2 * k + 1]));
            maxRef = Math.max(maxRef, Math.hypot(ref[2 * k], ref[2 * k + 1]));
            eGPU += X[2 * k] ** 2 + X[2 * k + 1] ** 2;
        }
        for (let i = 0; i < n; i++) eTime += x[i] * x[i];
        const rel = maxErr / maxRef;
        const parseval = Math.abs(eGPU / (n * eTime) - 1);
        t.diagnostic(`N=${n}: max |X_gpu - X_dft| / max|X| = ${rel.toExponential(2)}, Parseval error ${parseval.toExponential(2)}`);
        // f32 radix-2 FFT error grows ~ eps·log2(N); measured ~1e-7, bound 2e-6 leaves margin but
        // catches any index/twiddle error (those give O(1) errors).
        assert.ok(rel < 2e-6, `relative error ${rel}`);
        assert.ok(parseval < 1e-5, `Parseval ${parseval}`);
    }
});

test('a full-scale bin-centred sine reads 0 dB; band rows find it and stay silent elsewhere', (t) => {
    const n = 2048, fs = 44100;
    const { tw } = makeViz(n);
    const k0 = 93;                                   // 93·fs/N ≈ 2003 Hz
    const samples = Float32Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * k0 * i / n));
    const spec = magnitudes(gpuFFT(samples, tw), n);
    const peakDb = spec[2 * k0 + 1];
    t.diagnostic(`peak bin ${k0}: ${peakDb.toFixed(4)} dB; neighbours ${spec[2 * (k0 - 1) + 1].toFixed(2)}, ${spec[2 * (k0 + 1) + 1].toFixed(2)} dB (Hann: -6.02)`);
    assert.ok(Math.abs(peakDb) < 1e-3, '4/N scaling with Hann coherent gain 0.5 gives 0 dB');
    assert.ok(Math.abs(spec[2 * (k0 + 1) + 1] + 6.0206) < 0.01, 'Hann first neighbour at -6.02 dB');
    assert.ok(spec[2 * (k0 + 5) + 1] < -100, 'Hann sidelobes of a bin-centred sine vanish');

    const b = bands(spec, n, fs);
    const f = k0 * fs / n;
    const row = Math.floor(NUM_ROWS * Math.log(f / F_LO) / Math.log(F_HI / F_LO));
    t.diagnostic(`row ${row} (${f.toFixed(0)} Hz): ${b[row].toFixed(3)} dB; max of rows <${row - 10} or >${row + 10}: ${Math.max(...b.filter((_, r) => Math.abs(r - row) > 10)).toFixed(1)} dB`);
    assert.ok(Math.abs(b[row]) < 1e-3, 'row holding the sine shows the peak');
    for (let r = 0; r < NUM_ROWS; r++) if (Math.abs(r - row) > 10) assert.ok(b[r] < -60, `row ${r} = ${b[r]}`);
});

test('Params writer: one slot per stage, offsets and types match WGSL struct Params', (t) => {
    const layout = structLayout(moduleCode(fft, 'compute'), 'Params');
    const renderLayout = structLayout(moduleCode(fft, 'render'), 'Params');
    assert.deepEqual(renderLayout.offsets, layout.offsets, 'compute and render share struct Params');
    t.diagnostic(`struct Params: ${layout.size} bytes, offsets ${JSON.stringify(layout.offsets)}`);
    assert.equal(layout.size, 56, 'PARAMS_SIZE = 56 in fft-audio.js');

    const { viz, device } = makeViz(4096);
    viz.writeParams(viz.log2n);
    const w = device.lastWrite(viz.paramsBuffer);
    assert.equal(w.bytes.length, 12 * 256);
    for (let s = 0; s < 12; s++) {
        const p = decodeStruct(w.bytes, layout, s * 256);
        assert.deepEqual(p, {
            n: 4096, log2n: 12, stage: s, column: 311, sampleRate: 44100, smoothing: Math.fround(0.6),
            minDb: -93, maxDb: -3, mode: 2, trigger: 77, canvasSize: [1234, 567], waveGain: 2.5,
        }, `slot ${s}`);
    }
});

test('beat detector replica: one beat per kick (rising edge), silence and hiss never trigger', (t) => {
    // cs_beat on per-frame energy sequences (energy = sum of |X|²·(4/N)² over 40-150 Hz bins)
    const step = (st, energy) => {
        const above = energy > 1.6 * st.average && energy > 0.0001;
        let pulse = st.pulse * 0.88;
        let beat = false;
        if (above && st.above === 0) { pulse = 1; beat = true; }
        return { average: st.average + 0.03 * (energy - st.average), pulse, above: above ? 1 : 0, beat };
    };
    let st = { average: 0, pulse: 0, above: 0 };
    let beats = 0;
    // 120 BPM kicks at 60 fps: a 3-frame kick every 30 frames over a quiet bed (-50 dB)
    for (let f = 0; f < 600; f++) {
        const e = (f % 30) < 3 ? 0.25 : 1e-5;
        st = step(st, e);
        if (st.beat) beats++;
    }
    let hissBeats = 0;
    st = { average: 0, pulse: 0, above: 0 };
    const rand = mulberry32(7);
    for (let f = 0; f < 600; f++) { st = step(st, 5e-5 * rand()); if (st.beat) hissBeats++; }
    t.diagnostic(`kicks: 20 expected, ${beats} detected; hiss beats ${hissBeats}`);
    assert.equal(beats, 20);
    assert.equal(hissBeats, 0);
});
