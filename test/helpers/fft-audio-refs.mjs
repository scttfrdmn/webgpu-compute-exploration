// fft-audio references shared by the CPU suite (test/fft-audio.test.mjs) and the GPU kernel
// tests (test/gpu/fft-audio.gpu.test.mjs). Nothing here touches a GPU.

import { mockDevice } from './mock-gpu.mjs';
import * as fft from '../../js/examples/fft-audio.js';

export const NUM_ROWS = 256, F_LO = 20, F_HI = 20000, SILENT_DB = -200;
export const DB_PER_LOG2 = Math.fround(6.02059991);

export function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// A visualizer object with just the state the uniform/twiddle writers need (no canvas, no audio),
// on a recording mock device. Returns the shipped twiddle table that setFFTSize uploaded.
export function makeViz(n) {
    const device = mockDevice();
    const viz = Object.create(fft.FFTAudioVisualizer.prototype);
    viz.device = device;
    viz.paramsStride = 256;
    viz.params = new ArrayBuffer(viz.paramsStride * 12);
    viz.paramsF32 = new Float32Array(viz.params);
    viz.paramsU32 = new Uint32Array(viz.params);
    viz.twiddleBuffer = device.createBuffer({ size: 4096 * 4 });
    viz.spectrumBuffer = device.createBuffer({ size: 2049 * 8 });
    viz.samplesBuffer = device.createBuffer({ size: 4096 * 4 });
    viz.paramsBuffer = device.createBuffer({ size: 256 * 12 });
    viz.audio = { setFFTSize() {}, sampleRate: 44100 };
    viz.canvas = { width: 1234, height: 567 };
    viz.mode = 2; viz.smoothing = 0.6; viz.dbRange = 90; viz.maxDb = -3;
    viz.column = 311; viz.trigger = 77; viz.waveGain = 2.5;
    viz.setFFTSize(n);
    const tw = new Float32Array(device.lastWrite(viz.twiddleBuffer).bytes.buffer);
    return { viz, device, tw };
}

// Independent reference: float64 naive DFT of the float64 Hann-windowed input. With `windowed`
// given, transforms that sequence as is (no window).
export function naiveDFT(samples, windowed = null) {
    const n = samples.length;
    const x = windowed ? Float64Array.from(windowed)
        : Float64Array.from(samples, (v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n)));
    const cos = new Float64Array(n), sin = new Float64Array(n);
    for (let i = 0; i < n; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    const out = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) {
        let re = 0, im = 0;
        for (let i = 0, idx = 0; i < n; i++, idx = (idx + k) % n) { re += x[i] * cos[idx]; im -= x[i] * sin[idx]; }
        out[2 * k] = re; out[2 * k + 1] = im;
    }
    return { out, x };
}

// ---- replica of cs_magnitude and cs_bands ----
export function magnitudes(X, n, smoothing = 0, prev = null) {
    const spec = new Float32Array(2 * (n / 2 + 1));
    for (let k = 0; k <= n / 2; k++) {
        const mag = Math.fround(Math.hypot(X[2 * k], X[2 * k + 1]) * 4 / n);
        const blended = Math.fround(smoothing * (prev ? prev[2 * k] : 0) + (1 - smoothing) * mag);
        spec[2 * k] = blended;
        spec[2 * k + 1] = DB_PER_LOG2 * Math.log2(Math.max(blended, 1e-10));
    }
    return spec;
}
export function bands(spec, n, sampleRate) {
    const half = n / 2, binHz = sampleRate / n, ratio = F_HI / F_LO;
    const out = new Float32Array(NUM_ROWS);
    for (let r = 0; r < NUM_ROWS; r++) {
        const bLo = F_LO * Math.pow(ratio, r / NUM_ROWS) / binHz;
        const bHi = Math.min(F_LO * Math.pow(ratio, (r + 1) / NUM_ROWS) / binHz, half);
        let db = SILENT_DB;
        if (bLo < half) {
            const kStart = Math.ceil(bLo), kEnd = Math.floor(bHi);
            if (kEnd < kStart) {
                const c = Math.sqrt(bLo * bHi);
                const k0 = Math.min(Math.floor(c), half), k1 = Math.min(k0 + 1, half);
                const f = c - Math.floor(c);
                db = spec[2 * k0 + 1] * (1 - f) + spec[2 * k1 + 1] * f;
            } else {
                for (let i = 0; i < 128 && kStart + i <= kEnd; i++) db = Math.max(db, spec[2 * (kStart + i) + 1]);
            }
        }
        out[r] = db;
    }
    return out;
}
