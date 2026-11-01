// Measure experiments for the FFT audio visualizer (js/examples/fft-audio.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureFFT), which runs the shipped window,
// butterfly and analysis pipelines without a canvas or an AudioContext; no microphone, file or audio
// permission is involved: the inputs are synthetic signals generated here from a fixed seed. The CPU
// references are the float64 naive DFT and a float64 radix-2 FFT of js/refs/fft.js. Every sweep is a
// fixed list with hard caps (exported so test/fft-audio-measure.test.mjs can check them): N ≤ 4,096
// (the example's largest size), ≤ 5 timed FFTs per size, a naive DFT only up to N = 4,096.

import { measureFFT, MEASURE_FFT_SIZES } from '../../examples/fft-audio.js';
import { naiveDFT, fft64, spectrumError } from '../../refs/fft.js';
import { mulberry32 } from '../../refs/random.js';

export const SIZES = MEASURE_FFT_SIZES;   // 256 … 4,096
export const MAX_N = 4096;
export const SAMPLE_RATE = 48000;         // what a compute-only instance writes into Params
export const SEED = 42;
export const REPEATS = 5;

// Synthetic test signals of n samples at SAMPLE_RATE, all within [-1, 1]:
//   tones: 440 Hz at 0.5, 1,234.5 Hz at 0.25 and 5 kHz at 0.001 (66 dB below the first), none bin-centred
//   noise: uniform white noise from a seeded generator
//   chirp: a linear sweep from 100 Hz to 10 kHz over the frame, amplitude 0.8
export const SIGNALS = {
    tones: 'Three tones (one 66 dB down)',
    noise: 'White noise',
    chirp: 'Linear chirp 100 Hz – 10 kHz',
};
export function synthSignal(kind, n, seed = SEED) {
    const x = new Float32Array(n);
    if (kind === 'noise') {
        const random = mulberry32(seed);
        for (let i = 0; i < n; i++) x[i] = 2 * random() - 1;
        return x;
    }
    for (let i = 0; i < n; i++) {
        const t = i / SAMPLE_RATE;
        if (kind === 'chirp') {
            const T = n / SAMPLE_RATE, f0 = 100, f1 = 10000;
            x[i] = 0.8 * Math.sin(2 * Math.PI * (f0 * t + (f1 - f0) * t * t / (2 * T)));
        } else {
            x[i] = 0.5 * Math.sin(2 * Math.PI * 440 * t) + 0.25 * Math.sin(2 * Math.PI * 1234.5 * t + 1)
                + 0.001 * Math.sin(2 * Math.PI * 5000 * t + 2);
        }
    }
    return x;
}

// Measured in CI (software GPU, white noise, seed 42): butterflies alone vs a float64 DFT of the GPU's
// own windowed input, max |X − X_ref| / max |X_ref|
export const CI_FFT_ERROR_AT_4096 = 1.5e-7;

const signalSetting = {
    key: 'signal', label: 'Signal',
    options: Object.entries(SIGNALS).map(([value, label]) => ({ value, label })),
    default: 'noise',
};

const ms = (v) => v.toFixed(3);
const sci = (v) => v.toExponential(2);

// ---- 1. Accuracy vs FFT size ------------------------------------------------------------------------
const accuracy = {
    id: 'accuracy',
    title: 'FFT size vs accuracy',
    question: 'How large is the single-precision FFT error, how does it grow with N, and how much of it comes from the window rather than the butterflies?',
    settings: [signalSetting],
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'N' },
        { key: 'stages', label: 'stages log₂N' },
        { key: 'fftErr', label: 'max rel. error, butterflies', format: sci },
        { key: 'fftRms', label: 'rms rel. error, butterflies', format: sci },
        { key: 'endErr', label: 'max rel. error, end to end', format: sci },
        { key: 'parseval', label: 'Parseval error', format: sci },
        { key: 'gpuMs', label: 'GPU ms (window + butterflies)', format: ms },
        { key: 'wallMs', label: 'wall ms', format: ms },
    ],
    charts: [{
        x: 'n', xLabel: 'FFT size N', logX: true,
        y: ['fftErr', 'fftRms', 'endErr'], yLabel: 'error relative to the largest bin', logY: true,
        labels: { fftErr: 'butterflies, max', fftRms: 'butterflies, rms', endErr: 'end to end (f32 window), max' },
        // ε·log₂N growth through the first measured point
        reference: (rows) => {
            const first = rows.find(r => r.fftErr > 0);
            if (!first) return null;
            const c = first.fftErr / Math.log2(first.n);
            return { label: '∝ log₂N', points: [SIZES[0], SIZES[SIZES.length - 1]].map(n => [n, c * Math.log2(n)]) };
        },
    }],
    baseline: `CI (shipped shader on a software GPU, white noise): butterflies vs a float64 DFT of the same windowed input `
        + `8.4e-8 at N = 256 rising to ${CI_FFT_ERROR_AT_4096.toExponential(1)} at 4,096 (2–3 units of 6e-8, the f32 rounding unit), `
        + 'rms 3e-8 to 5e-8, Parseval error below 1.6e-7. End to end against a float64 Hann window: 9e-8 to 1.9e-7, because '
        + 'that driver\'s f32 cos is accurate to about 2e-7; WGSL only requires 2⁻¹¹ ≈ 5e-4, so other GPUs may show a larger '
        + 'end-to-end error with the same butterflies. Index or twiddle bugs give errors of order 1.',
    async *run(ctx, { signal }) {
        for (const n of SIZES) {
            if (n > MAX_N) break;
            const samples = synthSignal(signal, n);
            await ctx.step();
            const r = await measureFFT(ctx.gpuDevice, { samples, repeats: REPEATS, windowed: true });
            await ctx.step();
            // Butterflies alone: DFT of the GPU's own windowed input. End to end: DFT of the float64 Hann window.
            const ref = naiveDFT(samples, r.windowed);
            const fft = spectrumError(r.spectrum, ref.out, n, ref.x);
            const end = spectrumError(r.spectrum, naiveDFT(samples).out, n);
            yield { n, stages: r.log2n, fftErr: fft.maxRel, fftRms: fft.rmsRel, endErr: end.maxRel,
                parseval: fft.parseval, gpuMs: r.gpuMs, wallMs: r.wallMs };
        }
    },
};

// ---- 2. Cost vs FFT size: GPU FFT vs CPU FFT vs CPU DFT ----------------------------------------------
export const CPU_MIN_MS = 5;   // CPU timings repeat until this much time has passed (at most 200 runs)

function timeCpu(fn) {
    let runs = 0;
    const t0 = performance.now();
    do { fn(); runs++; } while (performance.now() - t0 < CPU_MIN_MS && runs < 200);
    return (performance.now() - t0) / runs;
}

const cost = {
    id: 'cost',
    title: 'FFT size vs time: GPU, CPU FFT, CPU DFT',
    question: 'Is one audio-sized FFT faster on the GPU than on the CPU, and where does the time go?',
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'N' },
        { key: 'dispatches', label: 'dispatches' },
        { key: 'gpuMs', label: 'GPU ms (window + butterflies)', format: ms },
        { key: 'analysisMs', label: 'GPU ms (magnitude, bands, beat)', format: ms },
        { key: 'wallMs', label: 'GPU wall ms', format: ms },
        { key: 'cpuFftMs', label: 'CPU FFT ms (float64)', format: ms },
        { key: 'cpuDftMs', label: 'CPU naive DFT ms', format: ms },
    ],
    charts: [{
        x: 'n', xLabel: 'FFT size N', logX: true,
        y: ['gpuMs', 'wallMs', 'cpuFftMs', 'cpuDftMs'], yLabel: 'ms per transform', logY: true,
        labels: { gpuMs: 'GPU (timestamps)', wallMs: 'GPU wall (submit → done)', cpuFftMs: 'CPU FFT', cpuDftMs: 'CPU naive DFT' },
    }],
    baseline: 'Work: the FFT does N/2·log₂N butterflies (24,576 at N = 4,096), the naive DFT N² multiply-adds (16.8 M). '
        + 'CI (CPU-emulated GPU): GPU wall time 0.5–0.9 ms at every size (12 to 16 dispatches, fixed costs dominate), '
        + 'the float64 JavaScript FFT 0.03 ms at N = 256 and 0.17 ms at 4,096, the naive DFT 0.21 ms and 38 ms (×180 for '
        + '16× the size, close to N²). The example keeps its FFT on the GPU because the spectrum is drawn there and never '
        + 'read back, not because one 4,096-point FFT is expensive.',
    async *run(ctx) {
        for (const n of SIZES) {
            if (n > MAX_N) break;
            const samples = synthSignal('noise', n);
            await ctx.step();
            const r = await measureFFT(ctx.gpuDevice, { samples, repeats: REPEATS });
            await ctx.step();
            const cpuFftMs = timeCpu(() => fft64(samples));
            await ctx.step();
            const cpuDftMs = timeCpu(() => naiveDFT(samples, samples));
            yield { n, dispatches: r.dispatches, gpuMs: r.gpuMs, analysisMs: r.analysisMs, wallMs: r.wallMs, cpuFftMs, cpuDftMs };
        }
    },
};

export default [accuracy, cost];
