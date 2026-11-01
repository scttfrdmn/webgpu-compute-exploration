// Measure experiments for the convolution reverb (js/examples/convolution-reverb.js). Interface: see the
// top of js/teach/measure.js.
//
// Offline only: nothing is played and no microphone or audio permission is needed. The GPU work goes
// through the example's compute-only API (measureConvolver), which drives the shipped GPUConvolver:
// its self-test (every partition against a direct convolution) and a few streamed blocks of seeded
// noise. The CPU reference is the float64 direct convolution of js/refs/convolution.js, applied to the
// first streamed blocks. Every sweep is a fixed list with hard caps (exported so
// test/convolution-reverb-measure.test.mjs can check them): IRs ≤ 6 s at 48 kHz (the example's
// maximum), ≤ 24 streamed blocks per row.

import { measureConvolver, BLOCK_SIZES } from '../../examples/convolution-reverb.js';
import { convolve, noiseIR } from '../../refs/convolution.js';

export const SAMPLE_RATE = 48000;
export const IR_SECONDS = [0.25, 0.5, 1, 2, 4, 6];
export const STREAM_BLOCKS = 24;
export const CHECK_BLOCKS = 4;             // streamed blocks compared with the float64 direct convolution
export const IR_SEED = 11;
// Measured in CI (software GPU): direct-convolution error of the streamed blocks, relative to their peak
export const CI_ERROR = 3e-7;

// Stereo exponentially decaying noise IR of `seconds`, falling 60 dB over its length (a "room")
export function makeIR(seconds) {
    const length = Math.round(seconds * SAMPLE_RATE);
    return noiseIR(length, IR_SEED, length / 6.9);
}

// Max |GPU − float64 direct convolution| over both channels of the first CHECK_BLOCKS streamed blocks,
// relative to the reference peak
export function streamError(r, [hl, hr]) {
    let err = 0, peak = 0;
    for (const [x, h, got] of [[r.input.left, hl, r.wet.left], [r.input.right, hr, r.wet.right]]) {
        const ref = convolve(x, h, x.length);
        for (let i = 0; i < ref.length; i++) {
            err = Math.max(err, Math.abs(got[i] - ref[i]));
            peak = Math.max(peak, Math.abs(ref[i]));
        }
    }
    return peak > 0 ? err / peak : 0;
}

const ms = (v) => v.toFixed(3);
const sci = (v) => v.toExponential(1);

const deadline = {
    id: 'deadline',
    title: 'IR length vs time per block and the real-time deadline',
    question: 'How does the cost of one block grow with the impulse response length, and how far is it from the deadline the audio hardware sets?',
    settings: [
        { key: 'blockSize', label: 'Block size', options: BLOCK_SIZES.map(b => ({ value: b, label: `${b} (${(1000 * b / SAMPLE_RATE).toFixed(1)} ms)` })), default: 1024 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'irSeconds', label: 'IR length (s)', format: String },
        { key: 'partitions', label: 'partitions P' },
        { key: 'gpuMs', label: 'GPU ms / block', format: ms },
        { key: 'wallMs', label: 'round trip ms / block', format: ms },
        { key: 'deadlineMs', label: 'deadline ms', format: (v) => v.toFixed(2) },
        { key: 'load', label: 'round trip / deadline', format: (v) => `${(100 * v).toFixed(1)}%` },
        { key: 'selfTestError', label: 'self-test error (rel.)', format: sci },
        { key: 'realtime', label: 'self-test speed (× real time)', format: (v) => v.toFixed(0) },
    ],
    charts: [
        { x: 'irSeconds', xLabel: 'impulse response length (s)', logX: true, y: ['wallMs', 'gpuMs', 'deadlineMs'],
            yLabel: 'ms per block', logY: true, labels: { wallMs: 'round trip', gpuMs: 'GPU', deadlineMs: 'deadline B / fs' } },
        { x: 'partitions', xLabel: 'partitions P', logX: true, y: ['selfTestError'], yLabel: 'max error / peak (self-test)', logY: true },
    ],
    baseline: 'Per block the GPU runs one 2B-point FFT, P·(B + 1) complex multiply-adds per channel and one inverse FFT, '
        + 'so the work grows linearly with the IR length but stays tiny next to a direct convolution (L multiply-adds per '
        + 'sample). Even on the CPU-emulated GPU in CI (B = 1024) the round trip stayed under a third of the 21.3 ms deadline '
        + 'at 6 s; on a real GPU expect the fixed cost of a submit and a readback to dominate. '
        + 'The page adds D blocks of slack on top of the B-frame partition latency. The self-test error stays about 2.5e-7 of the peak.',
    async *run(ctx, { blockSize }) {
        for (const irSeconds of IR_SECONDS) {
            await ctx.step();
            const [left, right] = makeIR(irSeconds);
            const r = await measureConvolver(ctx.gpuDevice, { blockSize, left, right, sampleRate: SAMPLE_RATE,
                blocks: STREAM_BLOCKS, keep: 0, between: () => ctx.step() });
            yield { irSeconds, partitions: r.partitions, gpuMs: r.gpuMs, wallMs: r.wallMs, deadlineMs: r.deadlineMs,
                load: r.wallMs / r.deadlineMs, selfTestError: r.selfTest.relative, realtime: r.selfTest.realtime };
        }
    },
};

const accuracy = {
    id: 'accuracy',
    title: 'Partitioned FFT convolution vs direct convolution',
    question: 'How close is the f32 partitioned FFT convolution to an exact float64 direct convolution, for both block sizes and every IR length?',
    budgetMs: 20000,
    columns: [
        { key: 'irSeconds', label: 'IR length (s)', format: String },
        { key: 'error1024', label: 'error, B = 1024', format: sci },
        { key: 'error2048', label: 'error, B = 2048', format: sci },
        { key: 'partitions1024', label: 'P (1024)' },
        { key: 'partitions2048', label: 'P (2048)' },
    ],
    charts: [{
        x: 'irSeconds', xLabel: 'impulse response length (s)', logX: true, y: ['error1024', 'error2048'],
        yLabel: 'max |GPU − direct| / peak', logY: true, labels: { error1024: 'B = 1024', error2048: 'B = 2048' },
        reference: { label: 'CI', points: [[IR_SECONDS[0], CI_ERROR], [IR_SECONDS[IR_SECONDS.length - 1], CI_ERROR]] },
    }],
    baseline: `CI (shipped shader on a software GPU): the first ${CHECK_BLOCKS} streamed noise blocks agree with a float64 direct `
        + `convolution to about ${CI_ERROR.toExponential(0)} of their peak for every IR length and both block sizes. That is a few `
        + 'f32 rounding units: the FFT error grows only like log N, and the partitions are summed in the frequency domain '
        + 'before a single inverse FFT.',
    async *run(ctx) {
        for (const irSeconds of IR_SECONDS) {
            const h = makeIR(irSeconds);
            const row = { irSeconds };
            for (const blockSize of BLOCK_SIZES) {
                await ctx.step();
                const r = await measureConvolver(ctx.gpuDevice, { blockSize, left: h[0], right: h[1], sampleRate: SAMPLE_RATE,
                    blocks: CHECK_BLOCKS, keep: CHECK_BLOCKS, between: () => ctx.step() });
                await ctx.step();
                row[`error${blockSize}`] = streamError(r, h);
                row[`partitions${blockSize}`] = r.partitions;
            }
            yield row;
        }
    },
};

export default [deadline, accuracy];
