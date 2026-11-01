// Convolution reverb Measure tab, CPU only: the compute-only API of js/examples/convolution-reverb.js
// (measureConvolver) and the experiments in js/teach/experiments/convolution-reverb.js, driven through
// the mock device (test/helpers/mock-gpu.mjs), which records buffers and writes but runs nothing.
// Checked: the streamed input blocks are uploaded as the overlap-save windows; the caps; the IR maker and
// the stream-error helper against js/refs/convolution.js; nothing touches Web Audio; every sweep within
// its caps and every row with the declared columns. Real results: test/gpu/convolution-reverb-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureConvolver, BLOCK_SIZES, MAX_IR_FRAMES, MAX_MEASURE_BLOCKS } from '../js/examples/convolution-reverb.js';
import { convolve } from '../js/refs/convolution.js';
import experiments, {
    makeIR, streamError, IR_SECONDS, SAMPLE_RATE, STREAM_BLOCKS, CHECK_BLOCKS,
} from '../js/teach/experiments/convolution-reverb.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { mockDevice } from './helpers/mock-gpu.mjs';

test('measureConvolver on a mock device: windows uploaded per block, kept blocks, partitions, deadline', async () => {
    const device = mockDevice();
    const [left, right] = makeIR(0.1);
    let calls = 0;
    const r = await measureConvolver({ device }, { blockSize: 1024, left, right, sampleRate: 48000, blocks: 5, keep: 3,
        between: async () => { calls++; } });
    assert.equal(r.partitions, Math.ceil(4800 / 1024));
    assert.equal(r.blockSize, 1024);
    assert.ok(Math.abs(r.deadlineMs - 1000 * 1024 / 48000) < 1e-12);
    assert.equal(calls, 1);
    assert.equal(r.input.left.length, 3 * 1024);
    assert.equal(r.wet.right.length, 3 * 1024);
    assert.ok(r.wallMs >= 0 && r.gpuMs === null);
    assert.ok(r.selfTest.blocks === r.partitions + 1);
    // The window of streamed block k is [block k − 1, block k], interleaved L/R (2B frames = 4B floats)
    const windows = device.writes.filter(w => w.bytes.byteLength === 4 * 2048 * 2).map(w => new Float32Array(w.bytes.buffer));
    const last = windows.slice(-5);
    for (let k = 1; k < 3; k++) {
        for (const i of [0, 511, 1023]) {
            assert.equal(last[k][2 * i], r.input.left[(k - 1) * 1024 + i], 'previous block, left');
            assert.equal(last[k][2 * (1024 + i) + 1], r.input.right[k * 1024 + i], 'current block, right');
        }
    }
});

test('measureConvolver enforces the caps', async () => {
    const device = mockDevice();
    const [l, r] = makeIR(0.05);
    await assert.rejects(measureConvolver({ device }, { blockSize: 512, left: l, right: r }), RangeError);
    await assert.rejects(measureConvolver({ device }, { blockSize: 1024, left: l, right: r.subarray(1) }), RangeError);
    await assert.rejects(measureConvolver({ device }, { blockSize: 1024, left: new Float32Array(0), right: new Float32Array(0) }), RangeError);
    const long = new Float32Array(MAX_IR_FRAMES + 1);
    await assert.rejects(measureConvolver({ device }, { blockSize: 1024, left: long, right: long }), RangeError);
    await assert.rejects(measureConvolver({ device }, { blockSize: 1024, left: l, right: r, blocks: MAX_MEASURE_BLOCKS + 1 }), RangeError);
});

test('makeIR and streamError: decay over the IR, zero error for the exact convolution', () => {
    const [hl, hr] = makeIR(0.5);
    assert.equal(hl.length, 24000);
    assert.notDeepEqual(hl.slice(0, 8), hr.slice(0, 8));
    const head = Math.max(...hl.slice(0, 2000).map(Math.abs)), tail = Math.max(...hl.slice(-2000).map(Math.abs));
    assert.ok(tail < 3e-3 * head, `about −60 dB at the end (${tail / head})`);
    const x = Float32Array.from({ length: 2048 }, (_, i) => Math.sin(i * 0.01));
    const exact = { input: { left: x, right: x }, wet: { left: Float32Array.from(convolve(x, hl, 2048)), right: Float32Array.from(convolve(x, hr, 2048)) } };
    assert.ok(streamError(exact, [hl, hr]) < 1e-6);
    exact.wet.left[100] += 1;
    assert.ok(streamError(exact, [hl, hr]) > 1e-3);
});

test('experiment sweeps stay within the caps; offline only', () => {
    assert.ok(IR_SECONDS.every(s => s * SAMPLE_RATE <= MAX_IR_FRAMES), 'IRs ≤ the example maximum');
    assert.ok(STREAM_BLOCKS <= MAX_MEASURE_BLOCKS && CHECK_BLOCKS <= STREAM_BLOCKS);
    assert.equal(typeof globalThis.AudioContext, 'undefined', 'no Web Audio in Node: the experiments must not need it');
    for (const e of experiments) assert.ok(e.budgetMs <= 20000);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [
        ['deadline', { blockSize: 1024 }, IR_SECONDS.length],
        ['deadline', { blockSize: 2048 }, IR_SECONDS.length],
        ['accuracy', {}, IR_SECONDS.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort());
        assert.deepEqual(rows.map(r => r.irSeconds), IR_SECONDS);
        if (id === 'deadline') {
            for (const r of rows) assert.equal(r.partitions, Math.ceil(r.irSeconds * SAMPLE_RATE / settings.blockSize));
        } else {
            for (const r of rows) for (const b of BLOCK_SIZES) assert.equal(r[`partitions${b}`], Math.ceil(r.irSeconds * SAMPLE_RATE / b));
        }
    }
});
