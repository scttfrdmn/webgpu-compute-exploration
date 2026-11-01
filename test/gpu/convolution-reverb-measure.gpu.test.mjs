// Convolution reverb compute-only API (measureConvolver in js/examples/convolution-reverb.js) and the
// Measure experiments built on it (js/teach/experiments/convolution-reverb.js), on a software GPU — CI
// only, see harness.mjs. Offline: no AudioContext, nothing is played.
//
// measureConvolver drives the shipped GPUConvolver (self-test, then streamed noise blocks through
// processBlock). Checked against the float64 direct convolution of js/refs/convolution.js on the
// streamed blocks, and the self-test's own direct-convolution check; experiments: every row has the
// declared columns and the errors stay at f32 rounding level. Printed numbers are the CI baselines.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureConvolver } from '../../js/examples/convolution-reverb.js';
import experiments, { makeIR, streamError } from '../../js/teach/experiments/convolution-reverb.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureConvolver (1 s IR, B = 1024 and 2048): self-test and streamed blocks vs float64 direct convolution', async (t) => {
    const h = makeIR(1);
    for (const blockSize of [1024, 2048]) {
        const t0 = performance.now();
        const r = await measureConvolver(gpuDevice, { blockSize, left: h[0], right: h[1], sampleRate: 48000, blocks: 8, keep: 4 });
        const err = streamError(r, h);
        t.diagnostic(`B=${blockSize}: P=${r.partitions}, self-test ${r.selfTest.relative.toExponential(2)} over blocks `
            + `${r.selfTest.checked.join(',')}, stream error ${err.toExponential(2)}, round trip ${r.wallMs.toFixed(2)} ms `
            + `(deadline ${r.deadlineMs.toFixed(2)} ms), ${((performance.now() - t0) / 1000).toFixed(1)} s`);
        assert.equal(r.partitions, Math.ceil(48000 / blockSize));
        assert.equal(r.input.left.length, 4 * blockSize);
        assert.ok(r.wet.left.some(v => v !== 0));
        assert.ok(r.selfTest.relative < 1e-5, `self-test ${r.selfTest.relative}`);
        assert.ok(err < 1e-5, `stream error ${err}`);
        assert.ok(r.wallMs > 0);
    }
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

test('experiment deadline (B = 1024): every IR length, self-test error at rounding level', async (t) => {
    const e = byId['deadline'];
    const { rows, seconds } = await runAll(e, { blockSize: 1024 });
    t.diagnostic(`${seconds.toFixed(1)} s: irSeconds P wallMs load selfTestError realtime: `
        + show(rows, ['irSeconds', 'partitions', 'wallMs', 'load', 'selfTestError', 'realtime']));
    assert.deepEqual(rows.map(r => r.irSeconds), [0.25, 0.5, 1, 2, 4, 6]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.equal(row.partitions, Math.ceil(row.irSeconds * 48000 / 1024));
        assert.ok(row.selfTestError < 1e-5 && row.wallMs > 0);
    }
});

test('experiment accuracy: both block sizes, every IR length, error at rounding level', async (t) => {
    const e = byId['accuracy'];
    const { rows, seconds } = await runAll(e, {});
    t.diagnostic(`${seconds.toFixed(1)} s: irSeconds error1024 error2048: ${show(rows, ['irSeconds', 'error1024', 'error2048'])}`);
    assert.equal(rows.length, 6);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.error1024 > 0 && row.error1024 < 1e-5 && row.error2048 > 0 && row.error2048 < 1e-5);
    }
});
