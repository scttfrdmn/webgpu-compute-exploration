// Vector addition (example #1).
// CPU replica of the `compute` shader; validates the algorithm and index math, not the WGSL itself — see #51.
// The shipped runVectorAddition() is driven with a mock device to capture the inputs it uploads.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runVectorAddition, wgslModules } from '../js/examples/vector-addition.js';
import { mockDevice } from './helpers/mock-gpu.mjs';

const WORKGROUP = 64;

// Replica: one invocation per global_id.x over ceil(n / 64) workgroups; out-of-range invocations
// (index >= arrayLength(&input1)) must not write.
function replica(a, b) {
    const out = new Float32Array(a.length).fill(NaN);
    const groups = Math.ceil(a.length / WORKGROUP);
    let writes = 0;
    for (let index = 0; index < groups * WORKGROUP; index++) {
        if (index < a.length) { out[index] = Math.fround(a[index] + b[index]); writes++; }
    }
    return { out, writes, invocations: groups * WORKGROUP };
}

test('shader declares workgroup size 64 and an arrayLength bound', () => {
    const code = wgslModules()[0].code;
    assert.match(code, /@workgroup_size\(64\)/);
    assert.match(code, /index < arrayLength\(&input1\)/);
});

test('shipped inputs + replica match float64 addition exactly', async (t) => {
    const device = mockDevice();
    const html = await runVectorAddition({ device });
    assert.match(html, /Vector size: 1,000,000/);
    const [b1, b2, out] = device.buffers;
    const a = new Float32Array(b1.getMappedRange());
    const b = new Float32Array(b2.getMappedRange());
    assert.equal(a.length, 1_000_000);
    assert.equal(out.size, a.byteLength, 'output buffer sized like the input');
    const { out: r, writes, invocations } = replica(a, b);
    let maxErr = 0;
    for (let i = 0; i < a.length; i++) maxErr = Math.max(maxErr, Math.abs(r[i] - (a[i] + b[i])));
    t.diagnostic(`n=${a.length} invocations=${invocations} writes=${writes} max|err|=${maxErr}`);
    assert.equal(writes, a.length, 'exactly one write per element, none past the end');
    // i + 2i = 3i < 2^24 is exactly representable in f32, so the f32 sum is exact.
    assert.equal(maxErr, 0);
});

test('bounds guard on a non-multiple-of-64 length', (t) => {
    const n = 1000;  // 16 workgroups = 1024 invocations, 24 must be masked
    const a = Float32Array.from({ length: n }, (_, i) => Math.sin(i));
    const b = Float32Array.from({ length: n }, (_, i) => Math.cos(i));
    const { out, writes, invocations } = replica(a, b);
    t.diagnostic(`n=${n} invocations=${invocations} masked=${invocations - writes}`);
    assert.equal(writes, n);
    let maxRel = 0;
    for (let i = 0; i < n; i++) {
        const exact = a[i] + b[i];
        maxRel = Math.max(maxRel, Math.abs(out[i] - exact) / Math.max(Math.abs(exact), 1e-30));
    }
    t.diagnostic(`max relative error vs float64 = ${maxRel.toExponential(2)}`);
    assert.ok(maxRel <= 2 ** -24, 'one f32 rounding (half ulp)');
});
