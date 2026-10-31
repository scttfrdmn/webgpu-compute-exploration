// convolution-reverb on a software GPU (CI only — see test/gpu/harness.mjs).
//
// GPUConvolver has no DOM or Web Audio dependency, so the SHIPPED class runs here unchanged on the
// real (software) device: its own buffers, explicit bind group layout with dynamic uniform offsets,
// Params writer, IR preparation (cs_pack_ir + batched forward FFT + cs_split into the filters) and
// per-block encode (cs_pack_block, forward FFT, cs_split into the delay line, cs_mac over all
// partitions, inverse FFT, cs_output) with the computeShader from wgslModules(). This is the same
// driver the CPU suite runs against its replica kernels (test/convolution-reverb.test.mjs), so the
// two suites share inputs and tolerances; here the kernels are the shipped WGSL.
//
// References: float64 direct convolution (test/helpers/convolution-reverb-refs.mjs) and the
// shipped directBlock().

import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import * as rv from '../../js/examples/convolution-reverb.js';
import { mulberry32, convolve, noiseIR } from '../helpers/convolution-reverb-refs.mjs';

const conv = new rv.GPUConvolver(device);
after(() => conv.destroy());

// Fail the test on any validation / OOM / internal error the class's encoding raised.
async function checked(label, fn) {
    for (const f of ['validation', 'out-of-memory', 'internal']) device.pushErrorScope(f);
    let value;
    try { value = await fn(); } finally {
        const errs = [];
        for (let i = 0; i < 3; i++) { const e = await device.popErrorScope(); if (e) errs.push(e.message); }
        if (errs.length) throw new Error(`${label}: ${errs.join('\n')}`);
    }
    return value;
}

test('one block and its tail vs shipped directBlock() (B = 1024, P = 2)', async (t) => {
    await checked('one block', async () => {
        const B = 1024;
        conv.configure(B);
        const [hl, hr] = noiseIR(Math.round(1.5 * B), 11, B / 2);
        conv.uploadIR(hl, hr, 48000);
        assert.equal(conv.partitions, 2);
        conv.resetStream();
        const rand = mulberry32(7);
        const xl = Float32Array.from({ length: B }, () => rand() * 2 - 1);
        const xr = Float32Array.from({ length: B }, () => rand() * 2 - 1);
        const zero = new Float32Array(B);
        let maxErr = 0, peak = 0;
        for (let m = 0; m < 3; m++) {
            const wet = await conv.processBlock(m === 0 ? xl : zero, m === 0 ? xr : zero);
            assert.equal(wet.index, m);
            [[xl, hl, wet.left], [xr, hr, wet.right]].forEach(([x, h, got]) => {
                const ref = rv.directBlock([x, zero, zero], h, m);
                for (let i = 0; i < B; i++) {
                    maxErr = Math.max(maxErr, Math.abs(got[i] - ref[i]));
                    peak = Math.max(peak, Math.abs(ref[i]));
                }
            });
        }
        const rel = maxErr / peak;
        t.diagnostic(`B=${B}, P=2, blocks 0..2: max |y_gpu - directBlock| = ${maxErr.toExponential(2)} (${rel.toExponential(2)} of peak)`);
        assert.ok(rel < 1e-5, `relative error ${rel}`);   // CPU-suite bound
    });
});

test('streaming output equals direct convolution across the delay-line wrap (B = 1024, 2048)', async (t) => {
    await checked('streaming', async () => {
        for (const B of [1024, 2048]) {
            conv.configure(B);
            const [hl, hr] = noiseIR(Math.round(2.6 * B), 5, B);   // P = 3 partitions, last one partial
            conv.uploadIR(hl, hr, 48000);
            assert.equal(conv.partitions, 3);
            conv.resetStream();
            conv.blockIndex = conv.ring - 3;   // start just before the delay-line wrap (state is cleared)
            const blocks = 8;
            const rand = mulberry32(99 + B);
            const xl = Float32Array.from({ length: blocks * B }, () => rand() * 2 - 1);
            const xr = Float32Array.from({ length: blocks * B }, () => rand() * 2 - 1);
            const heads = [];
            const out = [new Float32Array(blocks * B), new Float32Array(blocks * B)];
            for (let k = 0; k < blocks; k++) {
                heads.push(conv.blockIndex % conv.ring);
                const wet = await conv.processBlock(xl.subarray(k * B, (k + 1) * B), xr.subarray(k * B, (k + 1) * B));
                out[0].set(wet.left, k * B);
                out[1].set(wet.right, k * B);
            }
            assert.ok(heads.includes(0) && heads[0] === conv.ring - 3, `heads ${heads}`);
            let maxErr = 0, peak = 0;
            [[xl, hl], [xr, hr]].forEach(([x, h], c) => {
                const ref = convolve(x, h, blocks * B);
                for (let n = 0; n < ref.length; n++) {
                    maxErr = Math.max(maxErr, Math.abs(out[c][n] - ref[n]));
                    peak = Math.max(peak, Math.abs(ref[n]));
                }
            });
            const rel = maxErr / peak;
            t.diagnostic(`B=${B}, P=3, ${blocks} blocks through ring wrap at slot ${conv.ring}: max |y - x*h| = ${maxErr.toExponential(2)} (${rel.toExponential(2)} of peak)`);
            // f32 FFT of size 2B plus f32 MAC (twiddles from a float64 table, no in-shader trig):
            // same 1e-5 bound as the CPU suite; any bookkeeping error is O(1).
            assert.ok(rel < 1e-5, `relative error ${rel}`);
        }
    });
});

test('shipped GPUConvolver.selfTest() on generated IRs', async (t) => {
    await checked('selfTest', async () => {
        const fs = 16000;   // reduced rate keeps P small; the code path is the same
        for (const [space, B] of [['room', 1024], ['plate', 1024], ['room', 2048]]) {
            const s = rv.SPACES[space];
            const ir = rv.generateImpulseResponse(space, { sampleRate: fs, rt60: s.rt60, preDelay: s.preDelay });
            conv.configure(B);
            conv.uploadIR(ir.left, ir.right, fs);
            const r = await conv.selfTest();
            t.diagnostic(`${space} B=${B}: P=${conv.partitions}, blocks ${r.checked.join(',')} of ${r.blocks}: relative error ${r.relative.toExponential(2)}`);
            assert.equal(conv.partitions, Math.ceil(ir.left.length / B));
            assert.ok(r.peak > 0);
            assert.ok(r.relative < 1e-5, `relative ${r.relative}`);
        }
    });
});
