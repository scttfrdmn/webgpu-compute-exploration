// convolution-reverb: uniformly partitioned overlap-save (UPOLS) FFT convolution, procedural IRs,
// Schroeder RT60, and the worklet/stream bookkeeping.
//
// CPU replica of convolution-reverb.js computeShader (cs_pack_block, cs_pack_ir, cs_fft_forward,
// cs_fft_inverse, cs_split, cs_mac, cs_output); validates the algorithm and index math, not the
// WGSL itself — see #51.
//
// How: the SHIPPED GPUConvolver (and ReverbStream, and the AudioWorklet processor in
// WORKLET_SOURCE) run unmodified against an emulating device defined below. Its queue executes the
// recorded compute passes on the CPU: each dispatch runs the replica of the named entry point with
// the bind group's buffers and the Params decoded (with the layout parsed from the WGSL) from the
// dynamic uniform offset the JS selected. So the JS orchestration — ping-pong bind groups, stage
// slots, delay-line head/ring, partition counts, overlap-save history, readback copies — is real
// code; only the shader bodies are replicas. Results are compared against direct time-domain
// convolution in float64.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { installGPUGlobals, mockDevice } from './helpers/mock-gpu.mjs';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import * as rv from '../js/examples/convolution-reverb.js';
// mulberry32, float64 direct convolution and the noise IR, shared with test/gpu/.
import { mulberry32, convolve, noiseIR } from './helpers/convolution-reverb-refs.mjs';

const computeCode = moduleCode(rv, 'compute');
const PARAMS = structLayout(computeCode, 'Params');
const MAX_PARTITIONS = Number(/const MAX_PARTITIONS: u32 = (\d+)u/.exec(computeCode)[1]);
const WORKGROUP = Number(/@workgroup_size\((\d+)\)/.exec(computeCode)[1]);

function reverseBits32(i) {
    let r = 0;
    for (let b = 0; b < 32; b++) r = (r << 1) | ((i >>> b) & 1);
    return r >>> 0;
}
const fr = Math.fround;

const KERNELS = {
    // dst[bitReverse(i)] = timeIn[i]
    cs_pack_block(p, B, count) {
        for (let i = 0; i < count; i++) {
            if (i >= p.n) continue;
            const j = reverseBits32(i) >>> (32 - p.log2n);
            B.dst[2 * j] = B.timeIn[2 * i]; B.dst[2 * j + 1] = B.timeIn[2 * i + 1];
        }
    },
    cs_pack_ir(p, B, count) {
        for (let t = 0; t < count; t++) {
            if (t >= p.batch * p.n) continue;
            const b = t >>> p.log2n, i = t & (p.n - 1);
            let vx = 0, vy = 0;
            if (i < p.blockSize) { vx = B.timeIn[2 * (b * p.blockSize + i)]; vy = B.timeIn[2 * (b * p.blockSize + i) + 1]; }
            const j = (b << p.log2n) + (reverseBits32(i) >>> (32 - p.log2n));
            B.dst[2 * j] = vx; B.dst[2 * j + 1] = vy;
        }
    },
    cs_fft_forward(p, B, count) { butterfly(p, B, count, 1); },
    cs_fft_inverse(p, B, count) { butterfly(p, B, count, -1); },
    cs_split(p, B, count) {
        const bins = p.blockSize + 1;
        for (let t = 0; t < count; t++) {
            if (t >= p.batch * bins) continue;
            const b = Math.floor(t / bins), k = t - b * bins;
            const base = b << p.log2n;
            const zi = base + k, mi = base + ((p.n - k) & (p.n - 1));
            const zx = B.src[2 * zi], zy = B.src[2 * zi + 1];
            const cx = B.src[2 * mi], cy = -B.src[2 * mi + 1];
            const o = 4 * ((p.head + b) * bins + k);
            B.spectra[o] = 0.5 * (zx + cx); B.spectra[o + 1] = 0.5 * (zy + cy);
            const dx = zx - cx, dy = zy - cy;
            B.spectra[o + 2] = 0.5 * dy; B.spectra[o + 3] = 0.5 * -dx;
        }
    },
    cs_mac(p, B, count) {
        const bins = p.blockSize + 1;
        const br = (i) => reverseBits32(i) >>> (32 - p.log2n);
        for (let k = 0; k < count; k++) {
            if (k >= bins) continue;
            let lx = 0, ly = 0, rx = 0, ry = 0;
            for (let j = 0; j < MAX_PARTITIONS; j++) {
                if (j >= p.partitions) break;
                const slot = (p.head + p.ring - j) % p.ring;
                const x = 4 * (slot * bins + k), h = 4 * (j * bins + k);
                const [ax, ay, bx, by] = [B.spectra[x], B.spectra[x + 1], B.spectra[x + 2], B.spectra[x + 3]];
                const [hx, hy, gx, gy] = [B.filters[h], B.filters[h + 1], B.filters[h + 2], B.filters[h + 3]];
                lx = fr(lx + fr(fr(ax * hx) - fr(ay * hy))); ly = fr(ly + fr(fr(ax * hy) + fr(ay * hx)));
                rx = fr(rx + fr(fr(bx * gx) - fr(by * gy))); ry = fr(ry + fr(fr(bx * gy) + fr(by * gx)));
            }
            const d = br(k);
            B.dst[2 * d] = lx - ry; B.dst[2 * d + 1] = ly + rx;
            if (k > 0 && k < p.blockSize) {
                const m = br(p.n - k);
                B.dst[2 * m] = lx + ry; B.dst[2 * m + 1] = rx - ly;
            }
        }
    },
    cs_output(p, B, count) {
        for (let i = 0; i < count; i++) {
            if (i >= p.blockSize) continue;
            B.wet[2 * i] = B.src[2 * (p.blockSize + i)] * p.invN;
            B.wet[2 * i + 1] = B.src[2 * (p.blockSize + i) + 1] * p.invN;
        }
    },
};

function butterfly(p, B, count, sign) {
    const halfN = p.n >>> 1;
    const h = 1 << p.stage;
    for (let t = 0; t < count; t++) {
        if (t >= p.batch * halfN) continue;
        const b = t >>> (p.log2n - 1), q = t & (halfN - 1), j = q & (h - 1);
        const i0 = (b << p.log2n) + ((q >>> p.stage) << (p.stage + 1)) + j, i1 = i0 + h;
        const tk = j << (p.log2n - 1 - p.stage);
        const wx = B.twiddles[2 * tk], wy = sign * B.twiddles[2 * tk + 1];
        const cx = B.src[2 * i1], cy = B.src[2 * i1 + 1];
        const wcx = fr(fr(wx * cx) - fr(wy * cy)), wcy = fr(fr(wx * cy) + fr(wy * cx));
        const ax = B.src[2 * i0], ay = B.src[2 * i0 + 1];
        B.dst[2 * i0] = ax + wcx; B.dst[2 * i0 + 1] = ay + wcy;
        B.dst[2 * i1] = ax - wcx; B.dst[2 * i1 + 1] = ay - wcy;
    }
}

const BINDING_NAMES = ['timeIn', 'src', 'dst', 'twiddles', 'spectra', 'filters', 'wet'];

function emulatingDevice() {
    installGPUGlobals();
    const stats = { dispatches: {}, submits: 0 };
    const view = (res) => new Float32Array(res.buffer.data, res.offset ?? 0);
    function dispatch(pipeline, group, dyn, x) {
        const entry = pipeline.entryPoint;
        stats.dispatches[entry] = (stats.dispatches[entry] ?? 0) + 1;
        const B = {};
        let params;
        for (const e of group.entries) {
            if (e.binding === 7) params = decodeStruct(new Uint8Array(e.resource.buffer.data), PARAMS, (e.resource.offset ?? 0) + (dyn[0] ?? 0));
            else B[BINDING_NAMES[e.binding]] = view(e.resource);
        }
        KERNELS[entry](params, B, x * WORKGROUP);
    }
    const device = {
        stats,
        limits: {
            minUniformBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30,
            maxStorageBuffersPerShaderStage: 8, maxComputeWorkgroupsPerDimension: 65535,
        },
        lost: new Promise(() => {}),
        queue: {
            writeBuffer(buf, offset, data, dataOffset = 0, size) {
                const bpe = data.BYTES_PER_ELEMENT ?? 1;
                const src = ArrayBuffer.isView(data)
                    ? new Uint8Array(data.buffer, data.byteOffset + dataOffset * bpe, size === undefined ? data.byteLength - dataOffset * bpe : size * bpe)
                    : new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
                new Uint8Array(buf.data, offset, src.length).set(src);
            },
            submit(cmds) { stats.submits++; for (const c of cmds) c(); },
            onSubmittedWorkDone: () => Promise.resolve(),
        },
        createBuffer({ size }) {
            const data = new ArrayBuffer(size);
            return { size, data, destroy() {}, unmap() {}, mapAsync: () => Promise.resolve(),
                getMappedRange: (o = 0, s = size - o) => data.slice(o, o + s) };
        },
        createShaderModule: (d) => d,
        createBindGroupLayout: (d) => d,
        createPipelineLayout: (d) => d,
        createComputePipeline: (d) => ({ entryPoint: d.compute.entryPoint }),
        createBindGroup: (d) => d,
        createCommandEncoder() {
            const ops = [];
            return {
                beginComputePass() {
                    let pipe, group, dyn;
                    return {
                        setPipeline(p) { pipe = p; },
                        setBindGroup(_, g, d = []) { group = g; dyn = d; },
                        dispatchWorkgroups(x) { const a = [pipe, group, dyn]; ops.push(() => dispatch(...a, x)); },
                        end() {},
                    };
                },
                clearBuffer(b) { ops.push(() => new Uint8Array(b.data).fill(0)); },
                copyBufferToBuffer(s, so, d, dOff, n) {
                    ops.push(() => new Uint8Array(d.data, dOff, n).set(new Uint8Array(s.data, so, n)));
                },
                finish: () => () => ops.forEach(f => f()),
            };
        },
    };
    return device;
}

// ---------------------------------------------------------------------------------------------

test('Params writer bytes match WGSL struct Params (every stage slot)', (t) => {
    t.diagnostic(`struct Params: ${PARAMS.size} bytes, offsets ${JSON.stringify(PARAMS.offsets)}`);
    assert.equal(PARAMS.size, 36, 'PARAMS_SIZE = 36 (minBindingSize) in convolution-reverb.js');
    const device = mockDevice();
    const conv = new rv.GPUConvolver(device);
    conv.configure(2048);
    conv.partitions = 123;
    conv.writeParams(conv.log2n, 7, 456);
    const w = device.lastWrite(conv.paramsBuffer);
    assert.equal(w.bytes.length, conv.log2n * conv.paramsStride);
    for (let s = 0; s < conv.log2n; s++) {
        assert.deepEqual(decodeStruct(w.bytes, PARAMS, s * conv.paramsStride), {
            n: 4096, log2n: 12, stage: s, blockSize: 2048, batch: 7, partitions: 123, head: 456,
            ring: Math.ceil(576000 / 2048), invN: 1 / 4096,
        }, `slot ${s}`);
    }
});

test('View writer bytes match WGSL struct View', (t) => {
    const layout = structLayout(moduleCode(rv, 'render'), 'View');
    t.diagnostic(`struct View: ${layout.size} bytes, offsets ${JSON.stringify(layout.offsets)}`);
    assert.equal(layout.size, 80, 'VIEW_SIZE = 80 in convolution-reverb.js');
    const device = mockDevice();
    const r = Object.create(rv.ConvolutionReverb.prototype);
    Object.assign(r, {
        device, viewBuffer: device.createBuffer({ size: 80 }), view: new ArrayBuffer(80),
        canvas: { width: 1024, height: 512 }, irInfo: { seconds: 3.25, rt60: 2.125 }, space: 'hall',
        levels: new Float32Array([0.5, 0.25, 0.125, 0.0625]), holds: new Float32Array([1, 2, 3, 4]),
        outLevel: 0.75, outHold: 0.875, scopeGains: [3, 5], preDelay: 0.0625, hasScope: true,
    });
    r.viewF32 = new Float32Array(r.view);
    r.viewU32 = new Uint32Array(r.view);
    r.writeView();
    const v = decodeStruct(device.lastWrite(r.viewBuffer).bytes, layout);
    assert.deepEqual(v, {
        canvasSize: [1024, 512], irSeconds: 3.25, rt60: 2.125, levels: [0.5, 0.25, 0.125, 0.0625],
        holds: [1, 2, 3, 4], outLevel: 0.75, outHold: 0.875, gainIn: 3, gainOut: 5, preDelay: 0.0625, hasScope: 1,
    });
    r.space = 'reverse';
    r.hasScope = false;
    r.writeView();
    const v2 = decodeStruct(device.lastWrite(r.viewBuffer).bytes, layout);
    assert.equal(v2.rt60, 0, 'no RT60 readout for the reversed IR');
    assert.equal(v2.hasScope, 0);
});

test('GPUConvolver (replica kernels) streaming output equals direct convolution, across the ring wrap', async (t) => {
    const device = emulatingDevice();
    const conv = new rv.GPUConvolver(device);
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
        // f32 FFT of size 2B plus f32 MAC: ~1e-7 of the peak; 1e-5 catches any bookkeeping error
        assert.ok(rel < 1e-5, `relative error ${rel}`);
    }
    t.diagnostic(`dispatches: ${JSON.stringify(device.stats.dispatches)}`);
});

test('GPUConvolver.selfTest() on generated IRs (shipped self-test, replica kernels)', async (t) => {
    const device = emulatingDevice();
    const conv = new rv.GPUConvolver(device);
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

test('uploadIR partition bookkeeping; configure() rejects unsupported block sizes', () => {
    const conv = new rv.GPUConvolver(mockDevice());
    assert.throws(() => conv.configure(512), /Unsupported block size/);
    conv.configure(1024);
    const z = (n) => new Float32Array(n);
    for (const [len, P] of [[1, 1], [1024, 1], [1025, 2], [4096, 4], [10000, 10]]) {
        conv.uploadIR(z(len), z(len), 48000);
        assert.equal(conv.partitions, P, `L=${len}`);
    }
    conv.uploadIR(z(600000), z(600000), 96000);   // longer than the delay line: truncated to R partitions
    assert.equal(conv.partitions, conv.ring);
    assert.equal(conv.ring, Math.ceil(576000 / 1024));
    const v = conv.irVersion;
    conv.configure(2048);   // re-partitions the stored IR for the new block size
    assert.equal(conv.partitions, Math.ceil(576000 / 2048));
    assert.equal(conv.irVersion, v + 1);
    assert.equal(conv.blockIndex, 0);
});

test('directBlock matches full direct convolution', () => {
    const rand = mulberry32(3);
    const B = 64;
    const blocks = [0, 1, 2].map(() => Float32Array.from({ length: B }, () => rand() * 2 - 1));
    const h = Float32Array.from({ length: 150 }, () => rand() - 0.5);
    const x = new Float32Array(3 * B);
    blocks.forEach((b, i) => x.set(b, i * B));
    const ref = convolve(x, h, 5 * B);
    for (let m = 0; m < 3; m++) {
        const y = rv.directBlock(blocks, h, m);
        for (let i = 0; i < B; i++) assert.ok(Math.abs(y[i] - ref[m * B + i]) < 1e-12);
    }
});

test('schroeder / measureRT60 recover a known exponential decay', (t) => {
    const fs = 48000;
    for (const rt of [0.3, 1.0, 2.5]) {
        const length = Math.round(1.2 * rt * fs);
        const rand = mulberry32(17);
        const d = Math.pow(10, -3 / (rt * fs));   // amplitude -60 dB after rt seconds
        const l = new Float32Array(length), r = new Float32Array(length);
        for (let i = 0, e = 1; i < length; i++, e *= d) { l[i] = (rand() * 2 - 1) * e; r[i] = (rand() * 2 - 1) * e; }
        const edc = rv.schroeder(l, r);
        assert.equal(edc.length, length + 1);
        assert.equal(edc[length], 0);
        for (let i = 1; i <= length; i++) assert.ok(edc[i] <= edc[i - 1]);
        const m = rv.measureRT60(l, r, fs, edc);
        t.diagnostic(`RT60 ${rt} s -> measured ${m.toFixed(4)} s (${((m / rt - 1) * 100).toFixed(2)}%)`);
        // T30 fit on noise: a few % scatter; 5% bound
        assert.ok(Math.abs(m / rt - 1) < 0.05);
    }
    assert.equal(rv.measureRT60(new Float32Array(100), new Float32Array(100), fs), null, 'silence has no RT60');
});

test('generateImpulseResponse: length, pre-delay, normalization, RT60 per space', (t) => {
    const fs = 48000;
    for (const [key, s] of Object.entries(rv.SPACES)) {
        const ir = rv.generateImpulseResponse(key, { sampleRate: fs, rt60: s.rt60, preDelay: s.preDelay });
        const expectedLen = Math.round(rv.irSeconds(s, s.rt60, s.preDelay) * fs);
        assert.equal(ir.left.length, expectedLen);
        assert.equal(ir.seconds, expectedLen / fs);
        let e = 0;
        for (let i = 0; i < ir.left.length; i++) e += ir.left[i] ** 2 + ir.right[i] ** 2;
        const norm = Math.sqrt(e / 2);
        assert.ok(Math.abs(norm - 0.7) < 1e-5, `${key} norm ${norm}`);   // IR_NORM
        const first = ir.left.findIndex(v => v !== 0);
        const rt = rv.measureRT60(ir.left, ir.right, fs);
        if (s.reverse) {
            const q = ir.left.length >> 2;
            let head = 0, tail = 0;
            for (let i = 0; i < q; i++) { head += ir.left[i] ** 2; tail += ir.left[ir.left.length - 2 * q + i] ** 2; }
            t.diagnostic(`${key}: ${ir.seconds.toFixed(2)} s, energy rises (2nd-last/1st quarter ${(tail / head).toFixed(1)}x)`);
            assert.ok(tail > 10 * head, 'reversed tail swells');
            continue;
        }
        t.diagnostic(`${key}: ${ir.seconds.toFixed(2)} s, onset ${first} (pre-delay ${Math.round(s.preDelay * fs)}), RT60 ${rt.toFixed(3)} s vs mid-band ${s.rt60} s (${((rt / s.rt60 - 1) * 100).toFixed(1)}%)`);
        assert.ok(first >= Math.round(s.preDelay * fs), 'nothing before the pre-delay');
        assert.ok(first <= Math.round(s.preDelay * fs) + 2);
        // Broadband T30 mixes three bands (low ratio up to 1.3x, high down to 0.4x): within 25% of mid
        assert.ok(Math.abs(rt / s.rt60 - 1) < 0.25, `${key} RT60 ${rt}`);
    }
    // RT60 slider scales the decay
    const a = rv.generateImpulseResponse('hall', { sampleRate: fs, rt60: 1.0, preDelay: 0.01 });
    const b = rv.generateImpulseResponse('hall', { sampleRate: fs, rt60: 2.0, preDelay: 0.01 });
    const ra = rv.measureRT60(a.left, a.right, fs), rb = rv.measureRT60(b.left, b.right, fs);
    t.diagnostic(`hall rt60 1.0 -> ${ra.toFixed(3)} s, 2.0 -> ${rb.toFixed(3)} s (ratio ${(rb / ra).toFixed(3)})`);
    assert.ok(Math.abs(rb / ra - 2) < 0.1);
    assert.equal(rv.irSeconds(rv.SPACES.cathedral, 8, 0.5), 6, 'capped at 6 s');
    assert.equal(rv.irSeconds(rv.SPACES.room, 0.1, 0), 1, 'at least 1 s');
});

test('prepareUploadedIR: mono duplication, pre-delay, truncation, normalization', () => {
    const fs = 8000;
    const mono = Float32Array.from({ length: 1000 }, (_, i) => Math.exp(-i / 200));
    const ir = rv.prepareUploadedIR([mono], fs, 0.01);
    assert.equal(ir.left.length, 1080);
    assert.equal(ir.left[79], 0);
    assert.ok(ir.left[80] > 0);
    assert.deepEqual(ir.left, ir.right);
    let e = 0;
    for (let i = 0; i < ir.left.length; i++) e += ir.left[i] ** 2 + ir.right[i] ** 2;
    assert.ok(Math.abs(Math.sqrt(e / 2) - 0.7) < 1e-6);
    assert.equal(ir.truncated, false);
    const long = rv.prepareUploadedIR([new Float32Array(7 * fs).fill(0.1), new Float32Array(7 * fs).fill(-0.1)], fs, 0);
    assert.equal(long.left.length, 6 * fs);
    assert.equal(long.truncated, true);
    assert.ok(long.right[0] < 0, 'second channel kept');
});

test('analyzeImpulseResponse display data is consistent', () => {
    const fs = 48000;
    const ir = rv.generateImpulseResponse('room', { sampleRate: fs, rt60: 0.45, preDelay: 0.004 });
    const a = rv.analyzeImpulseResponse(ir.left, ir.right, fs);
    assert.equal(a.env.length, 640 * 4);
    let maxAbs = 0;
    for (const v of a.env) maxAbs = Math.max(maxAbs, Math.abs(v));
    assert.ok(Math.abs(maxAbs - 1) < 1e-6, 'envelope normalized to the peak');
    assert.equal(a.edc[0], 0);
    for (let c = 1; c < a.edc.length; c++) assert.ok(a.edc[c] <= a.edc[c - 1]);
    assert.ok(Math.abs(Math.max(...a.spec)) < 1e-6, 'spectrogram is dB re its maximum');
    assert.equal(a.rt60, rv.measureRT60(ir.left, ir.right, fs));
    assert.equal(a.seconds, ir.seconds);
});

test('end to end: worklet bridge + ReverbStream + GPUConvolver give (x*h) delayed by (1+D)·B, no underruns', async (t) => {
    const B = 1024, depth = 1, fs = 48000;
    // Load the shipped AudioWorkletProcessor source with a minimal worklet global scope
    let Processor;
    let pendingPort;
    class AudioWorkletProcessor { constructor() { this.port = pendingPort; } }
    new Function('AudioWorkletProcessor', 'registerProcessor', rv.WORKLET_SOURCE)(
        AudioWorkletProcessor, (name, cls) => { assert.equal(name, 'reverb-bridge'); Processor = cls; });

    // MessageChannel-like pair with asynchronous delivery
    const mkPort = () => ({ onmessage: null, other: null,
        postMessage(data) { const o = this.other; queueMicrotask(() => o.onmessage && o.onmessage({ data })); } });
    const workletPort = mkPort(), mainPort = mkPort();
    workletPort.other = mainPort; mainPort.other = workletPort;
    pendingPort = workletPort;
    const proc = new Processor({ processorOptions: { blockSize: B, depth, ringSize: 16 } });

    const conv = new rv.GPUConvolver(emulatingDevice());
    conv.configure(B);
    const [hl, hr] = noiseIR(2 * B, 8, 300);
    conv.uploadIR(hl, hr, fs);
    const stream = new rv.ReverbStream(conv, mainPort, null);

    const frames = 8 * B, quantum = 128;
    const rand = mulberry32(21);
    const xl = new Float32Array(frames), xr = new Float32Array(frames);
    // Quiet input (±0.1): the worklet clamps wet samples to ±4 to protect the speakers
    for (let i = 0; i < 3 * B; i++) { xl[i] = 0.1 * (rand() * 2 - 1); xr[i] = 0.1 * (rand() * 2 - 1); }
    const yl = new Float32Array(frames), yr = new Float32Array(frames);
    for (let f = 0; f < frames; f += quantum) {
        const outL = new Float32Array(quantum), outR = new Float32Array(quantum);
        proc.process([[xl.subarray(f, f + quantum), xr.subarray(f, f + quantum)]], [[outL, outR]]);
        yl.set(outL, f); yr.set(outR, f);
        await new Promise(r => setImmediate(r));   // let the main thread / GPU promises run
    }
    const delay = (1 + depth) * B;
    let maxErr = 0, peak = 0;
    [[xl, hl, yl], [xr, hr, yr]].forEach(([x, h, y]) => {
        const ref = convolve(x, h, frames);
        for (let n = 0; n < frames; n++) {
            const want = n >= delay ? ref[n - delay] : 0;
            maxErr = Math.max(maxErr, Math.abs(y[n] - want));
            peak = Math.max(peak, Math.abs(want));
        }
    });
    t.diagnostic(`latency ${delay} frames, played ${proc.played}, underruns ${proc.underruns}, late ${proc.late}, processed ${stream.processed}; max error ${(maxErr / peak).toExponential(2)} of peak`);
    assert.equal(stream.error, null);
    assert.equal(proc.underruns, 0);
    assert.equal(proc.late, 0);
    assert.equal(proc.played, (frames - delay) / B);
    assert.ok(peak < 4, 'test signal stays below the worklet clamp');
    assert.ok(maxErr / peak < 1e-5);
    stream.close();
});
