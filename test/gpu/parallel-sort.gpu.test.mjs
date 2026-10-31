// GPU execution of parallel-sort's shipped WGSL (js/examples/parallel-sort.js): bitonic sort
// (cs_bitonic_global, and cs_bitonic_local_sort / cs_bitonic_local_merge in the workgroup-memory
// variant), LSD radix sort (cs_radix_histogram / cs_radix_scan / cs_radix_scatter) and the f32 key
// conversion (cs_encode_f32 / cs_decode_f32). CI-only on a software GPU — see test/gpu/harness.mjs.
//
// Mirrors ParallelSort.runBenchmark: the example's explicit bind group layouts (uniforms with
// hasDynamicOffset, one 256-byte slot per stage/pass), the exact uniform bytes its writers produce
// (captured through the mock device), PAD_KEY padding to a power of two for bitonic, ping-pong
// keysA/keysB for radix, and every dispatch of a sort encoded in one compute pass. The harness'
// computePipeline() builds layouts without dynamic offsets, so pipelines are created locally here.
// Results must be bit-identical to TypedArray.prototype.sort.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device, buffer, write, read } from './harness.mjs';
import { sortInternals as S, ParallelSort, wgslModules } from '../../js/examples/parallel-sort.js';
import { mockDevice } from '../helpers/mock-gpu.mjs';
import { rng, edgeCases, cpuSort } from '../helpers/parallel-sort-refs.mjs';

const code = name => wgslModules().find(m => m.name === name).code;
const PARAMS_SIZE = 16;
const stride = device.limits.minUniformBufferOffsetAlignment;

async function checked(what, fn) {
    device.pushErrorScope('validation');
    device.pushErrorScope('internal');
    let v;
    try { v = await fn(); } finally {
        const errs = [await device.popErrorScope(), await device.popErrorScope()].filter(Boolean);
        if (errs.length) throw new Error(`${what}: ${errs.map(e => e.message).join('\n')}`);
    }
    return v;
}

async function makePipelines(moduleCode, entries, entryPoints) {
    return checked('pipelines', async () => {
        const module = device.createShaderModule({ code: moduleCode });
        const errs = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
        if (errs.length) throw new Error(errs.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
        const layout = device.createBindGroupLayout({
            entries: entries.map((buffer, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer })),
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
        const pipelines = {};
        for (const [key, entryPoint] of Object.entries(entryPoints)) {
            pipelines[key] = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } });
        }
        return { layout, pipelines };
    });
}

// Layouts exactly as ParallelSort.setupComputePipelines declares them
const dynamicUniform = { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE };
const bitonic = await makePipelines(code('bitonic'), [{ type: 'storage' }, dynamicUniform],
    { global: 'cs_bitonic_global', localSort: 'cs_bitonic_local_sort', localMerge: 'cs_bitonic_local_merge' });
const radix = await makePipelines(code('radix'),
    [{ type: 'read-only-storage' }, { type: 'storage' }, { type: 'storage' }, dynamicUniform],
    { histogram: 'cs_radix_histogram', scan: 'cs_radix_scan', scatter: 'cs_radix_scatter' });
const convert = await makePipelines(code('convert'), [{ type: 'storage' }, { type: 'uniform' }],
    { encode: 'cs_encode_f32', decode: 'cs_decode_f32' });

// Uniform bytes from the shipped writers, via a mock device (as test/parallel-sort.test.mjs does)
const mock = mockDevice();
const sorter = new ParallelSort(mock, { width: 64, height: 64,
    getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) }) });
assert.equal(sorter.paramsStride, stride, 'mock and real device share the 256-byte uniform offset alignment');

function shippedBitonicParams(stages, padded) {
    const buf = mock.createBuffer({ size: Math.max(stages.length, 1) * stride, usage: GPUBufferUsage.UNIFORM });
    sorter.writeBitonicParams(buf, stages, padded);
    return mock.lastWrite(buf).bytes;
}

async function shippedRadixParams(n) {
    mock.writes.length = 0;
    await sorter.runBenchmark('radix', n, 'u32');
    const isUniform = w => (w.buffer.usage & GPUBufferUsage.UNIFORM) !== 0;
    const w = mock.writes.find(x => isUniform(x) && x.buffer.size === S.RADIX_PASSES * stride);
    const c = mock.writes.find(x => isUniform(x) && x.buffer.size === PARAMS_SIZE);
    return { radixBytes: w.bytes, convertBytes: c.bytes };
}

// One sort, encoded like runBenchmark's encodeSort (one compute pass), optionally with the f32
// encode before and decode after. Returns the n sorted keys read back from keysA.
async function gpuSort(algorithm, inputBits, f32 = false) {
    const n = inputBits.length;
    const isRadix = algorithm === 'radix';
    const padded = isRadix ? n : S.nextPowerOfTwo(n);
    const keysA = buffer(padded * 4);
    write(keysA, inputBits);
    if (padded > n) write(keysA, new Uint32Array(padded - n).fill(S.PAD_KEY), n * 4);

    const { radixBytes, convertBytes } = await shippedRadixParams(n);   // ConvertParams { count: n }
    const convertParams = buffer(convertBytes);
    const convertBG = device.createBindGroup({ layout: convert.layout, entries: [
        { binding: 0, resource: { buffer: keysA } }, { binding: 1, resource: { buffer: convertParams } }] });
    const convertGrid = S.dispatchGrid(Math.ceil(n / S.WORKGROUP_SIZE), device.limits.maxComputeWorkgroupsPerDimension);

    let encodeSort, dispatches;
    if (isRadix) {
        const keysB = buffer(padded * 4);
        const numTiles = Math.ceil(n / S.RADIX_TILE);
        const histogram = buffer(S.RADIX_BUCKETS * numTiles * 4);
        const radixParams = buffer(radixBytes);
        const bgs = [[keysA, keysB], [keysB, keysA]].map(([src, dst]) => device.createBindGroup({
            layout: radix.layout, entries: [
                { binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: dst } },
                { binding: 2, resource: { buffer: histogram } },
                { binding: 3, resource: { buffer: radixParams, offset: 0, size: PARAMS_SIZE } }] }));
        const grid = S.dispatchGrid(numTiles, device.limits.maxComputeWorkgroupsPerDimension);
        dispatches = S.RADIX_PASSES * 3;
        encodeSort = pass => {
            for (let p = 0; p < S.RADIX_PASSES; p++) {
                pass.setBindGroup(0, bgs[p % 2], [p * stride]);
                pass.setPipeline(radix.pipelines.histogram); pass.dispatchWorkgroups(grid[0], grid[1]);
                pass.setPipeline(radix.pipelines.scan); pass.dispatchWorkgroups(1);
                pass.setPipeline(radix.pipelines.scatter); pass.dispatchWorkgroups(grid[0], grid[1]);
            }
        };
    } else {
        const stages = S.bitonicSchedule(padded, algorithm === 'bitonic-shared');
        const stageParams = buffer(shippedBitonicParams(stages, padded));
        const bg = device.createBindGroup({ layout: bitonic.layout, entries: [
            { binding: 0, resource: { buffer: keysA } },
            { binding: 1, resource: { buffer: stageParams, offset: 0, size: PARAMS_SIZE } }] });
        const grid = S.dispatchGrid(Math.ceil(padded / 2 / S.WORKGROUP_SIZE), device.limits.maxComputeWorkgroupsPerDimension);
        dispatches = stages.length;
        encodeSort = pass => stages.forEach((stage, s) => {
            pass.setPipeline(bitonic.pipelines[stage.kind]);
            pass.setBindGroup(0, bg, [s * stride]);
            pass.dispatchWorkgroups(grid[0], grid[1]);
        });
    }

    await checked(`${algorithm} n=${n}`, async () => {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        if (f32) { pass.setPipeline(convert.pipelines.encode); pass.setBindGroup(0, convertBG); pass.dispatchWorkgroups(...convertGrid); }
        encodeSort(pass);
        if (f32) { pass.setPipeline(convert.pipelines.decode); pass.setBindGroup(0, convertBG); pass.dispatchWorkgroups(...convertGrid); }
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
    });
    const out = await read(keysA, Uint32Array, n * 4);
    return { out, dispatches: dispatches + (f32 ? 2 : 0) };
}

function mismatches(a, b) {
    let m = 0;
    for (let i = 0; i < b.length; i++) if (a[i] !== b[i]) m++;
    return m;
}

const cases = edgeCases(11);
delete cases.empty;   // the page never sorts 0 keys (and a 0-byte histogram binding is invalid)
const r = rng(12);
cases.random70001 = Uint32Array.from({ length: 70001 }, () => (r() * 0x100000000) >>> 0);   // 69 tiles; padded 131072

for (const algorithm of ['bitonic', 'bitonic-shared', 'radix']) {
    test(`parallel-sort ${algorithm}: u32 edge cases and random keys match Uint32Array.sort`, async (t) => {
        let bad = 0;
        for (const [name, input] of Object.entries(cases)) {
            if (algorithm === 'bitonic' && input.length > 5000) continue;   // 153 global stages at 131072: keep CI short
            const { out, dispatches } = await gpuSort(algorithm, input);
            const m = mismatches(out, cpuSort(input));
            t.diagnostic(`${name}: n=${input.length} dispatches=${dispatches} mismatches=${m}`);
            if (m) bad++;
            assert.equal(m, 0, `${algorithm} ${name}`);
        }
        assert.equal(bad, 0);
    });
}

test('parallel-sort f32 keys (encode -> sort -> decode) are bit-identical to Float32Array.sort', async (t) => {
    const rf = rng(13);
    const specials = [0, -0, 1, -1, Infinity, -Infinity, 3.4028234663852886e38, -3.4028234663852886e38,
        1.401298464324817e-45, -1.401298464324817e-45, 1.1754943508222875e-38, -1.1754943508222875e-38, 0.5, -0.5];
    const f = new Float32Array(3001);
    for (let i = 0; i < f.length; i++) f[i] = i < specials.length * 4 ? specials[i % specials.length] : (rf() * 2 - 1) * 1e6;
    const bits = new Uint32Array(f.buffer);
    const expected = new Uint32Array(f.slice().sort().buffer);   // -0 before +0
    for (const algorithm of ['bitonic-shared', 'radix']) {
        const { out, dispatches } = await gpuSort(algorithm, bits, true);
        const m = mismatches(out, expected);
        t.diagnostic(`${algorithm}: 3001 f32 keys incl. ±0, ±Inf, ±denormal, ±FLT_MAX, dispatches=${dispatches} mismatches=${m}`);
        assert.equal(m, 0, algorithm);
    }
});
