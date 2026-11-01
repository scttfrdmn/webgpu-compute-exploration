// Measure experiments for parallel sorting (js/examples/parallel-sort.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureSort), which runs the shipped
// benchmark (runBenchmark: three sorts of the same random keys, one readback, a bit-for-bit check
// against TypedArray.prototype.sort) on a ParallelSort without a canvas. The CPU reference is that
// built-in sort (js/refs/sort.js cpuSort is the same call). Every sweep is a fixed list with hard caps
// (exported so test/parallel-sort-measure.test.mjs can check them): N ≤ 4,194,304 (the benchmark's
// largest size and the radix sort's limit), at most three algorithms × three sorts per size.

import { measureSort, MEASURE_MAX_KEYS } from '../../examples/parallel-sort.js';

export const COUNTS = [16384, 65536, 262144, 1048576, 4194304];
export const MAX_N = 4194304;
export const ALGORITHMS = ['bitonic', 'bitonic-shared', 'radix'];
const KEY = { bitonic: 'bitonic', 'bitonic-shared': 'shared', radix: 'radix' };

// Measured in CI (software GPU, test/gpu/parallel-sort-measure.gpu.test.mjs, N ≤ 262,144, u32 and f32)
export const CI_VERIFIED = 'every sort bit-identical to the CPU sort';

const ms = (v) => v.toFixed(2);

const scaling = {
    id: 'scaling',
    title: 'Sorting N keys: bitonic, bitonic with workgroup memory, radix, CPU',
    question: 'How do O(n log² n) bitonic sort and O(n) radix sort scale on your GPU, and at what N do they overtake the CPU\'s built-in sort?',
    settings: [
        { key: 'dataType', label: 'Keys', options: [{ value: 'u32', label: 'u32' }, { value: 'f32', label: 'f32 (encode → sort → decode)' }], default: 'u32' },
        { key: 'maxN', label: 'Largest N', options: [262144, 1048576, 4194304].map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 4194304 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'N' },
        { key: 'bitonicGpu', label: 'bitonic GPU ms', format: ms },
        { key: 'sharedGpu', label: 'bitonic (workgroup) GPU ms', format: ms },
        { key: 'radixGpu', label: 'radix GPU ms', format: ms },
        { key: 'cpuMs', label: 'CPU sort ms', format: ms },
        { key: 'bitonicWall', label: 'bitonic wall ms', format: ms },
        { key: 'sharedWall', label: 'bitonic (workgroup) wall ms', format: ms },
        { key: 'radixWall', label: 'radix wall ms', format: ms },
        { key: 'radixKeysPerUs', label: 'radix M keys/s (wall)', format: (v) => v.toFixed(0) },
        { key: 'dispatches', label: 'dispatches (bitonic / workgroup / radix)', format: String },
        { key: 'verified', label: 'verified vs CPU', format: String },
    ],
    charts: [
        { x: 'n', xLabel: 'keys N', logX: true, y: ['bitonicGpu', 'sharedGpu', 'radixGpu', 'cpuMs'], yLabel: 'ms per sort', logY: true,
            labels: { bitonicGpu: 'bitonic (GPU)', sharedGpu: 'bitonic, workgroup memory (GPU)', radixGpu: 'radix (GPU)', cpuMs: 'CPU sort' },
            title: 'GPU time per sort vs the CPU sort (needs timestamp-query)',
            emptyNote: 'GPU times need the timestamp-query feature; the wall-time chart below has every run.' },
        { x: 'n', xLabel: 'keys N', logX: true, y: ['bitonicWall', 'sharedWall', 'radixWall', 'cpuMs'], yLabel: 'wall ms per sort', logY: true,
            labels: { bitonicWall: 'bitonic', sharedWall: 'bitonic, workgroup memory', radixWall: 'radix', cpuMs: 'CPU sort' } },
    ],
    baseline: `CI (shipped shaders on a software GPU, N ≤ 262,144): ${CI_VERIFIED}. Work per sort at N = 2²⁰: `
        + 'global bitonic 210 dispatches of 524,288 compare-exchanges (110 M in all), with workgroup memory 78 dispatches, '
        + 'radix 24 dispatches in 8 passes (each key read twice and written once per pass). Wall ms on the CPU-emulated GPU '
        + '(u32, bitonic / workgroup / radix / CPU sort): 7.9 / 9.4 / 16 / 0.68 at N = 16,384 and 83 / 109 / 203 / 12.6 at '
        + '262,144. There, the CPU sort wins by 7–12×, and workgroup memory and radix sort lose because the "GPU" is CPU '
        + 'cores, where barriers are expensive and workgroup memory is no faster than any other. On a real GPU expect the '
        + 'workgroup-memory variant to beat the global one; whether radix sort beats both depends on N and the GPU.',
    async *run(ctx, { dataType, maxN }) {
        for (const n of COUNTS) {
            if (n > Math.min(maxN, MAX_N, MEASURE_MAX_KEYS)) break;
            const row = { n, cpuMs: null, verified: '' };
            let ok = 0;
            const cpu = [];
            const dispatches = [];
            for (const algorithm of ALGORITHMS) {
                await ctx.step();
                const r = await measureSort(ctx.gpuDevice, { algorithm, n, dataType });
                row[`${KEY[algorithm]}Gpu`] = r.bestKernelMs;
                row[`${KEY[algorithm]}Wall`] = r.bestMs;
                if (r.mismatches === 0 && r.inversions === 0) ok++;
                cpu.push(r.cpuMs);
                dispatches.push(r.dispatches);
            }
            row.cpuMs = Math.min(...cpu);
            row.radixKeysPerUs = row.radixWall > 0 ? n / row.radixWall / 1000 : null;
            row.dispatches = dispatches.join(' / ');
            row.verified = ok === ALGORITHMS.length ? 'yes' : `NO (${ALGORITHMS.length - ok} wrong)`;
            yield row;
        }
    },
};

export default [scaling];
