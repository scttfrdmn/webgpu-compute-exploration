import { createBuffer, readBuffer } from '../webgpu-utils.js';
import { GpuTimer, GPU_TIME_NOTE } from '../gpu-timer.js';
import { PhaseClock } from '../phase-clock.js';

const shaderCode = `
@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // M, N, K

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;

    if (row >= M || col >= K) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < N; i = i + 1u) {
        let a = matrixA[row * N + i];
        let b = matrixB[i * K + col];
        sum = sum + a * b;
    }

    result[row * K + col] = sum;
}
`;

// Tiled kernel (Measure tab only): classic workgroup-memory tiling, T = 16, one output per thread, no
// register blocking. Algorithm:
//   - The 16 × 16 workgroup at workgroup_id (x, y) owns the 16 × 16 block of C with rows 16·y … 16·y + 15
//     and columns 16·x … 16·x + 15; thread (lx, ly) owns c[row, col] with row = 16·y + ly, col = 16·x + lx.
//     (x runs along columns, so neighbouring threads read neighbouring addresses of B and write
//     neighbouring addresses of C.)
//   - The k loop is split into ⌈N / 16⌉ phases. In phase t every thread loads ONE element of A
//     (a[row, 16·t + lx]) and ONE of B (b[16·t + ly, col]) into workgroup memory; workgroupBarrier() makes
//     the two 16 × 16 tiles visible to the whole workgroup; then each thread does 16 multiply-adds from
//     workgroup memory; a second barrier keeps the next phase from overwriting tiles still being read.
//   - Each value loaded from global memory is used by 16 threads: 2 global loads per 16 multiply-adds
//     instead of 2 per 1 in the naive kernel (arithmetic intensity T / 4 = 4 flop per byte instead of 0.25).
//   - Edges: loads outside A or B store 0 (a zero term changes no sum), and only threads inside C write,
//     so any M, N, K works. Every thread reaches every barrier: the phase count depends only on the
//     uniform N, and the bounds checks guard the loads and the store, never a barrier.
//   - Summation order is k = 0, 1, …, N − 1, the same as the naive kernel, so both round identically
//     (unless the compiler contracts one of them into fused multiply-adds and not the other).
const tiledShaderCode = `
const T = 16u;

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // M, N, K

var<workgroup> tileA: array<array<f32, 16>, 16>;   // tileA[ly][kk] = a[16 wg.y + ly, 16 t + kk]
var<workgroup> tileB: array<array<f32, 16>, 16>;   // tileB[kk][lx] = b[16 t + kk, 16 wg.x + lx]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;
    let row = wg.y * T + lid.y;
    let col = wg.x * T + lid.x;

    var sum = 0.0;
    let phases = (N + T - 1u) / T;
    for (var t = 0u; t < phases; t = t + 1u) {
        // Cooperative load: one element of each tile per thread, zero outside the matrices
        let ka = t * T + lid.x;
        var a = 0.0;
        if (row < M && ka < N) {
            a = matrixA[row * N + ka];
        }
        tileA[lid.y][lid.x] = a;
        let kb = t * T + lid.y;
        var b = 0.0;
        if (kb < N && col < K) {
            b = matrixB[kb * K + col];
        }
        tileB[lid.y][lid.x] = b;
        workgroupBarrier();

        // 16 multiply-adds from workgroup memory, k ascending
        for (var kk = 0u; kk < T; kk = kk + 1u) {
            sum = sum + tileA[lid.y][kk] * tileB[kk][lid.x];
        }
        workgroupBarrier();
    }

    if (row < M && col < K) {
        result[row * K + col] = sum;
    }
}
`;

export async function runMatrixMultiplication(gpuDevice) {
    const { device } = gpuDevice;
    const startTime = performance.now();

    const M = 512;
    const N = 512;
    const K = 512;

    const matrixA = new Float32Array(M * N);
    const matrixB = new Float32Array(N * K);

    for (let i = 0; i < M * N; i++) {
        matrixA[i] = Math.random();
    }
    for (let i = 0; i < N * K; i++) {
        matrixB[i] = Math.random();
    }

    const matrixABuffer = createBuffer(
        device,
        matrixA,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const matrixBBuffer = createBuffer(
        device,
        matrixB,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const resultSize = M * K * 4;
    const resultBuffer = device.createBuffer({
        size: resultSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const dimensionsArray = new Uint32Array([M, N, K]);
    const dimensionsBuffer = createBuffer(
        device,
        dimensionsArray,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const shaderModule = device.createShaderModule({
        code: shaderCode
    });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: matrixABuffer } },
            { binding: 1, resource: { buffer: matrixBBuffer } },
            { binding: 2, resource: { buffer: resultBuffer } },
            { binding: 3, resource: { buffer: dimensionsBuffer } }
        ]
    });

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    const timer = new GpuTimer(device, { maxPasses: 1, poolSize: 1 });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass(timer.computePassDescriptor('matmul'));
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(K / 8));
    passEncoder.end();
    timer.resolve(commandEncoder);

    device.queue.submit([commandEncoder.finish()]);
    const timing = timer.read();

    const result = await readBuffer(device, resultBuffer, resultSize);

    const endTime = performance.now();
    const gpuMs = (await timing)?.total ?? null;
    timer.destroy();

    function cpuMatrixMultiply(a, b, m, n, k, sampleRow, sampleCol) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += a[sampleRow * n + i] * b[i * k + sampleCol];
        }
        return sum;
    }

    const sampleRow = 0;
    const sampleCol = 0;
    const cpuResult = cpuMatrixMultiply(matrixA, matrixB, M, N, K, sampleRow, sampleCol);
    const gpuResult = result[sampleRow * K + sampleCol];
    const isCorrect = Math.abs(cpuResult - gpuResult) < 0.01;

    matrixABuffer.destroy();
    matrixBBuffer.destroy();
    resultBuffer.destroy();
    dimensionsBuffer.destroy();

    const totalOps = 2 * M * N * K;
    const gflops = (totalOps / (endTime - startTime) / 1e6).toFixed(2);
    // Kernel-only throughput; with ~100 µs quantization a sub-ms kernel can read as 0 ms, so show "—" then.
    const gpuGflops = gpuMs > 0 ? `${(totalOps / gpuMs / 1e6).toFixed(2)} GFLOPS (kernel only, GPU time)` : '—';

    return `<span class="success">✓ Matrix Multiplication Complete</span>

<span class="info">Configuration:</span>
• Matrix A: ${M} × ${N}
• Matrix B: ${N} × ${K}
• Result Matrix: ${M} × ${K}
• Workgroup size: 8 × 8
• Total workgroups: ${Math.ceil(M / 8)} × ${Math.ceil(K / 8)}

<span class="info">Results:</span>
• Computation time: ${(endTime - startTime).toFixed(2)}ms (wall: setup, upload, kernel and readback)
• ${timer.summary(endTime - startTime)}
• Total operations: ${totalOps.toLocaleString()}
• Performance: ${gflops} GFLOPS (wall time)
${timer.supported ? `• Performance: ${gpuGflops}
` : ''}• Verification: ${isCorrect ? '<span class="success">PASSED</span>' : '<span class="error">FAILED</span>'}

<span class="info">Sample verification:</span>
• CPU result[0,0]: ${cpuResult.toFixed(6)}
• GPU result[0,0]: ${gpuResult.toFixed(6)}
• Difference: ${Math.abs(cpuResult - gpuResult).toFixed(8)}

<span class="info">Use case:</span>
Matrix multiplication is fundamental for:
• Neural network inference
• 3D transformations
• Scientific computing
• Linear algebra operations
${timer.supported ? `<span class="hint">${GPU_TIME_NOTE}</span>` : ''}`;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI. The one-shot run uses `compute` (naive); the Measure tab
// runs both kernels through measureMatmul.
export function wgslModules() {
    return [{ name: 'compute', code: shaderCode }, { name: 'tiled', code: tiledShaderCode }];
}

// ---- Compute-only API for the Measure tab (js/teach/experiments/matrix-multiplication.js) and test/gpu/ ----
//
// measureMatmul() runs one of the two kernels on caller-supplied matrices, with the same explicit layout
// (A, B read-only storage, C storage, dimensions vec3<u32> uniform) and no canvas:
//   naive  the shipped `compute` shader: 8 × 8 workgroups, global_id.x = row, ⌈M/8⌉ × ⌈K/8⌉ workgroups
//   tiled  tiledShaderCode: 16 × 16 workgroups, x = column, ⌈K/16⌉ × ⌈M/16⌉ workgroups

export const TILE = 8;                   // the naive kernel's @workgroup_size(8, 8)
export const TILED_T = 16;               // the tiled kernel's tile edge and @workgroup_size(16, 16)
export const KERNELS = {
    naive: { code: shaderCode, workgroups: (M, K) => [Math.ceil(M / TILE), Math.ceil(K / TILE)] },
    tiled: { code: tiledShaderCode, workgroups: (M, K) => [Math.ceil(K / TILED_T), Math.ceil(M / TILED_T)] },
};
export const MAX_MATMUL_DIM = 1024;      // M, N, K caps: 1024³ is 2.1 GFLOP in one dispatch
export const MAX_MEASURE_REPEATS = 10;

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((p, q) => p - q);
    return v.length ? v[v.length >> 1] : null;
};

// A: M×N and B: N×K Float32Arrays, row-major; kernel 'naive' or 'tiled'. Resolves to
// { M, N, K, kernel, flops, gpuMs, wallMs, phases, C } with flops = 2·M·N·K; gpuMs and wallMs are medians
// over `repeats` timed dispatches after `warmup` untimed ones (≤ 10 each), gpuMs from timestamp queries
// (null without timestamp-query) and wallMs from submit to onSubmittedWorkDone; C (M×K Float32Array) is
// read back when readback is true. phases are wall-clock ms as defined in js/phase-clock.js: compile
// (buffers, shader module, pipeline, bind group), upload (A, B and the dimensions), dispatch (= wallMs)
// and readback (copy, map, copy out). One call with warmup = 0, repeats = 1 is one independent
// measurement for ctx.measure (js/teach/measure.js), which then takes care of warm-up and repeats.
export async function measureMatmul(gpuDevice, { A, B, M, N, K, kernel = 'naive', warmup = 1, repeats = 3, readback = true }) {
    const device = gpuDevice.device ?? gpuDevice;
    for (const [name, v] of [['M', M], ['N', N], ['K', K]]) {
        if (!Number.isInteger(v) || v < 1 || v > MAX_MATMUL_DIM) throw new RangeError(`${name} = ${v} is outside 1 … ${MAX_MATMUL_DIM}`);
    }
    if (A?.length !== M * N || B?.length !== N * K) throw new RangeError('A must be M×N and B N×K');
    const spec = KERNELS[kernel];
    if (!spec) throw new RangeError(`unknown kernel ${kernel}`);
    const clock = new PhaseClock();
    const buffers = [];
    const timer = new GpuTimer(gpuDevice, { maxPasses: 1, poolSize: 1 });
    try {
        const { pipeline, bindGroup, aBuffer, bBuffer, cBuffer, dimsBuffer } = clock.time('compile', () => {
            const make = (size, usage) => { const b = device.createBuffer({ size, usage }); buffers.push(b); return b; };
            const input = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
            const aBuffer = make(4 * M * N, input), bBuffer = make(4 * N * K, input);
            const cBuffer = make(4 * M * K, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const dimsBuffer = make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            const layout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                ],
            });
            const pipeline = device.createComputePipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ code: spec.code }), entryPoint: 'main' },
            });
            const bindGroup = device.createBindGroup({
                layout,
                entries: [aBuffer, bBuffer, cBuffer, dimsBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
            });
            return { pipeline, bindGroup, aBuffer, bBuffer, cBuffer, dimsBuffer };
        });
        clock.time('upload', () => {
            device.queue.writeBuffer(aBuffer, 0, A);
            device.queue.writeBuffer(bBuffer, 0, B);
            device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([M, N, K, 0]));   // vec3<u32> M, N, K (+ pad)
        });
        const [wx, wy] = spec.workgroups(M, K);
        const timedRuns = Math.min(Math.max(repeats, 1), MAX_MEASURE_REPEATS);
        const total = Math.min(Math.max(warmup, 0), MAX_MEASURE_REPEATS) + timedRuns;
        const gpu = [], wall = [];
        for (let r = 0; r < total; r++) {
            const timed = r >= total - timedRuns;
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass(timed ? timer.computePassDescriptor('matmul') : undefined);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(wx, wy);
            pass.end();
            if (timed) timer.resolve(encoder);
            const t0 = performance.now();
            device.queue.submit([encoder.finish()]);
            const timing = timed ? timer.read() : null;
            await device.queue.onSubmittedWorkDone();
            if (!timed) continue;
            wall.push(performance.now() - t0);
            gpu.push((await timing)?.total ?? null);
        }
        const result = { M, N, K, kernel, flops: 2 * M * N * K, gpuMs: median(gpu), wallMs: median(wall) };
        clock.add('dispatch', result.wallMs);
        clock.add('readback', 0);
        if (readback) result.C = await clock.time('readback', () => readBuffer(device, cBuffer, 4 * M * K));
        result.phases = clock.phases;
        return result;
    } finally {
        timer.destroy();
        for (const buffer of buffers) buffer.destroy();
    }
}
