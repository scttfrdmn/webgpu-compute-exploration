import { createBuffer, readBuffer } from '../webgpu-utils.js';
import { GpuTimer, GPU_TIME_NOTE } from '../gpu-timer.js';

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
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'compute', code: shaderCode }]; }

// ---- Compute-only API for the Measure tab (js/teach/experiments/matrix-multiplication.js) and test/gpu/ ----
//
// measureMatmul() runs the shipped shader (same explicit layout and 8 × 8 dispatch as
// runMatrixMultiplication) on caller-supplied matrices. The inputs are uploaded once; only the kernel
// is timed, by GPU timestamps and by the CPU (submit → onSubmittedWorkDone); the product is read back
// afterwards for the accuracy check.

export const TILE = 8;                   // @workgroup_size(8, 8)
export const MAX_MATMUL_DIM = 1024;      // M, N, K caps: 1024³ is 2.1 GFLOP in one dispatch
export const MAX_MEASURE_REPEATS = 10;

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((p, q) => p - q);
    return v.length ? v[v.length >> 1] : null;
};

// A: M×N and B: N×K Float32Arrays, row-major. Resolves to { M, N, K, flops, gpuMs, wallMs, C } with
// flops = 2·M·N·K, medians over `repeats` timed dispatches (after `warmup` untimed ones; ≤ 10 each),
// gpuMs null without timestamp-query, and C (M×K Float32Array) when readback is true.
export async function measureMatmul(gpuDevice, { A, B, M, N, K, warmup = 1, repeats = 3, readback = true }) {
    const device = gpuDevice.device ?? gpuDevice;
    for (const [name, v] of [['M', M], ['N', N], ['K', K]]) {
        if (!Number.isInteger(v) || v < 1 || v > MAX_MATMUL_DIM) throw new RangeError(`${name} = ${v} is outside 1 … ${MAX_MATMUL_DIM}`);
    }
    if (A?.length !== M * N || B?.length !== N * K) throw new RangeError('A must be M×N and B N×K');
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const aBuffer = device.createBuffer({ size: 4 * M * N, usage });
    const bBuffer = device.createBuffer({ size: 4 * N * K, usage });
    const cBuffer = device.createBuffer({ size: 4 * M * K, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const dimsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const timer = new GpuTimer(gpuDevice, { maxPasses: 1, poolSize: 1 });
    try {
        device.queue.writeBuffer(aBuffer, 0, A);
        device.queue.writeBuffer(bBuffer, 0, B);
        device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([M, N, K, 0]));   // vec3<u32> M, N, K (+ pad)
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
            compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint: 'main' },
        });
        const bindGroup = device.createBindGroup({
            layout,
            entries: [aBuffer, bBuffer, cBuffer, dimsBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const timedRuns = Math.min(Math.max(repeats, 1), MAX_MEASURE_REPEATS);
        const total = Math.min(Math.max(warmup, 0), MAX_MEASURE_REPEATS) + timedRuns;
        const gpu = [], wall = [];
        for (let r = 0; r < total; r++) {
            const timed = r >= total - timedRuns;
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass(timed ? timer.computePassDescriptor('matmul') : undefined);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(M / TILE), Math.ceil(K / TILE));   // x covers rows, y columns
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
        const C = readback ? await readBuffer(device, cBuffer, 4 * M * K) : undefined;
        return { M, N, K, flops: 2 * M * N * K, gpuMs: median(gpu), wallMs: median(wall), C };
    } finally {
        timer.destroy();
        for (const buffer of [aBuffer, bBuffer, cBuffer, dimsBuffer]) buffer.destroy();
    }
}
