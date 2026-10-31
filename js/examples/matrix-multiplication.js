import { createBuffer, readBuffer } from '../webgpu-utils.js';

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

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(K / 8));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    const result = await readBuffer(device, resultBuffer, resultSize);

    const endTime = performance.now();

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

    return `<span class="success">✓ Matrix Multiplication Complete</span>

<span class="info">Configuration:</span>
• Matrix A: ${M} × ${N}
• Matrix B: ${N} × ${K}
• Result Matrix: ${M} × ${K}
• Workgroup size: 8 × 8
• Total workgroups: ${Math.ceil(M / 8)} × ${Math.ceil(K / 8)}

<span class="info">Results:</span>
• Computation time: ${(endTime - startTime).toFixed(2)}ms
• Total operations: ${totalOps.toLocaleString()}
• Performance: ${gflops} GFLOPS
• Verification: ${isCorrect ? '<span class="success">PASSED</span>' : '<span class="error">FAILED</span>'}

<span class="info">Sample verification:</span>
• CPU result[0,0]: ${cpuResult.toFixed(6)}
• GPU result[0,0]: ${gpuResult.toFixed(6)}
• Difference: ${Math.abs(cpuResult - gpuResult).toFixed(8)}

<span class="info">Use case:</span>
Matrix multiplication is fundamental for:
• Neural network inference
• 3D transformations
• Scientific computing
• Linear algebra operations`;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'compute', code: shaderCode }]; }
