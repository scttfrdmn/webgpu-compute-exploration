import { createBuffer, readBuffer } from '../webgpu-utils.js';

const shaderCode = `
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&input1)) {
        output[index] = input1[index] + input2[index];
    }
}
`;

export async function runVectorAddition(gpuDevice) {
    const { device } = gpuDevice;
    const startTime = performance.now();

    const size = 1000000;
    const input1 = new Float32Array(size);
    const input2 = new Float32Array(size);

    for (let i = 0; i < size; i++) {
        input1[i] = i;
        input2[i] = i * 2;
    }

    const input1Buffer = createBuffer(
        device,
        input1,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const input2Buffer = createBuffer(
        device,
        input2,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const outputBuffer = device.createBuffer({
        size: input1.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

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
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: input1Buffer } },
            { binding: 1, resource: { buffer: input2Buffer } },
            { binding: 2, resource: { buffer: outputBuffer } }
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
    passEncoder.dispatchWorkgroups(Math.ceil(size / 64));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    const result = await readBuffer(device, outputBuffer, input1.byteLength);

    const endTime = performance.now();

    let isCorrect = true;
    for (let i = 0; i < Math.min(100, size); i++) {
        if (Math.abs(result[i] - (input1[i] + input2[i])) > 0.001) {
            isCorrect = false;
            break;
        }
    }

    input1Buffer.destroy();
    input2Buffer.destroy();
    outputBuffer.destroy();

    return `<span class="success">✓ Vector Addition Complete</span>

<span class="info">Configuration:</span>
• Vector size: ${size.toLocaleString()} elements
• Workgroup size: 64
• Total workgroups: ${Math.ceil(size / 64).toLocaleString()}

<span class="info">Results:</span>
• Computation time: ${(endTime - startTime).toFixed(2)}ms
• Verification: ${isCorrect ? '<span class="success">PASSED</span>' : '<span class="error">FAILED</span>'}
• Sample results:
  input1[0] + input2[0] = ${input1[0]} + ${input2[0]} = ${result[0]}
  input1[10] + input2[10] = ${input1[10]} + ${input2[10]} = ${result[10]}
  input1[100] + input2[100] = ${input1[100]} + ${input2[100]} = ${result[100]}

<span class="info">Performance:</span>
• Elements per ms: ${(size / (endTime - startTime)).toFixed(0).toLocaleString()}
• Throughput: ${((size * 4) / (endTime - startTime) / 1024).toFixed(2)} MB/ms`;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'compute', code: shaderCode }]; }
