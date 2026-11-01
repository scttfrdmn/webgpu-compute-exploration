import { createBuffer, readBuffer } from '../webgpu-utils.js';
import { GpuTimer, GPU_TIME_NOTE } from '../gpu-timer.js';

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

    const timer = new GpuTimer(device, { maxPasses: 1, poolSize: 1 });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass(timer.computePassDescriptor('add'));
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(size / 64));
    passEncoder.end();
    timer.resolve(commandEncoder);

    device.queue.submit([commandEncoder.finish()]);
    const timing = timer.read();

    const result = await readBuffer(device, outputBuffer, input1.byteLength);

    const endTime = performance.now();
    await timing;
    timer.destroy();

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
• Computation time: ${(endTime - startTime).toFixed(2)}ms (wall: setup, upload, kernel and readback)
• ${timer.summary(endTime - startTime)}
• Verification: ${isCorrect ? '<span class="success">PASSED</span>' : '<span class="error">FAILED</span>'}
• Sample results:
  input1[0] + input2[0] = ${input1[0]} + ${input2[0]} = ${result[0]}
  input1[10] + input2[10] = ${input1[10]} + ${input2[10]} = ${result[10]}
  input1[100] + input2[100] = ${input1[100]} + ${input2[100]} = ${result[100]}

<span class="info">Performance:</span>
• Elements per ms: ${(size / (endTime - startTime)).toFixed(0).toLocaleString()}
• Throughput: ${((size * 4) / (endTime - startTime) / 1024).toFixed(2)} MB/ms
${timer.supported ? `<span class="hint">${GPU_TIME_NOTE}</span>` : ''}`;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'compute', code: shaderCode }]; }

// ---- Compute-only API for the Measure tab (js/teach/experiments/vector-addition.js) and test/gpu/ ----
//
// measureVectorAdd() runs the shipped shader on caller-supplied inputs, with the same explicit layout as
// runVectorAddition, and separates three costs that the one-shot readout lumps together: the kernel
// alone (GPU timestamps), the kernel as the CPU sees it (submit → onSubmittedWorkDone), and a full round
// trip (upload both inputs, dispatch, copy the sum back and map it). Buffers and the pipeline are created
// before anything is timed.

export const WORKGROUP_SIZE = 64;
// The shader indexes with global_id.x only, so one dispatch covers at most 65,535 workgroups of 64
export const MAX_DISPATCH_N = 65535 * WORKGROUP_SIZE;
export const MAX_MEASURE_REPEATS = 10;

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((p, q) => p - q);
    return v.length ? v[v.length >> 1] : null;
};

// a, b: Float32Arrays of equal length n ≤ MAX_DISPATCH_N. `warmup` untimed and `repeats` timed dispatches
// (≤ 10 each), then one round trip. Resolves to { n, workgroups, gpuMs, kernelWallMs, roundTripMs, sum }
// (medians over the repeats; gpuMs null without timestamp-query); sum is the read-back a + b (Float32Array)
// when readback is true.
export async function measureVectorAdd(gpuDevice, { a, b, warmup = 1, repeats = 5, readback = true }) {
    const device = gpuDevice.device ?? gpuDevice;
    const n = a?.length;
    if (!Number.isInteger(n) || n < 1 || n > MAX_DISPATCH_N || b?.length !== n) {
        throw new RangeError(`n = ${n} is outside 1 … ${MAX_DISPATCH_N} (or a and b differ in length)`);
    }
    const bytes = 4 * n;
    const input1Buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const input2Buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outputBuffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const timer = new GpuTimer(gpuDevice, { maxPasses: 1, poolSize: 1 });
    try {
        const layout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ code: shaderCode }), entryPoint: 'main' },
        });
        const bindGroup = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: input1Buffer } },
                { binding: 1, resource: { buffer: input2Buffer } },
                { binding: 2, resource: { buffer: outputBuffer } },
            ],
        });
        const workgroups = Math.ceil(n / WORKGROUP_SIZE);
        const encodeAdd = (encoder, timed) => {
            const pass = encoder.beginComputePass(timed ? timer.computePassDescriptor('add') : undefined);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
            if (timed) timer.resolve(encoder);
        };
        device.queue.writeBuffer(input1Buffer, 0, a);
        device.queue.writeBuffer(input2Buffer, 0, b);

        const timedRuns = Math.min(Math.max(repeats, 1), MAX_MEASURE_REPEATS);
        const total = Math.min(Math.max(warmup, 0), MAX_MEASURE_REPEATS) + timedRuns;
        const gpu = [], wall = [];
        for (let r = 0; r < total; r++) {
            const timed = r >= total - timedRuns;
            const encoder = device.createCommandEncoder();
            encodeAdd(encoder, timed);
            const t0 = performance.now();
            device.queue.submit([encoder.finish()]);
            const timing = timed ? timer.read() : null;
            await device.queue.onSubmittedWorkDone();
            if (!timed) continue;
            wall.push(performance.now() - t0);
            gpu.push((await timing)?.total ?? null);
        }

        // Round trip: what one GPU call costs end to end when the data starts and ends on the CPU
        const t0 = performance.now();
        device.queue.writeBuffer(input1Buffer, 0, a);
        device.queue.writeBuffer(input2Buffer, 0, b);
        const encoder = device.createCommandEncoder();
        encodeAdd(encoder, false);
        encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const sum = readback ? new Float32Array(staging.getMappedRange().slice(0)) : undefined;
        staging.unmap();
        const roundTripMs = performance.now() - t0;

        return { n, workgroups, gpuMs: median(gpu), kernelWallMs: median(wall), roundTripMs, sum };
    } finally {
        timer.destroy();
        for (const buffer of [input1Buffer, input2Buffer, outputBuffer, staging]) buffer.destroy();
    }
}
