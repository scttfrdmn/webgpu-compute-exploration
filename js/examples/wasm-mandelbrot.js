import { GpuTimer, GPU_TIME_NOTE } from '../gpu-timer.js';

const shaderCode = `
struct Params {
    width: f32,
    height: f32,
    max_iterations: f32,
    zoom: f32,
    center_x: f32,
    center_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let width = u32(params.width);
    let height = u32(params.height);

    if (x >= width || y >= height) {
        return;
    }

    let scale = 4.0 / (params.zoom * f32(min(width, height)));

    let cx = params.center_x + (f32(x) - f32(width) / 2.0) * scale;
    let cy = params.center_y + (f32(y) - f32(height) / 2.0) * scale;

    var zx = 0.0;
    var zy = 0.0;
    var iteration = 0.0;

    let max_iter = u32(params.max_iterations);

    for (var i = 0u; i < max_iter; i = i + 1u) {
        if (zx * zx + zy * zy >= 4.0) {
            break;
        }

        let temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        iteration = iteration + 1.0;
    }

    output[y * width + x] = iteration;
}
`;

let wasmModule = null;

async function loadWASM() {
    if (wasmModule) {
        return wasmModule;
    }

    try {
        const { default: init, MandelbrotConfig, render_to_canvas } = await import('../wasm-pkg/webgpu_wasm.js');
        await init();
        wasmModule = { MandelbrotConfig, render_to_canvas };
        return wasmModule;
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw new Error(`WASM module not found. Please build it first by running: cd wasm && chmod +x build.sh && ./build.sh`);
    }
}

export async function runWASMMandelbrot(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-mandelbrot');

    try {
        const wasm = await loadWASM();
        const { MandelbrotConfig, render_to_canvas } = wasm;

        const width = 800;
        const height = 600;
        canvas.width = width;
        canvas.height = height;

        const config = new MandelbrotConfig(width, height);
        config.set_zoom(1.0);
        config.set_center(-0.5, 0.0);

        const startTime = performance.now();

        const params = new Float32Array(config.get_shader_params());
        const paramsBuffer = device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(paramsBuffer.getMappedRange()).set(params);
        paramsBuffer.unmap();

        const outputSize = width * height * 4;
        const outputBuffer = device.createBuffer({
            size: outputSize,
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
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                }
            ]
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } }
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
        const passEncoder = commandEncoder.beginComputePass(timer.computePassDescriptor('mandelbrot'));
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        passEncoder.end();
        timer.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        const timing = timer.read();

        const readBuffer = device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);
        device.queue.submit([copyEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();

        const gpuTime = performance.now();
        await timing;
        timer.destroy();

        render_to_canvas(canvas, result, config);

        const endTime = performance.now();

        paramsBuffer.destroy();
        outputBuffer.destroy();

        const totalPixels = width * height;

        return `<span class="success">✓ WASM + WebGPU Mandelbrot Complete</span>

<span class="info">Configuration:</span>
• Canvas size: ${width} × ${height} (${totalPixels.toLocaleString()} pixels)
• Max iterations: ${config.get_max_iterations()}
• Center: (${config.get_shader_params()[4].toFixed(2)}, ${config.get_shader_params()[5].toFixed(2)})
• Zoom: ${config.get_shader_params()[3].toFixed(2)}x

<span class="info">Performance:</span>
• GPU computation: ${(gpuTime - startTime).toFixed(2)}ms (wall: setup, kernel and readback)
• ${timer.summary(gpuTime - startTime)}
• WASM rendering: ${(endTime - gpuTime).toFixed(2)}ms
• Total time: ${(endTime - startTime).toFixed(2)}ms
• Pixels per ms: ${(totalPixels / (endTime - startTime)).toFixed(0).toLocaleString()}

<span class="info">Architecture:</span>
1. Rust/WASM manages configuration and color mapping
2. WebGPU compute shader performs iteration calculations
3. WASM processes results and renders to canvas
4. Full pipeline: JS → WASM → WebGPU → WASM → Canvas

<span class="info">Benefits of WASM integration:</span>
• Complex logic in Rust (memory safe)
• Heavy computation on GPU
• Optimal performance for each task
• Reusable across platforms
${timer.supported ? `<span class="hint">${GPU_TIME_NOTE}</span>` : ''}`;

    } catch (error) {
        if (error.message.includes('WASM module not found')) {
            return `<span class="error">⚠ WASM Module Not Built</span>

To run this example, you need to build the WASM module first:

<span class="info">Prerequisites:</span>
• Install Rust: https://rustup.rs/
• Install wasm-pack: <code>cargo install wasm-pack</code>

<span class="info">Build steps:</span>
<code>cd wasm
chmod +x build.sh
./build.sh</code>

Then refresh the page and try again.

<span class="info">What this example demonstrates:</span>
This example shows how to integrate Rust/WASM with WebGPU for
high-performance computing. The WASM module handles configuration
and rendering while WebGPU performs the compute-intensive fractal
calculations on the GPU.`;
        }
        throw error;
    }
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'compute', code: shaderCode }]; }
