const shaderCode = `
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> dimensions: vec2<u32>;

// Kernel covers ±3σ so the Gaussian tails are not truncated visibly
const sigma = 4.0;
const kernelSize = 25;

fn gaussian(x: f32, y: f32) -> f32 {
    return exp(-(x*x + y*y) / (2.0 * sigma * sigma)) / (2.0 * 3.14159265 * sigma * sigma);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = dimensions.x;
    let height = dimensions.y;

    if (x >= width || y >= height) {
        return;
    }

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weightSum = 0.0;

    let halfKernel = kernelSize / 2;

    for (var ky = -halfKernel; ky <= halfKernel; ky = ky + 1) {
        for (var kx = -halfKernel; kx <= halfKernel; kx = kx + 1) {
            let sx = i32(x) + kx;
            let sy = i32(y) + ky;

            if (sx >= 0 && sx < i32(width) && sy >= 0 && sy < i32(height)) {
                let idx = u32(sy) * width + u32(sx);
                let weight = gaussian(f32(kx), f32(ky));
                sum = sum + input[idx] * weight;
                weightSum = weightSum + weight;
            }
        }
    }

    output[y * width + x] = sum / weightSum;
}
`;

// Sharp-edged test pattern (checkerboard, thin rings, fine diagonal lines) so the blur is obvious
function createTestImage(width, height) {
    const imageData = new Float32Array(width * height * 4);
    const centerX = width / 2;
    const centerY = height / 2;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);

            const checker = (Math.floor(x / 32) + Math.floor(y / 32)) % 2;
            let r = checker ? 0.95 : 0.1;
            let g = checker ? 0.75 : 0.15;
            let b = checker ? 0.2 : 0.35;

            // Thin bright rings around the center
            if (dist < 180 && Math.floor(dist) % 24 < 3) {
                r = 0.2; g = 0.9; b = 1.0;
            }

            // 1-pixel diagonal lines in the top-left quadrant
            if (x < centerX && y < centerY && (x + y) % 8 === 0) {
                r = g = b = 1.0;
            }

            imageData[idx] = r;
            imageData[idx + 1] = g;
            imageData[idx + 2] = b;
            imageData[idx + 3] = 1.0;
        }
    }

    return imageData;
}

function renderImageToCanvas(canvas, imageData, width, height) {
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const canvasImageData = ctx.createImageData(width, height);

    for (let i = 0; i < imageData.length; i++) {
        canvasImageData.data[i] = imageData[i] * 255;
    }

    ctx.putImageData(canvasImageData, 0, 0);
}

export async function runImageBlur(gpuDevice) {
    const { device } = gpuDevice;
    const startTime = performance.now();

    const width = 512;
    const height = 512;

    const inputImage = createTestImage(width, height);

    const inputBuffer = device.createBuffer({
        size: inputImage.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(inputBuffer.getMappedRange()).set(inputImage);
    inputBuffer.unmap();

    const outputBuffer = device.createBuffer({
        size: inputImage.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const dimensionsArray = new Uint32Array([width, height]);
    const dimensionsBuffer = device.createBuffer({
        size: dimensionsArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Uint32Array(dimensionsBuffer.getMappedRange()).set(dimensionsArray);
    dimensionsBuffer.unmap();

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
                buffer: { type: 'storage' }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
            }
        ]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: dimensionsBuffer } }
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
    passEncoder.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    const readBuffer = device.createBuffer({
        size: inputImage.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, inputImage.byteLength);
    device.queue.submit([copyEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const outputImage = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    const endTime = performance.now();

    const originalCanvas = document.getElementById('canvas-original');
    const blurredCanvas = document.getElementById('canvas-blurred');
    renderImageToCanvas(originalCanvas, inputImage, width, height);
    renderImageToCanvas(blurredCanvas, outputImage, width, height);

    inputBuffer.destroy();
    outputBuffer.destroy();
    dimensionsBuffer.destroy();

    const totalPixels = width * height;
    const kernelSize = 25;
    const opsPerPixel = kernelSize * kernelSize * 4;
    const totalOps = totalPixels * opsPerPixel;

    return `<span class="success">✓ Image Blur Complete</span>

<span class="info">Configuration:</span>
• Image dimensions: ${width} × ${height} (${totalPixels.toLocaleString()} pixels)
• Kernel size: ${kernelSize} × ${kernelSize}
• Blur type: Gaussian (σ = 4.0)
• Workgroup size: 8 × 8

<span class="info">Results:</span>
• Computation time: ${(endTime - startTime).toFixed(2)}ms
• Operations per pixel: ${opsPerPixel}
• Total operations: ${totalOps.toLocaleString()}
• Throughput: ${(totalPixels / (endTime - startTime)).toFixed(0).toLocaleString()} pixels/ms

<span class="info">Applications:</span>
• Real-time image processing
• Video filters
• Anti-aliasing
• Depth of field effects
• Noise reduction`;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'compute', code: shaderCode }]; }
