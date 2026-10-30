export async function checkWebGPUSupport() {
    if (!navigator.gpu) {
        return false;
    }
    return true;
}

export async function getGPUDevice() {
    if (!navigator.gpu) {
        throw new Error('WebGPU is not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Failed to get GPU adapter');
    }

    const device = await adapter.requestDevice();

    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);
        if (info.reason !== 'destroyed') {
            console.error('Device lost unexpectedly, may need to recreate');
        }
    });

    return { device, adapter };
}

export function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage,
        mappedAtCreation: true
    });

    new data.constructor(buffer.getMappedRange()).set(data);

    buffer.unmap();
    return buffer;
}

export async function readBuffer(device, buffer, size, ArrayType = Float32Array) {
    const readBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new ArrayType(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();

    return result;
}
