// Minimal recording mock of a GPUDevice for CPU-only tests. Nothing here touches a real GPU.
//
// mockDevice() returns a device whose createBuffer() hands back plain objects ({ label, size, usage })
// and whose queue.writeBuffer() records a *copy* of the bytes written. Every other method/property
// (createShaderModule, createComputePipeline, createCommandEncoder().beginComputePass()..., etc.)
// returns a permissive stub, so example classes can be constructed and their uniform writers called.
// Calling installGPUGlobals() defines GPUBufferUsage / GPUShaderStage / GPUMapMode / GPUTextureUsage.

export function installGPUGlobals() {
    globalThis.GPUBufferUsage ??= {
        MAP_READ: 0x1, MAP_WRITE: 0x2, COPY_SRC: 0x4, COPY_DST: 0x8, INDEX: 0x10, VERTEX: 0x20,
        UNIFORM: 0x40, STORAGE: 0x80, INDIRECT: 0x100, QUERY_RESOLVE: 0x200,
    };
    globalThis.GPUShaderStage ??= { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };
    globalThis.GPUMapMode ??= { READ: 0x1, WRITE: 0x2 };
    globalThis.GPUTextureUsage ??= {
        COPY_SRC: 0x1, COPY_DST: 0x2, TEXTURE_BINDING: 0x4, STORAGE_BINDING: 0x8, RENDER_ATTACHMENT: 0x10,
    };
}

// A stub that tolerates any property access or call chain. `then` is undefined so awaiting it
// does not hang; primitives coerce to 0 / ''.
export function stub(label = 'stub') {
    const target = function () {};
    return new Proxy(target, {
        get(_, prop) {
            if (prop === 'then') return undefined;
            if (prop === Symbol.toPrimitive) return hint => (hint === 'number' ? 0 : '');
            if (prop === 'label') return label;
            return stub(`${label}.${String(prop)}`);
        },
        apply() { return stub(`${label}()`); },
        set() { return true; },
    });
}

function bytesOf(data, dataOffset = 0, size) {
    if (ArrayBuffer.isView(data)) {
        const bpe = data.BYTES_PER_ELEMENT ?? 1;
        const start = data.byteOffset + dataOffset * bpe;
        const len = size === undefined ? data.byteLength - dataOffset * bpe : size * bpe;
        return new Uint8Array(data.buffer.slice(start, start + len));
    }
    const len = size === undefined ? data.byteLength - dataOffset : size;
    return new Uint8Array(data.slice(dataOffset, dataOffset + len));
}

export function mockDevice() {
    installGPUGlobals();
    const writes = [];   // { buffer, offset, bytes }
    const buffers = [];
    const queue = {
        writeBuffer(buffer, offset, data, dataOffset, size) {
            writes.push({ buffer, offset, bytes: bytesOf(data, dataOffset, size) });
        },
        writeTexture() {},
        copyExternalImageToTexture() {},
        submit() {},
        onSubmittedWorkDone: () => Promise.resolve(),
    };
    const device = {
        writes,
        buffers,
        queue,
        limits: {
            minUniformBufferOffsetAlignment: 256, minStorageBufferOffsetAlignment: 256,
            maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 1 << 30,
            maxBufferSize: 1 << 30, maxComputeInvocationsPerWorkgroup: 256,
            maxComputeWorkgroupStorageSize: 16384, maxStorageBuffersPerShaderStage: 10,
            maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 256, maxComputeWorkgroupSizeZ: 64,
        },
        features: new Set(),
        createBuffer(desc) {
            const size = desc.size;
            let mapped = desc.mappedAtCreation ? new ArrayBuffer(size) : null;
            const buf = {
                label: desc.label, size, usage: desc.usage,
                destroy() {}, unmap() {},
                getMappedRange: () => (mapped ??= new ArrayBuffer(size)),
                mapAsync: () => Promise.resolve(),
            };
            buffers.push(buf);
            return buf;
        },
        /** Bytes most recently written to `buffer` (whole write), or undefined. */
        lastWrite(buffer) {
            for (let i = writes.length - 1; i >= 0; i--) if (writes[i].buffer === buffer) return writes[i];
            return undefined;
        },
        destroy() {},
        lost: new Promise(() => {}),
        pushErrorScope() {}, popErrorScope: () => Promise.resolve(null),
    };
    return new Proxy(device, {
        get(t, prop) {
            if (prop in t) return t[prop];
            if (prop === 'then') return undefined;
            return stub(`device.${String(prop)}`);
        },
    });
}
