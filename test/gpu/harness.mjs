// GPU execution harness for test/gpu/*.gpu.test.mjs — CI ONLY, software GPU ONLY (#51).
//
// These tests compile the exact WGSL the examples ship (their wgslModules() exports), dispatch it on
// a real WebGPU implementation and compare the read-back results with the CPU references in test/.
//
// Runtime: Node + the `webgpu` npm package (Dawn's dawn.node bindings), version pinned in the
// `gpu-software` job of .github/workflows/ci.yml (WEBGPU_NPM_VERSION) and installed there with
// --no-save, so the site keeps zero npm dependencies. Why Dawn rather than Deno's built-in wgpu:
//   * Dawn is the WebGPU implementation in Chrome, the browser most visitors run the demos in, so
//     its WGSL front end (Tint), validation and default limits are the ones the shipped code meets;
//   * dawn.node ships prebuilt linux-x64 binaries and selects a Vulkan adapter by name
//     (`adapter=llvmpipe`), which pins it to Mesa's lavapipe CPU rasterizer on the runner;
//   * the tests stay on node:test like the CPU suite and can import its references directly.
//
// SAFETY GUARD (mandatory). On a developer machine this would drive the real GPU, and a GPU hang
// has crashed a maintainer's machine before. The harness therefore refuses to run unless
//   1. CI=true (GitHub Actions sets it) and the platform is Linux — checked *before* the WebGPU
//      runtime is even imported, and
//   2. the adapter Dawn returns is a software/CPU device: adapter.info vendor/architecture/device/
//      description mentions llvmpipe / lavapipe / SwiftShader, or the adapter is a fallback adapter.
// Anything else aborts the process with exit code 2 before a device is created.
// Never run test/gpu/ locally from automation; see CLAUDE.md.

import { after } from 'node:test';
import { structLayout } from '../helpers/wgsl-layout.mjs';

const SOFTWARE = /llvmpipe|lavapipe|swiftshader/i;

function refuse(why) {
    const msg = `\n[test/gpu] REFUSING TO RUN: ${why}\n` +
        '[test/gpu] GPU kernel tests execute only in CI on a software Vulkan driver (Mesa lavapipe).\n' +
        '[test/gpu] They must never run on a developer machine. Use `npm test` for the CPU suite.\n';
    process.stderr.write(msg);
    process.exit(2);
}

if (process.env.CI !== 'true') refuse(`CI is not "true" (CI=${process.env.CI ?? '<unset>'})`);
if (process.platform !== 'linux') refuse(`platform is ${process.platform}, not linux`);

const { create, globals } = await import('webgpu');
Object.assign(globalThis, globals);   // GPUBufferUsage, GPUMapMode, GPUShaderStage, ...

let gpu = create(['backend=vulkan', 'adapter=llvmpipe']);
const adapter = await gpu.requestAdapter();
if (!adapter) refuse('no WebGPU adapter (is mesa-vulkan-drivers installed and VK_ICD_FILENAMES set?)');

const info = adapter.info ?? {};
const infoText = ['vendor', 'architecture', 'device', 'description']
    .map(k => `${k}=${JSON.stringify(info[k] ?? '')}`).join(' ');
const isFallback = Boolean(adapter.isFallbackAdapter || info.isFallbackAdapter);
if (!isFallback && !SOFTWARE.test(`${info.vendor} ${info.architecture} ${info.device} ${info.description}`)) {
    refuse(`adapter is not a software device (${infoText})`);
}

/** The adapter description, printed once per test file. */
export const adapterInfo = `${infoText} fallback=${isFallback}`;
console.log(`[test/gpu] adapter: ${adapterInfo}`);

// Default limits, as a page calling requestDevice() without requiredLimits gets.
export const device = await adapter.requestDevice();
const uncaptured = [];
device.addEventListener?.('uncapturederror', e => uncaptured.push(e.error.message));
device.lost.then(i => { if (i.reason !== 'destroyed') console.error(`[test/gpu] device lost: ${i.message}`); });

after(() => {
    device.destroy();
    gpu = null;   // dawn.node keeps the process alive while the instance is referenced
});

// ---- WGSL binding declarations -> explicit bind group layouts ----

const stripComments = code => code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');

/** Every `@group(g) @binding(b) var<...> name: type;` in `code`: [{ group, binding, name, kind, type }]. */
export function parseBindings(code) {
    const out = [];
    const re = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+);/g;
    for (const m of stripComments(code).matchAll(re)) {
        const [, g, b, space = '', name, type] = m;
        const [as, access = 'read'] = space.split(',').map(s => s.trim());
        let entry;
        if (as === 'uniform') entry = { buffer: { type: 'uniform' } };
        else if (as === 'storage') entry = { buffer: { type: access === 'read_write' ? 'storage' : 'read-only-storage' } };
        else if (/^texture_storage_/.test(type)) {
            const [, fmt, acc] = type.match(/<\s*(\w+)\s*,\s*(\w+)\s*>/);
            entry = { storageTexture: { format: fmt, access: acc.replace('_', '-').replace('readwrite', 'read-write'),
                viewDimension: type.includes('3d') ? '3d' : '2d' } };
        } else if (/^texture_/.test(type)) entry = { texture: { sampleType: 'unfilterable-float', viewDimension: type.includes('3d') ? '3d' : '2d' } };
        else if (/^sampler/.test(type)) entry = { sampler: { type: 'non-filtering' } };
        else throw new Error(`cannot map binding ${name}: var<${space}> ${type}`);
        out.push({ group: +g, binding: +b, name, type: type.trim(), entry });
    }
    return out;
}

// ---- Error capture ----

async function scoped(what, fn) {
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('internal');
    let value;
    try { value = await fn(); } finally {
        const errs = [await device.popErrorScope(), await device.popErrorScope(), await device.popErrorScope()].filter(Boolean);
        if (errs.length) throw new Error(`${what}: ${errs.map(e => e.message).join('\n')}`);
    }
    if (uncaptured.length) throw new Error(`${what}: uncaptured WebGPU error(s): ${uncaptured.splice(0).join('\n')}`);
    return value;
}

// ---- Pipelines ----

const moduleCache = new Map();

/**
 * Compile `code` (exact WGSL string) and build a compute pipeline for `entryPoint` with an explicit
 * layout covering every binding the module declares (like the examples' own explicit layouts).
 * Returns { pipeline, bindings, layouts }.
 */
export async function computePipeline(code, entryPoint, constants) {
    let mod = moduleCache.get(code);
    if (!mod) {
        mod = await scoped('createShaderModule', async () => {
            const m = device.createShaderModule({ code });
            const ci = await m.getCompilationInfo();
            const errs = ci.messages.filter(x => x.type === 'error');
            if (errs.length) throw new Error(errs.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
            return m;
        });
        moduleCache.set(code, mod);
    }
    const bindings = parseBindings(code);
    const groups = Math.max(-1, ...bindings.map(b => b.group)) + 1;
    const layouts = Array.from({ length: groups }, (_, g) => device.createBindGroupLayout({
        entries: bindings.filter(b => b.group === g)
            .map(b => ({ binding: b.binding, visibility: GPUShaderStage.COMPUTE, ...b.entry })),
    }));
    const pipeline = await scoped(`createComputePipeline(${entryPoint})`, () => device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        compute: { module: mod, entryPoint, ...(constants ? { constants } : {}) },
    }));
    return { pipeline, bindings, layouts, entryPoint };
}

/**
 * Bind group `group` of pipeline `p` from `resources`, keyed by WGSL variable name or binding
 * number. Values: a GPUBuffer, { buffer, offset, size }, or a texture view / sampler.
 */
export function bindGroup(p, group, resources) {
    const entries = p.bindings.filter(b => b.group === group).map(b => {
        const r = resources[b.name] ?? resources[b.binding];
        if (r === undefined) throw new Error(`${p.entryPoint}: no resource for @group(${group}) @binding(${b.binding}) ${b.name}`);
        const isBuf = typeof r.mapAsync === 'function';
        return { binding: b.binding, resource: isBuf ? { buffer: r } : r };
    });
    return device.createBindGroup({ layout: p.layouts[group], entries });
}

// ---- Buffers ----

const ALL = () => GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC |
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;

/** A buffer usable as storage/uniform/copy src/dst, from a TypedArray / ArrayBuffer or a byte size (zeroed). */
export function buffer(dataOrSize) {
    const bytes = typeof dataOrSize === 'number' ? null
        : ArrayBuffer.isView(dataOrSize) ? new Uint8Array(dataOrSize.buffer, dataOrSize.byteOffset, dataOrSize.byteLength)
        : new Uint8Array(dataOrSize);
    const size = Math.max(16, Math.ceil((bytes ? bytes.byteLength : dataOrSize) / 16) * 16);
    const buf = device.createBuffer({ size, usage: ALL(), mappedAtCreation: !!bytes });
    if (bytes) { new Uint8Array(buf.getMappedRange()).set(bytes); buf.unmap(); }
    return buf;
}

/** queue.writeBuffer with a TypedArray / ArrayBuffer at a byte offset. */
export function write(buf, data, offset = 0) {
    device.queue.writeBuffer(buf, offset, ArrayBuffer.isView(data) ? data : new Uint8Array(data));
}

/** Copy `byteLength` bytes (default: whole buffer) back and return them as `Type`. */
export async function read(buf, Type = Float32Array, byteLength = buf.size, offset = 0) {
    const staging = device.createBuffer({ size: Math.ceil(byteLength / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offset, staging, 0, staging.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0, byteLength);
    staging.unmap();
    staging.destroy();
    return new Type(out);
}

// ---- Dispatch ----

/**
 * Encode and submit a sequence of dispatches, each its own compute pass, then wait for the GPU.
 * steps: [{ p, groups: [bindGroup, ...], workgroups: [x, y?, z?] } | { copy: [src, srcOff, dst, dstOff, size] }
 *         | { write: [buffer, data, offset] }   (writeBuffer: split the submission so it lands in order)]
 */
export async function run(steps, label = 'dispatch') {
    await scoped(label, async () => {
        let enc = device.createCommandEncoder();
        let pending = false;
        const flush = () => { if (pending) device.queue.submit([enc.finish()]); enc = device.createCommandEncoder(); pending = false; };
        for (const s of steps) {
            if (s.write) { flush(); write(...s.write); continue; }
            pending = true;
            if (s.copy) { const [a, ao, b, bo, n] = s.copy; enc.copyBufferToBuffer(a, ao, b, bo, n); continue; }
            const pass = enc.beginComputePass();
            pass.setPipeline(s.p.pipeline);
            s.groups.forEach((g, i) => pass.setBindGroup(i, g));
            const [x, y = 1, z = 1] = s.workgroups;
            pass.dispatchWorkgroups(x, y, z);
            pass.end();
        }
        flush();
        await device.queue.onSubmittedWorkDone();
    });
}

/** One dispatch of pipeline `p` with bind group 0 built from `resources`. */
export async function dispatch(p, resources, workgroups) {
    await run([{ p, groups: [bindGroup(p, 0, resources)], workgroups }], p.entryPoint);
}

// ---- Data / comparison helpers ----

/** mulberry32 PRNG returning floats in [0, 1). */
export function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** { maxAbs, maxRel, at } between `got` and `ref` (rel = |d| / max(|ref|, floor)). */
export function compare(got, ref, floor = 1e-6) {
    if (got.length < ref.length) throw new Error(`length ${got.length} < ${ref.length}`);
    let maxAbs = 0, maxRel = 0, at = -1, nonFinite = 0;
    for (let i = 0; i < ref.length; i++) {
        if (!Number.isFinite(got[i])) { nonFinite++; at = i; maxAbs = Infinity; maxRel = Infinity; continue; }
        const d = Math.abs(got[i] - ref[i]);
        const r = d / Math.max(Math.abs(ref[i]), floor);
        if (d > maxAbs) { maxAbs = d; at = i; }
        if (r > maxRel) maxRel = r;
    }
    return { maxAbs, maxRel, at, nonFinite };
}

/**
 * Pack `values` (member name -> number | number[]) into WGSL struct `name` declared in `code`.
 * Members may be scalars, vectors, or arrays of scalars/vectors (given flat); missing ones stay 0.
 */
export function packStruct(code, name, values) {
    const layout = structLayout(code, name);
    const out = new ArrayBuffer(layout.size);
    const view = new DataView(out);
    for (const k of Object.keys(values)) if (!(k in layout.offsets)) throw new Error(`struct ${name} has no member ${k}`);
    for (const f of layout.fields) {
        if (!(f.name in values)) continue;
        const flat = [values[f.name]].flat(3);
        const elem = f.layout.kind === 'array' ? f.layout.elem : f.layout;
        const stride = f.layout.kind === 'array' ? f.layout.stride : 0;
        const n = elem.kind === 'vec' ? elem.n : 1;
        const kind = elem.kind === 'vec' ? elem.scalar : elem.kind;
        flat.forEach((x, i) => {
            const o = f.offset + Math.floor(i / n) * stride + (i % n) * 4;
            if (kind === 'f32') view.setFloat32(o, x, true);
            else if (kind === 'i32') view.setInt32(o, x, true);
            else if (kind === 'u32') view.setUint32(o, x >>> 0, true);
            else throw new Error(`packStruct: unsupported member ${f.name}: ${f.layout.type}`);
        });
    }
    return new Uint8Array(out);
}
