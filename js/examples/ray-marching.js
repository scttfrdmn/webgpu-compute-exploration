// Ray Marching for 3D Fractals
// Uses distance field ray marching to render volumetric fractals

import { GpuTimer } from '../gpu-timer.js';
import { readBuffer } from '../webgpu-utils.js';

const RAY_MARCH_SHADER = `
struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    power: f32,
    maxSteps: u32,
    fractalType: u32,
    cameraRotX: f32,
    cameraRotY: f32,
    cameraZoom: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Mandelbulb distance estimator
fn mandelbulbDE(pos: vec3<f32>, power: f32) -> f32 {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;

    for (var i = 0; i < 15; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        // Convert to polar coordinates
        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // Scale and rotate the point
        let zr = pow(r, power);
        let newTheta = theta * power;
        let newPhi = phi * power;

        // Convert back to cartesian coordinates
        z = zr * vec3<f32>(
            sin(newTheta) * cos(newPhi),
            sin(newPhi) * sin(newTheta),
            cos(newTheta)
        );
        z += pos;
    }

    return 0.5 * log(r) * r / dr;
}

// Julia set distance estimator
fn juliaDE(pos: vec3<f32>) -> f32 {
    let c = vec4<f32>(0.18, 0.88, 0.24, 0.16);
    var z = vec4<f32>(pos, 0.0);
    var md2 = 1.0;
    var mz2 = dot(z, z);

    for (var i = 0; i < 12; i++) {
        md2 *= 4.0 * mz2;

        // z = z^2 + c
        z = vec4<f32>(
            z.x * z.x - dot(z.yzw, z.yzw),
            2.0 * z.x * z.yzw
        ) + c;

        mz2 = dot(z, z);
        if (mz2 > 4.0) { break; }
    }

    return 0.25 * sqrt(mz2 / md2) * log(mz2);
}

// Menger sponge distance estimator (Inigo Quilez's formulation)
fn floorMod(x: vec3<f32>, m: f32) -> vec3<f32> {
    return x - m * floor(x / m);
}

fn mengerDE(pos: vec3<f32>) -> f32 {
    let q = abs(pos) - vec3<f32>(1.0);
    var d = length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    var s = 1.0;

    for (var i = 0; i < 4; i++) {
        let a = floorMod(pos * s, 2.0) - 1.0;
        s *= 3.0;
        let r = abs(1.0 - 3.0 * abs(a));

        let da = max(r.x, r.y);
        let db = max(r.y, r.z);
        let dc = max(r.z, r.x);
        let c = (min(da, min(db, dc)) - 1.0) / s;

        d = max(d, c);
    }

    return d;
}

// Mandelbox distance estimator
fn mandelboxDE(pos: vec3<f32>) -> f32 {
    var p = pos;
    var dr = 1.0;
    // Negative scale keeps the whole fractal within about 2 units of the origin
    let scale = -1.5;
    let fixedRadius = 1.0;
    let minRadius = 0.5;
    let fixedRadius2 = fixedRadius * fixedRadius;
    let minRadius2 = minRadius * minRadius;

    for (var i = 0; i < 12; i++) {
        // Box fold
        p = clamp(p, vec3<f32>(-1.0), vec3<f32>(1.0)) * 2.0 - p;

        // Sphere fold
        let r2 = dot(p, p);
        if (r2 < minRadius2) {
            p *= fixedRadius2 / minRadius2;
            dr *= fixedRadius2 / minRadius2;
        } else if (r2 < fixedRadius2) {
            p *= fixedRadius2 / r2;
            dr *= fixedRadius2 / r2;
        }

        // Scale and translate
        p = p * scale + pos;
        dr = dr * abs(scale) + 1.0;
    }

    return length(p) / abs(dr);
}

fn sceneSDF(pos: vec3<f32>) -> f32 {
    switch uniforms.fractalType {
        case 0u: { return mandelbulbDE(pos, uniforms.power); }
        case 1u: { return juliaDE(pos); }
        case 2u: { return mengerDE(pos); }
        case 3u: { return mandelboxDE(pos * 3.0) / 3.0; } // shrink to fit the default view
        default: { return mandelbulbDE(pos, uniforms.power); }
    }
}

fn estimateNormal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    return normalize(vec3<f32>(
        sceneSDF(p + e.xyy) - sceneSDF(p - e.xyy),
        sceneSDF(p + e.yxy) - sceneSDF(p - e.yxy),
        sceneSDF(p + e.yyx) - sceneSDF(p - e.yyx)
    ));
}

fn rayMarch(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t = 0.0;
    for (var i = 0u; i < uniforms.maxSteps; i++) {
        let pos = ro + rd * t;
        let d = sceneSDF(pos);
        if (d < 0.001) {
            return t;
        }
        if (t > 50.0) {
            break;
        }
        t += d * 0.5; // March with damping for safety
    }
    return -1.0;
}

// Soft ambient occlusion from a few distance samples along the normal
fn ambientOcclusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    var weight = 1.0;
    for (var i = 1; i <= 5; i++) {
        let d = 0.03 * f32(i);
        occlusion += weight * (d - sceneSDF(p + n * d));
        weight *= 0.5;
    }
    return clamp(1.0 - 4.0 * occlusion, 0.0, 1.0);
}

fn rotateX(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

fn rotateY(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = fragCoord.xy;

    // Pixel y grows downward; flip so +y is up in camera space
    var uv = (coords - uniforms.resolution * 0.5) / uniforms.resolution.y;
    uv.y = -uv.y;

    // Camera setup
    var ro = vec3<f32>(0.0, 0.0, uniforms.cameraZoom);
    ro = rotateX(ro, uniforms.cameraRotX);
    ro = rotateY(ro, uniforms.cameraRotY + uniforms.time * 0.1);

    let lookAt = vec3<f32>(0.0, 0.0, 0.0);
    let forward = normalize(lookAt - ro);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), forward));
    let up = cross(forward, right);

    let rd = normalize(forward + uv.x * right + uv.y * up);

    // Ray march
    let t = rayMarch(ro, rd);

    // Background gradient so the silhouette stays visible against the page
    var color = mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.12, 0.14, 0.22), uv.y + 0.5);
    if (t > 0.0) {
        let pos = ro + rd * t;
        let normal = estimateNormal(pos);

        // Lighting
        let lightDir = normalize(vec3<f32>(1.0, 1.0, 2.0));
        let diffuse = max(dot(normal, lightDir), 0.0);
        let specular = pow(max(dot(reflect(-lightDir, normal), -rd), 0.0), 32.0);

        let ao = ambientOcclusion(pos, normal);

        // Color based on position and normal
        let baseColor = vec3<f32>(0.8, 0.4, 0.2) + normal * 0.2;
        color = baseColor * (diffuse * 0.7 + 0.3) * ao + specular * 0.5;

        // Fog
        color = mix(color, vec3<f32>(0.02, 0.02, 0.05), clamp(t / 20.0, 0.0, 1.0));
    }

    return vec4<f32>(color, 1.0);
}
`;

// Full-screen quad; the fragment shader ray-marches each pixel
const FULLSCREEN_VERTEX = `
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
    let pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}
`;

const RENDER_SHADER = FULLSCREEN_VERTEX + '\n' + RAY_MARCH_SHADER;

class RayMarchingRenderer {
    // canvas = null: compute-only (Measure tab, test/gpu/): the pipeline renders into an offscreen
    // bgra8unorm texture of size `target` instead, and there are no mouse handlers
    constructor(device, canvas, target = { width: 1000, height: 700 }) {
        this.device = device;
        this.canvas = canvas;
        this.target = canvas ? null : target;
        this.running = false;
        this.time = 0;

        this.fractalType = 0; // 0: Mandelbulb, 1: Julia, 2: Menger, 3: Mandelbox
        this.power = 8;
        this.maxSteps = 128;
        this.cameraRotX = 0;
        this.cameraRotY = 0;
        this.cameraZoom = 3;
        this.autoRotate = true;

        this.setupRenderPipeline();
        if (canvas) this.setupMouseControls();
    }

    setupMouseControls() {
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;

        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            this.autoRotate = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;

            this.cameraRotY += dx * 0.01;
            this.cameraRotX += dy * 0.01;
            this.cameraRotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.cameraRotX));

            lastX = e.clientX;
            lastY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.cameraZoom *= (1 + e.deltaY * 0.001);
            this.cameraZoom = Math.max(1.5, Math.min(6, this.cameraZoom));
        });
    }

    setupRenderPipeline() {
        if (this.canvas) {
            this.context = this.canvas.getContext('webgpu');
            this.context.configure({
                device: this.device,
                format: 'bgra8unorm'
            });
        } else {
            this.targetTexture = this.device.createTexture({
                size: [this.target.width, this.target.height],
                format: 'bgra8unorm',
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });
        }

        const shaderModule = this.device.createShaderModule({ code: RENDER_SHADER });

        this.uniformBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' }
            }]
        });

        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            }
        });
    }

    updateUniforms() {
        // maxSteps and fractalType are u32 in WGSL; written as floats they would read as ~1e9
        const uniforms = new Float32Array(16);
        const uniformsU32 = new Uint32Array(uniforms.buffer);
        const size = this.canvas ?? this.target;
        uniforms[0] = size.width;
        uniforms[1] = size.height;
        uniforms[2] = this.time;
        uniforms[3] = this.power;
        uniformsU32[4] = this.maxSteps;
        uniformsU32[5] = this.fractalType;
        uniforms[6] = this.cameraRotX;
        uniforms[7] = this.cameraRotY;
        uniforms[8] = this.cameraZoom;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    }

    render() {
        if (this.autoRotate) {
            this.time += 0.016;
        }

        this.updateUniforms();

        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(6);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.animate();
    }

    pause() {
        this.running = false;
    }

    animate() {
        if (!this.running) return;

        this.render();
        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.running = false;
        this.uniformBuffer.destroy();
        this.targetTexture?.destroy();
    }
}

export async function initRayMarching(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-fractal');
    const output = document.getElementById('output-fractal');

    // Not started here: main.js runs it only while the canvas is on screen
    const renderer = new RayMarchingRenderer(device, canvas);

    const fractalTypes = {
        'mandelbulb': 0,
        'julia': 1,
        'menger': 2,
        'mandelbox': 3
    };

    document.getElementById('fractal-type').addEventListener('change', (e) => {
        renderer.fractalType = fractalTypes[e.target.value];
        updateOutput();
    });

    const powerSlider = document.getElementById('fractal-power');
    const powerValue = document.getElementById('fractal-power-value');
    powerSlider.addEventListener('input', (e) => {
        renderer.power = parseInt(e.target.value);
        powerValue.textContent = e.target.value;
    });

    const qualitySlider = document.getElementById('fractal-quality');
    const qualityValue = document.getElementById('fractal-quality-value');
    qualitySlider.addEventListener('input', (e) => {
        renderer.maxSteps = parseInt(e.target.value);
        qualityValue.textContent = e.target.value;
        updateOutput();
    });

    function updateOutput() {
        const fractalNames = ['Mandelbulb', 'Julia Set', 'Menger Sponge', 'Mandelbox'];

        output.innerHTML = `<span class="success">✓ Ray Marching Active</span>

<span class="info">Configuration:</span>
• Fractal: ${fractalNames[renderer.fractalType]}
• Power: ${renderer.power}
• Max steps: ${renderer.maxSteps}
• Resolution: ${canvas.width} × ${canvas.height}

<span class="info">Technique:</span>
• Algorithm: Sphere tracing / Ray marching
• Distance field: Analytic distance estimators
• Lighting: Phong model with ambient occlusion
• Normals: Finite difference approximation

<span class="info">Performance:</span>
• Each pixel traces a ray through the fractal
• Cost per frame: ${canvas.width * canvas.height * renderer.maxSteps} evaluations
• Lower quality = faster rendering
• Fragment shader parallelism across all pixels

<span class="info">Fractals:</span>
• <strong>Mandelbulb</strong>: 3D extension of Mandelbrot set
• <strong>Julia Set</strong>: Quaternion Julia fractal
• <strong>Menger Sponge</strong>: Recursive cube subdivision
• <strong>Mandelbox</strong>: Box and sphere folding fractal`;
    }

    updateOutput();
    return renderer;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'render', code: RENDER_SHADER },
        { name: 'march-stats', code: MARCH_STATS_SHADER },
    ];
}

// Exported for the CPU tests in test/ray-marching.test.mjs.
export { RayMarchingRenderer };

// ---- Compute-only API for the Measure tab (js/teach/experiments/ray-marching.js) and test/gpu/ ----
//
// A fragment shader reports nothing but a color, so the Measure tab uses two pieces:
//   * the shipped render pipeline, drawn into an offscreen texture (RayMarchingRenderer without a canvas)
//     and timed, for the cost of a frame;
//   * MARCH_STATS_SHADER: the shipped shader source plus one compute entry point that builds each pixel's
//     ray exactly as fs_main does and runs the loop of rayMarch, recording how it ended.

export const MAX_MARCH_STEPS = 256;           // the Quality slider's maximum
export const MAX_MARCH_PIXELS = 1000 * 700;   // the canvas
export const MAX_MEASURE_REPEATS = 10;
export const FRACTALS = ['mandelbulb', 'julia', 'menger', 'mandelbox'];   // fractalType 0 … 3
export const OUTCOME = { HIT: 0, ESCAPED: 1, EXHAUSTED: 2 };

const MARCH_STATS_SHADER = RAY_MARCH_SHADER + `
// Measure tab only. One invocation per pixel: the camera ray of fs_main for the pixel center (x + 0.5,
// y + 0.5), then the loop of rayMarch with its exits recorded:
//   x = t at the hit (or -1), y = distance evaluations, z = outcome (0 hit: d < 0.001; 1 escaped: t > 50;
//   2 exhausted: maxSteps evaluations without either), w = the last distance estimate d.
@group(0) @binding(1) var<storage, read_write> marchOut: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn cs_march_stats(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(uniforms.resolution);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let coords = vec2<f32>(id.xy) + vec2<f32>(0.5);
    var uv = (coords - uniforms.resolution * 0.5) / uniforms.resolution.y;
    uv.y = -uv.y;
    var ro = vec3<f32>(0.0, 0.0, uniforms.cameraZoom);
    ro = rotateX(ro, uniforms.cameraRotX);
    ro = rotateY(ro, uniforms.cameraRotY + uniforms.time * 0.1);
    let forward = normalize(vec3<f32>(0.0) - ro);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), forward));
    let up = cross(forward, right);
    let rd = normalize(forward + uv.x * right + uv.y * up);

    var t = 0.0;
    var d = 0.0;
    var steps = 0u;
    var outcome = 2.0;
    for (var i = 0u; i < uniforms.maxSteps; i++) {
        steps = i + 1u;
        d = sceneSDF(ro + rd * t);
        if (d < 0.001) {
            outcome = 0.0;
            break;
        }
        if (t > 50.0) {
            outcome = 1.0;
            break;
        }
        t += d * 0.5;
    }
    marchOut[id.y * size.x + id.x] = vec4<f32>(select(-1.0, t, outcome == 0.0), f32(steps), outcome, d);
}
`;

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((p, q) => p - q);
    return v.length ? v[v.length >> 1] : null;
};

// One view of fractal `fractal` (a FRACTALS name) at width × height with at most maxSteps march steps, from
// the fixed camera of a fresh renderer (time 0, no rotation, zoom 3). Draws `warmup` untimed and `repeats`
// timed frames, then (stats = true) one statistics dispatch. Resolves to { width, height, fractal, maxSteps,
// gpuMs, wallMs, hit, escaped, exhausted, meanSteps, pixels? } with frame times as medians (gpuMs null
// without timestamp-query), outcome fractions of all pixels, the mean number of distance evaluations per
// pixel, and pixels (width × height × 4 floats, see MARCH_STATS_SHADER) when readback is true.
export async function measureRayMarch(gpuDevice, { width = 1000, height = 700, fractal = 'mandelbulb', power = 8,
    maxSteps = 128, warmup = 1, repeats = 3, stats = true, readback = false }) {
    const device = gpuDevice.device ?? gpuDevice;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > MAX_MARCH_PIXELS) {
        throw new RangeError(`${width} × ${height} is outside 1 … ${MAX_MARCH_PIXELS} pixels`);
    }
    if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > MAX_MARCH_STEPS) {
        throw new RangeError(`maxSteps = ${maxSteps} is outside 1 … ${MAX_MARCH_STEPS}`);
    }
    const fractalType = FRACTALS.indexOf(fractal);
    if (fractalType < 0) throw new RangeError(`unknown fractal ${fractal}`);
    const renderer = new RayMarchingRenderer(device, null, { width, height });
    Object.assign(renderer, { fractalType, power, maxSteps, autoRotate: false, time: 0 });
    const timer = new GpuTimer(gpuDevice, { maxPasses: 1, poolSize: 1 });
    const bytes = 16 * width * height;
    const out = stats ? device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }) : null;
    try {
        renderer.updateUniforms();
        const view = renderer.targetTexture.createView();
        const timedRuns = Math.min(Math.max(repeats, 1), MAX_MEASURE_REPEATS);
        const total = Math.min(Math.max(warmup, 0), MAX_MEASURE_REPEATS) + timedRuns;
        const gpu = [], wall = [];
        for (let r = 0; r < total; r++) {
            const timed = r >= total - timedRuns;
            const encoder = device.createCommandEncoder();
            const desc = { colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] };
            const pass = encoder.beginRenderPass(timed ? timer.renderPassDescriptor(desc, 'frame') : desc);
            pass.setPipeline(renderer.pipeline);
            pass.setBindGroup(0, renderer.bindGroup);
            pass.draw(6);
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
        const result = { width, height, fractal, maxSteps, gpuMs: median(gpu), wallMs: median(wall) };
        if (!stats) return result;

        const layout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ code: MARCH_STATS_SHADER }), entryPoint: 'cs_march_stats' },
        });
        const bindGroup = device.createBindGroup({
            layout,
            entries: [{ binding: 0, resource: { buffer: renderer.uniformBuffer } }, { binding: 1, resource: { buffer: out } }],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const pixels = await readBuffer(device, out, bytes);
        const counts = [0, 0, 0];
        let steps = 0;
        for (let p = 0; p < width * height; p++) {
            counts[pixels[4 * p + 2]] = (counts[pixels[4 * p + 2]] ?? 0) + 1;
            steps += pixels[4 * p + 1];
        }
        const all = width * height;
        Object.assign(result, {
            hit: counts[OUTCOME.HIT] / all, escaped: counts[OUTCOME.ESCAPED] / all, exhausted: counts[OUTCOME.EXHAUSTED] / all,
            meanSteps: steps / all,
        });
        if (readback) result.pixels = pixels;
        return result;
    } finally {
        timer.destroy();
        out?.destroy();
        renderer.destroy();
    }
}
