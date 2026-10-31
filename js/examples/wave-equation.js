// 2D wave equation u_tt = c²∇²u on a grid, solved with an explicit leapfrog finite-difference scheme.
//
// Uniform layout (struct Params, 72 bytes; shared by the compute and render shaders):
//   offset  0  size       vec2<u32>  grid width, height                -> Uint32 view [0], [1]
//   offset  8  cursor     vec2<f32>  mouse position in cells             -> Float32 view [2], [3]
//   offset 16  cursorPrev vec2<f32>  mouse position at the last frame    -> Float32 view [4], [5]
//   offset 24  courant2   f32        C² = (c·dt/dx)²                     -> Float32 view [6]
//   offset 28  damping    f32        g = γ·dt/2, velocity damping        -> Float32 view [7]
//   offset 32  step       u32        global step index of this substep   -> Uint32 view [8]
//   offset 36  period     u32        source period in steps              -> Uint32 view [9]
//   offset 40  dropOn     u32        1 = edit pass adds a ripple         -> Uint32 view [10]
//   offset 44  wallOn     u32        1 = edit pass draws walls           -> Uint32 view [11]
//   offset 48  rainOn     u32        1 = random rain drops               -> Uint32 view [12]
//   offset 52  planeWave  u32        1 = drive the source column         -> Uint32 view [13]
//   offset 56  absorb     u32        1 = absorbing sponge at the edges   -> Uint32 view [14]
//   offset 60  seed       u32        rain hash seed                      -> Uint32 view [15]
//   offset 64  dropAmp    f32        ripple amplitude                    -> Float32 view [16]
//   offset 68  (padding to the struct's 8-byte alignment)
// One copy per substep sits at a dynamic offset of PARAMS_STRIDE bytes, so each substep sees its
// own `step` (the source phase and rain hash depend on it) without a CPU round trip.
const PARAMS_SIZE = 72;
const MAX_SUBSTEPS = 16;

const paramsStruct = `
struct Params {
    size: vec2<u32>,
    cursor: vec2<f32>,
    cursorPrev: vec2<f32>,
    courant2: f32,
    damping: f32,
    step: u32,
    period: u32,
    dropOn: u32,
    wallOn: u32,
    rainOn: u32,
    planeWave: u32,
    absorb: u32,
    seed: u32,
    dropAmp: f32,
}
`;

const computeShader = paramsStruct + `
// Two height buffers hold consecutive time levels. A step reads u(t) from heightCur and overwrites
// heightPrev, which holds u(t - dt), with u(t + dt); the buffers then swap roles. Each invocation
// only reads and writes its own cell of heightPrev, so updating it in place is race-free.
@group(0) @binding(0) var<storage, read_write> heightCur: array<f32>;
@group(0) @binding(1) var<storage, read_write> heightPrev: array<f32>;
@group(0) @binding(2) var<storage, read_write> walls: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const PI: f32 = 3.14159265;
const DROP_SIGMA: f32 = 4.0;      // ripple width in cells
const RAIN_SIGMA: f32 = 2.0;
const RAIN_AMP: f32 = 0.35;
const RAIN_SLOTS: u32 = 2u;       // possible rain drops per step
const RAIN_CHANCE: f32 = 0.03;    // probability for each slot
const WALL_RADIUS: f32 = 2.5;     // wall brush radius in cells
const SOURCE_X: i32 = 30;         // column driven by the plane-wave source
const SOURCE_AMP: f32 = 0.6;
const SPONGE: f32 = 24.0;         // absorbing layer width in cells
const SPONGE_STRENGTH: f32 = 0.08;  // extra g at the very edge

// PCG hash (one round): a well-mixed u32 from a u32
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn unitFloat(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

// Ripple profile: a Gaussian-based "Mexican hat", (1 - s)·exp(-s) with s = r²/2σ².
// Its integral over the plane is zero, so drops add no net volume and the mean level stays 0.
fn ripple(p: vec2<f32>, center: vec2<f32>, sigma: f32) -> f32 {
    let d = p - center;
    let s = dot(d, d) / (2.0 * sigma * sigma);
    if (s > 9.0) {
        return 0.0;
    }
    return (1.0 - s) * exp(-s);
}

// Distance from p to the segment a-b (the wall stroke since the last frame)
fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let len2 = dot(ab, ab);
    var t = 0.0;
    if (len2 > 0.0) {
        t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    }
    return length(p - (a + ab * t));
}

// Outside the grid u = 0: fixed (reflecting) outer boundary
fn heightAt(x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.size.x) || y >= i32(params.size.y)) {
        return 0.0;
    }
    return heightCur[u32(y) * params.size.x + u32(x)];
}

// Edit pass (no time step): draws walls along the mouse stroke and adds a ripple to both time
// levels, so the drop starts at rest. Wall cells are forced to u = 0.
@compute @workgroup_size(8, 8)
fn cs_edit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let idx = gid.y * params.size.x + gid.x;
    let p = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

    var wall = walls[idx];
    if (params.wallOn != 0u && segmentDistance(p, params.cursorPrev, params.cursor) < WALL_RADIUS) {
        wall = 1u;
        walls[idx] = wall;
    }

    var cur = heightCur[idx];
    var prev = heightPrev[idx];
    if (params.dropOn != 0u) {
        let r = params.dropAmp * ripple(p, params.cursor, DROP_SIGMA);
        cur += r;
        prev += r;
    }
    if (wall != 0u) {
        cur = 0.0;
        prev = 0.0;
    }
    heightCur[idx] = cur;
    heightPrev[idx] = prev;
}

// Leapfrog step for the damped wave equation u_tt + γ·u_t = c²∇²u with dx = dt = 1, centered
// differences in time and the 5-point Laplacian in space (g = γ/2):
//   u(t+1) = (2u(t) - (1 - g)·u(t-1) + C²·(u(x-1) + u(x+1) + u(y-1) + u(y+1) - 4u)) / (1 + g)
// g = 0 is the classic 2u(t) - u(t-1) + C²∇²u. Stability (CFL) for the 5-point Laplacian in 2D
// requires C ≤ 1/√2 ≈ 0.707, independent of g ≥ 0 (damping only shrinks the amplification
// factor). The default is C = 0.5 and the speed slider stops at 0.70. A wavefront moves C cells
// per step. (Damping the velocity as u + d·(u - u_prev) instead would tighten the CFL limit.)
@compute @workgroup_size(8, 8)
fn cs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let idx = gid.y * params.size.x + gid.x;
    if (walls[idx] != 0u) {
        heightPrev[idx] = 0.0;   // walls are fixed at u = 0 and reflect waves
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = heightCur[idx];
    let lap = heightAt(x - 1, y) + heightAt(x + 1, y) + heightAt(x, y - 1) + heightAt(x, y + 1) - 4.0 * c;

    // Absorbing sponge: extra damping that ramps up quadratically toward the edges
    var g = params.damping;
    if (params.absorb != 0u) {
        let edge = f32(min(min(x, y), min(i32(params.size.x) - 1 - x, i32(params.size.y) - 1 - y)));
        if (edge < SPONGE) {
            let s = (SPONGE - edge) / SPONGE;
            g += SPONGE_STRENGTH * s * s;
        }
    }

    var next = (2.0 * c - (1.0 - g) * heightPrev[idx] + params.courant2 * lap) / (1.0 + g);

    // Rain: each step hashes (seed, step, slot) into a chance, a position and a sign
    if (params.rainOn != 0u) {
        let p = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
        for (var k = 0u; k < RAIN_SLOTS; k++) {
            let h = pcg(params.seed ^ pcg(params.step * RAIN_SLOTS + k));
            if (unitFloat(h) < RAIN_CHANCE) {
                let h2 = pcg(h);
                let h3 = pcg(h2);
                let center = vec2<f32>(unitFloat(h2), unitFloat(h3)) * vec2<f32>(params.size);
                next += RAIN_AMP * ripple(p, center, RAIN_SIGMA);
            }
        }
    }

    // Plane-wave source: a column driven sinusoidally (a "hard" source)
    if (params.planeWave != 0u && x == SOURCE_X) {
        let phase = f32(params.step % params.period) / f32(params.period);
        next = SOURCE_AMP * sin(2.0 * PI * phase);
    }

    heightPrev[idx] = next;
}
`;

const renderShader = paramsStruct + `
@group(0) @binding(0) var<storage, read> heights: array<f32>;
@group(0) @binding(1) var<storage, read> walls: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle that covers the whole screen; uv (0,0) is the top-left of the grid
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let p = corners[vi];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

const SLOPE: f32 = 16.0;         // exaggerates the surface slope for lighting
const HEIGHT_GAIN: f32 = 2.0;    // height -> color ramp position, soft-clipped with tanh

fn cellHeight(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.size.x) - 1);
    let cy = clamp(y, 0, i32(params.size.y) - 1);
    return heights[u32(cy) * params.size.x + u32(cx)];
}

// Bilinear interpolation between cell centers, p in cell units
fn heightBilinear(p: vec2<f32>) -> f32 {
    let q = p - vec2<f32>(0.5);
    let base = floor(q);
    let f = q - base;
    let x = i32(base.x);
    let y = i32(base.y);
    let top = mix(cellHeight(x, y), cellHeight(x + 1, y), f.x);
    let bottom = mix(cellHeight(x, y + 1), cellHeight(x + 1, y + 1), f.x);
    return mix(top, bottom, f.y);
}

// Deep navy -> blue -> teal -> pale aqua, t in [0, 1] (0.5 = rest level)
fn waterRamp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.010, 0.040, 0.120);
    let c1 = vec3<f32>(0.030, 0.200, 0.420);
    let c2 = vec3<f32>(0.060, 0.480, 0.600);
    let c3 = vec3<f32>(0.600, 0.920, 0.900);
    if (t < 0.4) { return mix(c0, c1, t / 0.4); }
    if (t < 0.7) { return mix(c1, c2, (t - 0.4) / 0.3); }
    return mix(c2, c3, (t - 0.7) / 0.3);
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let sizeF = vec2<f32>(params.size);
    let p = frag.uv * sizeF;
    let cellXY = min(vec2<u32>(p), params.size - vec2<u32>(1u));
    if (walls[cellXY.y * params.size.x + cellXY.x] != 0u) {
        return vec4<f32>(0.62, 0.56, 0.46, 1.0);   // sandstone walls
    }

    // Surface normal from central differences of the height field
    let h = heightBilinear(p);
    let dhx = heightBilinear(p + vec2<f32>(1.0, 0.0)) - heightBilinear(p - vec2<f32>(1.0, 0.0));
    let dhy = heightBilinear(p + vec2<f32>(0.0, 1.0)) - heightBilinear(p - vec2<f32>(0.0, 1.0));
    let n = normalize(vec3<f32>(-dhx * SLOPE, -dhy * SLOPE, 1.0));

    // Light from the upper left (screen y points down), viewer straight above
    let lightDir = normalize(vec3<f32>(-0.4, -0.6, 0.7));
    let viewDir = vec3<f32>(0.0, 0.0, 1.0);
    let diffuse = max(dot(n, lightDir), 0.0);
    let halfVec = normalize(lightDir + viewDir);
    let specular = pow(max(dot(n, halfVec), 0.0), 80.0);

    let base = waterRamp(0.5 + 0.5 * tanh(HEIGHT_GAIN * h));
    // Normalize diffuse so flat water (n = +z) keeps its base color
    let shade = diffuse / lightDir.z;
    let color = base * (0.25 + 0.75 * shade) + vec3<f32>(0.9, 0.95, 1.0) * specular * 0.5;
    return vec4<f32>(min(color, vec3<f32>(1.0)), 1.0);
}
`;

const PRESETS = {
    open:    { name: 'Open water' },
    slit:    { name: 'Double slit' },
    ellipse: { name: 'Ellipse' },
};

const DROP_SIGMA = 4.0;          // must match the compute shader
const SOURCE_PERIOD = 20;        // steps per source oscillation; wavelength = C · period cells
const HOLD_INTERVAL_MS = 150;    // holding the mouse drops a ripple this often

function rippleProfile(px, py, cx, cy, sigma) {
    const dx = px - cx;
    const dy = py - cy;
    const s = (dx * dx + dy * dy) / (2 * sigma * sigma);
    return s > 9 ? 0 : (1 - s) * Math.exp(-s);
}

class WaveEquation {
    constructor(device, canvas, width, height) {
        this.device = device;
        this.canvas = canvas;
        this.width = width;
        this.height = height;
        this.cellCount = width * height;
        this.running = false;
        this.destroyed = false;

        this.preset = 'open';
        this.courant = 0.5;
        this.dampingGamma = 0.004;  // γ·dt; amplitude decays by about e^(-γ/2) per step
        this.substeps = 4;
        this.rain = false;
        this.absorb = false;
        this.planeWave = false;
        this.dropAmp = 1.0;
        this.seedValue = (Math.random() * 0xffffffff) >>> 0;

        this.cursor = [0, 0];
        this.cursorPrev = [0, 0];
        this.holding = false;
        this.drawingWalls = false;
        this.dropPending = false;
        this.lastDropTime = 0;
        this.editFramePending = false;

        this.steps = 0;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;

        // One 256-byte-aligned slot of Params per substep (dynamic uniform offsets)
        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.params = new ArrayBuffer(this.paramsStride * MAX_SUBSTEPS);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);

        this.setupBuffers();
        this.setupComputePipelines();
        this.setupRenderPipeline();
        this.setupCanvas();
        this.setupMouseControls();
    }

    setupBuffers() {
        this.heightBuffers = [0, 1].map(() => this.device.createBuffer({
            size: this.cellCount * 4, // f32 per cell
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        }));
        this.wallBuffer = this.device.createBuffer({
            size: this.cellCount * 4, // u32 per cell
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.paramsBuffer = this.device.createBuffer({
            size: this.paramsStride * MAX_SUBSTEPS,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    setupComputePipelines() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE } }
            ]
        });
        const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.editPipeline = this.device.createComputePipeline({
            layout,
            compute: { module: shaderModule, entryPoint: 'cs_edit' }
        });
        this.stepPipeline = this.device.createComputePipeline({
            layout,
            compute: { module: shaderModule, entryPoint: 'cs_step' }
        });

        // computeBindGroups[i]: heightBuffers[i] is u(t), the other holds u(t-1) and receives u(t+1)
        this.computeBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.heightBuffers[i] } },
                { binding: 1, resource: { buffer: this.heightBuffers[1 - i] } },
                { binding: 2, resource: { buffer: this.wallBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        }));

        this.currentBuffer = 0;
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });

        // renderBindGroups[i] draws heightBuffers[i], i.e. u(t) when currentBuffer === i
        this.renderBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.heightBuffers[i] } },
                { binding: 1, resource: { buffer: this.wallBuffer } },
                { binding: 2, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        }));
    }

    setupCanvas() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });
    }

    setupMouseControls() {
        // The canvas is CSS-scaled, so convert client coordinates to canvas pixels, then to cells
        const toCells = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const px = (e.clientX - rect.left) * this.canvas.width / rect.width;
            const py = (e.clientY - rect.top) * this.canvas.height / rect.height;
            return [px * this.width / this.canvas.width, py * this.height / this.canvas.height];
        };

        this.onMouseDown = (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            this.cursor = toCells(e);
            this.cursorPrev = this.cursor.slice();
            if (e.shiftKey) {
                this.drawingWalls = true;
            } else {
                this.holding = true;
                this.dropPending = true;
                this.lastDropTime = performance.now();
            }
            this.requestEditFrame();
        };
        this.onMouseMove = (e) => {
            if (!this.holding && !this.drawingWalls) return;
            this.cursor = toCells(e);
            if (this.drawingWalls) this.requestEditFrame();
        };
        this.onMouseUp = () => {
            this.holding = false;
            this.drawingWalls = false;
        };

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        this.canvas.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
    }

    // While paused, clicks and wall strokes still apply: one edit-only pass per animation frame
    requestEditFrame() {
        if (this.running || this.editFramePending) return;
        this.editFramePending = true;
        requestAnimationFrame(() => {
            this.editFramePending = false;
            if (!this.running && !this.destroyed) this.frame(0);
        });
    }

    // Writes `slots` copies of Params (one per substep); see the layout table at the top of the file
    writeParams(slots, dropOn, wallOn) {
        const wordsPerSlot = this.paramsStride / 4;
        for (let k = 0; k < slots; k++) {
            const o = k * wordsPerSlot;
            this.paramsU32[o + 0] = this.width;
            this.paramsU32[o + 1] = this.height;
            this.paramsF32[o + 2] = this.cursor[0];
            this.paramsF32[o + 3] = this.cursor[1];
            this.paramsF32[o + 4] = this.cursorPrev[0];
            this.paramsF32[o + 5] = this.cursorPrev[1];
            this.paramsF32[o + 6] = this.courant * this.courant;
            this.paramsF32[o + 7] = this.dampingGamma / 2;
            this.paramsU32[o + 8] = (this.steps + k) >>> 0;
            this.paramsU32[o + 9] = SOURCE_PERIOD;
            this.paramsU32[o + 10] = dropOn ? 1 : 0;
            this.paramsU32[o + 11] = wallOn ? 1 : 0;
            this.paramsU32[o + 12] = this.rain ? 1 : 0;
            this.paramsU32[o + 13] = this.planeWave ? 1 : 0;
            this.paramsU32[o + 14] = this.absorb ? 1 : 0;
            this.paramsU32[o + 15] = this.seedValue;
            this.paramsF32[o + 16] = this.dropAmp;
        }
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params, 0, slots * this.paramsStride);
    }

    // Clears the water (both time levels); walls stay
    clearWater() {
        const zeros = new Float32Array(this.cellCount);
        this.device.queue.writeBuffer(this.heightBuffers[0], 0, zeros);
        this.device.queue.writeBuffer(this.heightBuffers[1], 0, zeros);
        this.currentBuffer = 0;
        this.steps = 0;
    }

    reset() {
        this.clearWater();
        this.render();
    }

    // Builds the preset's walls and initial water on the CPU and uploads them
    loadPreset(key) {
        const w = this.width;
        const h = this.height;
        const wallData = new Uint32Array(this.cellCount);
        const water = new Float32Array(this.cellCount);
        const addDrop = (cx, cy, amp) => {
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    water[y * w + x] += amp * rippleProfile(x + 0.5, y + 0.5, cx, cy, DROP_SIGMA);
                }
            }
        };

        this.preset = key;
        this.planeWave = false;
        this.absorb = false;

        if (key === 'slit') {
            // A 3-cell-thick wall with two 8-cell gaps 48 cells apart; a plane wave arrives from the left
            const wallX = Math.round(w * 0.35);
            const gapHalf = 4;
            const gapOffset = 24;
            const cy = h / 2;
            for (let y = 0; y < h; y++) {
                const inGap = Math.abs(y + 0.5 - (cy - gapOffset)) < gapHalf ||
                              Math.abs(y + 0.5 - (cy + gapOffset)) < gapHalf;
                if (inGap) continue;
                for (let x = wallX; x < wallX + 3; x++) wallData[y * w + x] = 1;
            }
            this.planeWave = true;
            this.absorb = true;
        } else if (key === 'ellipse') {
            // Everything outside an ellipse is wall; a drop at one focus refocuses at the other
            const cx = w / 2;
            const cy = h / 2;
            const a = w * 0.45;
            const b = h * 0.42;
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const ex = (x + 0.5 - cx) / a;
                    const ey = (y + 0.5 - cy) / b;
                    if (ex * ex + ey * ey > 1) wallData[y * w + x] = 1;
                }
            }
            const focus = Math.sqrt(a * a - b * b);
            addDrop(cx - focus, cy, this.dropAmp);
        } else {
            addDrop(w / 2, h / 2, this.dropAmp);
        }

        for (let i = 0; i < this.cellCount; i++) {
            if (wallData[i]) water[i] = 0;
        }
        this.device.queue.writeBuffer(this.wallBuffer, 0, wallData);
        // Same field in both time levels: the drop starts at rest
        this.device.queue.writeBuffer(this.heightBuffers[0], 0, water);
        this.device.queue.writeBuffer(this.heightBuffers[1], 0, water);
        this.currentBuffer = 0;
        this.steps = 0;
        this.render();
    }

    // Runs an edit pass if a drop or wall stroke is pending, then `substeps` time steps, then draws
    frame(substeps) {
        const passes = Math.min(Math.max(substeps, 0), MAX_SUBSTEPS);
        const dropOn = this.dropPending;
        const wallOn = this.drawingWalls;
        const edit = dropOn || wallOn;
        this.writeParams(Math.max(passes, 1), dropOn, wallOn);

        const commandEncoder = this.device.createCommandEncoder();
        if (edit || passes > 0) {
            const computePass = commandEncoder.beginComputePass();
            const wgX = Math.ceil(this.width / 8);
            const wgY = Math.ceil(this.height / 8);
            if (edit) {
                // Edits both time levels in place; no buffer swap
                computePass.setPipeline(this.editPipeline);
                computePass.setBindGroup(0, this.computeBindGroups[this.currentBuffer], [0]);
                computePass.dispatchWorkgroups(wgX, wgY);
            }
            if (passes > 0) {
                computePass.setPipeline(this.stepPipeline);
                for (let i = 0; i < passes; i++) {
                    computePass.setBindGroup(0, this.computeBindGroups[this.currentBuffer], [i * this.paramsStride]);
                    computePass.dispatchWorkgroups(wgX, wgY);
                    this.currentBuffer = 1 - this.currentBuffer; // u(t+1) is now the current level
                }
            }
            computePass.end();
        }
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);

        this.steps += passes;
        this.dropPending = false;
        this.cursorPrev = this.cursor.slice();
    }

    step() {
        // Holding the mouse button keeps dropping ripples at the cursor
        const now = performance.now();
        if (this.holding && now - this.lastDropTime >= HOLD_INTERVAL_MS) {
            this.dropPending = true;
            this.lastDropTime = now;
        }
        this.frame(this.substeps);
    }

    encodeRender(commandEncoder) {
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.01, g: 0.04, b: 0.12, a: 1.0 }
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBuffer]);
        renderPass.draw(3);
        renderPass.end();
    }

    render() {
        this.writeParams(1, false, false);
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
    }

    animate() {
        if (!this.running) return;

        this.step();

        this.framesSinceSample++;
        const now = performance.now();
        if (now - this.lastSampleTime >= 500) {
            this.fps = this.framesSinceSample * 1000 / (now - this.lastSampleTime);
            this.framesSinceSample = 0;
            this.lastSampleTime = now;
            if (this.onStats) this.onStats();
        }

        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.running = false;
        this.destroyed = true;
        this.canvas.removeEventListener('mousedown', this.onMouseDown);
        this.canvas.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.heightBuffers.forEach(b => b.destroy());
        this.wallBuffer.destroy();
        this.paramsBuffer.destroy();
    }
}

export async function initWaveEquation(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-wave');
    const output = document.getElementById('output-wave');

    const width = 512;
    const height = 384;

    const simulation = new WaveEquation(device, canvas, width, height);
    simulation.loadPreset('open');
    simulation.onStats = () => updateOutput();

    document.getElementById('btn-wave-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-wave-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-wave-reset').addEventListener('click', () => {
        simulation.reset();
        updateOutput();
    });

    const speedSlider = document.getElementById('wave-speed');
    const speedValue = document.getElementById('wave-speed-value');
    speedSlider.addEventListener('input', (e) => {
        // Clamp to the CFL limit C ≤ 1/√2 with a margin
        simulation.courant = Math.min(Math.max(parseFloat(e.target.value), 0.1), 0.7);
        speedValue.textContent = simulation.courant.toFixed(2);
        updateOutput();
    });

    const dampingSlider = document.getElementById('wave-damping');
    const dampingValue = document.getElementById('wave-damping-value');
    dampingSlider.addEventListener('input', (e) => {
        simulation.dampingGamma = Math.min(Math.max(parseFloat(e.target.value), 0), 0.02);
        dampingValue.textContent = simulation.dampingGamma.toFixed(3);
        updateOutput();
    });

    const stepsSlider = document.getElementById('wave-steps');
    const stepsValue = document.getElementById('wave-steps-value');
    stepsSlider.addEventListener('input', (e) => {
        simulation.substeps = Math.min(Math.max(parseInt(e.target.value), 1), MAX_SUBSTEPS);
        stepsValue.textContent = simulation.substeps;
        updateOutput();
    });

    const rainButton = document.getElementById('btn-wave-rain');
    rainButton.addEventListener('click', () => {
        simulation.rain = !simulation.rain;
        showToggles();
        updateOutput();
    });

    const absorbButton = document.getElementById('btn-wave-absorb');
    absorbButton.addEventListener('click', () => {
        simulation.absorb = !simulation.absorb;
        showToggles();
        updateOutput();
    });

    const presetButtons = document.querySelectorAll('.wave-preset');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!PRESETS[btn.dataset.preset]) return;
            simulation.loadPreset(btn.dataset.preset);
            showToggles();
            updateOutput();
        });
    });

    function showToggles() {
        rainButton.classList.toggle('active', simulation.rain);
        rainButton.textContent = simulation.rain ? 'Rain: on' : 'Rain: off';
        absorbButton.classList.toggle('active', simulation.absorb);
        absorbButton.textContent = simulation.absorb ? 'Absorbing edges: on' : 'Absorbing edges: off';
        presetButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === simulation.preset));
    }

    function updateOutput() {
        const stepsPerSecond = simulation.running ? Math.round(simulation.fps * simulation.substeps) : 0;
        const wavelength = (simulation.courant * SOURCE_PERIOD).toFixed(1);

        output.innerHTML = `<span class="success">✓ Wave Equation Active</span>

<span class="info">Configuration:</span>
• Grid size: ${width} × ${height} (${(width * height).toLocaleString()} cells)
• Preset: ${PRESETS[simulation.preset].name}${simulation.planeWave ? ` (plane wave, period ${SOURCE_PERIOD} steps, wavelength ${wavelength} cells)` : ''}
• Courant number C = c·dt/dx: ${simulation.courant.toFixed(2)} (stable for C ≤ 1/√2 ≈ 0.707)
• Damping γ·dt: ${simulation.dampingGamma.toFixed(3)} (waves lose about ${(simulation.dampingGamma * 50).toFixed(2)}% of their height per step)
• Steps per frame: ${simulation.substeps}
• Rain: ${simulation.rain ? 'on' : 'off'} · Edges: ${simulation.absorb ? 'absorbing sponge' : 'reflecting (u = 0)'}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Algorithm (leapfrog finite differences):</span>
• Damped wave equation u_tt + γu_t = c²∇²u, centered differences, 5-point Laplacian
• u(t+1) = (2u(t) − (1 − γ/2)·u(t−1) + C²·∇²u(t)) / (1 + γ/2)
• Two storage buffers hold u(t) and u(t−1); u(t+1) overwrites u(t−1), then they swap
• Walls and the outer boundary are fixed at u = 0, so waves reflect off them
• Shading: normal from height differences, diffuse + specular light, height → color ramp

<span class="info">Performance:</span>
• Frame rate: ${simulation.running ? simulation.fps.toFixed(0) : 0} FPS
• Simulation steps: ${simulation.steps.toLocaleString()} (${stepsPerSecond.toLocaleString()} / s)
• Cell updates per frame: ${(width * height * simulation.substeps).toLocaleString()}
• One compute dispatch per step (plus one for clicks and wall strokes); no CPU readback`;
    }

    showToggles();
    updateOutput();
    return simulation;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}
