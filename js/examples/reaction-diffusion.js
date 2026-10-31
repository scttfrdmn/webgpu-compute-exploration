// Gray-Scott reaction-diffusion on a toroidal 2D grid.
//
// Uniform layout (struct Params, 48 bytes; shared by the compute and render shaders):
//   offset  0  size        vec2<u32>  grid width, height      -> Uint32 view [0], [1]
//   offset  8  brushPos    vec2<f32>  brush position in cells -> Float32 view [2], [3]
//   offset 16  brushPrev   vec2<f32>  previous brush position -> Float32 view [4], [5]
//   offset 24  feed        f32                                -> Float32 view [6]
//   offset 28  kill        f32                                -> Float32 view [7]
//   offset 32  brushRadius f32        in cells                -> Float32 view [8]
//   offset 36  brushOn     u32        1 while painting        -> Uint32 view [9]
//   offset 40  simulate    u32        0 = only apply the brush -> Uint32 view [10]
//   offset 44  palette     u32        color ramp index        -> Uint32 view [11]

import { GpuTimer } from '../gpu-timer.js';

const PARAMS_SIZE = 48;

const paramsStruct = `
struct Params {
    size: vec2<u32>,
    brushPos: vec2<f32>,
    brushPrev: vec2<f32>,
    feed: f32,
    kill: f32,
    brushRadius: f32,
    brushOn: u32,
    simulate: u32,
    palette: u32,
}
`;

const computeShader = paramsStruct + `
// Each cell stores (U, V), the concentrations of the two chemicals.
@group(0) @binding(0) var<storage, read> cellsIn: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cellsOut: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const DU: f32 = 1.0;   // diffusion rate of U
const DV: f32 = 0.5;   // diffusion rate of V (slower, which is what makes patterns form)
const DT: f32 = 1.0;   // time step

// Toroidal wrap: the grid has no edges
fn cellAt(x: i32, y: i32) -> vec2<f32> {
    let w = i32(params.size.x);
    let h = i32(params.size.y);
    let wx = (x + w) % w;
    let wy = (y + h) % h;
    return cellsIn[u32(wy) * params.size.x + u32(wx)];
}

// Distance from p to the segment a-b (the brush stroke since the last frame)
fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let len2 = dot(ab, ab);
    var t = 0.0;
    if (len2 > 0.0) {
        t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    }
    return length(p - (a + ab * t));
}

// Gray-Scott model:
//   dU/dt = Du * lap(U) - U*V^2 + F * (1 - U)
//   dV/dt = Dv * lap(V) + U*V^2 - (F + k) * V
// U is fed in at rate F; the reaction U + 2V -> 3V converts U into V; V is removed at rate F + k.
// lap() is a 3x3 convolution: center -1, edge neighbors 0.2, corner neighbors 0.05.
@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = cellAt(x, y);
    var next = c;

    if (params.simulate != 0u) {
        let edges = cellAt(x - 1, y) + cellAt(x + 1, y) + cellAt(x, y - 1) + cellAt(x, y + 1);
        let corners = cellAt(x - 1, y - 1) + cellAt(x + 1, y - 1)
                    + cellAt(x - 1, y + 1) + cellAt(x + 1, y + 1);
        let lap = -c + 0.2 * edges + 0.05 * corners;

        let u = c.x;
        let v = c.y;
        let uvv = u * v * v;
        let du = DU * lap.x - uvv + params.feed * (1.0 - u);
        let dv = DV * lap.y + uvv - (params.feed + params.kill) * v;
        next = clamp(c + DT * vec2<f32>(du, dv), vec2<f32>(0.0), vec2<f32>(1.0));
    }

    // Painting: cells under the brush stroke get the same mix used for seeding
    if (params.brushOn != 0u) {
        let p = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
        if (segmentDistance(p, params.brushPrev, params.brushPos) < params.brushRadius) {
            next = vec2<f32>(0.5, 0.25);
        }
    }

    cellsOut[gid.y * params.size.x + gid.x] = next;
}
`;

const renderShader = paramsStruct + `
@group(0) @binding(0) var<storage, read> cells: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

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

// Piecewise-linear color ramp through five stops at t = 0, 0.2, 0.45, 0.7, 1
fn ramp(t: f32, c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, c4: vec3<f32>) -> vec3<f32> {
    if (t < 0.2) { return mix(c0, c1, t / 0.2); }
    if (t < 0.45) { return mix(c1, c2, (t - 0.2) / 0.25); }
    if (t < 0.7) { return mix(c2, c3, (t - 0.45) / 0.25); }
    return mix(c3, c4, (t - 0.7) / 0.3);
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let cellXY = min(vec2<u32>(frag.uv * vec2<f32>(params.size)), params.size - vec2<u32>(1u));
    let v = cells[cellXY.y * params.size.x + cellXY.x].y;

    // V rarely exceeds ~0.45; sqrt lifts faint fronts so they stay visible on the dark background
    let t = sqrt(clamp(v / 0.45, 0.0, 1.0));

    var color: vec3<f32>;
    if (params.palette == 1u) {
        // Magma
        color = ramp(t, vec3<f32>(0.020, 0.016, 0.050), vec3<f32>(0.230, 0.060, 0.380),
                        vec3<f32>(0.700, 0.200, 0.420), vec3<f32>(0.980, 0.520, 0.260),
                        vec3<f32>(0.990, 0.950, 0.700));
    } else if (params.palette == 2u) {
        // Mono
        color = ramp(t, vec3<f32>(0.030, 0.035, 0.045), vec3<f32>(0.150, 0.160, 0.180),
                        vec3<f32>(0.420, 0.440, 0.470), vec3<f32>(0.720, 0.740, 0.760),
                        vec3<f32>(0.960, 0.960, 0.960));
    } else {
        // Ocean
        color = ramp(t, vec3<f32>(0.020, 0.027, 0.055), vec3<f32>(0.040, 0.180, 0.300),
                        vec3<f32>(0.090, 0.520, 0.640), vec3<f32>(0.550, 0.880, 0.820),
                        vec3<f32>(1.000, 0.960, 0.820));
    }
    return vec4<f32>(color, 1.0);
}
`;

// (feed, kill) pairs, each checked against a CPU replica of this update rule
const PRESETS = {
    coral:   { name: 'Coral',   feed: 0.0545, kill: 0.062 },
    mitosis: { name: 'Mitosis', feed: 0.0367, kill: 0.0649 },
    maze:    { name: 'Maze',    feed: 0.029,  kill: 0.057 },
    holes:   { name: 'Holes',   feed: 0.039,  kill: 0.058 },
    chaos:   { name: 'Chaos',   feed: 0.018,  kill: 0.051 },
    cells:   { name: 'Cells',   feed: 0.062,  kill: 0.0609 },
};

const PALETTES = ['Ocean', 'Magma', 'Mono'];
const MAX_SUBSTEPS = 32;

class ReactionDiffusion {
    constructor(device, canvas, width, height) {
        this.device = device;
        this.gpuTimer = new GpuTimer(device, { maxPasses: 2 });
        this.canvas = canvas;
        this.width = width;
        this.height = height;
        this.cellCount = width * height;
        this.running = false;
        this.destroyed = false;

        this.preset = 'coral';
        this.feed = PRESETS.coral.feed;
        this.kill = PRESETS.coral.kill;
        this.substeps = 12;
        this.palette = 0;

        this.brushPos = [0, 0];
        this.brushPrev = [0, 0];
        this.brushRadius = 6;
        this.painting = false;
        this.paintFramePending = false;

        this.steps = 0;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;

        this.params = new ArrayBuffer(PARAMS_SIZE);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);

        this.setupBuffers();
        this.setupComputePipeline();
        this.setupRenderPipeline();
        this.setupCanvas();
        this.setupMouseControls();
    }

    setupBuffers() {
        const bufferSize = this.cellCount * 8; // vec2<f32> per cell
        this.cellBuffers = [0, 1].map(() => this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        }));

        this.paramsBuffer = this.device.createBuffer({
            size: PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    setupComputePipeline() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'cs_main' }
        });

        // computeBindGroups[i] reads cellBuffers[i] and writes the other one
        this.computeBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cellBuffers[i] } },
                { binding: 1, resource: { buffer: this.cellBuffers[1 - i] } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ]
        }));

        this.currentBuffer = 0;
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
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

        this.renderBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cellBuffers[i] } },
                { binding: 1, resource: { buffer: this.paramsBuffer } }
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
            this.painting = true;
            this.brushPos = toCells(e);
            this.brushPrev = this.brushPos.slice();
            this.requestPaintFrame();
        };
        this.onMouseMove = (e) => {
            if (!this.painting) return;
            this.brushPos = toCells(e);
            this.requestPaintFrame();
        };
        this.onMouseUp = () => {
            this.painting = false;
        };

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        this.canvas.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
    }

    // While paused, painting still updates the grid: one brush-only pass per animation frame
    requestPaintFrame() {
        if (this.running || this.paintFramePending) return;
        this.paintFramePending = true;
        requestAnimationFrame(() => {
            this.paintFramePending = false;
            if (!this.running && !this.destroyed) this.frame(0);
        });
    }

    // Writes the uniform buffer; see the layout table at the top of the file
    writeParams(simulate, brushOn) {
        this.paramsU32[0] = this.width;
        this.paramsU32[1] = this.height;
        this.paramsF32[2] = this.brushPos[0];
        this.paramsF32[3] = this.brushPos[1];
        this.paramsF32[4] = this.brushPrev[0];
        this.paramsF32[5] = this.brushPrev[1];
        this.paramsF32[6] = this.feed;
        this.paramsF32[7] = this.kill;
        this.paramsF32[8] = this.brushRadius;
        this.paramsU32[9] = brushOn ? 1 : 0;
        this.paramsU32[10] = simulate ? 1 : 0;
        this.paramsU32[11] = this.palette;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params);
    }

    // Reset: U = 1, V = 0 everywhere, plus a few random square patches of U = 0.5, V = 0.25
    seed() {
        const data = new Float32Array(this.cellCount * 2);
        for (let i = 0; i < this.cellCount; i++) {
            data[i * 2] = 1.0;
        }
        const patches = Math.max(4, Math.round(this.cellCount / 10000));
        const half = 5;
        for (let p = 0; p < patches; p++) {
            const cx = Math.floor(Math.random() * this.width);
            const cy = Math.floor(Math.random() * this.height);
            for (let dy = -half; dy < half; dy++) {
                for (let dx = -half; dx < half; dx++) {
                    const x = (cx + dx + this.width) % this.width;
                    const y = (cy + dy + this.height) % this.height;
                    const i = y * this.width + x;
                    data[i * 2] = 0.5;
                    data[i * 2 + 1] = 0.25 + (Math.random() - 0.5) * 0.02;
                }
            }
        }
        this.currentBuffer = 0;
        this.device.queue.writeBuffer(this.cellBuffers[0], 0, data);
        this.steps = 0;
        this.render();
    }

    // Runs `substeps` simulation steps (or one brush-only pass if substeps is 0), then draws
    frame(substeps) {
        const brushOn = this.painting;
        const passes = substeps > 0 ? Math.min(substeps, MAX_SUBSTEPS) : (brushOn ? 1 : 0);
        this.writeParams(substeps > 0, brushOn);

        const commandEncoder = this.device.createCommandEncoder();
        if (passes > 0) {
            const computePass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('sim'));
            computePass.setPipeline(this.computePipeline);
            const wgX = Math.ceil(this.width / 8);
            const wgY = Math.ceil(this.height / 8);
            for (let i = 0; i < passes; i++) {
                computePass.setBindGroup(0, this.computeBindGroups[this.currentBuffer]);
                computePass.dispatchWorkgroups(wgX, wgY);
                this.currentBuffer = 1 - this.currentBuffer;
            }
            computePass.end();
        }
        this.encodeRender(commandEncoder, true);
        this.gpuTimer.resolve(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        this.gpuTimer.read();

        if (substeps > 0) this.steps += passes;
        this.brushPrev = this.brushPos.slice();
    }

    step() {
        this.frame(this.substeps);
    }

    // timed: add timestamp writes (only frame() resolves the timer)
    encodeRender(commandEncoder, timed = false) {
        const desc = {
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.02, g: 0.027, b: 0.055, a: 1.0 }
            }]
        };
        const renderPass = commandEncoder.beginRenderPass(timed ? this.gpuTimer.renderPassDescriptor(desc, 'render') : desc);
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBuffer]);
        renderPass.draw(3);
        renderPass.end();
    }

    render() {
        this.writeParams(false, false);
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
        this.gpuTimer.destroy();
        this.canvas.removeEventListener('mousedown', this.onMouseDown);
        this.canvas.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.cellBuffers.forEach(b => b.destroy());
        this.paramsBuffer.destroy();
    }
}

export async function initReactionDiffusion(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-rd');
    const output = document.getElementById('output-rd');

    const width = 512;
    const height = 384;

    const simulation = new ReactionDiffusion(device, canvas, width, height);
    simulation.seed();
    simulation.onStats = () => updateOutput();

    document.getElementById('btn-rd-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-rd-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-rd-reset').addEventListener('click', () => {
        simulation.seed();
        updateOutput();
    });

    const feedSlider = document.getElementById('rd-feed');
    const feedValue = document.getElementById('rd-feed-value');
    const killSlider = document.getElementById('rd-kill');
    const killValue = document.getElementById('rd-kill-value');
    const presetButtons = document.querySelectorAll('.rd-preset');

    function showParams() {
        feedSlider.value = simulation.feed;
        killSlider.value = simulation.kill;
        feedValue.textContent = simulation.feed.toFixed(4);
        killValue.textContent = simulation.kill.toFixed(4);
        presetButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === simulation.preset));
    }

    feedSlider.addEventListener('input', (e) => {
        simulation.feed = parseFloat(e.target.value);
        simulation.preset = null;
        showParams();
        updateOutput();
    });

    killSlider.addEventListener('input', (e) => {
        simulation.kill = parseFloat(e.target.value);
        simulation.preset = null;
        showParams();
        updateOutput();
    });

    const speedSlider = document.getElementById('rd-speed');
    const speedValue = document.getElementById('rd-speed-value');
    speedSlider.addEventListener('input', (e) => {
        simulation.substeps = Math.min(Math.max(parseInt(e.target.value), 1), MAX_SUBSTEPS);
        speedValue.textContent = simulation.substeps;
        updateOutput();
    });

    document.getElementById('rd-palette').addEventListener('change', (e) => {
        simulation.palette = Math.min(Math.max(parseInt(e.target.value), 0), PALETTES.length - 1);
        if (!simulation.running) simulation.render();
        updateOutput();
    });

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const preset = PRESETS[btn.dataset.preset];
            if (!preset) return;
            simulation.preset = btn.dataset.preset;
            simulation.feed = preset.feed;
            simulation.kill = preset.kill;
            showParams();
            simulation.seed();
            updateOutput();
        });
    });

    function updateOutput() {
        const presetName = simulation.preset ? PRESETS[simulation.preset].name : 'Custom';
        const stepsPerSecond = simulation.running ? Math.round(simulation.fps * simulation.substeps) : 0;

        output.innerHTML = `<span class="success">✓ Reaction-Diffusion Active</span>

<span class="info">Configuration:</span>
• Grid size: ${width} × ${height} (${(width * height).toLocaleString()} cells, wraps at the edges)
• Preset: ${presetName} (feed ${simulation.feed.toFixed(4)}, kill ${simulation.kill.toFixed(4)})
• Substeps per frame: ${simulation.substeps}
• Palette: ${PALETTES[simulation.palette]}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Algorithm (Gray-Scott):</span>
• dU/dt = Du·∇²U − UV² + F(1 − U)
• dV/dt = Dv·∇²V + UV² − (F + k)V
• Du = 1.0, Dv = 0.5, dt = 1.0; 3×3 Laplacian (center −1, edges 0.2, corners 0.05)
• Two ping-pong storage buffers of (U, V) pairs; the color shows V

<span class="info">Performance:</span>
• Frame rate: ${simulation.running ? simulation.fps.toFixed(0) : 0} FPS
• ${simulation.gpuTimer.frameText(simulation.running ? simulation.fps : 0, ['sim', 'render'])}
• Simulation steps: ${simulation.steps.toLocaleString()} (${stepsPerSecond.toLocaleString()} / s)
• Cell updates per frame: ${(width * height * simulation.substeps).toLocaleString()}
• One compute dispatch per substep; no CPU readback`;
    }

    showParams();
    updateOutput();
    return simulation;
}

export { ReactionDiffusion, PRESETS };   // for test/reaction-diffusion.test.mjs

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}
