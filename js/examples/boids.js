// Boids Flocking Simulation
// Based on Craig Reynolds' algorithm. Neighbors are found by brute force: each boid loops over all N
// boids once per rule (separation, alignment, cohesion), so a step costs 3·N² distance tests; there is
// no spatial hashing.

import { GpuTimer } from '../gpu-timer.js';
import { readBuffer } from '../webgpu-utils.js';

const COMPUTE_SHADER = `
struct Boid {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

struct Params {
    numBoids: u32,
    separationRadius: f32,
    alignmentRadius: f32,
    cohesionRadius: f32,
    separationWeight: f32,
    alignmentWeight: f32,
    cohesionWeight: f32,
    maxSpeed: f32,
    maxForce: f32,
    boundsMin: vec2<f32>,
    boundsMax: vec2<f32>,
    mousePos: vec2<f32>,
    mouseRadius: f32,
    mouseForce: f32,
    behaviorMode: u32,
}

@group(0) @binding(0) var<storage, read> boidsIn: array<Boid>;
@group(0) @binding(1) var<storage, read_write> boidsOut: array<Boid>;
@group(0) @binding(2) var<uniform> params: Params;

// normalize() of a zero vector is NaN; return zero instead (e.g. neighbors' velocities cancel)
fn safeNormalize(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if (len < 1e-6) {
        return vec2<f32>(0.0);
    }
    return v / len;
}

fn limit(v: vec2<f32>, maxLen: f32) -> vec2<f32> {
    let len = length(v);
    if (len > maxLen) {
        return normalize(v) * maxLen;
    }
    return v;
}

fn separation(boid: Boid) -> vec2<f32> {
    var steering = vec2<f32>(0.0);
    var count = 0u;

    for (var i = 0u; i < params.numBoids; i++) {
        let other = boidsIn[i];
        let diff = boid.position - other.position;
        let dist = length(diff);

        if (dist > 0.0 && dist < params.separationRadius) {
            steering += normalize(diff) / dist;
            count++;
        }
    }

    if (count > 0u) {
        steering /= f32(count);
        if (length(steering) > 0.0) {
            steering = safeNormalize(steering) * params.maxSpeed - boid.velocity;
            steering = limit(steering, params.maxForce);
        }
    }

    return steering;
}

fn alignment(boid: Boid) -> vec2<f32> {
    var avgVel = vec2<f32>(0.0);
    var count = 0u;

    for (var i = 0u; i < params.numBoids; i++) {
        let other = boidsIn[i];
        let dist = distance(boid.position, other.position);

        if (dist > 0.0 && dist < params.alignmentRadius) {
            avgVel += other.velocity;
            count++;
        }
    }

    if (count > 0u) {
        avgVel /= f32(count);
        avgVel = safeNormalize(avgVel) * params.maxSpeed;
        var steering = avgVel - boid.velocity;
        steering = limit(steering, params.maxForce);
        return steering;
    }

    return vec2<f32>(0.0);
}

fn cohesion(boid: Boid) -> vec2<f32> {
    var avgPos = vec2<f32>(0.0);
    var count = 0u;

    for (var i = 0u; i < params.numBoids; i++) {
        let other = boidsIn[i];
        let dist = distance(boid.position, other.position);

        if (dist > 0.0 && dist < params.cohesionRadius) {
            avgPos += other.position;
            count++;
        }
    }

    if (count > 0u) {
        avgPos /= f32(count);
        var desired = avgPos - boid.position;
        desired = safeNormalize(desired) * params.maxSpeed;
        var steering = desired - boid.velocity;
        steering = limit(steering, params.maxForce);
        return steering;
    }

    return vec2<f32>(0.0);
}

fn mouseInteraction(boid: Boid) -> vec2<f32> {
    let diff = params.mousePos - boid.position;
    let dist = length(diff);

    // Positive mouseForce pulls boids toward the cursor, negative pushes them away;
    // strongest at the cursor and fading to zero at mouseRadius
    if (dist < params.mouseRadius && dist > 0.0) {
        return normalize(diff) * params.mouseForce * (1.0 - dist / params.mouseRadius);
    }

    return vec2<f32>(0.0);
}

fn vortexBehavior(boid: Boid) -> vec2<f32> {
    let center = (params.boundsMin + params.boundsMax) * 0.5;
    let diff = boid.position - center;
    let dist = length(diff);

    if (dist > 10.0) {
        // Tangential force for rotation
        let tangent = vec2<f32>(-diff.y, diff.x);
        let toCenter = -diff / dist;
        return normalize(tangent + toCenter * 0.3) * params.maxSpeed * 2.0;
    }

    return vec2<f32>(0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numBoids) {
        return;
    }

    var boid = boidsIn[idx];

    // Compute steering forces
    var sep = separation(boid) * params.separationWeight;
    var ali = alignment(boid) * params.alignmentWeight;
    var coh = cohesion(boid) * params.cohesionWeight;
    var mouse = mouseInteraction(boid);

    // Apply behavior modes
    var acceleration = sep + ali + coh + mouse;

    switch params.behaviorMode {
        case 1u: { // Predator mode
            if (idx == 0u) {
                // First boid is predator
                let center = (params.boundsMin + params.boundsMax) * 0.5;
                let toCenter = safeNormalize(center - boid.position) * params.maxSpeed * 0.5;
                acceleration = toCenter;
            } else {
                // Others flee from predator
                let predator = boidsIn[0];
                let diff = boid.position - predator.position;
                let dist = length(diff);
                if (dist < 150.0 && dist > 0.0) {
                    acceleration += normalize(diff) * params.maxForce * 3.0;
                }
            }
        }
        case 2u: { // Scatter
            acceleration *= 0.1;
            let center = (params.boundsMin + params.boundsMax) * 0.5;
            let away = safeNormalize(boid.position - center);
            acceleration += away * params.maxForce * 2.0;
        }
        case 3u: { // Vortex
            acceleration += vortexBehavior(boid);
        }
        default: {}
    }

    // Update velocity and position
    boid.velocity += acceleration;
    boid.velocity = limit(boid.velocity, params.maxSpeed);
    boid.position += boid.velocity;

    // Wrap around boundaries
    if (boid.position.x < params.boundsMin.x) {
        boid.position.x = params.boundsMax.x;
    }
    if (boid.position.x > params.boundsMax.x) {
        boid.position.x = params.boundsMin.x;
    }
    if (boid.position.y < params.boundsMin.y) {
        boid.position.y = params.boundsMax.y;
    }
    if (boid.position.y > params.boundsMax.y) {
        boid.position.y = params.boundsMin.y;
    }

    boidsOut[idx] = boid;
}
`;

const RENDER_SHADER = `
struct Boid {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

struct Uniforms {
    canvasSize: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> boids: array<Boid>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> VertexOutput {
    let boid = boids[instanceIdx];

    // Triangle vertices pointing in direction of velocity
    let vertices = array<vec2<f32>, 3>(
        vec2<f32>(0.0, -1.0),
        vec2<f32>(-0.5, 0.5),
        vec2<f32>(0.5, 0.5)
    );

    let vert = vertices[vertexIdx];

    // Rotate based on velocity direction
    let angle = atan2(boid.velocity.y, boid.velocity.x) + 1.5708;
    let c = cos(angle);
    let s = sin(angle);
    let rotated = vec2<f32>(
        vert.x * c - vert.y * s,
        vert.x * s + vert.y * c
    );

    // Scale and position
    let size = 5.0;
    let worldPos = boid.position + rotated * size;

    // Convert to clip space
    let clipPos = (worldPos / uniforms.canvasSize) * 2.0 - 1.0;
    let finalPos = vec4<f32>(clipPos.x, -clipPos.y, 0.0, 1.0);

    // Color based on speed
    let speed = length(boid.velocity);
    let t = clamp(speed / 5.0, 0.0, 1.0);
    let color = mix(
        vec3<f32>(0.3, 0.8, 0.3),
        vec3<f32>(1.0, 0.9, 0.3),
        t
    );

    // Special color for first boid in predator mode
    var finalColor = color;
    if (instanceIdx == 0u) {
        finalColor = vec3<f32>(1.0, 0.2, 0.2);
    }

    var output: VertexOutput;
    output.position = finalPos;
    output.color = finalColor;
    return output;
}

@fragment
fn fragmentMain(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
`;

class BoidsSimulation {
    constructor(device, canvas, numBoids) {
        this.device = device;
        this.canvas = canvas;
        this.numBoids = numBoids;
        this.running = false;
        this.frameCount = 0;

        // Boid parameters
        this.separationRadius = 25.0;
        this.alignmentRadius = 50.0;
        this.cohesionRadius = 50.0;
        this.separationWeight = 1.5;
        this.alignmentWeight = 1.0;
        this.cohesionWeight = 1.0;
        this.maxSpeed = 4.0;
        this.maxForce = 0.1;

        this.behaviorMode = 0; // 0: normal, 1: predator, 2: scatter, 3: vortex

        // canvas = null: compute-only (Measure tab, test/gpu/) in a 1000 × 700 world like index.html's canvas
        const size = canvas ?? COMPUTE_ONLY_SIZE;
        this.mousePos = [size.width / 2, size.height / 2];
        this.mouseRadius = 150.0;
        this.mouseForce = 0.0;

        if (canvas) this.setupMouseControls();
        this.initializeBoids();
        this.setupBuffers();
        this.setupComputePipeline();
        if (canvas) this.setupRenderPipeline();
    }

    setupMouseControls() {
        // The canvas is CSS-scaled, so convert client coordinates to canvas pixels
        const toCanvas = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            return [
                (e.clientX - rect.left) * this.canvas.width / rect.width,
                (e.clientY - rect.top) * this.canvas.height / rect.height
            ];
        };

        this.canvas.addEventListener('mousedown', (e) => {
            this.mousePos = toCanvas(e);
            this.mouseForce = e.button === 0 ? 1.0 : -1.0; // Left: attract, Right: repel
        });

        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        this.canvas.addEventListener('mousemove', (e) => {
            this.mousePos = toCanvas(e);
        });

        window.addEventListener('mouseup', () => {
            this.mouseForce = 0.0;
        });
    }

    initializeBoids() {
        this.boids = new Float32Array(this.numBoids * 4); // pos(2) + vel(2)
        const size = this.canvas ?? COMPUTE_ONLY_SIZE;

        for (let i = 0; i < this.numBoids; i++) {
            this.boids[i * 4 + 0] = Math.random() * size.width;
            this.boids[i * 4 + 1] = Math.random() * size.height;
            this.boids[i * 4 + 2] = (Math.random() - 0.5) * this.maxSpeed;
            this.boids[i * 4 + 3] = (Math.random() - 0.5) * this.maxSpeed;
        }
    }

    setupBuffers() {
        const boidBufferSize = this.numBoids * 4 * 4;

        // COPY_SRC: the Measure tab reads the boids back
        this.boidBuffers = [
            this.device.createBuffer({
                size: boidBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            }),
            this.device.createBuffer({
                size: boidBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            })
        ];

        new Float32Array(this.boidBuffers[0].getMappedRange()).set(this.boids);
        this.boidBuffers[0].unmap();

        this.paramsBuffer = this.device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.currentBuffer = 0;
    }

    setupComputePipeline() {
        const shaderModule = this.device.createShaderModule({ code: COMPUTE_SHADER });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'main' }
        });

        this.computeBindGroups = [
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.boidBuffers[0] } },
                    { binding: 1, resource: { buffer: this.boidBuffers[1] } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.boidBuffers[1] } },
                    { binding: 1, resource: { buffer: this.boidBuffers[0] } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            })
        ];
    }

    setupRenderPipeline() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });

        const shaderModule = this.device.createShaderModule({ code: RENDER_SHADER });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        this.renderBindGroups = [
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.boidBuffers[0] } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.boidBuffers[1] } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            })
        ];
    }

    updateParams() {
        // Uniform layout follows WGSL alignment: vec2 fields start on 8-byte boundaries,
        // so index 9 is padding. u32 fields must be written through a Uint32Array view.
        const params = new Float32Array(32);
        const paramsU32 = new Uint32Array(params.buffer);
        paramsU32[0] = this.numBoids;
        params[1] = this.separationRadius;
        params[2] = this.alignmentRadius;
        params[3] = this.cohesionRadius;
        params[4] = this.separationWeight;
        params[5] = this.alignmentWeight;
        params[6] = this.cohesionWeight;
        params[7] = this.maxSpeed;
        params[8] = this.maxForce;
        const size = this.canvas ?? COMPUTE_ONLY_SIZE;
        params[10] = 0; // boundsMin.x
        params[11] = 0; // boundsMin.y
        params[12] = size.width; // boundsMax.x
        params[13] = size.height; // boundsMax.y
        params[14] = this.mousePos[0];
        params[15] = this.mousePos[1];
        params[16] = this.mouseRadius;
        params[17] = this.mouseForce;
        paramsU32[18] = this.behaviorMode;

        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    }

    updateUniforms() {
        const uniforms = new Float32Array([
            this.canvas.width,
            this.canvas.height
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    }

    step() {
        this.updateParams();

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroups[this.currentBuffer]);
        computePass.dispatchWorkgroups(Math.ceil(this.numBoids / 64));
        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        this.currentBuffer = 1 - this.currentBuffer;
        this.frameCount++;
    }

    render() {
        this.updateUniforms();

        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.04, g: 0.04, b: 0.1, a: 1.0 }
            }]
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBuffer]);
        renderPass.draw(3, this.numBoids);
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

    reset() {
        this.frameCount = 0;
        this.initializeBoids();
        this.device.queue.writeBuffer(this.boidBuffers[0], 0, this.boids);
        this.currentBuffer = 0;
        this.behaviorMode = 0;
        this.render();
    }

    animate() {
        if (!this.running) return;

        this.step();
        this.render();

        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.running = false;
        this.boidBuffers.forEach(b => b.destroy());
        this.paramsBuffer.destroy();
        this.uniformBuffer.destroy();
    }
}

export async function initBoids(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-boids');
    const output = document.getElementById('output-boids');

    let numBoids = 2000;
    let simulation = new BoidsSimulation(device, canvas, numBoids);
    simulation.render();

    document.getElementById('btn-boids-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-boids-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-boids-reset').addEventListener('click', () => {
        const wasRunning = simulation.running;
        simulation.destroy();
        simulation = new BoidsSimulation(device, canvas, numBoids);
        if (wasRunning) simulation.start(); else simulation.render();
        updateOutput();
    });

    const countSlider = document.getElementById('boids-count');
    const countValue = document.getElementById('boids-count-value');
    countSlider.addEventListener('change', (e) => {
        numBoids = parseInt(e.target.value);
        const wasRunning = simulation.running;
        simulation.destroy();
        simulation = new BoidsSimulation(device, canvas, numBoids);
        if (wasRunning) simulation.start(); else simulation.render();
        updateOutput();
    });
    countSlider.addEventListener('input', (e) => {
        countValue.textContent = parseInt(e.target.value).toLocaleString();
    });

    const speedSlider = document.getElementById('boids-speed');
    const speedValue = document.getElementById('boids-speed-value');
    speedSlider.addEventListener('input', (e) => {
        simulation.maxSpeed = parseInt(e.target.value) / 10;
        speedValue.textContent = e.target.value;
    });

    document.querySelectorAll('.boids-behavior').forEach(btn => {
        btn.addEventListener('click', () => {
            const behavior = btn.dataset.behavior;
            const modes = { 'flock': 0, 'predator': 1, 'scatter': 2, 'vortex': 3 };
            simulation.behaviorMode = modes[behavior];
            updateOutput();
        });
    });

    function updateOutput() {
        const behaviorNames = ['Normal Flocking', 'Predator/Prey', 'Scatter', 'Vortex'];

        output.innerHTML = `<span class="success">✓ Boids Simulation Active</span>

<span class="info">Configuration:</span>
• Number of boids: ${simulation.numBoids.toLocaleString()}
• Max speed: ${simulation.maxSpeed.toFixed(1)}
• Behavior: ${behaviorNames[simulation.behaviorMode]}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Algorithm (Craig Reynolds):</span>
• <strong>Separation</strong>: Avoid crowding neighbors
• <strong>Alignment</strong>: Steer towards average heading
• <strong>Cohesion</strong>: Steer towards average position
• Radius: ${simulation.separationRadius}/${simulation.alignmentRadius}/${simulation.cohesionRadius}

<span class="info">Performance:</span>
• Frame: ${simulation.frameCount}
• Neighbor checks: O(N²) = ${(simulation.numBoids * simulation.numBoids).toLocaleString()}
• All calculations fully parallelized on GPU
• Note: Spatial hashing would improve to O(N)

<span class="info">Rendering:</span>
• Triangle per boid, rotated to face direction
• Color: Speed-based (green=slow, yellow=fast)
• Predator mode: First boid rendered in red`;
    }

    updateOutput();
    return simulation;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: COMPUTE_SHADER },
        { name: 'render', code: RENDER_SHADER },
    ];
}

// Exported for the CPU tests in test/boids.test.mjs.
export { BoidsSimulation };

// ---- Compute-only API for the Measure tab (js/teach/experiments/boids.js) and test/gpu/ ----
//
// measureBoids() builds a BoidsSimulation without a canvas (same buffers, pipeline, ping-pong bind groups
// and default parameters, in a 1000 × 700 world), uploads caller-supplied boids and runs `steps` steps of
// the compute shader in passes of stepsPerSubmit(n) dispatches (at most 60, and at most 1e9 distance tests),
// one submit each, each timed by GPU timestamps and submit → done. With samples > 0 the boids are read back
// at evenly spaced steps.

export const MAX_BOIDS = 10000;          // the Boids slider's maximum
export const MAX_MEASURE_STEPS = 2000;   // per call
export const MAX_STEPS_PER_SUBMIT = 60;  // one second of the interactive example
export const MAX_TESTS_PER_SUBMIT = 1e9; // distance tests (3N² per step): 3 steps per submit at N = 10,000
export const MAX_SAMPLES = 50;
export const stepsPerSubmit = (n) => Math.max(1, Math.min(MAX_STEPS_PER_SUBMIT, Math.floor(MAX_TESTS_PER_SUBMIT / (3 * n * n))));
const COMPUTE_ONLY_SIZE = { width: 1000, height: 700 };
export const WORLD = COMPUTE_ONLY_SIZE;

// boids: n × 4 floats (position xy, velocity xy). behaviorMode 0 … 3 as in the example (0 = flocking).
// Resolves to { n, steps, params, gpuMs, wallMs, gpuMsPerStep, wallMsPerStep, snapshots: [{ step, boids }] }
// where params are the uniform values used (separationRadius … maxForce, bounds), times are sums over the
// chunks (GPU time null without timestamp-query) and snapshots hold the start plus `samples` read-backs.
// between(), if given, is awaited after each sample (ctx.step in the Measure runner, which throws to cancel).
export async function measureBoids(gpuDevice, { boids, steps = 10, samples = 0, behaviorMode = 0, between = null }) {
    const device = gpuDevice.device ?? gpuDevice;
    const n = boids ? boids.length / 4 : NaN;
    if (!Number.isInteger(n) || n < 1 || n > MAX_BOIDS) throw new RangeError(`N = ${n} is outside 1 … ${MAX_BOIDS}`);
    if (!Number.isInteger(steps) || steps < 1 || steps > MAX_MEASURE_STEPS) throw new RangeError(`steps = ${steps} is outside 1 … ${MAX_MEASURE_STEPS}`);
    if (![0, 1, 2, 3].includes(behaviorMode)) throw new RangeError(`behaviorMode = ${behaviorMode} is not 0 … 3`);
    const sim = new BoidsSimulation(device, null, n);
    const timer = new GpuTimer(gpuDevice, { maxPasses: 1, poolSize: 1 });
    try {
        sim.behaviorMode = behaviorMode;
        sim.currentBuffer = 0;
        device.queue.writeBuffer(sim.boidBuffers[0], 0, boids);
        sim.updateParams();
        const params = {
            separationRadius: sim.separationRadius, alignmentRadius: sim.alignmentRadius, cohesionRadius: sim.cohesionRadius,
            separationWeight: sim.separationWeight, alignmentWeight: sim.alignmentWeight, cohesionWeight: sim.cohesionWeight,
            maxSpeed: sim.maxSpeed, maxForce: sim.maxForce, boundsMin: [0, 0], boundsMax: [WORLD.width, WORLD.height],
            mousePos: sim.mousePos, mouseRadius: sim.mouseRadius, mouseForce: sim.mouseForce, behaviorMode,
        };
        const count = Math.min(Math.max(samples, 0), MAX_SAMPLES, steps);
        const read = () => readBuffer(device, sim.boidBuffers[sim.currentBuffer], 16 * n);
        const snapshots = count > 0 ? [{ step: 0, boids: Float32Array.from(boids) }] : [];
        const targets = count > 0 ? Array.from({ length: count }, (_, s) => Math.round((s + 1) * steps / count)) : [steps];
        let done = 0, gpuMs = 0, wallMs = 0, timedAll = true;
        for (const target of targets) {
            while (done < target) {
                const chunk = Math.min(stepsPerSubmit(n), target - done);
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass(timer.computePassDescriptor('boids'));
                pass.setPipeline(sim.computePipeline);
                for (let k = 0; k < chunk; k++) {
                    pass.setBindGroup(0, sim.computeBindGroups[sim.currentBuffer]);
                    pass.dispatchWorkgroups(Math.ceil(n / 64));
                    sim.currentBuffer = 1 - sim.currentBuffer;
                }
                pass.end();
                timer.resolve(encoder);
                const t0 = performance.now();
                device.queue.submit([encoder.finish()]);
                const timing = timer.read();
                await device.queue.onSubmittedWorkDone();
                wallMs += performance.now() - t0;
                const g = (await timing)?.total;
                if (typeof g === 'number') gpuMs += g; else timedAll = false;
                done += chunk;
            }
            if (count > 0) {
                snapshots.push({ step: done, boids: await read() });
                if (between) await between();
            }
        }
        const gpu = timedAll ? gpuMs : null;
        return { n, steps, params, gpuMs: gpu, wallMs, gpuMsPerStep: gpu === null ? null : gpu / steps, wallMsPerStep: wallMs / steps, snapshots };
    } finally {
        timer.destroy();
        sim.destroy();
    }
}
