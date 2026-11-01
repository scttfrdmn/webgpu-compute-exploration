// SPH (Smoothed Particle Hydrodynamics) Fluid Simulation
// Based on "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.

import { GpuTimer } from '../gpu-timer.js';
import { readBuffer } from '../webgpu-utils.js';

const COMPUTE_DENSITY_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    pad: vec2<f32>, // keeps the stride at 32 bytes to match the JS packing (8 floats)
}

struct Params {
    numParticles: u32,
    smoothingRadius: f32,
    restDensity: f32,
    gasConstant: f32,
    mass: f32,
    viscosity: f32,
    gravity: f32,
    deltaTime: f32,
}

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> params: Params;

// Poly6 kernel for density (2D normalization)
fn poly6Kernel(r: f32, h: f32) -> f32 {
    if (r >= h) {
        return 0.0;
    }
    let coef = 4.0 / (3.14159265 * pow(h, 8.0));
    let diff = h * h - r * r;
    return coef * diff * diff * diff;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numParticles) {
        return;
    }

    let particle = particlesIn[idx];
    var density = 0.0;

    // Sum density contributions from all neighbors
    for (var j = 0u; j < params.numParticles; j = j + 1u) {
        let other = particlesIn[j];
        let diff = particle.position - other.position;
        let dist = length(diff);

        if (dist < params.smoothingRadius) {
            density += params.mass * poly6Kernel(dist, params.smoothingRadius);
        }
    }

    var output = particle;
    output.density = density;
    // Clamp to non-negative so sparse regions don't pull particles together into clumps
    output.pressure = max(params.gasConstant * (density - params.restDensity), 0.0);

    particlesOut[idx] = output;
}
`;

const COMPUTE_FORCES_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    pad: vec2<f32>, // keeps the stride at 32 bytes to match the JS packing (8 floats)
}

struct Params {
    numParticles: u32,
    smoothingRadius: f32,
    restDensity: f32,
    gasConstant: f32,
    mass: f32,
    viscosity: f32,
    gravity: f32,
    deltaTime: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Spiky kernel gradient for pressure (2D normalization)
fn spikyKernelGradient(r: vec2<f32>, h: f32) -> vec2<f32> {
    let dist = length(r);
    if (dist >= h || dist < 0.0001) {
        return vec2<f32>(0.0, 0.0);
    }
    let coef = -30.0 / (3.14159265 * pow(h, 5.0));
    let diff = h - dist;
    return coef * diff * diff * normalize(r);
}

// Viscosity Laplacian kernel (2D normalization)
fn viscosityLaplacian(r: f32, h: f32) -> f32 {
    if (r >= h) {
        return 0.0;
    }
    let coef = 40.0 / (3.14159265 * pow(h, 5.0));
    return coef * (h - r);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numParticles) {
        return;
    }

    let particle = particles[idx];
    var pressureForce = vec2<f32>(0.0, 0.0);
    var viscosityForce = vec2<f32>(0.0, 0.0);

    for (var j = 0u; j < params.numParticles; j = j + 1u) {
        if (j == idx) {
            continue;
        }

        let other = particles[j];
        let diff = particle.position - other.position;
        let dist = length(diff);

        if (dist < params.smoothingRadius && dist > 0.0001) {
            // Pressure force
            let pressureGrad = spikyKernelGradient(diff, params.smoothingRadius);
            let avgPressure = (particle.pressure + other.pressure) / 2.0;
            pressureForce -= params.mass * avgPressure / other.density * pressureGrad;

            // Viscosity force
            let velDiff = other.velocity - particle.velocity;
            let viscLap = viscosityLaplacian(dist, params.smoothingRadius);
            viscosityForce += params.mass * velDiff / other.density * viscLap;
        }
    }

    viscosityForce *= params.viscosity;

    // Gravity as a force density, so that force / density in the integrator yields acceleration g
    let gravityForce = vec2<f32>(0.0, params.gravity) * particle.density;

    forces[idx] = pressureForce + viscosityForce + gravityForce;
}
`;

const INTEGRATE_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    pad: vec2<f32>, // keeps the stride at 32 bytes to match the JS packing (8 floats)
}

struct Params {
    numParticles: u32,
    smoothingRadius: f32,
    restDensity: f32,
    gasConstant: f32,
    mass: f32,
    viscosity: f32,
    gravity: f32,
    deltaTime: f32,
    boundsMin: vec2<f32>,
    boundsMax: vec2<f32>,
    damping: f32,
}

@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read> forces: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numParticles) {
        return;
    }

    var particle = particlesIn[idx];
    let force = forces[idx];

    // Update velocity and position
    particle.velocity += force / particle.density * params.deltaTime;
    particle.position += particle.velocity * params.deltaTime;

    // Boundary conditions with damping
    if (particle.position.x < params.boundsMin.x) {
        particle.position.x = params.boundsMin.x;
        particle.velocity.x *= -params.damping;
    }
    if (particle.position.x > params.boundsMax.x) {
        particle.position.x = params.boundsMax.x;
        particle.velocity.x *= -params.damping;
    }
    if (particle.position.y < params.boundsMin.y) {
        particle.position.y = params.boundsMin.y;
        particle.velocity.y *= -params.damping;
    }
    if (particle.position.y > params.boundsMax.y) {
        particle.position.y = params.boundsMax.y;
        particle.velocity.y *= -params.damping;
    }

    particlesOut[idx] = particle;
}
`;

const RENDER_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    pad: vec2<f32>, // keeps the stride at 32 bytes to match the JS packing (8 floats)
}

struct Uniforms {
    canvasSize: vec2<f32>,
    boundsMin: vec2<f32>,
    boundsMax: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
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
    let particle = particles[instanceIdx];

    // Create a quad for each particle
    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );

    let vert = vertices[vertexIdx];
    let particleSize = 4.0; // pixels

    // Transform from world space to clip space
    let worldRange = uniforms.boundsMax - uniforms.boundsMin;
    let normalized = (particle.position - uniforms.boundsMin) / worldRange;
    var clipPos = normalized * 2.0 - 1.0;
    clipPos.y *= -1.0; // Flip Y for screen coordinates

    let pixelSize = vec2<f32>(2.0, 2.0) / uniforms.canvasSize * particleSize;
    let finalPos = clipPos + vert * pixelSize;

    // Color based on velocity
    let speed = length(particle.velocity);
    let t = clamp(speed / 200.0, 0.0, 1.0);
    let color = mix(
        vec3<f32>(0.2, 0.5, 1.0),
        vec3<f32>(0.0, 0.9, 1.0),
        t
    );

    var output: VertexOutput;
    output.position = vec4<f32>(finalPos, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fragmentMain(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 0.8);
}
`;

class SPHFluidSimulation {
    constructor(device, canvas, numParticles) {
        this.device = device;
        // 3 compute passes per substep (6 substeps) + the draw
        this.gpuTimer = new GpuTimer(device, { maxPasses: 24 });
        this.fps = 0;
        this.canvas = canvas;
        this.initialParticles = numParticles;
        this.numParticles = numParticles;
        this.maxParticles = numParticles * 2; // headroom for Spawn
        this.running = false;
        this.frameCount = 0;

        // SPH parameters, in pixel units. Mass is chosen so the initial lattice sits at rest
        // density 1, and the time step is split into substeps for stability at gasConstant 1e6.
        this.smoothingRadius = 16.0;
        this.particleSpacing = this.smoothingRadius * 0.5;
        this.restDensity = 1.0;
        this.mass = 1.0 / this.latticeDensity();
        this.gasConstant = 1e6;
        this.viscosity = 300.0;
        this.gravity = 500.0;
        this.damping = 0.3;
        this.substeps = 6;
        this.deltaTime = 1 / 60 / this.substeps;

        // canvas = null: compute-only (Measure tab, test/gpu/), sized like index.html's 1000 × 700 canvas
        const size = canvas ?? COMPUTE_ONLY_SIZE;
        this.boundsMin = [10, 10];
        this.boundsMax = [size.width - 10, size.height - 10];

        this.initializeParticles();
        this.setupBuffers();
        this.setupComputePipelines();
        if (canvas) this.setupRenderPipeline();
    }

    // Density at an interior lattice point for unit mass, using the 2D Poly6 kernel
    latticeDensity() {
        const h = this.smoothingRadius, s = this.particleSpacing;
        const coef = 4 / (Math.PI * Math.pow(h, 8));
        let density = 0;
        for (let i = -3; i <= 3; i++) {
            for (let j = -3; j <= 3; j++) {
                const r2 = (i * s) ** 2 + (j * s) ** 2;
                if (r2 < h * h) density += coef * Math.pow(h * h - r2, 3);
            }
        }
        return density;
    }

    initializeParticles() {
        // 8 floats per particle: pos(2) + vel(2) + density(1) + pressure(1) + padding(2)
        this.particles = new Float32Array(this.numParticles * 8);

        // Initialize particles in a dam-break configuration
        const particlesPerRow = Math.ceil(Math.sqrt(this.numParticles));
        const spacing = this.particleSpacing;
        const startX = this.boundsMin[0] + 50;
        const startY = this.boundsMin[1] + 50;

        for (let i = 0; i < this.numParticles; i++) {
            const row = Math.floor(i / particlesPerRow);
            const col = i % particlesPerRow;

            this.particles[i * 8 + 0] = startX + col * spacing; // position.x
            this.particles[i * 8 + 1] = startY + row * spacing; // position.y
            this.particles[i * 8 + 2] = 0; // velocity.x
            this.particles[i * 8 + 3] = 0; // velocity.y
            this.particles[i * 8 + 4] = this.restDensity; // density
            this.particles[i * 8 + 5] = 0; // pressure
        }
    }

    setupBuffers() {
        const particleBufferSize = this.maxParticles * 8 * 4;

        // COPY_SRC: the Measure tab reads particles (positions and densities) back
        this.particleBuffers = [
            this.device.createBuffer({
                size: particleBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            }),
            this.device.createBuffer({
                size: particleBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            })
        ];

        new Float32Array(this.particleBuffers[0].getMappedRange(), 0, this.particles.length).set(this.particles);
        this.particleBuffers[0].unmap();

        this.forcesBuffer = this.device.createBuffer({
            size: this.maxParticles * 2 * 4,
            usage: GPUBufferUsage.STORAGE
        });

        this.paramsBuffer = this.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

    }

    setupComputePipelines() {
        // Density computation pipeline
        const densityModule = this.device.createShaderModule({ code: COMPUTE_DENSITY_SHADER });
        const densityLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.densityPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [densityLayout] }),
            compute: { module: densityModule, entryPoint: 'main' }
        });

        this.densityBindGroups = [
            this.device.createBindGroup({
                layout: densityLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffers[0] } },
                    { binding: 1, resource: { buffer: this.particleBuffers[1] } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: densityLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffers[1] } },
                    { binding: 1, resource: { buffer: this.particleBuffers[0] } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            })
        ];

        // Force computation pipeline
        const forceModule = this.device.createShaderModule({ code: COMPUTE_FORCES_SHADER });
        const forceLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.forcePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [forceLayout] }),
            compute: { module: forceModule, entryPoint: 'main' }
        });

        this.forceBindGroups = [
            this.device.createBindGroup({
                layout: forceLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffers[1] } },
                    { binding: 1, resource: { buffer: this.forcesBuffer } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            })
        ];

        // Integration pipeline
        const integrateModule = this.device.createShaderModule({ code: INTEGRATE_SHADER });
        const integrateLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.integratePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [integrateLayout] }),
            compute: { module: integrateModule, entryPoint: 'main' }
        });

        this.integrateBindGroups = [
            this.device.createBindGroup({
                layout: integrateLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffers[1] } },
                    { binding: 1, resource: { buffer: this.forcesBuffer } },
                    { binding: 2, resource: { buffer: this.particleBuffers[0] } },
                    { binding: 3, resource: { buffer: this.paramsBuffer } }
                ]
            })
        ];
    }

    setupRenderPipeline() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm',
            alphaMode: 'premultiplied'
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
                targets: [{
                    format: 'bgra8unorm',
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        this.renderBindGroups = [
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffers[0] } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.particleBuffers[1] } },
                    { binding: 1, resource: { buffer: this.uniformBuffer } }
                ]
            })
        ];
    }

    updateParams() {
        const params = new Float32Array(32);
        new Uint32Array(params.buffer)[0] = this.numParticles;
        params[1] = this.smoothingRadius;
        params[2] = this.restDensity;
        params[3] = this.gasConstant;
        params[4] = this.mass;
        params[5] = this.viscosity;
        params[6] = this.gravity;
        params[7] = this.deltaTime;
        params[8] = this.boundsMin[0];
        params[9] = this.boundsMin[1];
        params[10] = this.boundsMax[0];
        params[11] = this.boundsMax[1];
        params[12] = this.damping;

        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    }

    updateUniforms() {
        const uniforms = new Float32Array([
            this.canvas.width,
            this.canvas.height,
            this.boundsMin[0],
            this.boundsMin[1],
            this.boundsMax[0],
            this.boundsMax[1]
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    }

    step() {
        this.updateParams();

        // Each substep: density/pressure (buffer 0 -> 1), forces (read 1), integrate (1 -> 0).
        // The simulation state therefore always lives in buffer 0 between substeps.
        const commandEncoder = this.device.createCommandEncoder();
        const workgroups = Math.ceil(this.numParticles / 64);
        for (let i = 0; i < this.substeps; i++) {
            const densityPass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('density'));
            densityPass.setPipeline(this.densityPipeline);
            densityPass.setBindGroup(0, this.densityBindGroups[0]);
            densityPass.dispatchWorkgroups(workgroups);
            densityPass.end();

            const forcePass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('forces'));
            forcePass.setPipeline(this.forcePipeline);
            forcePass.setBindGroup(0, this.forceBindGroups[0]);
            forcePass.dispatchWorkgroups(workgroups);
            forcePass.end();

            const integratePass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('integrate'));
            integratePass.setPipeline(this.integratePipeline);
            integratePass.setBindGroup(0, this.integrateBindGroups[0]);
            integratePass.dispatchWorkgroups(workgroups);
            integratePass.end();
        }

        this.device.queue.submit([commandEncoder.finish()]);

        this.frameCount++;
    }

    // timed: add timestamp writes and resolve the frame's timer (animate() only, after step())
    render(timed = false) {
        this.updateUniforms();

        const commandEncoder = this.device.createCommandEncoder();
        const desc = {
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.04, g: 0.04, b: 0.1, a: 1.0 }
            }]
        };
        const renderPass = commandEncoder.beginRenderPass(timed ? this.gpuTimer.renderPassDescriptor(desc, 'render') : desc);

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[0]);
        renderPass.draw(6, this.numParticles);
        renderPass.end();
        if (timed) this.gpuTimer.resolve(commandEncoder);

        this.device.queue.submit([commandEncoder.finish()]);
        if (timed) this.gpuTimer.read();
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

    reset() {
        this.frameCount = 0;
        this.numParticles = this.initialParticles;
        this.initializeParticles();
        this.device.queue.writeBuffer(this.particleBuffers[0], 0, this.particles);
        this.render();
    }

    // Drop a 10x10 block of particles centered at (x, y) into the free buffer slots
    spawnParticles(x, y) {
        const side = 10;
        const count = Math.min(side * side, this.maxParticles - this.numParticles);
        if (count <= 0) return;

        const block = new Float32Array(count * 8);
        const s = this.particleSpacing;
        for (let i = 0; i < count; i++) {
            block[i * 8 + 0] = x + ((i % side) - side / 2) * s;
            block[i * 8 + 1] = y + (Math.floor(i / side) - side / 2) * s;
            block[i * 8 + 4] = this.restDensity;
        }
        this.device.queue.writeBuffer(this.particleBuffers[0], this.numParticles * 8 * 4, block);
        this.numParticles += count;
        if (!this.running) this.render();
    }

    animate() {
        if (!this.running) return;

        // step() and render() are separate submits; the timer resolves in the second
        this.step();
        this.render(true);

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
        this.gpuTimer.destroy();
        this.particleBuffers.forEach(b => b.destroy());
        this.forcesBuffer.destroy();
        this.paramsBuffer.destroy();
        this.uniformBuffer.destroy();
    }
}

export async function initSPHFluid(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-sph');
    const output = document.getElementById('output-sph');

    let simulation = new SPHFluidSimulation(device, canvas, 2000);
    simulation.render();
    simulation.onStats = () => updateOutput();

    document.getElementById('btn-sph-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-sph-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-sph-reset').addEventListener('click', () => {
        simulation.reset();
        updateOutput();
    });

    document.getElementById('btn-sph-spawn').addEventListener('click', () => {
        simulation.spawnParticles(canvas.width / 2, 100);
        updateOutput();
    });

    const gravitySlider = document.getElementById('sph-gravity');
    const gravityValue = document.getElementById('sph-gravity-value');
    gravitySlider.addEventListener('input', (e) => {
        simulation.gravity = parseInt(e.target.value) * 10;
        gravityValue.textContent = e.target.value;
    });

    const viscositySlider = document.getElementById('sph-viscosity');
    const viscosityValue = document.getElementById('sph-viscosity-value');
    viscositySlider.addEventListener('input', (e) => {
        simulation.viscosity = parseInt(e.target.value) * 10;
        viscosityValue.textContent = e.target.value;
    });

    function updateOutput() {
        output.innerHTML = `<span class="success">✓ SPH Fluid Simulation Active</span>

<span class="info">Configuration:</span>
• Particles: ${simulation.numParticles.toLocaleString()}
• Smoothing radius: ${simulation.smoothingRadius.toFixed(1)}
• Rest density: ${simulation.restDensity}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Physics:</span>
• Algorithm: Smoothed Particle Hydrodynamics (SPH)
• Kernels: Poly6 (density), Spiky (pressure), Viscosity
• Gravity: ${(simulation.gravity / 10).toFixed(0)}
• Viscosity: ${simulation.viscosity.toFixed(0)}
• Integration: Explicit Euler

<span class="info">Performance:</span>
• Frame: ${simulation.frameCount}${simulation.running ? ` · ${simulation.fps.toFixed(0)} FPS` : ''}
• ${simulation.gpuTimer.frameText(simulation.running ? simulation.fps : 0, ['density', 'forces', 'integrate', 'render'])}
• Compute passes: 3 per frame (density, forces, integrate)
• Neighbor search: O(N²) brute force
• Note: Spatial hashing would improve performance significantly

<span class="info">Rendering:</span>
• Instanced particle quads
• Color: Velocity-based (blue=slow, cyan=fast)
• Alpha blending for fluid appearance`;
    }

    updateOutput();
    return simulation;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'density', code: COMPUTE_DENSITY_SHADER },
        { name: 'forces', code: COMPUTE_FORCES_SHADER },
        { name: 'integrate', code: INTEGRATE_SHADER },
        { name: 'render', code: RENDER_SHADER },
    ];
}

// Exported for the CPU tests in test/sph-fluid.test.mjs.
export { SPHFluidSimulation };

// ---- Compute-only API for the Measure tab (js/teach/experiments/sph-fluid.js) and test/gpu/ ----
//
// Both functions build an SPHFluidSimulation without a canvas (the same buffers, pipelines and bind
// groups, the same dam-break start) and encode frames exactly as step() does: per substep a density,
// a forces and an integrate pass. measureSPHFrame() times one frame per pass label; runSPH() runs whole
// frames (one submit each) and reads back particle buffer 1 at evenly spaced frames. Buffer 1 holds the
// output of the last density pass, so its positions and densities belong together (buffer 0 has the
// positions after the integrate pass and the densities from before it).

export const MAX_SPH_PARTICLES = 4000;   // the example's maximum (2,000 plus Add Water up to twice that)
export const MAX_SPH_FRAMES = 300;       // 5 simulated seconds at 60 frames, 6 substeps each
export const MAX_SAMPLES = 60;
export const MAX_MEASURE_REPEATS = 10;

const COMPUTE_ONLY_SIZE = { width: 1000, height: 700 };

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((p, q) => p - q);
    return v.length ? v[v.length >> 1] : null;
};

function computeOnlySPH(device, n, gasConstant) {
    if (!Number.isInteger(n) || n < 1 || n > MAX_SPH_PARTICLES) throw new RangeError(`N = ${n} is outside 1 … ${MAX_SPH_PARTICLES}`);
    const sim = new SPHFluidSimulation(device, null, n);
    if (gasConstant !== undefined) {
        if (!(gasConstant > 0)) throw new RangeError('the gas constant must be positive');
        sim.gasConstant = gasConstant;
    }
    return sim;
}

// One frame (sim.substeps substeps) on `encoder`, as step() encodes it; timer may be null
function encodeFrame(sim, encoder, timer) {
    const workgroups = Math.ceil(sim.numParticles / 64);
    for (let i = 0; i < sim.substeps; i++) {
        for (const [label, pipeline, bindGroup] of [
            ['density', sim.densityPipeline, sim.densityBindGroups[0]],
            ['forces', sim.forcePipeline, sim.forceBindGroups[0]],
            ['integrate', sim.integratePipeline, sim.integrateBindGroups[0]],
        ]) {
            const pass = encoder.beginComputePass(timer ? timer.computePassDescriptor(label) : undefined);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        }
    }
}

// The dam break with n particles: `warmup` untimed and `repeats` timed frames from the initial state.
// Resolves to { n, substeps, gpuMs, passes: { density, forces, integrate }, wallMs } per frame (medians;
// GPU times null without timestamp-query).
export async function measureSPHFrame(gpuDevice, { n, warmup = 1, repeats = 3 }) {
    const device = gpuDevice.device ?? gpuDevice;
    const sim = computeOnlySPH(device, n);
    const timer = new GpuTimer(gpuDevice, { maxPasses: 3 * sim.substeps, poolSize: 1 });
    try {
        sim.updateParams();
        const timedRuns = Math.min(Math.max(repeats, 1), MAX_MEASURE_REPEATS);
        const total = Math.min(Math.max(warmup, 0), MAX_MEASURE_REPEATS) + timedRuns;
        const runs = [];
        for (let r = 0; r < total; r++) {
            const timed = r >= total - timedRuns;
            const encoder = device.createCommandEncoder();
            encodeFrame(sim, encoder, timed ? timer : null);
            if (timed) timer.resolve(encoder);
            const t0 = performance.now();
            device.queue.submit([encoder.finish()]);
            const timing = timed ? timer.read() : null;
            await device.queue.onSubmittedWorkDone();
            if (timed) runs.push({ wallMs: performance.now() - t0, result: await timing });
        }
        const pass = (label) => median(runs.map(r => r.result?.passes?.[label] ?? null));
        return {
            n, substeps: sim.substeps,
            gpuMs: median(runs.map(r => r.result?.total ?? null)),
            passes: { density: pass('density'), forces: pass('forces'), integrate: pass('integrate') },
            wallMs: median(runs.map(r => r.wallMs)),
        };
    } finally {
        timer.destroy();
        sim.destroy();
    }
}

// `frames` frames of the dam break with n particles and gas constant k, read back `samples` times at even
// intervals (plus the start). between(), if given, is awaited after each sample (ctx.step in the Measure
// runner, which throws to cancel). Resolves to { n, frames, substeps, dt, smoothingRadius, mass, restDensity,
// gasConstant, wallMs, snapshots: [{ frame, time, particles }] } with particles n × 8 floats (position,
// velocity, density, pressure, pad) from buffer 1, i.e. the state the last density pass saw.
export async function runSPH(gpuDevice, { n, gasConstant, frames = 60, samples = 10, between = null }) {
    const device = gpuDevice.device ?? gpuDevice;
    if (!Number.isInteger(frames) || frames < 1 || frames > MAX_SPH_FRAMES) throw new RangeError(`frames = ${frames} is outside 1 … ${MAX_SPH_FRAMES}`);
    const sim = computeOnlySPH(device, n, gasConstant);
    try {
        sim.updateParams();
        const count = Math.min(Math.max(samples, 1), MAX_SAMPLES, frames);
        const read = () => readBuffer(device, sim.particleBuffers[1], 32 * n);
        const snapshots = [{ frame: 0, time: 0, particles: Float32Array.from(sim.particles) }];
        const t0 = performance.now();
        let done = 0;
        for (let s = 1; s <= count; s++) {
            const target = Math.round(s * frames / count);
            for (; done < target; done++) {
                const encoder = device.createCommandEncoder();
                encodeFrame(sim, encoder, null);
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
            }
            snapshots.push({ frame: done, time: done / 60, particles: await read() });
            if (between) await between();
        }
        return {
            n, frames, substeps: sim.substeps, dt: sim.deltaTime, smoothingRadius: sim.smoothingRadius, mass: sim.mass,
            restDensity: sim.restDensity, gasConstant: sim.gasConstant, wallMs: performance.now() - t0, snapshots,
        };
    } finally {
        sim.destroy();
    }
}
