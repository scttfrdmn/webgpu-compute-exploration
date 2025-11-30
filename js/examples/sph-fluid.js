// SPH (Smoothed Particle Hydrodynamics) Fluid Simulation
// Based on "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.

const COMPUTE_DENSITY_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
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

// Poly6 kernel for density
fn poly6Kernel(r: f32, h: f32) -> f32 {
    if (r >= h) {
        return 0.0;
    }
    let coef = 315.0 / (64.0 * 3.14159265 * pow(h, 9.0));
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
    output.pressure = params.gasConstant * (density - params.restDensity);

    particlesOut[idx] = output;
}
`;

const COMPUTE_FORCES_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
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

// Spiky kernel gradient for pressure
fn spikyKernelGradient(r: vec2<f32>, h: f32) -> vec2<f32> {
    let dist = length(r);
    if (dist >= h || dist < 0.0001) {
        return vec2<f32>(0.0, 0.0);
    }
    let coef = -45.0 / (3.14159265 * pow(h, 6.0));
    let diff = h - dist;
    return coef * diff * diff * normalize(r);
}

// Viscosity Laplacian kernel
fn viscosityLaplacian(r: f32, h: f32) -> f32 {
    if (r >= h) {
        return 0.0;
    }
    let coef = 45.0 / (3.14159265 * pow(h, 6.0));
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

    // Gravity
    let gravityForce = vec2<f32>(0.0, params.gravity);

    forces[idx] = pressureForce + viscosityForce + gravityForce;
}
`;

const INTEGRATE_SHADER = `
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
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
    let clipPos = normalized * 2.0 - 1.0;
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
        this.canvas = canvas;
        this.numParticles = numParticles;
        this.running = false;
        this.frameCount = 0;

        // SPH parameters
        this.smoothingRadius = 20.0;
        this.restDensity = 1000.0;
        this.gasConstant = 2000.0;
        this.mass = 1.0;
        this.viscosity = 0.3;
        this.gravity = 300.0;
        this.damping = 0.5;

        this.boundsMin = [10, 10];
        this.boundsMax = [canvas.width - 10, canvas.height - 10];

        this.initializeParticles();
        this.setupBuffers();
        this.setupComputePipelines();
        this.setupRenderPipeline();
    }

    initializeParticles() {
        // 8 floats per particle: pos(2) + vel(2) + density(1) + pressure(1) + padding(2)
        this.particles = new Float32Array(this.numParticles * 8);

        // Initialize particles in a dam-break configuration
        const particlesPerRow = Math.ceil(Math.sqrt(this.numParticles));
        const spacing = this.smoothingRadius * 0.5;
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
        const particleBufferSize = this.numParticles * 8 * 4;

        this.particleBuffers = [
            this.device.createBuffer({
                size: particleBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            }),
            this.device.createBuffer({
                size: particleBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            })
        ];

        new Float32Array(this.particleBuffers[0].getMappedRange()).set(this.particles);
        this.particleBuffers[0].unmap();

        this.forcesBuffer = this.device.createBuffer({
            size: this.numParticles * 2 * 4,
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

        this.currentBuffer = 0;
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
        params[0] = this.numParticles;
        params[1] = this.smoothingRadius;
        params[2] = this.restDensity;
        params[3] = this.gasConstant;
        params[4] = this.mass;
        params[5] = this.viscosity;
        params[6] = this.gravity;
        params[7] = 0.0016; // deltaTime
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

        const commandEncoder = this.device.createCommandEncoder();

        // Compute density and pressure
        const densityPass = commandEncoder.beginComputePass();
        densityPass.setPipeline(this.densityPipeline);
        densityPass.setBindGroup(0, this.densityBindGroups[this.currentBuffer]);
        densityPass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        densityPass.end();

        this.currentBuffer = 1 - this.currentBuffer;

        // Compute forces
        const forcePass = commandEncoder.beginComputePass();
        forcePass.setPipeline(this.forcePipeline);
        forcePass.setBindGroup(0, this.forceBindGroups[0]);
        forcePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        forcePass.end();

        // Integrate
        const integratePass = commandEncoder.beginComputePass();
        integratePass.setPipeline(this.integratePipeline);
        integratePass.setBindGroup(0, this.integrateBindGroups[0]);
        integratePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        integratePass.end();

        this.device.queue.submit([commandEncoder.finish()]);

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
        renderPass.draw(6, this.numParticles);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        this.running = true;
        this.animate();
    }

    pause() {
        this.running = false;
    }

    reset() {
        this.frameCount = 0;
        this.initializeParticles();
        this.device.queue.writeBuffer(this.particleBuffers[0], 0, this.particles);
        this.currentBuffer = 0;
        this.render();
    }

    spawnParticles(x, y) {
        // Add particles at mouse position
        const newParticles = 50;
        const oldCount = this.numParticles;
        this.numParticles = Math.min(this.numParticles + newParticles, 10000);

        if (this.numParticles === oldCount) return;

        // Would need to resize buffers - for now just reset
        console.log('Spawn not fully implemented - would need buffer resize');
    }

    animate() {
        if (!this.running) return;

        this.step();
        this.render();

        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.running = false;
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
        simulation.viscosity = parseInt(e.target.value) / 100;
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
• Viscosity: ${simulation.viscosity.toFixed(2)}
• Integration: Explicit Euler

<span class="info">Performance:</span>
• Frame: ${simulation.frameCount}
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
