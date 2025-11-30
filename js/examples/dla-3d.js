// 3D Diffusion Limited Aggregation
// Particles perform random walks in 3D space until they stick to the aggregate structure

const PARTICLE_UPDATE_SHADER = `
struct Particle {
    position: vec3<f32>,
    active: u32,
}

struct GridCell {
    occupied: atomic<u32>,
}

struct Params {
    numParticles: u32,
    gridSize: u32,
    stickiness: f32,
    time: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridCell>;
@group(0) @binding(2) var<uniform> params: Params;

// Simple random number generator (LCG)
fn random(seed: u32) -> f32 {
    let a = 1103515245u;
    let c = 12345u;
    let m = 2147483648u;
    return f32((a * seed + c) % m) / f32(m);
}

fn gridIndex(pos: vec3<i32>) -> u32 {
    let size = i32(params.gridSize);
    if (pos.x < 0 || pos.x >= size || pos.y < 0 || pos.y >= size || pos.z < 0 || pos.z >= size) {
        return 0xFFFFFFFFu;
    }
    return u32(pos.z * size * size + pos.y * size + pos.x);
}

fn isOccupied(pos: vec3<i32>) -> bool {
    let idx = gridIndex(pos);
    if (idx == 0xFFFFFFFFu) {
        return false;
    }
    return atomicLoad(&grid[idx].occupied) > 0u;
}

fn hasNeighbor(pos: vec3<i32>) -> bool {
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) { continue; }
                if (isOccupied(pos + vec3<i32>(dx, dy, dz))) {
                    return true;
                }
            }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numParticles) {
        return;
    }

    var particle = particles[idx];

    if (particle.active == 0u) {
        return;
    }

    // Generate random direction
    let seed = idx * 1000u + u32(params.time * 1000.0);
    let r1 = random(seed);
    let r2 = random(seed + 1u);
    let r3 = random(seed + 2u);

    var dir = vec3<i32>(0, 0, 0);
    if (r1 < 0.333) {
        dir.x = select(-1, 1, r2 > 0.5);
    } else if (r1 < 0.666) {
        dir.y = select(-1, 1, r2 > 0.5);
    } else {
        dir.z = select(-1, 1, r2 > 0.5);
    }

    let currentPos = vec3<i32>(particle.position);
    let newPos = currentPos + dir;

    let size = i32(params.gridSize);

    // Check if out of bounds - respawn at spawn radius
    if (newPos.x < 0 || newPos.x >= size || newPos.y < 0 || newPos.y >= size || newPos.z < 0 || newPos.z >= size) {
        // Respawn at outer radius
        let center = f32(size) / 2.0;
        let spawnRadius = f32(size) * 0.45;
        let theta = r1 * 6.28318;
        let phi = r2 * 3.14159;
        particle.position = vec3<f32>(
            center + spawnRadius * sin(phi) * cos(theta),
            center + spawnRadius * sin(phi) * sin(theta),
            center + spawnRadius * cos(phi)
        );
        particles[idx] = particle;
        return;
    }

    // Check if next to occupied cell
    if (hasNeighbor(newPos)) {
        // Stick with probability based on stickiness
        if (r3 < params.stickiness) {
            let gridIdx = gridIndex(newPos);
            if (gridIdx != 0xFFFFFFFFu) {
                atomicStore(&grid[gridIdx].occupied, 1u);
                particle.active = 0u;
            }
        }
    }

    particle.position = vec3<f32>(newPos);
    particles[idx] = particle;
}
`;

const RENDER_VERTEX_SHADER = `
struct Uniforms {
    viewProj: mat4x4<f32>,
}

struct GridCell {
    occupied: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> grid: array<GridCell>;
@group(0) @binding(2) var<uniform> gridSize: u32;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// Cube vertices
const CUBE_VERTICES = array<vec3<f32>, 36>(
    // Front
    vec3(-0.5, -0.5, 0.5), vec3(0.5, -0.5, 0.5), vec3(0.5, 0.5, 0.5),
    vec3(0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5), vec3(-0.5, -0.5, 0.5),
    // Back
    vec3(0.5, -0.5, -0.5), vec3(-0.5, -0.5, -0.5), vec3(-0.5, 0.5, -0.5),
    vec3(-0.5, 0.5, -0.5), vec3(0.5, 0.5, -0.5), vec3(0.5, -0.5, -0.5),
    // Top
    vec3(-0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, -0.5),
    vec3(0.5, 0.5, -0.5), vec3(-0.5, 0.5, -0.5), vec3(-0.5, 0.5, 0.5),
    // Bottom
    vec3(-0.5, -0.5, -0.5), vec3(0.5, -0.5, -0.5), vec3(0.5, -0.5, 0.5),
    vec3(0.5, -0.5, 0.5), vec3(-0.5, -0.5, 0.5), vec3(-0.5, -0.5, -0.5),
    // Right
    vec3(0.5, -0.5, 0.5), vec3(0.5, -0.5, -0.5), vec3(0.5, 0.5, -0.5),
    vec3(0.5, 0.5, -0.5), vec3(0.5, 0.5, 0.5), vec3(0.5, -0.5, 0.5),
    // Left
    vec3(-0.5, -0.5, -0.5), vec3(-0.5, -0.5, 0.5), vec3(-0.5, 0.5, 0.5),
    vec3(-0.5, 0.5, 0.5), vec3(-0.5, 0.5, -0.5), vec3(-0.5, -0.5, -0.5)
);

const CUBE_NORMALS = array<vec3<f32>, 6>(
    vec3(0.0, 0.0, 1.0),   // Front
    vec3(0.0, 0.0, -1.0),  // Back
    vec3(0.0, 1.0, 0.0),   // Top
    vec3(0.0, -1.0, 0.0),  // Bottom
    vec3(1.0, 0.0, 0.0),   // Right
    vec3(-1.0, 0.0, 0.0)   // Left
);

@vertex
fn main(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> VertexOutput {
    let size = gridSize;

    // Convert instance index to 3D grid position
    let z = instanceIdx / (size * size);
    let y = (instanceIdx / size) % size;
    let x = instanceIdx % size;

    // Check if this cell is occupied
    let occupied = grid[instanceIdx].occupied;

    var output: VertexOutput;

    if (occupied == 0u) {
        // Skip rendering
        output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return output;
    }

    let cubeVert = CUBE_VERTICES[vertexIdx];
    let normalIdx = vertexIdx / 6u;
    let normal = CUBE_NORMALS[normalIdx];

    let worldPos = vec3<f32>(f32(x), f32(y), f32(z)) + cubeVert;

    // Distance from center for coloring
    let center = vec3<f32>(f32(size) / 2.0);
    let dist = length(worldPos - center);
    let maxDist = f32(size) * 0.7;
    let t = clamp(dist / maxDist, 0.0, 1.0);

    // Color gradient from center to edge
    let color = mix(
        vec3<f32>(1.0, 0.9, 0.7),  // Gold center
        vec3<f32>(0.3, 0.6, 1.0),  // Blue edge
        t
    );

    output.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
    output.color = color;
    output.normal = normal;

    return output;
}
`;

const RENDER_FRAGMENT_SHADER = `
@fragment
fn main(
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>
) -> @location(0) vec4<f32> {
    let lightDir = normalize(vec3<f32>(1.0, 1.0, 2.0));
    let ambient = 0.3;
    let diffuse = max(dot(normalize(normal), lightDir), 0.0) * 0.7;
    let lighting = ambient + diffuse;

    return vec4<f32>(color * lighting, 1.0);
}
`;

// Matrix utilities (reusing from molecular-dynamics)
class Mat4 {
    static perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0
        ]);
    }

    static lookAt(eye, center, up) {
        const z = normalize(subtract(eye, center));
        const x = normalize(cross(up, z));
        const y = cross(z, x);

        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
        ]);
    }

    static multiply(a, b) {
        const result = new Float32Array(16);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                result[i * 4 + j] =
                    a[i * 4 + 0] * b[0 * 4 + j] +
                    a[i * 4 + 1] * b[1 * 4 + j] +
                    a[i * 4 + 2] * b[2 * 4 + j] +
                    a[i * 4 + 3] * b[3 * 4 + j];
            }
        }
        return result;
    }
}

function normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
}

function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

class Camera {
    constructor() {
        this.distance = 100;
        this.rotation = { x: 0.5, y: 0.5 };
        this.target = [0, 0, 0];
    }

    getViewMatrix() {
        const x = this.distance * Math.cos(this.rotation.y) * Math.cos(this.rotation.x);
        const y = this.distance * Math.sin(this.rotation.x);
        const z = this.distance * Math.sin(this.rotation.y) * Math.cos(this.rotation.x);

        const eye = [
            this.target[0] + x,
            this.target[1] + y,
            this.target[2] + z
        ];

        return Mat4.lookAt(eye, this.target, [0, 1, 0]);
    }
}

class DLA3D {
    constructor(device, canvas, gridSize, numParticles) {
        this.device = device;
        this.canvas = canvas;
        this.gridSize = gridSize;
        this.numParticles = numParticles;
        this.running = false;
        this.time = 0;
        this.particlesPerFrame = 10;
        this.stickiness = 0.5;
        this.aggregatedCount = 0;

        this.camera = new Camera();
        this.camera.target = [gridSize / 2, gridSize / 2, gridSize / 2];
        this.camera.distance = gridSize * 1.5;

        this.setupMouseControls();
        this.initializeParticles();
        this.initializeGrid();
        this.setupBuffers();
        this.setupComputePipeline();
        this.setupRenderPipeline();
    }

    setupMouseControls() {
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;

        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;

            this.camera.rotation.y += dx * 0.01;
            this.camera.rotation.x += dy * 0.01;
            this.camera.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.camera.rotation.x));

            lastX = e.clientX;
            lastY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera.distance *= (1 + e.deltaY * 0.001);
            this.camera.distance = Math.max(this.gridSize, Math.min(this.gridSize * 3, this.camera.distance));
        });
    }

    initializeParticles() {
        // position(3) + active(1)
        this.particles = new Float32Array(this.numParticles * 4);

        const center = this.gridSize / 2;
        const spawnRadius = this.gridSize * 0.45;

        for (let i = 0; i < this.numParticles; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            this.particles[i * 4 + 0] = center + spawnRadius * Math.sin(phi) * Math.cos(theta);
            this.particles[i * 4 + 1] = center + spawnRadius * Math.sin(phi) * Math.sin(theta);
            this.particles[i * 4 + 2] = center + spawnRadius * Math.cos(phi);
            this.particles[i * 4 + 3] = 1; // active
        }
    }

    initializeGrid(seedType = 'center') {
        const totalCells = this.gridSize * this.gridSize * this.gridSize;
        this.grid = new Uint32Array(totalCells);

        const center = Math.floor(this.gridSize / 2);

        switch (seedType) {
            case 'center':
                const idx = center * this.gridSize * this.gridSize + center * this.gridSize + center;
                this.grid[idx] = 1;
                this.aggregatedCount = 1;
                break;

            case 'line':
                for (let x = center - 10; x <= center + 10; x++) {
                    const idx = center * this.gridSize * this.gridSize + center * this.gridSize + x;
                    this.grid[idx] = 1;
                    this.aggregatedCount++;
                }
                break;

            case 'circle':
                const radius = 5;
                for (let y = -radius; y <= radius; y++) {
                    for (let x = -radius; x <= radius; x++) {
                        if (x * x + y * y <= radius * radius) {
                            const idx = center * this.gridSize * this.gridSize + (center + y) * this.gridSize + (center + x);
                            this.grid[idx] = 1;
                            this.aggregatedCount++;
                        }
                    }
                }
                break;

            case 'multiple':
                const offsets = [
                    [5, 5, 5], [-5, -5, -5], [5, -5, 5], [-5, 5, -5]
                ];
                for (const [dx, dy, dz] of offsets) {
                    const idx = (center + dz) * this.gridSize * this.gridSize + (center + dy) * this.gridSize + (center + dx);
                    this.grid[idx] = 1;
                    this.aggregatedCount++;
                }
                break;
        }
    }

    setupBuffers() {
        this.particleBuffer = this.device.createBuffer({
            size: this.particles.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.particleBuffer.getMappedRange()).set(this.particles);
        this.particleBuffer.unmap();

        this.gridBuffer = this.device.createBuffer({
            size: this.grid.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint32Array(this.gridBuffer.getMappedRange()).set(this.grid);
        this.gridBuffer.unmap();

        this.paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.gridSizeBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.gridSizeBuffer, 0, new Uint32Array([this.gridSize]));
    }

    setupComputePipeline() {
        const shaderModule = this.device.createShaderModule({ code: PARTICLE_UPDATE_SHADER });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'main' }
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.gridBuffer } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ]
        });
    }

    setupRenderPipeline() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });

        const shaderModule = this.device.createShaderModule({
            code: RENDER_VERTEX_SHADER + '\n' + RENDER_FRAGMENT_SHADER
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'main'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });

        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.gridBuffer } },
                { binding: 2, resource: { buffer: this.gridSizeBuffer } }
            ]
        });
    }

    updateParams() {
        const params = new Float32Array([
            this.numParticles,
            this.gridSize,
            this.stickiness,
            this.time
        ]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    }

    updateUniforms() {
        const aspect = this.canvas.width / this.canvas.height;
        const proj = Mat4.perspective(Math.PI / 4, aspect, 0.1, 500);
        const view = this.camera.getViewMatrix();
        const viewProj = Mat4.multiply(proj, view);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, viewProj);
    }

    step() {
        this.time += 0.016;
        this.updateParams();

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);

        // Run multiple times per frame
        for (let i = 0; i < this.particlesPerFrame; i++) {
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        }

        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    render() {
        this.updateUniforms();

        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1.0
            }
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);

        const totalInstances = this.gridSize * this.gridSize * this.gridSize;
        renderPass.draw(36, totalInstances);

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

    reset(seedType = 'center') {
        this.time = 0;
        this.initializeParticles();
        this.initializeGrid(seedType);
        this.device.queue.writeBuffer(this.particleBuffer, 0, this.particles);
        this.device.queue.writeBuffer(this.gridBuffer, 0, this.grid);
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
        this.particleBuffer.destroy();
        this.gridBuffer.destroy();
        this.paramsBuffer.destroy();
        this.uniformBuffer.destroy();
        this.gridSizeBuffer.destroy();
        this.depthTexture.destroy();
    }
}

export async function initDLA3D(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-dla');
    const output = document.getElementById('output-dla');

    const gridSize = 64;
    const numParticles = 5000;

    let simulation = new DLA3D(device, canvas, gridSize, numParticles);
    simulation.render();

    document.getElementById('btn-dla-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-dla-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-dla-reset').addEventListener('click', () => {
        simulation.reset();
        updateOutput();
    });

    const rateSlider = document.getElementById('dla-rate');
    const rateValue = document.getElementById('dla-rate-value');
    rateSlider.addEventListener('input', (e) => {
        simulation.particlesPerFrame = parseInt(e.target.value);
        rateValue.textContent = e.target.value;
    });

    const stickinessSlider = document.getElementById('dla-stickiness');
    const stickinessValue = document.getElementById('dla-stickiness-value');
    stickinessSlider.addEventListener('input', (e) => {
        simulation.stickiness = parseInt(e.target.value) / 100;
        stickinessValue.textContent = e.target.value + '%';
    });

    document.querySelectorAll('.dla-seed').forEach(btn => {
        btn.addEventListener('click', () => {
            const seed = btn.dataset.seed;
            simulation.reset(seed);
            updateOutput();
        });
    });

    function updateOutput() {
        output.innerHTML = `<span class="success">✓ 3D DLA Simulation Active</span>

<span class="info">Configuration:</span>
• Grid: ${gridSize}³ = ${(gridSize ** 3).toLocaleString()} cells
• Active particles: ${numParticles.toLocaleString()}
• Particles/frame: ${simulation.particlesPerFrame}
• Stickiness: ${(simulation.stickiness * 100).toFixed(0)}%

<span class="info">Algorithm:</span>
• Particles perform random walk in 3D space
• Stick when adjacent to aggregate structure
• Probability controlled by stickiness parameter
• Creates fractal-like dendritic structures

<span class="info">Rendering:</span>
• Instanced cube rendering for occupied cells
• Color gradient: gold (center) → blue (edge)
• Phong lighting for depth perception
• Drag to rotate, wheel to zoom

<span class="info">Growth Patterns:</span>
• <strong>Single Center</strong>: Classic radial growth
• <strong>Line</strong>: Growth along initial line seed
• <strong>Circle</strong>: Symmetric circular seed
• <strong>Multiple</strong>: Competing growth centers

<span class="info">Status:</span>
• ${simulation.running ? '<span class="success">Growing...</span>' : '<span class="info">Paused</span>'}
• Watch the structure evolve in real-time!`;
    }

    updateOutput();
    return simulation;
}
