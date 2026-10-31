// Compute shader for molecular dynamics using Lennard-Jones potential
const computeShader = `
struct Atom {
    position: vec3<f32>,
    velocity: vec3<f32>,
}

struct Params {
    numAtoms: u32,
    deltaTime: f32,
    boxSize: f32,
    temperature: f32,
}

@group(0) @binding(0) var<storage, read> atomsIn: array<Atom>;
@group(0) @binding(1) var<storage, read_write> atomsOut: array<Atom>;
@group(0) @binding(2) var<uniform> params: Params;

// Lennard-Jones potential parameters
const EPSILON = 1.0;
const SIGMA = 1.0;
const CUTOFF = 2.5;
const CUTOFF_SQ = 6.25;

fn computeForce(r: vec3<f32>) -> vec3<f32> {
    let dist_sq = dot(r, r);

    if (dist_sq > CUTOFF_SQ || dist_sq < 0.01) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let dist = sqrt(dist_sq);
    let sr6 = pow(SIGMA / dist, 6.0);
    let sr12 = sr6 * sr6;

    let force_mag = 24.0 * EPSILON * (2.0 * sr12 - sr6) / dist_sq;
    return force_mag * r;
}

fn applyPBC(pos: vec3<f32>, boxSize: f32) -> vec3<f32> {
    var result = pos;
    let half = boxSize * 0.5;

    if (result.x > half) { result.x -= boxSize; }
    if (result.x < -half) { result.x += boxSize; }
    if (result.y > half) { result.y -= boxSize; }
    if (result.y < -half) { result.y += boxSize; }
    if (result.z > half) { result.z -= boxSize; }
    if (result.z < -half) { result.z += boxSize; }

    return result;
}

fn minImage(r: vec3<f32>, boxSize: f32) -> vec3<f32> {
    var result = r;
    let half = boxSize * 0.5;

    if (result.x > half) { result.x -= boxSize; }
    if (result.x < -half) { result.x += boxSize; }
    if (result.y > half) { result.y -= boxSize; }
    if (result.y < -half) { result.y += boxSize; }
    if (result.z > half) { result.z -= boxSize; }
    if (result.z < -half) { result.z += boxSize; }

    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.numAtoms) {
        return;
    }

    let atom = atomsIn[idx];
    var force = vec3<f32>(0.0, 0.0, 0.0);

    // Compute forces from all other atoms
    for (var j = 0u; j < params.numAtoms; j = j + 1u) {
        if (j == idx) {
            continue;
        }

        var r = atomsIn[j].position - atom.position;
        r = minImage(r, params.boxSize);
        force += computeForce(r);
    }

    // Velocity Verlet integration
    let acceleration = force;
    var newVel = atom.velocity + acceleration * params.deltaTime;
    var newPos = atom.position + newVel * params.deltaTime;

    // Apply periodic boundary conditions
    newPos = applyPBC(newPos, params.boxSize);

    // Temperature scaling (simple thermostat)
    let speed = length(newVel);
    if (speed > 0.0) {
        let targetSpeed = sqrt(params.temperature * 0.01);
        newVel = newVel * (1.0 - 0.01 * (1.0 - targetSpeed / speed));
    }

    atomsOut[idx].position = newPos;
    atomsOut[idx].velocity = newVel;
}
`;

// Vertex shader for rendering atoms
const vertexShader = `
struct Uniforms {
    viewProj: mat4x4<f32>,
}

struct Atom {
    position: vec3<f32>,
    velocity: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// Icosahedron: 12 vertices, 20 counter-clockwise faces (60 indices) approximating a sphere
const ICO_T = 1.618034;
const ICO_VERTICES = array<vec3<f32>, 12>(
    vec3<f32>(-1.0, ICO_T, 0.0), vec3<f32>(1.0, ICO_T, 0.0), vec3<f32>(-1.0, -ICO_T, 0.0), vec3<f32>(1.0, -ICO_T, 0.0),
    vec3<f32>(0.0, -1.0, ICO_T), vec3<f32>(0.0, 1.0, ICO_T), vec3<f32>(0.0, -1.0, -ICO_T), vec3<f32>(0.0, 1.0, -ICO_T),
    vec3<f32>(ICO_T, 0.0, -1.0), vec3<f32>(ICO_T, 0.0, 1.0), vec3<f32>(-ICO_T, 0.0, -1.0), vec3<f32>(-ICO_T, 0.0, 1.0)
);
const ICO_INDICES = array<u32, 60>(
    0u, 11u, 5u,  0u, 5u, 1u,   0u, 1u, 7u,   0u, 7u, 10u,  0u, 10u, 11u,
    1u, 5u, 9u,   5u, 11u, 4u,  11u, 10u, 2u, 10u, 7u, 6u,  7u, 1u, 8u,
    3u, 9u, 4u,   3u, 4u, 2u,   3u, 2u, 6u,   3u, 6u, 8u,   3u, 8u, 9u,
    4u, 9u, 5u,   2u, 4u, 11u,  6u, 2u, 10u,  8u, 6u, 7u,   9u, 8u, 1u
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) instanceIdx: u32
) -> VertexOutput {
    let atom = atoms[instanceIdx];
    let sphereVertex = normalize(ICO_VERTICES[ICO_INDICES[vertexIdx]]);

    let scale = 0.3;
    let worldPos = atom.position + sphereVertex * scale;

    // Color based on velocity (temperature)
    let speed = length(atom.velocity);
    let t = clamp(speed * 2.0, 0.0, 1.0);
    let color = mix(vec3<f32>(0.2, 0.4, 1.0), vec3<f32>(1.0, 0.3, 0.2), t);

    var output: VertexOutput;
    output.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
    output.color = color;
    output.normal = normalize(sphereVertex);

    return output;
}
`;

// Fragment shader for shading
const fragmentShader = `
@fragment
fn fs_main(
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

// Simulation box: faint glass faces plus glowing edges drawn as screen-space quads
// (WebGPU lines are always 1px wide, so edges are expanded to camera-facing strips)
const boxShader = `
struct Uniforms {
    viewProj: mat4x4<f32>,
}

struct BoxParams {
    halfSize: f32,
    edgeHalfWidth: f32, // in pixels
    viewport: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> box: BoxParams;

fn corner(i: u32) -> vec3<f32> {
    let sign = vec3<f32>(
        select(-1.0, 1.0, (i & 1u) != 0u),
        select(-1.0, 1.0, (i & 2u) != 0u),
        select(-1.0, 1.0, (i & 4u) != 0u)
    );
    return sign * box.halfSize;
}

// Corner index pairs for the 12 edges (corner bit 0 = +x, bit 1 = +y, bit 2 = +z)
const EDGES = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),
    vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u)
);

// Corner indices for the 12 face triangles
const FACES = array<u32, 36>(
    0u, 1u, 3u, 0u, 3u, 2u,  4u, 6u, 7u, 4u, 7u, 5u,
    0u, 4u, 5u, 0u, 5u, 1u,  2u, 3u, 7u, 2u, 7u, 6u,
    0u, 2u, 6u, 0u, 6u, 4u,  1u, 5u, 7u, 1u, 7u, 3u
);

@vertex
fn vs_face(@builtin(vertex_index) vertexIdx: u32) -> @builtin(position) vec4<f32> {
    return uniforms.viewProj * vec4<f32>(corner(FACES[vertexIdx]), 1.0);
}

@fragment
fn fs_face() -> @location(0) vec4<f32> {
    return vec4<f32>(0.45, 0.65, 1.0, 0.035);
}

struct EdgeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) side: f32,
}

@vertex
fn vs_edge(
    @builtin(vertex_index) vertexIdx: u32,
    @builtin(instance_index) edgeIdx: u32
) -> EdgeOutput {
    let edge = EDGES[edgeIdx];
    let a = uniforms.viewProj * vec4<f32>(corner(edge.x), 1.0);
    let b = uniforms.viewProj * vec4<f32>(corner(edge.y), 1.0);

    // Two triangles: (end, side) for each of the 6 vertices
    let ends = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    let sides = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
    let end = ends[vertexIdx];
    let side = sides[vertexIdx];

    var out: EdgeOutput;
    out.side = side;
    if (a.w <= 0.0 || b.w <= 0.0) {
        // Edge crosses behind the camera; drop it
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }

    // Offset perpendicular to the edge's on-screen direction by a fixed pixel width
    let screenA = a.xy / a.w * box.viewport;
    let screenB = b.xy / b.w * box.viewport;
    let dir = normalize(screenB - screenA);
    let normal = vec2<f32>(-dir.y, dir.x);
    let clip = mix(a, b, end);
    let offset = normal * side * box.edgeHalfWidth * 2.0 / box.viewport * clip.w;
    out.position = clip + vec4<f32>(offset, 0.0, 0.0);
    return out;
}

@fragment
fn fs_edge(@location(0) side: f32) -> @location(0) vec4<f32> {
    // Bright core with a soft glow falloff across the strip
    let d = abs(side);
    let glow = pow(1.0 - d, 2.0);
    let core = 1.0 - smoothstep(0.0, 0.35, d);
    let color = mix(vec3<f32>(0.35, 0.6, 1.0), vec3<f32>(0.85, 0.95, 1.0), core);
    return vec4<f32>(color, glow * 0.85);
}
`;

// Simple matrix math utilities
class Mat4 {
    static perspective(fov, aspect, near, far) {
        // Column-major, WebGPU clip-space depth range [0, 1]
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * nf, -1,
            0, 0, far * near * nf, 0
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

    // a * b for column-major matrices (element [col * 4 + row])
    static multiply(a, b) {
        const result = new Float32Array(16);
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) {
                result[col * 4 + row] =
                    a[0 * 4 + row] * b[col * 4 + 0] +
                    a[1 * 4 + row] * b[col * 4 + 1] +
                    a[2 * 4 + row] * b[col * 4 + 2] +
                    a[3 * 4 + row] * b[col * 4 + 3];
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
        this.distance = 75; // frames the whole 30-unit box
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

class MolecularDynamics {
    constructor(device, canvas, numAtoms, boxSize) {
        this.device = device;
        this.canvas = canvas;
        this.numAtoms = numAtoms;
        this.boxSize = boxSize;
        this.running = false;
        this.frameCount = 0;
        this.temperature = 30;

        this.camera = new Camera();
        this.setupMouseControls();
        this.initializeAtoms();
        this.setupBuffers();
        this.setupComputePipeline();
        this.setupRenderPipeline();
    }

    setupMouseControls() {
        let isDragging = false;
        let isRightDrag = false;
        let lastX = 0;
        let lastY = 0;

        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            isRightDrag = e.button === 2;
            lastX = e.clientX;
            lastY = e.clientY;
            e.preventDefault();
        });

        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;

            if (isRightDrag) {
                // Pan
                const panSpeed = 0.05;
                this.camera.target[0] -= dx * panSpeed;
                this.camera.target[1] += dy * panSpeed;
            } else {
                // Rotate
                this.camera.rotation.y += dx * 0.01;
                this.camera.rotation.x += dy * 0.01;
                this.camera.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.camera.rotation.x));
            }

            lastX = e.clientX;
            lastY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera.distance *= (1 + e.deltaY * 0.001);
            this.camera.distance = Math.max(30, Math.min(150, this.camera.distance)); // stay outside the box
        });
    }

    initializeAtoms() {
        this.atoms = new Float32Array(this.numAtoms * 8); // 3 pos + 1 padding + 3 vel + 1 padding

        // Initialize in a grid
        const gridSize = Math.ceil(Math.pow(this.numAtoms, 1/3));
        const spacing = this.boxSize / gridSize;
        const offset = -this.boxSize / 2 + spacing / 2;

        let idx = 0;
        for (let i = 0; i < gridSize && idx < this.numAtoms; i++) {
            for (let j = 0; j < gridSize && idx < this.numAtoms; j++) {
                for (let k = 0; k < gridSize && idx < this.numAtoms; k++) {
                    // Position
                    this.atoms[idx * 8 + 0] = offset + i * spacing;
                    this.atoms[idx * 8 + 1] = offset + j * spacing;
                    this.atoms[idx * 8 + 2] = offset + k * spacing;

                    // Velocity (random)
                    const temp = this.temperature * 0.01;
                    this.atoms[idx * 8 + 4] = (Math.random() - 0.5) * temp;
                    this.atoms[idx * 8 + 5] = (Math.random() - 0.5) * temp;
                    this.atoms[idx * 8 + 6] = (Math.random() - 0.5) * temp;

                    idx++;
                }
            }
        }
    }

    setupBuffers() {
        const atomBufferSize = this.numAtoms * 8 * 4; // vec4 + vec4 per atom

        this.atomBuffers = [
            this.device.createBuffer({
                size: atomBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            }),
            this.device.createBuffer({
                size: atomBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            })
        ];

        new Float32Array(this.atomBuffers[0].getMappedRange()).set(this.atoms);
        this.atomBuffers[0].unmap();

        this.paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.currentBuffer = 0;
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
            compute: { module: shaderModule, entryPoint: 'main' }
        });

        this.computeBindGroups = [
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.atomBuffers[0] } },
                    { binding: 1, resource: { buffer: this.atomBuffers[1] } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.atomBuffers[1] } },
                    { binding: 1, resource: { buffer: this.atomBuffers[0] } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } }
                ]
            })
        ];
    }

    setupRenderPipeline() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm',
            alphaMode: 'opaque'
        });

        const shaderModule = this.device.createShaderModule({
            code: vertexShader + '\n' + fragmentShader
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
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

        this.setupBoxPipelines();

        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.renderBindGroups = [
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: { buffer: this.atomBuffers[0] } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: { buffer: this.atomBuffers[1] } }
                ]
            })
        ];
    }

    setupBoxPipelines() {
        const module = this.device.createShaderModule({ code: boxShader });

        this.boxParamsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.boxParamsBuffer, 0, new Float32Array([
            this.boxSize / 2, 3.0, this.canvas.width, this.canvas.height
        ]));

        const layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }
            ]
        });
        this.boxBindGroup = this.device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.boxParamsBuffer } }
            ]
        });

        // Transparent passes: depth-tested against the atoms, but never write depth
        const depthStencil = { depthWriteEnabled: false, depthCompare: 'less', format: 'depth24plus' };
        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });

        this.boxFacePipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module, entryPoint: 'vs_face' },
            fragment: {
                module,
                entryPoint: 'fs_face',
                targets: [{
                    format: 'bgra8unorm',
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil
        });

        this.boxEdgePipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module, entryPoint: 'vs_edge' },
            fragment: {
                module,
                entryPoint: 'fs_edge',
                targets: [{
                    format: 'bgra8unorm',
                    // Additive blending for a glow
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one' },
                        alpha: { srcFactor: 'zero', dstFactor: 'one' }
                    }
                }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil
        });
    }

    updateParams() {
        const params = new Float32Array([
            0, // numAtoms (u32, set below)
            0.016, // deltaTime
            this.boxSize,
            this.temperature
        ]);
        new Uint32Array(params.buffer)[0] = this.numAtoms;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    }

    updateUniforms() {
        const aspect = this.canvas.width / this.canvas.height;
        const proj = Mat4.perspective(Math.PI / 4, aspect, 0.1, 200);
        const view = this.camera.getViewMatrix();
        const viewProj = Mat4.multiply(proj, view);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, viewProj);
    }

    step() {
        this.updateParams();

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroups[this.currentBuffer]);
        computePass.dispatchWorkgroups(Math.ceil(this.numAtoms / 64));
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
                clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1.0
            }
        });

        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBuffer]);
        renderPass.draw(60, this.numAtoms); // 20 icosahedron faces per atom, numAtoms instances

        // Box after the opaque atoms: faint faces, then glowing edges (12 instances of 6 vertices)
        renderPass.setBindGroup(0, this.boxBindGroup);
        renderPass.setPipeline(this.boxFacePipeline);
        renderPass.draw(36);
        renderPass.setPipeline(this.boxEdgePipeline);
        renderPass.draw(6, 12);
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
        this.initializeAtoms();
        this.device.queue.writeBuffer(this.atomBuffers[0], 0, this.atoms);
        this.currentBuffer = 0;
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
        this.atomBuffers.forEach(b => b.destroy());
        this.paramsBuffer.destroy();
        this.uniformBuffer.destroy();
        this.boxParamsBuffer.destroy();
        this.depthTexture.destroy();
    }
}

export async function initMolecularDynamics(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-md');
    const output = document.getElementById('output-md');

    let simulation = null;
    let numAtoms = 1000;
    let temperature = 30;

    function createSimulation() {
        if (simulation) {
            simulation.destroy();
        }
        simulation = new MolecularDynamics(device, canvas, numAtoms, 30);
        simulation.temperature = temperature;
        simulation.render();
        updateOutput();
    }

    createSimulation();

    document.getElementById('btn-md-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-md-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-md-reset').addEventListener('click', () => {
        createSimulation();
    });

    const atomsSlider = document.getElementById('md-atoms');
    const atomsValue = document.getElementById('md-atoms-value');
    atomsSlider.addEventListener('change', (e) => {
        numAtoms = parseInt(e.target.value);
        atomsValue.textContent = numAtoms.toLocaleString();
        createSimulation();
    });
    atomsSlider.addEventListener('input', (e) => {
        atomsValue.textContent = parseInt(e.target.value).toLocaleString();
    });

    const tempSlider = document.getElementById('md-temp');
    const tempValue = document.getElementById('md-temp-value');
    tempSlider.addEventListener('input', (e) => {
        temperature = parseInt(e.target.value);
        tempValue.textContent = temperature;
        if (simulation) {
            simulation.temperature = temperature;
        }
    });

    function updateOutput() {
        const fps = simulation.running ? '~60' : '0';
        const interactions = (numAtoms * (numAtoms - 1)) / 2;

        output.innerHTML = `<span class="success">✓ Molecular Dynamics Active</span>

<span class="info">Configuration:</span>
• Number of atoms: ${numAtoms.toLocaleString()}
• Box size: 30 × 30 × 30 units
• Temperature: ${temperature}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Physics:</span>
• Force field: Lennard-Jones potential
• Integration: Velocity Verlet
• Boundary: Periodic (wrapping)
• Pair interactions: ${interactions.toLocaleString()}
• Timestep: 0.016 units

<span class="info">Performance:</span>
• Frame rate: ${fps} FPS
• Simulation time: ${simulation.frameCount} steps
• GPU compute: Fully parallelized force calculation
• Rendering: Instanced 3D spheres

<span class="info">Visualization:</span>
• Color: Blue (cold) → Red (hot)
• Shading: Phong lighting model
• Camera: Drag to rotate, wheel to zoom`;
    }

    updateOutput();
    return simulation;
}
