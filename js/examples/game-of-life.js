import { GpuTimer } from '../gpu-timer.js';
import { readBuffer } from '../webgpu-utils.js';

const computeShader = `
@group(0) @binding(0) var<storage, read> gridIn: array<u32>;
@group(0) @binding(1) var<storage, read_write> gridOut: array<u32>;
@group(0) @binding(2) var<uniform> dimensions: vec2<u32>;

fn getCell(x: i32, y: i32, width: u32, height: u32) -> u32 {
    let wx = (x + i32(width)) % i32(width);
    let wy = (y + i32(height)) % i32(height);
    return gridIn[u32(wy) * width + u32(wx)];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let width = dimensions.x;
    let height = dimensions.y;

    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let current = getCell(x, y, width, height);

    var neighbors = 0u;
    neighbors += getCell(x - 1, y - 1, width, height);
    neighbors += getCell(x, y - 1, width, height);
    neighbors += getCell(x + 1, y - 1, width, height);
    neighbors += getCell(x - 1, y, width, height);
    neighbors += getCell(x + 1, y, width, height);
    neighbors += getCell(x - 1, y + 1, width, height);
    neighbors += getCell(x, y + 1, width, height);
    neighbors += getCell(x + 1, y + 1, width, height);

    var nextState = 0u;
    if (current == 1u) {
        if (neighbors == 2u || neighbors == 3u) {
            nextState = 1u;
        }
    } else {
        if (neighbors == 3u) {
            nextState = 1u;
        }
    }

    gridOut[global_id.y * width + global_id.x] = nextState;
}
`;

const renderShader = `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    return output;
}

@group(0) @binding(0) var<storage, read> grid: array<u32>;
@group(0) @binding(1) var<uniform> dimensions: vec2<u32>;

@fragment
fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let x = u32(uv.x * f32(dimensions.x));
    let y = u32(uv.y * f32(dimensions.y));
    let cell = grid[y * dimensions.x + x];

    if (cell == 1u) {
        return vec4<f32>(0.2, 0.8, 0.4, 1.0);
    } else {
        return vec4<f32>(0.1, 0.1, 0.1, 1.0);
    }
}
`;

const SEED_PATTERNS = {
    glider: {
        width: 3,
        height: 3,
        cells: [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ]
    },
    lwss: {
        width: 5,
        height: 4,
        cells: [
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]
        ]
    },
    pulsar: {
        width: 13,
        height: 13,
        cells: [
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0]
        ]
    },
    pentadecathlon: {
        width: 10,
        height: 3,
        cells: [
            [0,0,1,0,0,0,0,1,0,0],
            [1,1,0,1,1,1,1,0,1,1],
            [0,0,1,0,0,0,0,1,0,0]
        ]
    },
    'gosper-gun': {
        width: 36,
        height: 9,
        cells: [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
            [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]
    },
    acorn: {
        width: 7,
        height: 3,
        cells: [
            [0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0],
            [1,1,0,0,1,1,1]
        ]
    },
    'r-pentomino': {
        width: 3,
        height: 3,
        cells: [
            [0,1,1],
            [1,1,0],
            [0,1,0]
        ]
    },
    diehard: {
        width: 8,
        height: 3,
        cells: [
            [0,0,0,0,0,0,1,0],
            [1,1,0,0,0,0,0,0],
            [0,1,0,0,0,1,1,1]
        ]
    }
};

class GameOfLife {
    constructor(device, canvas, width, height) {
        this.device = device;
        this.canvas = canvas;
        this.width = width;
        this.height = height;
        this.gridSize = width * height;
        this.running = false;
        this.generation = 0;
        this.lastFrameTime = 0;
        this.targetFPS = 10;
        this.totalCells = 0;

        this.setupBuffers();
        this.setupComputePipeline();
        // canvas = null: compute-only (Measure tab, test/gpu/), no render pipeline and nothing drawn
        if (this.canvas) {
            this.setupRenderPipeline();
            this.setupCanvas();
        }
    }

    setupBuffers() {
        const bufferSize = this.gridSize * 4;

        // COPY_SRC: toggleCell() and the Measure tab copy the grid back to the CPU
        this.gridBuffers = [
            this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            }),
            this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            })
        ];

        const dimensionsArray = new Uint32Array([this.width, this.height]);
        this.dimensionsBuffer = this.device.createBuffer({
            size: dimensionsArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint32Array(this.dimensionsBuffer.getMappedRange()).set(dimensionsArray);
        this.dimensionsBuffer.unmap();
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
                    { binding: 0, resource: { buffer: this.gridBuffers[0] } },
                    { binding: 1, resource: { buffer: this.gridBuffers[1] } },
                    { binding: 2, resource: { buffer: this.dimensionsBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.gridBuffers[1] } },
                    { binding: 1, resource: { buffer: this.gridBuffers[0] } },
                    { binding: 2, resource: { buffer: this.dimensionsBuffer } }
                ]
            })
        ];

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
            vertex: { module: shaderModule, entryPoint: 'vertexMain' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: 'bgra8unorm' }]
            }
        });

        this.renderBindGroups = [
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.gridBuffers[0] } },
                    { binding: 1, resource: { buffer: this.dimensionsBuffer } }
                ]
            }),
            this.device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.gridBuffers[1] } },
                    { binding: 1, resource: { buffer: this.dimensionsBuffer } }
                ]
            })
        ];
    }

    setupCanvas() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });
    }

    setGrid(data) {
        this.device.queue.writeBuffer(this.gridBuffers[this.currentBuffer], 0, data);
        this.totalCells = data.reduce((sum, val) => sum + val, 0);
        this.render();
    }

    clear() {
        const data = new Uint32Array(this.gridSize).fill(0);
        this.setGrid(data);
        this.generation = 0;
    }

    randomize() {
        const data = new Uint32Array(this.gridSize);
        for (let i = 0; i < this.gridSize; i++) {
            data[i] = Math.random() > 0.7 ? 1 : 0;
        }
        this.setGrid(data);
        this.generation = 0;
    }

    loadPattern(patternName) {
        const pattern = SEED_PATTERNS[patternName];
        if (!pattern) return;

        const data = new Uint32Array(this.gridSize).fill(0);
        const offsetX = Math.floor((this.width - pattern.width) / 2);
        const offsetY = Math.floor((this.height - pattern.height) / 2);

        for (let y = 0; y < pattern.height; y++) {
            for (let x = 0; x < pattern.width; x++) {
                if (pattern.cells[y][x] === 1) {
                    const idx = (offsetY + y) * this.width + (offsetX + x);
                    data[idx] = 1;
                }
            }
        }

        this.setGrid(data);
        this.generation = 0;
    }

    toggleCell(x, y) {
        const readBuffer = this.device.createBuffer({
            size: this.gridSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.gridBuffers[this.currentBuffer],
            0,
            readBuffer,
            0,
            this.gridSize * 4
        );
        this.device.queue.submit([commandEncoder.finish()]);

        readBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Uint32Array(readBuffer.getMappedRange().slice(0));
            const idx = y * this.width + x;
            data[idx] = data[idx] === 1 ? 0 : 1;
            this.setGrid(data);
            readBuffer.unmap();
            readBuffer.destroy();
        });
    }

    step() {
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroups[this.currentBuffer]);
        computePass.dispatchWorkgroups(
            Math.ceil(this.width / 8),
            Math.ceil(this.height / 8)
        );
        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        this.currentBuffer = 1 - this.currentBuffer;
        this.generation++;
        this.render();
    }

    render() {
        if (!this.context) return;   // compute-only
        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBuffer]);
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

    animate(timestamp = 0) {
        if (!this.running) return;

        const elapsed = timestamp - this.lastFrameTime;
        const frameInterval = 1000 / this.targetFPS;

        if (elapsed >= frameInterval) {
            this.step();
            this.lastFrameTime = timestamp;
        }

        requestAnimationFrame((t) => this.animate(t));
    }

    destroy() {
        this.running = false;
        this.gridBuffers.forEach(b => b.destroy());
        this.dimensionsBuffer.destroy();
    }
}

export async function initGameOfLife(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-gol');
    const output = document.getElementById('output-gol');

    const width = 200;
    const height = 150;

    const game = new GameOfLife(device, canvas, width, height);
    game.clear();

    document.getElementById('btn-gol-start').addEventListener('click', () => {
        game.start();
        updateOutput();
    });

    document.getElementById('btn-gol-pause').addEventListener('click', () => {
        game.pause();
        updateOutput();
    });

    document.getElementById('btn-gol-step').addEventListener('click', () => {
        game.step();
        updateOutput();
    });

    document.getElementById('btn-gol-clear').addEventListener('click', () => {
        game.clear();
        updateOutput();
    });

    document.getElementById('btn-gol-random').addEventListener('click', () => {
        game.randomize();
        updateOutput();
    });

    const speedSlider = document.getElementById('gol-speed');
    const speedValue = document.getElementById('gol-speed-value');
    speedSlider.addEventListener('input', (e) => {
        game.targetFPS = parseInt(e.target.value);
        speedValue.textContent = `${game.targetFPS} fps`;
    });

    document.querySelectorAll('.seed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const seed = btn.dataset.seed;
            game.loadPattern(seed);
            updateOutput();
        });
    });

    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * width);
        // The render shader draws row 0 at the bottom (uv.y = 0 at clip y = −1), so flip the screen y
        const y = height - 1 - Math.min(height - 1, Math.floor((e.clientY - rect.top) / rect.height * height));
        game.toggleCell(x, y);
    });

    function updateOutput() {
        output.innerHTML = `<span class="success">✓ Game of Life Active</span>

<span class="info">Configuration:</span>
• Grid size: ${width} × ${height} (${(width * height).toLocaleString()} cells)
• Generation: ${game.generation.toLocaleString()}
• Status: ${game.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}
• Speed: ${game.targetFPS} fps

<span class="info">Interaction:</span>
• Click cells to toggle them on/off
• Load classic patterns from buttons
• Use controls to start/pause/step simulation

<span class="info">Performance:</span>
• Fully GPU-accelerated using compute shaders
• Each generation computed in parallel
• Zero CPU overhead for simulation logic`;
    }

    updateOutput();
    return game;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}

// Exported for the CPU tests in test/game-of-life.test.mjs.
export { GameOfLife, SEED_PATTERNS };

// ---- Compute-only API for the Measure tab (js/teach/experiments/game-of-life.js) and test/gpu/ ----
//
// measureLife() builds a GameOfLife without a canvas (same buffers, pipeline and ping-pong bind groups as
// the interactive example), uploads a grid and advances it `generations` steps. Unlike the example, which
// submits one generation per animation frame, all generations go into one compute pass: WebGPU makes each
// dispatch's storage writes visible to the next dispatch, so the ping-pong needs no other synchronization,
// and the timestamps then measure the stencil itself rather than per-frame submission overhead.

export const MAX_LIFE_SIDE = 2048;              // 4.2 M cells, 16 MB per grid buffer
export const MAX_LIFE_GENERATIONS = 1000;       // per call (one pass)
export const MAX_LIFE_WORK = 2048 * 2048 * 100; // cell updates per pass (100 generations of the largest grid)
export const MAX_MEASURE_REPEATS = 10;

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((p, q) => p - q);
    return v.length ? v[v.length >> 1] : null;
};

// grid: width × height Uint32Array of 0/1 (row-major, toroidal). Runs `warmup` untimed and `repeats` timed
// batches of `generations` generations, each batch starting again from `grid`. Resolves to { width, height,
// generations, gpuMs, wallMs, grid? } with times per batch (medians; gpuMs null without timestamp-query)
// and grid the state after `generations` generations when readback is true.
export async function measureLife(gpuDevice, { grid, width, height, generations = 1, warmup = 0, repeats = 1, readback = true }) {
    const device = gpuDevice.device ?? gpuDevice;
    for (const [name, v] of [['width', width], ['height', height]]) {
        if (!Number.isInteger(v) || v < 3 || v > MAX_LIFE_SIDE) throw new RangeError(`${name} = ${v} is outside 3 … ${MAX_LIFE_SIDE}`);
    }
    if (!Number.isInteger(generations) || generations < 1 || generations > MAX_LIFE_GENERATIONS) {
        throw new RangeError(`generations = ${generations} is outside 1 … ${MAX_LIFE_GENERATIONS}`);
    }
    if (grid?.length !== width * height) throw new RangeError('grid must hold width × height cells');
    if (generations * width * height > MAX_LIFE_WORK) throw new RangeError(`more than ${MAX_LIFE_WORK} cell updates in one pass`);
    const game = new GameOfLife(device, null, width, height);
    const timer = new GpuTimer(gpuDevice, { maxPasses: 1, poolSize: 1 });
    try {
        const timedRuns = Math.min(Math.max(repeats, 1), MAX_MEASURE_REPEATS);
        const total = Math.min(Math.max(warmup, 0), MAX_MEASURE_REPEATS) + timedRuns;
        const gpu = [], wall = [];
        for (let r = 0; r < total; r++) {
            const timed = r >= total - timedRuns;
            game.currentBuffer = 0;
            device.queue.writeBuffer(game.gridBuffers[0], 0, grid);
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass(timed ? timer.computePassDescriptor('life') : undefined);
            pass.setPipeline(game.computePipeline);
            for (let g = 0; g < generations; g++) {
                pass.setBindGroup(0, game.computeBindGroups[game.currentBuffer]);
                pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
                game.currentBuffer = 1 - game.currentBuffer;
            }
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
        const out = readback
            ? await readBuffer(device, game.gridBuffers[game.currentBuffer], 4 * width * height, Uint32Array) : undefined;
        return { width, height, generations, gpuMs: median(gpu), wallMs: median(wall), grid: out };
    } finally {
        timer.destroy();
        game.destroy();
    }
}
