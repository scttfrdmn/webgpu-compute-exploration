// Voronoi diagrams and distance fields with the Jump Flooding Algorithm (Rong & Tan 2006).
// A seed pass writes each seed's index into its pixel; then log2(N) flood passes with step sizes
// N/2, N/4, ..., 1 let every pixel adopt the closest seed seen among itself and its 8 neighbors at
// ±step. The optional "JFA+1" pass repeats step 1 at the end to fix most remaining errors. A
// full-screen fragment pass colors the cells, the distance field, or both.
//
// Uniform layout (struct Params, 32 bytes; shared by every shader):
//   offset  0  size       vec2<u32>  grid width, height (= canvas pixels)    -> Uint32 view [0], [1]
//   offset  8  stepSize   u32        jump distance of this flood pass        -> Uint32 view [2]
//   offset 12  metric     u32        0 Euclidean, 1 Manhattan, 2 Chebyshev   -> Uint32 view [3]
//   offset 16  seedCount  u32        number of valid entries in seeds[]      -> Uint32 view [4]
//   offset 20  viewMode   u32        0 cells, 1 distance field, 2 combined   -> Uint32 view [5]
//   offset 24  selected   u32        seed being dragged, NONE if none        -> Uint32 view [6]
//   offset 28  distScale  f32        distance mapped to full brightness      -> Float32 view [7]
// One copy per flood pass sits at a dynamic offset of paramsStride bytes, so each dispatch sees its
// own stepSize (every writeBuffer lands before the submit runs, so one shared slot would give all
// passes the last value). Slot 0 also serves the seed, render and exact-check passes.
//
// Seed layout (struct Seed, 16 bytes, storage buffer of MAX_SEEDS entries):
//   offset  0  pos    vec2<f32>  position in pixels                   -> Float32 view [4i], [4i+1]
//   offset  8  color  u32        RGBA8, unpacked with unpack4x8unorm   -> Uint32 view [4i+2]
//   offset 12  pad    u32        unused                                -> Uint32 view [4i+3]
const PARAMS_SIZE = 32;
const SEED_SIZE = 16;
const MAX_SEEDS = 1024;
const MAX_PASSES = 16;
const NONE = 0xffffffff;

const METRICS = ['Euclidean', 'Manhattan', 'Chebyshev'];
const VIEWS = ['Cells', 'Distance field', 'Combined'];

const commonWGSL = `
struct Params {
    size: vec2<u32>,
    stepSize: u32,
    metric: u32,
    seedCount: u32,
    viewMode: u32,
    selected: u32,
    distScale: f32,
}

struct Seed {
    pos: vec2<f32>,
    color: u32,
    pad: u32,
}

const NONE: u32 = 0xffffffffu;
const MAX_SEEDS: u32 = ${MAX_SEEDS}u;

// Distance used to compare seeds. Euclidean returns the squared distance: it orders seeds the same
// way and skips the sqrt. Manhattan |dx| + |dy| gives diamond-shaped balls, Chebyshev max(|dx|, |dy|)
// square ones, so their cells have straight edges at 0°, 45° and 90°.
fn metricDistance(a: vec2<f32>, b: vec2<f32>, metric: u32) -> f32 {
    let d = abs(a - b);
    if (metric == 1u) {
        return d.x + d.y;
    }
    if (metric == 2u) {
        return max(d.x, d.y);
    }
    return dot(d, d);
}
`;

const seedShader = commonWGSL + `
@group(0) @binding(0) var<storage, read_write> ids: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> seeds: array<Seed>;
@group(0) @binding(2) var<uniform> params: Params;

// Pass 1: every pixel starts with no seed
@compute @workgroup_size(8, 8)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    atomicStore(&ids[gid.y * params.size.x + gid.x], NONE);
}

// Pass 2: one invocation per seed writes its index into the pixel that contains it. atomicMin makes
// the lowest index win when two seeds share a pixel, so the result does not depend on scheduling.
@compute @workgroup_size(64)
fn cs_seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.seedCount) {
        return;
    }
    let maxPixel = vec2<i32>(params.size) - vec2<i32>(1);
    let pixel = clamp(vec2<i32>(floor(seeds[i].pos)), vec2<i32>(0), maxPixel);
    atomicMin(&ids[u32(pixel.y) * params.size.x + u32(pixel.x)], i);
}
`;

const floodShader = commonWGSL + `
@group(0) @binding(0) var<storage, read> idsIn: array<u32>;
@group(0) @binding(1) var<storage, read_write> idsOut: array<u32>;
@group(0) @binding(2) var<storage, read> seeds: array<Seed>;
@group(0) @binding(3) var<uniform> params: Params;

// One jump-flooding pass with step k = params.stepSize. Pixel p looks at the seed stored at itself
// and at its 8 neighbors p + k·(dx, dy), dx, dy ∈ {-1, 0, 1}, and keeps the one closest to p's
// center (ties go to the lower seed index). After passes with k = N/2, N/4, ..., 1 every pixel has
// (almost always) its nearest seed: information travels up to N - 1 pixels in log2(N) passes.
// Errors are rare and happen where a closer seed's region is cut off by other cells before its
// information arrives; a second step-1 pass (JFA+1) repairs most of them.
@compute @workgroup_size(8, 8)
fn cs_flood(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let w = i32(params.size.x);
    let h = i32(params.size.y);
    let p = vec2<i32>(gid.xy);
    let center = vec2<f32>(gid.xy) + vec2<f32>(0.5);
    let k = i32(params.stepSize);

    var bestId = NONE;
    var bestDist = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = p + vec2<i32>(dx, dy) * k;
            if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
                continue;
            }
            let id = idsIn[u32(q.y) * params.size.x + u32(q.x)];
            if (id >= params.seedCount) {
                continue;   // NONE (no seed seen yet)
            }
            let d = metricDistance(center, seeds[id].pos, params.metric);
            if (bestId == NONE || d < bestDist || (d == bestDist && id < bestId)) {
                bestId = id;
                bestDist = d;
            }
        }
    }
    idsOut[gid.y * params.size.x + gid.x] = bestId;
}
`;

const exactShader = commonWGSL + `
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> seeds: array<Seed>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

// Brute force: every pixel tests every seed (O(pixels × seeds)) and compares with the JFA result.
// counters[0]: pixels whose JFA seed is strictly farther than the true nearest seed (real errors)
// counters[1]: pixels whose JFA seed index differs (also counts equidistant ties)
@compute @workgroup_size(8, 8)
fn cs_exact(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    let center = vec2<f32>(gid.xy) + vec2<f32>(0.5);
    var bestId = NONE;
    var bestDist = 0.0;
    for (var i = 0u; i < MAX_SEEDS; i++) {
        if (i >= params.seedCount) {
            break;
        }
        let d = metricDistance(center, seeds[i].pos, params.metric);
        if (bestId == NONE || d < bestDist) {
            bestId = i;
            bestDist = d;
        }
    }
    if (bestId == NONE) {
        return;
    }
    let jfaId = ids[gid.y * params.size.x + gid.x];
    if (jfaId != bestId) {
        atomicAdd(&counters[1], 1u);
    }
    if (jfaId >= params.seedCount || metricDistance(center, seeds[jfaId].pos, params.metric) > bestDist) {
        atomicAdd(&counters[0], 1u);
    }
}
`;

const renderShader = commonWGSL + `
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> seeds: array<Seed>;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

// One triangle that covers the whole screen
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(corners[vi], 0.0, 1.0);
    return out;
}

const CONTOUR: f32 = 16.0;      // distance between iso-distance lines, in pixels
const DOT_RADIUS: f32 = 3.5;    // seed dot radius, in pixels
const BACKGROUND: vec3<f32> = vec3<f32>(0.04, 0.05, 0.08);
const BORDER: vec3<f32> = vec3<f32>(0.05, 0.05, 0.07);

fn idAt(x: i32, y: i32, fallback: u32) -> u32 {
    if (x < 0 || y < 0 || x >= i32(params.size.x) || y >= i32(params.size.y)) {
        return fallback;
    }
    return ids[u32(y) * params.size.x + u32(x)];
}

// The canvas has one pixel per grid cell, so the fragment position is the cell (its center).
@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let x = i32(frag.position.x);
    let y = i32(frag.position.y);
    let id = idAt(x, y, NONE);
    if (id >= params.seedCount) {
        return vec4<f32>(BACKGROUND, 1.0);
    }
    let seed = seeds[id];
    let center = vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5);
    let cellColor = unpack4x8unorm(seed.color).rgb;

    // Distance to this pixel's seed in the selected metric (undo the Euclidean square)
    var d = metricDistance(center, seed.pos, params.metric);
    if (params.metric == 0u) {
        d = sqrt(d);
    }
    let t = clamp(d / params.distScale, 0.0, 1.0);
    // 1 on an iso-distance line, fading out over about a pixel on each side
    let toLine = abs(fract(d / CONTOUR + 0.5) - 0.5) * CONTOUR;
    let contour = 1.0 - smoothstep(0.5, 1.5, toLine);

    // A pixel is on a border if a 4-neighbor belongs to another seed (gives 2-pixel lines)
    let border = idAt(x + 1, y, id) != id || idAt(x - 1, y, id) != id ||
                 idAt(x, y + 1, id) != id || idAt(x, y - 1, id) != id;

    var color: vec3<f32>;
    if (params.viewMode == 1u) {
        // Distance field: black at the seeds, white at distScale, with darker contour lines
        color = vec3<f32>(0.03 + 0.92 * t) * (1.0 - 0.45 * contour);
    } else if (params.viewMode == 2u) {
        // Combined: cell color darkening with distance, contour lines and borders
        color = cellColor * (1.05 - 0.75 * t) * (1.0 - 0.3 * contour);
        if (border) {
            color = BORDER;
        }
    } else {
        color = cellColor;
        if (border) {
            color = BORDER;
        }
    }

    // Seed dot (always round): light fill with a dark rim; the dragged seed is highlighted
    let e = length(center - seed.pos);
    if (e < DOT_RADIUS) {
        if (id == params.selected) {
            color = vec3<f32>(1.0, 0.85, 0.2);
        } else {
            color = vec3<f32>(0.97, 0.97, 0.95);
        }
    } else if (e < DOT_RADIUS + 1.5) {
        color = BORDER;
    }
    return vec4<f32>(min(color, vec3<f32>(1.0)), 1.0);
}
`;

const PICK_RADIUS = 12;      // pixels: clicking this close to a seed grabs it instead of adding one
const MIN_SPEED = 25;        // animation speed range, pixels per second
const MAX_SPEED = 80;

const LAYOUTS = {
    random:    { name: 'Random' },
    grid:      { name: 'Jittered grid' },
    sunflower: { name: 'Sunflower' },
    clusters:  { name: 'Clusters' },
};

// Step sizes for one JFA run: N/2, N/4, ..., 1 with N the next power of two ≥ max(width, height),
// plus one more step-1 pass for JFA+1
export function jfaSteps(width, height, plusOne) {
    let n = 1;
    while (n < Math.max(width, height)) n *= 2;
    const steps = [];
    for (let k = n / 2; k >= 1; k /= 2) steps.push(k);
    if (plusOne || steps.length === 0) steps.push(1);
    return steps;
}

// HSL to a packed RGBA8 u32 (r in the low byte, as unpack4x8unorm expects)
function packColor(h, s, l) {
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
        const k = (n + h * 12) % 12;
        return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
    };
    return (f(0) | (f(8) << 8) | (f(4) << 16) | (255 << 24)) >>> 0;
}

class VoronoiJFA {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.width = canvas.width;
        this.height = canvas.height;
        this.pixelCount = this.width * this.height;
        this.running = false;
        this.destroyed = false;
        this.renderPending = false;

        this.seeds = [];          // { x, y, vx, vy, color }
        this.metric = 0;
        this.viewMode = 0;
        this.plusOne = false;
        this.layout = 'random';
        this.dragIndex = -1;

        this.checking = false;
        this.lastCheck = null;

        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;
        this.lastFrameTime = 0;

        // One 256-byte-aligned slot of Params per flood pass (dynamic uniform offsets)
        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.params = new ArrayBuffer(this.paramsStride * MAX_PASSES);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);

        this.seedData = new ArrayBuffer(SEED_SIZE * MAX_SEEDS);
        this.seedF32 = new Float32Array(this.seedData);
        this.seedU32 = new Uint32Array(this.seedData);

        this.setupBuffers();
        this.setupComputePipelines();
        this.setupRenderPipeline();
        this.setupCanvas();
        this.setupMouseControls();
    }

    setupBuffers() {
        this.idBuffers = [0, 1].map(() => this.device.createBuffer({
            size: this.pixelCount * 4, // u32 seed index per pixel
            usage: GPUBufferUsage.STORAGE
        }));
        this.seedBuffer = this.device.createBuffer({
            size: SEED_SIZE * MAX_SEEDS,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.paramsBuffer = this.device.createBuffer({
            size: this.paramsStride * MAX_PASSES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.counterBuffer = this.device.createBuffer({
            size: 8, // two u32 counters
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.readbackBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    setupComputePipelines() {
        const C = GPUShaderStage.COMPUTE;
        const paramsEntry = (binding) => ({
            binding, visibility: C,
            buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE }
        });
        const paramsResource = { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE };

        // Seed passes: clear + scatter into idBuffers[0]
        const seedLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'storage' } },
                { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
                paramsEntry(2)
            ]
        });
        const seedModule = this.device.createShaderModule({ code: seedShader });
        const seedPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [seedLayout] });
        this.clearPipeline = this.device.createComputePipeline({
            layout: seedPipelineLayout,
            compute: { module: seedModule, entryPoint: 'cs_clear' }
        });
        this.seedPipeline = this.device.createComputePipeline({
            layout: seedPipelineLayout,
            compute: { module: seedModule, entryPoint: 'cs_seed' }
        });
        this.seedBindGroup = this.device.createBindGroup({
            layout: seedLayout,
            entries: [
                { binding: 0, resource: { buffer: this.idBuffers[0] } },
                { binding: 1, resource: { buffer: this.seedBuffer } },
                { binding: 2, resource: paramsResource }
            ]
        });

        // Flood passes: floodBindGroups[i] reads idBuffers[i] and writes the other one
        const floodLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: C, buffer: { type: 'storage' } },
                { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
                paramsEntry(3)
            ]
        });
        this.floodPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [floodLayout] }),
            compute: { module: this.device.createShaderModule({ code: floodShader }), entryPoint: 'cs_flood' }
        });
        this.floodBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: floodLayout,
            entries: [
                { binding: 0, resource: { buffer: this.idBuffers[i] } },
                { binding: 1, resource: { buffer: this.idBuffers[1 - i] } },
                { binding: 2, resource: { buffer: this.seedBuffer } },
                { binding: 3, resource: paramsResource }
            ]
        }));

        // Exact check: exactBindGroups[i] compares idBuffers[i] against brute force
        const exactLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: C, buffer: { type: 'storage' } },
                paramsEntry(3)
            ]
        });
        this.exactPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [exactLayout] }),
            compute: { module: this.device.createShaderModule({ code: exactShader }), entryPoint: 'cs_exact' }
        });
        this.exactBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: exactLayout,
            entries: [
                { binding: 0, resource: { buffer: this.idBuffers[i] } },
                { binding: 1, resource: { buffer: this.seedBuffer } },
                { binding: 2, resource: { buffer: this.counterBuffer } },
                { binding: 3, resource: paramsResource }
            ]
        }));

        this.currentBuffer = 0;
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });
        const F = GPUShaderStage.FRAGMENT;
        const layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: F, buffer: { type: 'uniform' } }
            ]
        });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });
        // renderBindGroups[i] draws idBuffers[i], the one the last flood pass wrote
        this.renderBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.idBuffers[i] } },
                { binding: 1, resource: { buffer: this.seedBuffer } },
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
        // The canvas is CSS-scaled; convert client coordinates to canvas pixels (= grid cells)
        const toPixels = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * this.canvas.width / rect.width;
            const y = (e.clientY - rect.top) * this.canvas.height / rect.height;
            return [Math.min(Math.max(x, 0), this.width - 0.01), Math.min(Math.max(y, 0), this.height - 0.01)];
        };

        this.onMouseDown = (e) => {
            if (e.button !== 0 && e.button !== 2) return;
            e.preventDefault();
            const [x, y] = toPixels(e);
            const nearest = this.nearestSeed(x, y);
            if (e.button === 2 || e.shiftKey) {
                if (nearest.index >= 0) this.removeSeed(nearest.index);
                return;
            }
            if (nearest.index >= 0 && nearest.distance <= PICK_RADIUS) {
                this.dragIndex = nearest.index;
            } else if (this.seeds.length < MAX_SEEDS) {
                this.seeds.push(this.makeSeed(x, y));
                this.dragIndex = this.seeds.length - 1;
            }
            this.edited();
        };
        this.onMouseMove = (e) => {
            if (this.dragIndex < 0) return;
            const [x, y] = toPixels(e);
            const seed = this.seeds[this.dragIndex];
            seed.x = x;
            seed.y = y;
            this.edited();
        };
        this.onMouseUp = () => {
            if (this.dragIndex < 0) return;
            this.dragIndex = -1;
            this.edited();
        };
        this.onContextMenu = (e) => e.preventDefault();

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
    }

    makeSeed(x, y) {
        const angle = Math.random() * 2 * Math.PI;
        const speed = MIN_SPEED + Math.random() * (MAX_SPEED - MIN_SPEED);
        return {
            x, y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            color: packColor(Math.random(), 0.45 + Math.random() * 0.3, 0.48 + Math.random() * 0.18)
        };
    }

    nearestSeed(x, y) {
        let index = -1;
        let best = Infinity;
        this.seeds.forEach((s, i) => {
            const d = Math.hypot(s.x - x, s.y - y);
            if (d < best) {
                best = d;
                index = i;
            }
        });
        return { index, distance: best };
    }

    removeSeed(index) {
        this.seeds.splice(index, 1);
        if (this.dragIndex === index) this.dragIndex = -1;
        else if (this.dragIndex > index) this.dragIndex--;
        this.edited();
    }

    // Places `count` seeds in one of the LAYOUTS
    generate(count, layout = this.layout) {
        const n = Math.min(Math.max(Math.round(count), 0), MAX_SEEDS);
        const w = this.width;
        const h = this.height;
        const points = [];
        if (layout === 'grid') {
            // Roughly square cells, jittered by up to 35% of the spacing
            const cols = Math.max(1, Math.round(Math.sqrt(n * w / h)));
            const rows = Math.max(1, Math.ceil(n / cols));
            const sx = w / cols;
            const sy = h / rows;
            for (let i = 0; i < n; i++) {
                const c = i % cols;
                const r = Math.floor(i / cols);
                points.push([(c + 0.5 + (Math.random() - 0.5) * 0.7) * sx,
                             (r + 0.5 + (Math.random() - 0.5) * 0.7) * sy]);
            }
        } else if (layout === 'sunflower') {
            // Vogel's model: point i at radius ∝ √i and angle i·137.5° (the golden angle)
            const golden = Math.PI * (3 - Math.sqrt(5));
            const radius = 0.48 * Math.min(w, h);
            for (let i = 0; i < n; i++) {
                const r = radius * Math.sqrt((i + 0.5) / n);
                points.push([w / 2 + r * Math.cos(i * golden), h / 2 + r * Math.sin(i * golden)]);
            }
        } else if (layout === 'clusters') {
            // Gaussian blobs around a few random centers
            const k = Math.max(1, Math.min(6, Math.round(Math.sqrt(n) / 2)));
            const centers = Array.from({ length: k }, () => [w * (0.15 + 0.7 * Math.random()), h * (0.15 + 0.7 * Math.random())]);
            const sigma = 0.08 * Math.min(w, h);
            for (let i = 0; i < n; i++) {
                const [cx, cy] = centers[i % k];
                const r = sigma * Math.sqrt(-2 * Math.log(1 - Math.random()));
                const a = Math.random() * 2 * Math.PI;
                points.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
            }
        } else {
            for (let i = 0; i < n; i++) points.push([Math.random() * w, Math.random() * h]);
        }
        this.layout = LAYOUTS[layout] ? layout : 'random';
        this.seeds = points.map(([x, y]) => this.makeSeed(
            Math.min(Math.max(x, 0), w - 0.01), Math.min(Math.max(y, 0), h - 0.01)));
        this.dragIndex = -1;
        this.edited();
    }

    clear() {
        this.seeds = [];
        this.dragIndex = -1;
        this.edited();
    }

    // Any change: while paused, recompute and redraw once (coalesced per animation frame)
    edited() {
        if (this.onEdit) this.onEdit();
        this.requestRender();
    }

    requestRender() {
        if (this.running || this.renderPending || this.destroyed) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            if (!this.running && !this.destroyed) this.frame();
        });
    }

    steps() {
        return jfaSteps(this.width, this.height, this.plusOne);
    }

    // Typical nearest-seed distance: about the radius of an average cell
    distanceScale() {
        return Math.sqrt(this.pixelCount / Math.max(this.seeds.length, 1));
    }

    // Writes one Params slot per flood pass; see the layout table at the top of the file
    writeParams(steps) {
        const wordsPerSlot = this.paramsStride / 4;
        const slots = Math.max(steps.length, 1);
        for (let k = 0; k < slots; k++) {
            const o = k * wordsPerSlot;
            this.paramsU32[o + 0] = this.width;
            this.paramsU32[o + 1] = this.height;
            this.paramsU32[o + 2] = steps[k] ?? 1;
            this.paramsU32[o + 3] = this.metric;
            this.paramsU32[o + 4] = this.seeds.length;
            this.paramsU32[o + 5] = this.viewMode;
            this.paramsU32[o + 6] = this.dragIndex >= 0 ? this.dragIndex : NONE;
            this.paramsF32[o + 7] = this.distanceScale();
        }
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params, 0, slots * this.paramsStride);
    }

    // Uploads the seed positions and colors; see the Seed layout at the top of the file
    writeSeeds() {
        const n = this.seeds.length;
        for (let i = 0; i < n; i++) {
            const s = this.seeds[i];
            this.seedF32[4 * i + 0] = s.x;
            this.seedF32[4 * i + 1] = s.y;
            this.seedU32[4 * i + 2] = s.color;
            this.seedU32[4 * i + 3] = 0;
        }
        if (n > 0) this.device.queue.writeBuffer(this.seedBuffer, 0, this.seedData, 0, n * SEED_SIZE);
    }

    // Seed pass + all flood passes; afterwards idBuffers[currentBuffer] holds the diagram
    encodeCompute(commandEncoder) {
        const steps = this.steps();
        this.writeParams(steps);
        this.writeSeeds();

        const wgX = Math.ceil(this.width / 8);
        const wgY = Math.ceil(this.height / 8);
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.clearPipeline);
        pass.setBindGroup(0, this.seedBindGroup, [0]);
        pass.dispatchWorkgroups(wgX, wgY);
        this.currentBuffer = 0;
        if (this.seeds.length > 0) {
            pass.setPipeline(this.seedPipeline);
            pass.dispatchWorkgroups(Math.ceil(this.seeds.length / 64));
            pass.setPipeline(this.floodPipeline);
            for (let i = 0; i < steps.length; i++) {
                pass.setBindGroup(0, this.floodBindGroups[this.currentBuffer], [i * this.paramsStride]);
                pass.dispatchWorkgroups(wgX, wgY);
                this.currentBuffer = 1 - this.currentBuffer;
            }
        }
        pass.end();
    }

    encodeRender(commandEncoder) {
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1.0 }
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBuffer]);
        renderPass.draw(3);
        renderPass.end();
    }

    frame() {
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeCompute(commandEncoder);
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    // Recomputes the diagram, then brute-forces the nearest seed for every pixel on the GPU and
    // reads back the two mismatch counters
    async exactCheck() {
        if (this.checking || this.destroyed) return null;
        this.checking = true;
        const config = {
            seeds: this.seeds.length,
            metric: this.metric,
            plusOne: this.plusOne,
            passes: this.seeds.length > 0 ? this.steps().length : 0
        };
        try {
            const t0 = performance.now();
            this.device.queue.writeBuffer(this.counterBuffer, 0, new Uint32Array(2));
            const commandEncoder = this.device.createCommandEncoder();
            this.encodeCompute(commandEncoder);
            this.encodeRender(commandEncoder);
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.exactPipeline);
            pass.setBindGroup(0, this.exactBindGroups[this.currentBuffer], [0]);
            pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
            pass.end();
            commandEncoder.copyBufferToBuffer(this.counterBuffer, 0, this.readbackBuffer, 0, 8);
            this.device.queue.submit([commandEncoder.finish()]);

            await this.readbackBuffer.mapAsync(GPUMapMode.READ);
            const counts = new Uint32Array(this.readbackBuffer.getMappedRange().slice(0));
            this.readbackBuffer.unmap();
            this.lastCheck = { ...config, wrong: counts[0], differ: counts[1], ms: performance.now() - t0 };
            return this.lastCheck;
        } finally {
            this.checking = false;
        }
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.lastFrameTime = this.lastSampleTime;
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
    }

    // Seeds drift in straight lines and bounce off the edges; the dragged seed follows the mouse
    moveSeeds(dt) {
        const w = this.width;
        const h = this.height;
        this.seeds.forEach((s, i) => {
            if (i === this.dragIndex) return;
            s.x += s.vx * dt;
            s.y += s.vy * dt;
            if (s.x < 0) { s.x = -s.x; s.vx = Math.abs(s.vx); }
            if (s.x >= w) { s.x = 2 * w - s.x - 0.02; s.vx = -Math.abs(s.vx); }
            if (s.y < 0) { s.y = -s.y; s.vy = Math.abs(s.vy); }
            if (s.y >= h) { s.y = 2 * h - s.y - 0.02; s.vy = -Math.abs(s.vy); }
            s.x = Math.min(Math.max(s.x, 0), w - 0.01);
            s.y = Math.min(Math.max(s.y, 0), h - 0.01);
        });
    }

    animate() {
        if (!this.running) return;

        const now = performance.now();
        const dt = Math.min((now - this.lastFrameTime) / 1000, 0.1);
        this.lastFrameTime = now;
        this.moveSeeds(dt);
        this.frame();

        this.framesSinceSample++;
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
        this.canvas.removeEventListener('contextmenu', this.onContextMenu);
        window.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.idBuffers.forEach(b => b.destroy());
        this.seedBuffer.destroy();
        this.paramsBuffer.destroy();
        this.counterBuffer.destroy();
        this.readbackBuffer.destroy();
    }
}

export async function initVoronoiJFA(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-voronoi');
    const output = document.getElementById('output-voronoi');

    const voronoi = new VoronoiJFA(device, canvas);
    const width = voronoi.width;
    const height = voronoi.height;

    const countSlider = document.getElementById('voronoi-count');
    const countValue = document.getElementById('voronoi-count-value');
    const readCount = () => Math.min(Math.max(parseInt(countSlider.value) || 2, 2), MAX_SEEDS);

    let editsPending = false;
    voronoi.onStats = () => updateOutput();
    // Coalesce info-panel updates from drags into one per frame (and none while animating)
    voronoi.onEdit = () => {
        if (voronoi.running || editsPending) return;
        editsPending = true;
        requestAnimationFrame(() => {
            editsPending = false;
            updateOutput();
        });
    };

    const animateButton = document.getElementById('btn-voronoi-animate');
    animateButton.addEventListener('click', () => {
        if (voronoi.running) voronoi.pause(); else voronoi.start();
        animateButton.textContent = voronoi.running ? '⏸ Pause' : '▶ Animate';
        updateOutput();
    });

    document.getElementById('btn-voronoi-random').addEventListener('click', () => {
        voronoi.generate(readCount(), 'random');
        showLayout();
    });

    document.getElementById('btn-voronoi-clear').addEventListener('click', () => {
        voronoi.clear();
        updateOutput();
    });

    countSlider.addEventListener('input', () => {
        countValue.textContent = readCount();
        voronoi.generate(readCount());
        updateOutput();
    });

    document.getElementById('voronoi-metric').addEventListener('change', (e) => {
        voronoi.metric = Math.min(Math.max(parseInt(e.target.value) || 0, 0), METRICS.length - 1);
        voronoi.edited();
        updateOutput();
    });

    document.getElementById('voronoi-view').addEventListener('change', (e) => {
        voronoi.viewMode = Math.min(Math.max(parseInt(e.target.value) || 0, 0), VIEWS.length - 1);
        voronoi.edited();
        updateOutput();
    });

    const plusOneBox = document.getElementById('voronoi-jfa1');
    plusOneBox.checked = voronoi.plusOne;
    plusOneBox.addEventListener('change', (e) => {
        voronoi.plusOne = e.target.checked;
        voronoi.edited();
        updateOutput();
    });

    const checkButton = document.getElementById('btn-voronoi-check');
    checkButton.addEventListener('click', async () => {
        checkButton.disabled = true;
        try {
            await voronoi.exactCheck();
        } catch (error) {
            voronoi.lastCheck = { error: error.message };
        }
        checkButton.disabled = false;
        updateOutput();
    });

    const layoutButtons = document.querySelectorAll('.voronoi-layout');
    layoutButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!LAYOUTS[btn.dataset.layout]) return;
            voronoi.generate(readCount(), btn.dataset.layout);
            showLayout();
        });
    });

    function showLayout() {
        layoutButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.layout === voronoi.layout));
        updateOutput();
    }

    function describeCheck() {
        const c = voronoi.lastCheck;
        if (!c) return '• Press Exact check to compare JFA with brute force on the GPU';
        if (c.error) return `• <span class="error">Exact check failed: ${c.error}</span>`;
        const pct = (n) => (100 * n / (width * height)).toFixed(4);
        return `• Last exact check (${c.seeds} seeds, ${METRICS[c.metric]}, ${c.plusOne ? 'JFA+1' : 'JFA'}, ${c.passes} passes):
  ${c.wrong.toLocaleString()} pixels (${pct(c.wrong)}%) got a seed farther than the nearest one
  ${c.differ.toLocaleString()} pixels (${pct(c.differ)}%) got a different seed index (includes equidistant ties)
  Round trip ${c.ms.toFixed(1)} ms for ${(width * height * c.seeds).toLocaleString()} brute-force distance tests`;
    }

    function updateOutput() {
        const n = voronoi.seeds.length;
        const steps = voronoi.steps();
        const passes = n > 0 ? steps.length : 0;

        output.innerHTML = `<span class="success">✓ Voronoi (Jump Flooding) Active</span>

<span class="info">Configuration:</span>
• Grid: ${width} × ${height} (${(width * height).toLocaleString()} pixels, one u32 seed index each)
• Seeds: ${n} / ${MAX_SEEDS} · layout ${LAYOUTS[voronoi.layout].name}
• Metric: ${METRICS[voronoi.metric]} · View: ${VIEWS[voronoi.viewMode]}
• Flood steps: ${steps.join(', ')}${voronoi.plusOne ? ' (JFA+1)' : ''}
• Status: ${voronoi.running ? '<span class="success">Animating</span>' : '<span class="info">Paused</span>'}

<span class="info">Algorithm (Jump Flooding, Rong &amp; Tan 2006):</span>
• Seed pass: each seed writes its index into its pixel (atomicMin, lowest index wins, so seeds
  sharing a pixel lose all but one; the exact check counts that loss too)
• Flood pass with step k: each pixel checks itself and 8 neighbors at ±k and keeps the closest seed
• k = N/2, N/4, …, 1 for N = ${2 * steps[0]}: ${Math.log2(2 * steps[0])} passes instead of comparing every pixel with every seed
• Two ping-pong storage buffers; each pass reads its step from its own dynamic-offset uniform slot
• Borders: a pixel whose 4-neighbor has another seed; distance view: iso-lines every 16 pixels

<span class="info">Accuracy:</span>
${describeCheck()}

<span class="info">Performance:</span>
• Frame rate: ${voronoi.running ? voronoi.fps.toFixed(0) : 0} FPS
• Dispatches per frame: ${n > 0 ? passes + 2 : 1} (clear, seed, ${passes} flood) + 1 draw
• Pixel updates per frame: ${(width * height * passes).toLocaleString()} (${(9 * width * height * passes).toLocaleString()} neighbor lookups)`;
    }

    countValue.textContent = readCount();
    voronoi.generate(readCount(), 'random');
    showLayout();
    return voronoi;
}
