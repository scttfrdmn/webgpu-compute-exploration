// Procedural noise: a compute shader evaluates 3D Perlin or simplex noise, summed as fractal
// Brownian motion and optionally domain-warped, for every pixel into a storage buffer. A
// full-screen fragment pass then colors that field as terrain, clouds, marble or grayscale.
//
// Uniform layout (struct Params, 56 bytes; shared by the compute and render shaders):
//   offset  0  size          vec2<u32>  field width, height (= canvas pixels) -> Uint32 view [0], [1]
//   offset  8  center        vec2<f32>  view center in world units            -> Float32 view [2], [3]
//   offset 16  unitsPerPixel f32        world units per pixel (zoom)          -> Float32 view [4]
//   offset 20  frequency     f32        noise cycles per world unit           -> Float32 view [5]
//   offset 24  time          f32        third noise coordinate                -> Float32 view [6]
//   offset 28  octaves       u32        1..MAX_OCTAVES (clamped in WGSL too)  -> Uint32 view [7]
//   offset 32  lacunarity    f32        frequency multiplier per octave       -> Float32 view [8]
//   offset 36  gain          f32        amplitude multiplier per octave       -> Float32 view [9]
//   offset 40  warp          f32        domain-warp strength, 0 = off         -> Float32 view [10]
//   offset 44  noiseType     u32        0 = Perlin, 1 = simplex               -> Uint32 view [11]
//   offset 48  ridged        u32        1 = ridged fBm                        -> Uint32 view [12]
//   offset 52  colorMode     u32        0 terrain, 1 clouds, 2 marble, 3 gray -> Uint32 view [13]
const PARAMS_SIZE = 56;
const MAX_OCTAVES = 8;

const paramsStruct = `
struct Params {
    size: vec2<u32>,
    center: vec2<f32>,
    unitsPerPixel: f32,
    frequency: f32,
    time: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    warp: f32,
    noiseType: u32,
    ridged: u32,
    colorMode: u32,
}
`;

const computeShader = paramsStruct + `
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const MAX_OCTAVES: u32 = ${MAX_OCTAVES}u;

// PCG-based 3D hash (Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT 2020).
// Replaces Perlin's permutation table: every lattice corner gets 32 well-mixed bits.
fn pcg3d(v0: vec3<u32>) -> vec3<u32> {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn hashCorner(c: vec3<i32>) -> u32 {
    return pcg3d(bitcast<vec3<u32>>(c)).x;
}

// Gradient selection from Perlin 2002, "Improving Noise": the top 4 hash bits pick one of
// the 12 cube-edge directions (1,1,0), (1,0,1), (0,1,1) with signs (4 of them repeated),
// and the function returns dot(gradient, d) without a table.
fn grad(h: u32, d: vec3<f32>) -> f32 {
    let g = h >> 28u;
    let u = select(d.y, d.x, g < 8u);
    let v = select(select(d.z, d.x, g == 12u || g == 14u), d.y, g < 4u);
    return select(-u, u, (g & 1u) == 0u) + select(-v, v, (g & 2u) == 0u);
}

// Quintic fade 6t^5 - 15t^4 + 10t^3 (Perlin 2002): zero first and second derivatives at
// the lattice, so the noise and its gradient are continuous across cell boundaries.
fn fade(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Classic gradient noise (Perlin 2002): dot each of the 8 surrounding corners' gradient with
// the offset to that corner, then blend trilinearly with the fade curve. Range is about [-1, 1].
fn perlin3(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let i = vec3<i32>(cell);
    let f = p - cell;
    let u = fade(f);

    let n000 = grad(hashCorner(i), f);
    let n100 = grad(hashCorner(i + vec3<i32>(1, 0, 0)), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = grad(hashCorner(i + vec3<i32>(0, 1, 0)), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = grad(hashCorner(i + vec3<i32>(1, 1, 0)), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = grad(hashCorner(i + vec3<i32>(0, 0, 1)), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = grad(hashCorner(i + vec3<i32>(1, 0, 1)), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = grad(hashCorner(i + vec3<i32>(0, 1, 1)), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = grad(hashCorner(i + vec3<i32>(1, 1, 1)), f - vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

// One simplex corner: radial falloff (0.5 - r^2)^4 times the gradient ramp. The radius
// sqrt(0.5) reaches exactly to the opposite face of each simplex, so the sum has no seams
// (Gustavson's 0.6 leaves small discontinuities in 3D).
fn simplexCorner(h: u32, d: vec3<f32>) -> f32 {
    let t = 0.5 - dot(d, d);
    if (t <= 0.0) {
        return 0.0;
    }
    let t2 = t * t;
    return t2 * t2 * grad(h, d);
}

// Simplex noise (Perlin 2001; following Gustavson, "Simplex noise demystified", 2005).
// Skew space so the cube lattice becomes tetrahedra, find the tetrahedron containing p by
// ranking the offset's components, and sum 4 corner contributions instead of 8.
const F3: f32 = 1.0 / 3.0;   // skew factor (sqrt(n + 1) - 1) / n for n = 3
const G3: f32 = 1.0 / 6.0;   // unskew factor (1 - 1 / sqrt(n + 1)) / n
const SIMPLEX_SCALE: f32 = 72.0;  // maps the sum to about [-1, 1] (measured on the CPU)

fn simplex3(p: vec3<f32>) -> f32 {
    let s = (p.x + p.y + p.z) * F3;
    let cell = floor(p + vec3<f32>(s));
    let t = (cell.x + cell.y + cell.z) * G3;
    let x0 = p - (cell - vec3<f32>(t));

    // Which of the six tetrahedra in the skewed cube: order of x0's components
    var i1: vec3<i32>;
    var i2: vec3<i32>;
    if (x0.x >= x0.y) {
        if (x0.y >= x0.z) {
            i1 = vec3<i32>(1, 0, 0); i2 = vec3<i32>(1, 1, 0);
        } else if (x0.x >= x0.z) {
            i1 = vec3<i32>(1, 0, 0); i2 = vec3<i32>(1, 0, 1);
        } else {
            i1 = vec3<i32>(0, 0, 1); i2 = vec3<i32>(1, 0, 1);
        }
    } else {
        if (x0.y < x0.z) {
            i1 = vec3<i32>(0, 0, 1); i2 = vec3<i32>(0, 1, 1);
        } else if (x0.x < x0.z) {
            i1 = vec3<i32>(0, 1, 0); i2 = vec3<i32>(0, 1, 1);
        } else {
            i1 = vec3<i32>(0, 1, 0); i2 = vec3<i32>(1, 1, 0);
        }
    }

    let i = vec3<i32>(cell);
    let x1 = x0 - vec3<f32>(i1) + vec3<f32>(G3);
    let x2 = x0 - vec3<f32>(i2) + vec3<f32>(2.0 * G3);
    let x3 = x0 - vec3<f32>(1.0) + vec3<f32>(3.0 * G3);

    let n = simplexCorner(hashCorner(i), x0)
          + simplexCorner(hashCorner(i + i1), x1)
          + simplexCorner(hashCorner(i + i2), x2)
          + simplexCorner(hashCorner(i + vec3<i32>(1)), x3);
    return SIMPLEX_SCALE * n;
}

fn baseNoise(p: vec3<f32>) -> f32 {
    if (params.noiseType == 1u) {
        return simplex3(p);
    }
    return perlin3(p);
}

// Fractal Brownian motion: sum octaves of noise, each at lacunarity x the frequency and
// gain x the amplitude of the previous one. Dividing by the summed amplitudes keeps the
// result in the base noise's range, about [-1, 1]. Ridged mode folds each octave as
// 2(1 - |n|)^2 - 1, turning zero crossings into sharp crests. Each octave is shifted so
// the lattices of different octaves do not line up at the origin.
fn fbm(p0: vec3<f32>) -> f32 {
    let octaves = clamp(params.octaves, 1u, MAX_OCTAVES);
    var p = p0;
    var amplitude = 1.0;
    var total = 0.0;
    var norm = 0.0;
    for (var o = 0u; o < MAX_OCTAVES; o++) {
        if (o >= octaves) {
            break;
        }
        var n = baseNoise(p);
        if (params.ridged != 0u) {
            let r = 1.0 - abs(n);
            n = 2.0 * r * r - 1.0;
        }
        total += amplitude * n;
        norm += amplitude;
        amplitude *= params.gain;
        p = p * params.lacunarity + vec3<f32>(19.1, 7.3, 3.7);
    }
    return total / norm;
}

// Domain warping (Quilez, "Domain Warping"): displace the lookup point by a vector made
// of two more fBm evaluations, f(p + warp * (fbm(p), fbm(p + offset))).
fn warpedFbm(p: vec3<f32>) -> f32 {
    if (params.warp <= 0.0) {
        return fbm(p);
    }
    let q = vec2<f32>(fbm(p), fbm(p + vec3<f32>(5.2, 1.3, 0.0)));
    return fbm(p + vec3<f32>(params.warp * q, 0.0));
}

const MARBLE_STRIPES: f32 = 5.0;     // stripe frequency along (x + 0.5 y)
const MARBLE_TURBULENCE: f32 = 8.0;  // how far the noise bends the stripes

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }
    // Pixel -> world (y grows downward) -> noise space; time is the third coordinate, offset
    // half a cell so the first octave does not start on a lattice plane, where Perlin noise
    // is zero at every lattice point
    let pixel = vec2<f32>(gid.xy) + vec2<f32>(0.5);
    let world = params.center + (pixel - 0.5 * vec2<f32>(params.size)) * params.unitsPerPixel;
    let p = vec3<f32>(world * params.frequency, params.time + 0.5);

    var value = warpedFbm(p);
    if (params.colorMode == 2u) {
        // Marble (Perlin 1985): a sine of the coordinate, phase-shifted by the noise
        value = sin(MARBLE_STRIPES * (p.x + 0.5 * p.y) + MARBLE_TURBULENCE * value);
    }
    field[gid.y * params.size.x + gid.x] = value;
}
`;

const renderShader = paramsStruct + `
@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

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

fn valueAt(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(params.size.x) - 1));
    let cy = u32(clamp(y, 0, i32(params.size.y) - 1));
    return field[cy * params.size.x + cx];
}

const SEA_LEVEL: f32 = 0.0;
const RELIEF: f32 = 0.08;   // exaggeration of slopes for hillshading

// Height bands: deep/shallow water, sand, grass, rock, snow. Water is flattened to sea level
// for shading; land gets Lambert lighting from a northwest sun using central differences.
fn terrainColor(x: i32, y: i32, h: f32) -> vec3<f32> {
    var color: vec3<f32>;
    if (h < SEA_LEVEL) {
        let depth = clamp((SEA_LEVEL - h) / 0.35, 0.0, 1.0);
        return mix(vec3<f32>(0.20, 0.50, 0.70), vec3<f32>(0.03, 0.10, 0.28), depth);
    } else if (h < 0.03) {
        color = vec3<f32>(0.85, 0.80, 0.58);
    } else if (h < 0.28) {
        color = mix(vec3<f32>(0.32, 0.58, 0.24), vec3<f32>(0.16, 0.38, 0.15), (h - 0.03) / 0.25);
    } else if (h < 0.45) {
        color = mix(vec3<f32>(0.42, 0.37, 0.30), vec3<f32>(0.52, 0.49, 0.46), (h - 0.28) / 0.17);
    } else {
        color = vec3<f32>(0.95, 0.96, 0.98);
    }

    // Slope in height per world unit, so shading does not change with zoom
    let hl = max(valueAt(x - 1, y), SEA_LEVEL);
    let hr = max(valueAt(x + 1, y), SEA_LEVEL);
    let hu = max(valueAt(x, y - 1), SEA_LEVEL);
    let hd = max(valueAt(x, y + 1), SEA_LEVEL);
    let k = RELIEF / (2.0 * params.unitsPerPixel);
    let normal = normalize(vec3<f32>((hl - hr) * k, (hu - hd) * k, 1.0));
    let sun = normalize(vec3<f32>(-1.0, -1.0, 1.2));
    let light = 0.35 + 0.75 * max(dot(normal, sun), 0.0);
    return color * light;
}

// Coverage from the noise; the thick parts darken slightly like cloud undersides
fn cloudColor(y: i32, n: f32) -> vec3<f32> {
    let sky = mix(vec3<f32>(0.16, 0.36, 0.70), vec3<f32>(0.52, 0.72, 0.92), f32(y) / f32(params.size.y));
    let coverage = smoothstep(-0.15, 0.35, n);
    let cloud = mix(vec3<f32>(1.0), vec3<f32>(0.78, 0.80, 0.86), smoothstep(0.25, 0.7, n));
    return mix(sky, cloud, coverage);
}

// Value is sin(...) in [-1, 1]: cream stone with soft gray banding, and thin dark veins
// where the value is close to -1
fn marbleColor(v: f32) -> vec3<f32> {
    let stone = mix(vec3<f32>(0.95, 0.93, 0.88), vec3<f32>(0.70, 0.70, 0.72), 0.5 - 0.5 * v);
    let vein = 1.0 - smoothstep(-0.995, -0.93, v);
    return mix(stone, vec3<f32>(0.18, 0.19, 0.24), vein);
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let x = i32(frag.position.x);
    let y = i32(frag.position.y);
    let v = valueAt(x, y);

    var color: vec3<f32>;
    if (params.colorMode == 0u) {
        color = terrainColor(x, y, v);
    } else if (params.colorMode == 1u) {
        color = cloudColor(y, v);
    } else if (params.colorMode == 2u) {
        color = marbleColor(v);
    } else {
        color = vec3<f32>(clamp(0.5 + 0.5 * v, 0.0, 1.0));
    }
    return vec4<f32>(color, 1.0);
}
`;

const NOISE_TYPES = ['Perlin (gradient)', 'Simplex'];
const COLOR_MODES = ['Terrain', 'Clouds', 'Marble', 'Grayscale'];

// Each combination was rendered with a CPU port of these shaders and checked by eye
const PRESETS = {
    terrain:   { name: 'Terrain',   noiseType: 0, frequency: 2.5, octaves: 7, lacunarity: 2.0, gain: 0.5,  warp: 0.4, ridged: false, colorMode: 0 },
    mountains: { name: 'Mountains', noiseType: 1, frequency: 1.5, octaves: 7, lacunarity: 2.1, gain: 0.45, warp: 0.0, ridged: true,  colorMode: 0 },
    clouds:    { name: 'Clouds',    noiseType: 0, frequency: 2.0, octaves: 7, lacunarity: 2.0, gain: 0.5,  warp: 0.3, ridged: false, colorMode: 1 },
    marble:    { name: 'Marble',    noiseType: 0, frequency: 2.0, octaves: 6, lacunarity: 2.0, gain: 0.5,  warp: 0.0, ridged: false, colorMode: 2 },
    gray:      { name: 'Warped',    noiseType: 1, frequency: 1.5, octaves: 5, lacunarity: 2.0, gain: 0.5,  warp: 3.0, ridged: false, colorMode: 3 },
};

const TIME_SPEED = 0.15;   // noise units of the time axis per second

class NoiseField {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.width = canvas.width;
        this.height = canvas.height;
        this.running = false;
        this.destroyed = false;

        this.applyPreset('terrain');
        this.resetView();
        this.time = 0;

        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;
        this.lastFrameTime = 0;
        this.renderPending = false;

        this.params = new ArrayBuffer(PARAMS_SIZE);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);

        this.setupBuffers();
        this.setupComputePipeline();
        this.setupRenderPipeline();
        this.setupCanvas();
        this.setupMouseControls();
    }

    // Copies a preset's noise and color settings (not the view)
    applyPreset(key) {
        const { name, ...settings } = PRESETS[key];
        Object.assign(this, settings);
        this.preset = key;
    }

    // The field spans one world unit vertically, centered on the origin
    resetView() {
        this.center = [0, 0];
        this.unitsPerPixel = 1 / this.height;
    }

    setupBuffers() {
        this.fieldBuffer = this.device.createBuffer({
            size: this.width * this.height * 4, // f32 per pixel
            usage: GPUBufferUsage.STORAGE
        });
        this.paramsBuffer = this.device.createBuffer({
            size: PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    setupComputePipeline() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'cs_main' }
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.fieldBuffer } },
                { binding: 1, resource: { buffer: this.paramsBuffer } }
            ]
        });
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
        this.renderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.fieldBuffer } },
                { binding: 1, resource: { buffer: this.paramsBuffer } }
            ]
        });
    }

    setupCanvas() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });
    }

    setupMouseControls() {
        // The canvas is CSS-scaled, so convert client coordinates to canvas pixels
        const toPixels = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            return [
                (e.clientX - rect.left) * this.canvas.width / rect.width,
                (e.clientY - rect.top) * this.canvas.height / rect.height
            ];
        };

        this.dragging = false;
        this.lastPointer = [0, 0];

        this.onMouseDown = (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            this.dragging = true;
            this.lastPointer = toPixels(e);
        };
        this.onMouseMove = (e) => {
            if (!this.dragging) return;
            const p = toPixels(e);
            this.center[0] -= (p[0] - this.lastPointer[0]) * this.unitsPerPixel;
            this.center[1] -= (p[1] - this.lastPointer[1]) * this.unitsPerPixel;
            this.lastPointer = p;
            this.requestRender();
        };
        this.onMouseUp = () => {
            this.dragging = false;
        };
        // Zoom about the cursor: keep the world point under it fixed
        this.onWheel = (e) => {
            e.preventDefault();
            const [px, py] = toPixels(e);
            const offX = px - this.width / 2;
            const offY = py - this.height / 2;
            const worldX = this.center[0] + offX * this.unitsPerPixel;
            const worldY = this.center[1] + offY * this.unitsPerPixel;
            const base = 1 / this.height;
            const zoomed = this.unitsPerPixel * Math.exp(e.deltaY * 0.0015);
            this.unitsPerPixel = Math.min(Math.max(zoomed, base / 64), base * 16);
            this.center[0] = worldX - offX * this.unitsPerPixel;
            this.center[1] = worldY - offY * this.unitsPerPixel;
            this.requestRender();
            if (this.onViewChange) this.onViewChange();
        };

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
    }

    // While paused, coalesce control and mouse changes into one compute + render per frame
    requestRender() {
        if (this.running || this.renderPending || this.destroyed) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            if (!this.running && !this.destroyed) this.frame();
        });
    }

    // Writes the uniform buffer; see the layout table at the top of the file
    writeParams() {
        this.paramsU32[0] = this.width;
        this.paramsU32[1] = this.height;
        this.paramsF32[2] = this.center[0];
        this.paramsF32[3] = this.center[1];
        this.paramsF32[4] = this.unitsPerPixel;
        this.paramsF32[5] = this.frequency;
        this.paramsF32[6] = this.time;
        this.paramsU32[7] = Math.min(Math.max(this.octaves, 1), MAX_OCTAVES);
        this.paramsF32[8] = this.lacunarity;
        this.paramsF32[9] = this.gain;
        this.paramsF32[10] = this.warp;
        this.paramsU32[11] = this.noiseType;
        this.paramsU32[12] = this.ridged ? 1 : 0;
        this.paramsU32[13] = this.colorMode;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params);
    }

    // One compute pass over the field, then one full-screen draw
    frame() {
        this.writeParams();
        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        computePass.end();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    // Noise evaluations per pixel: one fBm, or three with domain warping
    noiseCallsPerPixel() {
        return this.octaves * (this.warp > 0 ? 3 : 1);
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

    animate() {
        if (!this.running) return;

        const now = performance.now();
        const dt = Math.min((now - this.lastFrameTime) / 1000, 0.1);
        this.lastFrameTime = now;
        this.time += dt * TIME_SPEED;
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
        window.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
        this.fieldBuffer.destroy();
        this.paramsBuffer.destroy();
    }
}

export async function initNoise(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-noise');
    const output = document.getElementById('output-noise');

    const noise = new NoiseField(device, canvas);
    noise.onStats = () => updateOutput();
    noise.onViewChange = () => updateOutput();

    const animateButton = document.getElementById('btn-noise-animate');
    animateButton.addEventListener('click', () => {
        if (noise.running) noise.pause(); else noise.start();
        animateButton.textContent = noise.running ? '⏸ Pause' : '▶ Animate';
        updateOutput();
    });

    document.getElementById('btn-noise-reset-view').addEventListener('click', () => {
        noise.resetView();
        noise.requestRender();
        updateOutput();
    });

    const typeSelect = document.getElementById('noise-type');
    const colorSelect = document.getElementById('noise-color');
    const ridgedBox = document.getElementById('noise-ridged');
    const presetButtons = document.querySelectorAll('.noise-preset');

    // Sliders: id suffix, property, parser, clamp range, display format
    const sliders = [
        { id: 'frequency',  key: 'frequency',  parse: parseFloat, min: 0.5, max: 10,          show: v => v.toFixed(1) },
        { id: 'octaves',    key: 'octaves',    parse: v => parseInt(v, 10), min: 1, max: MAX_OCTAVES, show: v => String(v) },
        { id: 'lacunarity', key: 'lacunarity', parse: parseFloat, min: 1.5, max: 3,           show: v => v.toFixed(2) },
        { id: 'gain',       key: 'gain',       parse: parseFloat, min: 0.2, max: 0.8,         show: v => v.toFixed(2) },
        { id: 'warp',       key: 'warp',       parse: parseFloat, min: 0,   max: 4,           show: v => v.toFixed(1) },
    ].map(s => ({
        ...s,
        input: document.getElementById(`noise-${s.id}`),
        label: document.getElementById(`noise-${s.id}-value`)
    }));

    function showParams() {
        for (const s of sliders) {
            s.input.value = noise[s.key];
            s.label.textContent = s.show(noise[s.key]);
        }
        typeSelect.value = String(noise.noiseType);
        colorSelect.value = String(noise.colorMode);
        ridgedBox.checked = noise.ridged;
        presetButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === noise.preset));
    }

    function changed() {
        noise.preset = null;
        showParams();
        noise.requestRender();
        updateOutput();
    }

    for (const s of sliders) {
        s.input.addEventListener('input', (e) => {
            const v = s.parse(e.target.value);
            if (Number.isNaN(v)) return;
            noise[s.key] = Math.min(Math.max(v, s.min), s.max);
            changed();
        });
    }

    typeSelect.addEventListener('change', (e) => {
        noise.noiseType = Math.min(Math.max(parseInt(e.target.value, 10) || 0, 0), NOISE_TYPES.length - 1);
        changed();
    });

    colorSelect.addEventListener('change', (e) => {
        noise.colorMode = Math.min(Math.max(parseInt(e.target.value, 10) || 0, 0), COLOR_MODES.length - 1);
        changed();
    });

    ridgedBox.addEventListener('change', (e) => {
        noise.ridged = e.target.checked;
        changed();
    });

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!PRESETS[btn.dataset.preset]) return;
            noise.applyPreset(btn.dataset.preset);
            showParams();
            noise.requestRender();
            updateOutput();
        });
    });

    function updateOutput() {
        const presetName = noise.preset ? PRESETS[noise.preset].name : 'Custom';
        const pixels = noise.width * noise.height;
        const calls = pixels * noise.noiseCallsPerPixel();
        const zoom = (1 / noise.height) / noise.unitsPerPixel;
        const callsPerSecond = noise.running ? calls * noise.fps : 0;

        output.innerHTML = `<span class="success">✓ Procedural Noise Active</span>

<span class="info">Configuration:</span>
• Field: ${noise.width} × ${noise.height} (${pixels.toLocaleString()} pixels, one f32 each)
• Preset: ${presetName} · ${NOISE_TYPES[noise.noiseType]} noise, ${COLOR_MODES[noise.colorMode]} colors
• fBm: ${noise.octaves} octave${noise.octaves === 1 ? '' : 's'}, lacunarity ${noise.lacunarity.toFixed(2)}, persistence ${noise.gain.toFixed(2)}${noise.ridged ? ', ridged' : ''}
• Frequency ${noise.frequency.toFixed(1)} · domain warp ${noise.warp.toFixed(1)} · zoom ${zoom.toFixed(2)}×
• Status: ${noise.running ? '<span class="success">Animating</span>' : '<span class="info">Paused</span>'} (time ${noise.time.toFixed(2)})

<span class="info">Algorithm:</span>
• Perlin: gradients at the 8 cube corners, quintic fade, trilinear blend (Perlin 2002)
• Simplex: skewed lattice, 4 tetrahedron corners with (0.5 − r²)⁴ falloff (Gustavson 2005)
• PCG hash of the lattice coordinates replaces the permutation table
• fBm = Σ gainⁱ · noise(lacunarityⁱ · p) / Σ gainⁱ; warp: fbm(p + w · (fbm(p), fbm(p + c)))
• Time is the third noise coordinate, so animation is smooth

<span class="info">Performance:</span>
• Frame rate: ${noise.running ? noise.fps.toFixed(0) : 0} FPS
• Noise evaluations per frame: ${calls.toLocaleString()} (${noise.noiseCallsPerPixel()} per pixel)
• Per second: ${noise.running ? (callsPerSecond / 1e6).toFixed(0) + ' million' : '—'}
• One compute dispatch + one draw per frame; no CPU readback`;
    }

    showParams();
    noise.frame();
    updateOutput();
    return noise;
}
