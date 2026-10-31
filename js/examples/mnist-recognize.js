// MNIST, part 2: draw a digit and classify it on the GPU with the shared CNN (mnist-model.js),
// using the weights cached in data/mnist/weights.bin or the weights handed over by example 21.
//
// Preprocessing runs on the CPU (it is 78k pixels once per inference, far less work than a
// dispatch round trip), the way the MNIST digits were made: the drawing's bounding box is scaled
// to fit 20×20 (aspect ratio kept, area-averaged, so strokes become gray at the edges), placed in
// a 28×28 frame and shifted so its center of mass sits at the frame's center (14, 14).
//
// Inference is one submit: the 784 floats are written into the work buffer, the forward kernels
// run with batch 1 (mode 2: no gather, no labels), cs_mapmax finds each activation map's maximum
// for display, a render pass draws the input and all 24 ReLU maps straight from the work buffer,
// and the 10 softmax probabilities (40 bytes) are copied back for the confidence bars. One
// inference is in flight at a time; while drawing, inference runs at most every LIVE_INTERVAL_MS.
import { MNISTGPU, NUM_CLASSES, IMG_PIXELS, workLayout, loadDataset, loadCachedWeights, sharedWeights } from './mnist-model.js';

const DRAW_SIZE = 280;
const BRUSH_WIDTH = 22;          // ≈ 2.4 px after scaling a 180-px-tall digit to 20 px, like MNIST strokes
const LIVE_INTERVAL_MS = 120;
const INK_THRESHOLD = 0.1;       // pixels brighter than this count for the bounding box

// Converts a w×h grayscale drawing (values in [0, 1], white digit on black) to the 28×28 MNIST
// format; returns null for an empty drawing
export function preprocessDrawing(pixels, w, h) {
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (pixels[y * w + x] > INK_THRESHOLD) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    if (maxX < 0) return null;

    // Scale the bounding box to fit 20×20, averaging the source area under each target pixel
    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const scale = 20 / Math.max(bw, bh);
    const nw = Math.max(1, Math.min(20, Math.round(bw * scale)));
    const nh = Math.max(1, Math.min(20, Math.round(bh * scale)));
    const sample = (x, y) => (x >= 0 && y >= 0 && x < w && y < h) ? pixels[y * w + x] : 0;
    const box = new Float32Array(nw * nh);
    for (let ty = 0; ty < nh; ty++) {
        const sy0 = minY + ty / scale;
        const sy1 = minY + (ty + 1) / scale;
        for (let tx = 0; tx < nw; tx++) {
            const sx0 = minX + tx / scale;
            const sx1 = minX + (tx + 1) / scale;
            let sum = 0;
            for (let yy = Math.floor(sy0); yy < Math.ceil(sy1); yy++) {
                const wy = Math.min(yy + 1, sy1) - Math.max(yy, sy0);
                if (wy <= 0) continue;
                for (let xx = Math.floor(sx0); xx < Math.ceil(sx1); xx++) {
                    const wx = Math.min(xx + 1, sx1) - Math.max(xx, sx0);
                    if (wx > 0) sum += sample(xx, yy) * wx * wy;
                }
            }
            box[ty * nw + tx] = Math.min(sum / ((sx1 - sx0) * (sy1 - sy0)), 1);
        }
    }

    // Center of mass of the box placed in the middle of the frame, then an integer shift that moves
    // it to (14, 14) without pushing the box out of the frame
    const ox = Math.floor((28 - nw) / 2);
    const oy = Math.floor((28 - nh) / 2);
    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < nh; y++) {
        for (let x = 0; x < nw; x++) {
            const v = box[y * nw + x];
            mass += v;
            mx += v * (ox + x + 0.5);
            my += v * (oy + y + 0.5);
        }
    }
    let dx = 0, dy = 0;
    if (mass > 0) {
        dx = Math.min(Math.max(Math.round(14 - mx / mass), -ox), 28 - ox - nw);
        dy = Math.min(Math.max(Math.round(14 - my / mass), -oy), 28 - oy - nh);
    }
    const out = new Float32Array(IMG_PIXELS);
    for (let y = 0; y < nh; y++) {
        for (let x = 0; x < nw; x++) out[(oy + dy + y) * 28 + ox + dx + x] = box[y * nw + x];
    }
    return out;
}

// Display layout of the maps canvas (pixels): the 28×28 input, the 8 conv1 maps (24×24, 4×2 grid)
// and the 16 conv2 maps (8×8, 4×4 grid), each map normalized by its own maximum
const MAPS_W = 570;
const MAPS_H = 232;
const L = workLayout(1);

const mapsShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> work: array<f32>;

const W_X: u32 = ${L.X}u;
const W_A1: u32 = ${L.A1}u;
const W_A2: u32 = ${L.A2}u;
const W_MAPMAX: u32 = ${L.MAPMAX}u;

const IN_ORIGIN: vec2<f32> = vec2<f32>(16.0, 60.0);
const IN_SCALE: f32 = 4.0;          // 28 cells × 4 px = 112 px
const C1_ORIGIN: vec2<f32> = vec2<f32>(160.0, 64.0);
const C1_SCALE: f32 = 2.0;          // 24 cells × 2 px = 48 px
const C1_GAP: f32 = 8.0;
const C2_ORIGIN: vec2<f32> = vec2<f32>(408.0, 43.0);
const C2_SCALE: f32 = 4.0;          // 8 cells × 4 px = 32 px
const C2_GAP: f32 = 6.0;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(corners[vi], 0.0, 1.0);
}

// Sequential ramp for activations: background -> blue -> near white
fn ramp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.075, 0.094, 0.125);
    let c1 = vec3<f32>(0.129, 0.588, 0.827);
    let c2 = vec3<f32>(0.900, 0.970, 1.000);
    if (t < 0.6) {
        return mix(c0, c1, t / 0.6);
    }
    return mix(c1, c2, (t - 0.6) / 0.4);
}

// Which map of a grid the pixel falls in (x, y = map column/row, z = cell index in the map) or
// none (w = 0)
fn gridCell(p: vec2<f32>, origin: vec2<f32>, scale: f32, cells: f32, gap: f32, cols: f32, rows: f32) -> vec4<f32> {
    let local = p - origin;
    let pitch = cells * scale + gap;
    let col = floor(local.x / pitch);
    let row = floor(local.y / pitch);
    if (local.x < 0.0 || local.y < 0.0 || col >= cols || row >= rows) {
        return vec4<f32>(0.0);
    }
    let inner = local - vec2<f32>(col, row) * pitch;
    if (inner.x >= cells * scale || inner.y >= cells * scale) {
        return vec4<f32>(0.0);
    }
    let cell = floor(inner / scale);
    return vec4<f32>(col, row, cell.y * cells + cell.x, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let bg = vec4<f32>(0.051, 0.067, 0.090, 1.0);
    let p = pos.xy;

    let input = gridCell(p, IN_ORIGIN, IN_SCALE, 28.0, 0.0, 1.0, 1.0);
    if (input.w > 0.0) {
        let v = clamp(work[W_X + u32(input.z)], 0.0, 1.0);
        return vec4<f32>(vec3<f32>(v), 1.0);
    }
    let c1 = gridCell(p, C1_ORIGIN, C1_SCALE, 24.0, C1_GAP, 4.0, 2.0);
    if (c1.w > 0.0) {
        let m = u32(c1.y * 4.0 + c1.x);
        let v = work[W_A1 + m * 576u + u32(c1.z)] / max(work[W_MAPMAX + m], 1e-6);
        return vec4<f32>(ramp(clamp(v, 0.0, 1.0)), 1.0);
    }
    let c2 = gridCell(p, C2_ORIGIN, C2_SCALE, 8.0, C2_GAP, 4.0, 4.0);
    if (c2.w > 0.0) {
        let m = u32(c2.y * 4.0 + c2.x);
        let v = work[W_A2 + m * 64u + u32(c2.z)] / max(work[W_MAPMAX + 8u + m], 1e-6);
        return vec4<f32>(ramp(clamp(v, 0.0, 1.0)), 1.0);
    }
    return bg;
}
`;

class MNISTRecognizer {
    constructor(device, drawCanvas, mapsCanvas) {
        this.device = device;
        this.drawCanvas = drawCanvas;
        this.mapsCanvas = mapsCanvas;
        this.gpu = new MNISTGPU(device, 1, 1);
        this.gpu.writeSlots([{ start: 0, count: 1, mode: 2, dataLen: 1 }]);
        this.probsBuffer = device.createBuffer({ size: NUM_CLASSES * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        this.busy = false;
        this.pending = null;
        this.destroyed = false;
        this.weightsReady = false;
        this.weightsLabel = 'loading…';
        this.probs = null;
        this.input = null;
        this.testDigit = null;      // { index, label } while a test image is shown
        this.latency = null;
        this.drawing = false;
        this.lastLive = 0;
        this.strokes = 0;

        this.setupDrawing();
        this.setupMapsRender();
        this.renderMaps();
    }

    setupDrawing() {
        const canvas = this.drawCanvas;
        canvas.width = DRAW_SIZE;
        canvas.height = DRAW_SIZE;
        canvas.style.touchAction = 'none';
        this.ctx2d = canvas.getContext('2d', { willReadFrequently: true });
        this.clear();

        const toCanvas = (e) => {
            const rect = canvas.getBoundingClientRect();
            return [(e.clientX - rect.left) * canvas.width / rect.width, (e.clientY - rect.top) * canvas.height / rect.height];
        };
        this.onPointerDown = (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            if (this.testDigit) this.clear();
            canvas.setPointerCapture(e.pointerId);
            this.drawing = true;
            this.last = toCanvas(e);
            this.stroke(this.last, this.last);
        };
        this.onPointerMove = (e) => {
            if (!this.drawing) return;
            const p = toCanvas(e);
            this.stroke(this.last, p);
            this.last = p;
            const now = performance.now();
            if (now - this.lastLive >= LIVE_INTERVAL_MS) {
                this.lastLive = now;
                this.inferDrawing();
            }
        };
        this.onPointerUp = () => {
            if (!this.drawing) return;
            this.drawing = false;
            this.strokes++;
            this.inferDrawing();
        };
        canvas.addEventListener('pointerdown', this.onPointerDown);
        canvas.addEventListener('pointermove', this.onPointerMove);
        canvas.addEventListener('pointerup', this.onPointerUp);
        canvas.addEventListener('pointercancel', this.onPointerUp);
    }

    stroke(a, b) {
        const ctx = this.ctx2d;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = BRUSH_WIDTH;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
    }

    clear() {
        this.ctx2d.fillStyle = '#000';
        this.ctx2d.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
        this.testDigit = null;
        this.probs = null;
        this.input = null;
        this.strokes = 0;
    }

    setupMapsRender() {
        this.mapsCanvas.width = MAPS_W;
        this.mapsCanvas.height = MAPS_H;
        this.context = this.mapsCanvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: 'bgra8unorm' });
        const module = this.device.createShaderModule({ code: mapsShader });
        const layout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }]
        });
        this.mapsPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            vertex: { module, entryPoint: 'vs_main' },
            fragment: { module, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
            primitive: { topology: 'triangle-list' }
        });
        this.mapsBindGroup = this.device.createBindGroup({
            layout, entries: [{ binding: 0, resource: { buffer: this.gpu.workBuffer } }]
        });
    }

    setWeights(weights, label) {
        this.gpu.setWeights(weights);
        this.weightsReady = true;
        this.weightsLabel = label;
        if (this.input) this.infer(this.input);
        else if (this.onUpdate) this.onUpdate();
    }

    encodeMaps(commandEncoder) {
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.051, g: 0.067, b: 0.09, a: 1.0 }
            }]
        });
        pass.setPipeline(this.mapsPipeline);
        pass.setBindGroup(0, this.mapsBindGroup);
        pass.draw(3);
        pass.end();
    }

    // Blank input and maps (the work buffer of a batch-1 network is 60 KB)
    clearMaps() {
        this.pending = null;
        this.device.queue.writeBuffer(this.gpu.workBuffer, 0, new Float32Array(L.total));
        this.renderMaps();
    }

    renderMaps() {
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeMaps(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    inferDrawing() {
        const rgba = this.ctx2d.getImageData(0, 0, DRAW_SIZE, DRAW_SIZE).data;
        const gray = new Float32Array(DRAW_SIZE * DRAW_SIZE);
        for (let i = 0; i < gray.length; i++) gray[i] = rgba[i * 4] / 255;
        const input = preprocessDrawing(gray, DRAW_SIZE, DRAW_SIZE);
        if (input) this.infer(input);
    }

    // Classifies one 28×28 input (Float32Array(784), values in [0, 1])
    infer(input) {
        this.input = input;
        if (!this.weightsReady || this.destroyed) return;
        if (this.busy) {
            this.pending = input;
            return;
        }
        this.busy = true;
        const t0 = performance.now();
        this.device.queue.writeBuffer(this.gpu.workBuffer, L.X * 4, input);
        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        this.gpu.encodeBatch(pass, this.gpu.inferenceBindGroup, 0, { count: 1, mode: 2 });
        this.gpu.encodeMapMax(pass, this.gpu.inferenceBindGroup, 0);
        pass.end();
        this.encodeMaps(commandEncoder);
        commandEncoder.copyBufferToBuffer(this.gpu.workBuffer, L.PROB * 4, this.probsBuffer, 0, NUM_CLASSES * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        this.probsBuffer.mapAsync(GPUMapMode.READ).then(() => {
            if (this.destroyed) return;
            const probs = Array.from(new Float32Array(this.probsBuffer.getMappedRange().slice(0)));
            this.probsBuffer.unmap();
            this.probs = this.input ? probs : null;   // null: cleared while this inference ran
            this.latency = performance.now() - t0;
            this.busy = false;
            if (this.onUpdate) this.onUpdate();
            if (this.pending) {
                const next = this.pending;
                this.pending = null;
                this.infer(next);
            }
        }).catch(() => {
            this.busy = false;
        });
    }

    async randomTestDigit() {
        const test = await loadDataset('test');
        const index = Math.floor(Math.random() * test.count);
        const pixels = test.images.subarray(index * IMG_PIXELS, (index + 1) * IMG_PIXELS);
        // Show the 28×28 test image scaled up 10× without smoothing
        const image = new ImageData(28, 28);
        for (let i = 0; i < IMG_PIXELS; i++) {
            image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = pixels[i];
            image.data[i * 4 + 3] = 255;
        }
        const small = document.createElement('canvas');
        small.width = 28;
        small.height = 28;
        small.getContext('2d').putImageData(image, 0, 0);
        this.clear();
        this.ctx2d.imageSmoothingEnabled = false;
        this.ctx2d.drawImage(small, 0, 0, DRAW_SIZE, DRAW_SIZE);
        this.testDigit = { index, label: test.labels[index] };
        // Test images are already in MNIST format: no preprocessing
        this.infer(Float32Array.from(pixels, v => v / 255));
    }

    destroy() {
        this.destroyed = true;
        this.drawCanvas.removeEventListener('pointerdown', this.onPointerDown);
        this.drawCanvas.removeEventListener('pointermove', this.onPointerMove);
        this.drawCanvas.removeEventListener('pointerup', this.onPointerUp);
        this.drawCanvas.removeEventListener('pointercancel', this.onPointerUp);
        this.gpu.destroy();
        this.probsBuffer.destroy();
    }
}

// Ten horizontal confidence bars (HTML), the top class highlighted
function createBars(container) {
    container.innerHTML = '';
    container.style.cssText = 'display:grid;grid-template-columns:1.2em minmax(0,1fr) 4.2em;gap:4px 10px;' +
        'align-items:center;max-width:420px;margin-top:12px;font-variant-numeric:tabular-nums;font-size:0.9rem;';
    const rows = [];
    for (let k = 0; k < NUM_CLASSES; k++) {
        const label = document.createElement('span');
        label.textContent = k;
        label.style.color = 'var(--text)';
        const track = document.createElement('div');
        track.style.cssText = 'height:12px;background:var(--surface-2);border-radius:4px;overflow:hidden;';
        const bar = document.createElement('div');
        bar.style.cssText = 'height:100%;width:0;border-radius:4px;background:#2196d3;transition:width 0.12s;';
        track.appendChild(bar);
        const value = document.createElement('span');
        value.style.cssText = 'color:var(--muted);text-align:right;';
        value.textContent = '–';
        container.append(label, track, value);
        rows.push({ label, bar, value });
    }
    return (probs) => {
        const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
        rows.forEach((row, k) => {
            const p = probs ? probs[k] : 0;
            row.bar.style.width = `${(100 * p).toFixed(1)}%`;
            row.bar.style.background = k === best ? '#4fc3f7' : '#2196d3';
            row.bar.style.opacity = k === best ? '1' : '0.7';
            row.label.style.fontWeight = k === best ? '700' : '400';
            row.value.textContent = probs ? `${(100 * p).toFixed(1)}%` : '–';
            row.value.style.color = k === best ? 'var(--text)' : 'var(--muted)';
        });
    };
}

export async function initMNISTRecognize(gpuDevice) {
    const { device } = gpuDevice;
    const drawCanvas = document.getElementById('canvas-mnist-draw');
    const mapsCanvas = document.getElementById('canvas-mnist-maps');
    const output = document.getElementById('output-mnist-recognize');
    const showBars = createBars(document.getElementById('mnist-bars'));

    const recognizer = new MNISTRecognizer(device, drawCanvas, mapsCanvas);
    recognizer.onUpdate = () => updateOutput();
    let source = 'cached';
    let error = null;

    async function useCached() {
        try {
            recognizer.setWeights(await loadCachedWeights(), 'cached weights (data/mnist/weights.bin, trained by tools/train-mnist.mjs)');
            source = 'cached';
            error = null;
        } catch (e) {
            error = `Could not load the cached weights: ${e.message}`;
        }
        updateOutput();
    }

    const unsubscribe = sharedWeights.subscribe((trained) => {
        recognizer.setWeights(trained.weights, trained.label);
        source = 'train';
        updateOutput();
    });
    const originalDestroy = recognizer.destroy.bind(recognizer);
    recognizer.destroy = () => {
        unsubscribe();
        originalDestroy();
    };

    document.getElementById('btn-mnist-clear').addEventListener('click', () => {
        recognizer.clear();
        recognizer.clearMaps();
        updateOutput();
    });
    document.getElementById('btn-mnist-random').addEventListener('click', async () => {
        try {
            await recognizer.randomTestDigit();
            error = null;
        } catch (e) {
            error = `Could not load the test images: ${e.message}`;
        }
        updateOutput();
    });
    document.getElementById('btn-mnist-cached').addEventListener('click', () => useCached());

    function updateOutput() {
        const r = recognizer;
        showBars(r.probs);
        const best = r.probs ? r.probs.indexOf(Math.max(...r.probs)) : -1;
        let prediction = '<span class="info">Draw a digit (0-9) in the black square, or load a random test digit</span>';
        if (r.probs) {
            prediction = `<span class="success">Prediction: ${best}</span> (${(100 * r.probs[best]).toFixed(1)}% confident)`;
            if (r.testDigit) {
                const ok = best === r.testDigit.label;
                prediction += ` · test image #${r.testDigit.index}, true label ${r.testDigit.label}: ` +
                    (ok ? '<span class="success">correct</span>' : '<span class="error">wrong</span>');
            }
        }

        output.innerHTML = `<span class="success">✓ MNIST Recognition</span>

${prediction}${error ? `\n<span class="error">${error}</span>` : ''}

<span class="info">Weights:</span> ${source === 'train' ? '<span class="success">from example 21</span> · ' : ''}${r.weightsLabel}

<span class="info">Pipeline:</span>
• Preprocessing (CPU): bounding box → scale to fit 20×20 (area average) → center of mass to the middle of 28×28${r.testDigit ? ' (skipped: test images are already in this format)' : ''}
• Inference (GPU): conv 5×5×8 → ReLU → max-pool → conv 5×5×16 → ReLU → max-pool → dense → softmax, batch 1, 7 dispatches
• Activation maps: each map scaled to its own maximum; dark = 0, bright = the map's strongest response
• Runs when a stroke ends and every ${LIVE_INTERVAL_MS} ms while drawing

<span class="info">Performance:</span>
• Latency (write input → probabilities back on the CPU): ${r.latency !== null ? `${r.latency.toFixed(1)} ms` : '–'}
• Readback: the 10 probabilities (40 bytes); the maps are drawn from the GPU buffer directly`;
    }

    updateOutput();
    if (sharedWeights.trained) {
        recognizer.setWeights(sharedWeights.trained.weights, sharedWeights.trained.label);
        source = 'train';
        updateOutput();
    } else {
        await useCached();
    }
    return recognizer;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() { return [{ name: 'maps', code: mapsShader }]; }
