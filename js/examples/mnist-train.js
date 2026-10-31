// MNIST, part 1: train the shared small CNN (mnist-model.js) in the browser, entirely in compute
// shaders: batch assembly, forward pass, softmax cross-entropy, backprop through the dense layer,
// both max-pools, both ReLUs and both convolutions, and SGD with momentum.
//
// Per mini-batch (one dynamic-offset uniform slot each, up to MAX_STEPS_PER_FRAME per frame):
//   cs_gather       batch images from the uploaded 10,000-image set by shuffled index (the shuffle
//                   is a seeded Fisher-Yates on the CPU, uploaded once per epoch: 40 KB), each
//                   translated by a hashed random shift of up to ±2 pixels (augmentation)
//   cs_conv1 … cs_dense, cs_softmax   forward pass and loss
//   cs_stats        one invocation adds the batch's loss and correct count to a 32-byte Stats struct
//   cs_dense_back … cs_pool1_back      backward pass (every kernel is a gather, so no write races)
//   cs_grads        one invocation per (sample, parameter) writes that sample's gradient to its own
//                   row of a [batch][5994] buffer
//   cs_update       one invocation per parameter sums its row entries in sample order, then
//                   g += λ·w (weight decay), v = μ·v + g, w = w − η·v, with η = η0 · decay^epoch
// After every epoch the 2,000 test images run forward in batches of 128 (mode 1) and count correct
// predictions into Stats.
//
// Readbacks: the 32-byte Stats struct, at most one in flight (if the previous one has not landed,
// the GPU keeps accumulating and the next copy picks it up), and the 24 KB weights on Export or
// Use in Recognize. The first-layer filters are drawn straight from the weights buffer.
import {
    MNISTGPU, DEFAULTS, MAX_BATCH, NUM_PARAMS, STATS_SIZE, loadDataset, initWeights, epochOrder,
    augmentSeed, serializeWeights, sharedWeights,
} from './mnist-model.js';

const MAX_STEPS_PER_FRAME = 16;
const EVAL_BATCH = MAX_BATCH;
const EVAL_STEPS_PER_FRAME = 4;
const PLOT_WINDOW = 500;            // examples per point of the training curves
const LEARNING_RATES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15];
const BATCH_SIZES = [32, 64, 128];

// First-layer filters: 8 filters of 5×5 taps drawn from the weights buffer (W1 at offset 0).
// Each filter is normalized by its largest |w| and colored on a diverging blue (+) / orange (−)
// ramp with a neutral gray at 0.
const FILTER_TAP = 9;               // pixels per tap
const FILTER_GAP = 12;
const FILTER_MARGIN = 8;
const FILTER_CANVAS_W = FILTER_MARGIN * 2 + 8 * 5 * FILTER_TAP + 7 * FILTER_GAP;
const FILTER_CANVAS_H = FILTER_MARGIN * 2 + 5 * FILTER_TAP;

const filterShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> weights: array<f32>;

const TAP: f32 = ${FILTER_TAP}.0;
const GAP: f32 = ${FILTER_GAP}.0;
const MARGIN: f32 = ${FILTER_MARGIN}.0;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(corners[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let bg = vec4<f32>(0.051, 0.067, 0.090, 1.0);
    let p = pos.xy - vec2<f32>(MARGIN);
    let pitch = 5.0 * TAP + GAP;
    if (p.x < 0.0 || p.y < 0.0 || p.y >= 5.0 * TAP) { return bg; }
    let f = u32(p.x / pitch);
    let local = p.x - f32(f) * pitch;
    if (f >= 8u || local >= 5.0 * TAP) { return bg; }
    let tap = u32(p.y / TAP) * 5u + u32(local / TAP);

    var maxAbs = 1e-6;
    for (var k = 0u; k < 25u; k++) {
        maxAbs = max(maxAbs, abs(weights[f * 25u + k]));
    }
    let v = clamp(weights[f * 25u + tap] / maxAbs, -1.0, 1.0);
    let neutral = vec3<f32>(0.19, 0.21, 0.24);
    let positive = vec3<f32>(0.40, 0.72, 0.95);
    let negative = vec3<f32>(0.96, 0.58, 0.22);
    var color = mix(neutral, positive, v);
    if (v < 0.0) {
        color = mix(neutral, negative, -v);
    }
    return vec4<f32>(color, 1.0);
}
`;

class MNISTTrainer {
    constructor(device, plotCanvas, filterCanvas) {
        this.device = device;
        this.plotCanvas = plotCanvas;
        this.filterCanvas = filterCanvas;
        this.gpu = new MNISTGPU(device, MAX_BATCH, MAX_STEPS_PER_FRAME);

        this.lr = DEFAULTS.lr;
        this.momentum = DEFAULTS.momentum;
        this.lrDecay = DEFAULTS.lrDecay;
        this.weightDecay = DEFAULTS.weightDecay;
        this.augment = true;
        this.batch = DEFAULTS.batch;
        this.epochs = DEFAULTS.epochs;
        this.stepsPerFrame = 4;
        this.resets = 0;

        this.trainData = null;      // GPU datasets, uploaded on the first Start
        this.testData = null;
        this.loading = null;
        this.running = false;
        this.destroyed = false;
        this.readbackBusy = false;
        this.statsDirty = false;    // steps were encoded since Stats was last copied
        this.generation = 0;
        this.error = null;
        this.loopActive = false;
        this.lastNotify = 0;

        this.readbackBuffer = device.createBuffer({ size: STATS_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.setupFilterRender();
        this.plot = new TrainingPlot(plotCanvas, () => this.epochs);
        this.reset();
    }

    setupFilterRender() {
        this.context = this.filterCanvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: 'bgra8unorm' });
        const module = this.device.createShaderModule({ code: filterShader });
        const layout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }]
        });
        this.filterPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            vertex: { module, entryPoint: 'vs_main' },
            fragment: { module, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
            primitive: { topology: 'triangle-list' }
        });
        this.filterBindGroup = this.device.createBindGroup({
            layout,
            entries: [{ binding: 0, resource: { buffer: this.gpu.weightsBuffer, offset: 0, size: 200 * 4 } }]
        });
    }

    // η = η0 · decay^epoch (the same schedule as tools/train-mnist.mjs)
    epochLearningRate() {
        return this.lr * Math.pow(this.lrDecay, this.epoch);
    }

    get seed() {
        return DEFAULTS.seed + this.resets;
    }

    // Fresh He-initialized weights (seed = default + number of resets), zero momentum, new curves
    reset() {
        this.running = false;
        this.generation++;
        this.gpu.setWeights(initWeights(this.seed));
        this.device.queue.writeBuffer(this.gpu.statsBuffer, 0, new Uint8Array(STATS_SIZE));
        this.epoch = 0;             // completed epochs
        this.pos = 0;               // samples consumed in the current epoch
        this.phase = 'train';       // 'train' or 'eval' (test pass after each epoch)
        this.evalPos = 0;
        this.orderEpoch = -1;       // epoch whose shuffle is in the order buffer
        this.batchesDone = 0;
        this.examplesEncoded = 0;
        this.examplesConfirmed = 0; // counted by the GPU and read back
        this.window = { loss: 0, correct: 0, count: 0 };
        this.testSum = { loss: 0, correct: 0, count: 0 };
        this.testResults = [];      // { epoch, accuracy, loss }
        this.lastLoss = null;
        this.lastAccuracy = null;
        this.rate = 0;
        this.rateMark = { time: performance.now(), examples: 0 };
        this.statsDirty = false;
        this.status = 'ready';
        this.plot.clear();
        this.renderFilters();
    }

    async ensureData() {
        if (this.trainData) return;
        if (!this.loading) {
            this.loading = Promise.all([loadDataset('train'), loadDataset('test')]).then(([train, test]) => {
                if (this.destroyed) return;
                this.trainData = this.gpu.uploadDataset(train, true);
                this.testData = this.gpu.uploadDataset(test, false);
            });
            this.loading.catch(() => { this.loading = null; });
        }
        await this.loading;
    }

    async start() {
        if (this.running || this.status === 'loading') return;
        if (this.epoch >= this.epochs || this.status === 'diverged') return;
        if (!this.trainData) {
            this.status = 'loading';
            this.notify();
            const generation = this.generation;
            try {
                await this.ensureData();
            } catch (error) {
                this.status = 'error';
                this.error = error.message;
                this.notify();
                return;
            }
            if (this.destroyed || generation !== this.generation || this.status !== 'loading') return;
        }
        this.running = true;
        this.status = 'running';
        this.rateMark = { time: performance.now(), examples: this.examplesConfirmed };
        this.notify();
        this.animate();
    }

    pause() {
        if (this.status === 'loading') this.status = 'ready';
        if (!this.running) return;
        this.running = false;
        this.status = 'paused';
        this.notify();
    }

    notify() {
        if (this.onStats) this.onStats();
    }

    // Plans this frame's mini-batches. A frame never crosses an epoch boundary, so the order buffer
    // (rewritten at the start of each epoch's first frame) is never changed under queued batches.
    planSteps() {
        const steps = [];
        if (this.phase === 'train') {
            const n = this.trainData.count;
            if (this.orderEpoch !== this.epoch) {
                this.device.queue.writeBuffer(this.trainData.orderBuffer, 0, epochOrder(n, this.seed, this.epoch));
                this.orderEpoch = this.epoch;
            }
            while (steps.length < this.stepsPerFrame && this.pos < n) {
                const count = Math.min(this.batch, n - this.pos);
                steps.push({
                    start: this.pos, count, mode: 0, dataLen: n,
                    lr: this.epochLearningRate(), momentum: this.momentum, weightDecay: this.weightDecay,
                    augSeed: augmentSeed(this.seed, this.epoch), maxShift: this.augment ? DEFAULTS.maxShift : 0,
                });
                this.pos += count;
                this.batchesDone++;
                this.examplesEncoded += count;
            }
            if (this.pos >= n) {
                this.phase = 'eval';
                this.evalPos = 0;
            }
        } else {
            const n = this.testData.count;
            while (steps.length < EVAL_STEPS_PER_FRAME && this.evalPos < n) {
                const count = Math.min(EVAL_BATCH, n - this.evalPos);
                steps.push({ start: this.evalPos, count, mode: 1, dataLen: n });
                this.evalPos += count;
            }
            if (this.evalPos >= n) {
                this.epoch++;
                this.pos = 0;
                this.phase = 'train';
                if (this.epoch >= this.epochs) this.running = false;
            }
        }
        return steps;
    }

    frame() {
        const steps = this.running ? this.planSteps() : [];
        const commandEncoder = this.device.createCommandEncoder();
        if (steps.length) {
            this.gpu.writeSlots(steps);
            const bindGroup = steps[0].mode === 0 ? this.trainData.bindGroup : this.testData.bindGroup;
            const pass = commandEncoder.beginComputePass();
            steps.forEach((step, i) => this.gpu.encodeBatch(pass, bindGroup, i, step));
            pass.end();
            this.statsDirty = true;
        }
        this.encodeFilters(commandEncoder);
        const copy = this.statsDirty && !this.readbackBusy;
        if (copy) {
            commandEncoder.copyBufferToBuffer(this.gpu.statsBuffer, 0, this.readbackBuffer, 0, STATS_SIZE);
            commandEncoder.clearBuffer(this.gpu.statsBuffer);
            this.statsDirty = false;
        }
        this.device.queue.submit([commandEncoder.finish()]);
        if (copy) this.readStats();
    }

    // Stats layout: trainLoss f32, trainCorrect u32, trainCount u32, pad, testLoss f32, testCorrect u32,
    // testCount u32, pad (see STATS_SIZE in mnist-model.js)
    readStats() {
        const generation = this.generation;
        this.readbackBusy = true;
        this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            if (this.destroyed) return;
            const bytes = this.readbackBuffer.getMappedRange().slice(0);
            this.readbackBuffer.unmap();
            this.readbackBusy = false;
            if (generation !== this.generation) return;
            const f32 = new Float32Array(bytes);
            const u32 = new Uint32Array(bytes);
            this.addStats(f32[0], u32[1], u32[2], f32[4], u32[5], u32[6]);
        }).catch(() => {
            this.readbackBusy = false;
        });
    }

    addStats(trainLoss, trainCorrect, trainCount, testLoss, testCorrect, testCount) {
        if (trainCount > 0) {
            if (!Number.isFinite(trainLoss)) {
                this.running = false;
                this.status = 'diverged';
                this.notify();
                return;
            }
            this.examplesConfirmed += trainCount;
            this.window.loss += trainLoss;
            this.window.correct += trainCorrect;
            this.window.count += trainCount;
            if (this.window.count >= PLOT_WINDOW) {
                this.lastLoss = this.window.loss / this.window.count;
                this.lastAccuracy = this.window.correct / this.window.count;
                this.plot.addTrain(this.examplesConfirmed, this.lastLoss, this.lastAccuracy);
                this.window = { loss: 0, correct: 0, count: 0 };
            }
        }
        if (testCount > 0) {
            this.testSum.loss += testLoss;
            this.testSum.correct += testCorrect;
            this.testSum.count += testCount;
            const n = this.testData.count;
            if (this.testSum.count >= n) {
                const epoch = this.testResults.length + 1;
                const result = { epoch, accuracy: this.testSum.correct / n, loss: this.testSum.loss / n };
                this.testResults.push(result);
                this.plot.addTest(epoch * this.trainData.count, result.accuracy);
                this.testSum = { loss: 0, correct: 0, count: 0 };
                if (this.epoch >= this.epochs && !this.running && this.status === 'running') this.status = 'done';
            }
        }
        const now = performance.now();
        if (now - this.rateMark.time >= 1000) {
            this.rate = (this.examplesConfirmed - this.rateMark.examples) * 1000 / (now - this.rateMark.time);
            this.rateMark = { time: now, examples: this.examplesConfirmed };
        }
        // The info panel is rebuilt at most 4 times a second, and after every test result
        if (testCount > 0 || now - this.lastNotify >= 250) {
            this.lastNotify = now;
            this.notify();
        }
    }

    encodeFilters(commandEncoder) {
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.051, g: 0.067, b: 0.09, a: 1.0 }
            }]
        });
        pass.setPipeline(this.filterPipeline);
        pass.setBindGroup(0, this.filterBindGroup);
        pass.draw(3);
        pass.end();
    }

    renderFilters() {
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeFilters(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    // One loop at a time (start() after a pause may come while the previous loop still drains)
    animate() {
        if (this.loopActive) return;
        this.loopActive = true;
        this.loop();
    }

    // Keeps drawing while running, and afterwards until the last Stats copy has been read back
    loop() {
        if (this.destroyed) {
            this.loopActive = false;
            return;
        }
        if (!this.running && !this.statsDirty) {
            if (this.readbackBusy) {
                requestAnimationFrame(() => this.loop());
                return;
            }
            if (this.status === 'running') this.status = this.epoch >= this.epochs ? 'done' : 'paused';
            this.loopActive = false;
            this.notify();
            return;
        }
        this.frame();
        requestAnimationFrame(() => this.loop());
    }

    // Copies the current weights (not the momentum) back to the CPU
    async readWeights() {
        const size = NUM_PARAMS * 4;
        const buffer = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.gpu.weightsBuffer, 0, buffer, 0, size);
        this.device.queue.submit([commandEncoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const weights = new Float32Array(buffer.getMappedRange().slice(0));
        buffer.unmap();
        buffer.destroy();
        return weights;
    }

    describeWeights() {
        const last = this.testResults[this.testResults.length - 1];
        const where = this.pos > 0 || this.phase === 'eval'
            ? `epoch ${this.epoch + 1}, ${this.pos.toLocaleString()} of 10,000 images`
            : `${this.epoch} epoch${this.epoch === 1 ? '' : 's'}`;
        return `weights from Train (seed ${this.seed}, ${where}${last ? `, test accuracy ${(100 * last.accuracy).toFixed(2)}%` : ''})`;
    }

    destroy() {
        this.running = false;
        this.destroyed = true;
        this.gpu.destroy();
        if (this.trainData) this.trainData.destroy();
        if (this.testData) this.testData.destroy();
        this.readbackBuffer.destroy();
    }
}

// Two small charts side by side on a 2D canvas: training loss (log scale) and accuracy (training
// window accuracy as a line, test accuracy after each epoch as points). Hover shows values.
const TRAIN_COLOR = '#2196d3';
const TEST_COLOR = '#d27a1e';
const INK = '#e6edf3';
const MUTED = '#9aa7b6';
const GRID = '#2a3342';

class TrainingPlot {
    constructor(canvas, getEpochs) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.getEpochs = getEpochs;
        this.hover = null;
        this.clear();
        canvas.addEventListener('pointermove', (e) => {
            const rect = canvas.getBoundingClientRect();
            this.hover = {
                x: (e.clientX - rect.left) * canvas.width / rect.width,
                y: (e.clientY - rect.top) * canvas.height / rect.height,
            };
            this.draw();
        });
        canvas.addEventListener('pointerleave', () => {
            this.hover = null;
            this.draw();
        });
    }

    clear() {
        this.train = [];    // { x: examples, loss, accuracy }
        this.test = [];     // { x: examples, accuracy }
        this.draw();
    }

    addTrain(x, loss, accuracy) {
        this.train.push({ x, loss, accuracy });
        this.draw();
    }

    addTest(x, accuracy) {
        this.test.push({ x, accuracy });
        this.draw();
    }

    panels() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const gap = 56;
        const left = 58;
        const right = 16;
        const top = 34;
        const bottom = 40;
        const pw = (w - left * 2 - gap - right) / 2;
        return [
            { x: left, y: top, w: pw, h: h - top - bottom },
            { x: left * 2 + pw + gap, y: top, w: pw, h: h - top - bottom },
        ];
    }

    draw() {
        const ctx = this.ctx;
        const { width, height } = this.canvas;
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);
        const [lossPanel, accPanel] = this.panels();
        const examplesPerEpoch = 10000;
        const xMax = Math.max(this.getEpochs(), 1) * examplesPerEpoch;
        const sx = (p, x) => p.x + (x / xMax) * p.w;

        // Loss: log10 scale from 0.01 to 3
        const lossLo = Math.log10(0.01);
        const lossHi = Math.log10(3);
        const lossY = (v) => lossPanel.y + lossPanel.h * (1 - (Math.log10(Math.max(v, 0.01)) - lossLo) / (lossHi - lossLo));
        // Accuracy: from a rounded-down minimum (at most 80%) to 100%
        const accs = this.train.map(p => p.accuracy).concat(this.test.map(p => p.accuracy));
        const accLo = Math.min(0.8, Math.floor((accs.length ? Math.min(...accs) : 0.8) * 10) / 10);
        const accY = (v) => accPanel.y + accPanel.h * (1 - (v - accLo) / (1 - accLo));

        ctx.font = '12px system-ui, sans-serif';
        const axes = (p, title, ticks, yOf, format) => {
            ctx.fillStyle = INK;
            ctx.textAlign = 'left';
            ctx.fillText(title, p.x, p.y - 14);
            ctx.strokeStyle = GRID;
            ctx.lineWidth = 1;
            ctx.fillStyle = MUTED;
            ctx.textAlign = 'right';
            for (const t of ticks) {
                const y = Math.round(yOf(t)) + 0.5;
                ctx.beginPath();
                ctx.moveTo(p.x, y);
                ctx.lineTo(p.x + p.w, y);
                ctx.stroke();
                ctx.fillText(format(t), p.x - 8, y + 4);
            }
            ctx.textAlign = 'center';
            const epochs = Math.max(this.getEpochs(), 1);
            const every = Math.ceil(epochs / 10);
            for (let e = 0; e <= epochs; e += every) {
                const x = Math.round(sx(p, e * examplesPerEpoch)) + 0.5;
                ctx.beginPath();
                ctx.moveTo(x, p.y + p.h);
                ctx.lineTo(x, p.y + p.h + 4);
                ctx.stroke();
                ctx.fillText(String(e), x, p.y + p.h + 17);
            }
            ctx.fillText('epoch', p.x + p.w / 2, p.y + p.h + 34);
        };
        axes(lossPanel, 'Training loss (cross-entropy, log scale)', [0.01, 0.03, 0.1, 0.3, 1, 3], lossY, v => String(v));
        const accTicks = [];
        for (let t = accLo; t <= 1.0001; t += (1 - accLo) / 4) accTicks.push(t);
        axes(accPanel, 'Accuracy', accTicks, accY, v => `${(100 * v).toFixed(0)}%`);

        const line = (p, points, yOf, color) => {
            if (!points.length) return;
            ctx.save();
            ctx.beginPath();
            ctx.rect(p.x, p.y - 2, p.w, p.h + 4);
            ctx.clip();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            points.forEach((pt, i) => {
                const x = sx(p, pt.x);
                const y = yOf(pt);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.restore();
        };
        line(lossPanel, this.train, pt => lossY(pt.loss), TRAIN_COLOR);
        line(accPanel, this.train, pt => accY(pt.accuracy), TRAIN_COLOR);
        line(accPanel, this.test, pt => accY(pt.accuracy), TEST_COLOR);
        for (const pt of this.test) {
            ctx.fillStyle = TEST_COLOR;
            ctx.strokeStyle = '#0d1117';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(sx(accPanel, pt.x), accY(pt.accuracy), 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }

        // Legend (accuracy panel has two series)
        const legend = [['training (last 500 images)', TRAIN_COLOR], ['test (2,000 images)', TEST_COLOR]];
        ctx.textAlign = 'left';
        let lx = accPanel.x + accPanel.w;
        for (let i = legend.length - 1; i >= 0; i--) {
            const [label, color] = legend[i];
            lx -= ctx.measureText(label).width + 26;
            ctx.fillStyle = color;
            ctx.fillRect(lx, accPanel.y - 22, 12, 4);
            ctx.fillStyle = MUTED;
            ctx.fillText(label, lx + 16, accPanel.y - 16);
        }

        if (!this.train.length) {
            ctx.fillStyle = MUTED;
            ctx.textAlign = 'center';
            ctx.fillText('Press Start to train', lossPanel.x + lossPanel.w / 2, lossPanel.y + lossPanel.h / 2);
        }
        this.drawHover(lossPanel, accPanel, sx, lossY, accY, xMax);
    }

    drawHover(lossPanel, accPanel, sx, lossY, accY, xMax) {
        if (!this.hover || !this.train.length) return;
        const panel = [lossPanel, accPanel].find(p => this.hover.x >= p.x && this.hover.x <= p.x + p.w &&
            this.hover.y >= p.y && this.hover.y <= p.y + p.h);
        if (!panel) return;
        const xData = (this.hover.x - panel.x) / panel.w * xMax;
        let nearest = this.train[0];
        for (const pt of this.train) if (Math.abs(pt.x - xData) < Math.abs(nearest.x - xData)) nearest = pt;
        const ctx = this.ctx;
        const x = Math.round(sx(panel, nearest.x)) + 0.5;
        ctx.strokeStyle = MUTED;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, panel.y);
        ctx.lineTo(x, panel.y + panel.h);
        ctx.stroke();
        const y = panel === lossPanel ? lossY(nearest.loss) : accY(nearest.accuracy);
        ctx.fillStyle = TRAIN_COLOR;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        const testNear = this.test.find(t => Math.abs(t.x - nearest.x) <= 10000 / 2 && t.x >= nearest.x - 10000 / 2);
        const lines = [
            `${nearest.x.toLocaleString()} images (epoch ${(nearest.x / 10000).toFixed(2)})`,
            `loss ${nearest.loss.toFixed(4)} · train acc ${(100 * nearest.accuracy).toFixed(1)}%`,
        ];
        if (testNear) lines.push(`test acc after epoch ${Math.round(testNear.x / 10000)}: ${(100 * testNear.accuracy).toFixed(2)}%`);
        ctx.font = '12px system-ui, sans-serif';
        const boxW = Math.max(...lines.map(l => ctx.measureText(l).width)) + 16;
        const boxH = lines.length * 16 + 10;
        const bx = Math.min(x + 10, this.canvas.width - boxW - 4);
        const by = panel.y + 6;
        ctx.fillStyle = 'rgba(28, 35, 48, 0.95)';
        ctx.strokeStyle = GRID;
        ctx.fillRect(bx, by, boxW, boxH);
        ctx.strokeRect(bx + 0.5, by + 0.5, boxW, boxH);
        ctx.fillStyle = INK;
        ctx.textAlign = 'left';
        lines.forEach((l, i) => ctx.fillText(l, bx + 8, by + 18 + i * 16));
    }
}

export async function initMNISTTrain(gpuDevice) {
    const { device } = gpuDevice;
    const plotCanvas = document.getElementById('canvas-mnist-train');
    const filterCanvas = document.getElementById('canvas-mnist-filters');
    const output = document.getElementById('output-mnist-train');
    filterCanvas.width = FILTER_CANVAS_W;
    filterCanvas.height = FILTER_CANVAS_H;

    const trainer = new MNISTTrainer(device, plotCanvas, filterCanvas);
    trainer.onStats = () => updateOutput();
    let message = '';

    const startButton = document.getElementById('btn-mnist-train-start');
    startButton.addEventListener('click', () => {
        trainer.start();
        updateOutput();
    });
    document.getElementById('btn-mnist-train-pause').addEventListener('click', () => {
        trainer.pause();
        updateOutput();
    });
    document.getElementById('btn-mnist-train-reset').addEventListener('click', () => {
        trainer.resets++;
        trainer.reset();
        message = '';
        updateOutput();
    });

    const exportButton = document.getElementById('btn-mnist-export');
    exportButton.addEventListener('click', async () => {
        exportButton.disabled = true;
        try {
            const weights = await trainer.readWeights();
            const url = URL.createObjectURL(new Blob([serializeWeights(weights)], { type: 'application/octet-stream' }));
            const link = document.createElement('a');
            link.href = url;
            link.download = `mnist-weights-seed${trainer.seed}-epoch${trainer.epoch}.bin`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            message = `Exported ${trainer.describeWeights()} as ${link.download}`;
        } catch (error) {
            message = `<span class="error">Export failed: ${error.message}</span>`;
        }
        exportButton.disabled = false;
        updateOutput();
    });

    const useButton = document.getElementById('btn-mnist-use');
    useButton.addEventListener('click', async () => {
        useButton.disabled = true;
        try {
            const weights = await trainer.readWeights();
            const label = trainer.describeWeights();
            sharedWeights.publish(weights, label);
            message = `Example 22 now uses the ${label}`;
        } catch (error) {
            message = `<span class="error">Hand-off failed: ${error.message}</span>`;
        }
        useButton.disabled = false;
        updateOutput();
    });

    const lrInput = document.getElementById('mnist-lr');
    const lrValue = document.getElementById('mnist-lr-value');
    lrInput.addEventListener('input', () => {
        const i = Math.min(Math.max(parseInt(lrInput.value, 10) || 0, 0), LEARNING_RATES.length - 1);
        trainer.lr = LEARNING_RATES[i];
        lrValue.textContent = trainer.lr;
        updateOutput();
    });
    lrInput.value = String(LEARNING_RATES.indexOf(DEFAULTS.lr));
    lrValue.textContent = DEFAULTS.lr;

    const batchSelect = document.getElementById('mnist-batch');
    batchSelect.value = String(DEFAULTS.batch);
    batchSelect.addEventListener('change', () => {
        const b = parseInt(batchSelect.value, 10);
        trainer.batch = BATCH_SIZES.includes(b) ? b : DEFAULTS.batch;
        updateOutput();
    });

    const epochsInput = document.getElementById('mnist-epochs');
    const epochsValue = document.getElementById('mnist-epochs-value');
    epochsInput.value = String(DEFAULTS.epochs);
    epochsValue.textContent = DEFAULTS.epochs;
    epochsInput.addEventListener('input', () => {
        trainer.epochs = Math.min(Math.max(parseInt(epochsInput.value, 10) || 1, 1), 30);
        epochsValue.textContent = trainer.epochs;
        if (trainer.status === 'done' && trainer.epoch < trainer.epochs) trainer.status = 'paused';
        trainer.plot.draw();
        updateOutput();
    });

    const augmentInput = document.getElementById('mnist-augment');
    augmentInput.checked = trainer.augment;
    augmentInput.addEventListener('change', () => {
        trainer.augment = augmentInput.checked;
        updateOutput();
    });

    const speedInput = document.getElementById('mnist-speed');
    const speedValue = document.getElementById('mnist-speed-value');
    speedInput.addEventListener('input', () => {
        trainer.stepsPerFrame = Math.min(Math.max(parseInt(speedInput.value, 10) || 1, 1), MAX_STEPS_PER_FRAME);
        speedValue.textContent = trainer.stepsPerFrame;
        updateOutput();
    });
    speedInput.value = String(trainer.stepsPerFrame);
    speedValue.textContent = trainer.stepsPerFrame;

    function updateOutput() {
        const t = trainer;
        const statusText = {
            ready: '<span class="info">Ready: press Start (loads 1.6 MB of training data on first use)</span>',
            loading: '<span class="info">Loading MNIST data…</span>',
            running: `<span class="success">Training${t.phase === 'eval' ? ' (testing after the epoch)' : ''}</span>`,
            paused: '<span class="info">Paused</span>',
            done: `<span class="success">Done: ${t.epoch} epoch${t.epoch === 1 ? '' : 's'}</span> (raise Epochs and press Start to continue)`,
            diverged: '<span class="error">The loss became NaN/∞: lower the learning rate and press Reset</span>',
            error: `<span class="error">Could not load the data: ${t.error}</span>`,
        }[t.status];
        const n = t.trainData ? t.trainData.count : 10000;
        const batchesPerEpoch = Math.ceil(n / t.batch);
        const last = t.testResults[t.testResults.length - 1];
        const testList = t.testResults.length
            ? t.testResults.map(r => `${r.epoch}: ${(100 * r.accuracy).toFixed(2)}%`).join(' · ')
            : '–';
        startButton.disabled = t.running || t.status === 'loading';

        output.innerHTML = `<span class="success">✓ MNIST Training</span>

<span class="info">Status:</span> ${statusText}${message ? `\n${message}` : ''}

<span class="info">Progress:</span>
• Epoch: ${Math.min(t.epoch + (t.pos > 0 || t.phase === 'eval' ? 1 : 0), t.epochs)} / ${t.epochs} · batch ${Math.ceil(t.pos / t.batch)} / ${batchesPerEpoch} · ${t.batchesDone.toLocaleString()} mini-batches in total
• Images trained on: ${t.examplesConfirmed.toLocaleString()} (${t.running ? Math.round(t.rate).toLocaleString() : 0} images / s)
• Training loss: ${t.lastLoss !== null ? t.lastLoss.toFixed(4) : '–'} · training accuracy: ${t.lastAccuracy !== null ? `${(100 * t.lastAccuracy).toFixed(1)}%` : '–'} (last ${PLOT_WINDOW} images)
• Test accuracy by epoch (2,000 images): ${testList}${last ? ` · test loss ${last.loss.toFixed(4)}` : ''}

<span class="info">Configuration:</span>
• Network: 28×28 → conv 5×5×8 → ReLU → max-pool → conv 5×5×16 → ReLU → max-pool → dense 256→10 → softmax (5,994 weights)
• Data: first 10,000 MNIST training images (shuffled every epoch, seed ${t.seed}), pixels / 255${t.augment ? `, random shifts of up to ±${DEFAULTS.maxShift} px` : ''}
• Optimizer: SGD, momentum ${t.momentum}, learning rate ${t.lr} × ${t.lrDecay}^epoch (now ${t.epochLearningRate().toPrecision(3)}), weight decay ${t.weightDecay}, batch ${t.batch}, gradients clamped to ±5
• ${t.stepsPerFrame} mini-batch${t.stepsPerFrame === 1 ? '' : 'es'} per frame

<span class="info">Algorithm (all on the GPU, 14 compute dispatches per mini-batch):</span>
• Batch assembly: gather shuffled images from the uploaded uint8 dataset (indices uploaded once per epoch)
• Forward: direct convolutions, one invocation per output value; log-sum-exp softmax cross-entropy
• Backward: dense, max-pool (gradient to the window's argmax), ReLU and conv input gradients as gathers
• Weight gradients: one invocation per (sample, weight) into a ${t.batch}×5,994 buffer, then one per weight
  sums its column in sample order (race-free, no atomics) and applies v = μv + g, w −= ηv
• Filters above: the 8 first-layer 5×5 kernels, drawn from the weights buffer every frame (blue +, orange −)

<span class="info">Readback:</span> 32 bytes of loss/accuracy sums per frame (at most one in flight); weights (24 KB) only on Export / Use in Recognize`;
    }

    updateOutput();
    return trainer;
}
