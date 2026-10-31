// GPU pass timing with the optional 'timestamp-query' feature.
//
// Wall-clock time around submit → onSubmittedWorkDone/mapAsync includes JavaScript, driver and queue
// latency. Timestamp queries instead record GPU timestamps at the beginning and end of each timed
// pass, so the difference is the time the GPU spent executing that pass.
//
// Usage, once per frame (or per one-shot run):
//
//     const timer = new GpuTimer(device, { maxPasses: 4 });
//     const pass = encoder.beginComputePass(timer.computePassDescriptor('force'));
//     const rp = encoder.beginRenderPass(timer.renderPassDescriptor(renderDesc, 'draw'));
//     timer.resolve(encoder);                    // after the last timed pass, before finish()
//     device.queue.submit([encoder.finish()]);
//     timer.read();                              // after submit; returns a Promise, never await it in a rAF loop
//
// read() resolves to { passes: { label: ms }, total, count, truncated, meta } (or null when this frame was
// not timed) and updates exponential moving averages in timer.ema / timer.emaTotal. Passes that share
// a label are summed. Results are copied into a small pool of MAP_READ buffers, so at most poolSize
// readbacks are in flight; when every buffer is still mapping, the frame is simply not timed (its pass
// descriptors are returned unchanged) instead of stalling. A buffer that is mapping is never reused.
//
// Without the feature (or after destroy()) every method is a cheap no-op that returns its input
// descriptor unchanged, and the text helpers report GPU_TIME_UNAVAILABLE.
//
// Quantization: Chrome rounds timestamp-query results to ~100 µs to limit timing side channels unless
// the browser runs with WebGPU developer features (chrome://flags/#enable-webgpu-developer-features,
// or --enable-webgpu-developer-features). Short passes therefore read as 0 or 0.1 ms; averages over
// many frames (the EMA) and long passes are still meaningful.

export const GPU_TIME_UNAVAILABLE = 'GPU time: not available (timestamp-query unsupported)';
export const GPU_TIME_NOTE = 'GPU times use timestamp queries; Chrome quantizes them to ~100 µs unless '
    + 'chrome://flags/#enable-webgpu-developer-features is enabled.';

export function timestampsSupported(device) {
    const dev = device?.device ?? device;
    // A real GPUSupportedFeatures.has returns a boolean; accept only true so stand-in device objects
    // (test mocks, wrappers) whose has() returns something truthy don't switch timing on.
    return dev?.features?.has?.('timestamp-query') === true;
}

export function formatMs(ms, digits = 2) {
    return ms === null || ms === undefined || !Number.isFinite(ms) ? '—' : ms.toFixed(digits);
}

export class GpuTimer {
    // device: a GPUDevice or the { device } object returned by getGPUDevice.
    // maxPasses: timed passes per frame (2 timestamps each); extra passes are left untimed.
    // poolSize: readback buffers, i.e. how many frames can be in flight.
    // minIntervalMs: time a frame at most this often (0 = whenever a readback buffer is free).
    // emaAlpha: weight of the newest sample in the moving averages.
    // onResult(result): called after each successful read.
    constructor(device, { maxPasses = 8, poolSize = 3, minIntervalMs = 0, emaAlpha = 0.1, onResult = null } = {}) {
        this.device = device?.device ?? device;
        this.supported = timestampsSupported(this.device);
        this.maxPasses = maxPasses;
        this.minIntervalMs = minIntervalMs;
        this.emaAlpha = emaAlpha;
        this.onResult = onResult;
        this.ema = {};
        this.emaTotal = null;
        this.last = null;
        this.samples = 0;
        this.skipped = 0;           // frames not timed because every readback buffer was busy
        this.frame = null;          // { slot, labels, truncated } while passes are being encoded
        this.toRead = null;         // slot resolved by the last resolve(), waiting for read()
        this.lastSample = -Infinity;
        this.live = false;
        if (!this.supported) return;

        const count = maxPasses * 2;
        this.querySet = this.device.createQuerySet({ type: 'timestamp', count, label: 'gpu-timer queries' });
        this.resolveBuffer = this.device.createBuffer({
            label: 'gpu-timer resolve',
            size: count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        });
        this.pool = Array.from({ length: poolSize }, (_, i) => ({
            buffer: this.device.createBuffer({
                label: `gpu-timer readback ${i}`,
                size: count * 8,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            }),
            state: 'free',          // 'free' → 'encoded' (copy recorded) → 'mapping' → 'free'
            labels: null,
            truncated: false
        }));
        this.live = true;
    }

    // Starts a frame explicitly (otherwise the first descriptor call does). Returns whether this frame is timed.
    beginFrame() {
        if (!this.live) return false;
        let slot = null;
        if (performance.now() - this.lastSample >= this.minIntervalMs) {
            slot = this.pool.find((s) => s.state === 'free') ?? null;
            // A frame that was resolved but never read gives up its (never mapped) buffer
            if (!slot && this.toRead?.state === 'encoded') {
                slot = this.toRead;
                this.toRead = null;
            }
            if (!slot) this.skipped++;
        }
        this.frame = { slot, labels: [], truncated: false };
        return !!slot;
    }

    get timing() {
        return !!this.frame?.slot;
    }

    nextWrites(label) {
        if (!this.live) return null;
        if (!this.frame) this.beginFrame();
        const frame = this.frame;
        if (!frame.slot) return null;
        if (frame.labels.length >= this.maxPasses) {
            frame.truncated = true;
            return null;
        }
        const i = frame.labels.length;
        frame.labels.push(label);
        return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
    }

    // Descriptor for beginComputePass: base (may be undefined) plus timestampWrites when this frame is timed.
    computePassDescriptor(label, base) {
        const timestampWrites = this.nextWrites(label);
        return timestampWrites ? { ...base, timestampWrites } : base;
    }

    // Descriptor for beginRenderPass: a copy of desc with timestampWrites when this frame is timed.
    renderPassDescriptor(desc, label) {
        const timestampWrites = this.nextWrites(label);
        return timestampWrites ? { ...desc, timestampWrites } : desc;
    }

    // Records the query resolve and the copy into the frame's readback buffer, and ends the frame.
    // Call it on the encoder of the last timed pass (or a later one in the same submit). meta is
    // passed through to the result (e.g. how many batches the frame ran).
    resolve(encoder, meta = null) {
        if (!this.live) return;
        const frame = this.frame;
        this.frame = null;
        if (!frame?.slot || frame.labels.length === 0) return;
        const queries = frame.labels.length * 2;
        encoder.resolveQuerySet(this.querySet, 0, queries, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, frame.slot.buffer, 0, queries * 8);
        // A previous frame that was resolved but never read gives its buffer back (it was never mapped).
        if (this.toRead && this.toRead.state === 'encoded') this.toRead.state = 'free';
        frame.slot.state = 'encoded';
        frame.slot.labels = frame.labels;
        frame.slot.truncated = frame.truncated;
        frame.slot.meta = meta;
        this.toRead = frame.slot;
        this.lastSample = performance.now();
    }

    // Maps the buffer from the last resolve(); call after queue.submit. Resolves to the result or null.
    read() {
        if (!this.live || !this.toRead) return Promise.resolve(null);
        const slot = this.toRead;
        this.toRead = null;
        slot.state = 'mapping';
        const labels = slot.labels;
        const bytes = labels.length * 16;
        return slot.buffer.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
            const stamps = new BigUint64Array(slot.buffer.getMappedRange(0, bytes));
            const passes = {};
            let total = 0;
            for (let i = 0; i < labels.length; i++) {
                // Timestamps are nanoseconds; a reset or reordered counter can give end < begin, read as 0.
                const ns = Number(stamps[2 * i + 1] - stamps[2 * i]);
                const ms = ns > 0 ? ns / 1e6 : 0;
                passes[labels[i]] = (passes[labels[i]] || 0) + ms;
                total += ms;
            }
            slot.buffer.unmap();
            slot.state = 'free';
            const result = { passes, total, count: labels.length, truncated: slot.truncated, meta: slot.meta };
            this.accumulate(result);
            if (this.onResult) this.onResult(result);
            return result;
        }, () => {
            // Rejected because the timer (or device) was destroyed; nothing to report.
            slot.state = this.live ? 'free' : 'destroyed';
            return null;
        });
    }

    accumulate(result) {
        const a = this.emaAlpha;
        for (const [label, ms] of Object.entries(result.passes)) {
            const prev = this.ema[label];
            this.ema[label] = prev === undefined ? ms : prev + a * (ms - prev);
        }
        this.emaTotal = this.emaTotal === null ? result.total : this.emaTotal + a * (result.total - this.emaTotal);
        this.last = result;
        this.samples++;
    }

    resetAverages() {
        this.ema = {};
        this.emaTotal = null;
        this.last = null;
        this.samples = 0;
    }

    // Averaged GPU ms for one label, or for all passes when label is omitted; null before the first sample.
    ms(label) {
        if (label === undefined) return this.emaTotal;
        return this.ema[label] ?? null;
    }

    // "GPU: 1.23 ms", "GPU: — ms" before the first sample, or GPU_TIME_UNAVAILABLE.
    gpuText(label, digits = 2) {
        if (!this.supported) return GPU_TIME_UNAVAILABLE;
        return `GPU: ${formatMs(this.ms(label), digits)} ms`;
    }

    // "GPU: x.xx ms · wall: y.yy ms" (the wall part is omitted when wallMs is null/undefined).
    summary(wallMs, { label, digits = 2, wallLabel = 'wall' } = {}) {
        const gpu = this.gpuText(label, digits);
        return wallMs === null || wallMs === undefined ? gpu : `${gpu} · ${wallLabel}: ${formatMs(wallMs, digits)} ms`;
    }

    // Per-frame readout for animation loops, e.g.
    // "GPU: 1.23 ms/frame (sim 1.00 · render 0.23) · wall: 16.67 ms/frame". parts lists labels to break out;
    // wall is the frame interval from the example's FPS counter (omitted when fps is 0).
    frameText(fps, parts = []) {
        let gpu = GPU_TIME_UNAVAILABLE;
        if (this.supported) {
            const detail = parts.length && this.emaTotal !== null
                ? ` (${parts.map((label) => `${label} ${formatMs(this.ema[label])}`).join(' · ')})` : '';
            gpu = `GPU: ${formatMs(this.emaTotal)} ms/frame${detail}`;
        }
        return fps > 0 ? `${gpu} · wall: ${formatMs(1000 / fps)} ms/frame` : gpu;
    }

    destroy() {
        if (!this.live) return;
        this.live = false;
        this.frame = null;
        this.toRead = null;
        this.querySet.destroy();
        this.resolveBuffer.destroy();
        for (const slot of this.pool) slot.buffer.destroy();
    }
}
