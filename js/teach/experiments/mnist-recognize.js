// Measure experiments for MNIST recognition (js/examples/mnist-recognize.js, model in
// js/examples/mnist-model.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the model's compute-only API (evaluateWeights, measureInference): the
// shipped forward kernels through MNISTGPU, on the committed test images and cached weights
// (data/mnist/weights.bin). The CPU reference is the float64 forward pass of js/refs/mnist.js. Every
// sweep is a fixed list with hard caps (exported so test/mnist-recognize-measure.test.mjs can check
// them): ≤ 2,000 test images, batch ≤ 128, ≤ 10 timed submits per batch size.

import { evaluateWeights, measureInference, loadDataset, loadCachedWeights, MAX_BATCH } from '../../examples/mnist-model.js';
import { mnistForward, argmax } from '../../refs/mnist.js';

export const TRAINER_ACCURACY = 0.98;           // tools/train-mnist.mjs on the CPU: 1,960 / 2,000 (data/mnist/README.md)
export const PREFIXES = [125, 250, 500, 1000, 2000];
export const LATENCY_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128];
export const LATENCY_REPEATS = 10;

const pct = (v) => `${(100 * v).toFixed(2)}%`;
const ms = (v) => v.toFixed(3);

const testAccuracy = {
    id: 'test-accuracy',
    title: 'Test accuracy of the cached weights',
    question: 'Does the GPU reproduce the trainer\'s 98.00% on the 2,000 test images, and how precisely does a test set of a given size measure accuracy?',
    budgetMs: 20000,
    columns: [
        { key: 'images', label: 'test images' },
        { key: 'accuracy', label: 'GPU accuracy', format: pct },
        { key: 'low', label: '−2σ', format: pct },
        { key: 'high', label: '+2σ', format: pct },
        { key: 'trainer', label: 'trainer (2,000)', format: pct },
        { key: 'cpuAccuracy', label: 'CPU float64 accuracy', format: pct },
        { key: 'agree', label: 'GPU = CPU prediction', format: pct },
        { key: 'maxDiff', label: 'max |p_GPU − p_CPU|', format: (v) => v.toExponential(1) },
        { key: 'wallMs', label: 'GPU wall ms', format: (v) => v.toFixed(1) },
    ],
    charts: [{
        x: 'images', xLabel: 'test images evaluated', logX: true, y: ['accuracy', 'low', 'high', 'trainer'],
        yLabel: 'accuracy', labels: { accuracy: 'GPU', low: '−2σ', high: '+2σ', trainer: 'trainer 98.00%' },
    }],
    baseline: 'CI (shipped kernels on a software GPU): 1,960 / 2,000 = 98.00%, the same images as the CPU trainer, with GPU '
        + 'and float64 CPU predictions identical on all 2,000 and probabilities within about 1e-6. The ±2σ band is the binomial '
        + 'standard error √(p(1 − p)/n): with 2,000 images it is ±0.6 percentage points, so two models 0.3 points apart '
        + 'cannot be told apart on this test set.',
    async *run(ctx) {
        const [test, weights] = await Promise.all([loadDataset('test'), loadCachedWeights()]);
        // The float64 reference once for all 2,000 images (about a second), in slices so the page stays responsive
        const cpu = new Float64Array(test.count * 10);
        for (let i = 0; i < test.count; i++) {
            if (i % 250 === 0) await ctx.step();
            cpu.set(mnistForward(weights, test.images, i), 10 * i);
        }
        for (const images of PREFIXES) {
            await ctx.step();
            const r = await evaluateWeights(ctx.gpuDevice, { weights, data: test, count: Math.min(images, test.count), batch: MAX_BATCH });
            let agree = 0, cpuCorrect = 0, maxDiff = 0;
            for (let i = 0; i < r.count; i++) {
                const g = argmax(r.probs, 10 * i), c = argmax(cpu, 10 * i);
                if (g === c) agree++;
                if (c === test.labels[i]) cpuCorrect++;
                for (let k = 0; k < 10; k++) maxDiff = Math.max(maxDiff, Math.abs(r.probs[10 * i + k] - cpu[10 * i + k]));
            }
            const p = r.accuracy;
            const se = Math.sqrt(p * (1 - p) / r.count);
            yield { images: r.count, accuracy: p, low: p - 2 * se, high: Math.min(1, p + 2 * se), trainer: TRAINER_ACCURACY,
                cpuAccuracy: cpuCorrect / r.count, agree: agree / r.count, maxDiff, wallMs: r.wallMs };
        }
    },
};

const latency = {
    id: 'latency',
    title: 'Inference latency vs batch size',
    question: 'How long does one classification take from submit to answer, and how much cheaper per image does batching make it?',
    budgetMs: 20000,
    columns: [
        { key: 'batch', label: 'images per submit' },
        { key: 'wallMs', label: 'round trip ms (median)', format: ms },
        { key: 'gpuMs', label: 'GPU ms', format: ms },
        { key: 'perImageUs', label: 'µs per image (wall)', format: (v) => v.toFixed(1) },
        { key: 'imagesPerSecond', label: 'images / s', format: (v) => Math.round(v).toLocaleString('en-US') },
    ],
    charts: [
        { x: 'batch', xLabel: 'images per submit', logX: true, y: ['wallMs', 'gpuMs'], yLabel: 'ms per submit', logY: true,
            labels: { wallMs: 'round trip', gpuMs: 'GPU' } },
        { x: 'batch', xLabel: 'images per submit', logX: true, y: ['perImageUs'], yLabel: 'µs per image', logY: true },
    ],
    baseline: 'The Recognize page classifies one drawing per submit (batch 1, input written by the CPU): its latency is the '
        + 'round trip of a submit plus a 40-byte readback, which is mostly fixed cost. The forward pass is 8 dispatches '
        + 'whatever the batch, so until the GPU is full a batch of 128 costs little more than one image. Even on the '
        + 'CPU-emulated GPU in CI, where every image costs real CPU time, the cost per image fell about 3× from batch 1 to 128.',
    async *run(ctx) {
        const [test, weights] = await Promise.all([loadDataset('test'), loadCachedWeights()]);
        for (const batch of LATENCY_BATCHES) {
            if (batch > MAX_BATCH) break;
            await ctx.step();
            const r = await measureInference(ctx.gpuDevice, { weights, data: test, batch, warmup: 2, repeats: LATENCY_REPEATS });
            yield { batch, wallMs: r.wallMs, gpuMs: r.gpuMs, perImageUs: 1000 * r.perImageMs, imagesPerSecond: 1000 / r.perImageMs };
        }
    },
};

export default [testAccuracy, latency];
