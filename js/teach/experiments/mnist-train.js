// Measure experiments for MNIST training (js/examples/mnist-train.js, model in js/examples/mnist-model.js).
// Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the model's compute-only API (measureTraining), which runs the shipped
// kernels through MNISTGPU from the seeded He initialization, exactly as the trainer does, on the
// committed MNIST subset. The CPU reference is the float64 forward pass of js/refs/mnist.js, run on a
// slice of the test set with the GPU-trained weights. Every sweep is a fixed list with hard caps
// (exported so test/mnist-train-measure.test.mjs can check them): ≤ 100 mini-batches per run and ≤ 500
// per experiment, batch ≤ 128.

import { measureTraining, loadDataset, DEFAULTS, MAX_BATCH } from '../../examples/mnist-model.js';
import { mnistForward, argmax } from '../../refs/mnist.js';

export const LEARNING_RATES = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15];
export const LR_BATCHES = 80;                 // 80 × 64 = 5,120 images, about half an epoch
export const CPU_CHECK = 200;                 // test images re-run on the CPU per row
export const BATCH_SIZES = [8, 16, 32, 64, 128];
export const SPEED_BATCHES = 16;              // timed mini-batches per batch size (after a warm-up submit)
// Measured in CI (software GPU, seed 1, batch 64, 80 mini-batches): test accuracy per learning rate,
// equal to the CPU trainer tools/train-mnist.mjs on the same run within a few images
export const CI_ACCURACY = { 0.005: 0.764, 0.01: 0.823, 0.02: 0.872, 0.04: 0.9115, 0.08: 0.904, 0.15: 0.740 };

const pct = (v) => `${(100 * v).toFixed(2)}%`;
const ms = (v) => v.toFixed(2);

// GPU vs float64 CPU predictions on the first `count` test images
function cpuAgreement(weights, test, probs, count) {
    let same = 0, maxDiff = 0;
    for (let i = 0; i < count; i++) {
        const p = mnistForward(weights, test.images, i);
        if (argmax(p) === argmax(probs, 10 * i)) same++;
        for (let k = 0; k < 10; k++) maxDiff = Math.max(maxDiff, Math.abs(p[k] - probs[10 * i + k]));
    }
    return { agree: same / count, maxDiff };
}

const learningRate = {
    id: 'learning-rate',
    title: 'Learning rate vs test accuracy',
    question: `Which learning rate learns fastest in a short run of ${LR_BATCHES} mini-batches, and what happens when it is too small or too large?`,
    budgetMs: 20000,
    columns: [
        { key: 'lr', label: 'learning rate η', format: String },
        { key: 'testAccuracy', label: 'test accuracy', format: pct },
        { key: 'testLoss', label: 'test loss', format: (v) => v.toFixed(3) },
        { key: 'trainLoss', label: 'train loss (last 8 batches)', format: (v) => v.toFixed(3) },
        { key: 'ciAccuracy', label: 'CI accuracy', format: pct },
        { key: 'cpuAgree', label: `CPU agrees (first ${CPU_CHECK})`, format: pct },
        { key: 'gpuMs', label: 'GPU ms / batch', format: ms },
        { key: 'wallMs', label: 'wall ms / batch', format: ms },
    ],
    charts: [
        { x: 'lr', xLabel: 'learning rate η', logX: true, y: ['testAccuracy', 'ciAccuracy'], yLabel: 'test accuracy (2,000 images)',
            labels: { testAccuracy: 'your GPU', ciAccuracy: 'CI' } },
        { x: 'lr', xLabel: 'learning rate η', logX: true, y: ['testLoss', 'trainLoss'], yLabel: 'cross-entropy loss', logY: true,
            labels: { testLoss: 'test', trainLoss: 'train' } },
    ],
    baseline: `CI (shipped kernels on a software GPU, seed ${DEFAULTS.seed}, batch ${DEFAULTS.batch}, ${LR_BATCHES} mini-batches): `
        + 'test accuracy 76% at η = 0.005, 87% at 0.02, 91% at 0.04 (the default), 90% at 0.08 and 74% at 0.15, the same '
        + 'as the CPU trainer on the identical run. Training is deterministic for a seed, so your GPU should land within a '
        + 'few images of these; larger differences come from f32 rounding growing through the run, most at the unstable η = 0.15.',
    async *run(ctx) {
        const [train, test] = await Promise.all([loadDataset('train'), loadDataset('test')]);
        for (const lr of LEARNING_RATES) {
            await ctx.step();
            const r = await measureTraining(ctx.gpuDevice, { train, test, batch: DEFAULTS.batch, lr, batches: LR_BATCHES,
                slots: 8, between: () => ctx.step() });
            await ctx.step();
            const check = cpuAgreement(r.weights, test, r.test.probs, CPU_CHECK);
            yield { lr, testAccuracy: r.test.accuracy, testLoss: r.test.loss, trainLoss: r.trainLoss,
                ciAccuracy: CI_ACCURACY[lr] ?? null, cpuAgree: check.agree, gpuMs: r.gpuMsPerBatch, wallMs: r.wallMsPerBatch };
        }
    },
};

const batchSpeed = {
    id: 'batch-speed',
    title: 'Batch size vs time per mini-batch',
    question: 'How does the cost of one SGD step grow with the batch size, and how many images per second does that give?',
    budgetMs: 20000,
    columns: [
        { key: 'batch', label: 'batch size' },
        { key: 'gpuMs', label: 'GPU ms / batch', format: ms },
        { key: 'wallMs', label: 'wall ms / batch', format: ms },
        { key: 'imagesPerSecond', label: 'images / s (wall)', format: (v) => Math.round(v).toLocaleString('en-US') },
        { key: 'trainLoss', label: 'train loss', format: (v) => v.toFixed(3) },
    ],
    charts: [
        { x: 'batch', xLabel: 'batch size', logX: true, y: ['gpuMs', 'wallMs'], yLabel: 'ms per mini-batch', logY: true,
            labels: { gpuMs: 'GPU', wallMs: 'wall' } },
        { x: 'batch', xLabel: 'batch size', logX: true, y: ['imagesPerSecond'], yLabel: 'training images per second', logY: true },
    ],
    baseline: 'A mini-batch is 14 dispatches whatever its size, plus one sum per parameter over the batch in cs_update, so small '
        + 'batches are dominated by fixed cost and the throughput grows with the batch. In CI, on a CPU-emulated GPU, the time '
        + 'per batch grows almost in proportion to the batch instead (throughput rose only 1.2–1.9× from batch 8 to 128).',
    async *run(ctx) {
        const train = await loadDataset('train');
        for (const batch of BATCH_SIZES) {
            if (batch > MAX_BATCH) break;
            await ctx.step();
            // One warm-up submit (pipelines, first use of the buffers), then the timed ones
            await measureTraining(ctx.gpuDevice, { train, batch, batches: 8, slots: 8 });
            await ctx.step();
            const r = await measureTraining(ctx.gpuDevice, { train, batch, batches: SPEED_BATCHES, slots: 8,
                between: () => ctx.step() });
            yield { batch, gpuMs: r.gpuMsPerBatch, wallMs: r.wallMsPerBatch, imagesPerSecond: 1000 * batch / r.wallMsPerBatch,
                trainLoss: r.trainLoss };
        }
    },
};

export default [learningRate, batchSpeed];
