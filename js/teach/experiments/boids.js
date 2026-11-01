// Measure experiments for boids (js/examples/boids.js). Interface: see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureBoids(), which runs the shipped flocking
// kernel with the example's buffers and default parameters in a 1000 × 700 world. The reference is the
// float64 step of js/refs/boids.js for SAMPLE boids, and its polarization() for the order parameter.
// Caps: N ≤ 10,000 (the slider's maximum), ≤ 1,200 steps per run, ≤ 60 steps and ≤ 1e9 distance tests per submit.

import { measureBoids, MAX_BOIDS, WORLD } from '../../examples/boids.js';
import { boidsStepAt, polarization } from '../../refs/boids.js';
import { mulberry32 } from '../../refs/random.js';

export const SEED = 21;
export const MAX_SPEED = 4;   // BoidsSimulation's initial maxSpeed (px per step)

// As initializeBoids: positions uniform over the world, velocity components uniform in ±maxSpeed/2
export function randomBoids(n, seed = SEED) {
    const rand = mulberry32(seed);
    const boids = new Float32Array(4 * n);
    for (let i = 0; i < n; i++) {
        boids[4 * i] = rand() * WORLD.width;
        boids[4 * i + 1] = rand() * WORLD.height;
        boids[4 * i + 2] = (rand() - 0.5) * MAX_SPEED;
        boids[4 * i + 3] = (rand() - 0.5) * MAX_SPEED;
    }
    return boids;
}

// ---- 1. Scaling ----------------------------------------------------------------------------------------
export const COUNTS = [250, 500, 1000, 2000, 4000, 8000, 10000];
export const SCALING_STEPS = 30;
export const SAMPLE = 64;

export const sampleTargets = (n, count = SAMPLE) => Array.from({ length: Math.min(count, n) }, (_, k) => Math.floor(k * n / Math.min(count, n)));

// Largest position difference (pixels) between one GPU step and the float64 step, over the sampled boids.
// A boid that wrapped around the world on one side only would differ by a world width; such pairs are
// compared modulo the world size.
export function stepError(before, after, n, params, targets) {
    const ref = boidsStepAt(before, n, params, targets);
    let worst = 0;
    targets.forEach((i, t) => {
        let dx = Math.abs(after[4 * i] - ref[4 * t]), dy = Math.abs(after[4 * i + 1] - ref[4 * t + 1]);
        dx = Math.min(dx, Math.abs(dx - WORLD.width));
        dy = Math.min(dy, Math.abs(dy - WORLD.height));
        worst = Math.max(worst, dx, dy);
    });
    return worst;
}

const ms = (v) => v.toFixed(3);

const scaling = {
    id: 'scaling',
    title: 'Boids vs time per step',
    question: 'How does the all-pairs neighbour search scale with N, and how many distance tests per second does your GPU run?',
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'boids N' },
        { key: 'gpuMsPerStep', label: 'GPU ms / step', format: ms },
        { key: 'wallMsPerStep', label: 'wall ms / step', format: ms },
        { key: 'testsGpu', label: 'G distance tests/s (GPU)', format: (v) => v.toFixed(2) },
        { key: 'testsWall', label: 'G distance tests/s (wall)', format: (v) => v.toFixed(2) },
        { key: 'stepError', label: 'one step vs float64 (px)', format: (v) => v.toExponential(1) },
    ],
    charts: [
        {
            x: 'n', xLabel: 'boids N', logX: true, y: ['gpuMsPerStep', 'wallMsPerStep'], yLabel: 'ms per step', logY: true,
            labels: { gpuMsPerStep: 'GPU', wallMsPerStep: 'wall' },
            reference: (rows) => {
                const last = rows.filter(r => r.wallMsPerStep > 0).at(-1);
                return last ? { label: 'N²', points: [COUNTS[0], last.n].map(n => [n, last.wallMsPerStep * (n / last.n) ** 2]) } : { label: 'N²', points: [] };
            },
        },
        {
            x: 'n', xLabel: 'boids N', logX: true, y: ['testsGpu', 'testsWall'], yLabel: 'billion distance tests per second (3N² per step)', logY: true,
            labels: { testsGpu: 'GPU time', testsWall: 'wall' },
        },
    ],
    baseline: 'Separation, alignment and cohesion each loop over all N boids, so a step is 3N² distance tests: '
        + '300 million at the slider\'s 10,000. The float64 comparison checks 64 boids after one step from the same start. '
        + 'CI (CPU-emulated GPU): 0.36 ms per step at N = 250, 532 ms at 10,000 (0.4–0.7 billion tests/s); one-step '
        + 'difference 3e-5 px, the f32 rounding of coordinates up to 1,000.',
    async *run(ctx) {
        for (const n of COUNTS) {
            if (n > MAX_BOIDS) break;
            const boids = randomBoids(n);
            await ctx.step();
            const one = await measureBoids(ctx.gpuDevice, { boids, steps: 1, samples: 1 });
            await ctx.step();
            const r = await measureBoids(ctx.gpuDevice, { boids, steps: SCALING_STEPS });
            const tests = 3 * n * n;
            const rate = (t) => (t > 0 ? tests / t / 1e6 : null);
            yield {
                n, gpuMsPerStep: r.gpuMsPerStep, wallMsPerStep: r.wallMsPerStep, testsGpu: rate(r.gpuMsPerStep), testsWall: rate(r.wallMsPerStep),
                stepError: stepError(boids, one.snapshots[1].boids, n, one.params, sampleTargets(n)),
            };
        }
    },
};

// ---- 2. Polarization (order parameter) ---------------------------------------------------------------------
export const ORDER_COUNTS = [500, 2000, 4000];
export const ORDER_STEPS = 1200;        // 20 s of the interactive example at 60 steps per second
export const ORDER_SAMPLES = 40;

const order = {
    id: 'polarization',
    title: 'Emergence of order',
    question: 'How quickly do randomly oriented boids align into flocks, measured by the polarization |Σ v̂| / N, and does it depend on N?',
    settings: [{ key: 'n', label: 'Boids', options: ORDER_COUNTS.map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 2000 }],
    budgetMs: 20000,
    columns: [
        { key: 'step', label: 'step' },
        { key: 'seconds', label: 'seconds at 60 steps/s', format: (v) => v.toFixed(1) },
        { key: 'polarization', label: 'polarization', format: (v) => v.toFixed(3) },
        { key: 'random', label: 'random headings 1/√N', format: (v) => v.toFixed(3) },
    ],
    charts: [{
        x: 'step', xLabel: 'step', y: ['polarization', 'random'], yLabel: 'polarization |Σ v̂| / N',
        labels: { polarization: 'GPU flock', random: 'random headings (1/√N)' },
    }],
    baseline: 'The polarization starts near 1/√N (random headings). Alignment acts only within 50 px, so flocks form '
        + 'locally with different headings first; the global value rises as they merge. CI (N = 500, seed 21): 0.08 at '
        + 'the start, 0.36 after 150 steps, 0.65 after 300, 0.92 after 600, then 0.87–0.92 as flocks meet and split.',
    async *run(ctx, { n }) {
        const N = Math.min(n, MAX_BOIDS);
        let boids = randomBoids(N);
        const every = ORDER_STEPS / ORDER_SAMPLES;
        const row = (step) => ({ step, seconds: step / 60, polarization: polarization(boids, N), random: 1 / Math.sqrt(N) });
        yield row(0);
        for (let s = 1; s <= ORDER_SAMPLES; s++) {
            await ctx.step();
            const r = await measureBoids(ctx.gpuDevice, { boids, steps: every, samples: 1 });
            boids = r.snapshots[1].boids;
            yield row(s * every);
        }
    },
};

export default [scaling, order];
