// Measure experiments for the Voronoi / jump flooding example (js/examples/voronoi-jfa.js). Interface:
// see the top of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureJFA), which runs the shipped clear,
// seed and flood passes (encodeCompute) and the shipped GPU brute-force comparison (exactCheck) without a
// canvas. The CPU reference is the float64 brute force of js/refs/voronoi.js (errorsVsBrute), used where
// it is cheap enough to run in the page. Every sweep is a fixed list with hard caps (exported so
// test/voronoi-measure.test.mjs can check them): grids ≤ 2048 × 2048, ≤ 1,024 seeds (the example's
// maximum), GPU exact checks ≤ 512² × 1,024 pixel-seed pairs, CPU checks ≤ 2²⁴ pairs.

import { measureJFA, MAX_SEEDS, MAX_MEASURE_SIDE, MAX_EXACT_WORK } from '../../examples/voronoi-jfa.js';
import { errorsVsBrute } from '../../refs/voronoi.js';
import { mulberry32 } from '../../refs/random.js';

export const SEED = 17;
export const ERROR_SIDE = 512;
export const SEED_COUNTS = [16, 64, 256, 1024];
export const TRIALS = 3;
export const CPU_CHECK_MAX = 2 ** 24;    // pixels × seeds for the in-page float64 check
export const SIDES = [256, 512, 1024, 2048];
export const RESOLUTION_SEEDS = 256;

export const LAYOUTS = { random: 'Uniform random', clusters: 'Clusters (Gaussian blobs)' };

// `count` seed positions (interleaved x, y in pixels) on a w × h grid, from a seeded generator:
// uniform, or clustered around 6 centers (σ = 6% of the shorter side, clamped to the grid). Every seed
// gets its own pixel: the seed pass stores one seed index per pixel, so of two seeds in one pixel the
// higher index would be lost, and its whole cell would count as JFA errors (see the Understand page).
export function makeSeeds(count, w, h, layout = 'random', seed = SEED) {
    if (count > w * h / 4) throw new RangeError('too many seeds for distinct pixels');
    const random = mulberry32(seed);
    const out = new Float32Array(2 * count);
    const clamp = (v, hi) => Math.min(Math.max(v, 0), hi - 0.01);
    const centers = Array.from({ length: 6 }, () => [w * (0.15 + 0.7 * random()), h * (0.15 + 0.7 * random())]);
    const sigma = 0.06 * Math.min(w, h);
    const taken = new Set();
    for (let i = 0; i < count; i++) {
        let x, y;
        do {
            if (layout === 'clusters') {
                const [cx, cy] = centers[i % centers.length];
                const r = sigma * Math.sqrt(-2 * Math.log(1 - random()));
                const a = 2 * Math.PI * random();
                x = clamp(cx + r * Math.cos(a), w);
                y = clamp(cy + r * Math.sin(a), h);
            } else {
                x = clamp(random() * w, w);
                y = clamp(random() * h, h);
            }
            x = Math.fround(x);   // the pixel the shader's floor() of the f32 position picks
            y = Math.fround(y);
        } while (taken.has(Math.floor(y) * w + Math.floor(x)));
        taken.add(Math.floor(y) * w + Math.floor(x));
        out[2 * i] = x;
        out[2 * i + 1] = y;
    }
    return out;
}

const METRICS = [{ value: 0, label: 'Euclidean' }, { value: 1, label: 'Manhattan' }, { value: 2, label: 'Chebyshev' }];
const pct = (v) => `${(100 * v).toFixed(4)}%`;
const ms = (v) => v.toFixed(2);

// ---- 1. Seeds × metric × (JFA, JFA+1) → error rate ---------------------------------------------------
const errorRate = {
    id: 'error-rate',
    title: 'JFA vs JFA+1: error rate against brute force',
    question: 'On what fraction of pixels does jump flooding pick a seed that is not the nearest, and how much does the extra step-1 pass of JFA+1 fix?',
    settings: [
        { key: 'metric', label: 'Metric', options: METRICS, default: 0 },
        { key: 'layout', label: 'Seeds', options: Object.entries(LAYOUTS).map(([value, label]) => ({ value, label })), default: 'random' },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'seeds', label: 'seeds' },
        { key: 'passes', label: 'flood passes (JFA)' },
        { key: 'jfaRate', label: 'JFA error rate', format: pct },
        { key: 'plusOneRate', label: 'JFA+1 error rate', format: pct },
        { key: 'jfaWrong', label: 'JFA wrong pixels' },
        { key: 'plusOneWrong', label: 'JFA+1 wrong pixels' },
        { key: 'cpuWrong', label: 'JFA wrong (float64 CPU check)' },
        { key: 'maxExcess', label: 'largest excess distance, px', format: (v) => v.toFixed(2) },
    ],
    charts: [{
        x: 'seeds', xLabel: 'seeds', logX: true, y: ['jfaRate', 'plusOneRate'], yLabel: 'fraction of pixels with a farther seed', logY: true,
        labels: { jfaRate: 'JFA', plusOneRate: 'JFA+1' },
        emptyNote: 'No errors at all: every pixel got its nearest seed (a log axis cannot show zero).',
    }],
    baseline: 'CI (software GPU, 512 × 512, 3 seed sets per row, uniform seeds), wrong pixels JFA / JFA+1 out of 786,432: '
        + 'Euclidean 0 / 0, 4 / 0, 32 / 2, 102 / 14 at 16, 64, 256, 1,024 seeds (0.013% / 0.0018% at 1,024); Manhattan 0 / 0, '
        + '0 / 0, 2 / 0, 40 / 18; Chebyshev 10 / 1, 14 / 6, 80 / 26, 122 / 12. The GPU check and the float64 CPU check counted '
        + 'the same pixels wherever both ran (up to 64 seeds), and the wrong seed was at most 1.4 px farther than the nearest.',
    async *run(ctx, { metric, layout }) {
        const w = ERROR_SIDE, h = ERROR_SIDE, pixels = w * h;
        for (const count of SEED_COUNTS) {
            if (count > MAX_SEEDS || pixels * count > MAX_EXACT_WORK) break;
            const row = { seeds: count, passes: 0, jfaWrong: 0, plusOneWrong: 0, cpuWrong: null, maxExcess: null };
            const cpuCheck = pixels * count <= CPU_CHECK_MAX;
            for (let trial = 0; trial < TRIALS; trial++) {
                const seeds = makeSeeds(count, w, h, layout, SEED + 101 * trial + count);
                for (const plusOne of [false, true]) {
                    await ctx.step();
                    const r = await measureJFA(ctx.gpuDevice, { width: w, height: h, seeds, metric, plusOne, warmup: 0, repeats: 1,
                        check: true, readIds: cpuCheck && !plusOne });
                    if (plusOne) row.plusOneWrong += r.wrong; else { row.jfaWrong += r.wrong; row.passes = r.passes; }
                    if (r.ids) {
                        await ctx.step();
                        const c = errorsVsBrute(r.ids, w, h, seeds, metric);
                        row.cpuWrong = (row.cpuWrong ?? 0) + c.wrong;
                        row.maxExcess = Math.max(row.maxExcess ?? 0, c.maxExcess);
                    }
                }
            }
            row.jfaRate = row.jfaWrong / (pixels * TRIALS);
            row.plusOneRate = row.plusOneWrong / (pixels * TRIALS);
            yield row;
        }
    },
};

// ---- 2. Resolution → time -------------------------------------------------------------------------
const resolution = {
    id: 'resolution',
    title: 'Grid size vs time',
    question: 'JFA does log₂N full-grid passes: how does its time grow with the number of pixels, and what does each pixel-pass cost?',
    settings: [{ key: 'plusOne', label: 'Variant', options: [{ value: false, label: 'JFA' }, { value: true, label: 'JFA+1' }], default: false }],
    budgetMs: 20000,
    columns: [
        { key: 'side', label: 'grid (side × side)' },
        { key: 'pixels', label: 'pixels' },
        { key: 'passes', label: 'flood passes' },
        { key: 'gpuMs', label: 'GPU ms', format: ms },
        { key: 'wallMs', label: 'wall ms', format: ms },
        { key: 'nsPerPixelPass', label: 'GPU ns per pixel-pass', format: (v) => v.toFixed(3) },
    ],
    charts: [{
        x: 'pixels', xLabel: 'pixels', logX: true, y: ['gpuMs', 'wallMs'], yLabel: `ms per diagram (${RESOLUTION_SEEDS} seeds)`, logY: true,
        labels: { gpuMs: 'GPU (timestamps)', wallMs: 'wall (submit → done)' },
        reference: (rows) => {
            const first = rows.find(r => r.wallMs > 0);
            if (!first) return null;
            const c = first.wallMs / (first.pixels * first.passes);
            return { label: '∝ pixels · log₂N (wall)', points: rows.map(r => [r.pixels, c * r.pixels * r.passes]) };
        },
    }],
    baseline: 'CI (CPU-emulated GPU, 256 seeds, JFA): wall 9.8 ms at 256² (8 passes), 16 ms at 512², 63 ms at 1,024² and '
        + '261 ms at 2,048² (11 passes): about 4× per doubling of the side, as pixels × passes predicts once fixed costs fade.',
    async *run(ctx, { plusOne }) {
        for (const side of SIDES) {
            if (side > MAX_MEASURE_SIDE) break;
            const seeds = makeSeeds(RESOLUTION_SEEDS, side, side, 'random', SEED);
            await ctx.step();
            const r = await measureJFA(ctx.gpuDevice, { width: side, height: side, seeds, plusOne, warmup: 1, repeats: 3 });
            const pixels = side * side;
            yield { side, pixels, passes: r.passes, gpuMs: r.gpuMs, wallMs: r.wallMs,
                nsPerPixelPass: r.gpuMs ? r.gpuMs * 1e6 / (pixels * r.passes) : null };
        }
    },
};

export default [errorRate, resolution];
