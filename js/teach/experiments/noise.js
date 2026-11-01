// Measure experiments for procedural noise (js/examples/noise.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measureNoise), which runs the shipped
// noise pipeline without a canvas; the CPU references are the float64 Perlin and simplex noise of
// js/refs/noise.js, summed as fBm with the same octave rule. Every sweep is a fixed list with hard caps
// (exported so test/noise-measure.test.mjs can check them): fields up to 2048², at most 8 octaves,
// 4,096 CPU samples per row.

import { measureNoise, SIMPLEX_SCALE, SIMPLEX_FALLOFF, MAX_OCTAVES } from '../../examples/noise.js';
import { perlinRef, simplexRef, fbmRef } from '../../refs/noise.js';

export const LACUNARITY = 2.0, GAIN = 0.5;

// ---- 1. Octaves and resolution vs time -------------------------------------------------------------
export const OCTAVES = [1, 2, 3, 4, 6, 8];
export const SIDES = [512, 1024, 2048];
export const REPEATS = 5;

const ms = (v) => v.toFixed(3);

const cost = {
    id: 'cost',
    title: 'Octaves and resolution vs time',
    question: 'Does the cost of a noise field grow linearly with the number of octaves, and is simplex noise cheaper than Perlin?',
    settings: [
        { key: 'side', label: 'Field', options: SIDES.map(s => ({ value: s, label: `${s} × ${s}` })), default: 1024 },
        { key: 'warp', label: 'Domain warp', options: [{ value: 0, label: 'off (1 fBm per pixel)' }, { value: 1, label: 'on (3 fBm per pixel)' }], default: 0 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'octaves', label: 'octaves' },
        { key: 'perlinGpu', label: 'Perlin GPU ms', format: ms },
        { key: 'simplexGpu', label: 'simplex GPU ms', format: ms },
        { key: 'perlinWall', label: 'Perlin wall ms', format: ms },
        { key: 'simplexWall', label: 'simplex wall ms', format: ms },
        { key: 'perlinRate', label: 'Perlin M noise calls / s', format: (v) => (v / 1e6).toFixed(0) },
        { key: 'simplexRate', label: 'simplex M noise calls / s', format: (v) => (v / 1e6).toFixed(0) },
    ],
    charts: [
        { x: 'octaves', xLabel: 'octaves', y: ['perlinGpu', 'simplexGpu'], yLabel: 'GPU ms per field',
            labels: { perlinGpu: 'Perlin', simplexGpu: 'simplex' },
            emptyNote: 'GPU time needs the timestamp-query feature, which this browser or GPU does not offer; the wall-time chart below still applies.' },
        { x: 'octaves', xLabel: 'octaves', y: ['perlinWall', 'simplexWall'], yLabel: 'wall ms per field',
            labels: { perlinWall: 'Perlin', simplexWall: 'simplex' } },
        { x: 'octaves', xLabel: 'octaves', y: ['perlinRate', 'simplexRate'], yLabel: 'noise calls per second',
            labels: { perlinRate: 'Perlin', simplexRate: 'simplex' } },
    ],
    baseline: 'CI, on a CPU-emulated GPU (512², no warp, wall time): Perlin 7.0 ms per field with 1 octave and 17 ms with 8, '
        + 'simplex 4.4 and 23 ms. A software GPU has a large fixed cost per pass, so the growth is far from linear there; '
        + 'on a real GPU at 1024² or more the time should be close to proportional to the octaves. Measure the Perlin / '
        + 'simplex ratio on yours: simplex has 4 corners instead of 8, but a falloff and a branchy tetrahedron search.',
    async *run(ctx, { side = 1024, warp = 0 }) {
        const n = Math.min(side, SIDES[SIDES.length - 1]);
        for (const octaves of OCTAVES) {
            const row = { octaves };
            for (const [key, noiseType] of [['perlin', 0], ['simplex', 1]]) {
                await ctx.step();
                const r = await measureNoise(ctx.gpuDevice, { width: n, height: n, noiseType, octaves, frequency: 2.5,
                    lacunarity: LACUNARITY, gain: GAIN, warp: warp ? 0.4 : 0, warmup: 1, repeats: REPEATS });
                row[`${key}Gpu`] = r.gpuMs;
                row[`${key}Wall`] = r.wallMs;
                row[`${key}Rate`] = r.evaluations / (r.gpuMs ?? r.wallMs) * 1e3;
            }
            yield row;
        }
    },
};

// ---- 2. Statistics and accuracy vs the float64 reference ---------------------------------------------
export const STAT_SIDE = 256;
export const STAT_STRIDE = 4;               // every 4th pixel in x and y: 64 × 64 = 4,096 CPU samples
export const STAT_OCTAVES = [1, 2, 3, 4, 6, 8];
export const STAT_FREQUENCY = 8;

// The noise-space point of pixel (x, y), in float64, exactly as cs_main computes it (Reset view)
export function noisePoint(x, y, side, frequency) {
    const upp = 1 / side;
    return [((x + 0.5) - 0.5 * side) * upp * frequency, ((y + 0.5) - 0.5 * side) * upp * frequency, 0.5];
}

export function moments(values) {
    let sum = 0, min = Infinity, max = -Infinity;
    for (const v of values) { sum += v; if (v < min) min = v; if (v > max) max = v; }
    const mean = sum / values.length;
    let sq = 0;
    for (const v of values) sq += (v - mean) ** 2;
    return { mean, std: Math.sqrt(sq / values.length), min, max };
}

// fBm of independent octaves with equal spread σ: σ · √(Σ gain²ᵒ) / Σ gainᵒ
export const fbmStdFactor = (octaves, gain = GAIN) => {
    let a = 1, s1 = 0, s2 = 0;
    for (let o = 0; o < octaves; o++) { s1 += a; s2 += a * a; a *= gain; }
    return Math.sqrt(s2) / s1;
};

const sci = (v) => v.toExponential(2);
const f3 = (v) => v.toFixed(3);

const statistics = {
    id: 'statistics',
    title: 'Field statistics vs the float64 reference',
    question: 'Does the GPU field match a float64 evaluation of the same noise, and how does adding octaves change its spread?',
    settings: [{ key: 'noiseType', label: 'Noise', options: [{ value: 0, label: 'Perlin' }, { value: 1, label: 'Simplex' }], default: 0 }],
    budgetMs: 20000,
    columns: [
        { key: 'octaves', label: 'octaves' },
        { key: 'mean', label: 'GPU mean', format: f3 },
        { key: 'std', label: 'GPU std', format: f3 },
        { key: 'min', label: 'GPU min', format: f3 },
        { key: 'max', label: 'GPU max', format: f3 },
        { key: 'cpuStd', label: 'float64 std (samples)', format: f3 },
        { key: 'predictedStd', label: 'std if octaves were independent', format: f3 },
        { key: 'maxError', label: 'max |GPU − float64|', format: sci },
    ],
    charts: [
        { x: 'octaves', xLabel: 'octaves', y: ['std', 'cpuStd', 'predictedStd'], yLabel: 'standard deviation of the field',
            labels: { std: 'GPU (all pixels)', cpuStd: 'float64 (samples)', predictedStd: 'independent octaves' } },
        { x: 'octaves', xLabel: 'octaves', y: ['maxError'], yLabel: 'max |GPU − float64| on the samples', logY: true },
    ],
    baseline: 'CI (shipped shader on a software GPU, 256², frequency 8): max |GPU − float64| on the samples is 1.0e-7 for one '
        + 'octave of Perlin noise, rising to 4.0e-6 with 8 octaves (simplex: 1.4e-6 to 1.5e-5). The standard deviation '
        + 'falls from 0.273 to 0.158 (Perlin) and from 0.369 to 0.210 (simplex), within 2% of the independent-octave '
        + 'estimate.',
    async *run(ctx, { noiseType = 0 }) {
        const side = STAT_SIDE;
        const base = noiseType === 1 ? (x, y, z) => simplexRef(x, y, z, SIMPLEX_SCALE, SIMPLEX_FALLOFF) : perlinRef;
        let sigma1 = null;
        for (const octaves of STAT_OCTAVES) {
            if (octaves > MAX_OCTAVES) break;
            await ctx.step();
            const r = await measureNoise(ctx.gpuDevice, { width: side, height: side, noiseType, octaves, frequency: STAT_FREQUENCY,
                lacunarity: LACUNARITY, gain: GAIN, warmup: 0, repeats: 1, readback: true });
            const gpu = moments(r.field);
            const cpu = [];
            let maxError = 0;
            for (let y = 0; y < side; y += STAT_STRIDE) {
                for (let x = 0; x < side; x += STAT_STRIDE) {
                    const [px, py, pz] = noisePoint(x, y, side, STAT_FREQUENCY);
                    const v = fbmRef(base, px, py, pz, { octaves, lacunarity: LACUNARITY, gain: GAIN });
                    cpu.push(v);
                    maxError = Math.max(maxError, Math.abs(r.field[y * side + x] - v));
                }
            }
            const c = moments(cpu);
            if (octaves === 1) sigma1 = c.std;
            yield { octaves, mean: gpu.mean, std: gpu.std, min: gpu.min, max: gpu.max, cpuStd: c.std,
                predictedStd: sigma1 === null ? null : sigma1 * fbmStdFactor(octaves), maxError };
        }
    },
};

export default [cost, statistics];
