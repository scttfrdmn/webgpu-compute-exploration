// Measure experiments for ray marching (js/examples/ray-marching.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureRayMarch(): the shipped render pipeline
// drawn into an offscreen texture (frame time) and a statistics kernel that replays rayMarch's loop per
// pixel and records how it ended (hit, escaped past t = 50, or out of steps). The reference is the float64
// sphere tracer of js/refs/ray-marching.js on SAMPLE pixels. Caps: ≤ 1000 × 700 pixels (the canvas) and
// ≤ 256 steps (the Quality slider's maximum).

import { measureRayMarch, FRACTALS, MAX_MARCH_STEPS } from '../../examples/ray-marching.js';
import { sceneDE, cameraRay, marchRay } from '../../refs/ray-marching.js';

export const STEPS = [32, 64, 96, 128, 160, 192, 224, 256];
export const RESOLUTIONS = [{ value: 500, label: '500 × 350' }, { value: 1000, label: '1000 × 700 (the canvas)' }];
export const SAMPLE = 200;
export const CAMERA = { cameraZoom: 3, cameraRotX: 0, cameraRotY: 0, time: 0 };   // a fresh renderer's view
export const POWER = 8;

// Evenly scattered pixel centers for the CPU comparison
export function samplePixels(w, h, count = SAMPLE) {
    return Array.from({ length: count }, (_, k) => [(k * 389 + 7) % w, Math.floor((k + 0.5) * h / count)]);
}

// Fraction of the sampled pixels whose GPU outcome (hit / escaped / out of steps) the float64 march repeats
export function agreement(pixels, w, h, fractalType, maxSteps, pts) {
    const sdf = sceneDE(fractalType, POWER);
    let same = 0;
    for (const [x, y] of pts) {
        const { ro, rd } = cameraRay(x + 0.5, y + 0.5, w, h, CAMERA);
        if (marchRay(sdf, ro, rd, maxSteps).outcome === pixels[4 * (y * w + x) + 2]) same++;
    }
    return same / pts.length;
}

const ms = (v) => v.toFixed(2);
const pct = (v) => `${(100 * v).toFixed(1)}%`;

const steps = {
    id: 'max-steps',
    title: 'Step budget vs cost and convergence',
    question: 'How many sphere-tracing steps does a pixel need before its ray converges, and what does a larger step budget cost per frame?',
    settings: [
        { key: 'fractal', label: 'Fractal', options: FRACTALS.map(f => ({ value: f, label: f[0].toUpperCase() + f.slice(1) })), default: 'mandelbulb' },
        { key: 'width', label: 'Resolution', options: RESOLUTIONS, default: 1000 },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'maxSteps', label: 'max steps' },
        { key: 'gpuMs', label: 'frame GPU ms', format: ms },
        { key: 'wallMs', label: 'frame wall ms', format: ms },
        { key: 'hit', label: 'hit', format: pct },
        { key: 'escaped', label: 'escaped', format: pct },
        { key: 'exhausted', label: 'out of steps', format: pct },
        { key: 'meanSteps', label: 'mean steps / pixel', format: (v) => v.toFixed(1) },
        { key: 'agreement', label: `agrees with float64 (${SAMPLE} px)`, format: pct },
    ],
    charts: [
        {
            x: 'maxSteps', xLabel: 'max steps (Quality)', y: ['gpuMs', 'wallMs'], yLabel: 'ms per frame',
            labels: { gpuMs: 'GPU', wallMs: 'wall' },
        },
        {
            x: 'maxSteps', xLabel: 'max steps (Quality)', y: ['hit', 'escaped', 'exhausted'], yLabel: 'fraction of pixels',
            labels: { hit: 'hit the surface', escaped: 'escaped (t > 50)', exhausted: 'out of steps' },
        },
    ],
    baseline: 'CI (shipped shader on a CPU-emulated GPU, Mandelbulb, 500 × 350): 45% of pixels run out of steps at 32 '
        + 'steps, 9.7% at 64, 0.8% at 128 and none from 192 on; 30.6% hit, 69.4% escape, 37.8 distance evaluations per '
        + 'pixel on average. The float64 tracer agrees on 100% of the sampled pixels. At 160 × 112 and 128 steps: Julia '
        + '4.3% out of steps, Menger 0.1%, Mandelbox 0%.',
    async *run(ctx, { fractal, width }) {
        const w = Math.min(width, 1000), h = Math.round(w * 0.7);
        const pts = samplePixels(w, h);
        for (const maxSteps of STEPS) {
            if (maxSteps > MAX_MARCH_STEPS) break;
            await ctx.step();
            const r = await measureRayMarch(ctx.gpuDevice, { width: w, height: h, fractal, power: POWER, maxSteps, repeats: 3, readback: true });
            await ctx.step();
            yield {
                maxSteps, gpuMs: r.gpuMs, wallMs: r.wallMs, hit: r.hit, escaped: r.escaped, exhausted: r.exhausted, meanSteps: r.meanSteps,
                agreement: agreement(r.pixels, w, h, FRACTALS.indexOf(fractal), maxSteps, pts),
            };
        }
    },
};

export default [steps];
