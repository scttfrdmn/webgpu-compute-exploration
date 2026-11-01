// Measure experiments for the path tracer (js/examples/path-tracer.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only API (measurePathTracer): the shipped cs_trace,
// one sample per pixel per dispatch, at a small fixed resolution. References: the closed furnace box,
// whose expected radiance is known exactly (js/refs/path-tracer.js furnaceRadiance), and for the
// Cornell box a high-sample GPU render with a different seed. Every sweep is a fixed list with hard
// caps (exported so test/path-tracer-measure.test.mjs can check them): ≤ 200 × 150 pixels, ≤ 2,048
// samples per pixel per call, ≤ 8 bounces.

import { measurePathTracer, FURNACE, MAX_BOUNCES } from '../../examples/path-tracer.js';
import { furnaceRadiance, imageRmse } from '../../refs/path-tracer.js';

export const NOISE_SIZE = [100, 75];
export const SPPS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
export const REF_SPP = 2048;               // Cornell reference; its own noise is √(256/2048) ≈ 35% of the 256-spp noise in RMS
export const BOUNCE_SIZE = [200, 150];
export const BOUNCES = [1, 2, 3, 4, 5, 6, 7, 8];
export const BOUNCE_SPP = 16;
const SEED = 1, REF_SEED = 2;
// Measured in CI (software GPU, furnace box ρ = 0.5, 5 bounces, 100 × 75, NEE on): relative RMSE at 1 spp
export const CI_FURNACE_RMSE_1 = 0.29;

const sci = (v) => v.toExponential(2);
const ms = (v) => v.toFixed(3);
const sceneSetting = {
    key: 'scene', label: 'Scene',
    options: [{ value: 'furnace', label: 'Furnace box (exact answer)' }, { value: 'cornell', label: 'Cornell box' }],
    default: 'furnace',
};

const sppNoise = {
    id: 'spp-noise',
    title: 'Samples per pixel vs noise',
    question: 'Does the Monte Carlo error really fall as 1/√N with the number of samples per pixel, and does next-event estimation lower the constant?',
    settings: [
        sceneSetting,
        { key: 'nee', label: 'Light sampling', options: [{ value: true, label: 'NEE + MIS' }, { value: false, label: 'BSDF only' }], default: true },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'spp', label: 'samples / pixel' },
        { key: 'relRmse', label: 'relative RMSE', format: sci },
        { key: 'rmse', label: 'RMSE', format: sci },
        { key: 'mean', label: 'mean radiance', format: (v) => v.toFixed(4) },
        { key: 'bias', label: 'mean − exact (furnace)', format: sci },
        { key: 'gpuMs', label: 'GPU ms / sample', format: ms },
        { key: 'wallMs', label: 'wall ms / sample', format: ms },
    ],
    charts: [{
        x: 'spp', xLabel: 'samples per pixel', logX: true, y: ['relRmse'], yLabel: 'relative RMS error per pixel', logY: true,
        // 1/√N through the visitor's own value at 1 spp
        reference: (rows) => {
            const first = rows.find(r => r.spp === 1)?.relRmse;
            return first > 0 ? { label: '1/√N', points: [1, SPPS[SPPS.length - 1]].map(n => [n, first / Math.sqrt(n)]) } : null;
        },
    }],
    baseline: `CI (shipped shader on a software GPU, furnace box ρ = ${FURNACE.rho}, 5 bounces, ${NOISE_SIZE.join(' × ')}): `
        + `relative RMSE ${CI_FURNACE_RMSE_1} at 1 spp and 0.018 at 256 with NEE + MIS, 0.33 and 0.021 with BSDF sampling only; `
        + 'the fitted slope is −0.500 in both cases, and the mean stays within its standard error of the exact '
        + `${furnaceRadiance(FURNACE.rho, FURNACE.le, 5).toFixed(5)}. The Cornell box is compared with a `
        + `${REF_SPP.toLocaleString('en-US')}-spp render of its own (another seed): 4.2 at 1 spp (fireflies from the glass `
        + 'sphere dominate) and 0.28 at 256, a slope of −0.47.',
    async *run(ctx, { scene, nee }) {
        const [width, height] = NOISE_SIZE;
        let reference;
        if (scene === 'furnace') {
            reference = furnaceRadiance(FURNACE.rho, FURNACE.le, 5);
        } else {
            await ctx.step();
            const ref = await measurePathTracer(ctx.gpuDevice, { scene, width, height, checkpoints: [REF_SPP],
                useNEE: true, seed: REF_SEED, between: () => ctx.step() });
            reference = ref.checkpoints[0].image;
        }
        await ctx.step();
        // One progressive run, read back at every checkpoint (rows arrive at the end of the run)
        const r = await measurePathTracer(ctx.gpuDevice, { scene, width, height, checkpoints: SPPS, bounces: scene === 'furnace' ? 5 : null,
            useNEE: nee, seed: SEED, between: () => ctx.step() });
        for (const c of r.checkpoints) {
            const e = imageRmse(c.image, reference);
            let mean = 0;
            for (const v of c.image) mean += v;
            mean /= c.image.length;
            yield { spp: c.spp, relRmse: e.relative, rmse: e.rmse, mean,
                bias: scene === 'furnace' ? mean - reference : null, gpuMs: c.gpuMsPerSample, wallMs: c.wallMsPerSample };
        }
    },
};

// ---- 2. Path length vs cost -------------------------------------------------------------------------
const bounceCost = {
    id: 'bounces',
    title: 'Bounces vs cost and energy',
    question: 'How much light do the longer paths carry, and what do they cost in rays and GPU time?',
    settings: [sceneSetting],
    budgetMs: 20000,
    columns: [
        { key: 'bounces', label: 'max bounces' },
        { key: 'raysPerPixel', label: 'rays / pixel / sample', format: (v) => v.toFixed(2) },
        { key: 'gpuMs', label: 'GPU ms / sample', format: ms },
        { key: 'wallMs', label: 'wall ms / sample', format: ms },
        { key: 'mean', label: 'mean radiance', format: (v) => v.toFixed(4) },
        { key: 'exact', label: 'exact (furnace)', format: (v) => v.toFixed(4) },
    ],
    charts: [
        { x: 'bounces', xLabel: 'maximum bounces', y: ['mean', 'exact'], yLabel: 'mean radiance',
            labels: { mean: 'measured', exact: 'exact Σ ρᵏ' } },
        { x: 'bounces', xLabel: 'maximum bounces', y: ['raysPerPixel'], yLabel: 'rays per pixel per sample' },
        { x: 'bounces', xLabel: 'maximum bounces', y: ['gpuMs', 'wallMs'], yLabel: 'ms per sample',
            labels: { gpuMs: 'GPU', wallMs: 'wall' } },
    ],
    baseline: `In the furnace box (ρ = ${FURNACE.rho}) each extra bounce adds ρᵏ: the mean climbs 1.5, 1.75, 1.875, … toward 2, `
        + 'and in CI the measured means match the exact sums to 3 decimals. Rays per pixel (camera, continuation and shadow '
        + 'rays) rise from 2.8 at 1 bounce to 5.9 at 5 and then level off, because Russian roulette ends paths after the '
        + 'third bounce. In the Cornell box (CI: 2.2 → 3.8 rays) most paths leave through the open front, and the bounces '
        + 'after the first add 34% to the mean.',
    async *run(ctx, { scene }) {
        const [width, height] = BOUNCE_SIZE;
        for (const bounces of BOUNCES) {
            if (bounces > MAX_BOUNCES) break;
            await ctx.step();
            const r = await measurePathTracer(ctx.gpuDevice, { scene, width, height, checkpoints: [BOUNCE_SPP], bounces,
                seed: SEED, between: () => ctx.step() });
            const c = r.checkpoints[0];
            let mean = 0;
            for (const v of c.image) mean += v;
            yield { bounces, raysPerPixel: c.raysPerPixel, gpuMs: c.gpuMsPerSample, wallMs: c.wallMsPerSample,
                mean: mean / c.image.length, exact: scene === 'furnace' ? furnaceRadiance(FURNACE.rho, FURNACE.le, bounces) : null };
        }
    },
};

export default [sppNoise, bounceCost];
