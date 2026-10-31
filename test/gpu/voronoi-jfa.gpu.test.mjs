// Voronoi JFA (js/examples/voronoi-jfa.js) on a software GPU: the shipped cs_clear / cs_seed /
// cs_flood / cs_exact WGSL, dispatched in the order VoronoiJFA.encodeCompute() + exactCheck() use,
// with the Params / Seed bytes captured from the shipped writers (writeParams / writeSeeds) through
// a mock device. CI-only — see test/gpu/harness.mjs.
//
// The example binds one Params slot per flood pass at a dynamic uniform offset; here each pass gets
// a bind group whose (static) buffer binding starts at that same slot offset — the shader sees the
// same 32 bytes either way.
//
// Compared with: the float64 brute-force nearest seed (error rate within the CPU-suite bounds from
// test/helpers/voronoi-refs.mjs) and the CPU replica (seed ids; differences only at f32 near-ties).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, write, read, run } from './harness.mjs';
import { jfaSteps, MAX_SEEDS, wgslModules } from '../../js/examples/voronoi-jfa.js';
import { moduleCode } from '../helpers/wgsl-layout.mjs';
import {
    NONE, PARAMS, SEED, makeJFA, generateSeeded, uploadedSeeds, seedPass, floodPass, errorsVsBrute,
    MAX_ERROR_RATE, MAX_ERROR_RATE_PLUS_ONE,
} from '../helpers/voronoi-refs.mjs';

const mod = { wgslModules };
const METRIC = ['euclid', 'manhattan', 'chebyshev'];

const pipes = {
    clear: await computePipeline(moduleCode(mod, 'seed'), 'cs_clear'),
    seed: await computePipeline(moduleCode(mod, 'seed'), 'cs_seed'),
    flood: await computePipeline(moduleCode(mod, 'flood'), 'cs_flood'),
    exact: await computePipeline(moduleCode(mod, 'exact'), 'cs_exact'),
};

// Runs the shipped compute sequence on the GPU; returns { ids, counters } (counters from cs_exact).
async function gpuJFA(device, jfa, W, H) {
    const steps = jfaSteps(W, H, jfa.plusOne);
    jfa.writeParams(steps);
    const paramBytes = device.lastWrite(jfa.paramsBuffer).bytes;
    assert.equal(paramBytes.byteLength, steps.length * jfa.paramsStride);
    jfa.writeSeeds();
    const seedBytes = new Uint8Array(SEED.size * MAX_SEEDS);
    seedBytes.set(device.lastWrite(jfa.seedBuffer).bytes);

    const params = buffer(paramBytes);
    const seeds = buffer(seedBytes);
    const ids = [buffer(W * H * 4), buffer(W * H * 4)];
    const counters = buffer(new Uint32Array(4));
    const slot = i => ({ buffer: params, offset: i * jfa.paramsStride, size: PARAMS.size });
    const wg = [Math.ceil(W / 8), Math.ceil(H / 8)];

    const seedGroup = bindGroup(pipes.clear, 0, { ids: ids[0], seeds, params: slot(0) });
    const list = [
        { p: pipes.clear, groups: [seedGroup], workgroups: wg },
        { p: pipes.seed, groups: [bindGroup(pipes.seed, 0, { ids: ids[0], seeds, params: slot(0) })],
            workgroups: [Math.ceil(jfa.seeds.length / 64)] },
    ];
    let cur = 0;
    steps.forEach((_, i) => {
        list.push({ p: pipes.flood, groups: [bindGroup(pipes.flood, 0,
            { idsIn: ids[cur], idsOut: ids[1 - cur], seeds, params: slot(i) })], workgroups: wg });
        cur = 1 - cur;
    });
    list.push({ p: pipes.exact, groups: [bindGroup(pipes.exact, 0,
        { ids: ids[cur], seeds, counters, params: slot(0) })], workgroups: wg });
    await run(list, 'voronoi JFA');
    return {
        ids: await read(ids[cur], Uint32Array, W * H * 4),
        counters: await read(counters, Uint32Array, 8),
        passes: steps.length,
    };
}

test('JFA and JFA+1 on the GPU vs float64 brute force (3 metrics × 4 layouts + a ragged grid)', async (t) => {
    const configs = [
        ...[['random', 1], ['grid', 2], ['sunflower', 3], ['clusters', 4]].map(([layout, seed]) => [256, 192, 160, layout, seed]),
        [250, 187, 97, 'random', 5],   // not a multiple of 8: exercises the bounds checks
    ];
    const total = { 0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0] };   // pixels, JFA errors, JFA+1 errors
    let replicaDiff = 0, comparedPixels = 0;
    for (const [W, H, COUNT, layout, seed] of configs) {
        const { device, jfa } = makeJFA(W, H);
        generateSeeded(jfa, COUNT, layout, seed);
        const seeds = uploadedSeeds(device, jfa);
        assert.equal(seeds.length / 2, COUNT);
        for (const metric of [0, 1, 2]) {
            jfa.metric = metric;
            jfa.dragIndex = -1;
            jfa.plusOne = false;
            const plain = await gpuJFA(device, jfa, W, H);
            jfa.plusOne = true;
            const plus = await gpuJFA(device, jfa, W, H);
            assert.equal(plus.passes, plain.passes + 1, 'JFA+1 adds one step-1 pass');

            assert.ok(plain.ids.every(v => v < COUNT), 'every pixel has a seed (no NONE left)');
            assert.ok(plus.ids.every(v => v < COUNT));
            const e = errorsVsBrute(plain.ids, W, H, seeds, metric);
            const ep = errorsVsBrute(plus.ids, W, H, seeds, metric);
            // The in-app checker (cs_exact on the GPU, f32) agrees with the f64 reference up to near-ties
            assert.ok(Math.abs(plain.counters[0] - e.wrong) <= Math.max(2, e.wrong * 0.05),
                `cs_exact ${plain.counters[0]} vs f64 ${e.wrong}`);
            assert.ok(ep.wrong <= e.wrong, 'JFA+1 never makes a pixel worse');

            // CPU replica of the same passes (f32, same tie rule)
            let ids = seedPass(W, H, seeds);
            for (const k of jfaSteps(W, H, false)) ids = floodPass(ids, W, H, seeds, k, metric);
            let diff = 0;
            for (let i = 0; i < ids.length; i++) if (ids[i] !== plain.ids[i]) diff++;
            replicaDiff += diff;
            comparedPixels += ids.length;

            total[metric][0] += W * H; total[metric][1] += e.wrong; total[metric][2] += ep.wrong;
            t.diagnostic(`${W}x${H} ${layout}/${METRIC[metric]}: JFA ${e.wrong} px wrong (max excess ${e.maxExcess.toFixed(3)} px), ` +
                `JFA+1 ${ep.wrong} px; cs_exact [${plain.counters[0]}, ${plain.counters[1]}]; ids != CPU replica: ${diff} px`);
        }
    }
    for (const m of [0, 1, 2]) {
        const [px, wrong, wrongPlus] = total[m];
        t.diagnostic(`${METRIC[m]}: GPU error rate JFA ${(100 * wrong / px).toFixed(4)}%, JFA+1 ${(100 * wrongPlus / px).toFixed(4)}%`);
        assert.ok(wrong / px < MAX_ERROR_RATE, `JFA error rate metric ${m}: ${wrong / px}`);
        assert.ok(wrongPlus / px < MAX_ERROR_RATE_PLUS_ONE, `JFA+1 error rate metric ${m}: ${wrongPlus / px}`);
    }
    t.diagnostic(`GPU seed ids differ from the CPU replica on ${replicaDiff} / ${comparedPixels} px`);
    // f32 equidistant ties may resolve differently if the GPU contracts dot(d, d) into an FMA; an
    // index-math bug would differ on a large share of pixels.
    assert.ok(replicaDiff / comparedPixels < 0.001, `replica mismatch ${replicaDiff} / ${comparedPixels}`);
});

test('seed pass on the GPU: lowest index wins a shared pixel, out-of-range positions clamp', async () => {
    const W = 8, H = 8;
    const { device, jfa } = makeJFA(W, H);
    // Seed 0 far outside (clamps to pixel (0, 7)); seeds 1 and 2 share pixel (3, 4)
    jfa.seeds = [[-5, 1000], [3.2, 4.9], [3.7, 4.1]].map(([x, y]) => ({ x, y, vx: 0, vy: 0, color: 0 }));
    jfa.metric = 0; jfa.plusOne = false; jfa.dragIndex = -1;
    jfa.writeParams([1]);
    const params = buffer(device.lastWrite(jfa.paramsBuffer).bytes);
    jfa.writeSeeds();
    const seedBytes = new Uint8Array(SEED.size * MAX_SEEDS);
    seedBytes.set(device.lastWrite(jfa.seedBuffer).bytes);
    const seeds = buffer(seedBytes);
    const ids = buffer(W * H * 4);
    write(ids, new Uint32Array(W * H).fill(123));
    const res = { ids, seeds, params: { buffer: params, offset: 0, size: PARAMS.size } };
    await run([
        { p: pipes.clear, groups: [bindGroup(pipes.clear, 0, res)], workgroups: [1, 1] },
        { p: pipes.seed, groups: [bindGroup(pipes.seed, 0, res)], workgroups: [1] },
    ], 'seed pass');
    const got = await read(ids, Uint32Array, W * H * 4);
    assert.equal(got[4 * W + 3], 1, 'atomicMin keeps the lower index');
    assert.equal(got[7 * W + 0], 0, 'clamped to the bottom-left pixel');
    assert.equal(got.filter(v => v === NONE).length, W * H - 2, 'cs_clear reset every other pixel');
});
