// Voronoi diagrams with the Jump Flooding Algorithm (js/examples/voronoi-jfa.js).
//
// CPU replica of the seed (cs_clear, cs_seed), flood (cs_flood) and exact-check (cs_exact) shaders;
// validates the algorithm and index math, not the WGSL itself — see #51. The replica keeps the
// shaders' f32 distances (Math.fround), pixel-center sampling, neighbor order, tie rule (lower seed
// index), atomicMin seeding and ping-pong passes over the shipped jfaSteps() schedule. It is
// compared with an independent float64 brute-force nearest-seed search for all three metrics,
// for plain JFA and JFA+1, over the example's own seed layouts (VoronoiJFA.generate, seeded RNG).
//
// Real code tested: jfaSteps, packColor, VoronoiJFA.generate, and the Params / Seed writers through
// a mock device, decoded with the WGSL struct layouts.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jfaSteps, packColor, VoronoiJFA, MAX_SEEDS, wgslModules } from '../js/examples/voronoi-jfa.js';
import { decodeStruct } from './helpers/wgsl-layout.mjs';
import {
    NONE, PARAMS, SEED, makeJFA, generateSeeded, uploadedSeeds, seedPass, floodPass, exactCheck, errorsVsBrute,
    MAX_ERROR_RATE, MAX_ERROR_RATE_PLUS_ONE,
} from './helpers/voronoi-refs.mjs';

const f = Math.fround;

// Replica and reference helpers (seeded generate(), uploaded seed decode, f32 metric, seed/flood/
// exact-check replicas, float64 brute force) live in test/helpers/voronoi-refs.mjs, shared with
// test/gpu/voronoi-jfa.gpu.test.mjs.

test('jfaSteps: N/2 ... 1 for N = next power of two ≥ max(w, h) (real code)', () => {
    assert.deepEqual(jfaSteps(640, 480, false), [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]);
    assert.deepEqual(jfaSteps(640, 480, true), [512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1]);
    assert.deepEqual(jfaSteps(512, 100, false), [256, 128, 64, 32, 16, 8, 4, 2, 1]);
    assert.deepEqual(jfaSteps(513, 100, false)[0], 512);
    assert.deepEqual(jfaSteps(1, 1, false), [1]);
    // Every pixel can be reached: sum of steps ≥ max(w, h) - 1
    for (const [w, h] of [[256, 192], [1000, 3], [7, 900]]) {
        assert.ok(jfaSteps(w, h, false).reduce((a, b) => a + b, 0) >= Math.max(w, h) - 1);
    }
});

test('packColor: RGBA8 with red in the low byte (real code)', () => {
    assert.equal(packColor(0, 1, 0.5), 0xff0000ff);         // red
    assert.equal(packColor(1 / 3, 1, 0.5), 0xff00ff00);     // green
    assert.equal(packColor(2 / 3, 1, 0.5), 0xffff0000);     // blue
    assert.equal(packColor(0.3, 0, 1), 0xffffffff);         // white
});

test('JFA replica vs brute-force nearest seed (3 metrics × 4 layouts, JFA and JFA+1)', (t) => {
    const W = 256, H = 192, COUNT = 160;
    const { device, jfa } = makeJFA(W, H);
    const rows = [];
    let totals = { 0: [0, 0], 1: [0, 0], 2: [0, 0] };
    const pixels = W * H;
    let layouts = 0;
    for (const [layout, seed] of [['random', 1], ['grid', 2], ['sunflower', 3], ['clusters', 4]]) {
        generateSeeded(jfa, COUNT, layout, seed);
        const seeds = uploadedSeeds(device, jfa);
        assert.equal(seeds.length / 2, COUNT);
        layouts++;
        for (const metric of [0, 1, 2]) {
            let ids = seedPass(W, H, seeds);
            const steps = jfaSteps(W, H, false);
            for (const k of steps) ids = floodPass(ids, W, H, seeds, k, metric);
            const plain = errorsVsBrute(ids, W, H, seeds, metric);
            const exact = exactCheck(ids, W, H, seeds, metric);
            const plus = floodPass(ids, W, H, seeds, 1, metric);   // JFA+1: one extra step-1 pass
            const plusErr = errorsVsBrute(plus, W, H, seeds, metric);
            assert.ok(plus.every(v => v < COUNT), 'every pixel has a seed');
            // The in-app checker (cs_exact, f32) and the f64 reference agree up to f32 near-ties
            assert.ok(Math.abs(exact[0] - plain.wrong) <= Math.max(2, plain.wrong * 0.05),
                `cs_exact ${exact[0]} vs f64 ${plain.wrong}`);
            totals[metric][0] += plain.wrong;
            totals[metric][1] += plusErr.wrong;
            rows.push(`${layout}/${['euclid', 'manhattan', 'chebyshev'][metric]}: JFA ${plain.wrong} px (${(100 * plain.wrong / pixels).toFixed(3)}%, ` +
                `max excess ${plain.maxExcess.toFixed(3)} px), JFA+1 ${plusErr.wrong} px; cs_exact counters [${exact}]`);
            // JFA+1 never makes a pixel worse (it only adds candidates)
            assert.ok(plusErr.wrong <= plain.wrong);
        }
    }
    for (const r of rows) t.diagnostic(r);
    const rate = m => totals[m][0] / (layouts * pixels);
    const ratePlus = m => totals[m][1] / (layouts * pixels);
    t.diagnostic(`error rate JFA / JFA+1: euclid ${(100 * rate(0)).toFixed(4)}% / ${(100 * ratePlus(0)).toFixed(4)}%, ` +
        `manhattan ${(100 * rate(1)).toFixed(4)}% / ${(100 * ratePlus(1)).toFixed(4)}%, chebyshev ${(100 * rate(2)).toFixed(4)}% / ${(100 * ratePlus(2)).toFixed(4)}%`);
    // Tolerances (Rong & Tan report well under 1% errors for JFA, fewer with JFA+1). Measured
    // rates (~0.01% JFA, ~0.002% JFA+1 here) are printed above; bounds of 0.1% / 0.05% leave ~5-10x
    // headroom while still catching index-math bugs, which
    // produce errors on a large fraction of pixels.
    for (const m of [0, 1, 2]) {
        assert.ok(rate(m) < MAX_ERROR_RATE, `JFA error rate metric ${m}: ${rate(m)}`);
        assert.ok(ratePlus(m) < MAX_ERROR_RATE_PLUS_ONE, `JFA+1 error rate metric ${m}: ${ratePlus(m)}`);
    }
});

test('seed pass: lowest index wins a shared pixel; out-of-range positions clamp', () => {
    const seeds = Float32Array.of(3.2, 4.9, 3.7, 4.1, -5, 1000);
    const ids = seedPass(8, 8, seeds);
    assert.equal(ids[4 * 8 + 3], 0);
    assert.equal(ids[7 * 8 + 0], 2);
});

test('uniform layouts: Params and Seed writers match WGSL', (t) => {
    assert.equal(PARAMS.size, 32);
    assert.equal(SEED.size, 16);
    const { device, jfa } = makeJFA(320, 200);
    generateSeeded(jfa, 37, 'random', 9);
    jfa.metric = 2; jfa.viewMode = 1; jfa.dragIndex = -1;
    const steps = jfaSteps(320, 200, true);
    jfa.writeParams(steps);
    const w = device.lastWrite(jfa.paramsBuffer);
    assert.equal(w.bytes.byteLength, steps.length * jfa.paramsStride);
    steps.forEach((k, s) => {
        const p = decodeStruct(w.bytes, PARAMS, s * jfa.paramsStride);
        assert.deepEqual(p, {
            size: [320, 200], stepSize: k, metric: 2, seedCount: 37, viewMode: 1, selected: NONE,
            distScale: f(Math.sqrt(320 * 200 / 37)),
        }, `slot ${s}`);
    });
    jfa.dragIndex = 5;
    jfa.writeParams([]);   // no steps: one slot with stepSize 1
    const p = decodeStruct(device.lastWrite(jfa.paramsBuffer).bytes, PARAMS);
    assert.equal(p.selected, 5);
    assert.equal(p.stepSize, 1);

    jfa.writeSeeds();
    const sb = device.lastWrite(jfa.seedBuffer).bytes;
    assert.equal(sb.byteLength, 37 * SEED.size);
    jfa.seeds.forEach((s, i) => {
        const d = decodeStruct(sb, SEED, i * SEED.size);
        assert.deepEqual(d, { pos: [f(s.x), f(s.y)], color: s.color, pad: 0 });
    });
    assert.ok(37 <= MAX_SEEDS);
    t.diagnostic(`${steps.length} Params slots at stride ${jfa.paramsStride}, 37 Seed records decoded OK`);
});

test('distinctPixels gives every seed its own pixel and leaves free seeds untouched', async () => {
    const { distinctPixels } = await import('../js/examples/voronoi-jfa.js');
    const pts = [[5.2, 5.7], [5.9, 5.1], [5.5, 5.5], [100.3, 7.8], [-3, 2], [2000, 2000]];
    const out = distinctPixels(pts, 64, 48);
    const keys = out.map(([x, y]) => `${Math.floor(x)},${Math.floor(y)}`);
    assert.equal(new Set(keys).size, pts.length);
    assert.deepEqual(out[0], [5.2, 5.7]);                  // first claimant keeps its exact position
    for (const [x, y] of out) assert.ok(x >= 0 && x < 64 && y >= 0 && y < 48);
    for (const [x, y] of out.slice(1, 3)) assert.ok(Math.hypot(x - 5.5, y - 5.5) < 2, 'collisions move to a neighbor');
});
