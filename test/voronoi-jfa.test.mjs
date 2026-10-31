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
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const f = Math.fround;
const NONE = 0xffffffff;
const mod = { wgslModules };
const PARAMS = structLayout(moduleCode(mod, 'flood'), 'Params');
const SEED = structLayout(moduleCode(mod, 'flood'), 'Seed');

globalThis.window ??= { addEventListener() {}, removeEventListener() {} };
globalThis.requestAnimationFrame ??= () => 0;

function rng(seed) {   // mulberry32
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function makeJFA(width, height) {
    const device = mockDevice();
    const canvas = {
        width, height, addEventListener() {}, removeEventListener() {},
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) }),
    };
    return { device, jfa: new VoronoiJFA(device, canvas) };
}

// Seeds placed by the shipped generate() with Math.random replaced by a seeded generator
function generateSeeded(jfa, count, layout, seed) {
    const saved = Math.random;
    Math.random = rng(seed);
    try { jfa.generate(count, layout); } finally { Math.random = saved; }
}

// Seed positions as the GPU sees them: decode the bytes writeSeeds() uploads
function uploadedSeeds(device, jfa) {
    jfa.writeSeeds();
    const bytes = device.lastWrite(jfa.seedBuffer).bytes;
    const n = bytes.byteLength / SEED.size;
    const out = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
        const s = decodeStruct(bytes, SEED, i * SEED.size);
        out[2 * i] = s.pos[0]; out[2 * i + 1] = s.pos[1];
    }
    return out;
}

// f32 metricDistance (Euclidean returns the squared distance)
function metricF32(cx, cy, sx, sy, metric) {
    const dx = Math.abs(f(cx - sx)), dy = Math.abs(f(cy - sy));
    if (metric === 1) return f(dx + dy);
    if (metric === 2) return Math.max(dx, dy);
    return f(f(dx * dx) + f(dy * dy));
}
function metric64(cx, cy, sx, sy, metric) {
    const dx = Math.abs(cx - sx), dy = Math.abs(cy - sy);
    if (metric === 1) return dx + dy;
    if (metric === 2) return Math.max(dx, dy);
    return dx * dx + dy * dy;
}

function seedPass(w, h, seeds) {
    const ids = new Uint32Array(w * h).fill(NONE);
    const count = seeds.length / 2;
    for (let i = 0; i < count; i++) {
        const px = Math.min(Math.max(Math.floor(seeds[2 * i]), 0), w - 1);
        const py = Math.min(Math.max(Math.floor(seeds[2 * i + 1]), 0), h - 1);
        const at = py * w + px;
        ids[at] = Math.min(ids[at], i);   // atomicMin
    }
    return ids;
}

function floodPass(idsIn, w, h, seeds, k, metric) {
    const count = seeds.length / 2;
    const out = new Uint32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const cx = f(x + 0.5), cy = f(y + 0.5);
        let bestId = NONE, bestDist = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const qx = x + dx * k, qy = y + dy * k;
            if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
            const id = idsIn[qy * w + qx];
            if (id >= count) continue;
            const d = metricF32(cx, cy, seeds[2 * id], seeds[2 * id + 1], metric);
            if (bestId === NONE || d < bestDist || (d === bestDist && id < bestId)) { bestId = id; bestDist = d; }
        }
        out[y * w + x] = bestId;
    }
    return out;
}

// cs_exact replica: counters[0] strictly-farther pixels, counters[1] differing index (f32)
function exactCheck(ids, w, h, seeds, metric) {
    const count = seeds.length / 2;
    const counters = [0, 0];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const cx = f(x + 0.5), cy = f(y + 0.5);
        let bestId = NONE, bestDist = 0;
        for (let i = 0; i < count; i++) {
            const d = metricF32(cx, cy, seeds[2 * i], seeds[2 * i + 1], metric);
            if (bestId === NONE || d < bestDist) { bestId = i; bestDist = d; }
        }
        const j = ids[y * w + x];
        if (j !== bestId) counters[1]++;
        if (j >= count || metricF32(cx, cy, seeds[2 * j], seeds[2 * j + 1], metric) > bestDist) counters[0]++;
    }
    return counters;
}

// Independent reference: float64 brute force. A pixel is an error if its JFA seed is farther than
// the true nearest by more than f32 rounding (relative 1e-6).
function errorsVsBrute(ids, w, h, seeds, metric) {
    const count = seeds.length / 2;
    let wrong = 0, maxExcess = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const cx = x + 0.5, cy = y + 0.5;
        let best = Infinity;
        for (let i = 0; i < count; i++) best = Math.min(best, metric64(cx, cy, seeds[2 * i], seeds[2 * i + 1], metric));
        const j = ids[y * w + x];
        const d = j < count ? metric64(cx, cy, seeds[2 * j], seeds[2 * j + 1], metric) : Infinity;
        if (d > best * (1 + 1e-6) + 1e-9) {
            wrong++;
            const lin = metric === 0 ? Math.sqrt(d) - Math.sqrt(best) : d - best;
            maxExcess = Math.max(maxExcess, lin);
        }
    }
    return { wrong, maxExcess };
}

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
        assert.ok(rate(m) < 0.001, `JFA error rate metric ${m}: ${rate(m)}`);
        assert.ok(ratePlus(m) < 0.0005, `JFA+1 error rate metric ${m}: ${ratePlus(m)}`);
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
