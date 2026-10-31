// Falling sand (example: js/examples/falling-sand.js).
//
// CPU replica of falling-sand.js computeShader (cs_main: Margolus block update, reactions, fire,
// movement); validates the algorithm and index math, not the WGSL itself — see #51.
// The replica uses exact u32 arithmetic for the PCG hash, so it follows the same random stream
// the shader does. Brush distances are evaluated in float64 (the shader uses f32); no test here
// depends on a brush edge case.
//
// Real code tested directly: buildSandbox (the shipped scene) and FallingSand.writeParams (through
// a mock device, decoded with struct Params parsed from the WGSL). Reaction and Block are
// function-scope structs in the shader (Reaction holds a bool), not host-shareable, so they have no
// host layout to check.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import * as sand from '../js/examples/falling-sand.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const EMPTY = 0, STONE = 1, SAND = 2, WATER = 3, OIL = 4, FIRE = 5, PLANT = 6, ACID = 7, WALL = 15;
const NAMES = ['empty', 'stone', 'sand', 'water', 'oil', 'fire', 'plant', 'acid'];

// ---- CPU replica of the WGSL helpers -------------------------------------------------------
const materialOf = c => c & 15;
const shadeOf = c => (c >>> 4) & 15;
const stateOf = c => (c >>> 8) & 255;
const makeCell = (m, shade, state) => (m | ((shade & 15) << 4) | ((state & 255) << 8)) >>> 0;

function pcg(v) {
    const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
    return ((w >>> 22) ^ w) >>> 0;
}
const add = (a, b) => (a + b) >>> 0;

function density(m) {
    switch (m) {
        case FIRE: return 0; case EMPTY: return 1; case OIL: return 2;
        case WATER: return 3; case ACID: return 4; case SAND: return 5;
        default: return 255;
    }
}
const isStatic = m => m === STONE || m === PLANT || m === WALL;
const isLiquid = m => m === WATER || m === OIL || m === ACID;
const nextRandom = r => (r.v = pcg(r.v));
const chance = (r, outOf256) => (nextRandom(r) & 255) < outOf256;

function reaction(a, b) {
    if (a === FIRE && b === WATER) return [256, EMPTY, WATER, 0, false];
    if (a === FIRE && b === OIL) return [24, FIRE, FIRE, 120, true];
    if (a === FIRE && b === PLANT) return [32, FIRE, FIRE, 60, true];
    if (a === ACID && b === STONE) return [6, EMPTY, EMPTY, 0, false];
    if (a === ACID && b === SAND) return [16, EMPTY, EMPTY, 0, false];
    if (a === ACID && b === PLANT) return [24, EMPTY, EMPTY, 0, false];
    if (a === PLANT && b === WATER) return [1, PLANT, PLANT, 0, false];
    return [0, a, b, 0, false];
}

function transform(c, m, fireLife, r) {
    if (materialOf(c) === m) return c;
    if (m === EMPTY) return 0;
    const rnd = nextRandom(r);
    let state = 0;
    if (m === FIRE) state = fireLife + ((rnd >>> 8) & 63);
    if (isLiquid(m)) state = (rnd >>> 8) & 1;
    return makeCell(m, rnd & 15, state);
}

function fallChance(upper, lower) {
    if (upper === EMPTY) return 48;
    if (lower === EMPTY) return isLiquid(upper) ? 192 : 256;
    return 128;
}
const slideChance = m => (m === SAND ? 128 : 256);
function spreadChance(m) {
    switch (m) { case WATER: return 240; case ACID: return 200; case OIL: return 144; default: return 96; }
}

function swapCells(b, i, j) {
    const t = b.c[i]; b.c[i] = b.c[j]; b.c[j] = t;
    b.moved[i] = true; b.moved[j] = true;
}
function tryFall(b, u, l, r) {
    const mu = materialOf(b.c[u]), ml = materialOf(b.c[l]);
    if (isStatic(mu) || isStatic(ml) || density(mu) <= density(ml)) return;
    if (!chance(r, fallChance(mu, ml))) return;
    swapCells(b, u, l);
}
function trySlide(b, t, s, g, u, r) {
    if (b.moved[t] || b.moved[g]) return;
    const mt = materialOf(b.c[t]), ms = materialOf(b.c[s]), mg = materialOf(b.c[g]), mu = materialOf(b.c[u]);
    if (mt === EMPTY || isStatic(mt) || isStatic(mg) || isStatic(mu)) return;
    if (density(mt) <= density(mg)) return;
    if (!isStatic(ms) && density(ms) < density(mt)) return;
    if (!chance(r, slideChance(mt))) return;
    swapCells(b, t, g);
}
const flowsRight = c => (stateOf(c) & 1) === 1;
const turned = c => (c ^ (1 << 8)) >>> 0;
const canEnter = (m, other) => !isStatic(other) && density(other) < density(m);
function trySpread(b, l, rt, r) {
    if (b.moved[l] || b.moved[rt]) return;
    const cl = b.c[l], cr = b.c[rt], ml = materialOf(cl), mr = materialOf(cr);
    if (isLiquid(ml) && flowsRight(cl)) {
        if (!canEnter(ml, mr)) b.c[l] = turned(cl);
        else if (chance(r, spreadChance(ml))) { swapCells(b, l, rt); return; }
    }
    if (isLiquid(mr) && !flowsRight(cr)) {
        if (!canEnter(mr, ml)) b.c[rt] = turned(cr);
        else if (chance(r, spreadChance(mr))) { swapCells(b, l, rt); return; }
    }
    if ((ml === FIRE && mr === EMPTY) || (mr === FIRE && ml === EMPTY)) {
        if (chance(r, spreadChance(FIRE))) swapCells(b, l, rt);
    }
}
function segmentDistance(px, py, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay, len2 = abx * abx + aby * aby;
    let t = 0;
    if (len2 > 0) t = Math.min(Math.max(((px - ax) * abx + (py - ay) * aby) / len2, 0), 1);
    return Math.hypot(px - (ax + abx * t), py - (ay + aby * t));
}

// One dispatch of cs_main over the whole grid (in place, like the shader). p mirrors struct Params.
function sandStep(cells, p) {
    const [W, H] = p.size;
    const blocksX = Math.floor(W / 2) + 1, blocksY = Math.floor(H / 2) + 1;
    const shift = p.step & 1;
    for (let gy = 0; gy < blocksY; gy++) {
        for (let gx = 0; gx < blocksX; gx++) {
            const x0 = gx * 2 - shift, y0 = gy * 2 - shift;
            const r = { v: pcg(add(gx, pcg(add(gy, pcg(p.step))))) };
            const b = { c: [0, 0, 0, 0], moved: [false, false, false, false] };
            const inside = [false, false, false, false];
            for (let k = 0; k < 4; k++) {
                const x = x0 + (k & 1), y = y0 + (k >> 1);
                inside[k] = x >= 0 && x < W && y >= 0 && y < H;
                b.c[k] = inside[k] ? cells[y * W + x] : WALL;
            }
            if (p.brushOn) {
                const paint = { v: add(r.v, p.paintSeed) };
                paint.v = pcg(paint.v);
                const m = p.brushMaterial;
                const spray = (m === STONE || m === PLANT || m === EMPTY) ? 256 : 128;
                for (let k = 0; k < 4; k++) {
                    const px = x0 + (k & 1) + 0.5, py = y0 + (k >> 1) + 0.5;
                    if (inside[k] && segmentDistance(px, py, ...p.brushPrev, ...p.brushPos) < p.brushRadius) {
                        if (chance(paint, spray)) b.c[k] = transform(0, m, 48, paint);
                    }
                }
            }
            if (p.simulate) {
                const reacted = [false, false, false, false];
                const hasAir = b.c.some(c => materialOf(c) === EMPTY);
                for (let i = 0; i < 4; i++) {
                    for (let j = 0; j < 4; j++) {
                        if (i === j || reacted[i] || reacted[j]) continue;
                        const [ch, newA, newB, fireLife, needsAir] = reaction(materialOf(b.c[i]), materialOf(b.c[j]));
                        if (ch === 0 || (needsAir && !hasAir) || !chance(r, ch)) continue;
                        b.c[i] = transform(b.c[i], newA, fireLife, r);
                        b.c[j] = transform(b.c[j], newB, fireLife, r);
                        reacted[i] = reacted[j] = true;
                    }
                }
                for (let k = 0; k < 4; k++) {
                    if (materialOf(b.c[k]) === FIRE) {
                        const life = stateOf(b.c[k]);
                        b.c[k] = life <= 1 ? 0 : makeCell(FIRE, shadeOf(b.c[k]), life - 1);
                    }
                }
                if ((nextRandom(r) & 1) === 0) {
                    tryFall(b, 0, 2, r); tryFall(b, 1, 3, r);
                    trySlide(b, 0, 2, 3, 1, r); trySlide(b, 1, 3, 2, 0, r);
                } else {
                    tryFall(b, 1, 3, r); tryFall(b, 0, 2, r);
                    trySlide(b, 1, 3, 2, 0, r); trySlide(b, 0, 2, 3, 1, r);
                }
                trySpread(b, 2, 3, r);
                trySpread(b, 0, 1, r);
            }
            for (let k = 0; k < 4; k++) {
                if (inside[k]) cells[(y0 + (k >> 1)) * W + x0 + (k & 1)] = b.c[k];
            }
        }
    }
}

function run(cells, w, h, steps, first = 0) {
    for (let s = 0; s < steps; s++) {
        sandStep(cells, { size: [w, h], step: first + s, simulate: 1, brushOn: 0 });
    }
}

function counts(cells) {
    const n = new Array(16).fill(0);
    for (const c of cells) n[materialOf(c)]++;
    return n;
}
// Multiset of (material, shade): movement only swaps cells, so this is invariant without reactions
function signature(cells) {
    const n = new Map();
    for (const c of cells) { const k = c & 0xff; n.set(k, (n.get(k) || 0) + 1); }
    return [...n.entries()].sort((a, b) => a[0] - b[0]);
}

// Small deterministic LCG for building test grids
function lcg(seed) {
    let s = seed >>> 0;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function randomCell(m, rnd) {
    if (m === EMPTY) return 0;
    const shade = Math.floor(rnd() * 16);
    const state = m === FIRE ? 40 + Math.floor(rnd() * 40) : isLiquid(m) ? Math.floor(rnd() * 2) : 0;
    return makeCell(m, shade, state);
}

// ---- Tests ---------------------------------------------------------------------------------

test('Margolus blocks tile the grid: every cell owned by exactly one block per step parity', (t) => {
    const sizes = [[320, 240], [7, 5], [8, 6], [1, 1], [3, 8]];
    for (const [W, H] of sizes) {
        for (const shift of [0, 1]) {
            const owner = new Int32Array(W * H);
            const blocksX = Math.floor(W / 2) + 1, blocksY = Math.floor(H / 2) + 1;
            for (let gy = 0; gy < blocksY; gy++) for (let gx = 0; gx < blocksX; gx++) {
                for (let k = 0; k < 4; k++) {
                    const x = gx * 2 - shift + (k & 1), y = gy * 2 - shift + (k >> 1);
                    if (x >= 0 && x < W && y >= 0 && y < H) owner[y * W + x]++;
                }
            }
            assert.ok(owner.every(o => o === 1), `${W}x${H} shift ${shift}: a cell is owned ${Math.max(...owner)}x or 0x`);
        }
    }
    // The host dispatches ceil((w/2 + 1) / 8) workgroups of 8: enough threads for every block
    for (const [W, H] of sizes) {
        assert.ok(Math.ceil((W / 2 + 1) / 8) * 8 >= Math.floor(W / 2) + 1);
        assert.ok(Math.ceil((H / 2 + 1) / 8) * 8 >= Math.floor(H / 2) + 1);
    }
    t.diagnostic(`exclusive ownership verified for ${sizes.length} grid sizes x 2 offsets`);
});

test('non-reacting materials are conserved exactly; stone never moves', (t) => {
    const W = 64, H = 48, rnd = lcg(11);
    const cells = new Uint32Array(W * H);
    for (let i = 0; i < W * H; i++) {
        const u = rnd();
        cells[i] = randomCell(u < 0.25 ? SAND : u < 0.4 ? WATER : u < 0.5 ? OIL : u < 0.58 ? STONE : EMPTY, rnd);
    }
    const before = signature(cells);
    const stone = cells.map(c => (materialOf(c) === STONE ? c : 0));
    const t0 = performance.now();
    run(cells, W, H, 400);
    const ms = performance.now() - t0;
    assert.deepEqual(signature(cells), before, '(material, shade) multiset changed');
    for (let i = 0; i < W * H; i++) if (stone[i]) assert.equal(cells[i], stone[i], `stone moved at ${i}`);
    const n = counts(cells);
    t.diagnostic(`64x48, 400 steps (${ms.toFixed(0)} ms): sand ${n[SAND]}, water ${n[WATER]}, oil ${n[OIL]}, stone ${n[STONE]} — all unchanged`);
});

test('gravity: a column of sand settles and the liquids layer by density', (t) => {
    const W = 32, H = 40;
    const cells = new Uint32Array(W * H);
    const rnd = lcg(5);
    for (let y = 2; y < 18; y++) for (let x = 12; x < 20; x++) cells[y * W + x] = randomCell(SAND, rnd);
    const meanY = () => {
        let s = 0, n = 0;
        cells.forEach((c, i) => { if (materialOf(c) === SAND) { s += Math.floor(i / W); n++; } });
        return s / n;
    };
    const y0 = meanY();
    run(cells, W, H, 600);
    const y1 = meanY();
    let floating = 0;
    for (let y = 0; y < H - 1; y++) for (let x = 0; x < W; x++) {
        if (materialOf(cells[y * W + x]) === SAND && materialOf(cells[(y + 1) * W + x]) === EMPTY) floating++;
    }
    t.diagnostic(`sand mean depth ${y0.toFixed(2)} -> ${y1.toFixed(2)} rows; grains above air after 600 steps: ${floating}`);
    assert.equal(floating, 0, 'settled sand must not hang over air');
    assert.ok(y1 > y0 + 15, 'sand should fall most of the way to the floor');

    // Oil over water, initially inverted: after mixing, oil's mean depth is above water's
    const L = new Uint32Array(W * H);
    for (let y = 20; y < 30; y++) for (let x = 0; x < W; x++) L[y * W + x] = randomCell(WATER, rnd);
    for (let y = 30; y < H; y++) for (let x = 0; x < W; x++) L[y * W + x] = randomCell(OIL, rnd);
    run(L, W, H, 800);
    const depth = m => {
        let s = 0, n = 0;
        L.forEach((c, i) => { if (materialOf(c) === m) { s += Math.floor(i / W); n++; } });
        return s / n;
    };
    t.diagnostic(`after 800 steps: mean oil row ${depth(OIL).toFixed(2)}, mean water row ${depth(WATER).toFixed(2)} (larger = lower)`);
    assert.ok(depth(OIL) < depth(WATER) - 3, 'oil should float on water');
});

test('reactions: fire burns out, water extinguishes fire, acid is used up 1:1', (t) => {
    const W = 24, H = 24, rnd = lcg(9);
    // Fire alone (life <= 79 here) is gone within its life
    const f = new Uint32Array(W * H);
    for (let i = 0; i < 40; i++) f[Math.floor(rnd() * W * H)] = randomCell(FIRE, rnd);
    run(f, W, H, 80);
    assert.equal(counts(f)[FIRE], 0, 'fire outlived its life counter');

    // Fire next to water is put out; the water stays
    const fw = new Uint32Array(W * H);
    for (let y = 18; y < H; y++) for (let x = 0; x < W; x++) fw[y * W + x] = randomCell(WATER, rnd);
    for (let x = 0; x < W; x++) fw[17 * W + x] = randomCell(FIRE, rnd);
    const water0 = counts(fw)[WATER];
    run(fw, W, H, 6);
    const n = counts(fw);
    assert.equal(n[WATER], water0, 'extinguishing fire must not consume water');

    // Acid with stone and sand: each reaction removes one acid and one victim
    const a = new Uint32Array(W * H);
    for (let i = 0; i < W * H; i++) {
        const u = rnd();
        a[i] = randomCell(u < 0.2 ? ACID : u < 0.35 ? STONE : u < 0.5 ? SAND : EMPTY, rnd);
    }
    const c0 = counts(a);
    run(a, W, H, 300);
    const c1 = counts(a);
    const acidUsed = c0[ACID] - c1[ACID];
    const eaten = (c0[STONE] - c1[STONE]) + (c0[SAND] - c1[SAND]);
    t.diagnostic(`fire-only grid burnt out in <= 80 steps; acid used ${acidUsed}, stone+sand eaten ${eaten}`);
    assert.ok(acidUsed > 0);
    assert.equal(acidUsed, eaten);
});

test('brush-only pass paints a solid material exactly inside the stroke capsule', () => {
    const W = 40, H = 30, cells = new Uint32Array(W * H);
    const p = { size: [W, H], step: 3, simulate: 0, brushOn: 1, brushMaterial: STONE, paintSeed: 1,
        brushPrev: [10.2, 12.7], brushPos: [25.6, 14.1], brushRadius: 4.5 };
    sandStep(cells, p);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const inStroke = segmentDistance(x + 0.5, y + 0.5, ...p.brushPrev, ...p.brushPos) < p.brushRadius;
        assert.equal(materialOf(cells[y * W + x]), inStroke ? STONE : EMPTY, `cell ${x},${y}`);
    }
});

test('shipped buildSandbox: valid encoding, deterministic, conserved under the replica', (t) => {
    const W = 320, H = 240;
    const grid = sand.buildSandbox(W, H);
    assert.deepEqual(sand.buildSandbox(W, H), grid, 'sandbox is not deterministic');
    for (const c of grid) {
        const m = materialOf(c);
        assert.ok(m <= ACID && m !== FIRE, `unexpected material ${m}`);
        if (m === EMPTY) assert.equal(c, 0, 'empty cells are stored as 0');
        if (!isLiquid(m)) assert.equal(stateOf(c), 0);
        else assert.ok(stateOf(c) <= 1);
    }
    const c0 = counts(grid);
    assert.ok(c0[SAND] > 0 && c0[WATER] > 0 && c0[OIL] > 0 && c0[STONE] > 0 && c0[PLANT] > 0);
    const cells = grid.slice();
    const t0 = performance.now();
    run(cells, W, H, 120);
    const ms = performance.now() - t0;
    const c1 = counts(cells);
    // No fire or acid in the scene: only PLANT + WATER -> PLANT + PLANT can react
    assert.equal(c1[SAND], c0[SAND]);
    assert.equal(c1[OIL], c0[OIL]);
    assert.equal(c1[STONE], c0[STONE]);
    assert.equal(c1[WATER] + c1[PLANT], c0[WATER] + c0[PLANT]);
    t.diagnostic(`sandbox 320x240, 120 replica steps in ${ms.toFixed(0)} ms: ` +
        NAMES.map((nm, m) => `${nm} ${c0[m]}->${c1[m]}`).join(', '));
});

function fakeCanvas() {
    return {
        width: 640, height: 480,
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
        addEventListener() {}, removeEventListener() {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 1, height: 1 }),
    };
}

test('FallingSand.writeParams matches struct Params in every dynamic-offset slot', (t) => {
    globalThis.window ??= { addEventListener() {}, removeEventListener() {} };
    const device = mockDevice();
    const sim = new sand.FallingSand(device, fakeCanvas(), 320, 240);
    sim.brushPos = [101.25, 37.5];
    sim.brushPrev = [99.75, 40.125];
    sim.brushRadius = 7;
    sim.paintMaterial = ACID;
    sim.steps = 4000000001;
    sim.paintSeed = 123456;

    const layout = structLayout(moduleCode(sand, 'compute'), 'Params');
    assert.deepEqual(structLayout(moduleCode(sand, 'render'), 'Params').offsets, layout.offsets);
    assert.equal(layout.size, 48);

    const STRIDE = 256;
    sim.writeParams(3, true, true);
    const wr = device.lastWrite(sim.paramsBuffer);
    assert.equal(wr.bytes.byteLength, 3 * STRIDE);
    for (let k = 0; k < 3; k++) {
        const p = decodeStruct(wr.bytes, layout, k * STRIDE);
        assert.deepEqual(p.size, [320, 240]);
        assert.deepEqual(p.brushPos, [101.25, 37.5]);
        assert.deepEqual(p.brushPrev, [99.75, 40.125]);
        assert.equal(p.brushRadius, 7);
        assert.equal(p.brushMaterial, ACID);
        assert.equal(p.brushOn, k === 0 ? 1 : 0, 'only the first step applies the brush');
        assert.equal(p.simulate, 1);
        assert.equal(p.step, 4000000001 + k);
        assert.equal(p.paintSeed, 123456);
    }
    // Brush-only pass: step counter does not advance
    sim.writeParams(1, false, true);
    const q = decodeStruct(device.lastWrite(sim.paramsBuffer).bytes, layout);
    assert.equal(q.simulate, 0);
    assert.equal(q.step, 4000000001);
    t.diagnostic(`Params ${layout.size} B in 256 B slots; offsets ${JSON.stringify(layout.offsets)}`);
});
