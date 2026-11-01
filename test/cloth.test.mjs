// Cloth (js/examples/cloth.js).
//
// Real shipped code: buildCloth for every preset (constraint counts, rest lengths, the 12-coloring,
// parents, Links, tethers), iterationStiffness, OrbitCamera, pickParticle, and the uniform writers
// ClothSimulation.writeSimParams / writeCamera / the ColorRange slots written by buildGrid, decoded
// through the WGSL SimParams / Camera / ColorRange / Constraint / Links structs.
//
// CPU replica of the compute shader (cs_integrate, cs_tether, cs_solve per color, cs_collide,
// cs_normals, in test/helpers/cloth-refs.mjs); validates the algorithm and index math, not the WGSL
// itself — see #51. The replica
// is driven by the SimParams bytes the shipped writer produces, and checked for: order
// independence within a color (the race-freedom the coloring promises), hanging-cloth stability
// (finite, bounded stretch, no energy gain), collision constraints, and tearing thresholds.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    buildCloth, PRESETS, NUM_COLORS, NONE, KIND_STRUCTURAL, KIND_SHEAR, KIND_BEND,
    iterationStiffness, OrbitCamera, FOV, pickParticle, SUBSTEP_DT,
    wgslModules,
} from '../js/examples/cloth.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { simWithMock, makeState, frame, simParamsFor, maxStretch, energy } from './helpers/cloth-refs.mjs';

const computeCode = wgslModules().find(m => m.name === 'compute').code;
const renderCode = wgslModules().find(m => m.name === 'render').code;
const SIM = structLayout(computeCode, 'SimParams');
const RANGE = structLayout(computeCode, 'ColorRange');
const CONSTRAINT = structLayout(computeCode, 'Constraint');
const LINKS = structLayout(computeCode, 'Links');
const CAMERA = structLayout(renderCode, 'Camera');

function mulberry(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function decodeConstraints(cloth) {
    return Array.from({ length: cloth.numConstraints }, (_, k) => decodeStruct(cloth.constraintData, CONSTRAINT, k * CONSTRAINT.size));
}
function decodeLinks(cloth) {
    return Array.from({ length: cloth.count }, (_, k) => decodeStruct(cloth.linkData, LINKS, k * LINKS.size));
}

// ================================================================ buildCloth (shipped)

test('struct sizes: Constraint 32 B, Links 48 B, SimParams 128 B, Camera 144 B, ColorRange 16 B', () => {
    assert.equal(CONSTRAINT.size, 32);
    assert.equal(LINKS.size, 48);
    assert.equal(SIM.size, 128);
    assert.equal(CAMERA.size, 144);
    assert.equal(RANGE.size, 16);
});

test('buildCloth: counts, rest lengths, coloring, parents, links and tethers for every preset', (t) => {
    for (const n of [8, 13]) {
        for (const [key, preset] of Object.entries(PRESETS)) {
            const cloth = buildCloth(n, key);
            const cs = decodeConstraints(cloth);
            const links = decodeLinks(cloth);
            const id = (i, j) => j * n + i;
            const restFn = preset.rest || preset.place;
            const restOf = (k) => restFn((k % n) / (n - 1), Math.floor(k / n) / (n - 1));

            // Counts per kind for an n × n grid
            const expect = [2 * n * (n - 1), 2 * (n - 1) ** 2, 2 * n * (n - 2)];
            assert.deepEqual(cloth.kindCounts, expect, `${key} n=${n} kind counts`);
            assert.equal(cloth.numConstraints, expect[0] + expect[1] + expect[2]);
            assert.equal(cloth.numStructural, expect[0]);
            assert.equal(cloth.count, n * n);
            assert.equal(cloth.constraintData.byteLength, cloth.numConstraints * CONSTRAINT.size);
            assert.equal(cloth.linkData.byteLength, cloth.count * LINKS.size);

            // Ranges tile [0, numConstraints) in order; structural links come first
            assert.equal(cloth.ranges.length, NUM_COLORS);
            let next = 0;
            for (const r of cloth.ranges) { assert.equal(r.start, next); next += r.count; }
            assert.equal(next, cloth.numConstraints);
            cs.forEach((c, k) => assert.equal(c.kind === KIND_STRUCTURAL, k < cloth.numStructural));

            // Coloring: no two constraints of one color share a particle
            const colorOf = new Int32Array(cloth.numConstraints);
            cloth.ranges.forEach((r, color) => {
                const used = new Set();
                for (let k = r.start; k < r.start + r.count; k++) {
                    colorOf[k] = color;
                    const { a, b } = cs[k];
                    assert.ok(!used.has(a) && !used.has(b), `${key} n=${n}: color ${color} shares a particle`);
                    used.add(a); used.add(b);
                }
            });

            // Rest lengths from the rest layout (f32), adjacency by kind, parents
            let maxRestErr = 0;
            const pairKey = (a, b) => `${Math.min(a, b)}-${Math.max(a, b)}`;
            const byPair = new Map(cs.map((c, k) => [pairKey(c.a, c.b), k]));
            cs.forEach((c, k) => {
                maxRestErr = Math.max(maxRestErr, Math.abs(c.rest - dist3(restOf(c.a), restOf(c.b))) / dist3(restOf(c.a), restOf(c.b)));
                const [ia, ja, ib, jb] = [c.a % n, Math.floor(c.a / n), c.b % n, Math.floor(c.b / n)];
                const di = Math.abs(ia - ib), dj = Math.abs(ja - jb);
                if (c.kind === KIND_STRUCTURAL) assert.equal(di + dj, 1);
                if (c.kind === KIND_SHEAR) assert.ok(di === 1 && dj === 1);
                if (c.kind === KIND_BEND) assert.ok((di === 2 && dj === 0) || (di === 0 && dj === 2));
                for (const parent of [c.parentA, c.parentB]) {
                    if (parent === NONE) continue;
                    assert.ok(parent < cloth.numConstraints);
                    assert.notEqual(colorOf[parent], colorOf[k], 'parents live in another color');
                }
                if (c.kind === KIND_BEND) {
                    // Spans two structural links: (a, mid) and (mid, b)
                    const mid = (c.a + c.b) / 2;
                    assert.equal(c.parentA, byPair.get(pairKey(c.a, mid)));
                    assert.equal(c.parentB, byPair.get(pairKey(mid, c.b)));
                } else if (c.kind === KIND_SHEAR && ia > ib) {
                    // Anti-diagonal (i+1, j)-(i, j+1) breaks with the diagonal (i, j)-(i+1, j+1)
                    const i = ib, j = ja;
                    assert.equal(c.parentA, byPair.get(pairKey(id(i, j), id(i + 1, j + 1))));
                    assert.equal(c.parentB, NONE);
                } else {
                    assert.equal(c.parentA, NONE);
                    assert.equal(c.parentB, NONE);
                }
            });
            assert.ok(maxRestErr < 1e-6, `${key}: rest length error ${maxRestErr}`);

            // Links: the structural / diagonal constraint indices around each particle
            const pins = [];
            for (let k = 0; k < cloth.count; k++) {
                const i = k % n, j = Math.floor(k / n);
                assert.equal(cloth.positions[k * 4 + 3], preset.pinned(i, j, n) ? 0 : 1);
                if (preset.pinned(i, j, n)) pins.push(k);
                const L = links[k];
                const expectLink = (cond, other) => (cond ? byPair.get(pairKey(k, other)) : NONE);
                assert.equal(L.xNeg, expectLink(i > 0, k - 1));
                assert.equal(L.xPos, expectLink(i + 1 < n, k + 1));
                assert.equal(L.yNeg, expectLink(j > 0, k - n));
                assert.equal(L.yPos, expectLink(j + 1 < n, k + n));
                assert.equal(L.diag, expectLink(i + 1 < n && j + 1 < n, k + n + 1));
            }
            assert.equal(cloth.pinnedCount, pins.length);
            // Tethers: the two nearest pins (brute force) and their rest-layout distances
            let maxTetherErr = 0;
            for (let k = 0; k < cloth.count; k++) {
                const L = links[k];
                if (cloth.positions[k * 4 + 3] === 0 || pins.length === 0) {
                    assert.equal(L.pinA, NONE); assert.equal(L.pinB, NONE);
                    continue;
                }
                const sorted = pins.map(p => [p, dist3(restOf(k), restOf(p))]).sort((x, y) => x[1] - y[1] || x[0] - y[0]);
                assert.ok(Math.abs(dist3(restOf(k), restOf(L.pinA)) - sorted[0][1]) < 1e-12, `${key}: pinA is a nearest pin`);
                maxTetherErr = Math.max(maxTetherErr, Math.abs(L.restA - sorted[0][1]));
                if (pins.length > 1) {
                    assert.ok(Math.abs(dist3(restOf(k), restOf(L.pinB)) - sorted[1][1]) < 1e-12);
                    assert.notEqual(L.pinA, L.pinB);
                    maxTetherErr = Math.max(maxTetherErr, Math.abs(L.restB - sorted[1][1]));
                } else {
                    assert.equal(L.pinB, NONE);
                }
            }
            assert.ok(maxTetherErr < 1e-6);
            if (n === 13) t.diagnostic(`${key} n=13: ${cloth.numConstraints} constraints [${cloth.kindCounts}], ${pins.length} pins, max rel rest error ${maxRestErr.toExponential(2)}`);
        }
    }
});

test('iterationStiffness compounds back to k over the iterations', (t) => {
    let maxErr = 0;
    for (const k of [0, 0.1, 0.2, 0.5, 0.9, 0.999, 1]) {
        for (const it of [1, 2, 4, 8, 16, 32]) {
            const ks = iterationStiffness(k, it);
            assert.ok(ks >= 0 && ks <= 1);
            maxErr = Math.max(maxErr, Math.abs(1 - (1 - ks) ** it - k));
        }
    }
    t.diagnostic(`max |1 − (1 − k_s)^iters − k| ${maxErr.toExponential(2)}`);
    assert.ok(maxErr < 1e-12);
    assert.equal(iterationStiffness(1, 8), 1);
    assert.equal(iterationStiffness(1.5, 8), 1, 'clamped to [0, 1]');
    assert.equal(iterationStiffness(-1, 8), 0);
    assert.ok(Math.abs(iterationStiffness(0.3, 1) - 0.3) < 1e-15, 'one iteration: k itself');
});

// ================================================================ camera and picking (shipped)

function project(m, p) {
    const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return [x / w, y / w, z / w, w];
}

test('OrbitCamera: orthonormal basis, viewProj and ray() agree pixel-for-pixel', (t) => {
    const r = mulberry(21);
    let maxBasis = 0, maxPix = 0;
    for (let k = 0; k < 200; k++) {
        const cam = new OrbitCamera();
        cam.distance = 1 + 8 * r();
        cam.rotation = { x: (r() - 0.5) * 2.8, y: r() * 2 * Math.PI };
        cam.target = [r() - 0.5, r() - 0.5, r() - 0.5];
        const { eye, forward, right, up } = cam.getBasis();
        const d = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        maxBasis = Math.max(maxBasis, Math.abs(d(forward, right)), Math.abs(d(forward, up)), Math.abs(d(right, up)),
            Math.abs(d(up, up) - 1), Math.abs(dist3(eye, cam.target) - cam.distance));
        const W = 800, H = 600;
        const vp = cam.getViewProj(W / H);
        const c = project(vp, cam.target);
        assert.ok(Math.abs(c[0]) < 1e-5 && Math.abs(c[1]) < 1e-5 && c[2] > 0 && c[2] < 1, 'target at screen center, depth in [0, 1]');
        // A point along ray(px, py) projects back onto (px, py)
        const px = r() * W, py = r() * H;
        const ray = cam.ray(px, py, W, H);
        const s = 0.5 + 3 * r();
        const q = project(vp, [0, 1, 2].map(i => ray.origin[i] + ray.dir[i] * s));
        maxPix = Math.max(maxPix, Math.abs((q[0] + 1) / 2 * W - px), Math.abs((1 - q[1]) / 2 * H - py));
    }
    t.diagnostic(`basis error ${maxBasis.toExponential(2)}, ray→viewProj reprojection error ${maxPix.toExponential(2)} px`);
    assert.ok(maxBasis < 1e-12);
    assert.ok(maxPix < 1e-2, 'f32 matrix: < 0.01 px');
    assert.equal(FOV, Math.PI / 4);
});

test('pickParticle picks the nearest particle near the ray, ignores ones behind, falls back, or misses', (t) => {
    const r = mulberry(8);
    const pixelAngle = 0.002;
    let agree = 0;
    for (let trial = 0; trial < 300; trial++) {
        const count = 200;
        const pos = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) pos.set([r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, 1], i * 4);
        const origin = [0, 0, 6];
        const dir = (() => { const v = [r() * 0.4 - 0.2, r() * 0.4 - 0.2, -1]; const l = Math.hypot(...v); return v.map(x => x / l); })();
        // Brute-force reference: smallest t among particles within 10 pixel angles, else the
        // smallest angle if within 30 pixel angles
        let best = -1, bestT = Infinity, fb = -1, fbA = Infinity;
        for (let i = 0; i < count; i++) {
            const p = [pos[i * 4] - origin[0], pos[i * 4 + 1] - origin[1], pos[i * 4 + 2] - origin[2]];
            const tt = p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2];
            if (tt <= 0) continue;
            const angle = Math.acos(Math.min(1, tt / Math.hypot(...p)));   // exact angle, ~ perp / t
            if (angle < pixelAngle * 10 && tt < bestT) { bestT = tt; best = i; }
            if (angle < fbA) { fbA = angle; fb = i; }
        }
        const expect = best >= 0 ? best : (fbA < pixelAngle * 30 ? fb : -1);
        if (pickParticle(pos, count, { origin, dir }, pixelAngle) === expect) agree++;
    }
    t.diagnostic(`random scenes: ${agree}/300 agree with brute force (tan vs angle may differ at the threshold)`);
    assert.ok(agree >= 297);
    // Deterministic cases
    const pos = new Float32Array([0, 0, -1, 1, 0, 0, -3, 1, 0, 0, 2, 1, 1, 1, 1, 1]);
    const ray = { origin: [0, 0, 1], dir: [0, 0, -1] };
    assert.equal(pickParticle(pos, 4, ray, 0.01), 0, 'nearer of two on-ray particles; one behind ignored');
    assert.equal(pickParticle(new Float32Array([0.8, 0, -1, 1]), 1, ray, 0.01), -1, 'nothing within 30 pixel angles');
    assert.equal(pickParticle(new Float32Array([0.25, 0, -1, 1]), 1, ray, 0.01), 0, 'fallback within 30 pixel angles');
});

// ================================================================ uniform writers (shipped)


test('writeSimParams bytes decode through the WGSL SimParams struct', (t) => {
    const { sim, device } = simWithMock({ grabIndex: 1234, grabTarget: [0.4, 0.8, -1.2], grabPos: [0.1, 0.2, 0.3] });
    sim.writeSimParams(4);
    const p = decodeStruct(device.lastWrite(sim.simParamsBuffer).bytes, SIM);
    const f = Math.fround;
    const preset = PRESETS.curtain;
    assert.deepEqual(p.gravity, [0, f(-9.8), 0, 0]);
    assert.deepEqual(p.wind, [0, 0, -5, f(preset.gust)]);
    assert.deepEqual(p.sphere, [...preset.sphere.center, preset.sphere.radius].map(f));
    assert.deepEqual(p.grabDelta, [0.3 / 4, 0.6 / 4, -1.5 / 4, 0].map(f));
    assert.deepEqual(p.stiffness.slice(0, 3), [0.9, 0.45, 0.18].map(k => f(iterationStiffness(k, 8))));
    assert.equal(p.dt, f(SUBSTEP_DT));
    assert.ok(p.damping > 0.99 && p.damping < 1);
    assert.equal(p.floorY, f(-1.2));
    assert.equal(p.time, 3.25);
    assert.equal(p.tearRatio, f(1.8));
    assert.equal(p.tearOn, 1);
    assert.equal(p.grabIndex, 1234);
    assert.equal(p.gridN, 64);
    assert.equal(p.sphereOn, 1);
    assert.equal(p.thickness, f(0.015));
    assert.equal(p.grabRadius, 2);
    t.diagnostic(`SimParams: ${Object.entries(SIM.offsets).map(([k, v]) => `${k}@${v}`).join(' ')}`);
    // No grab: grabIndex stays NONE (0xffffffff), not a float
    const { sim: s2, device: d2 } = simWithMock({ presetKey: 'flag', tearOn: false });
    s2.writeSimParams(2);
    const q = decodeStruct(d2.lastWrite(s2.simParamsBuffer).bytes, SIM);
    assert.equal(q.grabIndex, NONE);
    assert.equal(q.sphereOn, 0);
    assert.equal(q.tearOn, 0);
});

test('writeCamera bytes decode through the WGSL Camera struct', () => {
    const { sim, device } = simWithMock({ gridN: 32, grabIndex: 77, presetKey: 'drape' });
    sim.writeCamera();
    const c = decodeStruct(device.lastWrite(sim.cameraBuffer).bytes, CAMERA);
    const vp = sim.camera.getViewProj(2);
    assert.deepEqual(c.viewProj.flat(), Array.from(vp));
    assert.deepEqual(c.eye, [...sim.camera.getEye(), 1].map(Math.fround));
    assert.deepEqual(c.sphere, [0.04, -0.35, 0.02, 0.6].map(Math.fround));
    assert.ok(Math.abs(Math.hypot(...c.lightDir.slice(0, 3)) - 1) < 1e-6);
    assert.deepEqual(c.viewport, [1000, 500]);
    assert.equal(c.floorY, Math.fround(-1.2));
    assert.equal(c.pointSize, 3);
    assert.equal(c.gridN, 32);
    assert.equal(c.sphereOn, 1);
    assert.equal(c.grabIndex, 77);
});

test('buildGrid writes one ColorRange per dynamic-offset slot, matching cloth.ranges', () => {
    const { sim, device } = simWithMock({ gridN: 16, presetKey: 'corners' });
    Object.assign(sim, { colorStride: 256, colorBuffer: { label: 'colors' }, counterBuffer: { label: 'counter' }, generation: 0 });
    sim.render = () => {};
    sim.buildGrid();
    const w = device.lastWrite(sim.colorBuffer);
    assert.equal(w.bytes.byteLength, 256 * NUM_COLORS);
    sim.cloth.ranges.forEach((r, c) => {
        assert.deepEqual(decodeStruct(w.bytes, RANGE, c * 256), { start: r.start, count: r.count, pad0: 0, pad1: 0 });
    });
});

// ================================================================ compute-shader replica

test('replica: constraints within one color are independent (any invocation order, same bits)', (t) => {
    const n = 12;
    const cloth = buildCloth(n, 'curtain');
    const P = simParamsFor({ gridN: n, presetKey: 'curtain', tearOn: false });
    const A = makeState(cloth, P), B = makeState(cloth, P);
    const r = mulberry(31);
    const orders = cloth.ranges.map(rg => {
        const o = Array.from({ length: rg.count }, (_, i) => i);
        for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; }
        return o;
    });
    for (let f = 0; f < 10; f++) { frame(A, 4, 8); frame(B, 4, 8, orders); }
    let maxDiff = 0;
    for (let i = 0; i < A.pos.length; i++) maxDiff = Math.max(maxDiff, Math.abs(A.pos[i] - B.pos[i]));
    t.diagnostic(`max position difference between in-order and shuffled solves: ${maxDiff}`);
    assert.equal(maxDiff, 0);
});

test('replica: hanging cloth is stable — finite, bounded stretch, pins fixed, no energy gain', (t) => {
    const n = 16, g = 9.8;
    for (const presetKey of ['corners', 'edge', 'drape']) {
        const cloth = buildCloth(n, presetKey);
        const P = simParamsFor({ gridN: n, presetKey, tearOn: false, windStrength: 0, stiffness: 1, gravity: g });
        const S = makeState(cloth, P);
        const pinned0 = cloth.positions.filter((_, i) => i % 4 !== 3 && cloth.positions[i - (i % 4) + 3] === 0);
        const e0 = energy(S, g);
        let eMax = -Infinity, stretchMax = 0;
        const frames = 120;          // 480 substeps = 2 s of simulated time
        for (let f = 0; f < frames; f++) {
            P.time = f * 4 * SUBSTEP_DT;
            frame(S, 4, 8);
            eMax = Math.max(eMax, energy(S, g));
            if (f > frames / 2) stretchMax = Math.max(stretchMax, maxStretch(S));
        }
        assert.ok(S.pos.every(Number.isFinite), `${presetKey}: NaN/Inf`);
        const pinnedNow = S.pos.filter((_, i) => i % 4 !== 3 && S.pos[i - (i % 4) + 3] === 0);
        assert.deepEqual(Array.from(pinnedNow), Array.from(pinned0), 'pins never move');
        // Floor / sphere constraints hold (within the collision margin, after the last collide pass)
        let minFloor = Infinity, minSphere = Infinity;
        for (let i = 0; i < S.count; i++) {
            minFloor = Math.min(minFloor, S.pos[i * 4 + 1] - (P.floorY + P.thickness));
            if (P.sphereOn) minSphere = Math.min(minSphere, Math.hypot(S.pos[i * 4] - P.sphere[0], S.pos[i * 4 + 1] - P.sphere[1], S.pos[i * 4 + 2] - P.sphere[2]) - (P.sphere[3] + P.thickness));
        }
        // Energy scale: the potential energy range over the cloth's extent (2 m drop per particle)
        const scale = cloth.count * g * 2;
        const gain = (eMax - e0) / scale;
        t.diagnostic(`${presetKey}: max structural stretch ${stretchMax.toFixed(4)} (last second), energy gain ${gain.toExponential(2)} of m·g·2m, floor margin ${minFloor.toExponential(2)}, sphere margin ${P.sphereOn ? minSphere.toExponential(2) : '-'}`);
        assert.ok(stretchMax < 1.1, `${presetKey}: structural links stretch < 10% (tethers + 8 iterations)`);
        assert.ok(gain < 1e-3, `${presetKey}: PBD with damping must not gain energy`);
        assert.ok(minFloor >= -1e-5);
        if (P.sphereOn) assert.ok(minSphere >= -1e-5);
    }
});

test('replica: tearing — none below the threshold, tears above it, children follow parents', (t) => {
    const n = 16;
    const cloth = buildCloth(n, 'corners');
    // Gentle hang with tearing on: max stretch stays below tearRatio, so nothing breaks
    const P = simParamsFor({ gridN: n, presetKey: 'corners', tearOn: true, tearRatio: 1.8, windStrength: 0 });
    const S = makeState(cloth, P);
    for (let f = 0; f < 60; f++) frame(S, 4, 8);
    assert.equal(S.torn, 0);
    // Grab the middle of the bottom edge and yank it 0.6 m in one frame (4 substeps): links tear
    const grab = (n - 1) * n + n / 2;
    const P2 = simParamsFor({ gridN: n, presetKey: 'corners', tearOn: true, tearRatio: 1.8, windStrength: 0, grabIndex: grab, grabTarget: [0, 0, 0.6], grabPos: [0, 0, 0] });
    S.P = P2;
    for (let f = 0; f < 10; f++) frame(S, 4, 8);
    const brokenCount = S.broken.reduce((a, b) => a + b, 0);
    assert.ok(brokenCount > 0, 'yanking tears the cloth');
    assert.equal(S.torn, brokenCount, 'tornCount counts each break once');
    // Every broken bend / anti-diagonal broke with a parent; structural/diagonal ones by stretch
    let byParent = 0;
    for (let ci = 0; ci < cloth.numConstraints; ci++) {
        if (!S.broken[ci]) continue;
        const pa = S.cu[ci * 8 + 4], pb = S.cu[ci * 8 + 5], kind = S.cu[ci * 8 + 3];
        const parentBroken = (pa !== NONE && S.broken[pa]) || (pb !== NONE && S.broken[pb]);
        if (kind === KIND_BEND) assert.ok(parentBroken, 'bend links only break with a parent');
        if (parentBroken) byParent++;
    }
    // Same yank without tearing: nothing breaks, the cloth just stretches
    const S2 = makeState(cloth, simParamsFor({ gridN: n, presetKey: 'corners', tearOn: false, windStrength: 0 }));
    for (let f = 0; f < 60; f++) frame(S2, 4, 8);
    S2.P = { ...P2, tearOn: 0 };
    for (let f = 0; f < 10; f++) frame(S2, 4, 8);
    t.diagnostic(`yank: ${brokenCount} links broken (${byParent} via parents); without tearing max stretch ${maxStretch(S2).toFixed(3)}, broken ${S2.torn}`);
    assert.equal(S2.torn, 0);
    assert.ok(maxStretch(S2) > 1.8, 'the yank does stretch links beyond tearRatio');
    assert.ok(S.pos.every(Number.isFinite));
});
