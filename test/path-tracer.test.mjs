// Path tracer (js/examples/path-tracer.js).
//
// Real shipped code: SceneBuilder packing (vs the WGSL Prim struct), every SCENES entry,
// intersectScene (the CPU mirror of traceScene used for click-to-focus), cameraBasis, and the
// PathTracer.writeParams / writeDisplay uniform writers (vs the WGSL Params / Display structs).
//
// CPU replica of the trace shader's sampling and path loop (pcg rand, basisT/basisB, cosineSample,
// sampleLight, lightPdf, powerHeuristic, tracePath's diffuse branch with NEE + MIS and Russian
// roulette); validates the algorithm and index math, not the WGSL itself — see #51.
// The replica is compared against analytic references: the cosine-distribution CDF, rectangle and
// cone solid angles, and a closed emissive "furnace" box whose radiance is a geometric series.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SCENES, SceneBuilder, intersectScene, cameraBasis, PathTracer, wgslModules } from '../js/examples/path-tracer.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const traceCode = wgslModules().find(m => m.name === 'trace').code;
const displayCode = wgslModules().find(m => m.name === 'display').code;
const PRIM = structLayout(traceCode, 'Prim');
const PARAMS = structLayout(traceCode, 'Params');
const DISPLAY = structLayout(displayCode, 'Display');

const SHAPE = { sphere: 0, quad: 1, box: 2, plane: 3 };
const T_MIN = 1e-4;
const EPS = 1e-3;
const PI = 3.14159265;   // the shader's constant
const MAX_BOUNCES = 8;
const RR_START = 2;

// ---------------------------------------------------------------- small vector helpers
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a) => scale(a, 1 / len(a));

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
function randomUnit(r) {
    const z = 1 - 2 * r();
    const phi = 2 * Math.PI * r();
    const s = Math.sqrt(1 - z * z);
    return [s * Math.cos(phi), s * Math.sin(phi), z];
}

// Decode the packed scene buffer with the layout parsed from the WGSL Prim struct
function decodePrims(scene) {
    return Array.from({ length: scene.count }, (_, i) => decodeStruct(scene.data, PRIM, i * PRIM.size));
}

// ---------------------------------------------------------------- independent intersection reference
// Ray o + t d against parallelogram Q + a u + b v: solve [d, -u, -v](t, a, b) = Q - o by Cramer's rule.
function refParallelogram(o, d, Q, u, v) {
    const det3 = (c0, c1, c2) => dot(c0, cross(c1, c2));
    const nu = scale(u, -1), nv = scale(v, -1), rhs = sub(Q, o);
    const D = det3(d, nu, nv);
    if (Math.abs(D) < 1e-14) return Infinity;
    const t = det3(rhs, nu, nv) / D;
    const a = det3(d, rhs, nv) / D;
    const b = det3(d, nu, rhs) / D;
    return t >= T_MIN && a >= 0 && a <= 1 && b >= 0 && b <= 1 ? t : Infinity;
}
// Sphere with the general quadratic |o + t d - c|² = r² (a = d·d, not assuming |d| = 1)
function refSphere(o, d, c, r) {
    const oc = sub(o, c);
    const A = dot(d, d), B = 2 * dot(oc, d), C = dot(oc, oc) - r * r;
    const disc = B * B - 4 * A * C;
    if (disc < 0) return Infinity;
    const roots = [(-B - Math.sqrt(disc)) / (2 * A), (-B + Math.sqrt(disc)) / (2 * A)];
    return roots.find(t => t >= T_MIN) ?? Infinity;
}
// Box: the six faces as world-space parallelograms
function refBox(o, d, c, h, cs, sn) {
    const rot = (v) => [cs * v[0] + sn * v[2], v[1], -sn * v[0] + cs * v[2]];   // local -> world
    let best = Infinity;
    for (let k = 0; k < 3; k++) {
        const a = (k + 1) % 3, b = (k + 2) % 3;
        for (const s of [-1, 1]) {
            const corner = [0, 0, 0], u = [0, 0, 0], v = [0, 0, 0];
            corner[k] = s * h[k]; corner[a] = -h[a]; corner[b] = -h[b];
            u[a] = 2 * h[a]; v[b] = 2 * h[b];
            best = Math.min(best, refParallelogram(o, d, add(c, rot(corner)), rot(u), rot(v)));
        }
    }
    return best;
}
function refPlane(o, d, n, off) {
    const denom = dot(n, d);
    if (Math.abs(denom) < 1e-12) return Infinity;
    const t = (off - dot(n, o)) / denom;
    return t >= T_MIN ? t : Infinity;
}
function refIntersect(prims, o, d) {
    let best = Infinity, index = -1;
    prims.forEach((p, i) => {
        const g0 = p.geo0, g1 = p.geo1, g2 = p.geo2;
        let t = Infinity;
        switch (p.kind[0]) {
            case SHAPE.sphere: t = refSphere(o, d, g0.slice(0, 3), g0[3]); break;
            case SHAPE.quad: t = refParallelogram(o, d, g0.slice(0, 3), g1.slice(0, 3), g2.slice(0, 3)); break;
            case SHAPE.box: t = refBox(o, d, g0.slice(0, 3), g1.slice(0, 3), g2[0], g2[1]); break;
            case SHAPE.plane: t = refPlane(o, d, g0.slice(0, 3), g0[3]); break;
        }
        if (t < best) { best = t; index = i; }
    });
    return { t: best, index };
}

// ---------------------------------------------------------------- shader replica (trace module)
// PCG RXS-M-XS, exactly as rand()/pcgHash() in the shader (u32 wrap-around via Math.imul / >>> 0)
function pcgHash(v) {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> (((state >>> 28) + 4) >>> 0)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
function makeRand(seed) {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 747796405) + 2891336453) >>> 0;
        const s = state;
        const word = Math.imul(((s >>> (((s >>> 28) + 4) >>> 0)) ^ s) >>> 0, 277803737) >>> 0;
        return (((word >>> 22) ^ word) >>> 8) / 16777216;
    };
}
function basisT(n) {
    const s = n[2] >= 0 ? 1 : -1;
    const a = -1 / (s + n[2]);
    const b = n[0] * n[1] * a;
    return [1 + s * n[0] * n[0] * a, s * b, -s * n[0]];
}
function basisB(n) {
    const s = n[2] >= 0 ? 1 : -1;
    const a = -1 / (s + n[2]);
    const b = n[0] * n[1] * a;
    return [b, s + n[1] * n[1] * a, -n[1]];
}
function cosineSample(n, u1, u2) {
    const r = Math.sqrt(u1), phi = 2 * PI * u2, z = Math.sqrt(Math.max(0, 1 - u1));
    return normalize(add(add(scale(basisT(n), r * Math.cos(phi)), scale(basisB(n), r * Math.sin(phi))), scale(n, z)));
}
function sampleLight(p, x, u1, u2) {
    const none = { wi: [0, 1, 0], dist: 0, pdf: 0 };
    if (p.kind[0] === SHAPE.quad) {
        const u = p.geo1.slice(0, 3), v = p.geo2.slice(0, 3);
        const nArea = cross(u, v), area = len(nArea);
        const pt = add(add(p.geo0.slice(0, 3), scale(u, u1)), scale(v, u2));
        const toL = sub(pt, x), d2 = dot(toL, toL);
        if (d2 < 1e-8 || area < 1e-12) return none;
        const dist = Math.sqrt(d2), wi = scale(toL, 1 / dist);
        const cosL = -dot(scale(nArea, 1 / area), wi);
        if (cosL <= 1e-6) return none;
        return { wi, dist, pdf: d2 / (cosL * area) };
    }
    if (p.kind[0] === SHAPE.sphere) {
        const toC = sub(p.geo0.slice(0, 3), x), d2 = dot(toC, toC), r = p.geo0[3];
        if (d2 <= r * r * 1.0001) return none;
        const dist = Math.sqrt(d2), axis = scale(toC, 1 / dist);
        const cosMax = Math.sqrt(Math.max(0, 1 - r * r / d2));
        const cosT = 1 - u1 * (1 - cosMax), sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT)), phi = 2 * PI * u2;
        const wi = normalize(add(add(scale(basisT(axis), sinT * Math.cos(phi)), scale(basisB(axis), sinT * Math.sin(phi))), scale(axis, cosT)));
        return { wi, dist, pdf: 1 / Math.max(2 * PI * (1 - cosMax), 1e-12) };
    }
    return none;
}
function lightPdf(p, x, wi, t, nHit) {
    if (p.kind[0] === SHAPE.quad) {
        const area = len(cross(p.geo1.slice(0, 3), p.geo2.slice(0, 3)));
        const cosL = -dot(nHit, wi);
        if (cosL <= 1e-6 || area < 1e-12) return 0;
        return t * t / (cosL * area);
    }
    const toC = sub(p.geo0.slice(0, 3), x), d2 = dot(toC, toC), r = p.geo0[3];
    if (d2 <= r * r * 1.0001) return 0;
    const cosMax = Math.sqrt(Math.max(0, 1 - r * r / d2));
    return 1 / Math.max(2 * PI * (1 - cosMax), 1e-12);
}
function powerHeuristic(a, b) {
    const a2 = a * a, b2 = b * b;
    return a2 + b2 > 0 ? a2 / (a2 + b2) : 0;
}
// Geometric normal of a hit, as traceScene computes it (spheres, quads and planes)
function hitNormal(p, pos) {
    if (p.kind[0] === SHAPE.sphere) return scale(sub(pos, p.geo0.slice(0, 3)), 1 / p.geo0[3]);
    if (p.kind[0] === SHAPE.quad) return normalize(cross(p.geo1.slice(0, 3), p.geo2.slice(0, 3)));
    if (p.kind[0] === SHAPE.plane) return p.geo0.slice(0, 3);
    throw new Error('replica only covers spheres, quads and planes');
}
// tracePath() for a diffuse-only scene, starting from an explicit ray (no camera); uses the shipped
// intersectScene() for closest hits. env: radiance for escaped rays.
function tracePath(scene, prims, ro, rd, rand, { bounces, useNEE, env = 0 }) {
    let radiance = 0, throughput = 1, prevSpecular = true, prevPdf = 0, prevPos = ro;
    const lightCount = Math.min(scene.lightCount, scene.count);
    const nee = useNEE && lightCount > 0;
    for (let seg = 0; seg <= MAX_BOUNCES; seg++) {
        const hit = intersectScene(scene, ro, rd);
        if (!Number.isFinite(hit.t)) { radiance += throughput * env; break; }
        const p = prims[hit.index];
        if (p.kind[1] !== 0) throw new Error('replica only covers diffuse materials');
        const pos = add(ro, scale(rd, hit.t));
        const normal = hitNormal(p, pos);
        const frontFace = dot(normal, rd) < 0;
        const emission = p.emission[0];   // grey scenes: one channel is enough
        if (frontFace && emission > 0) {
            let w = 1;
            if (!prevSpecular && nee && hit.index < lightCount) {
                const pl = lightPdf(p, prevPos, rd, hit.t, normal) / lightCount;
                w = powerHeuristic(prevPdf, pl);
            }
            radiance += throughput * emission * w;
        }
        if (seg >= bounces) break;
        const nf = frontFace ? normal : scale(normal, -1);
        const albedo = p.albedo[0];
        if (nee) {
            const li = Math.min(Math.floor(rand() * lightCount), lightCount - 1);
            const ls = sampleLight(prims[li], pos, rand(), rand());
            const cosS = dot(nf, ls.wi);
            if (ls.pdf > 0 && cosS > 0) {
                const sro = add(pos, scale(nf, EPS));
                const shadow = intersectScene(scene, sro, ls.wi);
                if (shadow.index === li && Number.isFinite(shadow.t)) {
                    const sn = hitNormal(prims[li], add(sro, scale(ls.wi, shadow.t)));
                    if (dot(sn, ls.wi) < 0) {
                        const pdfL = ls.pdf / lightCount;
                        const w = powerHeuristic(pdfL, cosS / PI);
                        radiance += throughput * albedo * (cosS / PI) * prims[li].emission[0] * (w / pdfL);
                    }
                }
            }
        }
        rd = cosineSample(nf, rand(), rand());
        ro = add(pos, scale(nf, EPS));
        prevPdf = Math.max(dot(nf, rd), 0) / PI;
        throughput *= albedo;
        prevSpecular = false;
        prevPos = pos;
        if (seg >= RR_START) {
            const q = Math.min(Math.max(throughput, 0.05), 0.95);
            if (rand() >= q) break;
            throughput /= q;
        }
        if (throughput <= 0) break;
    }
    return radiance;
}

// ================================================================ tests

test('Prim / Params / Display struct sizes match the JS constants', (t) => {
    t.diagnostic(`Prim ${PRIM.size} B, Params ${PARAMS.size} B, Display ${DISPLAY.size} B`);
    assert.equal(PRIM.size, 96);
    assert.deepEqual(PRIM.offsets, { geo0: 0, geo1: 16, geo2: 32, albedo: 48, emission: 64, kind: 80 });
    assert.equal(PARAMS.size, 160);
    assert.equal(PARAMS.offsets.seed, 144);
    assert.equal(DISPLAY.size, 24);
});

test('SceneBuilder packs primitives byte-for-byte like the WGSL Prim struct, lights first', (t) => {
    const mat = (type, albedo, param, emission) => ({ type, albedo, param, emission });
    const scene = new SceneBuilder()
        .plane([0, 1, 0], -0.5, mat(0, [0.1, 0.2, 0.3], 0, [0, 0, 0]), [0.7, 0.8, 0.9], 3)
        .sphere([1, 2, 3], 0.75, mat(1, [0.4, 0.5, 0.6], 0.25, [0, 0, 0]))
        .box([-1, 0.5, 2], [0.3, 0.4, 0.5], 30, mat(2, [1, 1, 1], 1.5, [0, 0, 0]))
        .quad([-0.25, 1.9, -0.2], [0.5, 0, 0], [0, 0, 0.4], mat(0, [0, 0, 0], 0, [17, 12, 4]))
        .sphere([0, 5, 0], 0.2, mat(0, [0, 0, 0], 0, [60, 30, 10]))
        .build();
    assert.equal(scene.count, 5);
    assert.equal(scene.lightCount, 2);
    assert.equal(scene.data.byteLength, 5 * PRIM.size);
    const p = decodePrims(scene);
    const close = (a, b) => a.every((x, i) => Math.abs(x - b[i]) < 1e-6);
    // Lights (emissive quad, emissive sphere) are sorted to the front in insertion order
    assert.deepEqual(p[0].kind, [SHAPE.quad, 0, 0, 0]);
    assert.ok(close(p[0].geo0, [-0.25, 1.9, -0.2, 0]) && close(p[0].geo1, [0.5, 0, 0, 0]) && close(p[0].geo2, [0, 0, 0.4, 0]));
    assert.ok(close(p[0].emission, [17, 12, 4, 0]));
    assert.deepEqual(p[1].kind, [SHAPE.sphere, 0, 0, 0]);
    assert.ok(close(p[1].geo0, [0, 5, 0, 0.2]));
    assert.deepEqual(p[2].kind, [SHAPE.plane, 0, 0, 0]);
    assert.ok(close(p[2].geo0, [0, 1, 0, -0.5]) && close(p[2].geo1, [0.7, 0.8, 0.9, 3]));
    assert.ok(close(p[2].albedo, [0.1, 0.2, 0.3, 0]));
    assert.deepEqual(p[3].kind, [SHAPE.sphere, 1, 0, 0]);
    assert.ok(close(p[3].albedo, [0.4, 0.5, 0.6, 0.25]));
    assert.deepEqual(p[4].kind, [SHAPE.box, 2, 0, 0]);
    const a = 30 * Math.PI / 180;
    assert.ok(close(p[4].geo1, [0.3, 0.4, 0.5, 0]) && close(p[4].geo2, [Math.cos(a), Math.sin(a), 0, 0]));
    assert.ok(close(p[4].albedo, [1, 1, 1, 1.5]));
    t.diagnostic('5 primitives decoded through the WGSL Prim layout; all fields match');
});

test('every SCENES entry builds a valid scene', (t) => {
    for (const [key, def] of Object.entries(SCENES)) {
        const scene = def.build();
        const prims = decodePrims(scene);
        const isLight = (p) => (p.kind[0] === SHAPE.sphere || p.kind[0] === SHAPE.quad) && Math.max(...p.emission.slice(0, 3)) > 0;
        assert.ok(scene.count > 0 && scene.count <= 64, `${key}: ${scene.count} primitives`);
        assert.equal(scene.lightCount, prims.filter(isLight).length, `${key}: lightCount`);
        prims.forEach((p, i) => {
            assert.equal(isLight(p), i < scene.lightCount, `${key}: lights must lead (prim ${i})`);
            assert.ok(p.kind[0] <= 3 && p.kind[1] <= 2, `${key}: prim ${i} kind ${p.kind}`);
            if (p.kind[0] === SHAPE.sphere) assert.ok(p.geo0[3] > 0);
            if (p.kind[1] === 2) assert.ok(p.albedo[3] > 1, `${key}: glass IOR > 1`);
            for (const v of [...p.geo0, ...p.geo1, ...p.geo2, ...p.albedo, ...p.emission]) assert.ok(Number.isFinite(v));
        });
        assert.ok(def.bounces >= 1 && def.bounces <= MAX_BOUNCES);
        assert.ok(scene.lightCount > 0 || def.env.type !== 0, `${key} would render black`);
        // The camera looks at something: the center ray hits the scene
        const { eye, forward } = cameraBasis(def.camera);
        assert.ok(Number.isFinite(intersectScene(scene, eye, forward).t), `${key}: center ray misses`);
        t.diagnostic(`${key}: ${scene.count} prims, ${scene.lightCount} lights, ${def.bounces} bounces`);
    }
});

test('intersectScene matches an independent intersection reference on random rays', (t) => {
    // Both use float64 arithmetic on the same packed f32 geometry; distances agree to ~1e-15 except
    // for grazing rays, where the different formulations (Cramer vs plane/slab test) are
    // ill-conditioned. 1e-6 relative is the documented tolerance.
    const TOL = 1e-6;
    const r = mulberry(12345);
    for (const [key, def] of Object.entries(SCENES)) {
        const scene = def.build();
        const prims = decodePrims(scene);
        const { eye } = cameraBasis(def.camera);
        const target = def.camera.target;
        let hits = 0, misses = 0, maxRel = 0, indexMismatch = 0;
        for (let k = 0; k < 3000; k++) {
            // Origins: the camera eye, points around the target (inside boxes/spheres sometimes)
            const o = k % 3 === 0 ? eye : add(target, scale(randomUnit(r), 2 * r()));
            const d = randomUnit(r);
            const got = intersectScene(scene, o, d);
            const ref = refIntersect(prims, o, d);
            if (!Number.isFinite(ref.t)) {
                assert.equal(got.t, Infinity, `${key}: shipped hit at t=${got.t} where reference misses`);
                misses++;
                continue;
            }
            assert.ok(Number.isFinite(got.t), `${key}: shipped misses where reference hits at t=${ref.t}`);
            const rel = Math.abs(got.t - ref.t) / Math.max(1, ref.t);
            maxRel = Math.max(maxRel, rel);
            if (got.index !== ref.index) {
                // Only legitimate for ties (shared edges, a box resting on the floor plane): the
                // primitive the shipped code picked must be hit at the same distance
                indexMismatch++;
                const alt = refIntersect([prims[got.index]], o, d).t;
                assert.ok(Math.abs(alt - ref.t) / Math.max(1, ref.t) < TOL, `${key}: picked prim ${got.index}, reference ${ref.index}`);
            }
            hits++;
        }
        t.diagnostic(`${key}: ${hits} hits, ${misses} misses, max rel |Δt| ${maxRel.toExponential(2)}, index ties ${indexMismatch}`);
        assert.ok(maxRel < TOL, `${key}: max rel error ${maxRel}`);
    }
});

test('cameraBasis is orthonormal, looks at the target from the orbit distance', (t) => {
    const r = mulberry(7);
    let maxErr = 0;
    for (let k = 0; k < 500; k++) {
        const cam = { target: [r() * 4 - 2, r() * 2, r() * 4 - 2], yaw: r() * 2 * Math.PI, pitch: (r() - 0.5) * 3, distance: 0.5 + 10 * r() };
        const { eye, forward, right, up } = cameraBasis(cam);
        const errs = [
            Math.abs(len(forward) - 1), Math.abs(len(right) - 1), Math.abs(len(up) - 1),
            Math.abs(dot(forward, right)), Math.abs(dot(forward, up)), Math.abs(dot(right, up)),
            Math.abs(len(sub(cam.target, eye)) - cam.distance) / cam.distance,
            len(sub(normalize(sub(cam.target, eye)), forward)),
            Math.abs(right[1]),                                     // right stays horizontal
            len(sub(cross(right, forward), up)),                    // right-handed: up = right × forward
        ];
        maxErr = Math.max(maxErr, ...errs);
        assert.ok(up[1] > 0, 'up has a positive y component for |pitch| < π/2');
    }
    t.diagnostic(`max orthonormality / look-at error ${maxErr.toExponential(2)}`);
    assert.ok(maxErr < 1e-12);
});

// Builds a PathTracer without its constructor (which needs a canvas) and the mock device
function tracerWithMock() {
    const device = mockDevice();
    const pt = Object.create(PathTracer.prototype);
    pt.device = device;
    pt.paramsBuffer = { label: 'params' };
    pt.displayBuffer = { label: 'display' };
    pt.params = new ArrayBuffer(160);
    pt.paramsF32 = new Float32Array(pt.params);
    pt.paramsU32 = new Uint32Array(pt.params);
    pt.displayData = new ArrayBuffer(24);
    pt.displayF32 = new Float32Array(pt.displayData);
    pt.displayU32 = new Uint32Array(pt.displayData);
    return { pt, device };
}

test('writeParams bytes decode correctly through the WGSL Params struct', (t) => {
    const { pt, device } = tracerWithMock();
    pt.camera = { ...SCENES.spheres.camera };
    pt.env = SCENES.spheres.env;
    pt.scene = SCENES.spheres.build();
    pt.renderWidth = 600; pt.renderHeight = 450;
    pt.sampleCount = 777;
    pt.maxBounces = 12;           // clamped to MAX_BOUNCES
    pt.useNEE = true;
    pt.dofOn = true;
    pt.clampMax = 20;
    pt.seedValue = 0xdeadbeef;
    pt.writeParams();
    const w = device.lastWrite(pt.paramsBuffer);
    assert.ok(w, 'writeParams wrote the params buffer');
    assert.equal(w.bytes.byteLength, PARAMS.size);
    const p = decodeStruct(w.bytes, PARAMS);
    const { eye, forward, right, up } = cameraBasis(pt.camera);
    const f = Math.fround;
    assert.deepEqual(p.size, [600, 450]);
    assert.equal(p.sampleCount, 777);
    assert.equal(p.maxBounces, 8);
    assert.equal(p.primCount, pt.scene.count);
    assert.equal(p.lightCount, pt.scene.lightCount);
    assert.equal(p.envType, 1);
    assert.equal(p.useNEE, 1);
    assert.equal(p.seed, 0xdeadbeef);
    assert.deepEqual(p.camPos, [...eye, Math.tan(pt.camera.fov * Math.PI / 360)].map(f));
    assert.deepEqual(p.camForward, [...forward, pt.camera.aperture].map(f));
    assert.deepEqual(p.camRight, [...right, pt.camera.focus].map(f));
    assert.deepEqual(p.camUp, [...up, 20].map(f));
    assert.deepEqual(p.skyHorizon, [...pt.env.horizon, pt.env.intensity].map(f));
    assert.deepEqual(p.skyZenith, [...pt.env.zenith, 0].map(f));
    assert.deepEqual(p.skyGround, [...pt.env.ground, 0].map(f));
    t.diagnostic(`Params: ${Object.entries(PARAMS.offsets).map(([k, v]) => `${k}@${v}`).join(' ')}`);
});

test('writeDisplay bytes decode correctly through the WGSL Display struct', () => {
    const { pt, device } = tracerWithMock();
    pt.renderWidth = 400; pt.renderHeight = 300;
    pt.canvas = { width: 800, height: 600 };
    pt.toneMap = 1;
    pt.exposureEV = -1.5;
    pt.writeDisplay();
    const d = decodeStruct(device.lastWrite(pt.displayBuffer).bytes, DISPLAY);
    assert.deepEqual(d, { renderSize: [400, 300], canvasSize: [800, 600], toneMap: 1, exposure: Math.fround(2 ** -1.5) });
});

test('replica: PCG rand is uniform on [0, 1) and pcgHash decorrelates seeds', (t) => {
    const rand = makeRand(pcgHash(12345));
    const N = 200000, bins = new Array(16).fill(0);
    let sum = 0, min = 1, max = 0;
    for (let i = 0; i < N; i++) {
        const u = rand();
        sum += u; min = Math.min(min, u); max = Math.max(max, u);
        bins[Math.floor(u * 16)]++;
    }
    const chi2 = bins.reduce((s, b) => s + (b - N / 16) ** 2 / (N / 16), 0);
    t.diagnostic(`mean ${(sum / N).toFixed(5)}, range [${min.toFixed(6)}, ${max.toFixed(6)}], χ²(15 dof) ${chi2.toFixed(2)}`);
    assert.ok(min >= 0 && max < 1);
    assert.ok(Math.abs(sum / N - 0.5) < 0.005);
    assert.ok(chi2 < 40, 'χ² with 15 dof: p ≈ 5e-4 at 40');
    // Seeds of neighboring pixels give different streams
    const firsts = new Set(Array.from({ length: 1000 }, (_, i) => makeRand(pcgHash(i ^ pcgHash(9781)))()));
    assert.ok(firsts.size > 990);
});

test('replica: basisT/basisB/n form an orthonormal frame; cosineSample follows cos θ / π', (t) => {
    const r = mulberry(99);
    let maxErr = 0;
    const normals = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -1, 0]];
    for (let k = 0; k < 1000; k++) normals.push(randomUnit(r));
    for (const n of normals) {
        const T = basisT(n), B = basisB(n);
        maxErr = Math.max(maxErr, Math.abs(len(T) - 1), Math.abs(len(B) - 1), Math.abs(dot(T, B)), Math.abs(dot(T, n)), Math.abs(dot(B, n)));
    }
    t.diagnostic(`frame max error ${maxErr.toExponential(2)}`);
    assert.ok(maxErr < 1e-12);

    // Cosine distribution: P(θ < θ0) = sin² θ0, E[cos θ] = 2/3; azimuth uniform
    const n = normalize([0.3, -0.5, 0.8]);
    const rand = makeRand(4242);
    const N = 100000, thresholds = [0.2, 0.5, 0.8, 1.2, 1.5];
    const counts = thresholds.map(() => 0);
    let sumCos = 0, below = 0, quadrant = 0;
    const T = basisT(n);
    for (let i = 0; i < N; i++) {
        const w = cosineSample(n, rand(), rand());
        const c = dot(w, n);
        if (c < -1e-9) below++;
        sumCos += c;
        const theta = Math.acos(Math.min(1, c));
        thresholds.forEach((th, j) => { if (theta < th) counts[j]++; });
        if (dot(w, T) > 0) quadrant++;
    }
    let maxCdfErr = 0;
    thresholds.forEach((th, j) => { maxCdfErr = Math.max(maxCdfErr, Math.abs(counts[j] / N - Math.sin(th) ** 2)); });
    t.diagnostic(`E[cos θ] ${(sumCos / N).toFixed(5)} (2/3), max CDF error ${maxCdfErr.toFixed(5)}, half-space ${(quadrant / N).toFixed(4)}`);
    assert.equal(below, 0, 'samples stay in the hemisphere');
    assert.ok(Math.abs(sumCos / N - 2 / 3) < 0.004, '≈ 5 σ for N = 1e5');
    assert.ok(maxCdfErr < 0.008, 'binomial σ ≤ 0.0016 at N = 1e5');
    assert.ok(Math.abs(quadrant / N - 0.5) < 0.008);
});

test('replica: powerHeuristic weights of the two strategies sum to 1', (t) => {
    const r = mulberry(3);
    let maxErr = 0;
    for (let k = 0; k < 10000; k++) {
        const a = Math.exp(10 * (r() - 0.5)), b = Math.exp(10 * (r() - 0.5));
        maxErr = Math.max(maxErr, Math.abs(powerHeuristic(a, b) + powerHeuristic(b, a) - 1));
    }
    t.diagnostic(`max |w_light + w_bsdf − 1| ${maxErr.toExponential(2)}`);
    assert.ok(maxErr < 1e-14);
    assert.equal(powerHeuristic(0, 0), 0);
    assert.equal(powerHeuristic(1, 0), 1);
});

test('replica: sampleLight pdfs match the solid angle of quad and sphere lights', (t) => {
    const lightMat = { type: 0, albedo: [0, 0, 0], param: 0, emission: [1, 1, 1] };
    // Rectangle 2a × 2b, facing down (cross(u, v) = -y), centered above x at height d
    const a = 0.4, b = 0.25, d = 0.8;
    const scene = new SceneBuilder()
        .quad([-a, d, -b], [2 * a, 0, 0], [0, 0, 2 * b], lightMat)
        .sphere([2, 0.5, 0], 0.3, lightMat)
        .build();
    const prims = decodePrims(scene);
    const x = [0, 0, 0];
    const omegaRect = 4 * Math.asin(a * b / Math.sqrt((a * a + d * d) * (b * b + d * d)));
    const rand = makeRand(77);
    const N = 100000;
    // E[1/pdf] over light samples = solid angle; every sampled direction must hit the light and
    // lightPdf() (used for MIS of BSDF hits) must give back the same pdf
    let invSum = 0, maxPdfErr = 0;
    for (let i = 0; i < N; i++) {
        const ls = sampleLight(prims[0], x, rand(), rand());
        assert.ok(ls.pdf > 0);
        invSum += 1 / ls.pdf;
        const hit = intersectScene(scene, x, ls.wi);
        assert.equal(hit.index, 0);
        const back = lightPdf(prims[0], x, ls.wi, hit.t, hitNormal(prims[0], add(x, scale(ls.wi, hit.t))));
        maxPdfErr = Math.max(maxPdfErr, Math.abs(back - ls.pdf) / ls.pdf);
    }
    const omegaQuadMC = invSum / N;
    // Sphere: cone of half-angle asin(r/dist), solid angle 2π(1 − cos θmax)
    const r = 0.3, dist = Math.hypot(2, 0.5);
    const omegaCone = 2 * Math.PI * (1 - Math.sqrt(1 - r * r / (dist * dist)));
    let coneHits = 0, sphPdf = 0;
    for (let i = 0; i < 20000; i++) {
        const ls = sampleLight(prims[1], x, rand(), rand());
        sphPdf = ls.pdf;
        if (intersectScene(scene, x, ls.wi).index === 1) coneHits++;
    }
    // Independent check of the cone solid angle: uniform directions that hit the sphere
    const ur = mulberry(5);
    let uniformHits = 0;
    const M = 1000000;
    for (let i = 0; i < M; i++) if (intersectScene(scene, x, randomUnit(ur)).index === 1) uniformHits++;
    const omegaSphereMC = 4 * Math.PI * uniformHits / M;
    t.diagnostic(`quad Ω: MC ${omegaQuadMC.toFixed(5)} vs analytic ${omegaRect.toFixed(5)}, lightPdf round-trip max rel ${maxPdfErr.toExponential(2)}`);
    t.diagnostic(`sphere Ω: 1/pdf ${(1 / sphPdf).toFixed(5)} vs analytic ${omegaCone.toFixed(5)} vs uniform MC ${omegaSphereMC.toFixed(5)}; cone samples hitting ${coneHits}/20000`);
    assert.ok(Math.abs(omegaQuadMC - omegaRect) / omegaRect < 0.005);
    assert.ok(maxPdfErr < 1e-9);
    assert.ok(Math.abs(1 / sphPdf - omegaCone) / omegaCone < 1e-6, 'PI constant is 3.14159265 (f32 precision)');
    assert.ok(coneHits >= 19990, 'cone samples hit the sphere (a few graze the silhouette)');
    const sigma = Math.sqrt(uniformHits) / uniformHits;   // relative binomial σ (≈ 1.4% here)
    assert.ok(Math.abs(omegaSphereMC - omegaCone) / omegaCone < 4 * sigma, `uniform MC off by > 4σ (σ = ${sigma})`);
});

// Closed box of six inward-facing diffuse quads, each with emission Le and albedo ρ. Every path
// vertex sees emission Le, so the expected radiance with B bounces is Le · Σ_{k=0..B} ρ^k.
function furnaceBox(rho, le) {
    const m = { type: 0, albedo: [rho, rho, rho], param: 0, emission: [le, le, le] };
    // Unit cube [0,1]³; cross(u, v) must point into the box
    return new SceneBuilder()
        .quad([0, 0, 0], [0, 0, 1], [1, 0, 0], m)   // floor y=0: (0,0,1)×(1,0,0) = +y
        .quad([0, 1, 0], [1, 0, 0], [0, 0, 1], m)   // ceiling: -y
        .quad([0, 0, 0], [1, 0, 0], [0, 1, 0], m)   // z=0: +z
        .quad([0, 0, 1], [0, 1, 0], [1, 0, 0], m)   // z=1: -z
        .quad([0, 0, 0], [0, 1, 0], [0, 0, 1], m)   // x=0: +x
        .quad([1, 0, 0], [0, 0, 1], [0, 1, 0], m)   // x=1: -x
        .build();
}

test('replica: furnace box — tracePath (BSDF only, and NEE + MIS) is unbiased', (t) => {
    const rho = 0.5, le = 1, B = 5;
    const expected = le * (1 - rho ** (B + 1)) / (1 - rho);
    const scene = furnaceBox(rho, le);
    const prims = decodePrims(scene);
    assert.equal(scene.lightCount, 6);
    prims.forEach(p => assert.ok(dot(hitNormal(p, [0, 0, 0]), sub([0.5, 0.5, 0.5], p.geo0.slice(0, 3))) > 0, 'faces inward'));
    for (const useNEE of [false, true]) {
        const rand = makeRand(pcgHash(useNEE ? 2 : 1));
        const r = mulberry(useNEE ? 11 : 10);
        const N = 12000;
        let sum = 0, sum2 = 0;
        for (let i = 0; i < N; i++) {
            const o = [0.2 + 0.6 * r(), 0.2 + 0.6 * r(), 0.2 + 0.6 * r()];
            const L = tracePath(scene, prims, o, randomUnit(r), rand, { bounces: B, useNEE });
            sum += L; sum2 += L * L;
        }
        const mean = sum / N, se = Math.sqrt((sum2 / N - mean * mean) / N);
        t.diagnostic(`NEE ${useNEE ? 'on ' : 'off'}: L = ${mean.toFixed(4)} ± ${se.toFixed(4)} (expected ${expected.toFixed(4)}, ${((mean - expected) / se).toFixed(2)} σ)`);
        assert.ok(Math.abs(mean - expected) < 4 * se + 1e-3, `NEE ${useNEE}: ${mean} vs ${expected}`);
        assert.ok(Math.abs(mean - expected) / expected < 0.02);
    }
});

test('replica: white furnace — a diffuse sphere under a uniform environment returns ρ·L every sample', (t) => {
    const rho = 0.8, env = 1.5;
    const scene = new SceneBuilder().sphere([0, 0, 0], 1, { type: 0, albedo: [rho, rho, rho], param: 0, emission: [0, 0, 0] }).build();
    const prims = decodePrims(scene);
    const rand = makeRand(5), r = mulberry(5);
    let maxErr = 0;
    for (let i = 0; i < 2000; i++) {
        const o = scale(randomUnit(r), 3);
        const target = scale(randomUnit(r), 0.9 * r());
        const L = tracePath(scene, prims, o, normalize(sub(target, o)), rand, { bounces: 5, useNEE: true, env });
        maxErr = Math.max(maxErr, Math.abs(L - prims[0].albedo[0] * env));   // f32 albedo
    }
    // Convex sphere: one bounce then escape; the cosine pdf cancels f·cos exactly.
    t.diagnostic(`max |L − ρ·L_env| ${maxErr.toExponential(2)}`);
    assert.ok(maxErr < 1e-12);
});
