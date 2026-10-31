// Boids flocking (example #6).
// CPU replica of the `compute` shader (O(N²) separation/alignment/cohesion, behavior modes, wrap);
// validates the algorithm and index math, not the WGSL itself — see #51.
//
// Shipped code exercised directly: BoidsSimulation constructed on a mock device (buffer sizes) and its
// updateParams()/updateUniforms() writers, decoded with the WGSL `Params` / `Uniforms` structs.
// The replica is checked against invariants (speed limit, bounds), against hand-computed single-pair
// forces, and against emergent behavior (polarization rises from a random start).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoidsSimulation, wgslModules } from '../js/examples/boids.js';
import { mockDevice, stub } from './helpers/mock-gpu.mjs';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';

const mod = { wgslModules };
const f = Math.fround;

function makeSim(n = 100) {
    globalThis.window ??= { addEventListener() {} };
    const device = mockDevice();
    const canvas = { width: 800, height: 600, getContext: () => stub('ctx'), addEventListener() {} };
    return { device, sim: new BoidsSimulation(device, canvas, n) };
}

test('updateParams bytes match WGSL Params (u32 numBoids/behaviorMode, vec2 at 8-byte offsets)', (t) => {
    const layout = structLayout(moduleCode(mod, 'compute'), 'Params');
    t.diagnostic(`Params offsets ${JSON.stringify(layout.offsets)} size=${layout.size}`);
    const { device, sim } = makeSim(1234);
    Object.assign(sim, {
        separationRadius: 21, alignmentRadius: 43, cohesionRadius: 57, separationWeight: 1.25,
        alignmentWeight: 0.75, cohesionWeight: 0.5, maxSpeed: 3.5, maxForce: 0.125,
        mousePos: [111, 222], mouseRadius: 150, mouseForce: -1, behaviorMode: 3,
    });
    sim.updateParams();
    const w = device.lastWrite(sim.paramsBuffer);
    assert.ok(sim.paramsBuffer.size >= layout.size, `params buffer ${sim.paramsBuffer.size} >= ${layout.size}`);
    assert.ok(w.bytes.byteLength <= sim.paramsBuffer.size, 'write fits in the buffer');
    assert.deepEqual(decodeStruct(w.bytes, layout), {
        numBoids: 1234, separationRadius: 21, alignmentRadius: 43, cohesionRadius: 57,
        separationWeight: 1.25, alignmentWeight: 0.75, cohesionWeight: 0.5, maxSpeed: 3.5, maxForce: 0.125,
        boundsMin: [0, 0], boundsMax: [800, 600], mousePos: [111, 222], mouseRadius: 150, mouseForce: -1,
        behaviorMode: 3,
    });
});

test('updateUniforms bytes match WGSL Uniforms; boid buffers hold N vec2+vec2', () => {
    const layout = structLayout(moduleCode(mod, 'render'), 'Uniforms');
    const boid = structLayout(moduleCode(mod, 'compute'), 'Boid');
    const { device, sim } = makeSim(77);
    sim.updateUniforms();
    const w = device.lastWrite(sim.uniformBuffer);
    assert.ok(sim.uniformBuffer.size >= layout.size);
    assert.deepEqual(decodeStruct(w.bytes, layout), { canvasSize: [800, 600] });
    assert.equal(boid.size, 16);
    assert.equal(sim.boidBuffers[0].size, 77 * boid.size);
    // initializeBoids keeps positions on the canvas and velocity components within ±maxSpeed/2.
    for (let i = 0; i < 77; i++) {
        const [x, y, vx, vy] = sim.boids.subarray(i * 4, i * 4 + 4);
        assert.ok(x >= 0 && x <= 800 && y >= 0 && y <= 600 && Math.abs(vx) <= 2 && Math.abs(vy) <= 2);
    }
});

// ---- Replica of the compute shader (f32) ------------------------------------------------------
const length2 = (x, y) => f(Math.sqrt(f(f(x * x) + f(y * y))));
function normalize2(x, y) { const l = length2(x, y); return [f(x / l), f(y / l)]; }
function limit(v, maxLen) {
    const l = length2(v[0], v[1]);
    if (l > maxLen) { const n = normalize2(v[0], v[1]); return [f(n[0] * maxLen), f(n[1] * maxLen)]; }
    return v;
}
function replicaStep(boidsIn, p) {
    const n = p.numBoids;
    const out = new Float32Array(boidsIn.length);
    const B = i => boidsIn.subarray(i * 4, i * 4 + 4);
    for (let idx = 0; idx < n; idx++) {
        const [px, py, vx, vy] = B(idx);
        // separation
        let sx = 0, sy = 0, sc = 0, ax = 0, ay = 0, ac = 0, cx = 0, cy = 0, cc = 0;
        for (let i = 0; i < n; i++) {
            const [ox, oy, ovx, ovy] = B(i);
            const dx = f(px - ox), dy = f(py - oy);
            const dist = length2(dx, dy);
            if (dist > 0 && dist < p.separationRadius) {
                const nn = normalize2(dx, dy);
                sx = f(sx + f(nn[0] / dist)); sy = f(sy + f(nn[1] / dist)); sc++;
            }
            if (dist > 0 && dist < p.alignmentRadius) { ax = f(ax + ovx); ay = f(ay + ovy); ac++; }
            if (dist > 0 && dist < p.cohesionRadius) { cx = f(cx + ox); cy = f(cy + oy); cc++; }
        }
        let sep = [0, 0], ali = [0, 0], coh = [0, 0];
        if (sc > 0) {
            sx = f(sx / sc); sy = f(sy / sc);
            if (length2(sx, sy) > 0) {
                const nn = normalize2(sx, sy);
                sep = limit([f(f(nn[0] * p.maxSpeed) - vx), f(f(nn[1] * p.maxSpeed) - vy)], p.maxForce);
            }
        }
        if (ac > 0) {
            const nn = normalize2(f(ax / ac), f(ay / ac));
            ali = limit([f(f(nn[0] * p.maxSpeed) - vx), f(f(nn[1] * p.maxSpeed) - vy)], p.maxForce);
        }
        if (cc > 0) {
            const nn = normalize2(f(f(cx / cc) - px), f(f(cy / cc) - py));
            coh = limit([f(f(nn[0] * p.maxSpeed) - vx), f(f(nn[1] * p.maxSpeed) - vy)], p.maxForce);
        }
        let mouse = [0, 0];
        const mdx = f(p.mousePos[0] - px), mdy = f(p.mousePos[1] - py), md = length2(mdx, mdy);
        if (md < p.mouseRadius && md > 0) {
            const nn = normalize2(mdx, mdy), k = f(p.mouseForce * f(1 - f(md / p.mouseRadius)));
            mouse = [f(nn[0] * k), f(nn[1] * k)];
        }
        let acc = [
            f(f(f(f(sep[0] * p.separationWeight) + f(ali[0] * p.alignmentWeight)) + f(coh[0] * p.cohesionWeight)) + mouse[0]),
            f(f(f(f(sep[1] * p.separationWeight) + f(ali[1] * p.alignmentWeight)) + f(coh[1] * p.cohesionWeight)) + mouse[1]),
        ];
        const center = [f(f(p.boundsMin[0] + p.boundsMax[0]) * 0.5), f(f(p.boundsMin[1] + p.boundsMax[1]) * 0.5)];
        if (p.behaviorMode === 2) {
            const away = normalize2(f(px - center[0]), f(py - center[1]));
            acc = [f(f(acc[0] * 0.1) + f(away[0] * f(p.maxForce * 2))), f(f(acc[1] * 0.1) + f(away[1] * f(p.maxForce * 2)))];
        }
        let v = limit([f(vx + acc[0]), f(vy + acc[1])], p.maxSpeed);
        let x = f(px + v[0]), y = f(py + v[1]);
        if (x < p.boundsMin[0]) x = p.boundsMax[0];
        if (x > p.boundsMax[0]) x = p.boundsMin[0];
        if (y < p.boundsMin[1]) y = p.boundsMax[1];
        if (y > p.boundsMax[1]) y = p.boundsMin[1];
        out.set([x, y, v[0], v[1]], idx * 4);
    }
    return out;
}

const defaults = (n, W, H) => ({
    numBoids: n, separationRadius: 25, alignmentRadius: 50, cohesionRadius: 50, separationWeight: 1.5,
    alignmentWeight: 1, cohesionWeight: 1, maxSpeed: 4, maxForce: 0.1, boundsMin: [0, 0], boundsMax: [W, H],
    mousePos: [W / 2, H / 2], mouseRadius: 150, mouseForce: 0, behaviorMode: 0,
});

test('single pair: separation pushes apart, cohesion pulls together (hand-computed)', (t) => {
    // Two boids 10 px apart on x, at rest. Separation radius 25 (inside) — only separation active.
    const p = { ...defaults(2, 800, 600), alignmentRadius: 0, cohesionRadius: 0 };
    const b = new Float32Array([400, 300, 0, 0, 410, 300, 0, 0]);
    const out = replicaStep(b, p);
    // Expected: steering = normalize(diff)*maxSpeed - v = (∓4, 0), limited to maxForce 0.1, ×1.5.
    t.diagnostic(`separation step velocities: ${out[2]}, ${out[6]}`);
    assert.ok(Math.abs(out[2] - -0.15) < 1e-6 && Math.abs(out[6] - 0.15) < 1e-6);
    assert.equal(out[3], 0);
    // Cohesion only (separation off), 40 px apart: they accelerate toward each other by maxForce.
    const q = { ...defaults(2, 800, 600), separationRadius: 0, alignmentRadius: 0 };
    const c = replicaStep(new Float32Array([380, 300, 0, 0, 420, 300, 0, 0]), q);
    assert.ok(Math.abs(c[2] - 0.1) < 1e-6 && Math.abs(c[6] + 0.1) < 1e-6);
});

test('flock invariants and emergent alignment (fixed seed, 150 boids, 200 steps)', (t) => {
    const n = 150, W = 400, H = 300;
    let s = 2024;
    const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
    let b = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) b.set([rand() * W, rand() * H, (rand() - 0.5) * 4, (rand() - 0.5) * 4], i * 4);
    const p = defaults(n, W, H);
    const polarization = arr => {
        let mx = 0, my = 0, ms = 0;
        for (let i = 0; i < n; i++) { mx += arr[i * 4 + 2]; my += arr[i * 4 + 3]; ms += Math.hypot(arr[i * 4 + 2], arr[i * 4 + 3]); }
        return Math.hypot(mx, my) / ms;
    };
    const p0 = polarization(b);
    let maxSpeed = 0, outOfBounds = 0;
    for (let step = 0; step < 200; step++) {
        b = replicaStep(b, p);
        for (let i = 0; i < n; i++) {
            maxSpeed = Math.max(maxSpeed, Math.hypot(b[i * 4 + 2], b[i * 4 + 3]));
            if (b[i * 4] < 0 || b[i * 4] > W || b[i * 4 + 1] < 0 || b[i * 4 + 1] > H) outOfBounds++;
        }
    }
    const p1 = polarization(b);
    t.diagnostic(`polarization ${p0.toFixed(3)} -> ${p1.toFixed(3)}; max speed ${maxSpeed.toFixed(6)} (limit 4); out-of-bounds ${outOfBounds}`);
    assert.ok(maxSpeed <= 4 * (1 + 1e-6), 'limit() caps speed at maxSpeed');
    assert.equal(outOfBounds, 0, 'wrap keeps every boid inside bounds');
    assert.ok(p0 < 0.2, 'random start is disordered');
    assert.ok(p1 > 0.5, 'alignment + cohesion order the flock');
});

test('scatter mode drives boids away from the center', (t) => {
    const n = 60, W = 400, H = 300;
    let s = 7;
    const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
    let b = new Float32Array(n * 4);
    // Non-zero velocities: alignment() normalizes the neighbors' mean velocity with no zero check, so
    // an all-at-rest flock would produce NaN (same as normalize(vec2(0)) on the GPU).
    for (let i = 0; i < n; i++) b.set([150 + rand() * 100, 100 + rand() * 100, rand() - 0.5, rand() - 0.5], i * 4);
    const meanDist = arr => { let d = 0; for (let i = 0; i < n; i++) d += Math.hypot(arr[i * 4] - 200, arr[i * 4 + 1] - 150); return d / n; };
    const d0 = meanDist(b);
    const p = { ...defaults(n, W, H), behaviorMode: 2 };
    for (let i = 0; i < 20; i++) b = replicaStep(b, p);
    const d1 = meanDist(b);
    t.diagnostic(`mean distance from center ${d0.toFixed(1)} -> ${d1.toFixed(1)} after 20 scatter steps`);
    assert.ok(d1 > d0 + 10);
});
