// SPH fluid (example 1): CPU tests.
//
// CPU replica of sph-fluid.js COMPUTE_DENSITY_SHADER, COMPUTE_FORCES_SHADER and INTEGRATE_SHADER;
// validates the algorithm and index math, not the WGSL itself — see #51. The replica follows the
// shaders: Poly6 density (self included) and clamped pressure k·max(ρ − ρ0, 0); Spiky-gradient
// pressure force with the averaged pressure, viscosity Laplacian, gravity as a force density;
// explicit Euler integration with damped wall reflections. Per-pass results are stored as f32.
//
// The shaders use O(N²) brute-force neighbour loops. For speed the replica gathers neighbours from
// a uniform grid of cell size h (identical cutoff); a test checks it against the brute-force loop.
//
// Real shipped code exercised: SPHFluidSimulation constructed on a mock device (its parameters,
// latticeDensity()-derived mass, and initializeParticles() dam-break), updateParams() and
// updateUniforms() bytes against the WGSL structs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mockDevice, stub } from './helpers/mock-gpu.mjs';
import { structLayout, decodeStruct, moduleCode, parseStructs, typeLayout } from './helpers/wgsl-layout.mjs';
import * as sph from '../js/examples/sph-fluid.js';

const { SPHFluidSimulation } = sph;
const PI = 3.14159265;   // the shaders' constant

function makeSim(n) {
    const device = mockDevice();
    const canvas = { width: 1000, height: 700, getContext: () => stub('ctx') };   // index.html canvas-sph
    return { sim: new SPHFluidSimulation(device, canvas, n), device };
}

// Shader kernels (2D normalizations)
const poly6 = (r, h) => r >= h ? 0 : (4 / (PI * Math.pow(h, 8))) * (h * h - r * r) ** 3;
const spikyGradMag = (r, h) => (r >= h || r < 0.0001) ? 0 : (-30 / (PI * Math.pow(h, 5))) * (h - r) ** 2;
const viscLap = (r, h) => r >= h ? 0 : (40 / (PI * Math.pow(h, 5))) * (h - r);

// State in SoA form (independent of the GPU byte layout)
function stateFromSim(sim) {
    const n = sim.numParticles, p = sim.particles;
    const s = { n, x: new Float32Array(n), y: new Float32Array(n), vx: new Float32Array(n), vy: new Float32Array(n),
        rho: new Float32Array(n), pr: new Float32Array(n) };
    for (let i = 0; i < n; i++) {
        s.x[i] = p[i * 8]; s.y[i] = p[i * 8 + 1]; s.vx[i] = p[i * 8 + 2]; s.vy[i] = p[i * 8 + 3];
        s.rho[i] = p[i * 8 + 4]; s.pr[i] = p[i * 8 + 5];
    }
    return s;
}
const cloneState = s => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, typeof v === 'number' ? v : Float32Array.from(v)]));

// Neighbour lists: grid (cell size h, 3x3 cells) or brute force; both include j = i.
function neighboursGrid(s, h) {
    const cells = new Map();
    const key = (cx, cy) => cx * 65536 + cy;
    for (let i = 0; i < s.n; i++) {
        const k = key(Math.floor(s.x[i] / h), Math.floor(s.y[i] / h));
        if (!cells.has(k)) cells.set(k, []);
        cells.get(k).push(i);
    }
    const out = new Array(s.n);
    for (let i = 0; i < s.n; i++) {
        const cx = Math.floor(s.x[i] / h), cy = Math.floor(s.y[i] / h);
        const list = [];
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
            const c = cells.get(key(cx + dx, cy + dy));
            if (c) for (const j of c) if (Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]) < h) list.push(j);
        }
        out[i] = list.sort((a, b) => a - b);   // shader loop order: ascending j
    }
    return out;
}
function neighboursBrute(s, h) {
    const out = new Array(s.n);
    for (let i = 0; i < s.n; i++) {
        const list = [];
        for (let j = 0; j < s.n; j++) if (Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]) < h) list.push(j);
        out[i] = list;
    }
    return out;
}

// One substep: density pass, forces pass, integrate pass.
function substep(s, P, neighbours = neighboursGrid) {
    const h = P.smoothingRadius, n = s.n;
    const nb = neighbours(s, h);
    for (let i = 0; i < n; i++) {
        let d = 0;
        for (const j of nb[i]) d += P.mass * poly6(Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]), h);
        s.rho[i] = d;
        s.pr[i] = Math.max(P.gasConstant * (s.rho[i] - P.restDensity), 0);
    }
    const fx = new Float32Array(n), fy = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        let px = 0, py = 0, vx = 0, vy = 0;
        for (const j of nb[i]) {
            if (j === i) continue;
            const dx = s.x[i] - s.x[j], dy = s.y[i] - s.y[j];
            const dist = Math.hypot(dx, dy);
            if (dist < h && dist > 0.0001) {
                const g = spikyGradMag(dist, h);
                const avgP = (s.pr[i] + s.pr[j]) / 2;
                px -= P.mass * avgP / s.rho[j] * g * dx / dist;
                py -= P.mass * avgP / s.rho[j] * g * dy / dist;
                const lap = viscLap(dist, h);
                vx += P.mass * (s.vx[j] - s.vx[i]) / s.rho[j] * lap;
                vy += P.mass * (s.vy[j] - s.vy[i]) / s.rho[j] * lap;
            }
        }
        fx[i] = px + P.viscosity * vx;
        fy[i] = py + P.viscosity * vy + P.gravity * s.rho[i];
    }
    const dt = P.deltaTime;
    for (let i = 0; i < n; i++) {
        s.vx[i] += fx[i] / s.rho[i] * dt;
        s.vy[i] += fy[i] / s.rho[i] * dt;
        s.x[i] += s.vx[i] * dt;
        s.y[i] += s.vy[i] * dt;
        if (s.x[i] < P.boundsMin[0]) { s.x[i] = P.boundsMin[0]; s.vx[i] *= -P.damping; }
        if (s.x[i] > P.boundsMax[0]) { s.x[i] = P.boundsMax[0]; s.vx[i] *= -P.damping; }
        if (s.y[i] < P.boundsMin[1]) { s.y[i] = P.boundsMin[1]; s.vy[i] *= -P.damping; }
        if (s.y[i] > P.boundsMax[1]) { s.y[i] = P.boundsMax[1]; s.vy[i] *= -P.damping; }
    }
}

const params = sim => ({
    smoothingRadius: sim.smoothingRadius, restDensity: sim.restDensity, gasConstant: sim.gasConstant,
    mass: sim.mass, viscosity: sim.viscosity, gravity: sim.gravity, deltaTime: sim.deltaTime,
    boundsMin: sim.boundsMin, boundsMax: sim.boundsMax, damping: sim.damping,
});

function stats(s) {
    let ke = 0, vmax = 0, finite = true;
    for (let i = 0; i < s.n; i++) {
        const v2 = s.vx[i] ** 2 + s.vy[i] ** 2;
        if (!Number.isFinite(v2) || !Number.isFinite(s.x[i]) || !Number.isFinite(s.y[i])) finite = false;
        ke += 0.5 * v2;
        vmax = Math.max(vmax, Math.sqrt(v2));
    }
    return { ke, vmax, finite };
}

test('Params and Uniforms layouts: updateParams()/updateUniforms() bytes decode to the WGSL structs', t => {
    const { sim, device } = makeSim(300);
    Object.assign(sim, { numParticles: 1234, smoothingRadius: 16.5, restDensity: 1.25, gasConstant: 2e6, mass: 0.0625,
        viscosity: 310, gravity: 520, deltaTime: 0.0025, boundsMin: [11, 12], boundsMax: [987, 654], damping: 0.375 });
    sim.updateParams();
    const integrate = structLayout(moduleCode(sph, 'integrate'), 'Params');
    const bytes = device.lastWrite(sim.paramsBuffer).bytes;
    const p = decodeStruct(bytes, integrate);
    assert.deepEqual(p, { numParticles: 1234, smoothingRadius: 16.5, restDensity: 1.25, gasConstant: 2e6, mass: 0.0625,
        viscosity: 310, gravity: 520, deltaTime: Math.fround(0.0025), boundsMin: [11, 12], boundsMax: [987, 654], damping: 0.375 });
    assert.ok(sim.paramsBuffer.size >= integrate.size);
    // density/forces declare a prefix of the same struct; offsets must agree
    for (const name of ['density', 'forces']) {
        const l = structLayout(moduleCode(sph, name), 'Params');
        for (const f of l.fields) assert.equal(f.offset, integrate.offsets[f.name], `${name}.${f.name}`);
    }
    t.diagnostic(`Params ${integrate.size} bytes; offsets ${JSON.stringify(integrate.offsets)}`);

    sim.updateUniforms();
    const u = decodeStruct(device.lastWrite(sim.uniformBuffer).bytes, structLayout(moduleCode(sph, 'render'), 'Uniforms'));
    assert.deepEqual(u, { canvasSize: [1000, 700], boundsMin: [11, 12], boundsMax: [987, 654] });
});

// The JS packs 8 floats (32 bytes) per particle — "pos(2) + vel(2) + density(1) + pressure(1) +
// padding(2)" — but the WGSL Particle struct has no padding member, so array<Particle> has a
// 24-byte stride. The GPU therefore reads the initial upload and spawned blocks misaligned.
// This suite found a 24-byte WGSL stride against 32-byte JS packing; fixed in #63 by padding Particle.
test('Particle stride: JS packing matches array<Particle> stride in every module', t => {
    const jsStride = 8 * 4;
    for (const { name, code } of sph.wgslModules()) {
        const structs = parseStructs(code);
        if (!structs.has('Particle')) continue;
        const stride = typeLayout('array<Particle>', structs).stride;
        t.diagnostic(`${name}: array<Particle> stride ${stride} bytes, JS stride ${jsStride}`);
        assert.equal(stride, jsStride, `${name} Particle stride`);
    }
});

test('Kernels: Poly6 and Spiky integrate to 1 over their 2D support; Spiky gradient is dW/dr', t => {
    const h = 16;
    const M = 20000;
    let poly = 0, spiky = 0;
    // W_spiky(r) rebuilt from the shader's gradient: W(r) = ∫_r^h |∇W| ds (trapezoid, fine grid)
    const dr = h / M;
    const W = new Float64Array(M + 1);
    for (let k = M - 1; k >= 0; k--) {
        const a = Math.max(k * dr, 0.0001), b = (k + 1) * dr;
        W[k] = W[k + 1] - 0.5 * (spikyGradMag(a, h) + spikyGradMag(b, h)) * dr;
    }
    for (let k = 0; k < M; k++) {
        const r = (k + 0.5) * dr;
        poly += poly6(r, h) * 2 * Math.PI * r * dr;
        spiky += 0.5 * (W[k] + W[k + 1]) * 2 * Math.PI * r * dr;
    }
    let visc = 0;
    for (let k = 0; k < M; k++) { const r = (k + 0.5) * dr; visc += viscLap(r, h) * 2 * Math.PI * r * dr; }
    t.diagnostic(`∫Poly6 = ${poly.toFixed(7)}, ∫Spiky (from shader gradient) = ${spiky.toFixed(7)}, ∫∇²W_visc = ${visc.toExponential(4)} (= 40/(3h²) = ${(40 / (3 * h * h)).toExponential(4)})`);
    // The shader uses π = 3.14159265 (f32), so normalization is exact to ~1e-8 plus quadrature error.
    assert.ok(Math.abs(poly - 1) < 1e-6);
    assert.ok(Math.abs(spiky - 1) < 1e-4);
    assert.ok(Math.abs(visc - 40 / (3 * h * h)) / visc < 1e-6);
    assert.equal(poly6(h, h), 0); assert.equal(spikyGradMag(h, h), 0); assert.equal(viscLap(h, h), 0);
});

test('Initial dam-break lattice sits at rest density (shipped latticeDensity() mass)', t => {
    const { sim } = makeSim(400);
    const s = stateFromSim(sim);
    const P = params(sim);
    const nb = neighboursBrute(s, P.smoothingRadius);
    // interior particles: a full neighbourhood (same count as the lattice centre)
    const full = Math.max(...nb.map(l => l.length));
    let maxErr = 0, count = 0;
    for (let i = 0; i < s.n; i++) {
        if (nb[i].length !== full) continue;
        let d = 0;
        for (const j of nb[i]) d += P.mass * poly6(Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]), P.smoothingRadius);
        maxErr = Math.max(maxErr, Math.abs(d - P.restDensity));
        count++;
    }
    t.diagnostic(`${count} interior particles (${full} neighbours): max |ρ − ρ0| = ${maxErr.toExponential(2)}`);
    assert.ok(count > 100);
    assert.ok(maxErr < 1e-6);
});

test('Grid neighbour gathering matches the shader\'s brute-force loop', t => {
    const { sim } = makeSim(400);
    const P = params(sim);
    const a = stateFromSim(sim), b = cloneState(a);
    for (let k = 0; k < 30; k++) { substep(a, P, neighboursGrid); substep(b, P, neighboursBrute); }
    let maxDiff = 0;
    for (let i = 0; i < a.n; i++) maxDiff = Math.max(maxDiff, Math.abs(a.x[i] - b.x[i]), Math.abs(a.y[i] - b.y[i]));
    t.diagnostic(`max position difference after 30 substeps: ${maxDiff.toExponential(2)} px`);
    assert.equal(maxDiff, 0, 'same neighbours in the same order give bit-identical results');
});

test('Stability across slider extremes: finite, in bounds, bounded speed', t => {
    const { sim } = makeSim(400);
    const base = params(sim);
    const H = base.boundsMax[1] - base.boundsMin[1];
    // index.html: gravity slider 0..100 (×10), viscosity slider 1..100 (×10)
    for (const gravity of [0, 500, 1000]) {
        for (const viscosity of [10, 1000]) {
            const P = { ...base, gravity, viscosity };
            const s = stateFromSim(sim);
            let vmax = 0, finite = true;
            for (let frame = 0; frame < 150; frame++) {
                for (let k = 0; k < sim.substeps; k++) substep(s, P);
                const st = stats(s);
                vmax = Math.max(vmax, st.vmax);
                finite &&= st.finite;
            }
            let inBounds = true;
            for (let i = 0; i < s.n; i++) {
                if (s.x[i] < P.boundsMin[0] || s.x[i] > P.boundsMax[0] || s.y[i] < P.boundsMin[1] || s.y[i] > P.boundsMax[1]) inBounds = false;
            }
            // Free fall over the whole box gives sqrt(2gH); splash jets measured up to 2x; 3x flags a blow-up.
            const limit = 3 * Math.sqrt(2 * Math.max(gravity, 100) * H);
            t.diagnostic(`g=${gravity} visc=${viscosity}: max speed ${vmax.toFixed(1)} px/s over 2.5 s (limit ${limit.toFixed(0)})`);
            assert.ok(finite, 'no NaN/Inf');
            assert.ok(inBounds);
            assert.ok(vmax < limit, `speed ${vmax}`);
        }
    }
});

test('Dam break settles: kinetic energy decays after the splash', t => {
    const { sim } = makeSim(400);
    const P = params(sim);
    const s = stateFromSim(sim);
    const ke = [];
    for (let frame = 0; frame < 480; frame++) {   // 8 s at 60 fps
        for (let k = 0; k < sim.substeps; k++) substep(s, P);
        ke.push(stats(s).ke);
    }
    const peak = Math.max(...ke);
    const peakFrame = ke.indexOf(peak);
    const tail = ke.slice(-60).reduce((a, b) => a + b, 0) / 60;
    let maxY = -Infinity, minY = Infinity;
    for (let i = 0; i < s.n; i++) { maxY = Math.max(maxY, s.y[i]); minY = Math.min(minY, s.y[i]); }
    t.diagnostic(`peak KE ${peak.toExponential(3)} at frame ${peakFrame}; mean KE over last second ${tail.toExponential(3)} (${(100 * tail / peak).toFixed(2)}% of peak); fluid y range ${minY.toFixed(1)}..${maxY.toFixed(1)}`);
    // Viscosity and wall damping must dissipate the splash: last-second KE below 5% of the peak.
    assert.ok(tail < 0.05 * peak);
    // and the fluid has fallen to the floor (boundsMax.y, screen y down)
    assert.ok(maxY > P.boundsMax[1] - 1);
});
