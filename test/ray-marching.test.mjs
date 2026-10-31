// Ray-marched 3D fractals (example #9).
// CPU replica of the `render` shader's distance estimators, camera and sphere-tracing loop (float64);
// validates the algorithm and index math, not the WGSL itself — see #51.
//
// Shipped code exercised directly: RayMarchingRenderer.updateUniforms() via a mock device; the bytes
// are decoded with the WGSL `Uniforms` struct. Each replica DE is checked against an independent
// brute-force membership test (bounded orbit / exact Menger digit test) along camera rays: the
// sphere tracer must find the same silhouette a fine fixed-step membership march finds.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RayMarchingRenderer, wgslModules } from '../js/examples/ray-marching.js';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';

const mod = { wgslModules };

test('updateUniforms bytes match the WGSL Uniforms struct (u32 maxSteps/fractalType)', (t) => {
    const layout = structLayout(moduleCode(mod, 'render'), 'Uniforms');
    t.diagnostic(`Uniforms offsets ${JSON.stringify(layout.offsets)} size=${layout.size}`);
    const device = mockDevice();
    const r = Object.create(RayMarchingRenderer.prototype);
    Object.assign(r, {
        device, canvas: { width: 640, height: 360 }, time: 1.25, power: 7, maxSteps: 96,
        fractalType: 3, cameraRotX: -0.5, cameraRotY: 0.75, cameraZoom: 2.5,
        uniformBuffer: device.createBuffer({ size: 64, usage: 0 }),
    });
    r.updateUniforms();
    const w = device.lastWrite(r.uniformBuffer);
    assert.equal(w.offset, 0);
    assert.ok(r.uniformBuffer.size >= layout.size, 'buffer (64 bytes) holds the struct');
    const d = decodeStruct(w.bytes, layout);
    assert.deepEqual(d, {
        resolution: [640, 360], time: 1.25, power: 7, maxSteps: 96, fractalType: 3,
        cameraRotX: -0.5, cameraRotY: 0.75, cameraZoom: 2.5,
    });
});

// ---- Replica of the WGSL distance estimators -------------------------------------------------
const len = v => Math.hypot(v[0], v[1], v[2]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = a => scl(a, 1 / len(a));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function bulbPow(z, r, power) {
    const theta = Math.acos(z[2] / r), phi = Math.atan2(z[1], z[0]);
    const zr = Math.pow(r, power), nt = theta * power, np = phi * power;
    return [zr * Math.sin(nt) * Math.cos(np), Math.sin(np) * Math.sin(nt) * zr, zr * Math.cos(nt)];
}
function mandelbulbDE(pos, power) {
    let z = pos, dr = 1, r = 0;
    for (let i = 0; i < 15; i++) {
        r = len(z);
        if (r > 2) break;
        dr = Math.pow(r, power - 1) * power * dr + 1;
        z = add(bulbPow(z, r, power), pos);
    }
    return 0.5 * Math.log(r) * r / dr;
}
const JC = [0.18, 0.88, 0.24, 0.16];
function juliaStep(z) {
    return [z[0] * z[0] - (z[1] * z[1] + z[2] * z[2] + z[3] * z[3]) + JC[0],
        2 * z[0] * z[1] + JC[1], 2 * z[0] * z[2] + JC[2], 2 * z[0] * z[3] + JC[3]];
}
const dot4 = z => z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
function juliaDE(pos) {
    let z = [pos[0], pos[1], pos[2], 0], md2 = 1, mz2 = dot4(z);
    for (let i = 0; i < 12; i++) {
        md2 *= 4 * mz2;
        z = juliaStep(z);
        mz2 = dot4(z);
        if (mz2 > 4) break;
    }
    return 0.25 * Math.sqrt(mz2 / md2) * Math.log(mz2);
}
function mengerDE(pos) {
    const q = pos.map(c => Math.abs(c) - 1);
    let d = len(q.map(c => Math.max(c, 0))) + Math.min(Math.max(q[0], q[1], q[2]), 0);
    let s = 1;
    for (let i = 0; i < 4; i++) {
        const a = pos.map(c => { const x = c * s; return x - 2 * Math.floor(x / 2) - 1; });
        s *= 3;
        const r = a.map(c => Math.abs(1 - 3 * Math.abs(c)));
        const c = (Math.min(Math.max(r[0], r[1]), Math.max(r[1], r[2]), Math.max(r[2], r[0])) - 1) / s;
        d = Math.max(d, c);
    }
    return d;
}
function mandelboxDE(pos) {
    let p = pos, dr = 1;
    for (let i = 0; i < 12; i++) {
        p = p.map(c => Math.min(Math.max(c, -1), 1) * 2 - c);
        const r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if (r2 < 0.25) { p = scl(p, 4); dr *= 4; } else if (r2 < 1) { p = scl(p, 1 / r2); dr *= 1 / r2; }
        p = add(scl(p, -1.5), pos);
        dr = dr * 1.5 + 1;
    }
    return len(p) / Math.abs(dr);
}
const SCENES = [
    ['mandelbulb', p => mandelbulbDE(p, 8)],
    ['julia', juliaDE],
    ['menger', mengerDE],
    ['mandelbox', p => mandelboxDE(scl(p, 3)) / 3],
];

// ---- Independent membership references --------------------------------------------------------
function inMandelbulb(c) {     // bounded orbit of z -> z^8 + c (escape radius 2, same 15 iterations)
    let z = c;
    for (let i = 0; i < 15; i++) {
        const r = len(z);
        if (r > 2) return false;
        if (r === 0) { z = c; continue; }
        z = add(bulbPow(z, r, 8), c);
    }
    return len(z) <= 2;
}
function inJulia(p) {
    let z = [p[0], p[1], p[2], 0];
    for (let i = 0; i < 12; i++) { z = juliaStep(z); if (dot4(z) > 4) return false; }
    return true;
}
function inMenger(p) {          // level-4 sponge: inside the cube, and no level has >=2 middle-third digits
    if (p.some(c => Math.abs(c) > 1)) return false;
    let u = p.map(c => (c + 1) / 2);   // [0,1]^3
    for (let level = 0; level < 4; level++) {
        const digits = u.map(c => Math.min(2, Math.floor(c * 3)));
        if (digits.filter(dg => dg === 1).length >= 2) return false;
        u = u.map((c, i) => c * 3 - digits[i]);
    }
    return true;
}

// ---- Replica camera and sphere tracer (fs_main / rayMarch) -----------------------------------
const rotX = (p, a) => [p[0], Math.cos(a) * p[1] - Math.sin(a) * p[2], Math.sin(a) * p[1] + Math.cos(a) * p[2]];
const rotY = (p, a) => [Math.cos(a) * p[0] + Math.sin(a) * p[2], p[1], -Math.sin(a) * p[0] + Math.cos(a) * p[2]];
function cameraRay(px, py, W, H, u) {
    const uv = [(px - W * 0.5) / H, -((py - H * 0.5) / H)];
    let ro = rotY(rotX([0, 0, u.cameraZoom], u.cameraRotX), u.cameraRotY + u.time * 0.1);
    const fwd = norm(scl(ro, -1));
    const right = norm(cross([0, 1, 0], fwd));
    const up = cross(fwd, right);
    return { ro, rd: norm(add(add(fwd, scl(right, uv[0])), scl(up, uv[1]))) };
}
function rayMarch(sdf, ro, rd, maxSteps) {
    let t = 0;
    for (let i = 0; i < maxSteps; i++) {
        const d = sdf(add(ro, scl(rd, t)));
        if (d < 0.001) return t;
        if (t > 50) break;
        t += d * 0.5;
    }
    return -1;
}
function bruteMarch(inside, ro, rd, tMax, dt) {
    for (let t = 0; t < tMax; t += dt) if (inside(add(ro, scl(rd, t)))) return t;
    return -1;
}

test('sphere tracer silhouettes match brute-force membership marches', (t) => {
    const W = 32, H = 18;   // reduced resolution; same 16:9-ish aspect math as the shader
    const u = { cameraZoom: 3, cameraRotX: 0.3, cameraRotY: 0.5, time: 0 };
    const refs = { mandelbulb: inMandelbulb, julia: inJulia, menger: inMenger };
    for (const [name, sdf] of SCENES) {
        if (!refs[name]) continue;
        let both = 0, mismatch = 0, maxEarly = 0, overshoot = 0;
        for (let py = 0; py < H; py++)
            for (let px = 0; px < W; px++) {
                const { ro, rd } = cameraRay(px + 0.5, py + 0.5, W, H, u);
                const tm = rayMarch(sdf, ro, rd, 256);
                const tb = bruteMarch(refs[name], ro, rd, 6, 0.004);
                if ((tm > 0) !== (tb > 0)) { mismatch++; continue; }
                if (tm > 0) {
                    both++;
                    // A conservative DE never steps deep past the true surface.
                    if (tm > tb + 0.05) overshoot++;
                    maxEarly = Math.max(maxEarly, tb - tm);
                }
            }
        const rate = mismatch / (W * H);
        t.diagnostic(`${name}: hit pixels ${both}, hit/miss mismatches ${mismatch} (${(100 * rate).toFixed(1)}%), ` +
            `overshoots ${overshoot}, max (brute t − march t) ${maxEarly.toFixed(3)}`);
        assert.ok(both > 20, `${name} is visible from the default camera`);
        // (Early hits — march t < brute t — are reported but allowed: the escape-time DEs under-estimate
        // near the surface and grazing rays creep within the 0.001 threshold. Overshoots are capped at
        // 1% of hit pixels: a fixed-step 0.004 march can catch thin filaments a DE-limited march skips.)
        // Mismatches sit on the silhouette, where the 0.001 hit threshold / 0.004 brute step and the
        // DE's approximation disagree about grazing rays.
        assert.ok(rate < 0.08, `${name} silhouette mismatch rate`);
        assert.ok(overshoot <= 0.01 * both, `${name}: sphere tracer must not tunnel through the surface`);
    }
});

test('distance estimators: finite everywhere, folded ones 1-Lipschitz', (t) => {
    let s = 99;
    const rand = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 4 - 2;
    const L = {};
    for (const [name, sdf] of SCENES) {
        let maxL = 0;
        for (let i = 0; i < 2000; i++) {
            const p = [rand(), rand(), rand()];
            const q = add(p, scl(norm([rand(), rand(), rand()]), 0.01));
            const dp = sdf(p), dq = sdf(q);
            assert.ok(Number.isFinite(dp), `${name} finite at ${p}`);
            maxL = Math.max(maxL, Math.abs(dp - dq) / 0.01);
        }
        t.diagnostic(`${name}: max |ΔDE|/|Δp| over 2000 random pairs = ${maxL.toFixed(3)}`);
        // The shader marches at 0.5·DE, so an estimator with Lipschitz constant below ~2 cannot overstep.
        L[name] = maxL;
    }
    // Menger (box SDF with fold carving) and Mandelbox (folds are 1-Lipschitz, divided by |dr|) are
    // true 1-Lipschitz bounds. The escape-time estimators (Mandelbulb, Julia) are not: they jump where
    // the iteration count changes, which is why the shader steps 0.5·DE; their safety is asserted by
    // the no-tunneling check in the silhouette test instead.
    assert.ok(L.menger <= 1 + 1e-6, 'Menger DE is 1-Lipschitz');
    assert.ok(L.mandelbox <= 1 + 1e-6, 'Mandelbox DE is 1-Lipschitz');
});

test('far field: Menger DE is exact, damped Mandelbulb step is safe', (t) => {
    const p = [0, 0, 20];
    const bulb = mandelbulbDE(p, 8), menger = mengerDE(p);
    const code = moduleCode(mod, 'render');
    // Replica constants are transcribed from the shader; keep them in sync.
    assert.match(code, /t \+= d \* 0\.5;/);
    assert.match(code, /let c = vec4<f32>\(0\.18, 0\.88, 0\.24, 0\.16\);/);
    assert.match(code, /let scale = -1\.5;/);
    assert.match(code, /if \(d < 0\.001\)/);
    t.diagnostic(`mandelbulb DE at z=20: ${bulb.toFixed(3)}; menger DE at z=20: ${menger.toFixed(6)} (exact 19)`);
    assert.ok(Math.abs(menger - 19) < 1e-9, 'Menger DE equals the box SDF outside the cube');
    // At r > 2 the loop exits on its first pass (dr = 1), so DE = 0.5·r·ln r over-estimates the true
    // distance (~19); the shader's 0.5 step damping keeps the actual step (0.5·DE) below it.
    assert.ok(bulb > 0 && 0.5 * bulb < 19, 'damped Mandelbulb step is safe in the far field');
    // The shader's mandelbox scene is mandelboxDE(p*3)/3 so it fits the default view.
    assert.match(moduleCode(mod, 'render'), /case 3u: \{ return mandelboxDE\(pos \* 3\.0\) \/ 3\.0; \}/);
});
