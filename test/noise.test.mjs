// Procedural noise (example: js/examples/noise.js).
//
// CPU replica of noise.js computeShader (pcg3d, grad, fade, perlin3, simplex3, fbm, warpedFbm);
// validates the algorithm and index math, not the WGSL itself — see #51.
// The hash is exact u32 arithmetic; the noise itself is evaluated in float64 so that continuity
// checks measure the algorithm's seams, not f32 rounding. Shader constants (F3, G3, SIMPLEX_SCALE,
// the simplex falloff radius) are parsed from the WGSL so a change there is caught here.
//
// Independent references: Perlin noise as an explicit sum over the 8 corners with separable
// fade weights (instead of nested mixes), and simplex noise as a brute-force sum of the corner kernel
// over every skewed-lattice point near p (instead of picking the containing tetrahedron).
//
// Real code tested directly: NoiseField.writeParams (mock device, decoded with struct Params from the
// WGSL, incl. the octave clamp) and the shipped PRESETS.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as noise from '../js/examples/noise.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { hashCorner, grad, fade, perlinRef, simplexRef as simplexRef64 } from '../js/refs/noise.js';

const CODE = moduleCode(noise, 'compute');
const constF32 = name => {
    const m = new RegExp(`const ${name}: f32 = ([^;]+);`).exec(CODE);
    assert.ok(m, `const ${name} not found in WGSL`);
    return Function(`return (${m[1]});`)();
};
const F3 = constF32('F3'), G3 = constF32('G3'), SIMPLEX_SCALE = constF32('SIMPLEX_SCALE');
const FALLOFF = Number(/let t = ([\d.]+) - dot\(d, d\);/.exec(CODE)[1]);   // 0.5 = r^2 of the kernel
const MAX_OCTAVES = Number(/const MAX_OCTAVES: u32 = (\d+)u;/.exec(CODE)[1]);

// ---- replica ---------------------------------------------------------------------------------
// pcg3d / hashCorner, grad and fade are shared with the independent references (js/refs/noise.js).
const mix = (a, b, t) => a + (b - a) * t;

function perlin3(x, y, z) {
    const cx = Math.floor(x), cy = Math.floor(y), cz = Math.floor(z);
    const fx = x - cx, fy = y - cy, fz = z - cz;
    const ux = fade(fx), uy = fade(fy), uz = fade(fz);
    const n = (a, b, c) => grad(hashCorner(cx + a, cy + b, cz + c), fx - a, fy - b, fz - c);
    const nx00 = mix(n(0, 0, 0), n(1, 0, 0), ux);
    const nx10 = mix(n(0, 1, 0), n(1, 1, 0), ux);
    const nx01 = mix(n(0, 0, 1), n(1, 0, 1), ux);
    const nx11 = mix(n(0, 1, 1), n(1, 1, 1), ux);
    return mix(mix(nx00, nx10, uy), mix(nx01, nx11, uy), uz);
}

function simplexCorner(h, dx, dy, dz) {
    const t = FALLOFF - (dx * dx + dy * dy + dz * dz);
    if (t <= 0) return 0;
    const t2 = t * t;
    return t2 * t2 * grad(h, dx, dy, dz);
}

function simplex3(x, y, z) {
    const s = (x + y + z) * F3;
    const cx = Math.floor(x + s), cy = Math.floor(y + s), cz = Math.floor(z + s);
    const t = (cx + cy + cz) * G3;
    const x0 = x - (cx - t), y0 = y - (cy - t), z0 = z - (cz - t);
    let i1, i2;
    if (x0 >= y0) {
        if (y0 >= z0) { i1 = [1, 0, 0]; i2 = [1, 1, 0]; }
        else if (x0 >= z0) { i1 = [1, 0, 0]; i2 = [1, 0, 1]; }
        else { i1 = [0, 0, 1]; i2 = [1, 0, 1]; }
    } else {
        if (y0 < z0) { i1 = [0, 0, 1]; i2 = [0, 1, 1]; }
        else if (x0 < z0) { i1 = [0, 1, 0]; i2 = [0, 1, 1]; }
        else { i1 = [0, 1, 0]; i2 = [1, 1, 0]; }
    }
    const n = simplexCorner(hashCorner(cx, cy, cz), x0, y0, z0)
        + simplexCorner(hashCorner(cx + i1[0], cy + i1[1], cz + i1[2]), x0 - i1[0] + G3, y0 - i1[1] + G3, z0 - i1[2] + G3)
        + simplexCorner(hashCorner(cx + i2[0], cy + i2[1], cz + i2[2]), x0 - i2[0] + 2 * G3, y0 - i2[1] + 2 * G3, z0 - i2[2] + 2 * G3)
        + simplexCorner(hashCorner(cx + 1, cy + 1, cz + 1), x0 - 1 + 3 * G3, y0 - 1 + 3 * G3, z0 - 1 + 3 * G3);
    return SIMPLEX_SCALE * n;
}

function fbm(p, P) {
    const base = P.noiseType === 1 ? simplex3 : perlin3;
    const octaves = Math.min(Math.max(P.octaves, 1), MAX_OCTAVES);
    let [x, y, z] = p, amplitude = 1, total = 0, norm = 0;
    for (let o = 0; o < octaves; o++) {
        let n = base(x, y, z);
        if (P.ridged) { const r = 1 - Math.abs(n); n = 2 * r * r - 1; }
        total += amplitude * n;
        norm += amplitude;
        amplitude *= P.gain;
        x = x * P.lacunarity + 19.1; y = y * P.lacunarity + 7.3; z = z * P.lacunarity + 3.7;
    }
    return total / norm;
}
function warpedFbm(p, P) {
    if (P.warp <= 0) return fbm(p, P);
    const qx = fbm(p, P), qy = fbm([p[0] + 5.2, p[1] + 1.3, p[2]], P);
    return fbm([p[0] + P.warp * qx, p[1] + P.warp * qy, p[2]], P);
}

// ---- independent references (js/refs/noise.js) ------------------------------------------------
// Perlin: sum over the 8 corners, each weighted by the separable falloff w(t) = 1 - fade(|t|).
// Simplex: brute force over every skewed-lattice point in a 5x5x5 block around p, with the
// shader's output scale and kernel radius² parsed above.
const simplexRef = (x, y, z) => simplexRef64(x, y, z, SIMPLEX_SCALE, FALLOFF);

function lcg(seed) {
    let s = seed >>> 0;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function stats(values) {
    let mean = 0, min = Infinity, max = -Infinity;
    for (const v of values) { mean += v; min = Math.min(min, v); max = Math.max(max, v); }
    mean /= values.length;
    let v2 = 0;
    for (const v of values) v2 += (v - mean) ** 2;
    return { mean, std: Math.sqrt(v2 / values.length), min, max };
}
const fmt = s => `mean ${s.mean.toFixed(4)}, std ${s.std.toFixed(4)}, range [${s.min.toFixed(3)}, ${s.max.toFixed(3)}]`;

// ---- tests -----------------------------------------------------------------------------------

test('shader constants are the textbook simplex skew factors', () => {
    assert.ok(Math.abs(F3 - (Math.sqrt(4) - 1) / 3) < 1e-12);
    assert.ok(Math.abs(G3 - (1 - 1 / Math.sqrt(4)) / 3) < 1e-12);
    assert.equal(FALLOFF, 0.5, 'falloff radius^2 must be 0.5 for seam-free 3D simplex');
});

test('pcg3d gradient selection is uniform over the 16 codes', (t) => {
    const bins = new Array(16).fill(0), N = 64000;
    let n = 0;
    for (let i = -20; i < 20; i++) for (let j = -20; j < 20; j++) for (let k = 0; k < 40; k++) {
        bins[hashCorner(i, j, k) >>> 28]++; n++;
    }
    assert.equal(n, N);
    const e = N / 16;
    const chi2 = bins.reduce((s, b) => s + (b - e) ** 2 / e, 0);
    t.diagnostic(`chi^2 over 16 gradient codes (64000 corners) = ${chi2.toFixed(1)} (15 dof; p=0.001 cut 37.7)`);
    assert.ok(chi2 < 37.7);
});

test('perlin3 replica equals the 8-corner reference, vanishes on the lattice, has the expected statistics', (t) => {
    const rnd = lcg(1);
    let maxErr = 0;
    const vals = [];
    for (let i = 0; i < 20000; i++) {
        const x = (rnd() - 0.5) * 200, y = (rnd() - 0.5) * 200, z = (rnd() - 0.5) * 200;
        const v = perlin3(x, y, z);
        maxErr = Math.max(maxErr, Math.abs(v - perlinRef(x, y, z)));
        vals.push(v);
    }
    for (let i = -5; i <= 5; i++) assert.equal(perlin3(i, 2 * i, -i + 3), 0, 'Perlin noise is zero at lattice points');
    const s = stats(vals);
    t.diagnostic(`perlin3 vs reference max |diff| ${maxErr.toExponential(2)}; ${fmt(s)}`);
    assert.ok(maxErr < 1e-12);
    assert.ok(Math.abs(s.mean) < 0.01, 'mean ~ 0');
    assert.ok(s.std > 0.15 && s.std < 0.35, 'std in the usual 3D Perlin band');
    assert.ok(s.max <= 1.1 && s.min >= -1.1, 'range about [-1, 1]');
});

test('simplex3 replica equals the brute-force lattice sum; SIMPLEX_SCALE maps it to about [-1, 1]', (t) => {
    const rnd = lcg(2);
    let maxErr = 0;
    const vals = [];
    for (let i = 0; i < 20000; i++) {
        const x = (rnd() - 0.5) * 200, y = (rnd() - 0.5) * 200, z = (rnd() - 0.5) * 200;
        const v = simplex3(x, y, z);
        maxErr = Math.max(maxErr, Math.abs(v - simplexRef(x, y, z)));
        vals.push(v);
    }
    const s = stats(vals);
    t.diagnostic(`simplex3 vs brute force max |diff| ${maxErr.toExponential(2)}; ${fmt(s)}`);
    assert.ok(maxErr < 1e-9, 'tetrahedron selection must cover every non-zero kernel');
    assert.ok(Math.abs(s.mean) < 0.01);
    assert.ok(s.max <= 1.1 && s.min >= -1.1, 'scaled range about [-1, 1]');
    assert.ok(Math.max(s.max, -s.min) > 0.6, 'scale should use most of [-1, 1]');
});

test('both noises are continuous: Lipschitz-bounded, no jumps at cell / simplex / kernel boundaries', (t) => {
    // A jump of size J between two samples h apart shows up as a second difference ~J, while a
    // smooth (C2) field gives ~|f''| h^2 (~1e-6 here). With r^2 = 0.6 instead of 0.5, simplex has
    // jumps at simplex faces; a broken lattice/tetrahedron index gives jumps of ~1e-1.
    const h1 = 1e-6, h2 = 1e-4, JUMP = 1e-5;
    for (const [name, f, bound] of [['perlin', perlin3, 4], ['simplex', simplex3, 10]]) {
        const rnd = lcg(name === 'perlin' ? 3 : 4);
        let maxSlope = 0, maxSecond = 0;
        const probe = (x, y, z, dx, dy, dz) => {
            const len = Math.hypot(dx, dy, dz);
            dx /= len; dy /= len; dz /= len;
            const f0 = f(x, y, z);
            maxSlope = Math.max(maxSlope, Math.abs(f(x + h1 * dx, y + h1 * dy, z + h1 * dz) - f0) / h1);
            const s = f(x + h2 * dx, y + h2 * dy, z + h2 * dz) - 2 * f0 + f(x - h2 * dx, y - h2 * dy, z - h2 * dz);
            maxSecond = Math.max(maxSecond, Math.abs(s));
        };
        for (let i = 0; i < 40000; i++) {
            probe((rnd() - 0.5) * 50, (rnd() - 0.5) * 50, (rnd() - 0.5) * 50, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
        }
        for (let i = 0; i < 5000; i++) {
            const ci = Math.floor((rnd() - 0.5) * 50), cj = Math.floor((rnd() - 0.5) * 50), ck = Math.floor((rnd() - 0.5) * 50);
            const a = rnd(), b = rnd();
            if (name === 'perlin') {
                probe(ci, cj + a, ck + b, 1, 0.1 * (b - 0.5), 0.1 * (a - 0.5));   // across an x = integer plane
            } else {
                // A point where x0.x == x0.y (a face between two tetrahedra), stepped across it
                const tt = (ci + cj + ck) * G3;
                probe(ci - tt + a, cj - tt + a, ck - tt + b * a, 1, -1, 0.05);
            }
        }
        t.diagnostic(`${name}: max |df|/h ${maxSlope.toFixed(3)} (bound ${bound}); max |2nd difference| at h=1e-4: ${maxSecond.toExponential(2)} (jump bound ${JUMP})`);
        assert.ok(maxSlope < bound, `${name}: gradient too large`);
        assert.ok(maxSecond < JUMP, `${name} has a seam`);
    }
});

test('fbm: one octave is the base noise, normalization keeps the range, deterministic', (t) => {
    const rnd = lcg(5);
    for (const noiseType of [0, 1]) {
        const one = { noiseType, octaves: 1, gain: 0.5, lacunarity: 2, ridged: false, warp: 0 };
        const p = [rnd() * 10, rnd() * 10, 0.5];
        assert.equal(fbm(p, one), (noiseType ? simplex3 : perlin3)(...p));
        // octaves above MAX_OCTAVES are clamped like the shader does
        const big = { ...one, octaves: 40 }, max = { ...one, octaves: MAX_OCTAVES };
        assert.equal(fbm(p, big), fbm(p, max));
    }
    for (const [key, P] of Object.entries(noise.PRESETS)) {
        const vals = [];
        for (let i = 0; i < 1500; i++) {
            const p = [(rnd() - 0.5) * 8, (rnd() - 0.5) * 8, 0.5];
            const v = warpedFbm(p, P);
            assert.equal(v, warpedFbm(p, P));
            vals.push(v);
        }
        const s = stats(vals);
        t.diagnostic(`preset ${key}: ${fmt(s)}`);
        assert.ok(s.min >= -1.1 && s.max <= 1.1, `${key}: fbm left [-1, 1]`);
        assert.ok(s.std > 0.03, `${key}: field is flat`);
    }
});

test('shipped PRESETS are within the controls\' ranges', () => {
    for (const [key, P] of Object.entries(noise.PRESETS)) {
        assert.ok([0, 1].includes(P.noiseType), key);
        assert.ok(Number.isInteger(P.octaves) && P.octaves >= 1 && P.octaves <= MAX_OCTAVES, key);
        assert.ok(P.frequency > 0 && P.lacunarity > 1 && P.gain > 0 && P.gain < 1 && P.warp >= 0, key);
        assert.ok([0, 1, 2, 3].includes(P.colorMode), key);
    }
});

function fakeCanvas() {
    return {
        width: 512, height: 384,
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
        addEventListener() {}, removeEventListener() {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 1, height: 1 }),
    };
}

test('NoiseField.writeParams matches struct Params (u32 fields as integers, octaves clamped)', (t) => {
    globalThis.window ??= { addEventListener() {}, removeEventListener() {} };
    const device = mockDevice();
    const field = new noise.NoiseField(device, fakeCanvas());
    const layout = structLayout(moduleCode(noise, 'compute'), 'Params');
    assert.deepEqual(structLayout(moduleCode(noise, 'render'), 'Params').offsets, layout.offsets);
    assert.equal(layout.size, 56);

    Object.assign(field, {
        center: [0.375, -1.25], unitsPerPixel: 0.0025, frequency: 2.5, time: 3.5, octaves: 12,
        lacunarity: 2.125, gain: 0.45, warp: 0.75, noiseType: 1, ridged: true, colorMode: 2,
    });
    field.writeParams();
    const p = decodeStruct(device.lastWrite(field.paramsBuffer).bytes, layout);
    assert.deepEqual(p.size, [512, 384]);
    assert.deepEqual(p.center, [0.375, -1.25]);
    assert.equal(p.unitsPerPixel, Math.fround(0.0025));
    assert.equal(p.frequency, 2.5);
    assert.equal(p.time, 3.5);
    assert.equal(p.octaves, MAX_OCTAVES, 'octaves clamped to MAX_OCTAVES');
    assert.equal(p.lacunarity, 2.125);
    assert.equal(p.gain, Math.fround(0.45));
    assert.equal(p.warp, 0.75);
    assert.equal(p.noiseType, 1);
    assert.equal(p.ridged, 1);
    assert.equal(p.colorMode, 2);
    t.diagnostic(`Params ${layout.size} B; offsets ${JSON.stringify(layout.offsets)}`);
});
