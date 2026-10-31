// Gray-Scott reaction-diffusion (example: js/examples/reaction-diffusion.js).
//
// CPU replica of reaction-diffusion.js computeShader (cs_main); validates the algorithm and index
// math, not the WGSL itself — see #51. State is stored in Float32Array like the GPU buffers; the
// per-cell arithmetic runs in float64. DU, DV, DT and the Laplacian weights are parsed from the WGSL.
//
// Independent reference: a float64 Gray-Scott step that convolves an explicit 3x3 kernel with
// modulo-wrapped indices (the replica sums edges and corners the way the shader does).
//
// Real code tested directly: ReactionDiffusion.seed() (with a seeded Math.random, bytes captured
// from the mock device), ReactionDiffusion.writeParams (decoded with struct Params from the WGSL)
// and the shipped PRESETS.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as rd from '../js/examples/reaction-diffusion.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import * as refs from './helpers/reaction-diffusion-refs.mjs';

// Constants parsed from the WGSL, the replica and the float64 reference live in
// test/helpers/reaction-diffusion-refs.mjs (shared with test/gpu/).
const { W_EDGE, W_CORNER, W_CENTER, DU, DV, DT } = refs;
const { stepReplica, simulate, stepReference, shippedSeed } = refs;

function vStats(state) {
    let mean = 0, n = state.length / 2, covered = 0, nan = 0, outside = 0;
    for (let i = 0; i < n; i++) {
        const u = state[2 * i], v = state[2 * i + 1];
        if (Number.isNaN(u) || Number.isNaN(v)) nan++;
        if (u < 0 || u > 1 || v < 0 || v > 1) outside++;
        mean += v;
        if (v > 0.1) covered++;
    }
    mean /= n;
    let v2 = 0;
    for (let i = 0; i < n; i++) v2 += (state[2 * i + 1] - mean) ** 2;
    return { mean, std: Math.sqrt(v2 / n), covered: covered / n, nan, outside };
}

// ---- tests ----------------------------------------------------------------------------------

test('Laplacian stencil: weights sum to zero, isotropic 9-point form, explicit Euler is stable', (t) => {
    const sum = W_CENTER + 4 * W_EDGE + 4 * W_CORNER;
    assert.ok(Math.abs(sum) < 1e-12, `weights sum to ${sum}`);
    // Second moment: sum w * x^2 = 2 * (W_EDGE + 2 W_CORNER) — the kernel approximates c * lap with c = 0.3
    const c = W_EDGE + 2 * W_CORNER;
    // Fourier symbol lambda(kx, ky) = W_CENTER + 2 W_EDGE (cos kx + cos ky) + 4 W_CORNER cos kx cos ky;
    // forward Euler on dU/dt = D lap U - a U is stable iff DT * (D |lambda| + a) <= 2
    let minSymbol = 0;
    for (let i = 0; i <= 64; i++) for (let j = 0; j <= 64; j++) {
        const kx = Math.PI * i / 64, ky = Math.PI * j / 64;
        const l = W_CENTER + 2 * W_EDGE * (Math.cos(kx) + Math.cos(ky)) + 4 * W_CORNER * Math.cos(kx) * Math.cos(ky);
        assert.ok(l <= 1e-12, 'symbol must be non-positive (diffusion only damps)');
        minSymbol = Math.min(minSymbol, l);
    }
    let worst = 0;
    for (const { feed, kill } of Object.values(rd.PRESETS)) {
        worst = Math.max(worst, DT * (DU * -minSymbol + feed), DT * (DV * -minSymbol + feed + kill));
    }
    t.diagnostic(`weights center ${W_CENTER}, edge ${W_EDGE}, corner ${W_CORNER}; diffusion scale ${c}; ` +
        `min symbol ${minSymbol.toFixed(3)}; worst DT*(D|lambda|+rate) over presets ${worst.toFixed(3)} (stable <= 2)`);
    assert.ok(worst <= 2);
});

test('replica matches the float64 3x3-convolution reference', (t) => {
    const w = 48, h = 40;
    const init = shippedSeed(w, h, 3);
    let worst = 0;
    for (const [key, { feed, kill }] of Object.entries(rd.PRESETS)) {
        let a = Float32Array.from(init), b = new Float32Array(init.length);
        let U = new Float64Array(w * h), V = new Float64Array(w * h);
        for (let i = 0; i < w * h; i++) { U[i] = init[2 * i]; V[i] = init[2 * i + 1]; }
        let maxDiff = 0;
        for (let s = 0; s < 200; s++) {
            stepReplica(a, b, w, h, feed, kill); [a, b] = [b, a];
            [U, V] = stepReference(U, V, w, h, feed, kill);
        }
        for (let i = 0; i < w * h; i++) maxDiff = Math.max(maxDiff, Math.abs(a[2 * i] - U[i]), Math.abs(a[2 * i + 1] - V[i]));
        worst = Math.max(worst, maxDiff);
        t.diagnostic(`${key}: max |replica - reference| after 200 steps = ${maxDiff.toExponential(2)}`);
    }
    // f32 storage rounds each step by ~6e-8 relative; 200 steps of a contractive-ish map stay < 1e-4
    assert.ok(worst < 1e-4);
});

test('shipped seed(): U = 1 / V = 0 background plus max(4, n/10000) 10x10 patches of (0.5, 0.25 +- 0.01)', (t) => {
    const w = 200, h = 150;
    const data = shippedSeed(w, h, 42);
    assert.equal(data.length, 2 * w * h);
    let patchCells = 0;
    for (let i = 0; i < w * h; i++) {
        const u = data[2 * i], v = data[2 * i + 1];
        if (u === 1) assert.equal(v, 0);
        else {
            assert.equal(u, 0.5);
            assert.ok(Math.abs(v - 0.25) <= 0.01 + 1e-7, `V = ${v}`);
            patchCells++;
        }
    }
    const patches = Math.max(4, Math.round(w * h / 10000));
    assert.ok(patchCells > 0 && patchCells <= patches * 100);
    assert.deepEqual(shippedSeed(w, h, 42), data, 'seed() is not deterministic under a fixed Math.random');
    t.diagnostic(`${w}x${h}: ${patches} patches, ${patchCells} seeded cells`);
});

test('every preset is stable from the shipped seed and forms a persistent pattern', (t) => {
    const w = 96, h = 96, steps = 2500;
    const init = shippedSeed(w, h, 7);
    const s0 = vStats(init);
    for (const [key, { feed, kill }] of Object.entries(rd.PRESETS)) {
        const out = simulate(init, w, h, steps, feed, kill);
        const s = vStats(out);
        t.diagnostic(`${key} (F=${feed}, k=${kill}): V coverage ${(100 * s0.covered).toFixed(1)}% -> ${(100 * s.covered).toFixed(1)}%, ` +
            `V std ${s0.std.toFixed(4)} -> ${s.std.toFixed(4)}, mean V ${s.mean.toFixed(4)}`);
        assert.equal(s.nan, 0, `${key}: NaN`);
        assert.equal(s.outside, 0, `${key}: left [0, 1]`);
        // The pattern neither dies out (V -> 0 everywhere) nor saturates into a uniform state
        assert.ok(s.covered > 0.01, `${key}: pattern died out`);
        assert.ok(s.std > 0.02, `${key}: field became uniform`);
        // Every preset grows well beyond the seeded patches (measured: 3.8x for mitosis, 9-21x others)
        assert.ok(s.covered > 2 * s0.covered, `${key}: pattern did not spread`);
    }
});

test('ReactionDiffusion.writeParams matches struct Params', (t) => {
    globalThis.window ??= { addEventListener() {}, removeEventListener() {} };
    const canvas = {
        width: 512, height: 512,
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
        addEventListener() {}, removeEventListener() {},
    };
    const device = mockDevice();
    const sim = new rd.ReactionDiffusion(device, canvas, 256, 192);
    Object.assign(sim, { brushPos: [12.5, 40.25], brushPrev: [11.0, 39.75], feed: 0.0367, kill: 0.0649, brushRadius: 9, palette: 2 });
    const layout = structLayout(moduleCode(rd, 'compute'), 'Params');
    assert.deepEqual(structLayout(moduleCode(rd, 'render'), 'Params').offsets, layout.offsets);
    assert.equal(layout.size, 48);
    sim.writeParams(true, false);
    const p = decodeStruct(device.lastWrite(sim.paramsBuffer).bytes, layout);
    assert.deepEqual(p.size, [256, 192]);
    assert.deepEqual(p.brushPos, [12.5, 40.25]);
    assert.deepEqual(p.brushPrev, [11.0, 39.75]);
    assert.equal(p.feed, Math.fround(0.0367));
    assert.equal(p.kill, Math.fround(0.0649));
    assert.equal(p.brushRadius, 9);
    assert.equal(p.brushOn, 0);
    assert.equal(p.simulate, 1);
    assert.equal(p.palette, 2);
    sim.writeParams(false, true);
    const q = decodeStruct(device.lastWrite(sim.paramsBuffer).bytes, layout);
    assert.equal(q.brushOn, 1);
    assert.equal(q.simulate, 0);
    t.diagnostic(`Params ${layout.size} B; offsets ${JSON.stringify(layout.offsets)}`);
});
