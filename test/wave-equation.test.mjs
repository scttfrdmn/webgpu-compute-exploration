// Wave equation (example 11): CPU tests.
//
// CPU replica of wave-equation.js `cs_step` (compute shader); validates the algorithm and index
// math, not the WGSL itself — see #51. The replica follows the shader exactly: 5-point Laplacian
// with u = 0 outside the grid, damped leapfrog
//   u(t+1) = (2u - (1 - g)·u(t-1) + C²·lap) / (1 + g),
// the quadratic absorbing sponge (SPONGE = 24 cells, SPONGE_STRENGTH = 0.08), walls fixed at 0,
// and the hard plane-wave source column x = 30. Heights are stored as f32 like the GPU buffers.
//
// Real shipped code exercised: WaveEquation (constructed on a mock device) for writeParams() and
// loadPreset() — the preset walls and initial water the tests run on are the bytes the example
// uploads — plus PARAMS_SIZE / SOURCE_PERIOD and the Params struct from wgslModules().
//
// Independent references: the leapfrog scheme's exactly conserved discrete energy
//   E(n+½) = ‖u(n+1) − u(n)‖² − C²·⟨u(n+1), L u(n)⟩
// (positive-definite iff C ≤ 1/√2, i.e. the CFL limit), von Neumann's growth factor for the
// checkerboard mode above the limit, ellipse focus-to-focus travel time 2a/C, and two-slit
// interference maxima from exact path differences |r1 − r2| = mλ.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import * as wave from '../js/examples/wave-equation.js';
import * as refs from './helpers/wave-equation-refs.mjs';

const { WaveEquation, PARAMS_SIZE, SOURCE_PERIOD } = wave;

// Shader constants, the shipped-state capture (makeSim / presetState), the cs_step replica and the
// discrete energy live in test/helpers/wave-equation-refs.mjs (shared with test/gpu/).
const { SOURCE_X, SOURCE_AMP, SPONGE, SPONGE_STRENGTH, makeSim, presetState, stepReplica, energy } = refs;

function maxAbs(u) {
    let m = 0;
    for (const v of u) m = Math.max(m, Math.abs(v));
    return m;
}

function run(state, { courant, gamma = 0, absorb = false, planeWave = false, steps, onStep }) {
    const s = { W: state.W, H: state.H, cur: Float32Array.from(state.cur), prev: Float32Array.from(state.prev), walls: state.walls };
    const C2 = courant * courant;
    for (let n = 0; n < steps; n++) {
        const before = s.cur;
        const snapshot = onStep ? Float32Array.from(before) : null;
        stepReplica(s, { courant2: C2, damping: gamma / 2, absorb, planeWave, step: n, period: SOURCE_PERIOD });
        if (onStep) onStep(n, snapshot, s.cur, C2);
    }
    return s;
}

test('Params layout: writeParams() bytes decode to the WGSL struct (both modules)', t => {
    const { sim, device } = makeSim(512, 384);
    const computeLayout = structLayout(moduleCode(wave, 'compute'), 'Params');
    const renderLayout = structLayout(moduleCode(wave, 'render'), 'Params');
    assert.deepEqual(renderLayout.offsets, computeLayout.offsets);
    assert.equal(computeLayout.size, PARAMS_SIZE, 'PARAMS_SIZE (minBindingSize) must equal the WGSL struct size');
    t.diagnostic(`Params: ${computeLayout.size} bytes, stride ${sim.paramsStride}`);

    Object.assign(sim, {
        cursor: [12.5, 7.25], cursorPrev: [3.75, 99.5], courant: 0.6, dampingGamma: 0.012,
        steps: 1000, rain: true, planeWave: false, absorb: true, seedValue: 0xdeadbeef, dropAmp: 1.75,
    });
    device.writes.length = 0;
    sim.writeParams(3, true, false);
    const w = device.lastWrite(sim.paramsBuffer);
    assert.equal(w.bytes.byteLength, 3 * sim.paramsStride);
    for (let k = 0; k < 3; k++) {
        const p = decodeStruct(w.bytes, computeLayout, k * sim.paramsStride);
        assert.deepEqual(p.size, [512, 384]);
        assert.deepEqual(p.cursor, [12.5, 7.25]);
        assert.deepEqual(p.cursorPrev, [3.75, 99.5]);
        assert.equal(p.courant2, Math.fround(0.36));
        assert.equal(p.damping, Math.fround(0.006));
        assert.equal(p.step, 1000 + k, 'each substep slot carries its own step');
        assert.equal(p.period, SOURCE_PERIOD);
        assert.equal(p.dropOn, 1);
        assert.equal(p.wallOn, 0);
        assert.equal(p.rainOn, 1);
        assert.equal(p.planeWave, 0);
        assert.equal(p.absorb, 1);
        assert.equal(p.seed, 0xdeadbeef);
        assert.equal(p.dropAmp, 1.75);
    }
});

test('CFL: discrete energy conserved for C ≤ 0.70 (slider range), ellipse walls, no damping', t => {
    const W = 96, H = 72;
    const st = presetState('ellipse', W, H);
    const state = { W, H, cur: st.cur, prev: st.prev, walls: st.walls };
    for (const courant of [0.1, 0.5, 0.7]) {
        let e0 = null, maxDrift = 0;
        const s = run(state, {
            courant, steps: 1500, onStep: (n, u0, u1, C2) => {
                if (n % 25 !== 0) return;
                const e = energy(u0, u1, W, H, C2);
                if (e0 === null) e0 = e;
                maxDrift = Math.max(maxDrift, Math.abs(e - e0) / e0);
            },
        });
        const peak = maxAbs(s.cur);
        t.diagnostic(`C=${courant}: E0=${e0.toExponential(4)}, max relative energy drift ${maxDrift.toExponential(2)} over 1500 steps, max|u|=${peak.toFixed(4)}`);
        assert.ok(e0 > 0);
        // Only f32 rounding perturbs the invariant; 1e-4 leaves ~10x headroom over what was measured.
        assert.ok(maxDrift < 1e-4, `energy drift ${maxDrift} at C=${courant}`);
        assert.ok(peak < 2 * maxAbs(st.cur));
    }
});

test('CFL: damping and the absorbing sponge only remove energy', t => {
    const W = 96, H = 72;
    const st = presetState('open', W, H);
    const state = { W, H, cur: st.cur, prev: st.prev, walls: st.walls };
    for (const [gamma, absorb] of [[0.02, false], [0, true], [0.004, true]]) {
        let last = null, e0 = null, rises = 0;
        run(state, {
            courant: 0.7, gamma, absorb, steps: 800, onStep: (n, u0, u1, C2) => {
                const e = energy(u0, u1, W, H, C2);
                if (e0 === null) e0 = e;
                if (last !== null && e > last * (1 + 1e-5)) rises++;
                last = e;
            },
        });
        t.diagnostic(`C=0.7 γ=${gamma} sponge=${absorb}: E 800 steps / E0 = ${(last / e0).toExponential(3)}, increases ${rises}`);
        assert.equal(rises, 0, 'energy must be non-increasing with damping ≥ 0');
        assert.ok(last < e0);
    }
});

test('CFL limit is real: C = 0.72 > 1/√2 blows up at the von Neumann rate', t => {
    const W = 64, H = 48;
    const st = presetState('open', W, H);
    const state = { W, H, cur: st.cur, prev: st.prev, walls: st.walls };
    const courant = 0.72;
    let blowUp = -1;
    const amp = [];
    const s = run(state, {
        courant, steps: 400, onStep: (n, u0, u1) => {
            amp.push(maxAbs(u1));
            if (blowUp < 0 && amp[n] > 1e3) blowUp = n;
        },
    });
    // Checkerboard mode: r² − (2 − 8C²)·r + 1 = 0, |r| = 1.464 at C = 0.72
    const b = 2 - 8 * courant * courant;
    const r = Math.abs((b - Math.sqrt(b * b - 4)) / 2);
    // Growth measured over the 20 steps before max|u| reaches 1e3 (the unstable mode dominates)
    const measured = Math.pow(amp[blowUp] / amp[blowUp - 20], 1 / 20);
    t.diagnostic(`C=0.72: max|u| > 1e3 after ${blowUp} steps; growth ${measured.toFixed(3)}/step vs von Neumann ${r.toFixed(3)}/step; final max|u| ${maxAbs(s.cur).toExponential(2)}`);
    assert.ok(blowUp > 0 && blowUp < 400, 'must blow up just above the CFL limit');
    assert.ok(Math.abs(measured - r) / r < 0.05, 'growth rate matches the von Neumann factor within 5%');
    // Contrast: the slider maximum 0.70 stays bounded over the same steps
    const ok = run(state, { courant: 0.7, steps: 400 });
    assert.ok(maxAbs(ok.cur) < 2 * maxAbs(st.cur));
});

test('Ellipse preset: drop at one focus refocuses at the other after 2a/C steps', t => {
    const W = 128, H = 96;
    const st = presetState('ellipse', W, H);
    const a = W * 0.45, b = H * 0.42;
    const focus = Math.sqrt(a * a - b * b);
    const fx = Math.floor(W / 2 + focus), fy = Math.floor(H / 2);
    const courant = 0.5;
    let bestN = -1, best = 0;
    run({ W, H, cur: st.cur, prev: st.prev, walls: st.walls }, {
        courant, steps: 340, onStep: (n, u0, u1) => {
            if (n < 100) return;   // skip the initial direct wave
            let v = 0;
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) v = Math.max(v, Math.abs(u1[(fy + dy) * W + fx + dx]));
            if (v > best) { best = v; bestN = n + 1; }
        },
    });
    const predicted = 2 * a / courant;
    t.diagnostic(`peak |u| at far focus ${best.toFixed(4)} at step ${bestN}; predicted 2a/C = ${predicted.toFixed(1)} (${(100 * (bestN - predicted) / predicted).toFixed(1)}%)`);
    // The ripple has finite width (σ = 4 cells) and the grid is dispersive; 5% covers both.
    assert.ok(Math.abs(bestN - predicted) / predicted < 0.05);
});

test('Double-slit preset: interference maxima match exact path differences', t => {
    const W = 512, H = 384;   // the shipped grid; the slit geometry is in absolute cells
    const st = presetState('slit', W, H);
    assert.equal(st.sim.planeWave, true);
    assert.equal(st.sim.absorb, true);
    const wallX = Math.round(W * 0.35);
    // Recover the slit centers from the uploaded wall column
    const open = [];
    for (let y = 0; y < H; y++) if (st.walls[y * W + wallX] === 0) open.push(y);
    const gaps = [];
    for (const y of open) {
        if (gaps.length && y === gaps[gaps.length - 1].end + 1) gaps[gaps.length - 1].end = y;
        else gaps.push({ start: y, end: y });
    }
    assert.equal(gaps.length, 2, 'two slits');
    const centers = gaps.map(g => (g.start + g.end + 1) / 2);
    const d = centers[1] - centers[0];

    const courant = st.sim.courant, gamma = st.sim.dampingGamma;
    const screenX = 442, L = screenX + 0.5 - (wallX + 3);
    const steps = 1700, avgFrom = 1300;
    const intensity = new Float64Array(H);
    run({ W, H, cur: st.cur, prev: st.prev, walls: st.walls }, {
        courant, gamma, absorb: true, planeWave: true, steps, onStep: (n, u0, u1) => {
            if (n < avgFrom) return;
            for (let y = 0; y < H; y++) { const v = u1[y * W + screenX]; intensity[y] += v * v; }
        },
    });

    // Local maxima away from the sponge, above 10% of the peak
    const lo = SPONGE + 4, hi = H - SPONGE - 4;
    let peak = 0;
    for (let y = lo; y < hi; y++) peak = Math.max(peak, intensity[y]);
    const maxima = [];
    for (let y = lo + 1; y < hi - 1; y++) {
        if (intensity[y] > intensity[y - 1] && intensity[y] >= intensity[y + 1] && intensity[y] > 0.1 * peak) maxima.push(y + 0.5);
    }

    // Grid wavelength with numerical dispersion along x: sin(ω/2) = C·sin(k/2), ω = 2π/period
    const lambda0 = courant * SOURCE_PERIOD;
    const k = 2 * Math.asin(Math.sin(Math.PI / SOURCE_PERIOD) / courant);
    const lambda = 2 * Math.PI / k;
    const mid = (centers[0] + centers[1]) / 2;
    const pathDiff = y => Math.hypot(L, y - centers[0]) - Math.hypot(L, y - centers[1]);
    // Predicted maximum of order m: solve pathDiff(y) = mλ by bisection
    const predict = m => {
        let a = mid, b = m > 0 ? H : 0;
        for (let i = 0; i < 60; i++) { const c = (a + b) / 2; (Math.abs(pathDiff(c)) < Math.abs(m) * lambda) ? (a = c) : (b = c); }
        return (a + b) / 2;
    };
    const errors = [];
    for (const m of [-2, -1, 0, 1, 2]) {
        const y = m === 0 ? mid : predict(m);
        if (y < lo || y > hi) continue;
        const nearest = maxima.reduce((best, v) => Math.abs(v - y) < Math.abs(best - y) ? v : best, -1e9);
        errors.push({ m, predicted: y, measured: nearest, err: nearest - y });
    }
    const spacing = (errors.find(e => e.m === 1).measured - errors.find(e => e.m === -1).measured) / 2;
    t.diagnostic(`slits at y=${centers.join(', ')} (d=${d}), screen L=${L}, λ=C·period=${lambda0} (grid λ=${lambda.toFixed(3)})`);
    t.diagnostic(`maxima at y=${maxima.join(', ')}`);
    t.diagnostic(`order m: predicted -> measured: ${errors.map(e => `${e.m}: ${e.predicted.toFixed(1)} -> ${e.measured} (${e.err.toFixed(1)})`).join('; ')}`);
    t.diagnostic(`first-order spacing ${spacing.toFixed(1)} cells vs small-angle λL/d = ${(lambda * L / d).toFixed(1)}`);
    assert.ok(maxima.length >= 3, 'multiple interference maxima');
    // Measured maxima sit ~4.5% further off-axis than point sources at the wall exit predict (the
    // 8-cell slits in a 3-cell-thick wall are not point sources and the screen is at L ≈ 1.1·d²/λ,
    // not deep in the far field). Tolerance: 1 cell + 6% of the off-axis distance per order, and
    // the first-order spacing within 10% of the small-angle λL/d (measured 7%).
    for (const e of errors) {
        assert.ok(Math.abs(e.err) <= 1 + 0.06 * Math.abs(e.predicted - mid), `order ${e.m} off by ${e.err}`);
    }
    assert.ok(Math.abs(spacing - lambda * L / d) / (lambda * L / d) < 0.10);
});
