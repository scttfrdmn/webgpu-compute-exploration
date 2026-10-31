// Molecular dynamics, Lennard-Jones (js/examples/molecular-dynamics.js).
//
// Real code under test: MolecularDynamics.initializeAtoms / updateParams / updateUniforms /
// setupBoxPipelines (BoxParams write) through a mock device, Mat4 and Camera (exported for tests).
//
// CPU replica of the compute shader `main` (computeForce, minImage, applyPBC, the integrator and the
// thermostat); validates the algorithm and index math, not the WGSL itself — see #51.
// References: −dU/dr of U(r) = 4ε[(σ/r)¹² − (σ/r)⁶] by central finite differences, nearest image by
// brute force over the 27 periodic images, Newton's third law.
//
// History: this suite found that computeForce had its sign inverted (r = x_j − x_i, so atoms attracted
// at short range). Fixed in #64: the shader now uses r = x_i − x_j, matching the physical LJ force
// F_i = 24ε(2(σ/r)¹² − (σ/r)⁶)/r² · (x_i − x_j).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Mat4, Camera, MolecularDynamics, wgslModules } from '../js/examples/molecular-dynamics.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const f32 = Math.fround;

// ---- replica of the shader ----
const EPSILON = 1.0, SIGMA = 1.0, CUTOFF_SQ = 6.25;

// computeForce(r), r = x_i − x_j, exactly as in the shader (float64 arithmetic)
function computeForce(r) {
    const d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (d2 > CUTOFF_SQ || d2 < 0.01) return [0, 0, 0];
    const dist = Math.sqrt(d2);
    const sr6 = Math.pow(SIGMA / dist, 6);
    const sr12 = sr6 * sr6;
    const m = 24 * EPSILON * (2 * sr12 - sr6) / d2;
    return [m * r[0], m * r[1], m * r[2]];
}
function wrap(v, L) {   // applyPBC and minImage are the same per-component rule
    const half = L * 0.5;
    return v.map(c => (c > half ? c - L : c < -half ? c + L : c));
}
const applyPBC = wrap, minImage = wrap;

// One dispatch of `main` over all atoms (positions [n][3], velocities [n][3]); `thermostat` false
// drops the velocity rescale so the integrator core can be inspected on its own.
function stepReplica(pos, vel, { dt, L, T, thermostat = true }) {
    const n = pos.length;
    const outP = [], outV = [];
    for (let i = 0; i < n; i++) {
        const F = [0, 0, 0];
        for (let j = 0; j < n; j++) {
            if (j === i) continue;
            const r = minImage([0, 1, 2].map(c => pos[i][c] - pos[j][c]), L);
            const f = computeForce(r);
            F[0] += f[0]; F[1] += f[1]; F[2] += f[2];
        }
        let v = vel[i].map((c, k) => c + F[k] * dt);
        const p = applyPBC(pos[i].map((c, k) => c + v[k] * dt), L);
        const speed = Math.hypot(...v);
        if (thermostat && speed > 0) {
            const target = Math.sqrt(T * 0.01);
            v = v.map(c => c * (1 - 0.01 * (1 - target / speed)));
        }
        outP.push(p); outV.push(v);
    }
    return { pos: outP, vel: outV };
}

const U = r => 4 * EPSILON * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6);

// ---------------------------------------------------------------------------------------------

test('LJ force magnitude equals |dU/dr| (finite differences), cutoffs at r = 2.5σ and r² < 0.01', (t) => {
    let maxRel = 0;
    for (let r = 0.85; r < 2.5; r += 0.0137) {
        const dir = [0.48, -0.6, 0.64];   // unit vector
        const f = computeForce(dir.map(c => c * r));
        const h = 1e-6 * r;
        const dUdr = (U(r + h) - U(r - h)) / (2 * h);
        // shader convention: force on i is force_mag·r with r = x_i − x_j, so F_i·r̂ = −dU/dr
        const along = f[0] * dir[0] + f[1] * dir[1] + f[2] * dir[2];
        const perp = Math.hypot(f[0] - along * dir[0], f[1] - along * dir[1], f[2] - along * dir[2]);
        maxRel = Math.max(maxRel, Math.abs(Math.abs(along) - Math.abs(dUdr)) / Math.max(Math.abs(dUdr), 1e-3));
        assert.ok(perp < 1e-9 * Math.max(1, Math.abs(along)));
    }
    t.diagnostic(`max relative error |F| vs |dU/dr| over r ∈ [0.85, 2.5): ${maxRel.toExponential(2)}`);
    assert.ok(maxRel < 1e-6, 'central differences with h = 1e-6 r are accurate to ~1e-8');
    // zero force at the potential minimum r = 2^{1/6}σ
    const rmin = 2 ** (1 / 6);
    assert.ok(Math.abs(computeForce([rmin, 0, 0])[0]) < 1e-12);
    assert.deepEqual(computeForce([2.51, 0, 0]), [0, 0, 0]);
    assert.deepEqual(computeForce([0.099, 0, 0]), [0, 0, 0]);
    assert.notDeepEqual(computeForce([2.49, 0, 0]), [0, 0, 0]);
});

test('shader uses the same r = x_i − x_j convention as this replica', () => {
    // Keeps the replica honest: if the shader's direction changes again, this fails.
    const compute = wgslModules().map(m => m.code).join('\n');
    assert.match(compute, /var r = atom\.position - atomsIn\[j\]\.position;/);
});

test('LJ force is repulsive inside r_min = 2^{1/6}σ and attractive outside',
    () => {
        // r = x_i − x_j = +x (atom i sits on the +x side of j): repulsion pushes i further toward +x
        assert.ok(computeForce([1.0, 0, 0])[0] > 0, 'r = 1.0σ: force on i must point away from j');
        assert.ok(computeForce([1.5, 0, 0])[0] < 0, 'r = 1.5σ: force on i must point toward j');
    });

test('pair forces with minimum image obey Newton\'s third law (total force = 0)', (t) => {
    const L = 6;
    let s = 12345;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
    const pos = Array.from({ length: 40 }, () => [0, 1, 2].map(() => (rnd() - 0.5) * L));
    const F = [0, 0, 0];
    let maxF = 0;
    for (let i = 0; i < pos.length; i++) for (let j = 0; j < pos.length; j++) {
        if (i === j) continue;
        const f = computeForce(minImage([0, 1, 2].map(c => pos[i][c] - pos[j][c]), L));
        for (let c = 0; c < 3; c++) F[c] += f[c];
        maxF = Math.max(maxF, Math.hypot(...f));
    }
    const net = Math.hypot(...F);
    t.diagnostic(`40 atoms in a periodic box of ${L}: |Σ F| = ${net.toExponential(2)} (largest pair force ${maxF.toExponential(2)})`);
    assert.ok(net < 1e-9 * Math.max(1, maxF));
});

test('minImage = nearest periodic image (brute force over 27 images); applyPBC keeps atoms in the box', (t) => {
    const L = 30;
    let s = 99;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    for (let k = 0; k < 2000; k++) {
        const a = [0, 1, 2].map(() => (rnd() - 0.5) * L), b = [0, 1, 2].map(() => (rnd() - 0.5) * L);
        const r = minImage([0, 1, 2].map(c => b[c] - a[c]), L);
        let best = Infinity;
        for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let l = -1; l <= 1; l++) {
            best = Math.min(best, Math.hypot(b[0] - a[0] + i * L, b[1] - a[1] + j * L, b[2] - a[2] + l * L));
        }
        assert.ok(Math.abs(Math.hypot(...r) - best) < 1e-9);
        // a step smaller than half the box from inside stays inside after applyPBC
        const p = applyPBC(a.map(c => c + (rnd() - 0.5) * L * 0.9), L);
        assert.ok(p.every(c => c >= -L / 2 && c <= L / 2));
    }
    t.diagnostic('2000 random pairs: minImage distance equals the nearest of 27 images');
});

test('integrator replica: semi-implicit Euler drift, thermostat relaxes speed to √(0.01 T)', (t) => {
    const dt = 0.016, L = 30, T = 30;
    // one isolated atom: no forces; x advances by the *updated* velocity times dt
    let state = { pos: [[1, 2, 3]], vel: [[0.4, 0, 0]] };
    const next = stepReplica(state.pos, state.vel, { dt, L, T, thermostat: false });
    assert.deepEqual(next.pos[0], [1 + 0.4 * dt, 2, 3]);
    // thermostat: speed → target geometrically, rate 1 % per step
    const target = Math.sqrt(T * 0.01);
    for (let k = 0; k < 1500; k++) state = stepReplica(state.pos, state.vel, { dt, L, T });
    const speed = Math.hypot(...state.vel[0]);
    const expected = target + (0.4 - target) * 0.99 ** 1500;
    t.diagnostic(`after 1500 steps: speed ${speed.toFixed(6)}, target √(0.01·T) = ${target.toFixed(6)}`);
    assert.ok(Math.abs(speed - expected) < 1e-9);
    assert.ok(Math.abs(speed - target) < 1e-3);
    // the atom wrapped around the periodic box and stays inside it
    assert.ok(state.pos[0].every(c => c >= -L / 2 && c <= L / 2));
});

test('energy of a small LJ cluster is conserved by the integrator core (thermostat off)',
    (t) => {
        // 8 atoms on a cube of side 1.12σ (near r_min), small random velocities, dt = 0.002
        const L = 30, dt = 0.002;
        let pos = [], vel = [];
        let s = 7;
        const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5);
        for (let i = 0; i < 8; i++) {
            pos.push([(i & 1) * 1.12, ((i >> 1) & 1) * 1.12, ((i >> 2) & 1) * 1.12]);
            vel.push([rnd() * 0.1, rnd() * 0.1, rnd() * 0.1]);
        }
        const energy = () => {
            let E = 0;
            for (let i = 0; i < 8; i++) {
                E += 0.5 * (vel[i][0] ** 2 + vel[i][1] ** 2 + vel[i][2] ** 2);
                for (let j = i + 1; j < 8; j++) {
                    const r = Math.hypot(...minImage([0, 1, 2].map(c => pos[j][c] - pos[i][c]), L));
                    if (r * r <= CUTOFF_SQ) E += U(r);
                }
            }
            return E;
        };
        const E0 = energy();
        let maxErr = 0;
        for (let k = 0; k < 2000; k++) {
            ({ pos, vel } = stepReplica(pos, vel, { dt, L, T: 0, thermostat: false }));
            maxErr = Math.max(maxErr, Math.abs(energy() - E0) / Math.abs(E0));
        }
        t.diagnostic(`max |ΔE/E₀| over 2000 steps: ${maxErr.toExponential(2)}`);
        assert.ok(maxErr < 1e-2);
    });

// ---- shipped JS ----

function withSeededRandom(seed, fn) {
    const saved = Math.random;
    let a = seed >>> 0;
    Math.random = () => {   // mulberry32
        a = (a + 0x6d2b79f5) >>> 0;
        let t = Math.imul(a ^ (a >>> 15), a | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    try { return fn(); } finally { Math.random = saved; }
}

function makeSim(numAtoms, boxSize) {
    const md = Object.create(MolecularDynamics.prototype);
    Object.assign(md, {
        device: mockDevice(), canvas: { width: 960, height: 540 }, numAtoms, boxSize, temperature: 42,
        camera: new Camera(), paramsBuffer: { label: 'params' }, uniformBuffer: { label: 'uniforms' },
    });
    return md;
}

test('initializeAtoms: lattice inside the box, Atom stride matches WGSL (vec3 position @0, velocity @16)', (t) => {
    const atomLayout = structLayout(moduleCode(mod, 'compute'), 'Atom');
    assert.deepEqual(atomLayout.offsets, { position: 0, velocity: 16 });
    assert.equal(atomLayout.size, 32);   // initializeAtoms uses 8 floats per atom
    for (const n of [1000, 27, 100]) {
        const md = makeSim(n, 30);
        withSeededRandom(1, () => md.initializeAtoms());
        assert.equal(md.atoms.byteLength, n * atomLayout.size);
        const seen = new Set();
        let minSep = Infinity;
        const P = [];
        for (let i = 0; i < n; i++) {
            const a = decodeStruct(md.atoms, atomLayout, i * atomLayout.size);
            assert.ok(a.position.every(c => c > -15 && c < 15), `atom ${i} outside the box`);
            assert.ok(a.velocity.every(c => Math.abs(c) <= 0.5 * 42 * 0.01 + 1e-6));
            // every velocity component is drawn (nonzero) and the vec3 padding words stay zero
            assert.ok(a.velocity.every(c => c !== 0), `atom ${i}: velocity ${a.velocity}`);
            assert.equal(md.atoms[8 * i + 3], 0, `atom ${i}: position padding`);
            assert.equal(md.atoms[8 * i + 7], 0, `atom ${i}: velocity padding`);
            seen.add(a.position.join(','));
            P.push(a.position);
        }
        assert.equal(seen.size, n, 'no two atoms share a lattice site');
        for (let i = 0; i < Math.min(n, 200); i++) for (let j = i + 1; j < n; j++) {
            minSep = Math.min(minSep, Math.hypot(...minImage(P[j].map((c, k) => c - P[i][k]), 30)));
        }
        t.diagnostic(`N=${n}: nearest-neighbour spacing ${minSep.toFixed(3)}σ (cutoff 2.5σ)`);
        assert.ok(minSep > 1.0, 'no overlapping atoms at start');
    }
});

test('uniforms: Params, Uniforms and BoxParams bytes match the WGSL structs', () => {
    const md = makeSim(1234, 30);
    md.updateParams();
    const p = decodeStruct(md.device.lastWrite(md.paramsBuffer).bytes, structLayout(moduleCode(mod, 'compute'), 'Params'));
    assert.deepEqual(p, { numAtoms: 1234, deltaTime: f32(0.016), boxSize: 30, temperature: 42 });

    md.updateUniforms();
    const uLayout = structLayout(moduleCode(mod, 'render'), 'Uniforms');
    assert.deepEqual(structLayout(moduleCode(mod, 'box'), 'Uniforms').offsets, uLayout.offsets);
    const u = decodeStruct(md.device.lastWrite(md.uniformBuffer).bytes, uLayout);
    const ref = Mat4.multiply(Mat4.perspective(Math.PI / 4, 960 / 540, 0.1, 200), md.camera.getViewMatrix());
    assert.deepEqual(u.viewProj.flat(), Array.from(ref));

    md.setupBoxPipelines();
    const b = decodeStruct(md.device.lastWrite(md.boxParamsBuffer).bytes, structLayout(moduleCode(mod, 'box'), 'BoxParams'));
    assert.deepEqual(b, { halfSize: 15, edgeHalfWidth: 3, viewport: [960, 540] });
});

test('view-projection: camera frames the whole box; Mat4 matches hand-computed transforms', (t) => {
    const apply = (M, v) => [0, 1, 2, 3].map(r => M[r] * v[0] + M[4 + r] * v[1] + M[8 + r] * v[2] + M[12 + r] * v[3]);
    const cam = new Camera();
    const vp = Mat4.multiply(Mat4.perspective(Math.PI / 4, 16 / 9, 0.1, 200), cam.getViewMatrix());
    const c = apply(vp, [0, 0, 0, 1]);
    assert.ok(Math.abs(c[0] / c[3]) < 1e-6 && Math.abs(c[1] / c[3]) < 1e-6);
    let maxNdc = 0;
    for (let i = 0; i < 8; i++) {
        const corner = [(i & 1) ? 15 : -15, (i & 2) ? 15 : -15, (i & 4) ? 15 : -15, 1];
        const q = apply(vp, corner);
        const ndc = [q[0] / q[3], q[1] / q[3], q[2] / q[3]];
        maxNdc = Math.max(maxNdc, Math.abs(ndc[0]), Math.abs(ndc[1]));
        assert.ok(q[3] > 0 && ndc[2] > 0 && ndc[2] < 1, 'corner between near and far planes');
    }
    t.diagnostic(`default camera: box corners reach |NDC| ≤ ${maxNdc.toFixed(3)} (inside the view when < 1)`);
    assert.ok(maxNdc < 1, '"distance 75 frames the whole 30-unit box"');
    // eye sits at the camera distance from the target, looking at it
    const V = cam.getViewMatrix();
    const tgt = apply(V, [0, 0, 0, 1]);
    assert.ok(Math.abs(tgt[2] + cam.distance) < 1e-4 && Math.hypot(tgt[0], tgt[1]) < 1e-4);
});
