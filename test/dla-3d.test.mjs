// 3D diffusion-limited aggregation (js/examples/dla-3d.js).
//
// Real code under test: DLA3D.initializeParticles / initializeGrid / setupBuffers (gridSize u32
// write) / updateParams / updateUniforms through a mock device, Mat4 and Camera (exported for tests).
//
// CPU replica of PARTICLE_UPDATE_SHADER `main` (pcgHash, nextRandom, spawnPosition, gridIndex,
// isOccupied, hasNeighbor and the walk/stick rule); validates the algorithm and index math, not the
// WGSL itself — see #51. Walkers are processed one after another, one valid ordering of the GPU's
// atomic grid accesses. References: breadth-first search for 26-connectivity of the aggregate,
// the render shader's instance-index decomposition, the mass-radius fractal dimension of DLA.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Mat4, Camera, DLA3D, wgslModules } from '../js/examples/dla-3d.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const f32 = Math.fround;

// ---- replica ----
function pcgHash(v) {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> (((state >>> 28) + 4) >>> 0)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
function nextRandom(st) {
    st.s = pcgHash(st.s);
    return f32((st.s >>> 8) / 16777216);
}
function spawnPosition(st, gridSize) {
    const size = gridSize, center = size / 2, R = f32(size * 0.45);
    const theta = f32(nextRandom(st) * 6.28318);
    const cosPhi = f32(nextRandom(st) * 2 - 1);
    const sinPhi = f32(Math.sqrt(1 - cosPhi * cosPhi));
    return [f32(center + R * sinPhi * Math.cos(theta)), f32(center + R * sinPhi * Math.sin(theta)), f32(center + R * cosPhi)];
}
const NOT_FOUND = 0xffffffff;
function gridIndex(p, size) {
    if (p[0] < 0 || p[0] >= size || p[1] < 0 || p[1] >= size || p[2] < 0 || p[2] >= size) return NOT_FOUND;
    return p[2] * size * size + p[1] * size + p[0];
}
const isOccupied = (grid, p, size) => { const i = gridIndex(p, size); return i !== NOT_FOUND && grid[i] > 0; };
function hasNeighbor(grid, p, size) {
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        if (isOccupied(grid, [p[0] + dx, p[1] + dy, p[2] + dz], size)) return true;
    }
    return false;
}
// One invocation of main for walker idx; particles is the shipped Float32Array/Uint32Array pair.
// Returns the stuck cell (and records it) or null.
function walkerStep(pf, pu, idx, grid, size, stickiness) {
    let rng = pu[4 * idx + 3];
    if (rng === 0) return null;
    const st = { s: rng };
    const axis = nextRandom(st);
    const step = nextRandom(st) > 0.5 ? 1 : -1;
    const dir = [0, 0, 0];
    if (axis < f32(0.333)) dir[0] = step; else if (axis < f32(0.666)) dir[1] = step; else dir[2] = step;
    const cur = [Math.trunc(pf[4 * idx]), Math.trunc(pf[4 * idx + 1]), Math.trunc(pf[4 * idx + 2])];
    let np = cur.map((c, k) => c + dir[k]);
    if (np.some(c => c < 0 || c >= size)) {
        pf.set(spawnPosition(st, size), 4 * idx);
    } else {
        if (isOccupied(grid, np, size)) np = cur;
        if (hasNeighbor(grid, np, size) && nextRandom(st) < f32(stickiness)) {
            grid[gridIndex(np, size)] = 1;
            pu[4 * idx + 3] = 0;
            return np;
        }
        pf.set(np, 4 * idx);
    }
    pu[4 * idx + 3] = Math.max(st.s, 1);
    return null;
}

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

function makeDLA(gridSize, numParticles) {
    const d = Object.create(DLA3D.prototype);
    Object.assign(d, {
        device: mockDevice(), canvas: { width: 800, height: 600 }, gridSize, numParticles,
        stickiness: 0.5, time: 0, aggregatedCount: 0, camera: new Camera(),
    });
    d.camera.target = [gridSize / 2, gridSize / 2, gridSize / 2];
    d.camera.distance = gridSize * 1.5;
    return d;
}

// ---------------------------------------------------------------------------------------------

test('PCG random stream: values in [0, 1), uniform, deterministic', (t) => {
    const st = { s: 12345 };
    const xs = Array.from({ length: 50000 }, () => nextRandom(st));
    assert.ok(xs.every(x => x >= 0 && x < 1));
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const bins = new Array(10).fill(0);
    xs.forEach(x => bins[Math.floor(x * 10)]++);
    const chi2 = bins.reduce((s, b) => s + (b - 5000) ** 2 / 5000, 0);
    t.diagnostic(`mean ${mean.toFixed(4)}, χ² (9 dof) over 10 bins ${chi2.toFixed(2)}`);
    assert.ok(Math.abs(mean - 0.5) < 0.01);
    assert.ok(chi2 < 27.9, 'χ²₉ 99.9th percentile');
    assert.equal(pcgHash(0), pcgHash(0));
    assert.notEqual(pcgHash(1), pcgHash(2));
});

test('gridIndex matches the render shader decomposition and initializeGrid indexing', () => {
    const size = 13;
    for (let i = 0; i < size ** 3; i++) {
        const z = Math.floor(i / (size * size)), y = Math.floor(i / size) % size, x = i % size;   // render shader
        assert.equal(gridIndex([x, y, z], size), i);
    }
    for (const p of [[-1, 0, 0], [0, size, 0], [0, 0, size], [size, 0, 0], [0, -1, 5]]) assert.equal(gridIndex(p, size), NOT_FOUND);
});

test('initializeGrid seeds: center, line, circle, multiple', () => {
    const size = 64, c = 32;
    const expect = {
        center: [[c, c, c]],
        line: Array.from({ length: 21 }, (_, i) => [c - 10 + i, c, c]),
        circle: (() => { const r = []; for (let y = -5; y <= 5; y++) for (let x = -5; x <= 5; x++) if (x * x + y * y <= 25) r.push([c + x, c + y, c]); return r; })(),
        multiple: [[5, 5, 5], [-5, -5, -5], [5, -5, 5], [-5, 5, -5]].map(([x, y, z]) => [c + x, c + y, c + z]),
    };
    for (const [seed, cells] of Object.entries(expect)) {
        const d = makeDLA(size, 1);
        d.initializeGrid(seed);
        const got = [];
        d.grid.forEach((v, i) => { if (v) got.push(i); });
        assert.deepEqual(got, cells.map(p => gridIndex(p, size)).sort((a, b) => a - b), seed);
        assert.equal(d.aggregatedCount, cells.length, seed);
    }
});

test('initializeParticles: Particle layout (vec3 position, u32 rngState @12), live walkers on the spawn sphere', () => {
    const layout = structLayout(moduleCode(mod, 'particle'), 'Particle');
    assert.deepEqual(layout.offsets, { position: 0, rngState: 12 });
    assert.equal(layout.size, 16);
    const d = makeDLA(32, 500);
    withSeededRandom(3, () => d.initializeParticles());
    assert.equal(d.particles.byteLength, 500 * layout.size);
    for (let i = 0; i < 500; i++) {
        const p = decodeStruct(d.particles, layout, i * 16);
        assert.ok(p.rngState >= 1 && Number.isInteger(p.rngState), 'rngState is a nonzero u32');
        const r = Math.hypot(...p.position.map(x => x - 16));
        assert.ok(Math.abs(r - 32 * 0.45) < 1e-4);
    }
});

test('spawnPosition replica: uniform on the sphere of radius 0.45·size', (t) => {
    const st = { s: 99 };
    let sumZ = 0, sumZ2 = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
        const p = spawnPosition(st, 64);
        const r = Math.hypot(p[0] - 32, p[1] - 32, p[2] - 32);
        assert.ok(Math.abs(r - 28.8) < 1e-3);
        const z = (p[2] - 32) / 28.8;
        sumZ += z; sumZ2 += z * z;
    }
    t.diagnostic(`mean cos φ ${(sumZ / n).toFixed(4)} (0), mean cos² φ ${(sumZ2 / n).toFixed(4)} (1/3)`);
    assert.ok(Math.abs(sumZ / n) < 0.02 && Math.abs(sumZ2 / n - 1 / 3) < 0.02);
});

test('walk/stick replica: the aggregate stays 26-connected to the seed; DLA fractal dimension', (t) => {
    const size = 32, n = 1200;
    const d = makeDLA(size, n);
    withSeededRandom(17, () => { d.initializeParticles(); d.initializeGrid('center'); });
    const pf = d.particles, pu = new Uint32Array(pf.buffer);
    const grid = d.grid;
    let stuck = 0, dispatches = 0, violations = 0;
    const MAX_DISPATCH = 200000;
    while (stuck < n && dispatches < MAX_DISPATCH) {
        for (let i = 0; i < n; i++) {
            const cell = walkerStep(pf, pu, i, grid, size, d.stickiness);
            if (cell) {
                stuck++;
                // stick rule: the cell had an occupied 26-neighbour when it froze
                grid[gridIndex(cell, size)] = 0;
                if (!hasNeighbor(grid, cell, size)) violations++;
                grid[gridIndex(cell, size)] = 1;
            }
        }
        dispatches++;
        // live walkers stay inside the volume
        if (dispatches % 1000 === 0) {
            for (let i = 0; i < n; i++) if (pu[4 * i + 3]) assert.ok([0, 1, 2].every(k => pf[4 * i + k] >= 0 && pf[4 * i + k] < size));
        }
    }
    assert.equal(stuck, n, `all walkers stick within ${MAX_DISPATCH} dispatches`);
    assert.equal(violations, 0);
    assert.ok(grid.every(v => v === 0 || v === 1));
    // BFS over 26-neighbours from the seed reaches every occupied cell
    const c = size >> 1;
    const seed = gridIndex([c, c, c], size);
    const seen = new Uint8Array(grid.length);
    const queue = [seed];
    seen[seed] = 1;
    while (queue.length) {
        const i = queue.pop();
        const x = i % size, y = Math.floor(i / size) % size, z = Math.floor(i / (size * size));
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
            const j = gridIndex([x + dx, y + dy, z + dz], size);
            if (j !== NOT_FOUND && grid[j] && !seen[j]) { seen[j] = 1; queue.push(j); }
        }
    }
    let occupied = 0, unreached = 0;
    const radii = [];
    grid.forEach((v, i) => {
        if (!v) return;
        occupied++;
        if (!seen[i]) unreached++;
        const x = i % size, y = Math.floor(i / size) % size, z = Math.floor(i / (size * size));
        radii.push(Math.hypot(x - c, y - c, z - c));
    });
    assert.equal(unreached, 0, 'every stuck cell is 26-connected to the seed');
    // mass-radius dimension: least squares of log N(<r) vs log r over r ∈ [2, R_g]
    const rg = Math.sqrt(radii.reduce((s, r) => s + r * r, 0) / radii.length);
    const xs = [], ys = [];
    for (let r = 2; r <= rg; r += 0.5) { xs.push(Math.log(r)); ys.push(Math.log(radii.filter(q => q < r).length)); }
    const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
    const D = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
    t.diagnostic(`grid ${size}³, ${n} walkers: ${occupied} occupied cells after ${dispatches} dispatches; `
        + `R_g ${rg.toFixed(2)}; mass-radius dimension D ≈ ${D.toFixed(2)} (3D DLA ≈ 2.5, solid ball = 3)`);
    assert.ok(occupied >= n * 0.95 + 1 && occupied <= n + 1, 'nearly every walker adds a new cell');
    // small aggregate on a lattice: wide band, but clearly below a compact ball and above a chain
    assert.ok(D > 1.8 && D < 2.85, `D = ${D}`);
});

test('uniforms: Params, gridSize (u32) and Uniforms bytes match the WGSL', () => {
    const d = makeDLA(64, 5000);
    d.stickiness = 0.35; d.time = 1.25;
    withSeededRandom(1, () => { d.initializeParticles(); d.initializeGrid('center'); });
    d.setupBuffers();
    const gs = d.device.lastWrite(d.gridSizeBuffer).bytes;
    assert.equal(gs.byteLength, 4);
    assert.equal(new DataView(gs.buffer).getUint32(0, true), 64, '`var<uniform> gridSize: u32` gets the integer');
    assert.match(moduleCode(mod, 'render'), /var<uniform>\s+gridSize\s*:\s*u32/);

    d.updateParams();
    const p = decodeStruct(d.device.lastWrite(d.paramsBuffer).bytes, structLayout(moduleCode(mod, 'particle'), 'Params'));
    assert.deepEqual(p, { numParticles: 5000, gridSize: 64, stickiness: f32(0.35), time: 1.25 });

    d.updateUniforms();
    const u = decodeStruct(d.device.lastWrite(d.uniformBuffer).bytes, structLayout(moduleCode(mod, 'render'), 'Uniforms'));
    const ref = Mat4.multiply(Mat4.perspective(Math.PI / 4, 800 / 600, 0.1, 500), d.camera.getViewMatrix());
    assert.deepEqual(u.viewProj.flat(), Array.from(ref));
});

test('view-projection: grid center at screen center, camera at its orbit distance', (t) => {
    const apply = (M, v) => [0, 1, 2, 3].map(r => M[r] * v[0] + M[4 + r] * v[1] + M[8 + r] * v[2] + M[12 + r] * v[3]);
    const d = makeDLA(64, 1);
    const vp = Mat4.multiply(Mat4.perspective(Math.PI / 4, 800 / 600, 0.1, 500), d.camera.getViewMatrix());
    const c = apply(vp, [32, 32, 32, 1]);
    assert.ok(Math.abs(c[0] / c[3]) < 1e-5 && Math.abs(c[1] / c[3]) < 1e-5 && c[2] / c[3] > 0 && c[2] / c[3] < 1);
    const v = apply(d.camera.getViewMatrix(), [32, 32, 32, 1]);
    assert.ok(Math.abs(v[2] + 96) < 1e-3, 'target lies 1.5·gridSize in front of the eye');
    // the spawn sphere (radius 0.45·64 = 28.8) fits in the view at the default distance
    let maxNdc = 0;
    for (let k = 0; k < 500; k++) {
        const th = k * 2.39996, z = 1 - 2 * (k + 0.5) / 500, s = Math.sqrt(1 - z * z);
        const q = apply(vp, [32 + 28.8 * s * Math.cos(th), 32 + 28.8 * s * Math.sin(th), 32 + 28.8 * z, 1]);
        maxNdc = Math.max(maxNdc, Math.abs(q[0] / q[3]), Math.abs(q[1] / q[3]));
    }
    t.diagnostic(`spawn sphere reaches |NDC| ≤ ${maxNdc.toFixed(3)} at the default camera`);
    assert.ok(maxNdc < 1);
});
