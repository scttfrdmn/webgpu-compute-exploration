// N-body gravity (js/examples/nbody.js).
//
// Real code under test: generateBodies / PRESETS, makeRng, dispatchGrid, Mat4, OrbitCamera,
// spriteStyle, and the uniform writers NBodySimulation.writeSimParams / writeCamera / the RadixParams
// slots written in allocate() (decoded against the WGSL structs from wgslModules()).
//
// CPU replica of cs_morton, cs_radix_histogram/scan/scatter, cs_init_nodes, cs_build_tree,
// cs_bottom_up, cs_bh_force, cs_brute_force and cs_kick_drift/cs_kick; validates the algorithm and
// index math, not the WGSL itself — see #51. The replicas follow the shaders' node numbering
// (internal 0 … N − 2, leaf of sorted body k at N − 1 + k), Karras δ with the index tiebreak, the
// opening rule d < l/θ + δ (or body inside the box), the 64-entry stack and the MAX_VISITS cap.
// References: direct O(N²) float64 sums, a naive bit-interleave Morton code, Array.prototype.sort.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    WG, NONE, STACK_SIZE, MAX_VISITS, MAX_DEPTH, RADIX_PASSES, RADIX_TILE, TRAIL_LEN, PRESETS,
    makeRng, generateBodies, dispatchGrid, Mat4, OrbitCamera, FOV, spriteStyle, NBodySimulation,
    wgslModules,
} from '../js/examples/nbody.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const code = name => wgslModules().find(m => m.name === name).code;
const f32 = Math.fround;

// ---------------------------------------------------------------------------------------------
// Direct-sum reference (float64)
function directForces(pos, n, eps2) {
    const acc = new Float64Array(3 * n), phi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const xi = pos[4 * i], yi = pos[4 * i + 1], zi = pos[4 * i + 2];
        let ax = 0, ay = 0, az = 0, p = 0;
        for (let j = 0; j < n; j++) {
            if (j === i) continue;
            const dx = pos[4 * j] - xi, dy = pos[4 * j + 1] - yi, dz = pos[4 * j + 2] - zi;
            const invR = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
            const mInvR = pos[4 * j + 3] * invR;
            p -= mInvR;
            const s = mInvR * invR * invR;
            ax += dx * s; ay += dy * s; az += dz * s;
        }
        acc[3 * i] = ax; acc[3 * i + 1] = ay; acc[3 * i + 2] = az; phi[i] = p;
    }
    return { acc, phi };
}

// ---------------------------------------------------------------------------------------------
// Replica of cs_brute_force: 256-body tiles, zero-mass padding past N, self excluded by index,
// f32 accumulation (Math.fround after every operation the shader does in f32).
function bruteReplica(pos, n, eps2) {
    const numTiles = Math.ceil(n / WG);
    const acc = new Float32Array(4 * n);
    for (let i = 0; i < n; i++) {
        const px = pos[4 * i], py = pos[4 * i + 1], pz = pos[4 * i + 2];
        let ax = 0, ay = 0, az = 0, phi = 0;
        for (let t = 0; t < numTiles; t++) {
            for (let k = 0; k < WG; k++) {
                const j = t * WG + k;
                const bx = j < n ? pos[4 * j] : 0, by = j < n ? pos[4 * j + 1] : 0;
                const bz = j < n ? pos[4 * j + 2] : 0, bw = j < n ? pos[4 * j + 3] : 0;
                const dx = f32(bx - px), dy = f32(by - py), dz = f32(bz - pz);
                const invR = f32(1 / Math.sqrt(f32(f32(dx * dx + dy * dy + dz * dz) + eps2)));
                const mInvR = j === i ? 0 : f32(bw * invR);
                phi = f32(phi - mInvR);
                const s = f32(mInvR * invR * invR);
                ax = f32(ax + f32(dx * s)); ay = f32(ay + f32(dy * s)); az = f32(az + f32(dz * s));
            }
        }
        acc.set([ax, ay, az, phi], 4 * i);
    }
    return acc;
}

// ---------------------------------------------------------------------------------------------
// Replica of cs_morton
function expandBits(v) {
    let x = v & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    return x >>> 0;
}
function naiveInterleave(x, y, z) {
    let code = 0;
    for (let b = 9; b >= 0; b--) {
        code = code * 8 + ((x >> b) & 1) * 4 + ((y >> b) & 1) * 2 + ((z >> b) & 1);
    }
    return code >>> 0;
}
function boundsOf(pos, n) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], pos[4 * i + c]); hi[c] = Math.max(hi[c], pos[4 * i + c]);
    }
    return { lo, hi };
}
function mortonReplica(pos, n) {
    const { lo, hi } = boundsOf(pos, n);
    const side = Math.max(Math.max(hi[0] - lo[0], hi[1] - lo[1]), Math.max(hi[2] - lo[2], 1e-30));
    const scale = f32(1024 / f32(side));
    const keys = new Uint32Array(n), vals = new Uint32Array(n), cells = [];
    for (let i = 0; i < n; i++) {
        const q = [0, 1, 2].map(c => Math.min(Math.floor(Math.max(f32(f32(pos[4 * i + c] - lo[c]) * scale), 0)), 1023));
        cells.push(q);
        keys[i] = ((expandBits(q[0]) << 2) | (expandBits(q[1]) << 1) | expandBits(q[2])) >>> 0;
        vals[i] = i;
    }
    return { keys, vals, cells };
}

// ---------------------------------------------------------------------------------------------
// Replica of the radix sort: per pass histogram (digit-major offsets[d·numTiles + tile]),
// exclusive scan, and the stable scatter in 4 rounds of 256 with 16-bit packed rank counters.
function radixReplica(keysIn, valsIn) {
    const n = keysIn.length;
    const numTiles = Math.ceil(n / RADIX_TILE);
    let keys = Uint32Array.from(keysIn), vals = Uint32Array.from(valsIn);
    for (let pass = 0; pass < RADIX_PASSES; pass++) {
        const shift = 4 * pass;
        const digitOf = k => (k >>> shift) & 15;
        const offsets = new Uint32Array(16 * numTiles);
        for (let tile = 0; tile < numTiles; tile++) {
            for (let r = 0; r < RADIX_TILE; r++) {
                const idx = tile * RADIX_TILE + r;
                if (idx < n) offsets[digitOf(keys[idx]) * numTiles + tile]++;
            }
        }
        let running = 0;
        for (let i = 0; i < offsets.length; i++) { const c = offsets[i]; offsets[i] = running; running += c; }
        const kOut = new Uint32Array(n), vOut = new Uint32Array(n);
        for (let tile = 0; tile < numTiles; tile++) {
            const tileBase = Array.from({ length: 16 }, (_, d) => offsets[d * numTiles + tile]);
            for (let round = 0; round < RADIX_TILE / WG; round++) {
                // packed counters: 8 words, digit d in word d>>1 at bit (d&1)*16
                const prefix = new Uint32Array(8);   // exclusive scan as threads go in order
                for (let lid = 0; lid < WG; lid++) {
                    const idx = tile * RADIX_TILE + round * WG + lid;
                    if (idx >= n) continue;
                    const d = digitOf(keys[idx]);
                    const rank = (prefix[d >> 1] >>> ((d & 1) * 16)) & 0xffff;
                    const dst = tileBase[d] + rank;
                    kOut[dst] = keys[idx]; vOut[dst] = vals[idx];
                    prefix[d >> 1] = (prefix[d >> 1] + (1 << ((d & 1) * 16))) >>> 0;
                }
                for (let d = 0; d < 16; d++) tileBase[d] += (prefix[d >> 1] >>> ((d & 1) * 16)) & 0xffff;
            }
        }
        keys = kOut; vals = vOut;
    }
    return { keys, vals };
}

// ---------------------------------------------------------------------------------------------
// Replica of cs_build_tree (Karras 2012) with SEARCH_CAP = 32
const clz32 = Math.clz32;
function buildTreeReplica(keys) {
    const n = keys.length;
    const children = new Uint32Array(2 * Math.max(n - 1, 1));
    const parents = new Uint32Array(2 * n - 1).fill(0xdeadbeef);
    parents[0] = NONE;
    const delta = (i, j) => {
        if (j < 0 || j >= n) return -1;
        if (keys[i] === keys[j]) return 32 + clz32((i ^ j) >>> 0);
        return clz32((keys[i] ^ keys[j]) >>> 0);
    };
    const CAP = 32;
    for (let i = 0; i + 1 < n; i++) {
        const d = delta(i, i + 1) > delta(i, i - 1) ? 1 : -1;
        const deltaMin = delta(i, i - d);
        let lMax = 2;
        for (let it = 0; it < CAP; it++) { if (delta(i, i + lMax * d) <= deltaMin) break; lMax *= 2; }
        let l = 0, t = Math.trunc(lMax / 2);
        for (let it = 0; it < CAP; it++) {
            if (t < 1) break;
            if (delta(i, i + (l + t) * d) > deltaMin) l += t;
            t = Math.trunc(t / 2);
        }
        const j = i + l * d;
        const deltaNode = delta(i, j);
        let s = 0, div = 2;
        for (let it = 0; it < CAP; it++) {
            const stride = Math.trunc((l + div - 1) / div);
            if (delta(i, i + (s + stride) * d) > deltaNode) s += stride;
            if (stride <= 1) break;
            div *= 2;
        }
        const gamma = i + s * d + Math.min(d, 0);
        const first = Math.min(i, j), last = Math.max(i, j);
        const left = first === gamma ? n - 1 + gamma : gamma;
        const right = last === gamma + 1 ? n - 1 + gamma + 1 : gamma + 1;
        children[2 * i] = left; children[2 * i + 1] = right;
        parents[left] = i; parents[right] = i;
    }
    return { children, parents };
}

// Replicas of cs_init_nodes + cs_bottom_up (threads run one after another, which is one valid
// interleaving of the atomic visit-counter protocol: the second visitor completes the node).
function bottomUpReplica(pos, vals, children, parents) {
    const n = vals.length;
    const nodes = 2 * n - 1;
    const mass = new Float64Array(4 * nodes);
    const box = new Float64Array(8 * nodes);
    for (let k = 0; k < n; k++) {
        const b = vals[k], leaf = n - 1 + k;
        for (let c = 0; c < 3; c++) { mass[4 * leaf + c] = pos[4 * b + c]; box[8 * leaf + c] = pos[4 * b + c]; box[8 * leaf + 4 + c] = pos[4 * b + c]; }
        mass[4 * leaf + 3] = pos[4 * b + 3];
    }
    const visits = new Uint32Array(Math.max(n - 1, 1));
    let completions = 0;
    for (let k = 0; k < n; k++) {
        let node = n - 1 + k;
        for (let level = 0; level < MAX_DEPTH; level++) {
            const p = parents[node];
            if (p === NONE) break;
            if (visits[p]++ === 0) break;
            const a = children[2 * p], b = children[2 * p + 1];
            const m = mass[4 * a + 3] + mass[4 * b + 3];
            const lo = [0, 1, 2].map(c => Math.min(box[8 * a + c], box[8 * b + c]));
            const hi = [0, 1, 2].map(c => Math.max(box[8 * a + 4 + c], box[8 * b + 4 + c]));
            const com = [0, 1, 2].map(c => m > 0 ? (mass[4 * a + c] * mass[4 * a + 3] + mass[4 * b + c] * mass[4 * b + 3]) / m
                : 0.5 * (mass[4 * a + c] + mass[4 * b + c]));
            const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
            const offset = Math.hypot(...[0, 1, 2].map(c => com[c] - 0.5 * (lo[c] + hi[c])));
            for (let c = 0; c < 3; c++) { box[8 * p + c] = lo[c]; box[8 * p + 4 + c] = hi[c]; mass[4 * p + c] = com[c]; }
            box[8 * p + 3] = size; box[8 * p + 7] = offset; mass[4 * p + 3] = m;
            completions++;
            node = p;
        }
    }
    return { mass, box, completions };
}

// Replica of cs_bh_force
function forceReplica(pos, vals, tree, theta, eps2) {
    const n = vals.length;
    const { mass, box, children } = tree;
    const firstLeaf = n - 1;
    const invTheta = 1 / theta;
    const acc = new Float64Array(3 * n), phi = new Float64Array(n), inter = new Float64Array(n);
    let maxVisits = 0, maxSp = 0;
    const stack = new Uint32Array(STACK_SIZE);
    for (let k = 0; k < n; k++) {
        const body = vals[k];
        const px = pos[4 * body], py = pos[4 * body + 1], pz = pos[4 * body + 2];
        const myLeaf = n - 1 + k;
        stack[0] = 0;
        let sp = 1, ax = 0, ay = 0, az = 0, ph = 0, count = 0, it = 0;
        const add = (node) => {
            const dx = mass[4 * node] - px, dy = mass[4 * node + 1] - py, dz = mass[4 * node + 2] - pz;
            const invR = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
            const mInvR = mass[4 * node + 3] * invR;
            const s = mInvR * invR * invR;
            ax += dx * s; ay += dy * s; az += dz * s; ph -= mInvR; count++;
        };
        for (it = 0; it < MAX_VISITS; it++) {
            if (sp === 0) break;
            const node = stack[--sp];
            if (mass[4 * node + 3] <= 0) continue;
            if (node >= firstLeaf) { if (node !== myLeaf) add(node); continue; }
            const dx = mass[4 * node] - px, dy = mass[4 * node + 1] - py, dz = mass[4 * node + 2] - pz;
            const rOpen = box[8 * node + 3] * invTheta + box[8 * node + 7];
            const inside = px >= box[8 * node] && py >= box[8 * node + 1] && pz >= box[8 * node + 2]
                && px <= box[8 * node + 4] && py <= box[8 * node + 5] && pz <= box[8 * node + 6];
            if ((dx * dx + dy * dy + dz * dz < rOpen * rOpen || inside) && sp + 2 <= STACK_SIZE) {
                stack[sp] = children[2 * node + 1]; stack[sp + 1] = children[2 * node]; sp += 2;
                maxSp = Math.max(maxSp, sp);
            } else add(node);
        }
        for (let r = 0; r < sp; r++) {
            const node = stack[r];
            if (node !== myLeaf && mass[4 * node + 3] > 0) add(node);
        }
        maxVisits = Math.max(maxVisits, it);
        acc[3 * body] = ax; acc[3 * body + 1] = ay; acc[3 * body + 2] = az; phi[body] = ph; inter[body] = count;
    }
    return { acc, phi, inter, maxVisits, maxSp };
}

function buildBH(pos, n) {
    const { keys, vals } = mortonReplica(pos, n);
    const sorted = radixReplica(keys, vals);
    const { children, parents } = buildTreeReplica(sorted.keys);
    const { mass, box, completions } = bottomUpReplica(pos, sorted.vals, children, parents);
    return { keys: sorted.keys, vals: sorted.vals, children, parents, mass, box, completions };
}

function relErrors(acc, ref, n) {
    const e = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const dx = acc[3 * i] - ref[3 * i], dy = acc[3 * i + 1] - ref[3 * i + 1], dz = acc[3 * i + 2] - ref[3 * i + 2];
        e[i] = Math.hypot(dx, dy, dz) / Math.hypot(ref[3 * i], ref[3 * i + 1], ref[3 * i + 2]);
    }
    e.sort();
    return { median: e[n >> 1], p99: e[Math.floor(0.99 * n)], max: e[n - 1] };
}

// ---------------------------------------------------------------------------------------------

test('makeRng: deterministic, in [0, 1), uniform mean', () => {
    const a = makeRng(42), b = makeRng(42), c = makeRng(43);
    const xs = Array.from({ length: 20000 }, () => a());
    for (let i = 0; i < 100; i++) assert.equal(b(), xs[i]);
    assert.notEqual(c(), xs[0]);
    assert.ok(xs.every(x => x >= 0 && x < 1));
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    assert.ok(Math.abs(mean - 0.5) < 0.01, `mean ${mean}`);
});

test('generateBodies: every preset is deterministic, centered, mass 1/N, finite', (t) => {
    const n = 1024;
    for (const key of Object.keys(PRESETS)) {
        const { positions, velocities } = generateBodies(key, n, 0.05, 7);
        const again = generateBodies(key, n, 0.05, 7);
        assert.deepEqual(again.positions, positions, `${key} not deterministic`);
        assert.notDeepEqual(generateBodies(key, n, 0.05, 8).positions, positions);
        assert.equal(positions.length, 4 * n);
        const cm = [0, 0, 0], cv = [0, 0, 0];
        for (let i = 0; i < n; i++) {
            assert.equal(positions[4 * i + 3], f32(1 / n));
            for (let c = 0; c < 3; c++) {
                assert.ok(Number.isFinite(positions[4 * i + c]) && Number.isFinite(velocities[4 * i + c]));
                cm[c] += positions[4 * i + c] / n; cv[c] += velocities[4 * i + c] / n;
            }
        }
        const cmErr = Math.hypot(...cm), cvErr = Math.hypot(...cv);
        t.diagnostic(`${key}: |center of mass| ${cmErr.toExponential(2)}, |mean velocity| ${cvErr.toExponential(2)}`);
        // f32 rounding of positions ~1e-7 relative; centered to well below that times the radius
        assert.ok(cmErr < 1e-5 && cvErr < 1e-5);
    }
});

test('generateBodies: Plummer sphere is in virial equilibrium, spiral rotates about +y, collapse is cold', (t) => {
    const n = 2048, eps = 0.02;
    const pl = generateBodies('plummer', n, eps, 3);
    const { phi } = directForces(pl.positions, n, eps * eps);
    let K = 0, W = 0;
    for (let i = 0; i < n; i++) {
        const v = pl.velocities;
        K += 0.5 / n * (v[4 * i] ** 2 + v[4 * i + 1] ** 2 + v[4 * i + 2] ** 2);
        W += 0.5 / n * phi[i];
    }
    const virial = 2 * K / -W;
    t.diagnostic(`Plummer N=${n}: 2K/|W| = ${virial.toFixed(3)} (equilibrium = 1)`);
    assert.ok(Math.abs(virial - 1) < 0.1, 'Plummer sampling should be within 10% of virial equilibrium at N=2048');

    const sp = generateBodies('spiral', n, 0.05, 3);
    let Ly = 0;
    for (let i = 0; i < n; i++) Ly += sp.positions[4 * i + 2] * sp.velocities[4 * i] - sp.positions[4 * i] * sp.velocities[4 * i + 2];
    t.diagnostic(`spiral: L_y per unit mass = ${(Ly / n).toFixed(3)}`);
    assert.ok(Ly > 0);

    const co = generateBodies('collapse', n, 0.05, 3);
    let maxR = 0, maxV = 0;
    for (let i = 0; i < n; i++) {
        maxR = Math.max(maxR, Math.hypot(co.positions[4 * i], co.positions[4 * i + 1], co.positions[4 * i + 2]));
        maxV = Math.max(maxV, Math.hypot(co.velocities[4 * i], co.velocities[4 * i + 1], co.velocities[4 * i + 2]));
    }
    assert.equal(maxV, 0);
    assert.ok(maxR <= 2 + 0.1, `uniform sphere radius 2 (plus re-centering), got ${maxR}`);

    const col = generateBodies('collision', n, 0.05, 3);
    let xa = 0, xb = 0;
    for (let i = 0; i < n / 2; i++) { xa += col.positions[4 * i]; xb += col.positions[4 * (i + n / 2)]; }
    const sep = (xb - xa) / (n / 2);
    t.diagnostic(`collision: initial separation of the two galaxies ${sep.toFixed(3)} (D = 8)`);
    assert.ok(Math.abs(sep - 8) < 0.5);
});

test('dispatchGrid covers every workgroup within the per-dimension limit', () => {
    assert.deepEqual(dispatchGrid(0, 65535), [1, 1]);
    assert.deepEqual(dispatchGrid(1024, 65535), [1024, 1]);
    for (const [w, max] of [[65536, 65535], [100000, 65535], [1025, 1024], [1048576, 1024], [3, 2]]) {
        const [x, y] = dispatchGrid(w, max);
        assert.ok(x <= max && y <= max && x * y >= w && x * (y - 1) < w, `${w}/${max} -> ${x}x${y}`);
    }
    assert.throws(() => dispatchGrid(1024 * 1024 + 1, 1024));
});

test('Mat4 and OrbitCamera: projection, view and product match hand-computed matrices', (t) => {
    // perspective: a point on the near plane maps to depth 0, the far plane to depth 1 (WebGPU)
    const P = Mat4.perspective(FOV, 1.5, 0.05, 500);
    const apply = (M, v) => [0, 1, 2, 3].map(r => M[r] * v[0] + M[4 + r] * v[1] + M[8 + r] * v[2] + M[12 + r] * v[3]);
    const nearP = apply(P, [0, 0, -0.05, 1]), farP = apply(P, [0, 0, -500, 1]);
    assert.ok(Math.abs(nearP[2] / nearP[3]) < 1e-6 && Math.abs(farP[2] / farP[3] - 1) < 1e-6);
    const f = 1 / Math.tan(FOV / 2);
    const edge = apply(P, [Math.tan(FOV / 2) * 1.5 * 10, Math.tan(FOV / 2) * 10, -10, 1]);
    assert.ok(Math.abs(edge[0] / edge[3] - 1) < 1e-6 && Math.abs(edge[1] / edge[3] - 1) < 1e-6);
    assert.ok(Math.abs(P[5] - f) < 1e-6 && P[11] === -1);

    // lookAt: eye -> origin, center -> on the -z axis, rigid (orthonormal rotation)
    const eye = [3, 4, 5], center = [1, -2, 0.5];
    const V = Mat4.lookAt(eye, center, [0, 1, 0]);
    const e = apply(V, [...eye, 1]), c = apply(V, [...center, 1]);
    assert.ok(Math.hypot(e[0], e[1], e[2]) < 1e-5);
    const dist = Math.hypot(eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]);
    assert.ok(Math.abs(c[0]) < 1e-5 && Math.abs(c[1]) < 1e-5 && Math.abs(c[2] + dist) < 1e-5);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const d = V[a] * V[b] + V[4 + a] * V[4 + b] + V[8 + a] * V[8 + b];   // rows of R
        assert.ok(Math.abs(d - (a === b ? 1 : 0)) < 1e-6);
    }

    // multiply vs naive (row, col) indexing
    const rng = makeRng(5);
    const A = Float32Array.from({ length: 16 }, rng), B = Float32Array.from({ length: 16 }, rng);
    const AB = Mat4.multiply(A, B);
    let maxErr = 0;
    for (let r = 0; r < 4; r++) for (let col = 0; col < 4; col++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += A[k * 4 + r] * B[col * 4 + k];
        maxErr = Math.max(maxErr, Math.abs(AB[col * 4 + r] - s));
    }
    assert.ok(maxErr < 1e-6);

    // OrbitCamera: target projects to the screen center; basis is orthonormal
    const cam = new OrbitCamera();
    cam.target = [0.5, -1, 2];
    const VP = cam.getViewProj(16 / 9);
    const tc = apply(VP, [...cam.target, 1]);
    assert.ok(Math.abs(tc[0] / tc[3]) < 1e-5 && Math.abs(tc[1] / tc[3]) < 1e-5 && tc[2] / tc[3] > 0 && tc[2] / tc[3] < 1);
    const { right, up, forward } = cam.getBasis();
    const dd = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    assert.ok(Math.abs(dd(right, up)) < 1e-9 && Math.abs(dd(right, forward)) < 1e-9 && Math.abs(dd(up, forward)) < 1e-9);
    assert.ok(Math.abs(Math.hypot(...cam.getEye().map((x, i) => x - cam.target[i])) - cam.distance) < 1e-9);
    t.diagnostic(`Mat4.multiply max error vs naive ${maxErr.toExponential(2)}`);
});

test('uniforms: SimParams, Camera and RadixParams bytes match the WGSL structs', () => {
    const device = mockDevice();
    const sim = Object.create(NBodySimulation.prototype);
    Object.assign(sim, {
        device, count: 12345, dt: 0.0125, softening: 0.05, theta: 0.65, numPartials: 13, trailHead: 5,
        colorMode: 1, presetKey: 'collapse', camera: new OrbitCamera(), canvas: { width: 800, height: 450 },
    });
    sim.simData = new ArrayBuffer(32); sim.simF32 = new Float32Array(sim.simData); sim.simU32 = new Uint32Array(sim.simData);
    sim.camData = new ArrayBuffer(112); sim.camF32 = new Float32Array(sim.camData); sim.camU32 = new Uint32Array(sim.camData);
    sim.simBuffer = { label: 'sim' }; sim.cameraBuffer = { label: 'camera' };

    sim.writeSimParams();
    const simLayout = structLayout(code('force'), 'SimParams');
    assert.equal(simLayout.size, 32);
    const s = decodeStruct(device.lastWrite(sim.simBuffer).bytes, simLayout);
    assert.deepEqual(
        { count: s.count, numTiles: s.numTiles, numPartials: s.numPartials, trailSlot: s.trailSlot, pad0: s.pad0 },
        { count: 12345, numTiles: Math.ceil(12345 / WG), numPartials: 13, trailSlot: 5, pad0: 0 });
    assert.equal(s.dt, f32(0.0125));
    assert.equal(s.eps2, f32(0.05 * 0.05));
    assert.equal(s.theta, f32(0.65));
    // every module that embeds SimParams declares the same struct
    for (const m of wgslModules().filter(m => /struct SimParams/.test(m.code))) {
        assert.deepEqual(structLayout(m.code, 'SimParams').offsets, simLayout.offsets, m.name);
    }

    sim.writeCamera();
    const camLayout = structLayout(code('render'), 'Camera');
    assert.equal(camLayout.size, 112);
    const c = decodeStruct(device.lastWrite(sim.cameraBuffer).bytes, camLayout);
    const vp = sim.camera.getViewProj(800 / 450);
    assert.deepEqual(c.viewProj.flat(), Array.from(vp));
    assert.deepEqual(c.viewport, [800, 450]);
    const style = spriteStyle(12345);
    assert.equal(c.pointSize, f32(style.pointSize));
    assert.equal(c.brightness, f32(style.brightness));
    assert.equal(c.colorMode, 1);
    assert.equal(c.count, 12345);
    assert.equal(c.speedScale, f32(PRESETS.collapse.speedScale));
    assert.equal(c.trailHead, 5);
    assert.equal(c.trailLen, TRAIL_LEN);

    // RadixParams: one 16-byte struct per pass at a dynamic offset of radixStride (allocate())
    const d2 = mockDevice();
    const sim2 = Object.create(NBodySimulation.prototype);
    Object.assign(sim2, {
        device: d2, count: 5000, radixStride: 256, radixBuffer: { label: 'radix' }, layouts: {},
        simBuffer: {}, generation: 0, destroyBodyBuffers() {}, allocateTrail() {}, reset() {},
    });
    sim2.allocate();
    const radixLayout = structLayout(code('radix'), 'RadixParams');
    const bytes = d2.lastWrite(sim2.radixBuffer).bytes;
    assert.equal(bytes.byteLength, 256 * RADIX_PASSES);
    for (let p = 0; p < RADIX_PASSES; p++) {
        const r = decodeStruct(bytes, radixLayout, p * 256);
        assert.deepEqual(r, { shift: 4 * p, count: 5000, numTiles: Math.ceil(5000 / RADIX_TILE), pad0: 0 });
    }
});

test('Morton codes: expandBits matches a naive bit interleave; quantization stays in 0…1023', () => {
    const rng = makeRng(11);
    for (let i = 0; i < 5000; i++) {
        const x = Math.floor(rng() * 1024), y = Math.floor(rng() * 1024), z = Math.floor(rng() * 1024);
        assert.equal(((expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z)) >>> 0, naiveInterleave(x, y, z));
    }
    assert.equal(expandBits(1023), 0x09249249);
    const { positions } = generateBodies('collision', 3000, 0.05, 2);
    const { keys, cells } = mortonReplica(positions, 3000);
    assert.ok(cells.flat().every(q => q >= 0 && q <= 1023));
    assert.ok(cells.flat().includes(1023) && cells.flat().includes(0), 'extremes reach both ends of the cube');
    assert.ok(Array.from(keys).every(k => k < 2 ** 30));
});

test('radix sort replica: stable sort of (code, index) pairs vs Array.sort, edge cases', (t) => {
    const rng = makeRng(21);
    const cases = {
        random3000: Uint32Array.from({ length: 3000 }, () => Math.floor(rng() * 2 ** 30)),
        duplicates: Uint32Array.from({ length: 2500 }, () => Math.floor(rng() * 8) << 27),
        sorted: Uint32Array.from({ length: 1024 }, (_, i) => i * 1000),
        reversed: Uint32Array.from({ length: 2049 }, (_, i) => (2049 - i) * 77),
        allEqual: new Uint32Array(1500).fill(123456),
        single: Uint32Array.of(5),
        maxBits: Uint32Array.from({ length: 300 }, () => (Math.floor(rng() * 2 ** 32)) >>> 0),
    };
    for (const [name, keys] of Object.entries(cases)) {
        const vals = Uint32Array.from(keys, (_, i) => i);
        const out = radixReplica(keys, vals);
        const ref = Array.from(keys, (k, i) => [k, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        assert.deepEqual(Array.from(out.keys), ref.map(r => r[0]), `${name} keys`);
        assert.deepEqual(Array.from(out.vals), ref.map(r => r[1]), `${name} values (stability)`);
    }
    t.diagnostic(`${Object.keys(cases).length} cases sorted identically to Array.sort (stable)`);
});

test('Karras tree replica: a valid binary tree over the sorted leaves, also with duplicate codes', (t) => {
    const n = 4096;
    const { positions } = generateBodies('plummer', n, 0.05, 4);
    // 64 coincident bodies give equal Morton codes (the δ index tiebreak path)
    for (let i = 1; i < 64; i++) positions.set(positions.subarray(0, 3), 4 * i);
    const bh = buildBH(positions, n);
    const { children, parents, mass } = bh;
    // every node except the root has exactly one parent; each internal node's children are distinct
    const childCount = new Uint32Array(2 * n - 1);
    for (let i = 0; i < n - 1; i++) { childCount[children[2 * i]]++; childCount[children[2 * i + 1]]++; }
    assert.equal(childCount[0], 0);
    for (let v = 1; v < 2 * n - 1; v++) assert.equal(childCount[v], 1, `node ${v}`);
    assert.equal(parents[0], NONE);
    // DFS from the root reaches every leaf once, leaves under each node are contiguous, track depth
    let maxDepth = 0;
    const span = (node, depth) => {
        maxDepth = Math.max(maxDepth, depth);
        if (node >= n - 1) return [node - (n - 1), node - (n - 1)];
        const [a0, a1] = span(children[2 * node], depth + 1), [b0, b1] = span(children[2 * node + 1], depth + 1);
        assert.equal(a1 + 1, b0, `node ${node}: children ranges must be adjacent`);
        return [a0, b1];
    };
    assert.deepEqual(span(0, 0), [0, n - 1]);
    // bottom-up: every internal node completed exactly once, root mass = total mass, COM = mean
    assert.equal(bh.completions, n - 1);
    assert.ok(Math.abs(mass[3] - 1) < 1e-6, `root mass ${mass[3]}`);
    let cx = 0;
    for (let i = 0; i < n; i++) cx += positions[4 * i] * positions[4 * i + 3];
    assert.ok(Math.abs(mass[0] - cx) < 1e-6);
    t.diagnostic(`N=${n}: tree depth ${maxDepth} (bound 49, stack ${STACK_SIZE}); root mass ${mass[3].toFixed(6)}`);
    assert.ok(maxDepth <= 49);
});

test('brute-force replica (tiled, f32) vs direct float64 sum', (t) => {
    const n = 700, eps2 = 0.05 * 0.05;   // not a multiple of 256: exercises the zero-mass padding
    const { positions } = generateBodies('spiral', n, 0.05, 9);
    const rep = bruteReplica(positions, n, eps2);
    const ref = directForces(positions, n, eps2);
    const acc = Float64Array.from({ length: 3 * n }, (_, i) => rep[4 * Math.floor(i / 3) + (i % 3)]);
    const e = relErrors(acc, ref.acc, n);
    let phiErr = 0;
    for (let i = 0; i < n; i++) phiErr = Math.max(phiErr, Math.abs(rep[4 * i + 3] - ref.phi[i]) / Math.abs(ref.phi[i]));
    t.diagnostic(`N=${n}: accel rel. error median ${e.median.toExponential(2)}, max ${e.max.toExponential(2)}; potential max ${phiErr.toExponential(2)}`);
    // f32 accumulation over 768 terms: errors of a few ×1e-6, far from any algorithmic error
    assert.ok(e.max < 1e-4 && phiErr < 1e-5);
});

test('Barnes-Hut replica: force error vs direct sum per θ, decreasing with θ', (t) => {
    const n = 4096, eps = 0.05, eps2 = eps * eps;
    const { positions } = generateBodies('plummer', n, eps, 5);
    const bh = buildBH(positions, n);
    const ref = directForces(positions, n, eps2);
    // Median relative force error bounds for a monopole tree with the Barnes (1994) opening rule,
    // ~1.7× the values measured here (1.5e-3 / 4.2e-3 / 8.7e-3 / 1.8e-2, roughly ∝ θ²); p99 < 10×.
    const bounds = { 0.3: 2.5e-3, 0.5: 7e-3, 0.7: 1.5e-2, 1.0: 3e-2 };
    let prev = 0;
    for (const theta of [0.3, 0.5, 0.7, 1.0]) {
        const r = forceReplica(positions, bh.vals, bh, theta, eps2);
        const e = relErrors(r.acc, ref.acc, n);
        let phiErr = 0, inter = 0;
        for (let i = 0; i < n; i++) { phiErr = Math.max(phiErr, Math.abs(r.phi[i] - ref.phi[i]) / Math.abs(ref.phi[i])); inter += r.inter[i]; }
        t.diagnostic(`θ=${theta}: accel rel. err median ${e.median.toExponential(2)}, p99 ${e.p99.toExponential(2)}, max ${e.max.toExponential(2)}; `
            + `potential max ${phiErr.toExponential(2)}; interactions/body ${(inter / n).toFixed(0)} (brute ${n - 1}); max visits ${r.maxVisits}, max stack ${r.maxSp}`);
        assert.ok(e.median < bounds[theta], `θ=${theta} median ${e.median}`);
        assert.ok(e.p99 < 10 * bounds[theta], `θ=${theta} p99 ${e.p99}`);
        assert.ok(e.median > prev, 'error grows with θ');
        assert.ok(r.maxVisits < MAX_VISITS && r.maxSp <= STACK_SIZE);
        assert.ok(inter / n < n - 1, 'fewer interactions than brute force');
        prev = e.median;
    }
    // θ → 0 opens every node: exact up to float64 summation order
    const exact = forceReplica(positions, bh.vals, bh, 1e-6, eps2);
    const e0 = relErrors(exact.acc, ref.acc, n);
    t.diagnostic(`θ=1e-6 (all nodes opened): max rel. error ${e0.max.toExponential(2)}`);
    assert.ok(e0.max < 1e-10);
});

test('leapfrog (KDK) replica: energy error bounded, no secular drift; time-reversible', (t) => {
    const n = 256, eps = 0.05, eps2 = eps * eps, dt = 0.01, steps = 400;
    const { positions, velocities } = generateBodies('plummer', n, eps, 6);
    const x = Float64Array.from(positions), v = Float64Array.from(velocities);
    const energy = (acc) => {
        let K = 0, W = 0;
        for (let i = 0; i < n; i++) {
            const m = x[4 * i + 3];
            K += 0.5 * m * (v[4 * i] ** 2 + v[4 * i + 1] ** 2 + v[4 * i + 2] ** 2);
            W += 0.5 * m * acc.phi[i];
        }
        return K + W;
    };
    const step = (acc) => {
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {   // cs_kick_drift
            v[4 * i + c] += acc.acc[3 * i + c] * 0.5 * dt;
            x[4 * i + c] += v[4 * i + c] * dt;
        }
        const a = directForces(x, n, eps2);
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) v[4 * i + c] += a.acc[3 * i + c] * 0.5 * dt;   // cs_kick
        return a;
    };
    let acc = directForces(x, n, eps2);
    const E0 = energy(acc);
    let maxErr = 0, firstHalf = 0, secondHalf = 0;
    for (let s = 1; s <= steps; s++) {
        acc = step(acc);
        const err = Math.abs((energy(acc) - E0) / E0);
        maxErr = Math.max(maxErr, err);
        if (s <= steps / 2) firstHalf = Math.max(firstHalf, err); else secondHalf = Math.max(secondHalf, err);
    }
    t.diagnostic(`N=${n}, dt=${dt}, ${steps} steps: max |ΔE/E₀| ${maxErr.toExponential(2)} (first half ${firstHalf.toExponential(2)}, second half ${secondHalf.toExponential(2)})`);
    assert.ok(maxErr < 1e-3, 'symplectic leapfrog keeps |ΔE/E₀| < 1e-3 at dt = 0.01, ε = 0.05');
    assert.ok(secondHalf < 3 * firstHalf + 1e-6, 'energy error oscillates rather than drifting');

    // reverse velocities and integrate back: returns to the start (time-reversibility)
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) v[4 * i + c] = -v[4 * i + c];
    for (let s = 0; s < steps; s++) acc = step(acc);
    let back = 0;
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) back = Math.max(back, Math.abs(x[4 * i + c] - positions[4 * i + c]));
    t.diagnostic(`time reversal: max position error after ${steps} steps forward + back ${back.toExponential(2)}`);
    assert.ok(back < 1e-6);
});
