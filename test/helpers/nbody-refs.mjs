// N-body CPU references (js/examples/nbody.js), shared by test/nbody.test.mjs and test/gpu/nbody.gpu.test.mjs.
//
// Replicas of cs_morton, cs_radix_histogram/scan/scatter, cs_build_tree, cs_init_nodes + cs_bottom_up,
// cs_bh_force and cs_brute_force (test-only), plus the float64 direct-sum reference and relErrors()
// re-exported from js/refs/nbody.js (deployable, for in-browser comparisons).

import { WG, NONE, STACK_SIZE, MAX_VISITS, MAX_DEPTH, RADIX_PASSES, RADIX_TILE } from '../../js/examples/nbody.js';

export { directForces, relErrors } from '../../js/refs/nbody.js';

const f32 = Math.fround;

// ---------------------------------------------------------------------------------------------
// Replica of cs_brute_force: 256-body tiles, zero-mass padding past N, self excluded by index,
// f32 accumulation (Math.fround after every operation the shader does in f32).
export function bruteReplica(pos, n, eps2) {
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
export function expandBits(v) {
    let x = v & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    return x >>> 0;
}
export function naiveInterleave(x, y, z) {
    let code = 0;
    for (let b = 9; b >= 0; b--) {
        code = code * 8 + ((x >> b) & 1) * 4 + ((y >> b) & 1) * 2 + ((z >> b) & 1);
    }
    return code >>> 0;
}
export function boundsOf(pos, n) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], pos[4 * i + c]); hi[c] = Math.max(hi[c], pos[4 * i + c]);
    }
    return { lo, hi };
}
export function mortonReplica(pos, n) {
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
export function radixReplica(keysIn, valsIn) {
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
export const clz32 = Math.clz32;
export function buildTreeReplica(keys) {
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
export function bottomUpReplica(pos, vals, children, parents) {
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
export function forceReplica(pos, vals, tree, theta, eps2) {
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

export function buildBH(pos, n) {
    const { keys, vals } = mortonReplica(pos, n);
    const sorted = radixReplica(keys, vals);
    const { children, parents } = buildTreeReplica(sorted.keys);
    const { mass, box, completions } = bottomUpReplica(pos, sorted.vals, children, parents);
    return { keys: sorted.keys, vals: sorted.vals, children, parents, mass, box, completions };
}
