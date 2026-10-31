// N-body (js/examples/nbody.js) on a software GPU — CI only, see test/gpu/harness.mjs.
//
// Runs the exact shipped compute shaders in the order NBodySimulation.encodeForces() dispatches them:
//   brute force:  cs_brute_force
//   Barnes-Hut:   cs_bounds_partial, cs_bounds_final, cs_morton, 8 × (cs_radix_histogram, cs_radix_scan,
//                 cs_radix_scatter) ping-ponging A→B→A, cs_init_nodes, cs_build_tree, cs_bottom_up, cs_bh_force
// with the buffer sizes of allocate() and the uniform bytes of the shipped writers (writeSimParams and
// the RadixParams slots written by allocate(), captured through a mock device). The one deviation:
// the example binds RadixParams with a dynamic offset into one buffer; here each pass gets its own
// 16-byte uniform buffer holding the same bytes (the harness builds plain explicit layouts).
//
// Checked against the CPU references of the CPU suite (test/helpers/nbody-refs.mjs):
//   bounds exact; Morton codes vs mortonReplica (see the tolerance note there); the sorted (key, index)
//   pairs equal a stable sort of the GPU's own codes; the tree (children, parents) equals
//   buildTreeReplica of the GPU's sorted keys and is a valid binary radix tree; every internal node
//   was visited exactly twice; node mass / COM / box vs the float64 bottomUpReplica; forces vs the
//   float64 direct sum with the CPU-suite bounds.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, read, run } from './harness.mjs';
import {
    WG, NONE, RADIX_PASSES, RADIX_TILE, REDUCE_TILE, STACK_SIZE, MAX_VISITS,
    generateBodies, NBodySimulation, wgslModules,
} from '../../js/examples/nbody.js';
import { mockDevice } from '../helpers/mock-gpu.mjs';
import {
    directForces, boundsOf, mortonReplica, buildTreeReplica, bottomUpReplica, relErrors,
} from '../helpers/nbody-refs.mjs';

const code = name => wgslModules().find(m => m.name === name).code;
const EPS = 0.05, DT = 0.01;

// ---- Shipped uniform writers, through a mock device ----

function simParamsBytes(n, theta) {
    const device = mockDevice();
    const sim = Object.create(NBodySimulation.prototype);
    Object.assign(sim, { device, count: n, dt: DT, softening: EPS, theta, numPartials: Math.ceil(n / REDUCE_TILE), trailHead: 0 });
    sim.simData = new ArrayBuffer(32); sim.simF32 = new Float32Array(sim.simData); sim.simU32 = new Uint32Array(sim.simData);
    sim.simBuffer = { label: 'sim' };
    sim.writeSimParams();
    return device.lastWrite(sim.simBuffer).bytes;
}

function radixParamsBytes(n) {
    const device = mockDevice();
    const sim = Object.create(NBodySimulation.prototype);
    Object.assign(sim, {
        device, count: n, radixStride: 256, radixBuffer: { label: 'radix' }, layouts: {},
        simBuffer: {}, generation: 0, destroyBodyBuffers() {}, allocateTrail() {}, reset() {},
    });
    sim.allocate();
    const bytes = device.lastWrite(sim.radixBuffer).bytes;
    return Array.from({ length: RADIX_PASSES }, (_, p) => bytes.slice(p * 256, p * 256 + 16));
}

// ---- Pipelines (compiled once) ----

let P;
async function pipelines() {
    if (P) return P;
    const entries = {
        brute: ['cs_brute_force'], bounds: ['cs_bounds_partial', 'cs_bounds_final'], morton: ['cs_morton'],
        radix: ['cs_radix_histogram', 'cs_radix_scan', 'cs_radix_scatter'], initNodes: ['cs_init_nodes'],
        build: ['cs_build_tree'], bottomUp: ['cs_bottom_up'], force: ['cs_bh_force'],
    };
    P = {};
    for (const [mod, eps] of Object.entries(entries)) for (const e of eps) P[e] = await computePipeline(code(mod), e);
    return P;
}

// The workgroup grid of NBodySimulation.dispatch(): small jobs are one row
const grid = w => [Math.max(w, 1), 1];

// Buffers as allocate() sizes them (all zeroed here, then some prefilled with garbage below)
function allocate(n, positions) {
    const nodes = 2 * n - 1, internal = Math.max(n - 1, 1);
    const numPartials = Math.ceil(n / REDUCE_TILE), sortTiles = Math.ceil(n / RADIX_TILE);
    return {
        n, nodes, internal, numPartials, sortTiles,
        positions: buffer(positions),
        accel: buffer(16 * n), interactions: buffer(4 * n),
        keysA: buffer(4 * n), valsA: buffer(4 * n), keysB: buffer(4 * n), valsB: buffer(4 * n),
        offsets: buffer(4 * 16 * sortTiles),
        children: buffer(8 * internal), parents: buffer(4 * nodes),
        // stale values from a "previous frame": cs_build_tree must reset the visit counters
        visits: buffer(new Uint32Array(internal).fill(3)),
        nodeMass: buffer(new Float32Array(4 * nodes).fill(0.25)), nodeBox: buffer(32 * nodes),
        bounds: buffer(16 * (2 + 2 * numPartials)),
    };
}

async function barnesHut(positions, n, theta) {
    const p = await pipelines();
    const b = allocate(n, positions);
    const sim = buffer(simParamsBytes(n, theta));
    const radix = radixParamsBytes(n).map(bytes => buffer(bytes));
    const bodies = Math.ceil(n / WG);

    // Stage 1-2: bounds and Morton codes (read back before the sort overwrites keysA)
    const boundsG = { positions: b.positions, bounds: b.bounds, params: sim };
    await run([
        { p: p.cs_bounds_partial, groups: [bindGroup(p.cs_bounds_partial, 0, boundsG)], workgroups: grid(b.numPartials) },
        { p: p.cs_bounds_final, groups: [bindGroup(p.cs_bounds_final, 0, boundsG)], workgroups: [1] },
        { p: p.cs_morton, groups: [bindGroup(p.cs_morton, 0, { positions: b.positions, bounds: b.bounds, keys: b.keysA, vals: b.valsA, params: sim })], workgroups: grid(bodies) },
    ], 'bounds + morton');
    const bounds = await read(b.bounds, Float32Array, 32);
    const mortonKeys = await read(b.keysA, Uint32Array, 4 * n);
    const mortonVals = await read(b.valsA, Uint32Array, 4 * n);

    // Stage 3: 8 radix passes, A→B on even passes and B→A on odd ones (result back in A)
    const steps = [];
    for (let pass = 0; pass < RADIX_PASSES; pass++) {
        const [kIn, vIn, kOut, vOut] = pass % 2 === 0 ? [b.keysA, b.valsA, b.keysB, b.valsB] : [b.keysB, b.valsB, b.keysA, b.valsA];
        const res = { keysIn: kIn, valsIn: vIn, keysOut: kOut, valsOut: vOut, offsets: b.offsets, params: radix[pass] };
        for (const [e, w] of [['cs_radix_histogram', b.sortTiles], ['cs_radix_scan', 1], ['cs_radix_scatter', b.sortTiles]]) {
            steps.push({ p: p[e], groups: [bindGroup(p[e], 0, res)], workgroups: grid(w) });
        }
    }
    // Stage 4-6: leaves, Karras tree, bottom-up mass/box, traversal
    steps.push(
        { p: p.cs_init_nodes, groups: [bindGroup(p.cs_init_nodes, 0, { vals: b.valsA, positions: b.positions, nodeMass: b.nodeMass, nodeBox: b.nodeBox, params: sim })], workgroups: grid(bodies) },
        { p: p.cs_build_tree, groups: [bindGroup(p.cs_build_tree, 0, { keys: b.keysA, children: b.children, parents: b.parents, visits: b.visits, params: sim })], workgroups: grid(Math.ceil(Math.max(n - 1, 1) / WG)) },
        { p: p.cs_bottom_up, groups: [bindGroup(p.cs_bottom_up, 0, { parents: b.parents, children: b.children, visits: b.visits, massWords: b.nodeMass, boxWords: b.nodeBox, params: sim })], workgroups: grid(bodies) },
    );
    await run(steps, 'radix + tree');

    const out = {
        bounds, mortonKeys, mortonVals,
        keys: await read(b.keysA, Uint32Array, 4 * n), vals: await read(b.valsA, Uint32Array, 4 * n),
        children: await read(b.children, Uint32Array, 8 * (n - 1)), parents: await read(b.parents, Uint32Array, 4 * b.nodes),
        visits: await read(b.visits, Uint32Array, 4 * (n - 1)),
        mass: await read(b.nodeMass, Float32Array, 16 * b.nodes), box: await read(b.nodeBox, Float32Array, 32 * b.nodes),
    };

    // Traversal, for each requested θ (SimParams rewritten in between, as a θ change does)
    out.forces = {};
    for (const th of [theta].flat()) {
        const params = buffer(simParamsBytes(n, th));
        await run([{ p: p.cs_bh_force, groups: [bindGroup(p.cs_bh_force, 0, {
            positions: b.positions, vals: b.valsA, nodeMass: b.nodeMass, nodeBox: b.nodeBox, children: b.children,
            accel: b.accel, interactions: b.interactions, params })], workgroups: grid(bodies) }], `cs_bh_force θ=${th}`);
        out.forces[th] = { accel: await read(b.accel, Float32Array, 16 * n), inter: await read(b.interactions, Float32Array, 4 * n) };
    }
    return out;
}

// x9 y9 z9 … x0 y0 z0 -> [x, y, z]
function deinterleave(code) {
    const c = [0, 0, 0];
    for (let b = 9; b >= 0; b--) for (let a = 0; a < 3; a++) c[a] = (c[a] << 1) | ((code >>> (3 * b + 2 - a)) & 1);
    return c;
}

function checkTree(t, label, positions, n, g) {
    // Bounds: min / max are exact in f32
    const { lo, hi } = boundsOf(positions, n);
    assert.deepEqual([...g.bounds.subarray(0, 3)], lo.map(Math.fround), `${label}: bounds min`);
    assert.deepEqual([...g.bounds.subarray(4, 7)], hi.map(Math.fround), `${label}: bounds max`);

    // Morton codes. The CPU replica computes scale = 1024 / side with one correctly rounded f32
    // division; WGSL only guarantees 2.5 ulp for f32 division, so a body sitting exactly on a cell
    // boundary may quantize one cell over. Allow that for ≤ 0.1% of bodies, never more than ±1.
    const ref = mortonReplica(positions, n);
    let mismatched = 0;
    for (let i = 0; i < n; i++) {
        assert.equal(g.mortonVals[i], i, `${label}: vals[${i}] = ${i}`);
        if (g.mortonKeys[i] === ref.keys[i]) continue;
        mismatched++;
        const a = deinterleave(g.mortonKeys[i]);
        for (let c = 0; c < 3; c++) assert.ok(Math.abs(a[c] - ref.cells[i][c]) <= 1, `${label}: body ${i} cell ${a} vs ${ref.cells[i]}`);
    }
    assert.ok(mismatched <= Math.ceil(n / 1000), `${label}: ${mismatched} Morton codes differ from the CPU replica`);

    // Radix sort: exactly a stable sort of the GPU's own (code, index) pairs
    const sorted = Array.from(g.mortonKeys, (k, i) => [k, i]).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    assert.deepEqual(Array.from(g.keys), sorted.map(s => s[0]), `${label}: sorted keys`);
    assert.deepEqual(Array.from(g.vals), sorted.map(s => s[1]), `${label}: sorted values (stable)`);

    // Karras tree: identical to the replica on the same keys, and structurally valid
    const tree = buildTreeReplica(g.keys);
    assert.deepEqual(Array.from(g.children), Array.from(tree.children.subarray(0, 2 * (n - 1))), `${label}: children`);
    assert.deepEqual(Array.from(g.parents), Array.from(tree.parents), `${label}: parents`);
    assert.equal(g.parents[0], NONE);
    const childCount = new Uint32Array(2 * n - 1);
    for (let i = 0; i < n - 1; i++) {
        for (const c of [g.children[2 * i], g.children[2 * i + 1]]) { childCount[c]++; assert.equal(g.parents[c], i); }
    }
    assert.equal(childCount[0], 0);
    for (let v = 1; v < 2 * n - 1; v++) assert.equal(childCount[v], 1, `${label}: node ${v} has one parent`);
    let maxDepth = 0;
    const span = (node, depth) => {
        maxDepth = Math.max(maxDepth, depth);
        if (node >= n - 1) return [node - (n - 1), node - (n - 1)];
        const [a0, a1] = span(g.children[2 * node], depth + 1), [b0, b1] = span(g.children[2 * node + 1], depth + 1);
        assert.equal(a1 + 1, b0, `${label}: node ${node} children ranges adjacent`);
        return [a0, b1];
    };
    assert.deepEqual(span(0, 0), [0, n - 1], `${label}: every leaf reached once, in order`);
    assert.ok(maxDepth < STACK_SIZE, `${label}: depth ${maxDepth}`);

    // Bottom-up: each internal node completed by its second visitor (stale counters were reset)
    assert.ok(g.visits.every(v => v === 2), `${label}: every internal node visited exactly twice`);
    const bu = bottomUpReplica(positions, g.vals, tree.children, tree.parents);
    let massRel = 0, comAbs = 0, boxExact = true, sizeRel = 0, offAbs = 0;
    for (let v = 0; v < 2 * n - 1; v++) {
        massRel = Math.max(massRel, Math.abs(g.mass[4 * v + 3] - bu.mass[4 * v + 3]) / bu.mass[4 * v + 3]);
        for (let c = 0; c < 3; c++) {
            comAbs = Math.max(comAbs, Math.abs(g.mass[4 * v + c] - bu.mass[4 * v + c]));
            if (g.box[8 * v + c] !== Math.fround(bu.box[8 * v + c]) || g.box[8 * v + 4 + c] !== Math.fround(bu.box[8 * v + 4 + c])) boxExact = false;
        }
        if (v < n - 1) {
            sizeRel = Math.max(sizeRel, Math.abs(g.box[8 * v + 3] - bu.box[8 * v + 3]) / Math.max(bu.box[8 * v + 3], 1e-30));
            offAbs = Math.max(offAbs, Math.abs(g.box[8 * v + 7] - bu.box[8 * v + 7]));
        }
    }
    t.diagnostic(`${label}: Morton mismatches ${mismatched}/${n}; depth ${maxDepth}; root mass ${g.mass[3].toFixed(7)}; `
        + `node mass max rel err ${massRel.toExponential(2)}, COM max |err| ${comAbs.toExponential(2)}, `
        + `size max rel err ${sizeRel.toExponential(2)}, COM offset max |err| ${offAbs.toExponential(2)}, boxes exact ${boxExact}`);
    assert.ok(boxExact, `${label}: node boxes are exact unions of the leaf points`);
    // f32 sums of ≤ N masses of 1/N along ≤ ~50 levels: a few ×1e-7 relative per level
    assert.ok(massRel < 1e-5, `${label}: node mass`);
    assert.ok(Math.abs(g.mass[3] - 1) < 1e-5, `${label}: root mass`);
    // COM: f32 weighted means of coordinates |x| ≲ 10, rounding compounded over the depth
    assert.ok(comAbs < 1e-4, `${label}: COM`);
    assert.ok(sizeRel < 1e-6, `${label}: node size`);
    assert.ok(offAbs < 2e-4, `${label}: COM offset δ`);
}

test('Barnes-Hut tree build: bounds, Morton, radix sort, Karras tree, bottom-up (N = 4096, 3000, duplicates)', async (t) => {
    const cases = [
        ['plummer N=4096', 4096, generateBodies('plummer', 4096, EPS, 5).positions],
        // not a multiple of 256 or 1024: partial tiles in every stage
        ['collision N=3000', 3000, generateBodies('collision', 3000, EPS, 2).positions],
    ];
    // 64 coincident bodies give equal Morton codes (the δ index tiebreak path), as in the CPU suite
    const dup = generateBodies('plummer', 4096, EPS, 4).positions;
    for (let i = 1; i < 64; i++) dup.set(dup.subarray(0, 3), 4 * i);
    cases.push(['plummer N=4096 with 64 coincident', 4096, dup]);
    for (const [label, n, positions] of cases) {
        const g = await barnesHut(positions, n, 0.5);
        checkTree(t, label, positions, n, g);
    }
});

test('Barnes-Hut force (cs_bh_force) vs float64 direct sum: θ = 0.5 and θ → 0', async (t) => {
    const n = 4096, eps2 = EPS * EPS;
    const { positions } = generateBodies('plummer', n, EPS, 5);
    const g = await barnesHut(positions, n, [0.5, 1e-6]);
    const ref = directForces(positions, n, eps2);
    for (const [theta, bound] of [[0.5, 7e-3], [1e-6, null]]) {
        const { accel, inter } = g.forces[theta];
        const acc = Float64Array.from({ length: 3 * n }, (_, i) => accel[4 * Math.floor(i / 3) + (i % 3)]);
        const e = relErrors(acc, ref.acc, n);
        let phiErr = 0, total = 0, maxInter = 0;
        for (let i = 0; i < n; i++) {
            phiErr = Math.max(phiErr, Math.abs(accel[4 * i + 3] - ref.phi[i]) / Math.abs(ref.phi[i]));
            total += inter[i]; maxInter = Math.max(maxInter, inter[i]);
        }
        t.diagnostic(`θ=${theta}: accel rel. err median ${e.median.toExponential(2)}, p99 ${e.p99.toExponential(2)}, `
            + `max ${e.max.toExponential(2)}; potential max ${phiErr.toExponential(2)}; interactions/body ${(total / n).toFixed(0)} (brute ${n - 1})`);
        assert.ok(accel.every(Number.isFinite));
        if (bound) {
            // CPU-suite bounds for θ = 0.5 (forceReplica in float64 on the same bodies)
            assert.ok(e.median < bound, `median ${e.median}`);
            assert.ok(e.p99 < 10 * bound, `p99 ${e.p99}`);
            assert.ok(total / n < n - 1, 'fewer interactions than brute force');
        } else {
            // Every node opened: an exact pairwise sum, but accumulated in f32 in tree order (the CPU
            // replica is float64, hence 1e-10 there). Relative error is measured against the net force,
            // which nearly cancels for bodies near the center of the sphere, so the max is looser.
            assert.equal(total, n * (n - 1), 'all N−1 leaves summed for every body');
            assert.ok(maxInter < MAX_VISITS);
            assert.ok(e.p99 < 1e-4 && e.max < 1e-2, `θ→0 p99 ${e.p99} max ${e.max}`);
            assert.ok(phiErr < 1e-5, `θ→0 potential ${phiErr}`);
        }
    }
});

test('brute force (cs_brute_force, 256-body tiles) vs float64 direct sum (N = 700, 3000)', async (t) => {
    const p = await pipelines();
    for (const [preset, n, seed] of [['spiral', 700, 9], ['collision', 3000, 2]]) {
        const { positions } = generateBodies(preset, n, EPS, seed);
        const accel = buffer(16 * n);
        await run([{ p: p.cs_brute_force, groups: [bindGroup(p.cs_brute_force, 0, {
            positions: buffer(positions), accel, params: buffer(simParamsBytes(n, 0.5)) })], workgroups: grid(Math.ceil(n / WG)) }], 'cs_brute_force');
        const got = await read(accel, Float32Array, 16 * n);
        const ref = directForces(positions, n, EPS * EPS);
        const acc = Float64Array.from({ length: 3 * n }, (_, i) => got[4 * Math.floor(i / 3) + (i % 3)]);
        const e = relErrors(acc, ref.acc, n);
        let phiErr = 0;
        for (let i = 0; i < n; i++) phiErr = Math.max(phiErr, Math.abs(got[4 * i + 3] - ref.phi[i]) / Math.abs(ref.phi[i]));
        t.diagnostic(`${preset} N=${n}: accel rel. error median ${e.median.toExponential(2)}, max ${e.max.toExponential(2)}; potential max ${phiErr.toExponential(2)}`);
        // CPU-suite bound (N = 700 spiral, seed 9, as in test/nbody.test.mjs): f32 accumulation over
        // 768 terms. At N = 3000 the relative error of bodies whose net force nearly cancels can be
        // larger, so there only the 99th percentile gets the CPU bound and the max a 10x looser one.
        assert.ok(phiErr < 1e-5, `potential ${phiErr}`);
        if (n === 700) assert.ok(e.max < 1e-4, `max ${e.max}`);
        else assert.ok(e.p99 < 1e-4 && e.max < 1e-3, `p99 ${e.p99} max ${e.max}`);
    }
});
