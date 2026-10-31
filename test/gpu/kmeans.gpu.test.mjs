// GPU execution of kmeans.js cs_assign + cs_update (one Lloyd iteration per dispatch pair) on a
// software GPU. CI-only — see test/gpu/harness.mjs. Exact shipped WGSL. The initial centroid, trail
// and label buffers come from the shipped KMeans.reset() and the Params bytes from the shipped
// writeComputeParams(), captured on a mock device; each iteration clears Stats, then dispatches
// cs_assign (numGroups workgroups) and cs_update (1 workgroup), as KMeans.encodeIteration() does.
//
// Every GPU iteration is checked against a float64 Lloyd's step (test/helpers/kmeans-refs.mjs)
// taken from the centroids the GPU iteration started from, and the whole run against an
// independent float64 Lloyd's run from the same initial centroids.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, run, read } from './harness.mjs';
import {
    KMeans, generatePoints, mulberry32, MAX_K, TILE, MAX_GROUPS, TRAIL_LEN, wgslModules,
} from '../../js/examples/kmeans.js';
import { moduleCode } from '../helpers/wgsl-layout.mjs';
import { mockDevice } from '../helpers/mock-gpu.mjs';
import { lloyd64Step } from '../helpers/kmeans-refs.mjs';

const CODE = moduleCode({ wgslModules }, 'compute');
const STATS_SIZE = 80;

// Shipped host state for n points / k clusters: points, Params bytes, and the centroid / trail /
// label bytes reset() uploads (its Math.random replaced by a seeded mulberry32, as in the CPU suite)
function shippedState(n, k, blobs, seed, method, points = generatePoints(n, blobs, 0.05, seed), centroids) {
    const device = mockDevice();
    const canvas = { width: 960, height: 720, getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) }) };
    const km = new KMeans(device, canvas);
    Object.assign(km, { n, k, initMethod: method, data: points });
    const saved = Math.random;
    Math.random = mulberry32(seed + 1);
    try { km.reset(); } finally { Math.random = saved; }
    km.writeComputeParams();
    const bytes = b => Uint8Array.from(device.lastWrite(b).bytes);
    const cent = bytes(km.centroidBuffer);
    if (centroids) new Float32Array(cent.buffer).set(centroids);
    return { points, params: bytes(km.computeParamsBuffer), cent, trail: bytes(km.trailBuffer), labels: bytes(km.labelBuffer) };
}

async function gpuKMeans(n, k, init) {
    const assign = await computePipeline(CODE, 'cs_assign');
    const update = await computePipeline(CODE, 'cs_update');
    const b = {
        points: buffer(init.points), labels: buffer(init.labels), centroids: buffer(init.cent),
        partials: buffer(MAX_GROUPS * MAX_K * 16), stats: buffer(STATS_SIZE), trail: buffer(init.trail),
        params: buffer(init.params),
    };
    const ga = bindGroup(assign, 0, b), gu = bindGroup(update, 0, b);
    const numGroups = Math.ceil(n / TILE);
    return {
        async iterate() {
            await run([
                { write: [b.stats, new Uint8Array(STATS_SIZE)] },   // clearBuffer(statsBuffer)
                { p: assign, groups: [ga], workgroups: [numGroups] },
                { p: update, groups: [gu], workgroups: [1] },
            ], 'kmeans iteration');
        },
        async state() {
            const st = await read(b.stats, Uint8Array, STATS_SIZE);
            const u32 = new Uint32Array(st.buffer), f32 = new Float32Array(st.buffer);
            return {
                labels: await read(b.labels, Uint32Array, n * 4),
                centroids: await read(b.centroids, Float32Array, MAX_K * 8),
                trail: await read(b.trail, Float32Array, MAX_K * TRAIL_LEN * 8),
                changed: u32[0], inertia: f32[1], empty: u32[2], counts: Array.from(u32.subarray(4, 4 + k)),
            };
        },
    };
}

// Checks one GPU iteration (prev -> cur) against float64; returns measured errors
function checkIteration(points, n, k, prev, cur) {
    const ref = lloyd64Step(points, n, k, prev.centroids);
    const d2 = (i, j) => (points[2 * i] - prev.centroids[2 * j]) ** 2 + (points[2 * i + 1] - prev.centroids[2 * j + 1]) ** 2;
    let ties = 0, wrong = 0, changed = 0, inertia = 0;
    const sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
        const l = cur.labels[i];
        assert.ok(l < k, `label ${l} out of range`);
        if (l !== ref.labels[i]) {
            // f32 vs f64 distances can only disagree on near-ties
            if (d2(i, l) - d2(i, ref.labels[i]) <= 1e-6) ties++; else wrong++;
        }
        if (l !== prev.labels[i]) changed++;
        sx[l] += points[2 * i]; sy[l] += points[2 * i + 1]; cnt[l]++; inertia += d2(i, l);
    }
    let centErr = 0;
    for (let j = 0; j < k; j++) {
        const ex = cnt[j] ? sx[j] / cnt[j] : prev.centroids[2 * j], ey = cnt[j] ? sy[j] / cnt[j] : prev.centroids[2 * j + 1];
        centErr = Math.max(centErr, Math.abs(cur.centroids[2 * j] - ex), Math.abs(cur.centroids[2 * j + 1] - ey));
        // trail: newest entry is the new centroid, the previous newest moved to age 1
        assert.equal(cur.trail[j * TRAIL_LEN * 2], cur.centroids[2 * j]);
        assert.equal(cur.trail[(j * TRAIL_LEN + 1) * 2 + 1], prev.trail[j * TRAIL_LEN * 2 + 1]);
    }
    assert.equal(wrong, 0, 'every label is the nearest centroid (up to near-ties)');
    assert.equal(cur.changed, changed, 'Stats.changed counts label changes');
    assert.deepEqual(cur.counts, cnt, 'Stats.counts are the cluster sizes of this assignment');
    assert.equal(cur.empty, cnt.filter(c => c === 0).length);
    return { ties, centErr, inertiaRel: Math.abs(cur.inertia - inertia) / inertia, ref };
}

test('cs_assign + cs_update: Lloyd iterations to convergence vs float64', async (t) => {
    // 20,000 points = 20 assign workgroups (last one partial): both reduction levels; same configs
    // as the CPU suite's replica test.
    for (const [n, k, blobs, seed, method] of [[20000, 5, 5, 11, 'kmeans++'], [20000, 12, 6, 12, 'random'], [3001, 2, 3, 13, 'kmeans++']]) {
        const init = shippedState(n, k, blobs, seed, method);
        const { points } = init;
        const g = await gpuKMeans(n, k, init);
        let prev = { labels: new Uint32Array(init.labels.buffer), centroids: new Float32Array(init.cent.buffer), trail: new Float32Array(init.trail.buffer) };
        let cent64 = Float64Array.from(prev.centroids.subarray(0, 2 * k));
        let iters = 0, ties = 0, maxCent = 0, maxInertia = 0, sameAs64 = true, cur;
        for (let it = 1; it <= 300; it++) {
            await g.iterate();
            cur = await g.state();
            const r = checkIteration(points, n, k, prev, cur);
            ties += r.ties; maxCent = Math.max(maxCent, r.centErr); maxInertia = Math.max(maxInertia, r.inertiaRel);
            // independent float64 trajectory from the same initial centroids
            const ref64 = lloyd64Step(points, n, k, cent64);
            for (let i = 0; i < n && sameAs64; i++) if (ref64.labels[i] !== cur.labels[i]) sameAs64 = false;
            cent64 = ref64.next;
            iters = it;
            prev = cur;
            if (cur.changed === 0) break;
        }
        let traj = 0;
        for (let j = 0; j < 2 * k; j++) traj = Math.max(traj, Math.abs(cur.centroids[j] - cent64[j]));
        t.diagnostic(`n=${n} k=${k} ${method}: converged in ${iters} iterations; per-iteration max |Δcentroid| ${maxCent.toExponential(2)}, ` +
            `max |Δinertia|/inertia ${maxInertia.toExponential(2)}, near-tie label flips ${ties}; ` +
            `labels identical to the float64 run every iteration: ${sameAs64}, final |Δcentroid| vs float64 run ${traj.toExponential(2)}`);
        assert.equal(cur.changed, 0, 'converged');
        // Same tolerances as the CPU suite: f32 tree sums of <= 20,000 terms; GPU reduction order is
        // the shader's fixed tree, so no extra slack is needed.
        assert.ok(maxCent < 1e-5, `centroid error ${maxCent}`);
        assert.ok(maxInertia < 1e-5, `inertia error ${maxInertia}`);
        if (sameAs64) assert.ok(traj < 1e-5, `trajectory centroid error ${traj}`);
    }
});

test('cs_update: an empty cluster keeps its centroid and is counted', async () => {
    const points = Float32Array.of(0, 0, 0.1, 0, 0, 0.1);
    const init = shippedState(3, 2, 0, 1, 'random', points, [0, 0, 50, 50]);
    const g = await gpuKMeans(3, 2, init);
    await g.iterate();
    const s = await g.state();
    assert.equal(s.empty, 1);
    assert.equal(s.changed, 3);
    assert.deepEqual([...s.labels], [0, 0, 0]);
    assert.deepEqual([s.centroids[2], s.centroids[3]], [50, 50]);
    assert.deepEqual(s.counts, [3, 0]);
    assert.ok(Math.abs(s.centroids[0] - 0.1 / 3) < 1e-7 && Math.abs(s.centroids[1] - 0.1 / 3) < 1e-7);
});
