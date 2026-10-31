// K-means clustering (js/examples/kmeans.js).
//
// CPU replica of the compute shader (cs_assign + cs_update); validates the algorithm and index
// math, not the WGSL itself — see #51. The replica keeps the shader's f32 arithmetic (Math.fround),
// its point-to-invocation mapping (base + i·ASSIGN_WG + t), the per-invocation accumulation order,
// the two tree reductions (workgroup partials, then partials of groups t, t + 256), the tie rule
// (lower index wins), empty-cluster handling and the trail shift. It is compared with an independent
// float64 Lloyd's algorithm run from the same initial centroids.
//
// Real code tested: generatePoints, initCentroids (Forgy and k-means++), and the KMeans uniform
// writers (Params, RenderParams) and reset() buffer contents through a mock device.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    KMeans, generatePoints, initCentroids, mulberry32, MAX_K, TILE, ASSIGN_WG, UPDATE_WG, MAX_GROUPS,
    TRAIL_LEN, ASPECT, wgslModules,
} from '../js/examples/kmeans.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';
import { lloyd64Step } from './helpers/kmeans-refs.mjs';

const f = Math.fround;
const PPT = TILE / ASSIGN_WG;
const mod = { wgslModules };

// ---- Replica of one GPU iteration. State: points (Float32Array xy), labels (Uint32Array),
// centroids (Float32Array MAX_K·2), trail (Float32Array MAX_K·TRAIL_LEN·2). Returns Stats.
function gpuIteration(points, n, k, labels, centroids, trail) {
    const numGroups = Math.ceil(n / TILE);
    const partials = new Float32Array(numGroups * MAX_K * 4);
    let changed = 0;
    for (let g = 0; g < numGroups; g++) {
        const base = g * TILE;
        const lab = new Uint32Array(ASSIGN_WG * PPT).fill(MAX_K);
        const px = new Float32Array(ASSIGN_WG * PPT), py = new Float32Array(ASSIGN_WG * PPT);
        const dist = new Float32Array(ASSIGN_WG * PPT);
        for (let t = 0; t < ASSIGN_WG; t++) {
            for (let i = 0; i < PPT; i++) {
                const idx = base + i * ASSIGN_WG + t;
                if (idx >= n) continue;
                const x = points[2 * idx], y = points[2 * idx + 1];
                let best = 0, bestD = f(3.0e38);
                for (let j = 0; j < k; j++) {
                    const dx = f(x - centroids[2 * j]), dy = f(y - centroids[2 * j + 1]);
                    const dd = f(f(dx * dx) + f(dy * dy));
                    if (dd < bestD) { bestD = dd; best = j; }
                }
                if (labels[idx] !== best) { changed++; labels[idx] = best; }
                const s = t * PPT + i;
                lab[s] = best; px[s] = x; py[s] = y; dist[s] = bestD;
            }
        }
        for (let c = 0; c < k; c++) {
            const sums = new Float32Array(ASSIGN_WG * 4);
            for (let t = 0; t < ASSIGN_WG; t++) {
                for (let i = 0; i < PPT; i++) {
                    const s = t * PPT + i;
                    if (lab[s] !== c) continue;
                    sums[4 * t] += px[s]; sums[4 * t + 1] += py[s]; sums[4 * t + 2] += 1; sums[4 * t + 3] += dist[s];
                }
            }
            for (let s = ASSIGN_WG / 2; s > 0; s >>= 1) {
                for (let t = 0; t < s; t++) for (let q = 0; q < 4; q++) sums[4 * t + q] += sums[4 * (t + s) + q];
            }
            partials.set(sums.subarray(0, 4), (g * MAX_K + c) * 4);
        }
    }
    // cs_update
    let inertia = 0, empty = 0;
    const counts = new Uint32Array(MAX_K);
    for (let c = 0; c < k; c++) {
        const sums = new Float32Array(UPDATE_WG * 4);
        for (let t = 0; t < UPDATE_WG; t++) {
            for (let j = 0; j < MAX_GROUPS / UPDATE_WG; j++) {
                const g = j * UPDATE_WG + t;
                if (g < numGroups) for (let q = 0; q < 4; q++) sums[4 * t + q] += partials[(g * MAX_K + c) * 4 + q];
            }
        }
        for (let s = UPDATE_WG / 2; s > 0; s >>= 1) {
            for (let t = 0; t < s; t++) for (let q = 0; q < 4; q++) sums[4 * t + q] += sums[4 * (t + s) + q];
        }
        if (sums[2] > 0) {
            centroids[2 * c] = f(sums[0] / sums[2]);
            centroids[2 * c + 1] = f(sums[1] / sums[2]);
        } else {
            empty++;
        }
        counts[c] = sums[2] >>> 0;
        inertia = f(inertia + sums[3]);
        for (let h = TRAIL_LEN - 1; h > 0; h--) {
            trail[(c * TRAIL_LEN + h) * 2] = trail[(c * TRAIL_LEN + h - 1) * 2];
            trail[(c * TRAIL_LEN + h) * 2 + 1] = trail[(c * TRAIL_LEN + h - 1) * 2 + 1];
        }
        trail[c * TRAIL_LEN * 2] = centroids[2 * c];
        trail[c * TRAIL_LEN * 2 + 1] = centroids[2 * c + 1];
    }
    return { changed, inertia, empty, counts };
}

// Independent float64 Lloyd's step: lloyd64Step in test/helpers/kmeans-refs.mjs (shared with test/gpu/).

function runBoth(n, k, blobs, seed, method) {
    const points = generatePoints(n, blobs, 0.05, seed);
    const init = initCentroids(points, n, k, method, mulberry32(seed + 1));
    const centroids = new Float32Array(MAX_K * 2); centroids.set(init);
    const trail = new Float32Array(MAX_K * TRAIL_LEN * 2);
    const labels = new Uint32Array(n).fill(0xffffffff);
    let cent64 = Float64Array.from(init);
    let gpuIters = 0, maxInertiaRel = 0, labelsAgreeAll = true;
    let stats;
    for (let it = 1; it <= 300; it++) {
        const ref = lloyd64Step(points, n, k, cent64);
        stats = gpuIteration(points, n, k, labels, centroids, trail);
        gpuIters = it;
        maxInertiaRel = Math.max(maxInertiaRel, Math.abs(stats.inertia - ref.inertia) / ref.inertia);
        let disagree = 0;
        for (let i = 0; i < n; i++) if (labels[i] !== ref.labels[i]) disagree++;
        if (disagree) labelsAgreeAll = false;
        // Stats.counts are the cluster sizes of this assignment
        for (let j = 0; j < k; j++) assert.equal(stats.counts[j], ref.counts[j]);
        // Trail: newest entry is the new centroid
        for (let j = 0; j < k; j++) assert.equal(trail[j * TRAIL_LEN * 2], centroids[2 * j]);
        cent64 = ref.next;
        if (stats.changed === 0) break;
    }
    let maxCentDiff = 0;
    for (let j = 0; j < 2 * k; j++) maxCentDiff = Math.max(maxCentDiff, Math.abs(centroids[j] - cent64[j]));
    return { gpuIters, maxInertiaRel, labelsAgreeAll, maxCentDiff, stats, points, labels, centroids, k, n };
}

test('generatePoints: deterministic, in-bounds, noise fraction (real code)', (t) => {
    const a = generatePoints(5000, 4, 0.1, 123);
    const b = generatePoints(5000, 4, 0.1, 123);
    assert.deepEqual(a, b);
    assert.notDeepEqual(a, generatePoints(5000, 4, 0.1, 124));
    for (let i = 0; i < 5000; i++) {
        assert.ok(Math.abs(a[2 * i]) <= f(ASPECT) && Math.abs(a[2 * i + 1]) <= 1);
    }
    t.diagnostic('5000 points within [-4/3, 4/3] x [-1, 1], identical for equal seeds');
});

test('initCentroids: Forgy picks distinct data points; k-means++ beats Forgy (real code)', (t) => {
    const n = 8000, k = 8;
    let kpp = 0, forgy = 0, ratioMax = 0;
    const seeds = [1, 2, 3, 4, 5, 6, 7, 8];
    for (const seed of seeds) {
        const pts = generatePoints(n, k, 0.02, seed);
        const cost = c => lloyd64Step(pts, n, k, c).inertia;
        const r = initCentroids(pts, n, k, 'random', mulberry32(seed));
        const keys = new Set();
        for (let c = 0; c < k; c++) {
            keys.add(`${r[2 * c]},${r[2 * c + 1]}`);
            let found = false;
            for (let i = 0; i < n && !found; i++) found = pts[2 * i] === r[2 * c] && pts[2 * i + 1] === r[2 * c + 1];
            assert.ok(found, 'Forgy centroid is a data point');
        }
        assert.equal(keys.size, k);
        const p = initCentroids(pts, n, k, 'kmeans++', mulberry32(seed));
        // Converged Lloyd's from k-means++ as a stand-in for OPT
        let c = Float64Array.from(p);
        for (let it = 0; it < 100; it++) c = lloyd64Step(pts, n, k, c).next;
        const opt = cost(c);
        kpp += cost(p); forgy += cost(r);
        ratioMax = Math.max(ratioMax, cost(p) / opt);
    }
    t.diagnostic(`mean initial inertia over ${seeds.length} seeds: k-means++ ${(kpp / seeds.length).toFixed(2)}, Forgy ${(forgy / seeds.length).toFixed(2)}; worst k-means++/converged ${ratioMax.toFixed(2)}`);
    assert.ok(kpp < forgy, 'k-means++ seeding has lower mean initial inertia than Forgy');
    // Arthur & Vassilvitskii: E[cost] <= 8(ln k + 2)·OPT; a single draw can exceed the expectation,
    // so this is a loose sanity bound.
    assert.ok(ratioMax < 8 * (Math.log(k) + 2));
});

test('GPU iteration replica matches float64 Lloyd\'s', (t) => {
    // 20,000 points = 20 assign workgroups (last one partial) exercise both reduction levels
    for (const [n, k, blobs, seed, method] of [[20000, 5, 5, 11, 'kmeans++'], [20000, 12, 6, 12, 'random'], [3001, 2, 3, 13, 'kmeans++']]) {
        const r = runBoth(n, k, blobs, seed, method);
        t.diagnostic(`n=${n} k=${k} ${method}: converged in ${r.gpuIters} iterations, max |Δinertia|/inertia ${r.maxInertiaRel.toExponential(2)}, ` +
            `max |Δcentroid| ${r.maxCentDiff.toExponential(2)}, labels identical every iteration: ${r.labelsAgreeAll}`);
        assert.equal(r.stats.changed, 0, 'replica converged');
        // f32 sums of ≤ 20,000 squared distances in tree order: relative error ~ 1e-7, allow 1e-5
        assert.ok(r.maxInertiaRel < 1e-5, `inertia rel err ${r.maxInertiaRel}`);
        // centroids: f32 means of ~1e4 points, coordinates ≤ 4/3, allow 1e-5
        assert.ok(r.maxCentDiff < 1e-5, `centroid diff ${r.maxCentDiff}`);
        assert.equal(r.stats.empty, 0);
        // At convergence every point's label is its nearest centroid (f64 check, near-ties allowed)
        let wrong = 0;
        for (let i = 0; i < r.n; i++) {
            const x = r.points[2 * i], y = r.points[2 * i + 1];
            const d = j => (x - r.centroids[2 * j]) ** 2 + (y - r.centroids[2 * j + 1]) ** 2;
            let best = 0;
            for (let j = 1; j < r.k; j++) if (d(j) < d(best)) best = j;
            if (r.labels[i] !== best && d(r.labels[i]) - d(best) > 1e-6) wrong++;
        }
        assert.equal(wrong, 0);
    }
});

test('replica: empty cluster keeps its centroid and is counted', () => {
    const points = Float32Array.of(0, 0, 0.1, 0, 0, 0.1);
    const centroids = new Float32Array(MAX_K * 2); centroids.set([0, 0, 50, 50]);
    const trail = new Float32Array(MAX_K * TRAIL_LEN * 2);
    const s = gpuIteration(points, 3, 2, new Uint32Array(3).fill(0xffffffff), centroids, trail);
    assert.equal(s.empty, 1);
    assert.equal(s.changed, 3);
    assert.deepEqual([centroids[2], centroids[3]], [50, 50]);
    assert.deepEqual([...s.counts.subarray(0, 2)], [3, 0]);
});

// ---- Uniform layouts (real writers) ----
function makeKMeans() {
    const device = mockDevice();
    const canvas = { width: 960, height: 720, getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) }) };
    return { device, km: new KMeans(device, canvas) };
}

test('uniform layouts: Params and RenderParams writers match WGSL', (t) => {
    const P = structLayout(moduleCode(mod, 'compute'), 'Params');
    const R = structLayout(moduleCode(mod, 'render'), 'RenderParams');
    const St = structLayout(moduleCode(mod, 'compute'), 'Stats');
    assert.equal(P.size, 16);
    assert.equal(R.size, 40);
    assert.equal(St.size, 80);   // STATS_SIZE and the readStats() indices [0], [1], [2], [4..]
    assert.deepEqual(St.offsets, { changed: 0, inertia: 4, empty: 8, pad0: 12, counts: 16 });

    const { device, km } = makeKMeans();
    km.n = 300001; km.k = 13; km.iteration = 7;
    km.writeComputeParams();
    assert.deepEqual(decodeStruct(device.lastWrite(km.computeParamsBuffer).bytes, P),
        { n: 300001, k: 13, numGroups: Math.ceil(300001 / TILE), pad0: 0 });
    km.writeRenderParams();
    const r = decodeStruct(device.lastWrite(km.renderParamsBuffer).bytes, R);
    assert.deepEqual(r.scale, [f(1 / ASPECT), 1]);
    assert.deepEqual(r.pixelClip, [f(2 / 960), f(2 / 720)]);
    assert.equal(r.k, 13);
    assert.equal(r.trailCount, 8);
    assert.equal(r.pointSize, 1.5);
    assert.equal(r.markerSize, 20);
    assert.equal(r.pixelWorld, f(2 / 720));
    assert.equal(r.voronoiAlpha, f(0.16));
    t.diagnostic(`Params ${P.size} B, RenderParams ${R.size} B (offsets ${JSON.stringify(R.offsets)}) decoded OK`);
});

test('reset(): centroid, trail and label buffers (real code)', () => {
    const { device, km } = makeKMeans();
    km.n = 2000; km.k = 4; km.initMethod = 'random';
    km.data = generatePoints(km.n, 4, 0.05, 99);
    km.reset();
    const cent = new Float32Array(device.lastWrite(km.centroidBuffer).bytes.buffer);
    const trail = new Float32Array(device.lastWrite(km.trailBuffer).bytes.buffer);
    const labels = new Uint32Array(device.lastWrite(km.labelBuffer).bytes.buffer);
    assert.equal(cent.length, MAX_K * 2);
    assert.equal(labels.length, km.n);
    assert.ok(labels.every(v => v === 0xffffffff));
    for (let c = 0; c < km.k; c++) {
        for (let h = 0; h < TRAIL_LEN; h++) {
            assert.equal(trail[(c * TRAIL_LEN + h) * 2], cent[2 * c]);
            assert.equal(trail[(c * TRAIL_LEN + h) * 2 + 1], cent[2 * c + 1]);
        }
    }
});
