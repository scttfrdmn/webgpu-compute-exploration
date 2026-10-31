// Voronoi / JFA references shared by test/voronoi-jfa.test.mjs (CPU replica) and
// test/gpu/voronoi-jfa.gpu.test.mjs (shipped WGSL on a software GPU).
// The float64 brute-force reference (metric64, errorsVsBrute) and the error-rate bounds live in
// js/refs/voronoi.js (deployable, for in-browser comparisons); the replicas and mock helpers here
// are test-only.

import { VoronoiJFA, wgslModules } from '../../js/examples/voronoi-jfa.js';
import { structLayout, decodeStruct, moduleCode } from './wgsl-layout.mjs';
import { mockDevice } from './mock-gpu.mjs';
import { mulberry32 as rng } from '../../js/refs/random.js';

export { rng };
export { metric64, errorsVsBrute, MAX_ERROR_RATE, MAX_ERROR_RATE_PLUS_ONE } from '../../js/refs/voronoi.js';

const f = Math.fround;
export const NONE = 0xffffffff;
export const PARAMS = structLayout(moduleCode({ wgslModules }, 'flood'), 'Params');
export const SEED = structLayout(moduleCode({ wgslModules }, 'flood'), 'Seed');

globalThis.window ??= { addEventListener() {}, removeEventListener() {} };
globalThis.requestAnimationFrame ??= () => 0;

/** A VoronoiJFA on a recording mock device (no GPU). */
export function makeJFA(width, height) {
    const device = mockDevice();
    const canvas = {
        width, height, addEventListener() {}, removeEventListener() {},
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) }),
    };
    return { device, jfa: new VoronoiJFA(device, canvas) };
}

/** Seeds placed by the shipped generate() with Math.random replaced by a seeded generator. */
export function generateSeeded(jfa, count, layout, seed) {
    const saved = Math.random;
    Math.random = rng(seed);
    try { jfa.generate(count, layout); } finally { Math.random = saved; }
}

/** Seed positions as the GPU sees them: decode the bytes writeSeeds() uploads. */
export function uploadedSeeds(device, jfa) {
    jfa.writeSeeds();
    const bytes = device.lastWrite(jfa.seedBuffer).bytes;
    const n = bytes.byteLength / SEED.size;
    const out = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
        const s = decodeStruct(bytes, SEED, i * SEED.size);
        out[2 * i] = s.pos[0]; out[2 * i + 1] = s.pos[1];
    }
    return out;
}

/** f32 metricDistance (Euclidean returns the squared distance). */
export function metricF32(cx, cy, sx, sy, metric) {
    const dx = Math.abs(f(cx - sx)), dy = Math.abs(f(cy - sy));
    if (metric === 1) return f(dx + dy);
    if (metric === 2) return Math.max(dx, dy);
    return f(f(dx * dx) + f(dy * dy));
}

/** Replica of cs_clear + cs_seed. */
export function seedPass(w, h, seeds) {
    const ids = new Uint32Array(w * h).fill(NONE);
    const count = seeds.length / 2;
    for (let i = 0; i < count; i++) {
        const px = Math.min(Math.max(Math.floor(seeds[2 * i]), 0), w - 1);
        const py = Math.min(Math.max(Math.floor(seeds[2 * i + 1]), 0), h - 1);
        const at = py * w + px;
        ids[at] = Math.min(ids[at], i);   // atomicMin
    }
    return ids;
}

/** Replica of one cs_flood pass with step k. */
export function floodPass(idsIn, w, h, seeds, k, metric) {
    const count = seeds.length / 2;
    const out = new Uint32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const cx = f(x + 0.5), cy = f(y + 0.5);
        let bestId = NONE, bestDist = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const qx = x + dx * k, qy = y + dy * k;
            if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
            const id = idsIn[qy * w + qx];
            if (id >= count) continue;
            const d = metricF32(cx, cy, seeds[2 * id], seeds[2 * id + 1], metric);
            if (bestId === NONE || d < bestDist || (d === bestDist && id < bestId)) { bestId = id; bestDist = d; }
        }
        out[y * w + x] = bestId;
    }
    return out;
}

/** cs_exact replica: counters[0] strictly-farther pixels, counters[1] differing index (f32). */
export function exactCheck(ids, w, h, seeds, metric) {
    const count = seeds.length / 2;
    const counters = [0, 0];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const cx = f(x + 0.5), cy = f(y + 0.5);
        let bestId = NONE, bestDist = 0;
        for (let i = 0; i < count; i++) {
            const d = metricF32(cx, cy, seeds[2 * i], seeds[2 * i + 1], metric);
            if (bestId === NONE || d < bestDist) { bestId = i; bestDist = d; }
        }
        const j = ids[y * w + x];
        if (j !== bestId) counters[1]++;
        if (j >= count || metricF32(cx, cy, seeds[2 * j], seeds[2 * j + 1], metric) > bestDist) counters[0]++;
    }
    return counters;
}
