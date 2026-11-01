// Test-only CPU replica of the cloth compute shader (js/examples/cloth.js computeShader): cs_integrate,
// cs_tether, cs_solve per color, cs_collide and cs_normals, driven by the SimParams bytes the shipped
// writer produces. Used by test/cloth.test.mjs (CPU) and test/gpu/cloth-measure.gpu.test.mjs, which
// compares the shipped WGSL on a software GPU with it. Validates the algorithm and index math, not the
// WGSL itself — see #51.

import {
    ClothSimulation, OrbitCamera, NONE, NUM_COLORS, KIND_STRUCTURAL, KIND_BEND, SUBSTEP_DT, MAX_SPEED,
    WIND_NORMAL_DRAG, WIND_TANGENT_DRAG, wgslModules,
} from '../../js/examples/cloth.js';
import { structLayout, decodeStruct } from './wgsl-layout.mjs';
import { mockDevice } from './mock-gpu.mjs';

const SIM = structLayout(wgslModules().find(m => m.name === 'compute').code, 'SimParams');

// A ClothSimulation without a constructor run (no GPU objects): the uniform writers on a mock device
export function simWithMock(fields = {}) {
    const device = mockDevice();
    const sim = Object.create(ClothSimulation.prototype);
    Object.assign(sim, {
        device, canvas: { width: 1000, height: 500 },
        gridN: 64, presetKey: 'curtain', iterations: 8, stiffness: 0.9, windStrength: 5, gravity: 9.8,
        tearOn: true, tearRatio: 1.8, time: 3.25, grabIndex: NONE, grabTarget: [0, 0, 0], grabPos: [0, 0, 0],
        camera: new OrbitCamera(),
        simParams: new ArrayBuffer(128), cameraData: new ArrayBuffer(144),
        simParamsBuffer: { label: 'sim' }, cameraBuffer: { label: 'camera' },
    }, fields);
    sim.simF32 = new Float32Array(sim.simParams);
    sim.simU32 = new Uint32Array(sim.simParams);
    sim.camF32 = new Float32Array(sim.cameraData);
    sim.camU32 = new Uint32Array(sim.cameraData);
    return { sim, device };
}

// Unpacks a decoded SimParams into plain numbers the replica uses
export function makeState(cloth, P) {
    return {
        n: cloth.n, count: cloth.count, P,
        pos: Float32Array.from(cloth.positions),
        prev: Float32Array.from(cloth.positions),
        normals: new Float32Array(cloth.count * 4),
        cu: new Uint32Array(cloth.constraintData), cf: new Float32Array(cloth.constraintData),
        lu: new Uint32Array(cloth.linkData), lf: new Float32Array(cloth.linkData),
        broken: new Uint32Array(cloth.numConstraints), torn: 0, ranges: cloth.ranges, pinnedCount: cloth.pinnedCount,
    };
}

export function windAt(P, x) {
    const t = P.time, g = P.wind[3], mean = P.wind.slice(0, 3);
    const strength = Math.hypot(...mean);
    const surge = 1 + g * 0.5 * (Math.sin(1.7 * t + 1.3 * x[0] + 0.7 * x[1]) + 0.5 * Math.sin(2.9 * t - 0.9 * x[2] + 1.1 * x[1]));
    const cw = [Math.sin(2.3 * t + 2.1 * x[0] + 0.5 * x[2]), 0.4 * Math.sin(1.9 * t + 1.7 * x[2] + 0.8 * x[0]), Math.cos(1.6 * t + 2.4 * x[1] + 1.3 * x[0])];
    return [0, 1, 2].map(k => mean[k] * surge + cw[k] * (strength * g * 0.35));
}
export function gripped(S, i) {
    const P = S.P;
    if (P.grabIndex === NONE) return false;
    const n = P.gridN, r = P.grabRadius;
    return Math.abs((i % n) - (P.grabIndex % n)) <= r && Math.abs(Math.floor(i / n) - Math.floor(P.grabIndex / n)) <= r;
}
export function integrate(S) {
    const { pos, prev, normals, P } = S;
    const h = P.dt;
    for (let i = 0; i < S.count; i++) {
        const o = i * 4;
        if (gripped(S, i)) {
            for (let k = 0; k < 3; k++) { pos[o + k] += P.grabDelta[k]; prev[o + k] = pos[o + k]; }
            prev[o + 3] = pos[o + 3];
            continue;
        }
        if (pos[o + 3] === 0) { prev.set(pos.subarray(o, o + 4), o); continue; }
        const p = [pos[o], pos[o + 1], pos[o + 2]];
        let vel = [0, 1, 2].map(k => (p[k] - prev[o + k]) * (P.damping / h));
        const speed = Math.hypot(...vel);
        if (speed > MAX_SPEED) vel = vel.map(v => v * (MAX_SPEED / speed));
        const nn = [normals[o], normals[o + 1], normals[o + 2]];
        const w = windAt(P, p);
        const rel = [0, 1, 2].map(k => w[k] - vel[k]);
        const dn = nn[0] * rel[0] + nn[1] * rel[1] + nn[2] * rel[2];
        for (let k = 0; k < 3; k++) {
            const accel = P.gravity[k] + WIND_NORMAL_DRAG * dn * nn[k] + WIND_TANGENT_DRAG * rel[k];
            prev[o + k] = p[k];
            pos[o + k] = p[k] + vel[k] * h + accel * (h * h);
        }
    }
}
export function tetherStep(S) {
    if (S.torn !== 0) return;
    const { pos, lu, lf } = S;
    for (let i = 0; i < S.count; i++) {
        const o = i * 4;
        if (pos[o + 3] === 0 || gripped(S, i)) continue;
        let x = [pos[o], pos[o + 1], pos[o + 2]];
        for (const [pin, rest] of [[lu[i * 12 + 5], lf[i * 12 + 8]], [lu[i * 12 + 6], lf[i * 12 + 9]]]) {
            if (pin === NONE) continue;
            const anchor = [pos[pin * 4], pos[pin * 4 + 1], pos[pin * 4 + 2]];
            const d = [0, 1, 2].map(k => x[k] - anchor[k]);
            const l = Math.hypot(...d);
            if (l > rest) x = [0, 1, 2].map(k => anchor[k] + d[k] * (rest / l));
        }
        pos[o] = x[0]; pos[o + 1] = x[1]; pos[o + 2] = x[2];
    }
}
// One cs_solve dispatch for a color; `order` permutes the invocations (they must not interact)
export function solveColor(S, color, order) {
    const { pos, cu, cf, broken, P } = S;
    const r = S.ranges[color];
    for (let g = 0; g < r.count; g++) {
        const ci = r.start + (order ? order[g] : g);
        if (broken[ci] !== 0) continue;
        const o = ci * 8;
        const a = cu[o], b = cu[o + 1], rest = cf[o + 2], kind = cu[o + 3], pa = cu[o + 4], pb = cu[o + 5];
        let parentTorn = false;
        if (pa !== NONE) parentTorn = broken[pa] !== 0;
        if (pb !== NONE) parentTorn = parentTorn || broken[pb] !== 0;
        if (parentTorn) { broken[ci] = 1; S.torn++; continue; }
        const d = [0, 1, 2].map(k => pos[b * 4 + k] - pos[a * 4 + k]);
        const l = Math.hypot(...d);
        if (l < 1e-7) continue;
        if (P.tearOn !== 0 && kind < KIND_BEND && l > rest * P.tearRatio) { broken[ci] = 1; S.torn++; continue; }
        const wa = gripped(S, a) ? 0 : pos[a * 4 + 3];
        const wb = gripped(S, b) ? 0 : pos[b * 4 + 3];
        const wsum = wa + wb;
        if (wsum <= 0) continue;
        const s = P.stiffness[kind] * (l - rest) / (l * wsum);
        for (let k = 0; k < 3; k++) {
            pos[a * 4 + k] += d[k] * s * wa;
            pos[b * 4 + k] -= d[k] * s * wb;
        }
    }
}
export function contact(P, x, v, n, depth) {
    const vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
    const vt = [0, 1, 2].map(k => v[k] - vn * n[k]);
    const slide = Math.hypot(...vt);
    const limit = P.friction * depth;
    const kept = slide > limit ? vt.map(c => c * (1 - limit / slide)) : [0, 0, 0];
    return { x: [0, 1, 2].map(k => x[k] - (vt[k] - kept[k])), v: [0, 1, 2].map(k => kept[k] + Math.max(vn, 0) * n[k]) };
}
export function collide(S) {
    const { pos, prev, P } = S;
    for (let i = 0; i < S.count; i++) {
        const o = i * 4;
        if (pos[o + 3] === 0 || gripped(S, i)) continue;
        let x = [pos[o], pos[o + 1], pos[o + 2]];
        let v = [0, 1, 2].map(k => x[k] - prev[o + k]);
        let hit = false;
        if (P.sphereOn !== 0) {
            const c = P.sphere.slice(0, 3);
            const d = [0, 1, 2].map(k => x[k] - c[k]);
            const r = P.sphere[3] + P.thickness;
            const dist = Math.hypot(...d);
            if (dist < r && dist > 1e-6) {
                const nrm = d.map(z => z / dist);
                ({ x, v } = contact(P, nrm.map((z, k) => c[k] + z * r), v, nrm, r - dist));
                hit = true;
            }
        }
        const floorTop = P.floorY + P.thickness;
        if (x[1] < floorTop) {
            ({ x, v } = contact(P, [x[0], floorTop, x[2]], v, [0, 1, 0], floorTop - x[1]));
            hit = true;
        }
        if (hit) {
            for (let k = 0; k < 3; k++) { pos[o + k] = x[k]; prev[o + k] = x[k] - v[k]; }
        }
    }
}
export function computeNormals(S) {
    const { pos, lu, broken, normals, n } = S;
    const P3 = (i) => [pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]];
    const ok = (ci) => ci !== NONE && broken[ci] === 0;
    for (let i = 0; i < S.count; i++) {
        const L = i * 12, p = P3(i);
        const xp = ok(lu[L + 1]) ? P3(i + 1) : p, xn = ok(lu[L]) ? P3(i - 1) : p;
        const yp = ok(lu[L + 3]) ? P3(i + n) : p, yn = ok(lu[L + 2]) ? P3(i - n) : p;
        const a = [0, 1, 2].map(k => xp[k] - xn[k]), b = [0, 1, 2].map(k => yp[k] - yn[k]);
        const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        const l = Math.hypot(...c);
        normals.set(l > 1e-12 ? [c[0] / l, c[1] / l, c[2] / l, 0] : [0, 0, 0, 0], i * 4);
    }
}
// encodeCompute(): substeps × (integrate, tether if pinned, iterations × colors, collide), normals
export function frame(S, substeps, iterations, orders) {
    for (let s = 0; s < substeps; s++) {
        integrate(S);
        if (S.pinnedCount > 0) tetherStep(S);
        for (let it = 0; it < iterations; it++) {
            for (let c = 0; c < NUM_COLORS; c++) if (S.ranges[c].count) solveColor(S, c, orders?.[c]);
        }
        collide(S);
    }
    computeNormals(S);
}
// SimParams exactly as the shipped writer encodes them
export function simParamsFor(fields) {
    const { sim, device } = simWithMock(fields);
    sim.writeSimParams(4);
    return decodeStruct(device.lastWrite(sim.simParamsBuffer).bytes, SIM);
}
export function maxStretch(S, kinds = [KIND_STRUCTURAL]) {
    let m = 0;
    for (let ci = 0; ci < S.cu.length / 8; ci++) {
        if (S.broken[ci] || !kinds.includes(S.cu[ci * 8 + 3])) continue;
        const a = S.cu[ci * 8], b = S.cu[ci * 8 + 1];
        const l = Math.hypot(S.pos[b * 4] - S.pos[a * 4], S.pos[b * 4 + 1] - S.pos[a * 4 + 1], S.pos[b * 4 + 2] - S.pos[a * 4 + 2]);
        m = Math.max(m, l / S.cf[ci * 8 + 2]);
    }
    return m;
}
export function energy(S, g) {
    let e = 0;
    for (let i = 0; i < S.count; i++) {
        if (S.pos[i * 4 + 3] === 0) continue;
        let v2 = 0;
        for (let k = 0; k < 3; k++) v2 += ((S.pos[i * 4 + k] - S.prev[i * 4 + k]) / SUBSTEP_DT) ** 2;
        e += 0.5 * v2 + g * S.pos[i * 4 + 1];
    }
    return e;
}
