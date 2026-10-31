// Gray-Scott references shared by test/reaction-diffusion.test.mjs and test/gpu/.
//
// stepReplica: replica of reaction-diffusion.js cs_main (simulate = 1, brush off) with f32 storage
// and float64 per-cell arithmetic. stepReference: independent float64 step convolving an explicit
// 3x3 kernel with modulo-wrapped indices. shippedSeed: the shipped ReactionDiffusion.seed() run on a
// mock device with a deterministic Math.random. DU, DV, DT and the Laplacian weights are parsed
// from the shipped WGSL and passed to the deployable reference in js/refs/reaction-diffusion.js.

import * as rd from '../../js/examples/reaction-diffusion.js';
import { moduleCode } from './wgsl-layout.mjs';
import { mockDevice } from './mock-gpu.mjs';
import { W_CENTER, laplacianKernel, stepReference as stepReference64 } from '../../js/refs/reaction-diffusion.js';

export const CODE = moduleCode(rd, 'compute');
export const constF32 = name => Number(new RegExp(`const ${name}: f32 = ([\\d.]+);`).exec(CODE)[1]);
export const DU = constF32('DU'), DV = constF32('DV'), DT = constF32('DT');
export const lapMatch = /let lap = -c \+ ([\d.]+) \* edges \+ ([\d.]+) \* corners;/.exec(CODE);
if (!lapMatch) throw new Error('Laplacian line not found in the WGSL');
export const W_EDGE = Number(lapMatch[1]), W_CORNER = Number(lapMatch[2]);
export { W_CENTER };

// ---- replica of cs_main (simulate = 1, brush off) ----------------------------------------------
export function stepReplica(cin, cout, w, h, feed, kill) {
    // cellAt(x, y): (x + w) % w, (y + h) % h — the shader's toroidal wrap for offsets of -1..1
    for (let y = 0; y < h; y++) {
        const ym = ((y - 1 + h) % h) * w, y0 = y * w, yp = ((y + 1 + h) % h) * w;
        for (let x = 0; x < w; x++) {
            const xm = (x - 1 + w) % w, xp = (x + 1 + w) % w;
            const i = (y0 + x) * 2;
            const e = [(y0 + xm) * 2, (y0 + xp) * 2, (ym + x) * 2, (yp + x) * 2];
            const c = [(ym + xm) * 2, (ym + xp) * 2, (yp + xm) * 2, (yp + xp) * 2];
            const u = cin[i], v = cin[i + 1];
            const lapU = -u + W_EDGE * (cin[e[0]] + cin[e[1]] + cin[e[2]] + cin[e[3]])
                + W_CORNER * (cin[c[0]] + cin[c[1]] + cin[c[2]] + cin[c[3]]);
            const lapV = -v + W_EDGE * (cin[e[0] + 1] + cin[e[1] + 1] + cin[e[2] + 1] + cin[e[3] + 1])
                + W_CORNER * (cin[c[0] + 1] + cin[c[1] + 1] + cin[c[2] + 1] + cin[c[3] + 1]);
            const uvv = u * v * v;
            const du = DU * lapU - uvv + feed * (1 - u);
            const dv = DV * lapV + uvv - (feed + kill) * v;
            cout[i] = Math.min(Math.max(u + DT * du, 0), 1);
            cout[i + 1] = Math.min(Math.max(v + DT * dv, 0), 1);
        }
    }
}
export function simulate(state, w, h, steps, feed, kill) {
    let a = Float32Array.from(state), b = new Float32Array(state.length);
    for (let s = 0; s < steps; s++) { stepReplica(a, b, w, h, feed, kill); [a, b] = [b, a]; }
    return a;
}

// ---- independent float64 reference (js/refs/reaction-diffusion.js), bound to the WGSL constants --
export const KERNEL = laplacianKernel(W_EDGE, W_CORNER);
export const MODEL = { DU, DV, DT, W_EDGE, W_CORNER };
export const stepReference = (U, V, w, h, F, k) => stepReference64(U, V, w, h, F, k, MODEL);

// Runs the shipped seed() with a deterministic Math.random; returns the uploaded cell data
export function shippedSeed(w, h, seed) {
    const device = mockDevice();
    const sim = Object.create(rd.ReactionDiffusion.prototype);
    Object.assign(sim, { device, width: w, height: h, cellCount: w * h, cellBuffers: [{ id: 0 }, { id: 1 }], render() {} });
    const saved = Math.random;
    let s = seed >>> 0;
    Math.random = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
    try { sim.seed(); } finally { Math.random = saved; }
    const wr = device.lastWrite(sim.cellBuffers[0]);
    return new Float32Array(wr.bytes.buffer, wr.bytes.byteOffset, wr.bytes.byteLength / 4);
}

