// GPU execution of reaction-diffusion.js cs_main (Gray-Scott step + brush) on a software GPU.
// CI-only — see test/gpu/harness.mjs. Exact shipped WGSL; the uniform bytes come from the shipped
// ReactionDiffusion.writeParams() and the initial state from the shipped seed(), both captured on a
// mock device. Ping-pong like ReactionDiffusion.frame(): bind group i reads buffer i, writes 1 - i.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, run, read } from './harness.mjs';
import * as rd from '../../js/examples/reaction-diffusion.js';
import { moduleCode } from '../helpers/wgsl-layout.mjs';
import { mockDevice } from '../helpers/mock-gpu.mjs';
import { stepReplica, stepReference, shippedSeed } from '../helpers/reaction-diffusion-refs.mjs';

globalThis.window ??= { addEventListener() {}, removeEventListener() {} };

// Params bytes exactly as the shipped writer produces them
function shippedParams(w, h, fields, simulate, brushOn) {
    const device = mockDevice();
    const canvas = {
        width: 512, height: 512,
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
        addEventListener() {}, removeEventListener() {},
    };
    const sim = new rd.ReactionDiffusion(device, canvas, w, h);
    Object.assign(sim, fields);
    sim.writeParams(simulate, brushOn);
    return Uint8Array.from(device.lastWrite(sim.paramsBuffer).bytes);
}

// Runs `steps` dispatches of cs_main from `init`; returns the final state (Float32Array of u, v pairs)
async function gpuRun(p, w, h, init, params, steps) {
    const cells = [buffer(init), buffer(init.byteLength)];
    const ubuf = buffer(params);
    const groups = [0, 1].map(i => bindGroup(p, 0, { cellsIn: cells[i], cellsOut: cells[1 - i], params: ubuf }));
    const wg = [Math.ceil(w / 8), Math.ceil(h / 8)];
    const list = [];
    for (let s = 0; s < steps; s++) list.push({ p, groups: [groups[s % 2]], workgroups: wg });
    await run(list, 'cs_main');
    return read(cells[steps % 2], Float32Array, init.byteLength);
}

test('cs_main: 200 Gray-Scott steps per preset vs the float64 3x3-convolution reference', async (t) => {
    const p = await computePipeline(moduleCode(rd, 'compute'), 'cs_main');
    // 45 x 38 is not a multiple of the 8x8 workgroup, so the bounds guard and the toroidal wrap at
    // a ragged edge are both exercised.
    const w = 45, h = 38, steps = 200;
    const init = shippedSeed(w, h, 3);
    let worst = 0;
    for (const [key, { feed, kill }] of Object.entries(rd.PRESETS)) {
        const params = shippedParams(w, h, { feed, kill }, true, false);
        const got = await gpuRun(p, w, h, init, params, steps);

        let U = new Float64Array(w * h), V = new Float64Array(w * h);
        for (let i = 0; i < w * h; i++) { U[i] = init[2 * i]; V[i] = init[2 * i + 1]; }
        let a = Float32Array.from(init), b = new Float32Array(init.length);
        for (let s = 0; s < steps; s++) {
            [U, V] = stepReference(U, V, w, h, feed, kill);
            stepReplica(a, b, w, h, feed, kill); [a, b] = [b, a];
        }
        let dRef = 0, dRep = 0, nonFinite = 0;
        for (let i = 0; i < w * h; i++) {
            if (!Number.isFinite(got[2 * i]) || !Number.isFinite(got[2 * i + 1])) nonFinite++;
            dRef = Math.max(dRef, Math.abs(got[2 * i] - U[i]), Math.abs(got[2 * i + 1] - V[i]));
            dRep = Math.max(dRep, Math.abs(got[2 * i] - a[2 * i]), Math.abs(got[2 * i + 1] - a[2 * i + 1]));
        }
        worst = Math.max(worst, dRef);
        t.diagnostic(`${key} (F=${feed}, k=${kill}): ${steps} steps on ${w}x${h}, max |gpu - f64 reference| = ${dRef.toExponential(2)}, ` +
            `max |gpu - CPU replica| = ${dRep.toExponential(2)}`);
        assert.equal(nonFinite, 0);
    }
    // Same bound as the CPU suite's replica-vs-reference test (f32 storage, 200 steps). The GPU also
    // does the per-cell arithmetic in f32 (and may contract to FMA), which stays well inside it.
    assert.ok(worst < 1e-4, `max error ${worst}`);
});

test('cs_main: brush-only pass (simulate = 0) paints exactly the cells within the stroke radius', async (t) => {
    const p = await computePipeline(moduleCode(rd, 'compute'), 'cs_main');
    const w = 61, h = 29;
    const init = shippedSeed(w, h, 5);
    const brush = { brushPos: [40.25, 20.5], brushPrev: [12.75, 6.0], brushRadius: 4.5, feed: 0.03, kill: 0.06 };
    const got = await gpuRun(p, w, h, init, shippedParams(w, h, brush, false, true), 1);
    // float64 segment distance from each cell center to brushPrev -> brushPos
    const [ax, ay] = brush.brushPrev, [bx, by] = brush.brushPos;
    let painted = 0, ambiguous = 0, wrong = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const px = x + 0.5, py = y + 0.5, abx = bx - ax, aby = by - ay;
        const tt = Math.min(1, Math.max(0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)));
        const d = Math.hypot(px - (ax + abx * tt), py - (ay + aby * tt));
        const i = y * w + x;
        const expect = d < brush.brushRadius ? [0.5, 0.25] : [init[2 * i], init[2 * i + 1]];
        if (d < brush.brushRadius) painted++;
        if (got[2 * i] === expect[0] && got[2 * i + 1] === expect[1]) continue;
        // f32 rounding can only flip cells sitting on the radius
        if (Math.abs(d - brush.brushRadius) < 1e-4) ambiguous++; else wrong++;
    }
    t.diagnostic(`${w}x${h}: ${painted} cells under the stroke, ${ambiguous} on-radius flips, ${wrong} wrong`);
    assert.ok(painted > 50);
    assert.equal(wrong, 0, 'cells away from the radius: painted iff inside, unchanged otherwise (no simulation)');
});
