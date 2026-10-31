// GPU execution of wave-equation.js cs_step / cs_edit on a software GPU.
// CI-only — see test/gpu/harness.mjs. Exact shipped WGSL; walls and both time levels come from the
// shipped loadPreset(), and every substep's Params slot from the shipped writeParams(), captured on
// a mock device. Like WaveEquation.frame(), bind group i has heightCur = buffer i and
// heightPrev = buffer 1 - i, each substep reads its own 256-byte-aligned Params slot (bound here at
// a static offset instead of a dynamic one), and the buffers swap after every step.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computePipeline, bindGroup, buffer, run, read } from './harness.mjs';
import * as wave from '../../js/examples/wave-equation.js';
import { moduleCode } from '../helpers/wgsl-layout.mjs';
import { presetState, stepReplica, energy, SOURCE_X } from '../helpers/wave-equation-refs.mjs';

const { PARAMS_SIZE, SOURCE_PERIOD } = wave;
const CODE = moduleCode(wave, 'compute');
const f = Math.fround;
const MAX_SUBSTEPS = 16;

// cs_step / cs_edit on the GPU, starting from a shipped preset
async function gpuSim(key, W, H) {
    const st = presetState(key, W, H);
    const { sim } = st;
    const stepP = await computePipeline(CODE, 'cs_step');
    const editP = await computePipeline(CODE, 'cs_edit');
    const stride = sim.paramsStride;
    const heights = [buffer(st.cur), buffer(st.prev)];
    const walls = buffer(st.walls);
    const ubuf = buffer(stride * MAX_SUBSTEPS);
    const cache = new Map();
    const group = (p, slot, i) => {
        const k = `${p.entryPoint}/${slot}/${i}`;
        if (!cache.has(k)) cache.set(k, bindGroup(p, 0, {
            heightCur: heights[i], heightPrev: heights[1 - i], walls,
            params: { buffer: ubuf, offset: slot * stride, size: PARAMS_SIZE },
        }));
        return cache.get(k);
    };
    const wg = [Math.ceil(W / 8), Math.ceil(H / 8)];
    let current = 0;
    const lastParams = slots => sim.device.lastWrite(sim.paramsBuffer).bytes.subarray(0, slots * stride);
    return {
        st, sim,
        // n time steps, in frames of up to 16 substeps (one Params write per frame, like frame())
        async steps(n) {
            const list = [];
            while (n > 0) {
                const slots = Math.min(n, MAX_SUBSTEPS);
                sim.writeParams(slots, false, false);
                list.push({ write: [ubuf, Uint8Array.from(lastParams(slots))] });
                for (let i = 0; i < slots; i++) {
                    list.push({ p: stepP, groups: [group(stepP, i, current)], workgroups: wg });
                    current = 1 - current;
                }
                sim.steps += slots;
                n -= slots;
            }
            await run(list, 'cs_step');
        },
        async edit(dropOn, wallOn) {
            sim.writeParams(1, dropOn, wallOn);
            await run([{ write: [ubuf, Uint8Array.from(lastParams(1))] },
                { p: editP, groups: [group(editP, 0, current)], workgroups: wg }], 'cs_edit');
        },
        // [u(t), u(t-1)] and the wall mask
        async state() {
            return [await read(heights[current]), await read(heights[1 - current]), await read(walls, Uint32Array)];
        },
    };
}

// ---- JS ports of the shader's rain and ripple (not in the CPU replica) ----
function pcg(v) {
    const state = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}
const unitFloat = h => (h >>> 8) / 16777216;
function ripple(px, py, cx, cy, sigma) {
    const s = ((px - cx) ** 2 + (py - cy) ** 2) / (2 * sigma * sigma);
    return s > 9 ? 0 : (1 - s) * Math.exp(-s);
}
// Rain drops (center x, y) the shader adds at `step`
function rainDrops(seed, step, W, H) {
    const drops = [];
    for (let k = 0; k < 2; k++) {
        const h = pcg((seed ^ pcg((step * 2 + k) >>> 0)) >>> 0);
        if (unitFloat(h) < 0.03) {
            const h2 = pcg(h), h3 = pcg(h2);
            drops.push([unitFloat(h2) * W, unitFloat(h3) * H]);
        }
    }
    return drops;
}
// Replica step plus rain (added before the hard source overrides its column; walls stay 0)
function replicaStep(s, o) {
    stepReplica(s, o);
    if (!o.rain) return 0;
    const drops = rainDrops(o.seed, o.step, s.W, s.H);
    for (const [cx, cy] of drops) for (let y = 0; y < s.H; y++) for (let x = 0; x < s.W; x++) {
        const i = y * s.W + x;
        if (s.walls[i] || (o.planeWave && x === SOURCE_X)) continue;
        s.cur[i] += 0.35 * ripple(x + 0.5, y + 0.5, cx, cy, 2.0);
    }
    return drops.length;
}

const maxAbs = u => u.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
function maxDiff(a, b) {
    let m = 0;
    for (let i = 0; i < b.length; i++) m = Math.max(m, Number.isFinite(a[i]) ? Math.abs(a[i] - b[i]) : Infinity);
    return m;
}

test('cs_step: ellipse walls, C = 0.5, no damping — discrete energy conserved and replica agreement', async (t) => {
    const W = 96, H = 72;
    const g = await gpuSim('ellipse', W, H);
    Object.assign(g.sim, { courant: 0.5, dampingGamma: 0, rain: false, absorb: false, planeWave: false });
    const C2 = f(0.25);
    const rep = { W, H, cur: Float32Array.from(g.st.cur), prev: Float32Array.from(g.st.prev), walls: g.st.walls };
    const u0 = maxAbs(g.st.cur);
    let e0 = null, maxDrift = 0, repErr = 0;
    for (let chunk = 0; chunk < 60; chunk++) {        // 60 x 25 = 1500 steps, as in the CPU suite
        await g.steps(25);
        const [cur, prev] = await g.state();
        const e = energy(prev, cur, W, H, C2);
        if (e0 === null) e0 = e;
        maxDrift = Math.max(maxDrift, Math.abs(e - e0) / e0);
        if (chunk < 12) {                               // replica for the first 300 steps
            for (let n = 0; n < 25; n++) stepReplica(rep, { courant2: C2, damping: 0, absorb: false, planeWave: false, step: 0, period: SOURCE_PERIOD });
            repErr = Math.max(repErr, maxDiff(cur, rep.cur), maxDiff(prev, rep.prev));
        }
    }
    const [cur, , walls] = await g.state();
    let wallLeak = 0;
    for (let i = 0; i < W * H; i++) if (walls[i] && cur[i] !== 0) wallLeak++;
    t.diagnostic(`96x72 ellipse: max relative energy drift ${maxDrift.toExponential(2)} over 1500 steps; ` +
        `max |gpu - replica| over 300 steps ${repErr.toExponential(2)} (max|u0| ${u0.toFixed(3)}); final max|u| ${maxAbs(cur).toFixed(4)}`);
    assert.ok(maxDrift < 1e-4, `energy drift ${maxDrift} (CPU-suite bound)`);
    // f32 storage in both; the GPU's f32 (possibly fused) arithmetic adds a few ulp per step on a
    // linear, non-amplifying scheme: 1e-4 of the initial amplitude after 300 steps.
    assert.ok(repErr < 1e-4 * u0, `replica error ${repErr}`);
    assert.equal(wallLeak, 0, 'wall cells stay at u = 0');
});

test('cs_step: double-slit preset with sponge, damping, plane-wave source and rain vs replica', async (t) => {
    const W = 128, H = 96, steps = 160;
    const g = await gpuSim('slit', W, H);
    assert.equal(g.sim.planeWave, true);
    assert.equal(g.sim.absorb, true);
    Object.assign(g.sim, { courant: 0.6, dampingGamma: 0.004, rain: true, seedValue: 0x9e3779b9 });
    await g.steps(steps);
    const [cur, prev] = await g.state();
    const rep = { W, H, cur: Float32Array.from(g.st.cur), prev: Float32Array.from(g.st.prev), walls: g.st.walls };
    let drops = 0;
    for (let n = 0; n < steps; n++) {
        drops += replicaStep(rep, { courant2: f(0.36), damping: f(0.002), absorb: true, planeWave: true,
            step: n, period: SOURCE_PERIOD, rain: true, seed: 0x9e3779b9 });
    }
    const err = Math.max(maxDiff(cur, rep.cur), maxDiff(prev, rep.prev));
    t.diagnostic(`128x96 slit: ${steps} steps, ${drops} rain drops, max|u| ${maxAbs(cur).toFixed(3)}, max |gpu - replica| = ${err.toExponential(2)}`);
    assert.ok(drops > 0, 'the rain hash fired at least once');
    // sin() (source) and exp() (rain ripples) are not correctly rounded on GPUs (Vulkan allows
    // ~2^-11 absolute for sin); 2e-3 still rejects any sign, index or hash error (O(0.1) effects).
    assert.ok(err < 2e-3, `max error ${err}`);
    // The hard source column holds SOURCE_AMP·sin(2π·phase) of the last step (index steps - 1)
    const src = 0.6 * Math.sin(2 * Math.PI * ((steps - 1) % SOURCE_PERIOD) / SOURCE_PERIOD);
    for (let y = 0; y < H; y++) assert.ok(Math.abs(cur[y * W + SOURCE_X] - src) < 2e-3, `source column y=${y}`);
});

test('cs_edit: wall stroke and ripple drop on both time levels', async (t) => {
    const W = 64, H = 48;
    const g = await gpuSim('open', W, H);
    const fields = { cursor: [40.3, 30.7], cursorPrev: [12.1, 9.6], dropAmp: 0.8 };
    Object.assign(g.sim, fields);
    await g.edit(true, true);
    const [cur, prev, walls] = await g.state();
    const [ax, ay] = fields.cursorPrev, [bx, by] = fields.cursor;
    let wallCells = 0, ambiguous = 0, wrongWall = 0, hErr = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, px = x + 0.5, py = y + 0.5, abx = bx - ax, aby = by - ay;
        const tt = Math.min(1, Math.max(0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)));
        const d = Math.hypot(px - (ax + abx * tt), py - (ay + aby * tt));
        const wall = d < 2.5 ? 1 : 0;
        if (walls[i] !== wall) { if (Math.abs(d - 2.5) < 1e-4) ambiguous++; else wrongWall++; continue; }
        wallCells += wall;
        // ripple() cuts off at s > 9 (a jump of ~8e-4 here); skip cells within f32 rounding of it
        if (Math.abs(((px - bx) ** 2 + (py - by) ** 2) / 32 - 9) < 1e-4) continue;
        const r = fields.dropAmp * ripple(px, py, bx, by, 4.0);
        const ec = wall ? 0 : g.st.cur[i] + r, ep = wall ? 0 : g.st.prev[i] + r;
        hErr = Math.max(hErr, Math.abs(cur[i] - ec), Math.abs(prev[i] - ep));
    }
    t.diagnostic(`64x48: ${wallCells} wall cells drawn, ${ambiguous} on-radius flips, ${wrongWall} wrong; max |height - f64| = ${hErr.toExponential(2)}`);
    assert.ok(wallCells > 100);
    assert.equal(wrongWall, 0);
    // One f32 add of an exp()-based ripple of amplitude <= 0.8 to values <= ~1
    assert.ok(hErr < 1e-5, `height error ${hErr}`);
});
