// Ray marching Measure tab, CPU only: compute-only RayMarchingRenderer (canvas = null, offscreen target),
// MARCH_STATS_SHADER and measureRayMarch() in js/examples/ray-marching.js, the experiment in
// js/teach/experiments/ray-marching.js, and the float64 tracer in js/refs/ray-marching.js, on the mock
// device. Checked: the uniforms written for the offscreen target, the statistics module is the shipped
// source plus one compute entry point that copies fs_main's camera and rayMarch's constants, the caps,
// rows carry the declared columns, and the reference tracer on cases with known answers.
// Real frames: test/gpu/ray-marching-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RayMarchingRenderer, measureRayMarch, wgslModules, FRACTALS, MAX_MARCH_STEPS, OUTCOME } from '../js/examples/ray-marching.js';
import experiments, { STEPS, RESOLUTIONS, samplePixels, agreement } from '../js/teach/experiments/ray-marching.js';
import { sceneDE, cameraRay, marchRay } from '../js/refs/ray-marching.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const mod = { wgslModules };
const mockCtx = (device, budgetMs) => makeContext({ device, timestamps: false }, { budgetMs, waitForUI: async () => {}, deviceLost: () => false });

test('statistics module = shipped render source + a compute entry with fs_main\'s camera and rayMarch\'s exits', () => {
    const render = moduleCode(mod, 'render');
    const stats = moduleCode(mod, 'march-stats');
    const fragmentPart = render.slice(render.indexOf('struct Uniforms'));   // everything after the vertex stage
    assert.ok(stats.includes(fragmentPart), 'the shipped fragment source, unchanged');
    const entry = stats.slice(stats.indexOf('fn cs_march_stats'));
    for (const line of ['uv.y = -uv.y;', 'rotateY(ro, uniforms.cameraRotY + uniforms.time * 0.1)', 'if (d < 0.001)', 'if (t > 50.0)', 't += d * 0.5;']) {
        assert.ok(entry.includes(line) && render.includes(line), line);
    }
    assert.deepEqual(OUTCOME, { HIT: 0, ESCAPED: 1, EXHAUSTED: 2 });
});

test('compute-only renderer: offscreen target size in the uniforms; measureRayMarch shape and caps', async () => {
    const device = mockDevice();
    const r = new RayMarchingRenderer(device, null, { width: 320, height: 224 });
    Object.assign(r, { fractalType: 2, maxSteps: 96, time: 0 });
    r.updateUniforms();
    const u = decodeStruct(device.lastWrite(r.uniformBuffer).bytes, structLayout(moduleCode(mod, 'render'), 'Uniforms'));
    assert.deepEqual(u, { resolution: [320, 224], time: 0, power: 8, maxSteps: 96, fractalType: 2, cameraRotX: 0, cameraRotY: 0, cameraZoom: 3 });
    r.destroy();

    const m = await measureRayMarch({ device: mockDevice() }, { width: 40, height: 28, fractal: 'julia', maxSteps: 64, readback: true });
    assert.equal(m.pixels.length, 4 * 40 * 28);
    assert.equal(m.gpuMs, null);
    assert.equal(m.hit + m.escaped + m.exhausted, 1);
    for (const bad of [{ maxSteps: MAX_MARCH_STEPS + 1 }, { width: 1001, height: 701 }, { fractal: 'sierpinski' }, { maxSteps: 0 }]) {
        await assert.rejects(measureRayMarch({ device: mockDevice() }, bad), RangeError);
    }
});

test('reference tracer: Menger far field is an exact box SDF; a ray past everything escapes', () => {
    const menger = sceneDE(FRACTALS.indexOf('menger'));
    assert.ok(Math.abs(menger([0, 0, 3]) - 2) < 1e-12);
    const center = cameraRay(500, 350, 1000, 700, { cameraZoom: 3, cameraRotX: 0, cameraRotY: 0, time: 0 });
    assert.ok(Math.abs(center.rd[2] + 1) < 1e-12, 'the center pixel looks down −z');
    assert.equal(marchRay(menger, center.ro, center.rd, 256).outcome, 1, 'the center ray flies through the sponge\'s central tunnel');
    const hit = marchRay(menger, [0.8, 0.8, 3], [0, 0, -1], 256);   // (0.8, 0.8) on the z = 1 face is solid
    assert.equal(hit.outcome, 0);
    assert.ok(Math.abs(hit.t - 2) < 0.002, `hits the cube face at t ≈ 2 (${hit.t})`);
    const away = marchRay(menger, [0, 0, 3], [0, 0, 1], 256);
    assert.equal(away.outcome, 1, 'escaped');
    assert.equal(marchRay(menger, [0, 0, 3], [0, 0, 1], 5).outcome, 2, 'out of steps first');
    const pts = samplePixels(1000, 700);
    assert.ok(pts.every(([x, y]) => x >= 0 && x < 1000 && y >= 0 && y < 700) && new Set(pts.map(String)).size === pts.length);
    const fake = new Float32Array(4 * 1000 * 700).fill(OUTCOME.ESCAPED);
    assert.equal(agreement(fake, 1000, 700, 2, 256, [[0, 0]]), 1, 'the top-left pixel misses the sponge');
    assert.equal(agreement(fake, 1000, 700, 2, 256, [[0, 0], [700, 455]]), 0.5, 'pixel (700, 455) hits the front face near (0.57, −0.30)');
});

test('experiment on a mock device: one row per step budget, declared columns', async () => {
    const [e] = experiments;
    assert.ok(STEPS.every(s => s <= MAX_MARCH_STEPS) && RESOLUTIONS.every(r => r.value <= 1000));
    const device = mockDevice();
    const rows = await runExperiment(e, { fractal: 'menger', width: 500 }, mockCtx(device, e.budgetMs));
    assert.deepEqual(rows.map(r => r.maxSteps), STEPS);
    for (const row of rows) assert.deepEqual(Object.keys(row).sort(), e.columns.map(c => c.key).sort());
    assert.ok(Math.max(...device.buffers.map(b => b.size)) <= 16 * 1000 * 700);
});
