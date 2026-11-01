// Falling-sand compute-only API (measureSand in js/examples/falling-sand.js) and the Measure experiments
// built on it (js/teach/experiments/falling-sand.js), on a software GPU — CI only, see harness.mjs. This
// runs the example's own pipeline and dynamic-offset bind group through a FallingSand without a canvas,
// exactly as the Measure tab does in a browser. Checked against exact material counts
// (js/refs/falling-sand.js): with no reactions every count is unchanged; reactions change only the
// materials they involve. The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureSand, MATERIAL_IDS as M } from '../../js/examples/falling-sand.js';
import { materialCounts, MATERIAL_NAMES } from '../../js/refs/falling-sand.js';
import experiments, { scene, SANDBOX_W, SANDBOX_H } from '../../js/teach/experiments/falling-sand.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };
const byId = Object.fromEntries(experiments.map(e => [e.id, e]));

async function runAll(experiment, settings) {
    const rows = [];
    const t0 = performance.now();
    for await (const row of experiment.run(makeTestContext(gpuDevice), settings)) rows.push(row);
    return { rows, seconds: (performance.now() - t0) / 1000 };
}
const show = (rows, keys) => rows.map(r => keys.map(k => {
    const v = r[k];
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toExponential(2)) : String(v);
}).join(' ')).join('; ');

test('measureSand (no reactions): counts exact, settling sand, batched and single submits give the same grid', async (t) => {
    const grid = scene('inert');
    const c0 = materialCounts(grid);
    const a = await measureSand(gpuDevice, { width: SANDBOX_W, height: SANDBOX_H, grid, steps: 200, stepsPerSubmit: 8, warmup: 0, readAt: [200] });
    const b = await measureSand(gpuDevice, { width: SANDBOX_W, height: SANDBOX_H, grid, steps: 200, stepsPerSubmit: 1, readAt: [200] });
    const cells = a.samples[0].cells;
    const c = materialCounts(cells);
    t.diagnostic(`after 200 steps: ${MATERIAL_NAMES.map((n, i) => `${n} ${c[i]}`).join(', ')}; ids ≥ 8: ${c.slice(8).reduce((s, x) => s + x, 0)}; `
        + `wall ms/step ${a.wallMsPerStep.toFixed(3)} (8 per submit) vs ${b.wallMsPerStep.toFixed(3)} (1 per submit)`);
    assert.deepEqual(c, c0, 'every material count unchanged');
    // the step counter (and so every random choice) depends only on the step index, not on batching
    assert.deepEqual(b.samples[0].cells, cells, 'deterministic: same grid with 1 or 8 steps per submit');
    // the sand heap has started to fall: its top rows (y 30-34 in the scene) are emptier
    let top = 0;
    for (let y = 30; y < 35; y++) for (let x = 20; x < 70; x++) if ((cells[y * SANDBOX_W + x] & 15) === M.SAND) top++;
    assert.ok(top < 250, `sand left in the heap's top rows: ${top} of 250`);
});

test('experiment conservation: exact counts without reactions; reactions change only their materials', async (t) => {
    const e = byId.conservation;
    const out = {};
    for (const key of ['plants', 'fire']) {
        const { rows, seconds } = await runAll(e, { scene: key });
        t.diagnostic(`${key} ${seconds.toFixed(1)} s: steps Δsand Δstone Δoil Δwater Δplant Δ(w+p) fire changed: `
            + show(rows, ['steps', 'sand', 'stone', 'oil', 'water', 'plant', 'waterPlant', 'fire', 'changed']));
        assert.equal(rows.length, 6);
        for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        out[key] = rows;
    }
    for (const r of out.plants) {
        assert.deepEqual([r.sand, r.stone, r.oil, r.waterPlant, r.fire], [0, 0, 0, 0, 0], 'only water turns into plant');
        assert.ok(r.plant >= 0);
    }
    assert.ok(out.plants[5].plant > 0, 'the plants grew');
    for (const r of out.fire) assert.deepEqual([r.sand, r.stone], [0, 0]);
    assert.ok(out.fire[5].oil < 0 && out.fire[5].fire === 0, 'fire burns oil, then goes out');
});

test('experiment scaling (up to 640 x 480): rows, positive throughput', async (t) => {
    const e = byId.scaling;
    const { rows, seconds } = await runAll(e, { maxCells: 640 * 480 });
    t.diagnostic(`${seconds.toFixed(1)} s: size steps wall8 wall1 Mcells/s: ${show(rows, ['size', 'steps', 'wall8', 'wall1', 'rate'])}`);
    assert.deepEqual(rows.map(r => r.cells), [19200, 76800, 307200]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.wall8 > 0 && row.wall1 > 0 && row.rate > 0);
    }
});
