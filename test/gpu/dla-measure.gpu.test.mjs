// 3D DLA compute-only API (measureDLA in js/examples/dla-3d.js) and the Measure experiments built on it
// (js/teach/experiments/dla.js), on a software GPU — CI only, see harness.mjs. This runs the example's
// own walk/stick pipeline through a DLA3D without a canvas, exactly as the Measure tab does in a browser.
// Checked with js/refs/dla.js: the aggregate is 26-connected to the seed (the stick rule needs an
// occupied neighbour, which atomics cannot break), nearly every stuck walker adds a cell, and the
// mass-radius dimension is in the DLA range. The printed numbers are the CI baselines quoted in the
// experiment file and the Understand page.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureDLA } from '../../js/examples/dla-3d.js';
import { aggregateStats } from '../../js/refs/dla.js';
import experiments from '../../js/teach/experiments/dla.js';
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
    return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v);
}).join(' ')).join('; ');

test('measureDLA (64³, 5,000 walkers, stickiness 1): connected aggregate, DLA dimension', async (t) => {
    const t0 = performance.now();
    const r = await measureDLA(gpuDevice, { gridSize: 64, walkers: 5000, stickiness: 1, rngSeed: 1, steps: 20000, checkEvery: 100, stopAt: 0.99 });
    const a = aggregateStats(r.grid, 64);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: ${r.steps} steps, ${r.stuck} stuck, ${a.cells} cells `
        + `(${r.stuck - (a.cells - r.seedCells)} walkers stuck on an occupied cell), R_g ${a.rg.toFixed(2)}, R_max ${a.rmax.toFixed(1)}, `
        + `D ${a.dimension.toFixed(3)}, connected ${(100 * a.connected).toFixed(2)}%; wall ${r.wallMsPerStep.toFixed(3)} ms/step; `
        + `progress ${r.progress.filter((_, i) => i % 3 === 0).map(p => `${p.step}:${p.stuck}`).join(' ')}`);
    assert.ok(r.stuck >= 0.99 * 5000);
    assert.equal(a.connected, 1, 'every cell 26-connected to the seed');
    assert.ok(a.cells > 0.9 * r.stuck && a.cells <= r.stuck + 1, 'nearly every stuck walker adds a new cell');
    assert.ok(a.dimension > 2.0 && a.dimension < 2.9, `D = ${a.dimension}`);
});

test('experiment stickiness (32³, 1,200 walkers): connected, more compact and slower at low stickiness', async (t) => {
    const e = byId.stickiness;
    const { rows, seconds } = await runAll(e, { grid: 32 });
    t.diagnostic(`${seconds.toFixed(1)} s: s steps half stuck cells Rg D connected ms: `
        + show(rows, ['stickiness', 'steps', 'half', 'stuck', 'cells', 'rg', 'dimension', 'connected', 'msPerStep']));
    assert.equal(rows.length, 6);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.equal(row.connected, 1);
        assert.ok(row.stuck >= 0.95 * 1200, `stickiness ${row.stickiness}: ${row.stuck} stuck`);
    }
    assert.ok(rows[5].rg < rows[0].rg, 'low stickiness packs the aggregate closer to the seed');
    assert.ok(rows[5].half > rows[0].half, 'and takes longer to grow');
});

test('experiment scaling (up to 65,536 walkers): rows, positive throughput', async (t) => {
    const e = byId.scaling;
    const { rows, seconds } = await runAll(e, { maxWalkers: 65536 });
    t.diagnostic(`${seconds.toFixed(1)} s: walkers wallMs/step Msteps/s: ${show(rows, ['walkers', 'wallMsPerStep', 'rate'])}`);
    assert.deepEqual(rows.map(r => r.walkers), [1024, 4096, 16384, 65536]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
        assert.ok(row.wallMsPerStep > 0 && row.rate > 0);
    }
});
