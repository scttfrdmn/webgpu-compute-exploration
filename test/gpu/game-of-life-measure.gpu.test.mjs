// Game of Life compute-only API (measureLife in js/examples/game-of-life.js) and its Measure experiments,
// on a software GPU — CI only, see harness.mjs. Checked against the exact CPU rule (js/refs/life.js):
// grids must match cell for cell. The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureLife } from '../../js/examples/game-of-life.js';
import { lifeStep } from '../../js/refs/life.js';
import experiments, { randomGrid, mismatches } from '../../js/teach/experiments/game-of-life.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureLife: 50 generations of a 67 × 45 soup (odd sizes, partial workgroups) equal lifeStep', async (t) => {
    const [w, h] = [67, 45];
    const grid = randomGrid(w, h, 0.35, 9);
    const r = await measureLife(gpuDevice, { grid, width: w, height: h, generations: 50 });
    let cpu = grid;
    for (let g = 0; g < 50; g++) cpu = lifeStep(cpu, w, h);
    t.diagnostic(`mismatches ${mismatches(r.grid, cpu)}, live ${cpu.reduce((s, c) => s + c, 0)}, wall ${r.wallMs.toFixed(2)} ms`);
    assert.equal(mismatches(r.grid, cpu), 0);
});

test('experiment soup (256 × 256): no mismatching cell at any checkpoint', async (t) => {
    const e = experiments.find(x => x.id === 'soup');
    const rows = [];
    for await (const row of e.run(makeTestContext(gpuDevice), { side: 256 })) rows.push(row);
    t.diagnostic('generation density mismatches: ' + rows.map(r => `${r.generation} ${r.density.toFixed(4)} ${r.mismatches}`).join('; '));
    assert.equal(rows.length, 10);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, c.key);
        assert.equal(row.mismatches, 0, `generation ${row.generation}`);
    }
    assert.ok(rows.at(-1).density < rows[0].density);
});

test('experiment throughput: rows up to 2048², positive rates', async (t) => {
    const e = experiments.find(x => x.id === 'throughput');
    const rows = [];
    const t0 = performance.now();
    for await (const row of e.run(makeTestContext(gpuDevice), {})) rows.push(row);
    t.diagnostic(`${((performance.now() - t0) / 1000).toFixed(1)} s: side wallMs/gen Gcells/s(wall) gens/s: ` + rows.map(r => [r.side,
        r.wallMsPerGen.toFixed(4), r.gcellsWall.toFixed(3), r.gensPerSec.toFixed(0)].join(' ')).join('; '));
    assert.deepEqual(rows.map(r => r.side), [128, 256, 512, 1024, 2048]);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, c.key);
        assert.ok(row.gcellsWall > 0);
    }
});
