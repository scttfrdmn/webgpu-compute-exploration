// Procedural-noise compute-only API (measureNoise in js/examples/noise.js) and the Measure experiments
// built on it (js/teach/experiments/noise.js), on a software GPU — CI only, see harness.mjs. This runs
// the example's own pipeline through a NoiseField without a canvas, exactly as the Measure tab does in a
// browser. Checked against the float64 Perlin / simplex fBm of js/refs/noise.js on sampled pixels. The
// printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureNoise, SIMPLEX_SCALE, SIMPLEX_FALLOFF } from '../../js/examples/noise.js';
import { perlinRef, simplexRef, fbmRef } from '../../js/refs/noise.js';
import experiments, { noisePoint } from '../../js/teach/experiments/noise.js';
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

test('measureNoise: every pixel of a 1-octave field vs float64 Perlin and simplex', async (t) => {
    const side = 96;
    for (const [noiseType, base] of [[0, perlinRef], [1, (x, y, z) => simplexRef(x, y, z, SIMPLEX_SCALE, SIMPLEX_FALLOFF)]]) {
        const r = await measureNoise(gpuDevice, { width: side, height: side, noiseType, octaves: 1, frequency: 6, readback: true });
        let max = 0;
        for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
            const [px, py, pz] = noisePoint(x, y, side, 6);
            max = Math.max(max, Math.abs(r.field[y * side + x] - fbmRef(base, px, py, pz, { octaves: 1 })));
        }
        t.diagnostic(`${noiseType ? 'simplex' : 'Perlin'} ${side}², 1 octave: max |gpu − f64| = ${max.toExponential(2)}; wall ${r.wallMs.toFixed(2)} ms`);
        assert.ok(r.field.every(Number.isFinite));
        assert.ok(max < 1e-4, `max error ${max}`);
    }
});

test('experiment statistics (Perlin, simplex): GPU matches float64; spread falls with octaves', async (t) => {
    const e = byId.statistics;
    for (const noiseType of [0, 1]) {
        const { rows, seconds } = await runAll(e, { noiseType });
        t.diagnostic(`${noiseType ? 'simplex' : 'Perlin'} ${seconds.toFixed(1)} s: octaves mean std min max cpuStd predicted maxErr: `
            + show(rows, ['octaves', 'mean', 'std', 'min', 'max', 'cpuStd', 'predictedStd', 'maxError']));
        assert.equal(rows.length, 6);
        for (const row of rows) {
            for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
            assert.ok(row.min >= -1.05 && row.max <= 1.05 && Math.abs(row.mean) < 0.2);
            assert.ok(row.maxError < 5e-3, `octaves ${row.octaves}: ${row.maxError}`);
            assert.ok(Math.abs(row.std - row.cpuStd) < 0.03, 'the samples see the same spread as the whole field');
        }
        assert.ok(rows[0].maxError < 1e-4);
        assert.ok(rows[5].std < rows[0].std, 'normalized fBm has a smaller spread than one octave');
    }
});

test('experiment cost (512², no warp): time grows with the octaves', async (t) => {
    const e = byId.cost;
    const { rows, seconds } = await runAll(e, { side: 512, warp: 0 });
    t.diagnostic(`${seconds.toFixed(1)} s: octaves perlinWall simplexWall perlinRate simplexRate: `
        + show(rows, ['octaves', 'perlinWall', 'simplexWall', 'perlinRate', 'simplexRate']));
    assert.equal(rows.length, 6);
    for (const row of rows) for (const c of e.columns) assert.ok(c.key in row, `column ${c.key}`);
    assert.ok(rows[5].perlinWall > rows[0].perlinWall && rows[5].simplexWall > rows[0].simplexWall, '8 octaves cost more than 1');
});
