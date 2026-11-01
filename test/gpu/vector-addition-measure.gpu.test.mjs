// Vector addition compute-only API (measureVectorAdd in js/examples/vector-addition.js) and its Measure
// experiment, on a software GPU — CI only, see harness.mjs. Checked against the exact reference: every
// sum equals Math.fround(a + b). The printed numbers are the CI baselines quoted in the experiment file.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { device } from './harness.mjs';
import { measureVectorAdd } from '../../js/examples/vector-addition.js';
import experiments, { makeInputs, roundingMismatches } from '../../js/teach/experiments/vector-addition.js';
import { makeTestContext } from '../../js/teach/measure.js';

const gpuDevice = { device, timestamps: device.features.has('timestamp-query') };

test('measureVectorAdd: correctly rounded sums, masked tail', async (t) => {
    for (const n of [1000, 1_000_000]) {
        const { a, b } = makeInputs(n, 2);
        const r = await measureVectorAdd(gpuDevice, { a, b, repeats: 2 });
        let maxRel = 0;
        for (let i = 0; i < n; i++) maxRel = Math.max(maxRel, Math.abs(r.sum[i] - (a[i] + b[i])) / Math.abs(a[i] + b[i]));
        t.diagnostic(`n=${n}: mismatches ${roundingMismatches(a, b, r.sum)}, max rel ${maxRel.toExponential(2)}, `
            + `kernel wall ${r.kernelWallMs.toFixed(3)} ms, round trip ${r.roundTripMs.toFixed(3)} ms`);
        assert.equal(roundingMismatches(a, b, r.sum), 0);
        assert.ok(maxRel <= 2 ** -24);
    }
});

test('experiment bandwidth: every row exact, times positive', async (t) => {
    const [e] = experiments;
    const rows = [];
    for await (const row of e.run(makeTestContext(gpuDevice), {})) rows.push(row);
    t.diagnostic('N gpuMs kernelWallMs roundTripMs GB/s(wall) GB/s(roundtrip): ' + rows.map(r => [r.n, r.gpuMs?.toFixed(3) ?? '—',
        r.kernelWallMs.toFixed(3), r.roundTripMs.toFixed(3), r.gbWall?.toFixed(2), r.gbRoundTrip?.toFixed(3)].join(' ')).join('; '));
    assert.equal(rows.length, 7);
    for (const row of rows) {
        for (const c of e.columns) assert.ok(c.key in row, c.key);
        assert.equal(row.mismatches, 0, `N = ${row.n}`);
        assert.ok(row.kernelWallMs > 0 && row.roundTripMs > 0);
    }
    assert.ok(rows.at(-1).roundTripMs > rows[0].roundTripMs, 'the round trip grows with N');
});
