// Teaching-layer framework (js/teach/): deep-link routes, the Measure runner's limits, CSV/table
// formatting, and the chart's pure scale/tick helpers. No DOM and no GPU: the runner gets a mock device.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRoute, formatRoute, TABS } from '../js/teach/routes.js';
import { runExperiment, makeContext, ExperimentStopped, MAX_BUDGET_MS, toCSV, formatCell } from '../js/teach/measure.js';
import { niceTicks, logTicks, makeScale, extent, formatTick } from '../js/teach/chart.js';

const ids = new Set(['nbody', 'fft-audio']);

test('routes: parse and format deep links', () => {
    assert.deepEqual(TABS, ['explore', 'understand', 'measure']);
    assert.deepEqual(parseRoute('#nbody', ids), { id: 'nbody', tab: 'explore' });
    assert.deepEqual(parseRoute('#nbody/understand', ids), { id: 'nbody', tab: 'understand' });
    assert.deepEqual(parseRoute('#fft-audio/measure', ids), { id: 'fft-audio', tab: 'measure' });
    for (const bad of ['', '#', '#unknown', '#nbody/other', '#nbody/measure/x', '#/measure']) {
        assert.equal(parseRoute(bad, ids), null, bad);
    }
    assert.equal(formatRoute('nbody', 'explore'), '#nbody');
    assert.equal(formatRoute('nbody', 'measure'), '#nbody/measure');
    for (const tab of TABS) assert.deepEqual(parseRoute(formatRoute('nbody', tab), ids), { id: 'nbody', tab });
});

function fakeGPU() {
    let pending = 0;
    return { device: { queue: { onSubmittedWorkDone: async () => { pending++; } } }, timestamps: false, pendingCount: () => pending };
}

test('runner: collects one row per yield and waits for the GPU between steps', async () => {
    const gpu = fakeGPU();
    const ctx = makeContext(gpu, { budgetMs: 1000, waitForUI: async () => {}, deviceLost: () => false });
    const experiment = { async *run(c, settings) { for (const n of settings.sizes) { await c.step(); yield { n, ms: n / 10 }; } } };
    const seen = [];
    const rows = await runExperiment(experiment, { sizes: [1, 2, 4] }, ctx, (row) => seen.push(row.n));
    assert.deepEqual(rows, [{ n: 1, ms: 0.1 }, { n: 2, ms: 0.2 }, { n: 4, ms: 0.4 }]);
    assert.deepEqual(seen, [1, 2, 4]);
    assert.equal(gpu.pendingCount(), 3);
});

test('runner: cancel, time budget and device loss stop the run between steps', async () => {
    const steps = (ctx) => ({ async *run(c) { for (let i = 0; i < 100; i++) { await c.step(); yield { i }; } } });

    const controller = new AbortController();
    let ctx = makeContext(fakeGPU(), { signal: controller.signal, waitForUI: async () => {}, deviceLost: () => false });
    const rows = [];
    await assert.rejects(runExperiment(steps(), {}, ctx, (row) => { rows.push(row); if (row.i === 2) controller.abort(); }),
        (e) => e instanceof ExperimentStopped && /Cancelled/.test(e.message));
    assert.equal(rows.length, 3);

    let t = 0;
    ctx = makeContext(fakeGPU(), { budgetMs: 50, now: () => t, waitForUI: async () => { t += 20; }, deviceLost: () => false });
    await assert.rejects(runExperiment(steps(), {}, ctx), (e) => e instanceof ExperimentStopped && /budget/.test(e.message));

    let lost = false;
    ctx = makeContext(fakeGPU(), { waitForUI: async () => {}, deviceLost: () => lost });
    await assert.rejects(runExperiment(steps(), {}, ctx, (row) => { if (row.i === 1) lost = true; }),
        (e) => e instanceof ExperimentStopped && /lost/.test(e.message));
});

test('runner: budgets are capped at MAX_BUDGET_MS', () => {
    let t = 0;
    const ctx = makeContext(fakeGPU(), { budgetMs: 10 * MAX_BUDGET_MS, now: () => t, deviceLost: () => false });
    assert.equal(ctx.remainingMs(), MAX_BUDGET_MS);
    t = MAX_BUDGET_MS + 1;
    assert.throws(() => ctx.check(), ExperimentStopped);
});

test('results: CSV escaping and cell formatting', () => {
    const cols = [{ key: 'n', label: 'N' }, { key: 'note', label: 'note, "quoted"' }, { key: 'err', label: 'error' }];
    const csv = toCSV(cols, [{ n: 1024, note: 'a,b', err: 1.5e-7 }, { n: 2048, note: 'x\ny', err: undefined }]);
    assert.equal(csv, 'N,"note, ""quoted""",error\n1024,"a,b",1.5e-7\n2048,"x\ny",\n');
    assert.equal(formatCell({}, 1234567), '1,234,567');
    assert.equal(formatCell({}, 0.000012), '1.20e-5');
    assert.equal(formatCell({}, 3.14159), '3.14');
    assert.equal(formatCell({}, NaN), '—');
    assert.equal(formatCell({ format: v => `${v} ms` }, 2), '2 ms');
});

test('chart: nice linear ticks, decade log ticks, scales and extents', () => {
    assert.deepEqual(niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
    assert.deepEqual(niceTicks(3, 97, 5), [20, 40, 60, 80]);
    assert.deepEqual(logTicks(1, 1e4), [1, 10, 100, 1000, 10000]);
    assert.deepEqual(logTicks(2, 80), [2, 5, 10, 20, 50]);
    assert.deepEqual(logTicks(-1, 10), []);
    const s = makeScale({ min: 1, max: 100, log: true }, [0, 200]);
    assert.equal(s(1), 0); assert.equal(s(10), 100); assert.equal(s(100), 200);
    const lin = makeScale({ min: 0, max: 10 }, [100, 0]);
    assert.equal(lin(5), 50);
    assert.deepEqual(extent([{ a: 1, b: 5 }, { a: -2, b: NaN }], ['a', 'b'], false), { min: -2, max: 5 });
    assert.deepEqual(extent([{ a: -1 }, { a: 4 }], ['a'], true), { min: 2, max: 8 });
    assert.equal(extent([{ a: 'x' }], ['a'], false), null);
    assert.equal(formatTick(25000), '25K');
    assert.equal(formatTick(0.5), '0.5');
    assert.equal(formatTick(1e-5), '1e-5');
});
