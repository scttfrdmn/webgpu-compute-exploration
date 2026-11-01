// Teaching-layer framework (js/teach/): deep-link routes, the Measure runner's limits, CSV/table
// formatting, and the chart's pure scale/tick helpers. No DOM and no GPU: the runner gets a mock device.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRoute, formatRoute, TABS } from '../js/teach/routes.js';
import {
    runExperiment, makeContext, makeTestContext, ExperimentStopped, MAX_BUDGET_MS, MAX_WARMUP, MAX_REPEATS, STOP_MESSAGES,
    INTRO, outcomeOf, measureRepeated, toCSV, formatCell, formatRowCell,
} from '../js/teach/measure.js';
import { niceTicks, logTicks, makeScale, extent, formatTick, rangeBars, rangeExtentKeys, formatWithRange } from '../js/teach/chart.js';
import { withSpread, rangeOf, hasRange } from '../js/teach/stats.js';

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
    assert.equal(formatCell({}, 0.5), '0.5');
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

test('runner: stop reasons and messages are consistent (#76)', async () => {
    const steps = { async *run(c) { for (let i = 0; i < 100; i++) { await c.step(); yield { i }; } } };
    const controller = new AbortController();
    controller.abort();
    let t = 0;
    const cases = [
        [makeContext(fakeGPU(), { signal: controller.signal, waitForUI: async () => {}, deviceLost: () => false }), 'cancelled', /^Cancelled$/],
        [makeContext(fakeGPU(), { waitForUI: async () => {}, deviceLost: () => true }), 'device-lost', /^Stopped: the GPU device was lost$/],
        [makeContext(fakeGPU(), { budgetMs: 5000, now: () => (t += 1000), waitForUI: async () => {}, deviceLost: () => false }),
            'budget', /^Stopped at the first step boundary after the 5 s time budget$/],
    ];
    for (const [ctx, reason, message] of cases) {
        const error = await runExperiment(steps, {}, ctx).catch(e => e);
        assert.ok(error instanceof ExperimentStopped);
        assert.equal(error.reason, reason);
        assert.match(error.message, message);
        assert.deepEqual(outcomeOf(error), { status: reason, message: error.message });
        assert.equal(STOP_MESSAGES[reason](ctx.budgetMs), error.message);
    }
    assert.deepEqual(outcomeOf(null), { status: 'done' });
    assert.deepEqual(outcomeOf(new TypeError('boom')), { status: 'error', message: 'Error: boom' });
    // the intro states the budget precisely: checked between steps, each step capped
    assert.match(INTRO, /checked between steps/);
    assert.match(INTRO, /first step boundary after it/);
    assert.match(INTRO, /fixed, capped amount of GPU work/);
});

test('ctx.measure: warm-ups discarded, repeats kept, step() between every two runs, statistics per quantity', async () => {
    const gpu = fakeGPU();
    const ctx = makeContext(gpu, { waitForUI: async () => {}, deviceLost: () => false });
    const calls = [];
    let k = 0;
    const walls = [100, 1, 2, 3, 4, 5];               // the 100 ms warm-up must not show up
    const m = await ctx.measure(async (info) => {
        calls.push({ ...info, stepsBefore: gpu.pendingCount() });
        const wallMs = walls[k++];
        return { wallMs, gpuMs: wallMs / 2, phases: { compile: 10, upload: 0.5, dispatch: wallMs, readback: 0.25 }, tag: k };
    }, { warmup: 1, repeats: 5 });
    assert.deepEqual(calls.map(c => [c.warmup, c.index]), [[true, 0], [false, 0], [false, 1], [false, 2], [false, 3], [false, 4]]);
    assert.deepEqual(calls.map(c => c.stepsBefore), [0, 1, 2, 3, 4, 5], 'one ctx.step() between every two runs');
    assert.equal(m.warmup, 1);
    assert.equal(m.repeats, 5);
    assert.equal(m.n, 5);
    assert.deepEqual(m.samples.map(s => s.tag), [2, 3, 4, 5, 6]);
    assert.deepEqual([m.wallMs.n, m.wallMs.median, m.wallMs.min, m.wallMs.max, m.wallMs.q1, m.wallMs.q3], [5, 3, 1, 5, 2, 4]);
    assert.equal(m.gpuMs.median, 1.5);
    assert.deepEqual(Object.keys(m.phases).sort(), ['compile', 'dispatch', 'readback', 'upload']);
    assert.equal(m.phases.compile.median, 10);
    assert.equal(m.phases.compile.cv, 0);
    assert.equal(m.phases.dispatch.median, 3);
    assert.deepEqual(ctx.measured, { calls: 1, warmup: 1, repeats: 5 });

    // no GPU time (no timestamp-query): gpuMs has n = 0 and null statistics; wall only
    const w = await ctx.measure(async () => ({ wallMs: 2, gpuMs: null }), { warmup: 0, repeats: 3 });
    assert.equal(w.gpuMs.n, 0);
    assert.equal(w.gpuMs.median, null);
    assert.deepEqual(w.phases, {});
    assert.equal(w.wallMs.median, 2);
    assert.deepEqual(ctx.measured, { calls: 2, warmup: 0, repeats: 3 });

    // caps and defaults
    let runs = 0;
    const capped = await measureRepeated(async () => ({ wallMs: runs++ }), { warmup: 99, repeats: 999 });
    assert.equal(capped.warmup, MAX_WARMUP);
    assert.equal(capped.repeats, MAX_REPEATS);
    assert.equal(runs, MAX_WARMUP + MAX_REPEATS);
    const defaults = await measureRepeated(async () => ({ wallMs: 1 }));
    assert.deepEqual([defaults.warmup, defaults.repeats], [1, 5]);
});

test('ctx.measure: cancel and budget stop it between repeats', async () => {
    const controller = new AbortController();
    const ctx = makeContext(fakeGPU(), { signal: controller.signal, waitForUI: async () => {}, deviceLost: () => false });
    let runs = 0;
    await assert.rejects(ctx.measure(async () => { if (++runs === 3) controller.abort(); return { wallMs: 1 }; }, { warmup: 1, repeats: 5 }),
        (e) => e instanceof ExperimentStopped && e.reason === 'cancelled');
    assert.equal(runs, 3, 'the run in flight finishes; the next one does not start');
    assert.equal(ctx.measured.calls, 0);

    let t = 0;
    const budgeted = makeContext(fakeGPU(), { budgetMs: 2500, now: () => t, waitForUI: async () => {}, deviceLost: () => false });
    runs = 0;
    await assert.rejects(budgeted.measure(async () => { runs++; t += 1000; return { wallMs: 1000 }; }, { warmup: 1, repeats: 5 }),
        (e) => e instanceof ExperimentStopped && e.reason === 'budget');
    assert.equal(runs, 3, 'stops at the first boundary after 2.5 s');

    // the CI test context has the same measure, without a budget
    const testCtx = makeTestContext({ device: { queue: { onSubmittedWorkDone: async () => {} } } });
    const m = await testCtx.measure(async () => ({ wallMs: 1 }), { repeats: 2 });
    assert.equal(m.n, 2);
    assert.equal(testCtx.measured.repeats, 2);
});

test('ranges: row-key convention, table cells, CSV columns, chart range bars', () => {
    const row = { n: 4, ...withSpread('ms', { median: 2, q1: 1.5, q3: 3 }), other: 7 };
    assert.deepEqual(row, { n: 4, ms: 2, msLo: 1.5, msHi: 3, other: 7 });
    assert.deepEqual(withSpread('x', null), { x: null, xLo: null, xHi: null });
    assert.deepEqual(rangeOf(row, 'ms'), [1.5, 3]);
    assert.equal(rangeOf(row, 'other'), null);
    assert.equal(rangeOf({ a: 1, aLo: null, aHi: 2 }, 'a'), null);
    assert.ok(hasRange([{}, row], 'ms'));
    assert.ok(!hasRange([row], 'other'));

    const fmt = (v) => v.toFixed(1);
    assert.equal(formatWithRange(fmt, row, 'ms'), '2.0 (1.5–3.0)');
    assert.equal(formatWithRange(fmt, row, 'other'), '7.0');
    assert.equal(formatRowCell({ key: 'ms', format: fmt }, row), '2.0 (1.5–3.0)');
    assert.equal(formatRowCell({ key: 'ms' }, { ms: null, msLo: 1, msHi: 2 }), '— (1–2)');
    assert.equal(formatRowCell({ key: 'other' }, row), '7');

    // CSV: q1/q3 columns only for columns with a spread; rows without one stay empty there
    const cols = [{ key: 'n', label: 'N' }, { key: 'ms', label: 'wall ms' }];
    assert.equal(toCSV(cols, [row, { n: 8, ms: 5 }]), 'N,wall ms,wall ms q1,wall ms q3\n4,2,1.5,3\n8,5,,\n');
    assert.equal(toCSV(cols, [{ n: 8, ms: 5 }]), 'N,wall ms\n8,5\n', 'no spread, no extra columns');

    // Chart: extents cover the ranges; bars at each ranged point, dodged per series, log-clamped
    assert.deepEqual(rangeExtentKeys(['a', 'b']), ['aLo', 'aHi', 'bLo', 'bHi']);
    const rows = [{ x: 1, a: 2, aLo: 1, aHi: 4, b: 3 }, { x: 2, a: 5, aLo: 0, aHi: 6, b: 4, bLo: 3, bHi: 5 }];
    assert.deepEqual(extent(rows, ['a', 'b', ...rangeExtentKeys(['a', 'b'])], false), { min: 0, max: 6 });
    const id = (v) => v * 10;
    const bars = rangeBars(rows, 'x', ['a', 'b'], id, id);
    assert.deepEqual(bars, [
        { key: 'a', series: 0, x: 8, y1: 10, y2: 40 }, { key: 'a', series: 0, x: 18, y1: 0, y2: 60 },
        { key: 'b', series: 1, x: 22, y1: 30, y2: 50 },
    ]);
    // one ranged series: no dodge; on a log axis a non-positive low end clamps to the axis minimum
    const logBars = rangeBars(rows, 'x', ['a'], id, id, { logY: true, yMin: 0.5 });
    assert.deepEqual(logBars.map(b => [b.x, b.y1, b.y2]), [[10, 10, 40], [20, 5, 60]]);
    assert.deepEqual(rangeBars([{ x: 1, a: 1 }], 'x', ['a'], id, id), [], 'no ranges: nothing drawn (old experiments unchanged)');
});
