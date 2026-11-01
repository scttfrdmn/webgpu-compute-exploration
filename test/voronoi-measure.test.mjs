// Voronoi Measure tab, CPU only: the compute-only API of js/examples/voronoi-jfa.js (measureJFA) and the
// experiments in js/teach/experiments/voronoi.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: a compute-only VoronoiJFA needs no canvas, window or requestAnimationFrame; the Params and
// Seed bytes match the WGSL structs (one Params slot per flood pass, the given grid and seeds); the ids
// grid can be read back; the caps are enforced; the seed generator is deterministic and stays on the
// grid; every sweep stays within its caps and every row carries the declared columns. The real results
// are checked on a software GPU in test/gpu/voronoi-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureJFA, jfaSteps, VoronoiJFA, MAX_SEEDS, MAX_MEASURE_SIDE, MAX_EXACT_WORK, wgslModules } from '../js/examples/voronoi-jfa.js';
import experiments, {
    makeSeeds, LAYOUTS, SEED_COUNTS, SIDES, ERROR_SIDE, TRIALS, CPU_CHECK_MAX, RESOLUTION_SEEDS,
} from '../js/teach/experiments/voronoi.js';
import { runExperiment, makeContext } from '../js/teach/measure.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const PARAMS = structLayout(moduleCode({ wgslModules }, 'flood'), 'Params');
const SEED = structLayout(moduleCode({ wgslModules }, 'flood'), 'Seed');
const paramWrites = (device) => device.writes.filter(w => w.buffer.size === 256 * 16 && (w.buffer.usage & GPUBufferUsage.UNIFORM));

test('compute-only VoronoiJFA: no canvas, no window, no render pipeline', () => {
    assert.equal(typeof globalThis.window, 'undefined');
    assert.equal(typeof globalThis.requestAnimationFrame, 'undefined');
    let rendered = 0;
    const device = mockDevice();
    const counting = new Proxy(device, { get: (t, p) => (p === 'createRenderPipeline' ? () => { rendered++; } : t[p]) });
    const jfa = new VoronoiJFA(counting, null, { width: 300, height: 200 });
    assert.equal(rendered, 0);
    assert.deepEqual([jfa.width, jfa.height, jfa.pixelCount], [300, 200, 60000]);
    jfa.edited();   // would schedule a redraw with a canvas
    jfa.destroy();
});

test('measureJFA on a mock device: Params per pass, Seed bytes, id buffers, result shape', async () => {
    const device = mockDevice();
    const w = 300, h = 200;
    const seeds = makeSeeds(37, w, h, 'clusters');
    const r = await measureJFA({ device }, { width: w, height: h, seeds, metric: 2, plusOne: true, warmup: 1, repeats: 2, check: true, readIds: true });
    const steps = jfaSteps(w, h, true);
    const writes = paramWrites(device);
    assert.equal(writes.length, 4, '1 warm-up + 2 timed + the exact check');
    for (const wr of writes) {
        assert.equal(wr.bytes.byteLength, 256 * steps.length);
        steps.forEach((step, k) => {
            const p = decodeStruct(wr.bytes, PARAMS, 256 * k);
            assert.deepEqual({ size: p.size, stepSize: p.stepSize, metric: p.metric, seedCount: p.seedCount },
                { size: [w, h], stepSize: step, metric: 2, seedCount: 37 });
        });
    }
    const seedWrite = device.writes.find(x => x.bytes.byteLength === 37 * SEED.size);
    for (let i = 0; i < 37; i++) {
        const s = decodeStruct(seedWrite.bytes, SEED, i * SEED.size);
        assert.deepEqual(s.pos, [seeds[2 * i], seeds[2 * i + 1]]);
    }
    assert.equal(device.buffers.filter(b => b.size === 4 * w * h && (b.usage & GPUBufferUsage.COPY_SRC)).length, 2, 'id buffers readable');
    assert.deepEqual({ passes: r.passes, seeds: r.seeds, metric: r.metric, plusOne: r.plusOne }, { passes: steps.length, seeds: 37, metric: 2, plusOne: true });
    assert.equal(r.ids.length, w * h);
    assert.equal(r.wrong, 0);
    assert.equal(r.gpuMs, null);
    assert.ok(r.wallMs >= 0);
});

test('measureJFA enforces the caps', async () => {
    const device = mockDevice();
    const s = makeSeeds(4, 64, 64);
    await assert.rejects(measureJFA({ device }, { width: 0, height: 64, seeds: s }), RangeError);
    await assert.rejects(measureJFA({ device }, { width: 64, height: MAX_MEASURE_SIDE + 1, seeds: s }), RangeError);
    await assert.rejects(measureJFA({ device }, { width: 64, height: 64, seeds: new Float32Array(0) }), RangeError);
    await assert.rejects(measureJFA({ device }, { width: 64, height: 64, seeds: new Float32Array(2 * (MAX_SEEDS + 1)) }), RangeError);
    await assert.rejects(measureJFA({ device }, { width: 64, height: 64, seeds: s, metric: 3 }), RangeError);
    await assert.rejects(measureJFA({ device }, { width: 2048, height: 2048, seeds: makeSeeds(256, 2048, 2048), check: true }), RangeError);
});

test('makeSeeds: deterministic, on the grid, both layouts', () => {
    for (const layout of Object.keys(LAYOUTS)) {
        const a = makeSeeds(500, 640, 480, layout, 3);
        assert.deepEqual(a, makeSeeds(500, 640, 480, layout, 3));
        assert.notDeepEqual(a, makeSeeds(500, 640, 480, layout, 4));
        for (let i = 0; i < 500; i++) assert.ok(a[2 * i] >= 0 && a[2 * i] < 640 && a[2 * i + 1] >= 0 && a[2 * i + 1] < 480);
        // one seed per pixel (as cs_seed sees it: floor of the f32 position), even when crowded
        const crowded = makeSeeds(1024, 64, 64, layout, 5);
        const pixels = new Set(Array.from({ length: 1024 }, (_, i) => Math.floor(crowded[2 * i + 1]) * 64 + Math.floor(crowded[2 * i])));
        assert.equal(pixels.size, 1024, `${layout}: distinct pixels`);
    }
    assert.throws(() => makeSeeds(1025, 64, 64), RangeError);
});

test('experiment sweeps stay within the caps', () => {
    assert.ok(SEED_COUNTS.every(c => c <= MAX_SEEDS && ERROR_SIDE ** 2 * c <= MAX_EXACT_WORK));
    assert.ok(SIDES.every(s => s <= MAX_MEASURE_SIDE) && RESOLUTION_SEEDS <= MAX_SEEDS);
    assert.ok(TRIALS <= 3 && CPU_CHECK_MAX <= 2 ** 24);
    for (const e of experiments) assert.ok(e.budgetMs <= 20000);
});

test('experiments run on a mock device: one row per sweep value, declared columns', async (t) => {
    const cases = [
        ['error-rate', { metric: 0, layout: 'random' }, SEED_COUNTS.length],
        ['error-rate', { metric: 1, layout: 'clusters' }, SEED_COUNTS.length],
        ['resolution', { plusOne: false }, SIDES.length],
        ['resolution', { plusOne: true }, SIDES.length],
    ];
    for (const [id, settings, expected] of cases) {
        const experiment = experiments.find(e => e.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${id} ${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.equal(rows.length, expected);
        for (const row of rows) assert.deepEqual(Object.keys(row).sort(), experiment.columns.map(c => c.key).sort(), `${id}: row keys`);
        if (id === 'error-rate') {
            // the float64 CPU check runs only where pixels × seeds ≤ CPU_CHECK_MAX
            for (const row of rows) assert.equal(row.cpuWrong === null, ERROR_SIDE ** 2 * row.seeds > CPU_CHECK_MAX);
        }
    }
});
