// Parallel sort Measure tab, CPU only: the compute-only API of js/examples/parallel-sort.js (measureSort)
// and the experiment in js/teach/experiments/parallel-sort.js, driven through the mock device
// (test/helpers/mock-gpu.mjs), which records buffers and uniform writes but runs nothing.
// Checked: a compute-only ParallelSort creates no render pipeline or canvas context; the uniform bytes
// measureSort writes match the WGSL structs; the key buffers are sized for the padded length; the caps
// are enforced; the sweep respects its caps and the "Largest N" setting, and every row carries the
// declared columns; measureSort reports its wall-clock phases; the experiment follows the measurement
// protocol (one ctx.measure per N, warm-up + repeats, fresh seeded keys per trial, the same keys for every
// algorithm within a trial, medians with q1/q3); verify() and interpret() on synthetic rows (known slopes
// agree, wrong ones disagree, too few sizes cannot decide), crossovers and the ranking; the scope note; the
// timer-resolution helpers (performance.now() resolution, granularity of quantized medians, the CPU sort
// batch), the fit range (≥ 1 ms and resolvable, up to the largest N) and the timestamp consistency check.
// test/parallel-sort-bundles.test.mjs applies interpret() to the recorded Apple silicon runs.
// The real sorts are checked on a software GPU in test/gpu/parallel-sort-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureSort, computeSorter, generateKeys, MEASURE_MAX_KEYS, ParallelSort, sortInternals as S, wgslModules } from '../js/examples/parallel-sort.js';
import experiments, {
    COUNTS, MAX_N, ALGORITHMS, KEY, PROTOCOL, LARGE_N, LARGE_PROTOCOL, protocolFor, keySeed, MODELS, predictedSlope,
    SLOPE_TOLERANCE, crossover, f32KeyMapFailures, FIT_MIN_MS, MIN_FIT_SIZES, RESOLUTION_FACTOR, timerResolution, granularity,
    resolutions, resolvable, clockFor, fitRange, cpuSortBatch, calibrateCpuBatch, cpuBatchTarget, cpuRepsCap, CPU_MAX_REPS, CPU_MAX_BATCH_KEYS,
} from '../js/teach/experiments/parallel-sort.js';
import { runExperiment, makeContext, MAX_REPEATS, MAX_WARMUP } from '../js/teach/measure.js';
import { PHASES } from '../js/phase-clock.js';
import { structLayout, decodeStruct } from './helpers/wgsl-layout.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const L = {
    bitonic: structLayout(wgslModules().find(m => m.name === 'bitonic').code, 'BitonicParams'),
    radix: structLayout(wgslModules().find(m => m.name === 'radix').code, 'RadixParams'),
};
const isUniform = b => (b.usage & GPUBufferUsage.UNIFORM) !== 0;

test('compute-only ParallelSort: no canvas context, no render pipeline', () => {
    const device = mockDevice();
    let rendered = 0;
    const counting = new Proxy(device, { get: (t, p) => (p === 'createRenderPipeline' ? () => { rendered++; } : t[p]) });
    const sorter = new ParallelSort(counting, null);
    assert.equal(rendered, 0);
    assert.equal(sorter.context, undefined);
    assert.ok(sorter.bitonicPipelines && sorter.radixPipelines, 'compute pipelines exist');
    sorter.destroy();
});

test('measureSort on a mock device: RadixParams / BitonicParams bytes, padded buffers, result shape', async () => {
    let device = mockDevice();
    const n = 5000;
    const r = await measureSort({ device }, { algorithm: 'radix', n, dataType: 'u32' });
    const radix = device.writes.find(w => isUniform(w.buffer) && w.buffer.size === S.RADIX_PASSES * 256);
    for (let p = 0; p < S.RADIX_PASSES; p++) {
        assert.deepEqual(decodeStruct(radix.bytes, L.radix, 256 * p), { shift: 4 * p, count: n, numTiles: Math.ceil(n / S.RADIX_TILE), pad0: 0 });
    }
    assert.ok(device.buffers.filter(b => b.size === 4 * n && (b.usage & GPUBufferUsage.STORAGE)).length >= 2, 'keysA/keysB of n keys');
    assert.deepEqual({ algorithm: r.algorithm, n: r.n, padded: r.padded, dataType: r.dataType, dispatches: r.dispatches },
        { algorithm: 'radix', n, padded: n, dataType: 'u32', dispatches: 24 });
    assert.equal(r.gpuTimes.length, 1, 'one timed sort by default (one independent measurement)');
    assert.equal(r.bestKernelMs, null, 'no timestamp-query on the mock');
    assert.ok(r.cpuMs >= 0 && r.bestMs >= 0);

    device = mockDevice();
    const b = await measureSort({ device }, { algorithm: 'bitonic', n, dataType: 'f32' });
    const stages = S.bitonicSchedule(8192, false);
    const w = device.writes.find(x => isUniform(x.buffer) && x.buffer.size === stages.length * 256);
    stages.forEach((stage, s) => assert.deepEqual(decodeStruct(w.bytes, L.bitonic, 256 * s), { k: stage.k, j: stage.j, threads: 4096, pad0: 0 }));
    assert.ok(device.buffers.some(x => x.size === 4 * 8192 && (x.usage & GPUBufferUsage.STORAGE)), 'keys padded to 8,192');
    assert.equal(b.padded, 8192);
    assert.equal(b.dispatches, stages.length + 2, 'f32 adds encode + decode');
});

test('measureSort enforces the caps', async () => {
    const device = mockDevice();
    await assert.rejects(measureSort({ device }, { algorithm: 'quick', n: 1024 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, dataType: 'f64' }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 0 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'bitonic', n: MEASURE_MAX_KEYS + 1 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, runs: 0 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, runs: 11 }), RangeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, keys: new Uint32Array(1000) }), TypeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, dataType: 'f32', keys: new Uint32Array(1024) }), TypeError);
    await assert.rejects(measureSort({ device }, { algorithm: 'radix', n: 1024, expected: new Float32Array(1024) }), TypeError);
    assert.equal(MEASURE_MAX_KEYS, S.RADIX_MAX_KEYS);
});

test('measureSort phases: compile / upload / dispatch / readback; given keys are uploaded, given expected skips the CPU sort', async () => {
    const n = 4096;
    const keys = generateKeys(n, 'u32', 7);
    const expected = Uint32Array.from(keys).sort();
    for (const runs of [1, 3]) {
        const device = mockDevice();
        const r = await measureSort({ device }, { algorithm: 'radix', n, keys, expected, runs });
        assert.deepEqual(Object.keys(r.phases).sort(), [...PHASES].sort());
        for (const v of Object.values(r.phases)) assert.ok(v >= 0);
        assert.equal(r.gpuTimes.length, runs);
        assert.ok(Math.abs(r.phases.dispatch - r.gpuTimes.reduce((a, b) => a + b, 0)) < 1e-9, 'dispatch = the sorts\' wall times');
        assert.equal(r.cpuMs, null, 'no CPU sort when expected is given');
        assert.equal(r.gpuMs, null);
        assert.ok(r.wallMs >= Math.min(...r.gpuTimes) && r.wallMs <= Math.max(...r.gpuTimes));
        const uploads = device.writes.filter(w => w.bytes.byteLength === 4 * n && (w.buffer.usage & GPUBufferUsage.STORAGE));
        assert.equal(uploads.length, runs, 'the keys are uploaded once per sort');
        for (const u of uploads) assert.deepEqual(new Uint32Array(u.bytes.buffer), keys);
        assert.equal(r.mismatches, expected.filter(x => x !== 0).length, 'the mock reads back zeros');
    }
    // A shared sorter is reused, not destroyed; without one, measureSort builds its own
    const device = mockDevice();
    const sorter = computeSorter({ device });
    let destroyed = 0;
    const destroy = sorter.destroy.bind(sorter);
    sorter.destroy = () => { destroyed++; destroy(); };
    await measureSort({ device }, { algorithm: 'bitonic', n, dataType: 'f32', keys: generateKeys(n, 'f32', 1), sorter });
    assert.equal(destroyed, 0);
    sorter.destroy();
    // generateKeys: seeded and reproducible; f32 keys finite and in ±1e6
    assert.deepEqual(generateKeys(1000, 'u32', 3), generateKeys(1000, 'u32', 3));
    assert.notDeepEqual(generateKeys(1000, 'u32', 3), generateKeys(1000, 'u32', 4));
    assert.ok(generateKeys(1000, 'f32', 3).every(x => Number.isFinite(x) && Math.abs(x) <= 1e6));
});

test('experiment sweep stays within the caps and the budget plan', () => {
    assert.ok(COUNTS.every(n => n <= MAX_N) && MAX_N <= MEASURE_MAX_KEYS);
    assert.deepEqual(ALGORITHMS, ['bitonic', 'bitonic-shared', 'radix']);
    for (const p of [PROTOCOL, LARGE_PROTOCOL]) assert.ok(p.warmup >= 1 && p.warmup <= MAX_WARMUP && p.repeats >= 3 && p.repeats <= MAX_REPEATS);
    assert.equal(protocolFor(LARGE_N), LARGE_PROTOCOL);
    assert.equal(protocolFor(LARGE_N / 2), PROTOCOL);
    // Keys sorted per run at the default "Largest N": (warm-up + repeats) × (3 GPU sorts + 1 CPU sort) per N
    const keys = COUNTS.reduce((s, n) => s + n * (protocolFor(n).warmup + protocolFor(n).repeats) * 4, 0);
    assert.ok(keys < 1.6e8, `${keys} keys sorted per run`);
    for (const e of experiments) {
        assert.ok(e.budgetMs <= 20000);
        for (const s of e.settings ?? []) for (const o of s.options) if (typeof o.value === 'number') assert.ok(o.value <= MAX_N);
    }
    // Seeds: distinct per (N, trial), deterministic
    const seeds = COUNTS.flatMap(n => [100, 1, 2, 3, 4, 5].map(t => keySeed(n, t)));
    assert.equal(new Set(seeds).size, seeds.length);
    assert.equal(keySeed(65536, 2), keySeed(65536, 2));
});

test('scope note: shows / does not show, as lists of sentences', () => {
    const { scope } = experiments[0];
    assert.deepEqual(Object.keys(scope).sort(), ['doesNotShow', 'shows']);
    assert.ok(scope.shows.length >= 3 && scope.doesNotShow.length >= 4);
    const text = scope.doesNotShow.join(' ');
    for (const pattern of [/one device/, /sorted/, /adversarial/, /one thread/, /upload/]) assert.match(text, pattern);
});

// The key uploads of one run: writes of 4·N bytes (N in COUNTS) to a storage buffer, in order
const keyUploads = (device) => device.writes.filter(w => (w.buffer.usage & GPUBufferUsage.STORAGE) && COUNTS.includes(w.bytes.byteLength / 4));
const digest = (bytes) => { let h = 0; for (let i = 0; i < bytes.length; i += 97) h = (Math.imul(h, 31) + bytes[i]) >>> 0; return h; };

test('experiment runs on a mock device: the protocol per N, fresh seeded keys per trial, declared columns', async (t) => {
    const experiment = experiments[0];
    const digests = [];
    for (const settings of [{ dataType: 'u32', maxN: 262144 }, { dataType: 'f32', maxN: 262144 }, { dataType: 'u32', maxN: 262144 }]) {
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: experiment.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const t0 = performance.now();
        const rows = await runExperiment(experiment, settings, ctx);
        t.diagnostic(`${JSON.stringify(settings)}: ${rows.length} rows in ${(performance.now() - t0).toFixed(0)} ms on the mock`);
        assert.deepEqual(rows.map(r => r.n), COUNTS.filter(n => n <= 262144));
        assert.equal(ctx.measured.calls, rows.length, 'one ctx.measure per N');
        assert.deepEqual([ctx.measured.warmup, ctx.measured.repeats], [PROTOCOL.warmup, PROTOCOL.repeats]);
        const spread = ['bitonicWall', 'sharedWall', 'radixWall', 'cpuMs', 'uploadMs', 'readbackMs', 'setupMs', 'bitonicGpu', 'sharedGpu', 'radixGpu'];
        const allowed = [...experiment.columns.map(c => c.key), ...spread.flatMap(k => [`${k}Lo`, `${k}Hi`])].sort();
        rows.forEach((row, i) => {
            assert.deepEqual(Object.keys(row).sort(), allowed, 'declared columns and their spreads, nothing else');
            assert.equal(row.repeats, PROTOCOL.repeats);
            for (const k of ['bitonicWall', 'sharedWall', 'radixWall', 'cpuMs', 'uploadMs', 'readbackMs', 'setupMs']) {
                assert.ok(row[`${k}Lo`] <= row[k] && row[k] <= row[`${k}Hi`], `${k}: q1 ≤ median ≤ q3`);
            }
            for (const k of ['bitonicGpu', 'sharedGpu', 'radixGpu']) assert.equal(row[k], null, 'no timestamp-query on the mock');
            assert.equal(row.pipelineMs === null, i > 0, 'pipeline compile on the first row only');
            assert.ok(Number.isInteger(row.cpuReps) && row.cpuReps >= 1 && row.cpuReps <= cpuRepsCap(row.n), `cpuReps ${row.cpuReps}`);
            assert.ok(row.timerMs > 0 && row.timerMs < 1, `performance.now() resolution ${row.timerMs} ms in Node`);
            if (i === 0) t.diagnostic(`N = ${row.n}: ${row.cpuReps} CPU sorts per sample, CPU ${row.cpuMs.toFixed(3)} ms per sort, timer ${row.timerMs} ms`);
            assert.match(row.dispatches, /^\d+ \/ \d+ \/ \d+$/);
            for (const a of ALGORITHMS) assert.ok(row[`${KEY[a]}Wrong`] > 0, 'the mock reads back zeros, which must not verify');
        });
        // Every trial uploads one key set three times (one per algorithm); key sets differ between trials
        const uploads = keyUploads(device);
        const trials = rows.length * (PROTOCOL.warmup + PROTOCOL.repeats);
        assert.equal(uploads.length, 3 * trials);
        const sets = [];
        for (let i = 0; i < uploads.length; i += 3) {
            const [a, b, c] = uploads.slice(i, i + 3).map(u => digest(u.bytes));
            assert.ok(a === b && b === c, 'the same keys for every algorithm within a trial');
            sets.push(a);
        }
        assert.equal(new Set(sets).size, sets.length, 'fresh keys in every trial');
        const want = generateKeys(COUNTS[0], settings.dataType, keySeed(COUNTS[0], 100));
        assert.deepEqual(new Uint32Array(uploads[0].bytes.buffer), new Uint32Array(want.buffer), 'warm-up keys from the fixed seed sequence');
        digests.push(sets);
        // verify() on these rows fails every sort check
        const checks = experiment.verify(rows, settings);
        assert.deepEqual(checks.slice(0, 3).map(c => c.status), ['fail', 'fail', 'fail']);
        assert.equal(checks.length, settings.dataType === 'f32' ? 5 : 4);
        const counts = device.writes.filter(w => isUniform(w.buffer) && w.buffer.size === S.RADIX_PASSES * 256)
            .map(w => decodeStruct(w.bytes, L.radix).count);
        assert.ok(counts.length === trials && Math.max(...counts) <= 262144);
    }
    assert.deepEqual(digests[2], digests[0], 'a second run uses exactly the same keys (reproducible)');
});

// ---- verify / interpret on synthetic rows ---------------------------------------------------------------

const e = experiments[0];
// Rows whose medians follow times[series](n), with q1/q3 at ±spread and a deterministic ±jitter
function syntheticRows(ns, times, { spread = 0.02, jitter = 0.01, gpu = true } = {}) {
    return ns.map((n, i) => {
        const row = { n, repeats: protocolFor(n).repeats, bitonicWrong: 0, sharedWrong: 0, radixWrong: 0 };
        const put = (key, v) => {
            row[key] = v;
            row[`${key}Lo`] = v === null ? null : v * (1 - spread);
            row[`${key}Hi`] = v === null ? null : v * (1 + spread);
        };
        const j = 1 + jitter * (i % 2 ? 1 : -1);
        for (const k of ['bitonic', 'shared', 'radix']) {
            put(`${k}Wall`, times[k](n) * j);
            put(`${k}Gpu`, gpu ? 0.8 * times[k](n) * j : null);
        }
        put('cpuMs', times.cpu(n) * j);
        return row;
    });
}
const NS = COUNTS;   // 16,384 … 4,194,304
// A "real GPU": costs in ms such that radix is fastest and global bitonic slowest at the largest N
const GOOD = {
    bitonic: (n) => 2e-7 * MODELS.bitonic(n),
    shared: (n) => 0.8e-7 * MODELS.sharedCompute(n),
    radix: (n) => 1.2e-6 * MODELS.radix(n),
    cpu: (n) => 4e-6 * MODELS.cpu(n),
};
const byClaim = (claims, prefix) => claims.find(c => c.claim.startsWith(prefix));
const SLOPES = ['Bitonic sort (global)', 'Bitonic sort (workgroup', 'Radix sort', 'CPU sort'];

test('verify(): pass with zero mismatches, fail with any, not-checked without rows; f32 key map checks', () => {
    const rows = syntheticRows(NS.slice(0, 3), GOOD);
    let checks = e.verify(rows, { dataType: 'u32' });
    assert.deepEqual(checks.map(c => c.status), ['pass', 'pass', 'pass', 'not-checked']);
    assert.match(checks[0].reference, /Uint32Array\.prototype\.sort.*18 sorts/);
    checks = e.verify(rows, { dataType: 'f32' });
    assert.deepEqual(checks.map(c => c.status), ['pass', 'pass', 'pass', 'pass', 'pass']);
    assert.match(checks[0].reference, /Float32Array/);
    const bad = rows.map((r, i) => (i === 1 ? { ...r, sharedWrong: 3 } : r));
    checks = e.verify(bad, { dataType: 'u32' });
    assert.deepEqual(checks.slice(0, 3).map(c => [c.status, c.value]), [['pass', 0], ['fail', 3], ['pass', 0]]);
    assert.deepEqual(e.verify([], { dataType: 'u32' }).map(c => c.status), Array(4).fill('not-checked'));
    assert.deepEqual(f32KeyMapFailures(1000), { values: 1014, roundTrip: 0, order: 0 });
});

test('interpret(): predicted slopes come from the models at the fitted N', () => {
    const tail = NS.slice(-4);
    const mid = Math.log(Math.sqrt(tail[0] * tail[tail.length - 1]));
    const close = (a, b, tol) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);
    // the local slopes of the asymptotic forms at the middle of the range: N log² N ≈ 1 + 2/ln N, N log N ≈ 1 + 1/ln N
    close(predictedSlope(MODELS.bitonic, tail), 1 + 2 / mid, 0.01);
    close(predictedSlope(MODELS.cpu, tail), 1 + 1 / mid, 0.01);
    close(predictedSlope(MODELS.radix, tail), 1, 1e-12);
    assert.ok(predictedSlope(MODELS.sharedTraffic, tail) > predictedSlope(MODELS.sharedCompute, tail) + 0.05,
        'the workgroup variant\'s dispatch count grows faster than its stage count');
});

test('interpret(): known slopes agree, wrong slopes disagree, too few sizes cannot decide', () => {
    const claims = e.interpret(syntheticRows(NS, GOOD));
    assert.equal(claims.length, 8);
    for (const c of claims) {
        for (const k of ['claim', 'predicted', 'measured']) assert.ok(typeof c[k] === 'string' && c[k].length, `${c.claim}: ${k}`);
        assert.ok([true, false, null].includes(c.agrees));
    }
    for (const prefix of SLOPES) {
        const c = byClaim(claims, prefix);
        assert.equal(c.agrees, true, `${prefix}: ${c.measured} vs ${c.predicted}`);
        assert.match(c.measured, /95% CI .* medians at \d+ sizes/);
    }
    assert.match(byClaim(claims, 'Radix sort').measured, /GPU-timestamp medians/);
    assert.match(byClaim(claims, 'Radix sort').predicted, /slope 1\.00 over N = 131,072 … 4,194,304/);
    // the workgroup variant at the traffic-bound slope also agrees (its prediction is an interval)
    const traffic = e.interpret(syntheticRows(NS, { ...GOOD, shared: (n) => 1e-6 * MODELS.sharedTraffic(n) }));
    assert.equal(byClaim(traffic, 'Bitonic sort (workgroup').agrees, true);

    // wrong exponents: radix ∝ N², a flat (overhead-bound) bitonic, a CPU sort ∝ N^1.5
    const wrong = e.interpret(syntheticRows(NS, { ...GOOD, radix: (n) => 1e-10 * n * n, bitonic: () => 5, cpu: (n) => 1e-6 * n ** 1.5 }));
    assert.equal(byClaim(wrong, 'Radix sort').agrees, false);
    assert.match(byClaim(wrong, 'Radix sort').note, /Steeper/);
    assert.equal(byClaim(wrong, 'Bitonic sort (global)').agrees, false);
    assert.match(byClaim(wrong, 'Bitonic sort (global)').note, /Flatter.*per-dispatch/);
    assert.equal(byClaim(wrong, 'CPU sort').agrees, false);

    // two sizes: a slope but no interval; one size or none: no slope
    for (const ns of [NS.slice(0, 2), NS.slice(0, 1), []]) {
        const few = e.interpret(syntheticRows(ns, GOOD));
        for (const prefix of SLOPES) assert.equal(byClaim(few, prefix).agrees, null, `${ns.length} sizes: ${prefix}`);
    }

    // an interval too wide to tell the models apart decides nothing (the f32 run in CI: ± 1.0)
    const noisy = e.interpret(syntheticRows(NS, GOOD, { jitter: 0.4 }));
    for (const prefix of SLOPES) {
        const c = byClaim(noisy, prefix);
        assert.equal(c.agrees, null, `${prefix}: ${c.measured}`);
        assert.match(c.note, /Too noisy/);
    }

    // without timestamps the wall medians are fitted
    assert.match(byClaim(e.interpret(syntheticRows(NS, GOOD, { gpu: false })), 'Radix sort').measured, /wall-clock medians/);
    assert.ok(SLOPE_TOLERANCE > 0 && SLOPE_TOLERANCE <= 0.1);
});

test('crossover(): bracketed between sweep points with spreads; none in range, below range, unresolved, reversed', () => {
    // radix with a 2 ms fixed cost vs CPU 4e-6·N log₂N: the CPU is faster at small N, radix at large N
    const rows = syntheticRows(NS, { ...GOOD, radix: (n) => 2 + 1.2e-6 * MODELS.radix(n) }, { jitter: 0 });
    const first = rows.findIndex(r => r.radixGpuHi < r.cpuMsLo);   // GPU timestamps decide when present
    assert.ok(first > 0 && rows.slice(first).every(r => r.radixGpuHi < r.cpuMsLo));
    const c = crossover(rows, 'radix');
    assert.equal(c.kind, 'between');
    assert.deepEqual([c.below.n, c.above.n], [rows[first - 1].n, rows[first].n]);
    const claim = byClaim(e.interpret(rows), 'Crossover: radix');
    assert.ok(claim.measured.startsWith(`between N = ${rows[first - 1].n.toLocaleString('en-US')} and ${rows[first].n.toLocaleString('en-US')}`), claim.measured);
    assert.equal(claim.agrees, null, 'no device-independent prediction');

    const slow = syntheticRows(NS, { ...GOOD, radix: (n) => 10 * GOOD.cpu(n) });
    assert.equal(crossover(slow, 'radix').kind, 'none');
    assert.match(byClaim(e.interpret(slow), 'Crossover: radix').measured, /^none in range/);
    assert.equal(crossover(syntheticRows(NS, { ...GOOD, radix: (n) => 0.1 * GOOD.cpu(n) }), 'radix').kind, 'below');
    assert.equal(crossover(syntheticRows(NS, { ...GOOD, radix: (n) => GOOD.cpu(n) / 0.8 }), 'radix').kind, 'unresolved');   // GPU time = 0.8 × wall
    assert.equal(crossover(syntheticRows(NS, { ...GOOD, radix: (n) => (n < 1e6 ? 0.1 : 10) * GOOD.cpu(n) }), 'radix').kind, 'reversed');
    assert.equal(crossover([], 'radix').kind, 'no-data');
});

test('ranking at the largest N: expected order agrees, inverted (software GPU) disagrees, overlapping spreads undecided', () => {
    const rank = (times, opts) => byClaim(e.interpret(syntheticRows(NS, times, opts)), 'Ranking');
    let c = rank(GOOD);
    assert.equal(c.agrees, true, c.measured);
    assert.match(c.claim, /N = 4,194,304, GPU timestamps/);
    assert.match(c.measured, /^radix .* < bitonic \(workgroup memory\) .* < bitonic \(global\)/);
    // CI's software GPU: global bitonic fastest, radix slowest
    c = rank({ bitonic: (n) => 1e-7 * MODELS.bitonic(n), shared: (n) => 1.3e-7 * MODELS.bitonic(n), radix: (n) => 1e-5 * MODELS.radix(n), cpu: GOOD.cpu });
    assert.equal(c.agrees, false);
    assert.match(c.note, /inverted/);
    c = rank({ ...GOOD, shared: (n) => GOOD.bitonic(n) * 1.01 }, { spread: 0.05, gpu: false });
    assert.equal(c.agrees, null, 'workgroup and global bitonic within each other\'s spread');
    assert.match(c.claim, /wall clock/);
    assert.equal(byClaim(e.interpret([]), 'Ranking').agrees, null);
});

// ---- timer resolution, CPU batches, fit range, timestamp consistency --------------------------------------

// A clock that advances `stepMs` of true time per call and reports it floored to `tickMs` (Safari: 1 ms)
const coarseClock = (tickMs, stepMs = 0.003) => { let t = 0; return () => { t += stepMs; return Math.floor(t / tickMs) * tickMs; }; };

test('timerResolution(): the smallest step of a coarsened clock; Node\'s clock is fine-grained; a frozen clock gives null', (t) => {
    for (const tick of [1, 0.1, 0.005]) {
        const r = timerResolution(coarseClock(tick));
        assert.ok(Math.abs(r - tick) < 1e-9, `${tick} ms clock: measured ${r}`);
    }
    const node = timerResolution();
    t.diagnostic(`performance.now() resolution in Node: ${node} ms`);
    assert.ok(node > 0 && node < 0.05);
    assert.equal(timerResolution(() => 7, { maxCalls: 1000 }), null);
});

test('granularity(): the step of quantized medians, null with too few distinct values', () => {
    // Chrome's GPU timestamps: multiples of 2¹⁶ ns, with float noise
    const ticks = [3, 4, 5, 9, 15, 36, 65].map(k => k * 0.065536 * (1 + 1e-9));
    assert.ok(Math.abs(granularity(ticks) - 0.065536) < 1e-5);
    assert.equal(granularity([0, 1, 1, 2, 3, 6, 6.0000001]), 1, 'Safari wall medians: 1 ms');
    assert.equal(granularity([1, 2, 3]), null);
    const r = resolutions([{ n: 1, timerMs: 0.1, radixWall: 0.5 }, { n: 2, timerMs: 0.1, radixWall: 0.55 }]);
    assert.equal(r.wall, 0.1, 'a measured resolution wins over the inferred one');
    assert.equal(r.wallSource, 'measured');
    assert.ok(resolvable(1, 0.1) && !resolvable(0.99, 0.1) && resolvable(0.2, 0.1, 5) && resolvable(0.001, null));
    assert.equal(RESOLUTION_FACTOR, 10);
});

test('CPU sort batch: copies sorted back to back, per-sort time, calibrated to the target, capped', () => {
    const keys = generateKeys(4096, 'u32', 5);
    const b = cpuSortBatch(keys, 3);
    assert.equal(b.reps, 3);
    assert.deepEqual(b.sorted, keys.slice().sort());
    assert.ok(Math.abs(b.cpuMs * 3 - b.totalMs) < 1e-9);
    assert.notDeepEqual(keys, b.sorted, 'the input keys are not sorted in place');
    // A 1 ms clock and 0.25 ms per sort: reps double until a batch spans the 20 ms target (Safari: 20 × 1 ms)
    let t = 0;
    const now = () => Math.floor(t);
    const fake = (length, ms) => ({ length, slice: () => ({ sort() { t += ms; } }) });
    assert.equal(cpuBatchTarget(1), 20);
    assert.equal(cpuBatchTarget(0.0001), 10);
    const cal = calibrateCpuBatch(fake(16384, 0.25), cpuBatchTarget(1), now);
    assert.equal(cal.reps, CPU_MAX_REPS, '64 × 0.25 ms = 16 ms: the cap stops it');
    assert.ok(resolvable(cal.cpuMs, 1, cal.reps), `${cal.reps} × ${cal.cpuMs} ms resolvable on a 1 ms clock`);
    t = 0;
    assert.equal(calibrateCpuBatch(fake(1 << 20, 0.25), 20, now).reps, cpuRepsCap(1 << 20));
    assert.equal(cpuRepsCap(1 << 20), CPU_MAX_BATCH_KEYS >> 20, 'at most 16 MB of keys per batch');
    assert.equal(cpuRepsCap(MAX_N), 1);
    assert.ok(COUNTS.every(n => cpuRepsCap(n) * n <= Math.max(n, CPU_MAX_BATCH_KEYS)));
    t = 0;
    assert.equal(calibrateCpuBatch(fake(MAX_N, 300), 20, now).reps, 1, 'a sort longer than the target is timed once');
    t = 0;
    assert.equal(calibrateCpuBatch(fake(65536, 3), 20, now).reps, 8, '3, 6, 12 ms are short of 20 ms; 24 ms is not');
});

test('fitRange(): every size up to the largest N with a median ≥ 1 ms and resolvable; < 3 sizes undetermined', () => {
    const rows = syntheticRows(NS, GOOD);
    const { ns } = fitRange(rows, 'radixGpu');
    assert.deepEqual(ns, rows.filter(r => r.radixGpu >= FIT_MIN_MS).map(r => r.n));
    assert.ok(ns.length >= MIN_FIT_SIZES && ns.length < NS.length, 'the smallest sizes are left out');
    assert.equal(ns.at(-1), MAX_N, 'up to the largest N of the run');
    // a 1 ms clock: only medians ≥ 10 ms count
    assert.deepEqual(fitRange(rows, 'radixGpu', 1).ns, rows.filter(r => r.radixGpu >= 10).map(r => r.n));
    // Swept to 1,048,576 only (the first Apple silicon runs), radix reaches 1 ms at too few sizes: undetermined
    const short = syntheticRows(NS.filter(n => n <= 1048576), { ...GOOD, radix: (n) => 2.5e-6 * n });
    const c = byClaim(e.interpret(short), 'Radix sort');
    assert.ok(fitRange(short, 'radixGpu').ns.length < MIN_FIT_SIZES);
    assert.equal(c.agrees, null);
    assert.match(c.note, /Undetermined: needs at least 3 sizes/);
    // The predicted slope is computed over the fitted N, not a fixed range
    const b = byClaim(e.interpret(rows), 'Bitonic sort (global)');
    const fitted = fitRange(rows, 'bitonicGpu').ns;
    assert.ok(b.predicted.includes(`slope ${predictedSlope(MODELS.bitonic, fitted).toFixed(2)} over N = ${fitted[0].toLocaleString('en-US')} … 4,194,304`), b.predicted);
});

test('clockFor(): GPU timestamps longer than the wall time of the same sorts are not used', () => {
    const rows = syntheticRows(NS, GOOD);
    assert.equal(clockFor(rows, 'radix').key, 'radixGpu');
    const bad = rows.map(r => (r.n === 1048576 ? { ...r, radixGpu: 3 * r.radixWall, radixGpuLo: 2.5 * r.radixWall, radixGpuHi: 3.5 * r.radixWall } : r));
    const clock = clockFor(bad, 'radix');
    assert.deepEqual([clock.key, clock.gpu, clock.inconsistent], ['radixWall', false, [1048576]]);
    const claims = e.interpret(bad);
    assert.match(byClaim(claims, 'Crossover: radix (wall)').note, /GPU timestamps not used: at N = 1,048,576/);
    assert.match(byClaim(claims, 'Ranking').claim, /wall clock/, 'the ranking needs consistent timestamps for every algorithm');
    assert.ok(byClaim(claims, 'Crossover: bitonic (global) (GPU timestamps)'));
});

test('crossover(): sizes below timer resolution are not compared; all below → undetermined', () => {
    // Safari-like: wall and CPU times quantized to 1 ms, no timestamps; CPU up to 17 ms, radix up to 3 ms
    const rows = NS.filter(n => n <= 1048576).map((n, i) => {
        const row = { n, repeats: 5, cpuReps: 1 };
        for (const [k, v] of [['cpuMs', 17 * n / 1048576], ['radixWall', 3 * n / 1048576], ['bitonicWall', 6 * n / 1048576 + i % 2], ['sharedWall', 6 * n / 1048576]]) {
            row[k] = Math.round(v); row[`${k}Lo`] = Math.round(v); row[`${k}Hi`] = Math.round(v) + (i % 2);
        }
        return row;
    });
    assert.equal(resolutions(rows).wall, 1);
    assert.equal(crossover(rows, 'radix').kind, 'below-resolution');
    assert.match(byClaim(e.interpret(rows), 'Crossover: radix (wall)').measured, /^undetermined: below timer resolution/);
    // The CPU sort timed over 64-sort batches: CPU times resolvable, radix still not
    assert.equal(crossover(rows.map(r => ({ ...r, cpuReps: 64 })), 'radix').kind, 'below-resolution');
    // Resolvable GPU times from 524,288 on: faster there, undetermined below
    const x = (r, k, f) => ({ [k]: r[k] * f, [`${k}Lo`]: r[`${k}Lo`] * f, [`${k}Hi`]: r[`${k}Hi`] * f });
    const later = rows.map(r => ({ ...r, cpuReps: 64, ...x(r, 'radixWall', 10), ...x(r, 'cpuMs', 100) }));
    const l = crossover(later, 'radix');
    assert.equal(l.kind, 'below');
    assert.ok(l.undeterminedBelow && l.skipped.length > 0 && l.skipped.every(n => n < l.at.n));
    assert.match(byClaim(e.interpret(later), 'Crossover: radix').measured, /whether the crossover lies lower is undetermined; not compared/);
});
