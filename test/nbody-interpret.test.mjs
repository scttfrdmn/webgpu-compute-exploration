// N-body Measure interpretation (measured vs predicted, #75), CPU only: interpret() of the three experiments in
// js/teach/experiments/nbody.js on synthetic rows with known exponents (agrees), wrong exponents (disagrees)
// and too few points (undetermined); the crossover bracket; the brute-force budget guard; scope shape; and
// the interpretation entries of a schema-valid result record. The same claims on real rows from a software GPU
// are checked in test/gpu/nbody-measure.gpu.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import experiments, {
    THETAS, REPLICA_MEDIANS, REPLICA_FACTOR, DTS, PROTOCOL, FLOOR_FACTOR, FIT_MIN_MS, BH_DISPATCHES,
    crossover, skipBrute, rangeVerdict, slopeText, fitRows, floorFit, nlognSlope, scalingCounts, scalingLimits, NLOGN_EFFECTIVE,
} from '../js/teach/experiments/nbody.js';
import { fitPowerLaw } from '../js/teach/stats.js';
import { buildResultRecord, validateResultRecord, SCHEMA_PATH } from '../js/teach/result-record.js';

const byId = Object.fromEntries(experiments.map(e => [e.id, e]));
const wiggle = (i) => 1 + 0.03 * (i % 2 ? 1 : -1);          // ±3% scatter, so the fits have an interval
const verdicts = (claims) => claims.map(c => c.agrees);

// θ-error rows: median ∝ θ^p, interactions ∝ θ^q
const thetaRows = (p, q, thetas = THETAS) => thetas.map((theta, i) => ({
    theta, median: 4e-3 * (theta / 0.5) ** p * wiggle(i), p99: 3e-2, interactions: 487 * (theta / 0.5) ** q * wiggle(i + 1),
}));

test('theta-error interpret: θ² and θ^-1.5 agree; θ¹, θ³ and θ⁻⁴ disagree; one or two θ are undetermined', () => {
    const e = byId['theta-error'];
    const settings = { n: 1234, preset: 'plummer' };          // no replica values: the third claim is null
    let claims = e.interpret(thetaRows(2, -1.5), settings);
    assert.deepEqual(claims.map(c => c.claim.split(' ')[0]), ['Median', 'Interactions', 'Your']);
    assert.deepEqual(verdicts(claims), [true, true, null]);
    assert.match(claims[0].measured, /^2\.\d\d \(95% CI .* 6 points\)$/);
    assert.match(claims[0].predicted, /θ²/);
    assert.deepEqual(verdicts(e.interpret(thetaRows(2.2, -2.8), settings)), [true, true, null], 'inside the model tolerance');
    assert.deepEqual(verdicts(e.interpret(thetaRows(1, -4), settings)).slice(0, 2), [false, false]);
    assert.match(e.interpret(thetaRows(1, -1), settings)[0].note, /Flatter than θ²/);
    assert.match(e.interpret(thetaRows(3, -1), settings)[0].note, /Steeper than θ²/);
    assert.equal(e.interpret(thetaRows(2, 0.5), settings)[1].agrees, false, 'interactions that grow with θ');
    for (const k of [1, 2]) {
        claims = e.interpret(thetaRows(2, -1.5, THETAS.slice(0, k)), settings);
        assert.deepEqual(verdicts(claims), [null, null, null], `${k} θ value(s)`);
        assert.match(claims[0].note, /needs at least 3 θ values/);
    }
    assert.deepEqual(e.interpret([], settings).map(c => c.agrees), [null, null, null]);
});

test('theta-error interpret: the visitor\'s medians against the CPU replica', () => {
    const e = byId['theta-error'];
    const settings = { n: 4096, preset: 'plummer' };
    const replica = THETAS.map((theta, i) => ({ theta, median: REPLICA_MEDIANS['plummer/4096'][i], interactions: 1000 / (1 + i) }));
    let claim = e.interpret(replica, settings)[2];
    assert.equal(claim.agrees, true);
    assert.equal(claim.measured, 'ratio 1.00 … 1.00 over 6 θ values');
    // the replica's own slope agrees with θ²
    assert.equal(e.interpret(replica, settings)[0].agrees, true);
    claim = e.interpret(replica.map((r, i) => (i === 3 ? { ...r, median: r.median * (REPLICA_FACTOR + 0.1) } : r)), settings)[2];
    assert.equal(claim.agrees, false);
    assert.match(claim.note, /does not reproduce the reference/);
    claim = e.interpret(replica.map(r => ({ ...r, median: r.median * 1.5 })), settings)[2];
    assert.equal(claim.agrees, true, `× ${REPLICA_FACTOR} is the tolerance`);
});

// scaling rows: brute = bruteFloor + brute0 · s^p (to the brute-force limit of maxN), Barnes-Hut = bhFloor + bh0 · s^q,
// interactions ∝ s^k, s = N / 1,024; spreads ±5%; GPU times optional
function scalingRows({ p = 2, q = 1.1, k = 0.37, bh0 = 16, brute0 = 2, bhFloor = 0, bruteFloor = 0, gpu = false, share = null,
    maxN } = {}) {
    const limits = scalingLimits(maxN);
    return scalingCounts(maxN).map((n, i) => {
        const s = n / 1024;
        const bh = (bhFloor + bh0 * s ** q) * wiggle(i);
        const brute = n <= limits.brute ? (bruteFloor + brute0 * s ** p) * wiggle(i + 1) : null;
        const spread = (key, v) => ({ [key]: v, [`${key}Lo`]: v === null ? null : 0.95 * v, [`${key}Hi`]: v === null ? null : 1.05 * v });
        return { n, ...spread('bhWall', bh), ...spread('bruteWall', brute),
            ...spread('bhGpu', gpu ? bh / 2 : null), ...spread('bruteGpu', gpu && brute !== null ? brute / 2 : null),
            treeShare: share ? share(s) : null, interactions: 248 * s ** k * wiggle(i) };
    });
}
const claimOf = (claims, word) => claims.find(c => c.claim.includes(word));

test('scaling interpret: N² and N^1.1 agree; wrong exponents disagree; too few N are undetermined', () => {
    const e = byId.scaling;
    let claims = e.interpret(scalingRows());
    assert.equal(claims.length, 6);
    assert.deepEqual(claims.map(c => c.key), ['brute-n2', 'bh-nlogn', 'bh-interactions', 'bh-crossover', 'tree-share', 'fixed-floor']);
    assert.deepEqual(verdicts(claims), [true, true, true, true, null, null], JSON.stringify(claims, null, 1));
    assert.match(claims[0].claim, /wall ms/);
    assert.match(claimOf(claims, 'Tree-build').note, /timestamp-query/);
    assert.ok(Math.abs(NLOGN_EFFECTIVE - 1.113) < 0.005);

    // brute force at N^1.4 (overhead-bound) and Barnes-Hut at N^1.6
    claims = e.interpret(scalingRows({ p: 1.4, q: 1.6, brute0: 40 }));
    assert.equal(claims[0].agrees, false);
    assert.match(claims[0].note, /Below 2 even above the floor/);
    assert.equal(claims[1].agrees, false);
    assert.match(claims[1].note, /Above 1\.3/);
    assert.equal(e.interpret(scalingRows({ q: 0.6, maxN: 262144 }), { maxN: 262144 })[1].agrees, false, 'flat Barnes-Hut');
    assert.equal(claimOf(e.interpret(scalingRows({ k: 1.2 })), 'interactions').agrees, false, 'super-linear interactions');

    // two N: slopes without an interval
    claims = e.interpret(scalingRows().slice(0, 2));
    assert.deepEqual(verdicts(claims).slice(0, 3), [null, null, null]);
    assert.match(claims[0].note, /At least 3 sizes above the threshold are needed/);
    assert.deepEqual(verdicts(e.interpret([])), [null, null, null, null, null, null]);

    // GPU times when timestamp-query is available, and a falling tree-build share
    claims = e.interpret(scalingRows({ gpu: true, share: (s) => 0.6 * s ** -0.3 }));
    assert.match(claims[0].claim, /GPU ms/);
    assert.match(claims[1].claim, /GPU ms/);
    const tree = claimOf(claims, 'Tree-build');
    assert.equal(tree.agrees, true);
    assert.match(tree.measured, /^60% at N = 1,024 → \d+% at N = 65,536/);
    assert.equal(claimOf(e.interpret(scalingRows({ gpu: true, share: (s) => 0.2 * s ** 0.2 })), 'Tree-build').agrees, false);
});

test('scaling interpret: the crossover bracket, below the sweep, above it, and a skipped brute-force point', () => {
    const e = byId.scaling;
    // CI-like: brute faster to 8,192, Barnes-Hut from 16,384
    let x = crossover(scalingRows());
    assert.deepEqual([x.state, x.lo, x.hi, x.clear, x.monotone, x.basis], ['bracketed', 8192, 16384, true, true, 'wall ms']);
    assert.ok(x.estimate > 8192 && x.estimate < 16384);
    let claim = claimOf(e.interpret(scalingRows()), 'Crossover');
    assert.equal(claim.agrees, true);
    assert.match(claim.measured, /^between N = 8,192 and N = 16,384 \(log-interpolated ≈ [\d,]+\)$/);

    // Barnes-Hut faster everywhere
    x = crossover(scalingRows({ brute0: 40 }));
    assert.deepEqual([x.state, x.firstN], ['below', 1024]);
    assert.equal(claimOf(e.interpret(scalingRows({ brute0: 40 })), 'Crossover').agrees, true);

    // brute force faster everywhere it ran: an extrapolated estimate, verdict undetermined
    claim = claimOf(e.interpret(scalingRows({ brute0: 0.01 })), 'Crossover');
    assert.equal(claim.agrees, null);
    assert.match(claim.measured, /^above N = 16,384: .*extrapolating the two fits gives N ≈ [\d,]+$/);
    // ... and slopes that never cross contradict the prediction
    assert.equal(claimOf(e.interpret(scalingRows({ brute0: 0.01, p: 1.0, q: 1.1 })), 'Crossover').agrees, false);

    // overlapping spreads at a bracket point are reported
    const rows = scalingRows();
    const r = rows.find(row => row.n === 16384);
    Object.assign(r, { bhWall: 0.99 * r.bruteWall, bhWallLo: 0.9 * r.bruteWall, bhWallHi: 1.1 * r.bruteWall });
    assert.equal(crossover(rows).clear, false);
    assert.match(claimOf(e.interpret(rows), 'Crossover').note, /overlap/);

    // brute force skipped at 16,384 by the budget guard: said in the brute-force note
    const skipped = scalingRows().map(row => (row.n === 16384 ? { ...row, bruteWall: null, bruteWallLo: null, bruteWallHi: null } : row));
    assert.match(e.interpret(skipped)[0].note, /skipped at N = 16,384/);
});

test('skipBrute: skips only when the projected brute-force and remaining Barnes-Hut points exceed the budget', () => {
    const calls = PROTOCOL.warmup + PROTOCOL.repeats;
    // 100 ms per brute evaluation at 8,192 → 400 ms at 16,384; Barnes-Hut 50 ms at 16,384 → 32,768 and 65,536 left
    const perCall = { bh: 50, brute: 100 };
    const bh = calls * 50 * (2 ** 1.2 + 4 ** 1.2);
    assert.equal(skipBrute(16384, 8192, perCall, calls * 400 + bh + 1), false);
    assert.equal(skipBrute(16384, 8192, perCall, calls * 400 + bh - 1), true);
    assert.equal(skipBrute(16384, 8192, perCall, Infinity), false, 'CI test context: no budget');
    assert.equal(skipBrute(2048, 1024, { bh: 1, brute: null }, 0), false, 'no brute-force time yet');
    // Largest N 262,144: the Barnes-Hut points at 131,072 and 262,144 count too
    const bhBig = bh + calls * 50 * (8 ** 1.2 + 16 ** 1.2);
    assert.equal(skipBrute(16384, 8192, perCall, calls * 400 + bhBig - 1, 262144), true);
    assert.equal(skipBrute(16384, 8192, perCall, calls * 400 + bhBig + 1, 262144), false);
});

// ---- Fixed floor (#85) ------------------------------------------------------------------------------------

test('floorFit: floor at the smallest N, threshold max(4 × floor, 1 ms), floor subtracted, plain fit for contrast', () => {
    const rows = [1024, 2048, 4096, 8192, 16384].map((n, i) => ({ n, t: [0.5, 0.9, 2.1, 6, 20][i] }));
    const f = floorFit(rows, 't');
    assert.deepEqual([f.floor, f.floorN, f.threshold, f.used], [0.5, 1024, FLOOR_FACTOR * 0.5, [4096, 8192, 16384]]);
    const expected = fitPowerLaw([4096, 8192, 16384], [1.6, 5.5, 19.5]);
    assert.ok(Math.abs(f.fit.slope - expected.slope) < 1e-12);
    assert.equal(f.allFit.n, 5);
    // a tiny (or zero, from a coarse wall clock) floor: the 1 ms minimum decides
    assert.equal(floorFit(rows.map(r => ({ ...r, t: r.t / 10 })), 't').threshold, FIT_MIN_MS);
    assert.deepEqual(floorFit([{ n: 1024, t: 0 }, { n: 2048, t: 1 }, { n: 4096, t: 3 }], 't').used, [2048, 4096]);
    assert.equal(floorFit([], 't').floor, null);
    // unsorted rows and missing values
    assert.equal(floorFit([{ n: 4096, t: 9 }, { n: 1024, t: 1 }, { n: 2048, t: null }], 't').floor, 1);
});

test('scaling interpret: a fixed floor flattens the plain fit, the floor-corrected fit recovers N² and N log N', (t) => {
    const e = byId.scaling;
    // A fast GPU (GPU ms = half of these): Barnes-Hut 0.5 ms floor + 0.05 ms · s^1.15, brute force 0.05 ms floor +
    // 0.025 ms · s², Largest N 262,144
    const rows = scalingRows({ maxN: 262144, gpu: true, bhFloor: 1, bh0: 0.1, q: 1.15, bruteFloor: 0.1, brute0: 0.05, p: 2 });
    const claims = e.interpret(rows, { maxN: 262144 });
    const [brute, bh] = claims;
    const plainBh = floorFit(rows, 'bhGpu').allFit, plainBrute = floorFit(rows, 'bruteGpu').allFit;
    t.diagnostic(`plain slopes: brute ${plainBrute.slope.toFixed(2)}, Barnes-Hut ${plainBh.slope.toFixed(2)}; `
        + `corrected: ${brute.measured} | ${bh.measured}`);
    assert.ok(plainBh.slope < 1, 'over every size the floor flattens Barnes-Hut below 1');
    assert.equal(slopeAgreesBand(plainBh), false, 'the old all-sizes test would disagree');
    assert.deepEqual([brute.agrees, bh.agrees], [true, true]);
    assert.match(bh.measured, /over N = [\d,]+ … 262,144, floor subtracted/);
    assert.match(bh.note, /at least 4× the floor/);
    assert.match(bh.note, new RegExp(`Over all 9 sizes, without the correction, the slope is ${plainBh.slope.toFixed(2)}`));
    // the predicted N log N slope is computed over the fitted sizes, not the whole sweep
    const used = floorFit(rows, 'bhGpu').used;
    assert.ok(used[0] > 1024);
    assert.ok(bh.predicted.startsWith(`about N log N: ${nlognSlope(used).toFixed(2)} over N = ${used[0].toLocaleString('en-US')} … 262,144 (the fitted sizes)`));
    // the informational floor claim
    const floor = claims.find(c => c.key === 'fixed-floor');
    assert.equal(floor.agrees, null);
    const bhFloor = rows[0].bhGpu;
    assert.match(floor.measured, new RegExp(`^Barnes-Hut ${bhFloor.toFixed(2)} ms at N = 1,024 \\(GPU ms; ≤ ${(1000 * bhFloor / BH_DISPATCHES).toFixed(0)} µs per dispatch over 31\\); brute force`));
    assert.match(floor.note, /floor is \d+% of the time/);
});

// slopeAgrees with the Barnes-Hut band over the default sweep (the pre-#85 test: 1.15 ± 0.15)
const slopeAgreesBand = (fit) => fit.ci95[0] - 0.15 <= 1.15 && 1.15 <= fit.ci95[1] + 0.15;

test('scaling interpret: with no floor at all, subtracting the smallest-N time steepens the fit only a little', (t) => {
    const e = byId.scaling;
    for (const maxN of [65536, 262144]) {
        const claims = e.interpret(scalingRows({ maxN }), { maxN });
        const slopes = claims.slice(0, 2).map(c => Number(c.measured.split(' ')[0]));
        t.diagnostic(`pure N² and N^1.1 (wall ms), Largest N ${maxN}: corrected slopes ${slopes.join(', ')}`);
        assert.ok(slopes[0] >= 2 && slopes[0] <= 2.1, `brute ${slopes[0]}`);
        assert.ok(slopes[1] >= 1.1 && slopes[1] <= 1.2, `Barnes-Hut ${slopes[1]}`);
        assert.deepEqual(verdicts(claims).slice(0, 2), [true, true]);
    }
    // a noisy fit (±40% scatter on 3 points) is too wide to decide
    const noisy = scalingRows({ maxN: 65536 }).map((r, i) => ({ ...r, bruteWall: r.bruteWall === null ? null : r.bruteWall * (i % 2 ? 1.4 : 0.6) }));
    const claim = e.interpret(noisy)[0];
    assert.equal(claim.agrees, null);
    assert.match(claim.note, /Too noisy to decide/);
});

// The first physical-GPU bundles (#85): Apple M4 Pro, Chrome 156 and Safari 27.2, default Largest N (65,536).
// Over every size the Barnes-Hut slope was 0.72 / 0.66 and Safari's brute-force slope 1.16 (disagree); with the
// floor correction too few sizes are far enough above the floor, so the verdicts are undetermined, not wrong.
const BUNDLES = {
    chrome: 'data/validation/apple-silicon-chrome/2026-11-02-6fd3b07.json',
    safari: 'data/validation/apple-silicon-safari26/2026-11-02-6fd3b07.json',
};
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const scalingRecord = (path) => JSON.parse(readFileSync(join(root, path), 'utf8')).records
    .find(r => r.experiment.exampleId === 'nbody' && r.experiment.id === 'scaling');

test('scaling interpret on the Apple M4 Pro bundles: the old disagreements become undetermined, floors reported', (t) => {
    const e = byId.scaling;
    for (const [name, path] of Object.entries(BUNDLES)) {
        const record = scalingRecord(path);
        const old = Object.fromEntries(record.interpretation.map(c => [c.claim.split(' (')[0], c]));
        const claims = e.interpret(record.rows, record.settings);
        for (const c of claims) t.diagnostic(`${name} ${c.key}: ${c.agrees} | ${c.measured}${c.note ? ` | ${c.note}` : ''}`);
        const byKey = Object.fromEntries(claims.map(c => [c.key, c]));
        assert.equal(old['Barnes-Hut time vs N'].agrees, false, `${name}: the bundle's own verdict`);
        assert.equal(byKey['bh-nlogn'].agrees, null, `${name}: Barnes-Hut undetermined`);
        assert.equal(byKey['brute-n2'].agrees, null, `${name}: brute force undetermined`);
        assert.match(byKey['bh-nlogn'].note, /Over all 7 sizes, without the correction, the slope is 0\.(72|66)/);
        assert.match(byKey['brute-n2'].note, /choose a larger Largest N/);
        // unchanged claims keep their verdicts
        assert.equal(byKey['bh-crossover'].agrees, true);
        assert.equal(byKey['tree-share'].agrees, true);
        assert.equal(byKey['bh-interactions'].agrees, true);
        const floor = byKey['fixed-floor'];
        assert.equal(floor.agrees, null);
        assert.match(floor.measured, /^Barnes-Hut 0\.(59|60) ms at N = 1,024 \(GPU ms; ≤ 19 µs per dispatch over 31\); brute force (66|85) µs at N = 1,024/);
    }
});

const driftRows = (p, base = 1.3e-6) => DTS.map((dt, i) => ({ dt, steps: Math.round(2 / dt), maxDrift: base * (dt / 0.01) ** p * wiggle(i) }));

test('energy-drift interpret: dt² agrees for brute force, dt¹ does not; a flat floor agrees for Barnes-Hut', () => {
    const e = byId['energy-drift'];
    let claims = e.interpret(driftRows(2), { mode: 'brute' });
    assert.equal(claims.length, 1);
    assert.equal(claims[0].agrees, true);
    assert.equal(e.interpret(driftRows(1), { mode: 'brute' })[0].agrees, false);
    assert.match(e.interpret(driftRows(1), { mode: 'brute' })[0].note, /Below 2/);
    assert.match(e.interpret(driftRows(3), { mode: 'brute' })[0].note, /Above 2/);
    assert.equal(e.interpret(driftRows(2).slice(0, 2), { mode: 'brute' })[0].agrees, null);

    claims = e.interpret(driftRows(0.04, 6.5e-4), { mode: 'bh' });
    assert.equal(claims.length, 2);
    assert.deepEqual(verdicts(claims), [true, true]);
    assert.match(claims[1].measured, /^6\.\d\de-4 \(median over 4 dt/);
    claims = e.interpret(driftRows(1, 6.5e-4), { mode: 'bh' });
    assert.equal(claims[0].agrees, false, 'a dt-dependent Barnes-Hut error');
    assert.match(claims[0].note, /still falls with dt/);
    assert.equal(e.interpret(driftRows(0, 3e-6), { mode: 'bh' })[1].agrees, false, 'a floor far below the tree error');
    assert.equal(e.interpret(driftRows(0, 5e-2), { mode: 'bh' })[1].agrees, false, 'a floor far above it');
    assert.deepEqual(verdicts(e.interpret(driftRows(0, 6.5e-4).slice(0, 2), { mode: 'bh' })), [null, true]);
    assert.deepEqual(verdicts(e.interpret([], { mode: 'bh' })), [null, null]);
});

test('helpers: slopeText and rangeVerdict', () => {
    assert.equal(slopeText(fitRows([{ x: 1, y: 1 }], 'x', 'y')), 'no fit: 1 usable point');
    assert.equal(slopeText(fitRows([{ x: 1, y: 1 }, { x: 2, y: 4 }], 'x', 'y')), '2.00 (2 points: no uncertainty)');
    const fit = (lo, hi) => ({ ci95: [lo, hi] });
    assert.equal(rangeVerdict(fit(0.3, 0.4), 0, 1), true);
    assert.equal(rangeVerdict(fit(-0.2, 0.4), 0, 1), null);
    assert.equal(rangeVerdict(fit(1.1, 1.4), 0, 1), false);
    assert.equal(rangeVerdict({ ci95: null }, 0, 1), null);
});

test('every N-body experiment has interpret() and a scope with what it shows and does not show', () => {
    for (const e of experiments) {
        assert.equal(typeof e.interpret, 'function', e.id);
        assert.deepEqual(Object.keys(e.scope).sort(), ['doesNotShow', 'shows'], e.id);
        for (const list of Object.values(e.scope)) {
            assert.ok(list.length >= 2 && list.every(s => typeof s === 'string' && s.length > 20 && s.endsWith('.')), e.id);
        }
    }
});

const schema = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', SCHEMA_PATH), 'utf8'));

test('result records carry the interpretation and stay schema-valid', () => {
    const cases = [
        ['theta-error', { n: 4096, preset: 'plummer' }, thetaRows(2, -1.5), 3],
        ['scaling', { preset: 'plummer', maxN: 65536 }, scalingRows(), 6],
        ['scaling', { preset: 'plummer', maxN: 262144 }, scalingRows({ maxN: 262144 }), 6],
        ['energy-drift', { n: 2048, mode: 'brute', preset: 'plummer' }, driftRows(2), 1],
        ['energy-drift', { n: 2048, mode: 'bh', preset: 'plummer' }, driftRows(0, 6.5e-4), 2],
    ];
    for (const [id, settings, rows, claims] of cases) {
        const experiment = byId[id];
        const record = buildResultRecord({ exampleId: 'nbody', experiment, settings, rows, measured: null,
            gpuDevice: null, source: { version: 'test', commit: 'local' }, budgetMs: experiment.budgetMs });
        assert.deepEqual(validateResultRecord(record, schema).errors, [], id);
        assert.equal(record.interpretation.length, claims, id);
        for (const c of record.interpretation) {
            assert.ok(c.claim && c.predicted && c.measured, `${id}: ${JSON.stringify(c)}`);
            assert.ok([true, false, null].includes(c.agrees));
        }
    }
});
