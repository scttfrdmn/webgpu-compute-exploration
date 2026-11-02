// Statistics of the measurement protocol (js/teach/stats.js) and wall-clock phase timing (js/phase-clock.js).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { summarize, quantile, quantileSorted, finiteValues, SPREAD } from '../js/teach/stats.js';
import { PhaseClock, PHASES } from '../js/phase-clock.js';

const close = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test('quantiles: type 7 (linear between order statistics), as R and NumPy', () => {
    const x = [1, 2, 3, 4, 5];
    assert.deepEqual([0, 0.25, 0.5, 0.75, 1].map(p => quantile(x, p)), [1, 2, 3, 4, 5]);
    // numpy.percentile([1, 2, 3, 4], [25, 50, 75]) = 1.75, 2.5, 3.25
    assert.deepEqual([0.25, 0.5, 0.75].map(p => quantile([4, 1, 3, 2], p)), [1.75, 2.5, 3.25]);
    // R: quantile(c(10, 20, 40, 80, 160, 320), c(.1, .9)) = 15, 240
    close(quantile([10, 20, 40, 80, 160, 320], 0.1), 15);
    close(quantile([10, 20, 40, 80, 160, 320], 0.9), 240);
    assert.equal(quantileSorted([], 0.5), null);
    assert.throws(() => quantileSorted([1], 1.5), RangeError);
});

test('summarize: median, range, quartiles, mean, sample std, cv', () => {
    const s = summarize([2, 4, 4, 4, 5, 5, 7, 9]);
    assert.equal(s.n, 8);
    assert.equal(s.median, 4.5);
    assert.equal(s.min, 2);
    assert.equal(s.max, 9);
    assert.equal(s.q1, 4);
    assert.equal(s.q3, 5.5);
    assert.equal(s.mean, 5);
    close(s.std, Math.sqrt(32 / 7));
    close(s.cv, Math.sqrt(32 / 7) / 5);
    // order does not matter; the input is not modified
    const input = [9, 2, 5, 4, 7, 4, 5, 4];
    assert.deepEqual(summarize(input), s);
    assert.deepEqual(input, [9, 2, 5, 4, 7, 4, 5, 4]);
    // negative mean: cv uses |mean|
    assert.ok(summarize([-1, -2, -3]).cv > 0);
    assert.equal(SPREAD, 'iqr');
});

test('summarize: edge cases n = 0, 1, 2; NaN and non-numbers dropped; zero mean', () => {
    const empty = summarize([]);
    assert.deepEqual(empty, { n: 0, median: null, min: null, max: null, q1: null, q3: null, mean: null, std: null, cv: null });
    assert.deepEqual(summarize(undefined), empty);
    assert.deepEqual(summarize([null, undefined, NaN, Infinity, -Infinity, '3', {}]), empty);

    const one = summarize([7]);
    assert.deepEqual(one, { n: 1, median: 7, min: 7, max: 7, q1: 7, q3: 7, mean: 7, std: null, cv: null });

    const two = summarize([1, 3]);
    assert.deepEqual([two.n, two.median, two.q1, two.q3, two.min, two.max], [2, 2, 1.5, 2.5, 1, 3]);
    close(two.std, Math.SQRT2);
    close(two.cv, Math.SQRT2 / 2);

    const mixed = summarize([1, NaN, 2, null, 3, undefined, Infinity]);
    assert.equal(mixed.n, 3);
    assert.equal(mixed.median, 2);
    assert.deepEqual(finiteValues([1, NaN, '2', 3]), [1, 3]);

    const constant = summarize([4, 4, 4]);
    assert.equal(constant.std, 0);
    assert.equal(constant.cv, 0);
    assert.equal(summarize([-1, 1]).cv, null, 'cv undefined for a zero mean');
});

test('PhaseClock: sync and async phases, sums repeated phases, records time on throw/reject', async () => {
    let t = 0;
    const clock = new PhaseClock({ now: () => t });
    assert.deepEqual(PHASES, ['compile', 'upload', 'dispatch', 'readback']);
    const sim = clock.time('compile', () => { t += 5; return 'sim'; });
    assert.equal(sim, 'sim');
    clock.time('upload', () => { t += 1; });
    const value = await clock.time('dispatch', async () => { t += 10; await null; t += 2; return 42; });
    assert.equal(value, 42);
    await clock.time('readback', async () => { t += 3; });
    await clock.time('readback', async () => { t += 4; });
    assert.deepEqual(clock.phases, { compile: 5, upload: 1, dispatch: 12, readback: 7 });

    assert.throws(() => clock.time('upload', () => { t += 2; throw new Error('bad'); }), /bad/);
    await assert.rejects(clock.time('dispatch', async () => { t += 1; throw new Error('lost'); }), /lost/);
    assert.equal(clock.phases.upload, 3);
    assert.equal(clock.phases.dispatch, 13);
    clock.add('compile', 0.5);
    assert.equal(clock.phases.compile, 5.5);
    // phases is a copy
    clock.phases.compile = 99;
    assert.equal(clock.phases.compile, 5.5);
});

test('fitPowerLaw: recovers exponents with a 95% interval; slopeAgrees', async () => {
    const { fitPowerLaw, slopeAgrees, t975 } = await import('../js/teach/stats.js');
    const xs = [64, 128, 256, 512, 1024];
    const exact = fitPowerLaw(xs, xs.map(x => 2e-6 * x ** 3));
    assert.ok(Math.abs(exact.slope - 3) < 1e-12 && exact.r2 > 0.999999);
    assert.ok(exact.slopeStdErr < 1e-9);
    const noisy = fitPowerLaw(xs, xs.map((x, i) => x ** 2 * (1 + [0.05, -0.04, 0.03, -0.05, 0.02][i])));
    assert.ok(noisy.ci95[0] < 2 && noisy.ci95[1] > 2, 'interval covers the true exponent');
    assert.equal(slopeAgrees(noisy, 2), true);
    assert.equal(slopeAgrees(noisy, 3), false);
    assert.equal(fitPowerLaw([1, 2], [1, 4]).ci95, null);
    assert.equal(slopeAgrees(fitPowerLaw([1, 2], [1, 4]), 2), null);
    assert.equal(fitPowerLaw([0, -1, NaN], [1, 2, 3]).slope, null);
    assert.equal(t975(3), 3.182); assert.ok(Math.abs(t975(100) - 1.984) < 0.01);
});
