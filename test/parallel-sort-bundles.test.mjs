// Parallel sort interpret() on the recorded physical-GPU runs (data/validation/apple-silicon-*, an M4 Pro in
// Chrome 156 and Safari 27.2, swept to N = 1,048,576 before the fixes for #83/#84). CPU only: the stored rows
// are fed to the current interpret() and its verdicts printed. Checked: the timer resolution inferred from the
// rows (Chrome 0.1 ms wall, 2¹⁶ ns timestamps; Safari 1 ms wall); no crossover is declared from times below
// 10× their clock's resolution; GPU-vs-CPU crossovers use GPU timestamps where they are consistent with the
// wall clock; Safari's radix timestamps, longer than the wall time of the same sorts, are rejected; the
// slopes stay undetermined where fewer than 3 sizes reach 1 ms (these runs stop at 2²⁰).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import experiments, { resolutions, clockFor, crossover } from '../js/teach/experiments/parallel-sort.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const BUNDLES = { chrome: 'apple-silicon-chrome/2026-11-02-6fd3b07.json', safari: 'apple-silicon-safari26/2026-11-02-6fd3b07.json' };
const e = experiments[0];

function records(browser) {
    const bundle = JSON.parse(readFileSync(join(root, 'data/validation', BUNDLES[browser]), 'utf8'));
    const out = {};
    for (const r of bundle.records) {
        if (r.experiment.exampleId === 'parallel-sort' && r.experiment.id === 'scaling') out[r.settings.dataType] = r;
    }
    assert.deepEqual(Object.keys(out).sort(), ['f32', 'u32'], `${browser}: u32 and f32 records`);
    return out;
}

function show(t, label, claims) {
    for (const c of claims) t.diagnostic(`${label} [${c.agrees}] ${c.key}: ${c.claim} | predicted ${c.predicted} | measured ${c.measured}`);
}

const verdicts = (claims) => Object.fromEntries(claims.map(c => [c.key, c.agrees]));

test('Chrome (M4 Pro): 0.1 ms wall clock, 2¹⁶ ns timestamps; crossovers on timestamps, only where resolvable', (t) => {
    for (const [type, record] of Object.entries(records('chrome'))) {
        const { rows } = record;
        const res = resolutions(rows);
        t.diagnostic(`chrome ${type}: wall resolution ${res.wall} ms (${res.wallSource}), timestamps ${res.gpu} ms`);
        assert.ok(Math.abs(res.wall - 0.1) < 1e-6);
        assert.ok(Math.abs(res.gpu - 0.065536) < 1e-3);
        const claims = e.interpret(rows, record.settings);
        show(t, `chrome ${type}`, claims);
        for (const k of ['bitonic', 'shared', 'radix']) {
            assert.equal(clockFor(rows, k, res).key, `${k}Gpu`, `${type} ${k}: timestamps consistent with the wall clock`);
            const c = crossover(rows, k);
            assert.equal(c.kind, 'below', `${type} ${k}: faster from the smallest resolvable N`);
            assert.ok(c.undeterminedBelow && c.skipped.includes(16384), 'the small sizes are below resolution');
            assert.ok(c.at.n >= 131072);
            const claim = claims.find(x => x.key === `crossover-${k}`);
            assert.match(claim.claim, /\(GPU timestamps\)/);
            assert.match(claim.measured, /whether the crossover lies lower is undetermined/);
        }
        assert.equal(verdicts(claims)['cpu-slope'], true, 'the CPU sort grows like N log N');
    }
    const u32 = verdicts(e.interpret(records('chrome').u32.rows));
    assert.equal(u32['radix-slope'], null, 'radix reaches 1 ms only at 2²⁰: a run to 4,194,304 is needed');
    assert.equal(u32.ranking, null, 'global and workgroup bitonic overlap at 2²⁰');
    assert.equal(verdicts(e.interpret(records('chrome').f32.rows)).ranking, true);
});

test('Safari (M4 Pro): 1 ms wall clock; radix timestamps exceed the wall time and are not used; no quantized crossovers', (t) => {
    for (const [type, record] of Object.entries(records('safari'))) {
        const { rows } = record;
        const res = resolutions(rows);
        t.diagnostic(`safari ${type}: wall resolution ${res.wall} ms (${res.wallSource}), timestamps ${res.gpu} ms`);
        assert.equal(res.wall, 1);
        assert.ok(res.gpu < 0.02, 'Safari does not quantize timestamps');
        const claims = e.interpret(rows, record.settings);
        show(t, `safari ${type}`, claims);
        // Radix: GPU time 5.6 ms where submit → done took 3 ms at 2²⁰
        const radix = clockFor(rows, 'radix', res);
        assert.equal(radix.key, 'radixWall');
        assert.ok(radix.inconsistent.includes(1048576));
        // Every compared size has both medians ≥ 10 ms on the 1 ms clock, so a wall-clock radix sort of ≤ 3 ms
        // decides nothing
        assert.equal(crossover(rows, 'radix').kind, 'below-resolution');
        for (const k of ['bitonic', 'shared', 'radix']) {
            const c = crossover(rows, k);
            assert.ok(['below', 'below-resolution'].includes(c.kind), `${type} ${k}: ${c.kind}`);
            if (c.kind === 'below') assert.ok(c.at.cpuMs >= 10, 'compared only where the CPU sort spans ≥ 10 ticks');
        }
        assert.match(claims.find(c => c.key === 'ranking').claim, /wall clock/, 'not every algorithm has consistent timestamps');
        // The CPU sort time is wall-only and quantized: in u32 only 2²⁰ (17 ms) is ≥ 10 ms
        if (type === 'u32') assert.equal(verdicts(claims)['cpu-slope'], null);
    }
    const u32 = e.interpret(records('safari').u32.rows);
    assert.match(u32.find(c => c.key === 'ranking').measured, /^radix 3\.00 ms/, 'by the wall clock radix is fastest, as in Chrome');
});
