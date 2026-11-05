// The Measure tab at each visitor level (#93), CPU only: the pure helpers of js/teach/measure-levels.js (level
// filtering, the plain-language claim table, the verification summary, the simpler timing words), and the plain
// texts of the three flagship experiments computed by interpret() from the rows of the committed physical-GPU
// bundles (data/validation/apple-silicon-*/2026-11-02-097c558.json: an M4 Pro in Chrome and Safari). Every
// flagship claim shown at Beginner with a verdict must have a plain text, and the Beginner texts must stay within
// LEVEL_RULES.beginner of js/teach/readability.js (no math, no avoided terms, grade and long-word share).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import {
    experimentLevels, experimentsForLevel, nextLevelWithExperiments, noExperimentsNote, resultView, claimsForLevel,
    plainOf, rawClaimsOf, verdictText, agreesKey, checksSummary, simpleProtocolText, rangeHeader, chartSpreadText,
    chartsForLevel, timesText, doublingFactor, INTRO_SIMPLE,
} from '../js/teach/measure-levels.js';
import { interpretationOf, NOT_CHECKED } from '../js/teach/result-record.js';
import { LEVELS } from '../js/teach/levels.js';
import { MEASURE } from '../js/teach/registry.js';
import { FLAGSHIPS } from '../js/teach/flagships.js';
import { LEVEL_RULES, readability, avoidedTerms } from '../js/teach/readability.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const BUNDLES = {
    chrome: 'apple-silicon-chrome/2026-11-02-097c558.json',
    safari: 'apple-silicon-safari26/2026-11-02-097c558.json',
};
const experimentsOf = async (id) => (await import(pathToFileURL(join(root, 'js/teach/experiments', `${id}.js`)).href)).default;
const bundle = (name) => JSON.parse(readFileSync(join(root, 'data/validation', BUNDLES[name]), 'utf8'));

test('level filtering: default all levels, declared levels, the next level up', () => {
    const all = { id: 'a' }, upper = { id: 'b', levels: ['intermediate', 'advanced'] }, adv = { id: 'c', levels: ['advanced'] };
    assert.deepEqual(experimentLevels(all), LEVELS);
    assert.deepEqual(experimentsForLevel([all, upper, adv], 'beginner'), [all]);
    assert.deepEqual(experimentsForLevel([all, upper, adv], 'intermediate'), [all, upper]);
    assert.deepEqual(experimentsForLevel([all, upper, adv], 'advanced'), [all, upper, adv]);
    assert.deepEqual(experimentsForLevel([all, upper], 'nonsense'), [all, upper], 'an unknown level is the default (Advanced)');
    assert.equal(nextLevelWithExperiments([upper, adv], 'beginner'), 'intermediate');
    assert.equal(nextLevelWithExperiments([adv], 'beginner'), 'advanced');
    assert.equal(nextLevelWithExperiments([adv], 'advanced'), null);
    assert.match(noExperimentsNote([upper], 'beginner'), /not written for Beginner yet\. Switch the level to Intermediate/);
    assert.match(noExperimentsNote([adv], 'intermediate'), /Switch the level to Advanced/);
});

test('what each level shows: Advanced is the full view', () => {
    assert.deepEqual(resultView('advanced'), { level: 'advanced', simple: false, scope: true, checks: 'list', charts: 'all', table: 'open' });
    assert.deepEqual(resultView('intermediate'), { level: 'intermediate', simple: true, scope: true, checks: 'list', charts: 'all', table: 'open' });
    assert.deepEqual(resultView('beginner'), { level: 'beginner', simple: true, scope: false, checks: 'summary', charts: 'first', table: 'disclosure' });
    for (const a of [true, false, null]) assert.equal(verdictText(a, 'advanced'), agreesKey(a));
    assert.equal(verdictText(null, 'advanced'), 'undetermined');
    assert.equal(verdictText(null, 'beginner'), 'not decided yet');
    assert.equal(verdictText(true, 'intermediate'), 'agrees');
    assert.equal(verdictText(false, 'beginner'), 'disagrees');
    assert.equal(rangeHeader('wall ms', 'advanced'), 'wall ms (q1–q3)');
    assert.equal(rangeHeader('wall ms', 'beginner'), 'wall ms (middle half)');
    assert.equal(chartSpreadText('advanced'), undefined);
    for (const text of [INTRO_SIMPLE, simpleProtocolText({ statistic: 'median', repeats: 5, warmup: 1, spread: 'iqr', timing: 'wall' }),
        chartSpreadText('beginner').legend, chartSpreadText('beginner').aria]) {
        assert.doesNotMatch(text, /median|quartile|IQR|q1|95%|log-log|slope/i, text);
    }
    assert.match(simpleProtocolText({ statistic: 'median', repeats: 5, warmup: 1, spread: 'iqr', timing: 'timestamp-query' }),
        /middle value of 5 runs, after 1 warm-up run that is not counted\..*middle half.*GPU's own clock/);
    assert.match(simpleProtocolText({ statistic: 'as-reported', timing: 'wall' }), /as the experiment computed them/);
});

test('charts: all of them, or at Beginner the first one with data', () => {
    const charts = [{ y: ['gpuMs'] }, { y: ['wallMs'] }, { y: ['err'] }];
    const noGpu = [{ gpuMs: null, wallMs: 2, err: 1e-6 }];
    assert.deepEqual(chartsForLevel(charts, noGpu, 'advanced'), charts);
    assert.deepEqual(chartsForLevel(charts, noGpu, 'intermediate'), charts);
    assert.deepEqual(chartsForLevel(charts, noGpu, 'beginner'), [charts[1]], 'the GPU-time chart is empty without timestamps');
    assert.deepEqual(chartsForLevel(charts, [{ gpuMs: 1, wallMs: 2 }], 'beginner'), [charts[0]]);
    assert.deepEqual(chartsForLevel(charts, [], 'beginner'), [charts[0]], 'no rows yet: the first chart');
});

test('Measured vs predicted: plain texts at Beginner and Intermediate, recorded claims at Advanced', () => {
    const raw = [
        { key: 'a', claim: 'slope', predicted: '3', measured: '2.98 (95% CI 2.90 … 3.07)', agrees: true, note: 'n',
            plain: { claim: 'Twice as wide', predicted: 'About 8 times as long', measured: '7.9 times as long' } },
        { key: 'b', claim: 'bandwidth', predicted: '≤ 0.25 × bandwidth', measured: '1460 GB/s', agrees: null, note: null },
        { key: 'c', claim: 'bad plain', predicted: 'x', measured: 'y', agrees: false, plain: { claim: 'only a claim' } },
    ];
    const experiment = { interpret: () => raw };
    const { interpretation: recorded } = interpretationOf(experiment, [], {});
    for (const c of recorded) assert.ok(!('plain' in c), 'records never carry plain');
    assert.deepEqual(claimsForLevel(recorded, raw, 'advanced'), recorded.map(({ claim, predicted, measured, agrees, note }) => ({ claim, predicted, measured, agrees, note })));
    const inter = claimsForLevel(recorded, raw, 'intermediate');
    assert.deepEqual(inter.map(c => c.claim), ['Twice as wide', 'bandwidth', 'bad plain'], 'without (valid) plain: as written');
    assert.equal(inter[0].note, null, 'the statistics note is not shown with a plain text');
    assert.deepEqual(claimsForLevel(recorded, raw, 'beginner'), [{ claim: 'Twice as wide', predicted: 'About 8 times as long',
        measured: '7.9 times as long', note: null, agrees: true }], 'without plain: left out at Beginner');
    assert.equal(plainOf(raw[2]), null);
    assert.equal(plainOf({ plain: { claim: 'a', predicted: 'b', measured: '' } }), null);
    assert.deepEqual(plainOf({ plain: { claim: 'a', predicted: 'b', measured: 'c', note: 'd' } }), { claim: 'a', predicted: 'b', measured: 'c', note: 'd' });
    assert.deepEqual(rawClaimsOf({ interpret: () => { throw new Error('boom'); } }, [], {}), []);
    assert.deepEqual(rawClaimsOf({}, [], {}), []);
    // interpret() failed: the record's fallback claim, no plain text, so nothing at Beginner
    const failed = interpretationOf({ interpret: () => { throw new Error('boom'); } }, [], {}).interpretation;
    assert.equal(claimsForLevel(failed, [], 'beginner').length, 0);
    assert.equal(claimsForLevel(failed, [], 'intermediate').length, 1);
});

test('verification at Beginner: one summary line', () => {
    const pass = { metric: 'm', status: 'pass' }, fail = { metric: 'm', status: 'fail' }, nc = { metric: 'm', status: 'not-checked' };
    assert.equal(checksSummary([pass, pass, pass, pass]), 'All 4 checks passed: the GPU\'s answers match a careful CPU calculation.');
    assert.equal(checksSummary([pass]), 'The check passed: the GPU\'s answer matches a careful CPU calculation.');
    assert.match(checksSummary([pass, fail, pass]), /^1 of 3 checks failed: some of the GPU's answers do not match/);
    assert.equal(checksSummary([pass, pass, nc]), '2 of 3 checks passed, and 1 could not be run on these results.');
    assert.match(checksSummary([nc, nc]), /^None of the checks could be run/);
    assert.equal(checksSummary([{ ...NOT_CHECKED }]), 'This experiment has no automatic check of its answers.');
    assert.equal(checksSummary([]), 'This experiment has no automatic check of its answers.');
});

test('plain-text helpers', () => {
    assert.equal(timesText(doublingFactor(3)), '8.0');
    assert.equal(timesText(doublingFactor(2.98)), '7.9');
    assert.equal(timesText(12.4), '12');
    assert.equal(timesText(9.96), '10');
});

test('levels of the registered experiments: one Beginner experiment per flagship and per Beginner Understand page', async (t) => {
    const atBeginner = {};
    for (const id of MEASURE) {
        const experiments = await experimentsOf(id);
        for (const level of LEVELS) {
            const shown = experimentsForLevel(experiments, level).map(e => e.id);
            if (level === 'beginner') atBeginner[id] = shown;
            if (level !== 'beginner') assert.ok(shown.length > 0, `${id}: no experiment at ${level}`);
        }
        assert.equal(experimentsForLevel(experiments, 'advanced').length, experiments.length, `${id}: every experiment at Advanced`);
        t.diagnostic(`${id}: Beginner ${atBeginner[id].join(', ') || '—'}`);
    }
    assert.deepEqual(atBeginner['matrix-multiplication'], ['cpu-naive-tiled']);
    assert.deepEqual(atBeginner.nbody, ['theta-error']);
    assert.deepEqual(atBeginner['parallel-sort'], ['scaling']);
    for (const id of ['boids', 'falling-sand', 'game-of-life', 'wave-equation', 'wasm-mandelbrot']) {
        assert.equal(atBeginner[id].length, 1, `${id}: one Beginner experiment`);
    }
    for (const f of FLAGSHIPS) assert.ok(atBeginner[f.id].length, `${f.id}: flagship without a Beginner experiment`);
});

// A Beginner Understand page that says "run **<title>**" in Measure must name an experiment shown at Beginner
test('Beginner Understand pages point to experiments that are shown at Beginner', async () => {
    const { UNDERSTAND_LEVELS } = await import('../js/teach/registry.js');
    let found = 0;
    for (const id of MEASURE) {
        if (!UNDERSTAND_LEVELS.beginner.has(id)) continue;
        const md = readFileSync(join(root, 'content/understand', `${id}.beginner.md`), 'utf8');
        const titles = experimentsForLevel(await experimentsOf(id), 'beginner').map(e => e.title);
        for (const m of md.matchAll(/\(#[a-z0-9-]+\/measure\), run \*\*([^*]+)\*\*/g)) {
            assert.ok(titles.includes(m[1]), `${id}.beginner.md asks for "${m[1]}", not shown at Beginner (${titles.join(', ')})`);
            found++;
        }
    }
    assert.ok(found >= 5, `${found} experiment names found in the Beginner pages`);
});

// The flagship experiments on the recorded M4 Pro rows: the plain texts a Beginner would see
async function beginnerClaims(name) {
    const out = [];
    for (const r of bundle(name).records) {
        if (!FLAGSHIPS.some(f => f.id === r.experiment.exampleId)) continue;
        const e = experimentsForLevel(await experimentsOf(r.experiment.exampleId), 'beginner').find(x => x.id === r.experiment.id);
        if (!e) continue;
        out.push({ record: r, experiment: e, claims: rawClaimsOf(e, r.rows, r.settings) });
    }
    return out;
}

for (const name of Object.keys(BUNDLES)) {
    test(`${name} bundle: every flagship claim with a verdict has a plain text at Beginner`, async (t) => {
        const runs = await beginnerClaims(name);
        assert.deepEqual([...new Set(runs.map(r => `${r.record.experiment.exampleId}/${r.experiment.id}`))].sort(),
            ['matrix-multiplication/cpu-naive-tiled', 'nbody/theta-error', 'parallel-sort/scaling']);
        for (const { record, experiment, claims } of runs) {
            const where = `${record.experiment.exampleId}/${experiment.id} ${JSON.stringify(record.settings)}`;
            const recorded = interpretationOf(experiment, record.rows, record.settings).interpretation;
            assert.equal(recorded.length, claims.length);
            for (const c of claims) {
                if (c.agrees !== null) assert.ok(plainOf(c), `${where} ${c.key}: a verdict but no plain text`);
                const p = plainOf(c);
                t.diagnostic(`${where} [${verdictText(c.agrees, 'beginner')}] ${c.key}: ${p ? `${p.claim} | ${p.predicted} | ${p.measured}` : '(not shown at Beginner)'}`);
            }
            assert.ok(claimsForLevel(recorded, claims, 'beginner').length > 0, `${where}: something to show at Beginner`);
        }
    });
}

test('the Beginner plain texts stay within the Beginner reading rules', async (t) => {
    const rules = LEVEL_RULES.beginner;
    const texts = [];
    for (const name of Object.keys(BUNDLES)) {
        for (const { claims } of await beginnerClaims(name)) {
            for (const c of claims) {
                const p = plainOf(c);
                if (p) texts.push(...[p.claim, p.predicted, p.measured, p.note].filter(Boolean));
            }
        }
    }
    assert.ok(texts.length > 0);
    for (const text of texts) {
        assert.ok(rules.math || !/\$/.test(text), `math in a Beginner text: ${text}`);
        assert.deepEqual(avoidedTerms(text, rules.avoid), [], `avoided terms in: ${text}`);
        assert.doesNotMatch(text, /median|quartile|IQR|q1|q3|95%|CI\b|log-log|slope|O\(/i, `statistics or complexity terms in: ${text}`);
    }
    // Each text is one short sentence, too short for a stable grade: the bounds apply to all of them together
    const r = readability(texts.map(s => `${s}.`).join(' '));
    t.diagnostic(`${texts.length} texts, ${r.words} words: grade ${r.grade.toFixed(1)} (≤ ${rules.maxGrade}), long words ${(100 * r.longWords).toFixed(1)}% (≤ ${(100 * rules.maxLongWords).toFixed(0)}%)`);
    assert.ok(r.grade <= rules.maxGrade, `grade ${r.grade.toFixed(1)}`);
    assert.ok(r.longWords <= rules.maxLongWords, `long words ${(100 * r.longWords).toFixed(1)}%`);
});
