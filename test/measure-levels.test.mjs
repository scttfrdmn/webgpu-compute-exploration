// The Measure tab at each visitor level (#93), CPU only: the pure helpers of js/teach/measure-levels.js (level
// filtering, the plain-language claim table, the run summary that keeps completion, verification and interpretation
// apart (#105), the simpler timing words), and the plain
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
    chartsForLevel, timesText, doublingFactor, INTRO_SIMPLE, runSummary, completionText, checksLine, countChecks, claimsLine,
} from '../js/teach/measure-levels.js';
import { runState, ExperimentStopped } from '../js/teach/measure.js';
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

// ---- Completion, verification and interpretation kept apart (#105) ----------------------------------------
const PASS = { metric: 'm', status: 'pass' }, FAIL = { metric: 'bad', status: 'fail' }, NC = { metric: 'm', status: 'not-checked' };
const CHECK_MIXES = {
    'all pass': [PASS, PASS, PASS, PASS],
    'one pass': [PASS],
    'some fail': [PASS, FAIL, PASS, NC],
    'one fail': [FAIL],
    'some not-checked': [PASS, PASS, PASS, NC],
    'all not-checked': [NC, NC],
    'no checks (no verify)': [{ ...NOT_CHECKED }],
    'no checks (empty)': [],
};
const OUTCOMES = {
    done: { status: 'done' },
    budget: { status: 'budget', message: 'Stopped at the first step boundary after the 20 s time budget' },
    cancelled: { status: 'cancelled', message: 'Cancelled' },
    'device-lost': { status: 'device-lost', message: 'Stopped: the GPU device was lost' },
    error: { status: 'error', message: 'Error: boom' },
};
const CLAIMS = [{ claim: 'c', predicted: 'p', measured: 'm', agrees: true, note: null }];
const runOf = (status, checks, { steps = 3, claims = CLAIMS } = {}) => ({
    outcome: OUTCOMES[status], steps, plannedSteps: 5, elapsedMs: 4321, budgetMs: 20000,
    covered: { label: 'N', first: '64', last: '256' }, checks, claims,
});

test('completion: the original status at Advanced, how much of the run there is below it', () => {
    assert.equal(completionText(runOf('done'), 'advanced'), 'Done: 3 steps in 4.3 s.');
    assert.equal(completionText(runOf('budget'), 'advanced'), 'Stopped at the first step boundary after the 20 s time budget (3 steps completed).');
    assert.equal(completionText(runOf('cancelled', null, { steps: 1 }), 'advanced'), 'Cancelled (1 step completed).');
    assert.equal(completionText(runOf('cancelled', null, { steps: 0 }), 'advanced'), 'Cancelled.');
    assert.equal(completionText(runOf('error'), 'advanced'), 'Error: boom (3 steps completed).');
    for (const level of ['beginner', 'intermediate']) {
        assert.equal(completionText(runOf('done'), level), 'Finished: 3 steps in 4.3 s.');
        assert.equal(completionText(runOf('budget'), level), 'Stopped early by the 20 s time limit after 3 of 5 steps: these results cover only N = 64 … 256.');
        assert.equal(completionText(runOf('cancelled'), level), 'Cancelled after 3 of 5 steps: these results cover only N = 64 … 256.');
        assert.equal(completionText(runOf('device-lost'), level), 'Stopped early because the browser lost the GPU after 3 of 5 steps: these results cover only N = 64 … 256.');
        assert.equal(completionText(runOf('error'), level), 'Stopped early by an error after 3 of 5 steps: these results cover only N = 64 … 256. Error: boom.');
        assert.equal(completionText({ ...runOf('budget'), steps: 0, covered: null }, level), 'Stopped early by the 20 s time limit before the first step finished: there are no results.');
        assert.equal(completionText({ ...runOf('budget'), steps: 1, covered: { label: 'N', first: '64', last: '64' } }, level),
            'Stopped early by the 20 s time limit after 1 of 5 steps: these results cover only N = 64.');
        assert.equal(completionText({ ...runOf('cancelled'), plannedSteps: null, covered: null }, level), 'Cancelled after 3 steps: these results are only part of the run.');
        assert.equal(completionText({ ...runOf('cancelled'), plannedSteps: 2 }, level), 'Cancelled after 3 steps: these results cover only N = 64 … 256.', 'a planned count below the rows is ignored');
    }
});

test('the check line counts pass, fail and not-checked honestly', () => {
    const line = (checks, status = 'done') => checksLine(checks, OUTCOMES[status]);
    assert.deepEqual(line(CHECK_MIXES['all pass']), { status: 'pass', text: 'All 4 checks passed: the GPU\'s answers match a careful CPU calculation.' });
    assert.equal(line(CHECK_MIXES['all pass'], 'budget').text, 'All 4 checks passed on the steps that finished: the GPU\'s answers match a careful CPU calculation.');
    assert.equal(line(CHECK_MIXES['one pass']).text, 'The check passed: the GPU\'s answer matches a careful CPU calculation.');
    assert.equal(line(CHECK_MIXES['some fail']).text, '1 of 4 checks failed, 2 passed and 1 could not be checked: some of the GPU\'s answers do not match a careful CPU calculation.');
    assert.equal(line([FAIL, FAIL, PASS]).text, '2 of 3 checks failed, 1 passed: some of the GPU\'s answers do not match a careful CPU calculation.');
    assert.equal(line(CHECK_MIXES['one fail']).text, 'The check failed: some of the GPU\'s answers do not match a careful CPU calculation.');
    assert.deepEqual(line(CHECK_MIXES['some not-checked']), { status: 'partial', text: '3 of 4 checks passed, 1 could not be checked.' });
    assert.equal(line(CHECK_MIXES['all not-checked']).text, 'None of the 2 checks could be run on these results.');
    assert.equal(line([NC]).text, 'The check could not be run on these results.');
    assert.equal(line(CHECK_MIXES['no checks (no verify)']).status, 'none');
    assert.equal(line(CHECK_MIXES['no checks (empty)']).text, 'This experiment has no automatic check of its answers.');
    assert.equal(checksSummary(CHECK_MIXES['all pass']), line(CHECK_MIXES['all pass']).text);
    assert.deepEqual(countChecks(CHECK_MIXES['some fail']), { n: 4, pass: 2, fail: 1, unchecked: 1, none: false });
    assert.deepEqual(countChecks([{ ...NOT_CHECKED }]), { n: 0, pass: 0, fail: 0, unchecked: 0, none: true });
});

// Every outcome × check mix × level: completion first, then checks, then claims; the wording rules of #105
test('run summary: every outcome × check mix × level keeps completion, checks and claims apart', (t) => {
    let cases = 0;
    for (const [status, outcome] of Object.entries(OUTCOMES)) {
        for (const [mix, checks] of Object.entries(CHECK_MIXES)) {
            for (const level of LEVELS) {
                for (const claims of [CLAIMS, [{ ...CLAIMS[0], agrees: false }], []]) {
                    const where = `${status} / ${mix} / ${level} / ${claims.length ? `claim ${claims[0].agrees}` : 'no claims'}`;
                    const lines = runSummary(runOf(status, checks, { claims }), level);
                    cases++;
                    assert.equal(lines[0].kind, 'completion', where);
                    assert.equal(lines[0].status, status, where);
                    assert.equal(lines[0].text, completionText(runOf(status, checks), level), where);
                    if (level === 'advanced') {
                        assert.equal(lines.length, 1, `${where}: Advanced shows its checks as the list`);
                        continue;
                    }
                    const kinds = lines.map(l => l.kind);
                    assert.deepEqual(kinds, ['completion', 'checks', ...(claims.length ? ['claims'] : [])], where);
                    const done = status === 'done';
                    // A run that did not complete says so first, before any check
                    if (!done) assert.match(lines[0].text, /^(Stopped early|Cancelled)\b.*: these results cover only N = 64 … 256\./, where);
                    else assert.match(lines[0].text, /^Finished: /, where);
                    const c = lines[1], fails = checks.filter(x => x.status === 'fail').length;
                    const real = checks.filter(x => x.metric !== 'none');
                    // "All … passed" only when there are checks and every one passed
                    if (/\bAll \d+ checks passed|^The check passed/.test(c.text)) {
                        assert.ok(real.length > 0 && real.every(x => x.status === 'pass'), `${where}: ${c.text}`);
                    }
                    if (!real.length) assert.equal(c.status, 'none', where);
                    if (fails) {
                        assert.equal(c.status, 'fail', where);
                        assert.match(c.text, /failed/, where);
                        assert.equal(c.failed.length, fails, `${where}: every failed check is listed`);
                    } else assert.doesNotMatch(c.text, /failed/, where);
                    if (real.some(x => x.status === 'not-checked')) assert.match(c.text, /could not be (checked|run)|could be run/, where);
                    if (real.length && !done && c.status === 'pass') assert.match(c.text, /on the steps that finished/, where);
                    // A verdict never stands in for a check: the claims line says so, and after a failure it warns
                    if (claims.length) {
                        const note = lines[2];
                        if (fails) assert.match(note.text, /^A check failed/, where);
                        else assert.match(note.text, /does not say if the GPU's answers are right/, where);
                        assert.doesNotMatch(note.text, /passed|correct\b/, where);
                    }
                }
            }
        }
    }
    t.diagnostic(`${cases} combinations`);
    assert.equal(cases, 5 * 8 * 3 * 3);
});

test('run summary: without a record only the completion line', () => {
    for (const level of LEVELS) {
        const lines = runSummary({ ...runOf('cancelled', null, { steps: 0, claims: [] }), covered: null }, level);
        assert.deepEqual(lines.map(l => l.kind), ['completion']);
    }
});

test('the Beginner run summaries stay within the Beginner reading rules', (t) => {
    const rules = LEVEL_RULES.beginner;
    const texts = new Set();
    for (const status of Object.keys(OUTCOMES)) {
        for (const checks of Object.values(CHECK_MIXES)) {
            for (const claims of [CLAIMS, []]) {
                for (const steps of [0, 1, 3]) {
                    // the error message and the column label are the experiment's, not ours
                    const run = { ...runOf(status, checks, { steps, claims }), outcome: { ...OUTCOMES[status], message: undefined } };
                    for (const line of runSummary(run, 'beginner')) texts.add(line.text);
                }
            }
        }
    }
    texts.add(checksLine([FAIL, FAIL, PASS]).text);
    texts.add(claimsLine('fail').text);
    for (const text of texts) {
        assert.ok(rules.math || !/\$/.test(text), `math in: ${text}`);
        assert.deepEqual(avoidedTerms(text, rules.avoid), [], `avoided terms in: ${text}`);
        assert.doesNotMatch(text, /median|quartile|IQR|95%|budget|verif|interpret/i, text);
    }
    const r = readability([...texts].join(' '));
    t.diagnostic(`${texts.size} texts, ${r.words} words: grade ${r.grade.toFixed(1)} (≤ ${rules.maxGrade}), long words ${(100 * r.longWords).toFixed(1)}% (≤ ${(100 * rules.maxLongWords).toFixed(0)}%)`);
    assert.ok(r.grade <= rules.maxGrade, `grade ${r.grade.toFixed(1)}`);
    assert.ok(r.longWords <= rules.maxLongWords, `long words ${(100 * r.longWords).toFixed(1)}%`);
});

test('runState: outcome, steps of the planned count, and the first column covered', () => {
    const experiment = { columns: [{ key: 'n', label: 'N' }, { key: 'ms', label: 'ms' }], plannedSteps: ({ maxN }) => Math.log2(maxN) - 5 };
    const rows = [{ n: 64 }, { n: 128 }, { n: 16384 }];
    const s = runState(experiment, { maxN: 1024 }, rows, new ExperimentStopped('budget', 20000), { elapsedMs: 21000, budgetMs: 20000 });
    assert.deepEqual(s, { outcome: { status: 'budget', message: 'Stopped at the first step boundary after the 20 s time budget' },
        steps: 3, plannedSteps: 5, elapsedMs: 21000, budgetMs: 20000, covered: { label: 'N', first: '64', last: '16,384' } });
    assert.equal(runState({ ...experiment, plannedSteps: () => 2 }, { maxN: 1024 }, rows, null).plannedSteps, null, 'fewer than the rows: unknown');
    assert.equal(runState({ ...experiment, plannedSteps: () => { throw new Error('x'); } }, {}, rows, null).plannedSteps, null);
    assert.equal(runState({ columns: [] }, {}, [], null).covered, null);
});

// plannedSteps of the registered experiments, with their default settings, against the lists their run() walks
test('plannedSteps of the registered experiments', async (t) => {
    let found = 0;
    for (const id of MEASURE) {
        for (const e of await experimentsOf(id)) {
            if (typeof e.plannedSteps !== 'function') continue;
            const settings = Object.fromEntries((e.settings ?? []).map(s => [s.key, s.default ?? s.options[0].value]));
            const p = e.plannedSteps(settings);
            assert.ok(Number.isInteger(p) && p > 0 && p <= 100, `${id}/${e.id}: ${p}`);
            t.diagnostic(`${id}/${e.id}: ${p} steps with ${JSON.stringify(settings)}`);
            found++;
        }
    }
    const sort = (await experimentsOf('parallel-sort')).find(e => e.id === 'scaling');
    assert.equal(sort.plannedSteps({ maxN: 262144 }), 5);
    assert.equal(sort.plannedSteps({ maxN: 4194304 }), 9);
    const mm = (await experimentsOf('matrix-multiplication')).find(e => e.id === 'cpu-naive-tiled');
    assert.equal(mm.plannedSteps({ maxN: 1024 }), 7);
    assert.equal((await experimentsOf('nbody')).find(e => e.id === 'theta-error').plannedSteps({}), 6);
    // every experiment shown at Beginner says how many steps it plans
    for (const id of MEASURE) {
        for (const e of experimentsForLevel(await experimentsOf(id), 'beginner')) {
            assert.equal(typeof e.plannedSteps, 'function', `${id}/${e.id}: shown at Beginner without plannedSteps`);
        }
    }
    assert.ok(found >= 8);
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
