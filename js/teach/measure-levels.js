// The Measure tab at each visitor level (#93). Pure: no DOM, no WebGPU; js/teach/measure.js renders with these,
// test/measure-levels.test.mjs tests them.
//
// Experiments may declare `levels: ['intermediate', 'advanced']`; without it an experiment is shown at every
// level. Advanced (the default level) is the full view and must not change: every helper returns the
// Advanced text unchanged there. Beginner and Intermediate get a simpler result view:
//   - "Measured vs predicted" uses a claim's optional `plain: { claim, predicted, measured, note? }` (strings,
//     computed by interpret() from the same rows). A claim without `plain` is left out at Beginner and shown
//     as written at Intermediate.
//   - the verdict "undetermined" reads "not decided yet";
//   - Beginner sees one summary line for the verification checks instead of the list, no scope notes, only the
//     first chart that has data, and the data table behind a "Show the numbers" disclosure;
//   - timing notes, table headers and chart legends say "middle value" and "middle half of the runs" instead of
//     median and interquartile range.
// Result records (js/teach/result-record.js) are the same at every level: interpretationOf copies only the known
// claim fields, so `plain` never reaches a record.

import { LEVELS, LEVEL_INFO, DEFAULT_LEVEL, parseLevel } from './levels.js';

const levelOf = (level) => parseLevel(level) ?? DEFAULT_LEVEL;

// The levels an experiment is shown at (all of them unless it says otherwise)
export function experimentLevels(experiment) {
    return Array.isArray(experiment?.levels) ? experiment.levels : LEVELS;
}

export const experimentsForLevel = (experiments, level) =>
    experiments.filter(e => experimentLevels(e).includes(levelOf(level)));

// The nearest level above `level` that has experiments, or null
export function nextLevelWithExperiments(experiments, level) {
    return LEVELS.slice(LEVELS.indexOf(levelOf(level)) + 1).find(l => experimentsForLevel(experiments, l).length) ?? null;
}

// The note shown when no experiment of an example is written for the level
export function noExperimentsNote(experiments, level) {
    const next = nextLevelWithExperiments(experiments, level);
    const here = LEVEL_INFO[levelOf(level)].label;
    return next
        ? `The experiments for this example are not written for ${here} yet. Switch the level to ${LEVEL_INFO[next].label} to try them.`
        : `There are no experiments for ${here} here yet.`;
}

export const INTRO_SIMPLE = 'Each experiment runs on your GPU only when you press Run, and stops by itself after about '
    + '20 seconds at most. Where it can, it checks the GPU\'s answers against a careful calculation on your computer\'s main '
    + 'processor (the CPU). Your results are for your computer only: another computer will get other numbers.';

// What the result view shows at a level
export function resultView(level) {
    const l = levelOf(level);
    return {
        level: l,
        simple: l !== 'advanced',
        scope: l !== 'beginner',
        checks: l === 'beginner' ? 'summary' : 'list',
        charts: l === 'beginner' ? 'first' : 'all',
        table: l === 'beginner' ? 'disclosure' : 'open',
    };
}

// ---- Measured vs predicted ----------------------------------------------------------------------------
const AGREES = { true: 'agrees', false: 'disagrees', null: 'undetermined' };
export const agreesKey = (agrees) => AGREES[agrees === true ? 'true' : agrees === false ? 'false' : 'null'];
export function verdictText(agrees, level) {
    const key = agreesKey(agrees);
    return key === 'undetermined' && levelOf(level) !== 'advanced' ? 'not decided yet' : key;
}

// A claim's plain-language text, or null when it has none (or it is malformed): claim, predicted and measured
// must be non-empty strings, note an optional string
export function plainOf(claim) {
    const p = claim?.plain;
    if (!p || typeof p !== 'object') return null;
    if (!['claim', 'predicted', 'measured'].every(k => typeof p[k] === 'string' && p[k].trim())) return null;
    return { claim: p.claim, predicted: p.predicted, measured: p.measured,
        note: typeof p.note === 'string' && p.note.trim() ? p.note : null };
}

// interpret(rows, settings) as the experiment returns it (with `plain`), or [] when it has none or throws
export function rawClaimsOf(experiment, rows, settings) {
    if (typeof experiment?.interpret !== 'function') return [];
    try {
        const claims = experiment.interpret(rows, settings);
        return Array.isArray(claims) ? claims : [];
    } catch {
        return [];
    }
}

// The rows of the "Measured vs predicted" table at a level: [{ claim, predicted, measured, agrees, note }].
// `recorded` is record.interpretation (normalized, no `plain`), `raw` the claims interpret() returned in the
// same order. Advanced: the recorded claims as they are.
export function claimsForLevel(recorded, raw, level) {
    const l = levelOf(level);
    const out = [];
    (recorded ?? []).forEach((c, i) => {
        const plain = l === 'advanced' ? null : plainOf(raw?.[i]);
        if (plain) out.push({ ...plain, agrees: c.agrees });
        else if (l !== 'beginner') out.push({ claim: c.claim, predicted: c.predicted, measured: c.measured, agrees: c.agrees, note: c.note });
    });
    return out;
}

// ---- Verification ---------------------------------------------------------------------------------------
// One line for Beginner from the record's checks
export function checksSummary(verification) {
    const checks = verification ?? [];
    const n = checks.length;
    const count = (s) => checks.filter(c => c.status === s).length;
    const pass = count('pass'), fail = count('fail'), unchecked = count('not-checked');
    const plural = (k, word) => `${k} ${word}${k === 1 ? '' : 's'}`;
    if (!n || (n === 1 && checks[0].metric === 'none')) return 'This experiment has no automatic check of its answers.';
    if (fail) return `${fail} of ${plural(n, 'check')} failed: some of the GPU's answers do not match a careful CPU `
        + 'calculation. The Intermediate level lists every check.';
    if (!pass) return 'None of the checks could be run on these results yet.';
    if (unchecked) return `${pass} of ${plural(n, 'check')} passed, and ${unchecked} could not be run on these results.`;
    return n === 1 ? 'The check passed: the GPU\'s answer matches a careful CPU calculation.'
        : `All ${n} checks passed: the GPU's answers match a careful CPU calculation.`;
}

// ---- Timing words -----------------------------------------------------------------------------------------
// The protocol line under the results at Beginner and Intermediate (Advanced keeps measure.js's own wording)
export function simpleProtocolText(protocol) {
    const p = protocol ?? {};
    const clock = p.timing === 'timestamp-query'
        ? 'Times come from the GPU\'s own clock and from the browser\'s clock.'
        : 'Times come from the browser\'s clock (this GPU does not report its own times).';
    if (p.statistic !== 'median') return `${clock} The values are as the experiment computed them.`;
    const warm = p.warmup > 0 ? `, after ${p.warmup} warm-up run${p.warmup === 1 ? '' : 's'} that ${p.warmup === 1 ? 'is' : 'are'} not counted` : '';
    const spread = p.spread === 'iqr' ? ' The numbers in brackets, and the bars on the charts, show the range of the middle half of the runs.' : '';
    return `Each time is the middle value of ${p.repeats} runs${warm}.${spread} ${clock} One run on one computer is not `
        + 'a fixed number: run it again and the times move a little.';
}

// Results-table header of a column with a spread
export const rangeHeader = (label, level) => (levelOf(level) === 'advanced' ? `${label} (q1–q3)` : `${label} (middle half)`);

// Chart legend and aria text for the range bars (undefined at Advanced: the chart's own wording)
export const chartSpreadText = (level) => (levelOf(level) === 'advanced' ? undefined : {
    legend: 'middle value, bar = middle half of the runs',
    aria: 'Points are the middle value of repeated runs; vertical bars show the range of the middle half of the runs. ',
});

// The charts to draw: all of them, or at Beginner only the first one with a value in its rows (else the first)
export function chartsForLevel(charts, rows, level) {
    const list = charts ?? [];
    if (resultView(level).charts === 'all' || list.length <= 1) return list;
    const hasData = (c) => (rows ?? []).some(r => c.y.some(k => typeof r[k] === 'number' && Number.isFinite(r[k])));
    return [list.find(hasData) ?? list[0]];
}

// ---- Helpers for plain texts in experiments' interpret() ----------------------------------------------------
// "7.9" / "12" for a factor
export const timesText = (x) => (x >= 9.95 ? x.toFixed(0) : x.toFixed(1));
// The growth factor 2^slope of a fitted power law: how much longer twice the input takes
export const doublingFactor = (slope) => 2 ** slope;
export const countText = (n) => Math.round(n).toLocaleString('en-US');
