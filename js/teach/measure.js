// Measure tab: user-started, bounded experiments that run on the visitor's GPU and compare against CPU
// references in the browser.
//
// An experiment module (js/teach/experiments/<id>.js) default-exports an array of experiments:
//   {
//     id, title, question,                       // question: one sentence the experiment answers
//     settings: [{ key, label, options: [{ value, label }], default }],   // optional fixed choices
//     budgetMs,                                  // time budget for the run (≤ MAX_BUDGET_MS, 20 s), see "Budget"
//     columns: [{ key, label, format?, unit? }], // results table; unit (e.g. 'ms') is recorded in result files
//     charts: [{ x, xLabel, logX?, y: [keys], yLabel, logY?, labels?: {key: label},
//                reference?: { label, points } | (rows) => ({ label, points }),   // e.g. a trend through your data
//                emptyNote? }],                // shown when no series has a value (e.g. GPU ms without timestamps)
//     baseline,                                  // optional text: what CI measured, for comparison
//     verify?(rows, settings) → [{ metric, value, tolerance, reference, status: 'pass'|'fail'|'not-checked' }],
//                                                // optional checks recorded with the result (see "Verification")
//     async *run(ctx, settings) { ... yield row ... }   // one row per sweep step
//   }
//
// Budget. Every experiment's problem sizes come from fixed lists with hard caps, so each step is bounded
// GPU work; that cap, not the clock, is what bounds the GPU. Call `await ctx.step()` between steps: it waits
// for submitted GPU work, yields to the page, and throws ExperimentStopped if the run was cancelled, the
// device was lost, or the budget is spent. The budget is only checked there, between steps (and between
// the repeats of ctx.measure): a run stops at the first step boundary after the budget, never in the middle
// of a submission, so a run can end a little after budgetMs. Stop reasons: 'cancelled', 'budget',
// 'device-lost' (ExperimentStopped#reason; the messages are STOP_MESSAGES).
//
// Measurement protocol (for every timed quantity):
//   const m = await ctx.measure(runOnce, { warmup = 1, repeats = 5 });
// runOnce() performs one independent measurement (set up, upload, dispatch, read back; no state carried
// over from the previous call) and resolves to { wallMs, gpuMs?, phases?: { compile?, upload?, dispatch?,
// readback? } } in ms, plus anything else the experiment needs (e.g. the read-back results). ctx.measure
// runs `warmup` calls and discards them (first-use shader compilation, pipeline caches, clocks ramping up),
// then `repeats` calls, with ctx.step() between every two calls, and resolves to
//   { warmup, repeats, n, wallMs: S, gpuMs: S, phases: { <phase>: S }, samples: [runOnce results] }
// where S = summarize(values) from js/teach/stats.js: { n, median, min, max, q1, q3, mean, std, cv }
// (type-7 quantiles; non-finite values such as a missing gpuMs are dropped, so gpuMs.n = 0 without
// timestamp-query). Caps: warmup ≤ MAX_WARMUP, repeats ≤ MAX_REPEATS.
//   - Report a timed quantity as its median with its spread: `...withSpread('wallMs', m.wallMs)` (stats.js)
//     sets row.wallMs = median, row.wallMsLo = q1, row.wallMsHi = q3. The table then shows
//     "median (q1–q3)", the chart draws a range bar at each point, the CSV gets q1/q3 columns, and the
//     result record marks the column `range: true`. Only <key>, <key>Lo, <key>Hi: no other extra row keys.
//   - Phases are wall-clock times with the meanings of js/phase-clock.js (compile, upload, dispatch,
//     readback); compute-only APIs time them with a PhaseClock. GPU time (timestamp-query, gpuMs) is a
//     separate quantity, never mixed into the phases.
//   - Deterministic outputs (errors, interaction counts) are not repeated statistics; take them from one
//     sample (e.g. m.samples.at(-1)).
//   - Long simulations (e.g. energy drift over hundreds of steps) may report a single wall time; their
//     result records say statistic 'as-reported' instead of 'median'.
// When GPU work goes through an example's compute-only export (e.g. measureForces in nbody.js), that
// function owns its GpuTimer and returns gpuMs/wallMs; experiments that dispatch directly use
// ctx.newTimer(). Columns like θ or dt that should print as written use `format: String`.
//
// Verification. verify(rows, settings) checks the rows (possibly a partial run) against independent
// references and CI-derived bounds; return 'not-checked' for a check whose rows are missing. Without
// verify the record says a single 'not-checked'. Use check() from js/teach/result-record.js for bounds.
//
// Results. Each finished run (also a stopped one, if it produced rows) becomes a result record, schema v1
// (schema/experiment-result.v1.json, js/teach/result-record.js), kept for the device report, downloadable
// from the card and comparable with a downloaded file.

import { getGPUDevice, isDeviceLost, on, recordExperimentResult } from './app-state.js';
import { GpuTimer } from '../gpu-timer.js';
import { renderLineChart, formatWithRange } from './chart.js';
import { experimentsModule } from './registry.js';
import { pauseExplore } from './tabs.js';
import { summarize, hasRange, SPREAD_LABEL } from './stats.js';
import { buildResultRecord, validateResultRecord, compareRecords, ratioTable, resultFileName, SCHEMA_PATH } from './result-record.js';
import { sourceInfo } from '../source-info.js';

export const MAX_BUDGET_MS = 20000;
export const MAX_WARMUP = 5;
export const MAX_REPEATS = 20;
export const MAX_IMPORT_BYTES = 2 * 1024 * 1024;
let busy = false;                 // one experiment at a time, page-wide
const runButtons = new Set();

export const STOP_MESSAGES = {
    cancelled: () => 'Cancelled',
    'device-lost': () => 'Stopped: the GPU device was lost',
    budget: (budgetMs) => `Stopped at the first step boundary after the ${(budgetMs / 1000).toFixed(0)} s time budget`,
};

export class ExperimentStopped extends Error {
    constructor(reason, budgetMs = MAX_BUDGET_MS) {
        super((STOP_MESSAGES[reason] ?? (() => `Stopped: ${reason}`))(budgetMs));
        this.reason = reason;
    }
}

// The record's outcome for a run that ended with `error` (null: completed)
export function outcomeOf(error) {
    if (!error) return { status: 'done' };
    if (error instanceof ExperimentStopped) return { status: error.reason, message: error.message };
    return { status: 'error', message: `Error: ${error.message}` };
}

// Drives one experiment. Pure apart from the ctx it is given, so it is testable with a mock device.
export async function runExperiment(experiment, settings, ctx, onRow) {
    const rows = [];
    for await (const row of experiment.run(ctx, settings)) {
        rows.push(row);
        onRow?.(row, rows);
    }
    return rows;
}

const clampInt = (v, lo, hi, fallback) => (Number.isFinite(v) ? Math.min(Math.max(Math.round(v), lo), hi) : fallback);

// Per-quantity statistics over measurement samples (see "Measurement protocol" above)
export function summarizeSamples(samples) {
    const phaseNames = [...new Set(samples.flatMap(s => Object.keys(s?.phases ?? {})))];
    return {
        n: samples.length,
        wallMs: summarize(samples.map(s => s?.wallMs)),
        gpuMs: summarize(samples.map(s => s?.gpuMs)),
        phases: Object.fromEntries(phaseNames.map(p => [p, summarize(samples.map(s => s?.phases?.[p]))])),
        samples,
    };
}

// The protocol itself: warm-ups discarded, repeats kept, step() between every two calls
export async function measureRepeated(runOnce, { warmup = 1, repeats = 5, step = async () => {} } = {}) {
    const w = clampInt(warmup, 0, MAX_WARMUP, 1);
    const r = clampInt(repeats, 1, MAX_REPEATS, 5);
    const samples = [];
    for (let i = 0; i < w + r; i++) {
        if (i > 0) await step();
        const sample = await runOnce({ warmup: i < w, index: i < w ? i : i - w });
        if (i >= w) samples.push(sample);
    }
    return { warmup: w, repeats: r, ...summarizeSamples(samples) };
}

// ctx.measure plus ctx.measured: what the run's measure calls used, for the result record
function withMeasure(ctx) {
    ctx.measured = { calls: 0, warmup: null, repeats: null };
    ctx.measure = async (runOnce, options = {}) => {
        const result = await measureRepeated(runOnce, { ...options, step: () => ctx.step() });
        ctx.measured = { calls: ctx.measured.calls + 1, warmup: result.warmup, repeats: result.repeats };
        return result;
    };
    return ctx;
}

// A ctx for CI tests of experiments on the software GPU: no wall budget (software rendering is slow),
// no UI yielding, but the same step() contract (waits for submitted GPU work) and the same ctx.measure.
export function makeTestContext(gpuDevice) {
    const device = gpuDevice.device;
    return withMeasure({
        device, gpuDevice, timestamps: gpuDevice.timestamps === true,
        newTimer: (options) => new GpuTimer(gpuDevice, options),
        elapsedMs: () => 0, remainingMs: () => Infinity, check() {},
        async step() { await device.queue.onSubmittedWorkDone(); },
    });
}

// Builds the ctx an experiment receives. `now` and `waitForUI` are injectable for tests.
export function makeContext(gpuDevice, { budgetMs, signal, now = () => performance.now(),
    waitForUI = () => new Promise(resolve => setTimeout(resolve, 0)), deviceLost = isDeviceLost } = {}) {
    const device = gpuDevice.device;
    const start = now();
    const budget = Math.min(budgetMs ?? MAX_BUDGET_MS, MAX_BUDGET_MS);
    const check = () => {
        if (signal?.aborted) throw new ExperimentStopped('cancelled');
        if (deviceLost()) throw new ExperimentStopped('device-lost');
        if (now() - start > budget) throw new ExperimentStopped('budget', budget);
    };
    return withMeasure({
        device,
        gpuDevice,
        timestamps: gpuDevice.timestamps === true,
        budgetMs: budget,
        newTimer: (options) => new GpuTimer(gpuDevice, options),
        elapsedMs: () => now() - start,
        remainingMs: () => budget - (now() - start),
        check,
        // Between steps: wait for queued GPU work, let the page breathe, then re-check the limits
        async step() {
            check();
            await device.queue.onSubmittedWorkDone();
            await waitForUI();
            check();
        },
    });
}

export const INTRO = 'Each experiment runs on your GPU only when you press Run, and compares against a CPU '
    + 'reference in your browser. Every step is a fixed, capped amount of GPU work; the time budget (at most '
    + `${MAX_BUDGET_MS / 1000} s) is checked between steps, so a run stops at the first step boundary after it. `
    + 'Timings are medians over repeated runs after a discarded warm-up, with their spread. Results can be '
    + 'downloaded as JSON, compared with someone else\'s file, and are added to the device report.';

export async function mountMeasure(panel, exampleId) {
    const { default: experiments } = await import(experimentsModule(exampleId));
    const intro = document.createElement('p');
    intro.className = 'hint';
    intro.textContent = INTRO;
    panel.append(intro);
    for (const experiment of experiments) panel.append(experimentCard(exampleId, experiment, panel));
}

let schemaPromise = null;
function loadSchema() {
    schemaPromise ??= fetch(SCHEMA_PATH).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
    }).catch(error => { schemaPromise = null; throw error; });
    return schemaPromise;
}

function button(text, className) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = text;
    if (className) b.className = className;
    return b;
}

function experimentCard(exampleId, experiment, panel) {
    const card = document.createElement('div');
    card.className = 'experiment';
    const title = document.createElement('h4');
    title.textContent = experiment.title;
    const question = document.createElement('p');
    question.className = 'experiment-question';
    question.textContent = experiment.question;
    card.append(title, question);

    const controls = document.createElement('div');
    controls.className = 'controls';
    const values = {};
    for (const setting of experiment.settings ?? []) {
        const label = document.createElement('label');
        label.textContent = setting.label;
        const select = document.createElement('select');
        select.id = `${exampleId}-${experiment.id}-${setting.key}`;
        label.htmlFor = select.id;
        for (const option of setting.options) {
            const o = document.createElement('option');
            o.value = String(option.value);
            o.textContent = option.label;
            select.append(o);
        }
        select.value = String(setting.default ?? setting.options[0].value);
        values[setting.key] = () => setting.options.find(o => String(o.value) === select.value).value;
        controls.append(label, select);
    }
    const run = button('▶ Run', 'primary');
    const cancel = button('Cancel');
    cancel.disabled = true;
    const status = document.createElement('span');
    status.className = 'experiment-status';
    status.setAttribute('role', 'status');
    controls.append(run, cancel, status);
    card.append(controls);

    if (experiment.baseline) {
        const baseline = document.createElement('p');
        baseline.className = 'hint';
        baseline.textContent = `Reference: ${experiment.baseline}`;
        card.append(baseline);
    }
    const charts = document.createElement('div');
    charts.className = 'experiment-charts';
    const table = document.createElement('div');
    table.className = 'experiment-table';
    const protocol = document.createElement('p');
    protocol.className = 'hint experiment-protocol';
    protocol.hidden = true;
    const checks = document.createElement('ul');
    checks.className = 'experiment-checks';
    checks.setAttribute('aria-label', 'Verification');
    checks.hidden = true;

    const recordBar = document.createElement('div');
    recordBar.className = 'experiment-record';
    recordBar.hidden = true;
    const copy = button('Copy CSV');
    const download = button('Download result (JSON)');
    const compareButton = button('Compare with a result file…');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json,application/json';
    fileInput.hidden = true;
    fileInput.setAttribute('aria-label', `Result file to compare with ${experiment.title}`);
    recordBar.append(copy, download, compareButton, fileInput);
    const compare = document.createElement('div');
    compare.className = 'experiment-compare';
    compare.setAttribute('aria-live', 'polite');
    card.append(charts, table, protocol, checks, recordBar, compare);

    runButtons.add(run);
    let rows = [];
    let record = null;
    let controller = null;
    const setIdle = () => {
        busy = false;
        for (const b of runButtons) b.disabled = !getGPUDevice() || isDeviceLost();
        cancel.disabled = true;
    };
    setIdle();
    on('device', setIdle);
    on('device-lost', () => { controller?.abort(); setIdle(); });

    run.addEventListener('click', async () => {
        const gpuDevice = getGPUDevice();
        if (busy || !gpuDevice || isDeviceLost()) return;
        busy = true;
        for (const b of runButtons) b.disabled = true;
        cancel.disabled = false;
        controller = new AbortController();
        // Stop the example's own loop so the measurement isn't sharing the GPU with it
        pauseExplore(panel.parentElement.querySelector('.tab-panel-explore'));
        const settings = Object.fromEntries(Object.entries(values).map(([k, get]) => [k, get()]));
        rows = [];
        record = null;
        recordBar.hidden = true;
        protocol.hidden = true;
        checks.hidden = true;
        compare.replaceChildren();
        status.textContent = 'Running…';
        status.className = 'experiment-status';
        const redraw = () => drawResults(experiment, rows, charts, table);
        let failure = null;
        const ctx = makeContext(gpuDevice, { budgetMs: experiment.budgetMs, signal: controller.signal });
        try {
            await runExperiment(experiment, settings, ctx, (_row, all) => {
                rows = all;
                status.textContent = `Running… ${all.length} step${all.length === 1 ? '' : 's'}, `
                    + `${(ctx.elapsedMs() / 1000).toFixed(1)} s`;
                redraw();
            });
            status.textContent = `Done: ${rows.length} steps in ${(ctx.elapsedMs() / 1000).toFixed(1)} s.`;
        } catch (error) {
            failure = error;
            const { message } = outcomeOf(error);
            status.textContent = rows.length ? `${message} (${rows.length} step${rows.length === 1 ? '' : 's'} completed).` : `${message}.`;
            if (!(error instanceof ExperimentStopped)) {
                status.className = 'experiment-status error';
                console.error(`[Measure] ${exampleId}/${experiment.id}:`, error);
            }
        } finally {
            setIdle();
        }
        redraw();
        if (!rows.length) return;
        record = buildResultRecord({
            exampleId, experiment, settings, rows, outcome: outcomeOf(failure), measured: ctx.measured,
            gpuDevice, source: await sourceInfo(), userAgent: navigator.userAgent, budgetMs: ctx.budgetMs,
        });
        recordExperimentResult(record);
        showProtocol(protocol, record);
        showChecks(checks, record.verification);
        recordBar.hidden = false;
    });
    cancel.addEventListener('click', () => controller?.abort());
    copy.addEventListener('click', async () => {
        const csv = toCSV(experiment.columns, rows);
        try {
            await navigator.clipboard.writeText(csv);
            copy.textContent = 'Copied';
        } catch {
            copy.textContent = 'Copy failed';
        }
        setTimeout(() => { copy.textContent = 'Copy CSV'; }, 2000);
    });
    download.addEventListener('click', () => {
        if (!record) return;
        loadSchema().then(schema => {
            const { valid, errors } = validateResultRecord(record, schema);
            if (!valid) console.warn('[Measure] result record does not match the schema:', errors);
        }).catch(() => {});
        const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2) + '\n'], { type: 'application/json' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = resultFileName(record);
        document.body.append(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    compareButton.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files?.[0];
        fileInput.value = '';
        if (!file || !record) return;
        await showComparison(compare, record, file);
    });
    return card;
}

function showProtocol(el, record) {
    const p = record.protocol;
    const timing = p.timing === 'timestamp-query'
        ? 'GPU times from timestamp queries plus wall-clock times'
        : 'wall-clock times only (this device has no timestamp-query)';
    el.textContent = p.statistic === 'median'
        ? `Timed columns: median of ${p.repeats} runs after ${p.warmup} discarded warm-up${p.warmup === 1 ? '' : 's'}`
            + `${p.spread === 'iqr' ? `; parentheses and chart bars show the ${SPREAD_LABEL}` : ''}. ${timing}. `
            + 'One run on one device is not a universal number: compare spreads, not single values.'
        : `Timing: ${timing}; values as computed by the experiment.`;
    el.hidden = false;
}

const STATUS_TEXT = { pass: 'PASS', fail: 'FAIL', 'not-checked': 'NOT CHECKED' };

function showChecks(list, verification) {
    list.replaceChildren();
    for (const c of verification) {
        const li = document.createElement('li');
        const s = document.createElement('span');
        s.className = 'check-status';
        s.textContent = STATUS_TEXT[c.status];
        const value = c.value === null ? '' : ` = ${formatCell({}, c.value)}`;
        const detail = [c.tolerance && `required ${c.tolerance}`, c.reference].filter(Boolean).join('; ');
        li.append(s, `${c.metric}${value}${detail ? ` (${detail})` : ''}`);
        list.append(li);
    }
    list.hidden = false;
}

function cellsRow(tr, values, header = false) {
    for (const v of values) {
        const cell = document.createElement(header ? 'th' : 'td');
        cell.textContent = typeof v === 'number' ? formatCell({}, v) : (v ?? '—');
        if (typeof v === 'number') cell.className = 'num';
        tr.append(cell);
    }
}

async function showComparison(holder, mine, file) {
    holder.replaceChildren();
    const fail = (text) => {
        const p = document.createElement('p');
        p.className = 'error';
        p.textContent = text;
        holder.append(p);
    };
    if (file.size > MAX_IMPORT_BYTES) return fail(`${file.name} is larger than ${MAX_IMPORT_BYTES / 1024 / 1024} MB; not a result file.`);
    let theirs, schema;
    try {
        theirs = JSON.parse(await file.text());
    } catch (error) {
        return fail(`${file.name} is not valid JSON (${error.message}).`);
    }
    try {
        schema = await loadSchema();
    } catch (error) {
        return fail(`Could not load the result schema (${SCHEMA_PATH}): ${error.message}.`);
    }
    const { valid, errors } = validateResultRecord(theirs, schema);
    if (!valid) {
        fail(`${file.name} is not a result record (schema v1): ${errors.slice(0, 5).join('; ')}${errors.length > 5 ? ` and ${errors.length - 5} more` : ''}.`);
        return;
    }
    const comparison = compareRecords(mine, theirs);
    const heading = document.createElement('p');
    heading.textContent = `Compared with ${file.name}:`;
    const checks = document.createElement('table');
    cellsRow(checks.createTHead().insertRow(), ['', 'This run', 'File', 'Same?'], true);
    const body = checks.createTBody();
    for (const c of comparison.checks) cellsRow(body.insertRow(), [c.label, c.mine, c.theirs, c.same ? 'yes' : 'no']);
    holder.append(heading, checks);

    if (!comparison.rows.length || !comparison.quantities.length) {
        const p = document.createElement('p');
        p.className = 'hint';
        p.textContent = 'The two results have no sweep points (or numeric columns) in common.';
        holder.append(p);
        return;
    }
    const controls = document.createElement('div');
    controls.className = 'controls';
    const label = document.createElement('label');
    label.textContent = 'Quantity';
    const select = document.createElement('select');
    select.id = `compare-${mine.experiment.exampleId}-${mine.experiment.id}`;
    label.htmlFor = select.id;
    for (const q of comparison.quantities) {
        const o = document.createElement('option');
        o.value = q.key;
        o.textContent = q.label;
        select.append(o);
    }
    controls.append(label, select);
    const values = document.createElement('table');
    const draw = () => {
        values.replaceChildren();
        const q = comparison.quantities.find(c => c.key === select.value);
        cellsRow(values.createTHead().insertRow(), [comparison.key.label, `${q.label}: this run`, 'file', 'file / this run'], true);
        const tb = values.createTBody();
        for (const r of ratioTable(comparison, q.key)) {
            cellsRow(tb.insertRow(), [String(r.x), r.mine, r.theirs, r.ratio === null ? '—' : `${formatCell({}, r.ratio)}×`]);
        }
    };
    select.addEventListener('change', draw);
    draw();
    const note = document.createElement('p');
    note.className = 'hint';
    note.textContent = 'Values are the medians each run recorded, matched on the first column. A ratio is only '
        + 'meaningful when settings and protocol match; a different adapter, browser or commit is what you are comparing.';
    holder.append(controls, values, note);
}

function drawResults(experiment, rows, charts, table) {
    charts.replaceChildren();
    for (const chart of experiment.charts ?? []) {
        const holder = document.createElement('figure');
        holder.className = 'chart';
        charts.append(holder);
        renderLineChart(holder, { ...chart, rows, columns: experiment.columns });
    }
    table.replaceChildren(resultsTable(experiment.columns, rows));
}

export function formatCell(column, value) {
    if (value === null || value === undefined || (typeof value === 'number' && !Number.isFinite(value))) return '—';
    if (column.format) return column.format(value);
    if (typeof value === 'number') {
        if (Number.isInteger(value)) return value.toLocaleString('en-US');
        const abs = Math.abs(value);
        // three significant digits without trailing zeros (0.5 stays "0.5", 0.12345 becomes "0.123")
        return abs !== 0 && (abs < 1e-3 || abs >= 1e5) ? value.toExponential(2) : String(+value.toPrecision(3));
    }
    return String(value);
}

// A results-table cell: "median (q1–q3)" when the row has a spread for this column
export function formatRowCell(column, row) {
    return formatWithRange((v) => formatCell(column, v), row, column.key);
}

function resultsTable(columns, rows) {
    const t = document.createElement('table');
    const head = t.createTHead().insertRow();
    for (const c of columns) {
        const th = document.createElement('th');
        th.textContent = hasRange(rows, c.key) ? `${c.label} (q1–q3)` : c.label;
        head.append(th);
    }
    const body = t.createTBody();
    for (const row of rows) {
        const tr = body.insertRow();
        for (const c of columns) tr.insertCell().textContent = formatRowCell(c, row);
    }
    return t;
}

// Columns with a spread get two more CSV columns, "<label> q1" and "<label> q3"
export function toCSV(columns, rows) {
    const esc = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
    const fields = columns.flatMap(c => (hasRange(rows, c.key)
        ? [[c.key, c.label], [`${c.key}Lo`, `${c.label} q1`], [`${c.key}Hi`, `${c.label} q3`]]
        : [[c.key, c.label]]));
    const value = (v) => (v === undefined || v === null ? '' : String(v));
    const lines = [fields.map(([, label]) => esc(label)).join(',')];
    for (const row of rows) lines.push(fields.map(([key]) => esc(value(row[key]))).join(','));
    return lines.join('\n') + '\n';
}
