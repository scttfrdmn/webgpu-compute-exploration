// Measure tab: user-started, bounded experiments that run on the visitor's GPU and compare against CPU
// references in the browser.
//
// An experiment module (js/teach/experiments/<id>.js) default-exports an array of experiments:
//   {
//     id, title, question,                       // question: one sentence the experiment answers
//     settings: [{ key, label, options: [{ value, label }], default }],   // optional fixed choices
//     budgetMs,                                  // hard wall-clock cap for the whole run (≤ MAX_BUDGET_MS, 20 s)
//     columns: [{ key, label, format? }],        // results table
//     charts: [{ x, xLabel, logX?, y: [keys], yLabel, logY?, labels?: {key: label},
//                reference?: { label, points } | (rows) => ({ label, points }),   // e.g. a trend through your data
//                emptyNote? }],                // shown when no series has a value (e.g. GPU ms without timestamps)
//     baseline,                                  // optional text: what CI measured, for comparison
//     async *run(ctx, settings) { ... yield row ... }   // one row per sweep step
//   }
// Rules every experiment follows: problem sizes come from fixed lists with hard caps; each step is
// bounded GPU work; call `await ctx.step()` between steps (it waits for the GPU, yields to the page and
// throws if the run was cancelled, the budget is spent, or the device was lost).
// Timing: when GPU work goes through an example's compute-only export (e.g. measureForces in nbody.js),
// that function owns its GpuTimer and returns gpuMs/wallMs; experiments that dispatch directly use
// ctx.newTimer(). Columns like θ or dt that should print as written use `format: String`.

import { getGPUDevice, isDeviceLost, on, recordExperimentResult } from './app-state.js';
import { GpuTimer } from '../gpu-timer.js';
import { renderLineChart } from './chart.js';
import { experimentsModule } from './registry.js';
import { pauseExplore } from './tabs.js';

export const MAX_BUDGET_MS = 20000;
let busy = false;                 // one experiment at a time, page-wide
const runButtons = new Set();

export class ExperimentStopped extends Error {}

// Drives one experiment. Pure apart from the ctx it is given, so it is testable with a mock device.
export async function runExperiment(experiment, settings, ctx, onRow) {
    const rows = [];
    for await (const row of experiment.run(ctx, settings)) {
        rows.push(row);
        onRow?.(row, rows);
    }
    return rows;
}

// A ctx for CI tests of experiments on the software GPU: no wall budget (software rendering is slow),
// no UI yielding, but the same step() contract (waits for submitted GPU work).
export function makeTestContext(gpuDevice) {
    const device = gpuDevice.device;
    return {
        device, gpuDevice, timestamps: gpuDevice.timestamps === true,
        newTimer: (options) => new GpuTimer(gpuDevice, options),
        elapsedMs: () => 0, remainingMs: () => Infinity, check() {},
        async step() { await device.queue.onSubmittedWorkDone(); },
    };
}

// Builds the ctx an experiment receives. `now` and `waitForUI` are injectable for tests.
export function makeContext(gpuDevice, { budgetMs, signal, now = () => performance.now(),
    waitForUI = () => new Promise(resolve => setTimeout(resolve, 0)), deviceLost = isDeviceLost } = {}) {
    const device = gpuDevice.device;
    const start = now();
    const budget = Math.min(budgetMs ?? MAX_BUDGET_MS, MAX_BUDGET_MS);
    const check = () => {
        if (signal?.aborted) throw new ExperimentStopped('Cancelled');
        if (deviceLost()) throw new ExperimentStopped('GPU device lost');
        if (now() - start > budget) throw new ExperimentStopped(`Stopped at the ${(budget / 1000).toFixed(0)} s time budget`);
    };
    return {
        device,
        gpuDevice,
        timestamps: gpuDevice.timestamps === true,
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
    };
}

export async function mountMeasure(panel, exampleId) {
    const { default: experiments } = await import(experimentsModule(exampleId));
    const intro = document.createElement('p');
    intro.className = 'hint';
    intro.textContent = 'Each experiment runs on your GPU only when you press Run, within a time budget, and '
        + 'compares against a CPU reference in your browser. Results are added to the device report.';
    panel.append(intro);
    for (const experiment of experiments) panel.append(experimentCard(exampleId, experiment, panel));
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
    const run = document.createElement('button');
    run.className = 'primary';
    run.textContent = '▶ Run';
    const cancel = document.createElement('button');
    cancel.textContent = 'Cancel';
    cancel.disabled = true;
    const status = document.createElement('span');
    status.className = 'experiment-status';
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
    const copy = document.createElement('button');
    copy.textContent = 'Copy CSV';
    copy.hidden = true;
    card.append(charts, table, copy);

    runButtons.add(run);
    let rows = [];
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
        status.textContent = 'Running…';
        status.className = 'experiment-status';
        const redraw = () => drawResults(experiment, rows, charts, table);
        let outcome = 'done';
        try {
            const ctx = makeContext(gpuDevice, { budgetMs: experiment.budgetMs, signal: controller.signal });
            await runExperiment(experiment, settings, ctx, (_row, all) => {
                rows = all;
                status.textContent = `Running… ${all.length} step${all.length === 1 ? '' : 's'}, `
                    + `${(ctx.elapsedMs() / 1000).toFixed(1)} s`;
                redraw();
            });
            status.textContent = `Done: ${rows.length} steps.`;
        } catch (error) {
            outcome = error instanceof ExperimentStopped ? error.message : `Error: ${error.message}`;
            status.textContent = rows.length ? `${outcome} after ${rows.length} steps.` : outcome;
            if (!(error instanceof ExperimentStopped)) {
                status.className = 'experiment-status error';
                console.error(`[Measure] ${exampleId}/${experiment.id}:`, error);
            }
        } finally {
            setIdle();
        }
        redraw();
        copy.hidden = rows.length === 0;
        if (rows.length) {
            recordExperimentResult({ exampleId, experimentId: experiment.id, title: experiment.title,
                settings, rows, outcome, columns: experiment.columns, finishedAt: new Date().toISOString() });
        }
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
    return card;
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

function resultsTable(columns, rows) {
    const t = document.createElement('table');
    const head = t.createTHead().insertRow();
    for (const c of columns) {
        const th = document.createElement('th');
        th.textContent = c.label;
        head.append(th);
    }
    const body = t.createTBody();
    for (const row of rows) {
        const tr = body.insertRow();
        for (const c of columns) tr.insertCell().textContent = formatCell(c, row[c.key]);
    }
    return t;
}

export function toCSV(columns, rows) {
    const esc = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
    const lines = [columns.map(c => esc(c.label)).join(',')];
    for (const row of rows) lines.push(columns.map(c => esc(row[c.key] === undefined ? '' : String(row[c.key]))).join(','));
    return lines.join('\n') + '\n';
}
