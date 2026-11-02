// validate.html: runs the physical-GPU validation protocol (#78, VALIDATION_PLAN in js/teach/validation.js) on
// one click and offers the validation bundle for download. The experiments run exactly as on the Measure tab:
// the same experiment objects, makeContext (budget, step(), cancel, device loss) and runExperiment from
// js/teach/measure.js, and buildResultRecord from js/teach/result-record.js — one at a time, never in parallel.
// runPlan is pure apart from the ctx factory and experiments it is given, so test/validation.test.mjs drives
// it with a mock device. Nothing touches the DOM or the GPU at import time.

import { getGPUDevice } from '../webgpu-utils.js';
import { setGPUDevice, markDeviceLost, isDeviceLost } from './app-state.js';
import { makeContext, runExperiment, outcomeOf, ExperimentStopped, formatCell } from './measure.js';
import { buildResultRecord, SCHEMA_PATH } from './result-record.js';
import { sourceInfo, LOCAL_COMMIT } from '../source-info.js';
import {
    planFor, experimentOf, entryBudgetMs, planDuration, settingsText, detectBrowser, matchEnvironment,
    validationEnvironment, missingFields, buildValidationBundle, validateValidationBundle, bundleFileName, bundlePath,
    verificationVerdict, interpretationCounts, VERDICT_TEXT, agreesText, MATRIX_PATH, BUNDLE_SCHEMA_PATH,
} from './validation.js';
import matrixMultiplication from './experiments/matrix-multiplication.js';
import nbody from './experiments/nbody.js';
import parallelSort from './experiments/parallel-sort.js';

export const EXPERIMENTS = Object.freeze({
    'matrix-multiplication': matrixMultiplication,
    nbody,
    'parallel-sort': parallelSort,
});

// Runs every plan entry in order. Returns [{ entry, status, record, seconds }] where status is the record's
// outcome status, or 'not-run' for the entries after a cancel or a device loss (a budget stop or an error
// only ends that entry). Every entry that started gets a record, even one with no rows, so a failure is
// part of the bundle.
//   options: { gpuDevice, signal, source, userAgent, experiments, newContext(gpuDevice, { budgetMs, signal }),
//              onProgress({ index, entry, rows, elapsedMs }), onEntryDone(result, index), now }
export async function runPlan(plan, { gpuDevice, signal, source, userAgent = '', experiments = EXPERIMENTS,
    newContext = makeContext, onProgress, onEntryDone, now = () => performance.now() } = {}) {
    const results = [];
    let halted = null;
    for (const [index, entry] of plan.entries.entries()) {
        if (!halted && signal?.aborted) halted = 'cancelled';
        if (!halted && isDeviceLost()) halted = 'device-lost';
        if (halted) {
            results.push({ entry, status: 'not-run', record: null, seconds: 0 });
            continue;
        }
        const experiment = experimentOf(entry, experiments);
        const settings = { ...entry.settings };
        const ctx = newContext(gpuDevice, { budgetMs: entryBudgetMs(entry, experiment), signal });
        const t0 = now();
        let rows = [], failure = null;
        try {
            await gpuDevice.device.queue.onSubmittedWorkDone();   // nothing left over from the previous entry
            rows = await runExperiment(experiment, settings, ctx, (_row, all) => {
                rows = all;
                onProgress?.({ index, entry, rows: all, elapsedMs: now() - t0 });
            });
        } catch (error) {
            failure = error;
            if (!(error instanceof ExperimentStopped)) console.error(`[Validate] ${entry.exampleId}/${entry.experimentId}:`, error);
        }
        const record = buildResultRecord({
            exampleId: entry.exampleId, experiment, settings, rows, outcome: outcomeOf(failure), measured: ctx.measured,
            gpuDevice, source, userAgent, budgetMs: ctx.budgetMs,
        });
        const result = { entry, status: record.outcome.status, record, seconds: (now() - t0) / 1000 };
        results.push(result);
        onEntryDone?.(result, index);
        if (record.outcome.status === 'cancelled' || record.outcome.status === 'device-lost') halted = record.outcome.status;
    }
    return results;
}

// ---- Page ------------------------------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const sec = (s) => (s < 10 ? s.toFixed(1) : s.toFixed(0));

async function fetchJSON(path) {
    const r = await fetch(path, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
}

function cells(tr, values, header = false) {
    for (const v of values) {
        const cell = document.createElement(header ? 'th' : 'td');
        if (v instanceof Node) cell.append(v);
        else cell.textContent = v ?? '—';
        tr.append(cell);
    }
    return tr;
}

function detectedText(detected, gpuDevice, source) {
    const info = gpuDevice.adapter.info ?? {};
    const adapter = ['vendor', 'architecture', 'device', 'description'].map(k => info[k]).filter(Boolean).join(' / ') || '(adapter info not reported)';
    const tsAdapter = gpuDevice.adapter.features.has('timestamp-query');
    return [
        `Browser: ${detected.browser.name} ${detected.browser.version ?? '(version unknown)'}`,
        `OS: ${detected.os.name} ${detected.os.version ?? '(version not exposed by this browser: enter it below)'}`
            + (detected.architecture ? ` · ${detected.architecture}` : ''),
        `Adapter: ${adapter}${info.isFallbackAdapter ? ' · FALLBACK (software) adapter' : ''}`,
        `timestamp-query: ${tsAdapter ? (gpuDevice.timestamps ? 'available (GPU times recorded)' : 'offered but not enabled')
            : 'not available — GPU times fall back to wall-clock'}`,
        `Site: ${source.version} @ ${source.commit === LOCAL_COMMIT ? 'local (no version.json: run `node tools/local-version.mjs` before serving, so the bundle names its commit)' : source.commit.slice(0, 12)}`,
    ];
}

function planTable(table, plan) {
    table.replaceChildren();
    cells(table.createTHead().insertRow(), ['#', 'Experiment', 'Settings', 'Budget', 'Expected', 'Why'], true);
    const body = table.createTBody();
    plan.entries.forEach((entry, i) => {
        const experiment = experimentOf(entry, EXPERIMENTS);
        cells(body.insertRow(), [String(i + 1), `${entry.exampleId} / ${experiment.title}`, settingsText(entry.settings),
            `${entryBudgetMs(entry, experiment) / 1000} s`, `~${entry.expectedSeconds} s`, entry.why ?? '']);
    });
}

function resultRow(tr, i, result) {
    tr.replaceChildren();
    const { entry, record, status } = result;
    if (!record) return cells(tr, [String(i + 1), `${entry.exampleId} / ${entry.experimentId}`, 'not run', '—', '—', '—', '—']);
    const v = verificationVerdict(record);
    const k = interpretationCounts(record);
    const outcome = `${status}${record.outcome.message && status !== 'done' ? ` — ${record.outcome.message}` : ''} (${record.outcome.steps} steps)`;
    const verdict = document.createElement('strong');
    verdict.textContent = VERDICT_TEXT[v.verdict];
    verdict.className = v.verdict === 'fail' ? 'error' : v.verdict === 'pass' ? 'success' : '';
    const verification = document.createElement('span');
    verification.append(verdict, ` ${v.pass} pass, ${v.fail} fail, ${v.notChecked} not checked`);
    const interp = record.interpretation?.length ? `${k.agrees} agree, ${k.disagrees} disagree, ${k.undetermined} undetermined` : '—';
    return cells(tr, [String(i + 1), `${entry.exampleId} / ${record.experiment.title} (${settingsText(record.settings)})`, outcome,
        verification, record.protocol.timing, interp, `${sec(result.seconds)} s`]);
}

function details(result) {
    const box = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `${result.entry.exampleId} / ${result.record.experiment.title} (${settingsText(result.record.settings)}): checks and claims`;
    const checks = document.createElement('ul');
    checks.className = 'experiment-checks';
    for (const c of result.record.verification) {
        const li = document.createElement('li');
        const value = c.value === null ? '' : ` = ${formatCell({}, c.value)}`;
        li.textContent = `${VERDICT_TEXT[c.status]} ${c.metric}${value}${c.tolerance ? ` (required ${c.tolerance})` : ''}`;
        checks.append(li);
    }
    box.append(summary, checks);
    if (result.record.interpretation?.length) {
        const claims = document.createElement('ul');
        claims.className = 'experiment-checks';
        for (const c of result.record.interpretation) {
            const li = document.createElement('li');
            li.textContent = `${agreesText(c.agrees)}: ${c.claim} — predicted ${c.predicted}; measured ${c.measured}`;
            claims.append(li);
        }
        box.append(claims);
    }
    return box;
}

function download(bundle) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2) + '\n'], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = bundleFileName(bundle);
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function init() {
    const status = $('gpu-status');
    const plan = planFor(location.search);
    let matrix = null;
    try {
        matrix = await fetchJSON(MATRIX_PATH);
    } catch (error) {
        console.warn(`[Validate] could not load ${MATRIX_PATH}:`, error);
    }
    const envSelect = $('validation-env');
    for (const env of [...(matrix?.environments ?? []), { id: 'unlisted', label: 'Not in the matrix' }]) {
        const o = document.createElement('option');
        o.value = env.id;
        o.textContent = `${env.label} (${env.id})`;
        envSelect.append(o);
    }
    planTable($('validation-plan'), plan);
    const { expectedSeconds, worstCaseSeconds } = planDuration(plan, EXPERIMENTS);
    $('validation-plan-summary').textContent = `${plan.smoke ? 'SMOKE PLAN (validate.html?smoke=1: reduced settings for CI; '
        + 'its bundle is not part of the validation matrix). ' : ''}Validation plan v${plan.version}: ${plan.entries.length} experiments, run one `
        + `after another. Expected about ${sec(expectedSeconds)} s on an Apple-silicon Mac; at most about ${sec(worstCaseSeconds)} s `
        + '(the sum of the time budgets: a slow experiment stops at its budget and the rest still run).';

    if (!navigator.gpu) {
        status.className = 'status not-supported';
        status.textContent = 'WebGPU not available in this browser — use Chrome 113+ or Safari 26+';
        return;
    }
    let gpuDevice;
    let controller = null;
    try {
        gpuDevice = await getGPUDevice((info) => {
            markDeviceLost();
            controller?.abort();
            status.className = 'status not-supported';
            status.textContent = info.message ? `GPU device lost: ${info.message}` : 'GPU device lost';
            $('validation-run').disabled = true;
        });
    } catch (error) {
        status.className = 'status not-supported';
        status.textContent = `WebGPU failed to initialize: ${error.message}`;
        return;
    }
    setGPUDevice(gpuDevice);
    const source = await sourceInfo();
    let hints = null;
    try {
        hints = await navigator.userAgentData?.getHighEntropyValues?.(['platformVersion', 'fullVersionList', 'architecture']) ?? null;
    } catch { hints = null; }
    const detected = detectBrowser(navigator.userAgent, hints);
    const info = gpuDevice.adapter.info ?? {};
    status.className = 'status supported';
    status.textContent = `WebGPU ready · ${[info.vendor, info.architecture].filter(Boolean).join(' ') || info.description || 'adapter'} · `
        + (gpuDevice.timestamps ? 'timestamp-query available' : 'no timestamp-query (wall-clock times only)');

    const detectedBox = $('validation-detected');
    detectedBox.replaceChildren(...detectedText(detected, gpuDevice, source).map(line => {
        const p = document.createElement('div');
        p.textContent = line;
        return p;
    }));
    detectedBox.hidden = false;
    const suggested = matchEnvironment(matrix, detected);
    envSelect.value = suggested?.id ?? 'unlisted';
    $('validation-browser-version').value = detected.browser.version ?? '';
    $('validation-os-version').value = detected.os.version ?? '';
    const envHint = () => {
        const env = matrix?.environments?.find(e => e.id === envSelect.value);
        $('validation-env-hint').textContent = env ? `${env.hardware}. ${env.notes}` : 'Not a matrix environment: the bundle can still be '
            + 'downloaded, but tools/validation-summary.mjs only reads bundles of listed environments.';
    };
    envSelect.addEventListener('change', envHint);
    envHint();

    const run = $('validation-run'), cancel = $('validation-cancel'), progress = $('validation-progress');
    const results = $('validation-results'), detailsBox = $('validation-details');
    const downloadButton = $('validation-download'), downloadHint = $('validation-download-hint');
    const inputs = ['validation-env', 'validation-browser-version', 'validation-os-version', 'validation-hardware', 'validation-notes'].map($);
    let bundle = null;
    run.disabled = isDeviceLost();

    run.addEventListener('click', async () => {
        if (controller || isDeviceLost()) return;
        controller = new AbortController();
        bundle = null;
        run.disabled = true;
        cancel.disabled = false;
        downloadButton.disabled = true;
        downloadHint.textContent = '';
        inputs.forEach(i => { i.disabled = true; });
        const environment = validationEnvironment({ gpuDevice, detected, userAgent: navigator.userAgent, source, entered: {
            id: envSelect.value, browserVersion: $('validation-browser-version').value, osVersion: $('validation-os-version').value,
            hardware: $('validation-hardware').value, notes: $('validation-notes').value } });
        $('validation-results-section').hidden = false;
        results.replaceChildren();
        detailsBox.replaceChildren();
        cells(results.createTHead().insertRow(), ['#', 'Experiment', 'Outcome', 'Verification', 'Timing', 'Interpretation', 'Time'], true);
        const body = results.createTBody();
        const rows = plan.entries.map((entry, i) => cells(body.insertRow(), [String(i + 1), `${entry.exampleId} / ${entry.experimentId}`, 'waiting', '', '', '', '']));
        progress.hidden = false;
        progress.max = plan.entries.length;
        progress.value = 0;
        const statusEl = $('validation-status');
        statusEl.textContent = 'Running…';
        const all = await runPlan(plan, {
            gpuDevice, signal: controller.signal, source, userAgent: navigator.userAgent,
            onProgress: ({ index, entry, rows: done, elapsedMs }) => {
                statusEl.textContent = `Running ${index + 1}/${plan.entries.length}: ${entry.exampleId} / ${entry.experimentId} — `
                    + `${done.length} step${done.length === 1 ? '' : 's'}, ${sec(elapsedMs / 1000)} s`;
                rows[index].cells[2].textContent = 'running';
            },
            onEntryDone: (result, index) => {
                resultRow(rows[index], index, result);
                detailsBox.append(details(result));
                progress.value = index + 1;
            },
        });
        all.forEach((result, i) => { if (!result.record) resultRow(rows[i], i, result); });
        const records = all.filter(r => r.record).map(r => r.record);
        const ran = records.length, fails = records.filter(r => verificationVerdict(r).verdict === 'fail').length;
        const halted = all.find(r => r.status === 'cancelled' || r.status === 'device-lost')?.status;
        statusEl.textContent = `${halted === 'cancelled' ? 'Cancelled' : halted === 'device-lost' ? 'Stopped: the GPU device was lost' : 'Done'}: `
            + `${ran} of ${plan.entries.length} experiments ran, ${fails} with a failed check.`;
        controller = null;
        cancel.disabled = true;
        run.disabled = isDeviceLost();
        inputs.forEach(i => { i.disabled = false; });
        if (!records.length) return;
        bundle = buildValidationBundle({ plan, environment, records });
        const missing = missingFields(environment, matrix);
        downloadButton.disabled = false;
        downloadHint.textContent = `Save as ${bundlePath(bundle)} in the repository`
            + (missing.length ? `. Not recorded: ${missing.join(', ')} — fill them in and run again, or add them to the file.` : '.');
        try {
            const [bundleSchema, recordSchema] = await Promise.all([fetchJSON(BUNDLE_SCHEMA_PATH), fetchJSON(SCHEMA_PATH)]);
            const { valid, errors } = validateValidationBundle(bundle, bundleSchema, recordSchema);
            if (!valid) {
                console.warn('[Validate] bundle does not match the schema:', errors);
                downloadHint.textContent += ` Warning: the bundle does not match ${BUNDLE_SCHEMA_PATH} (${errors.slice(0, 3).join('; ')}).`;
            }
        } catch (error) {
            console.warn('[Validate] could not load the schemas:', error);
        }
    });
    cancel.addEventListener('click', () => controller?.abort());
    downloadButton.addEventListener('click', () => { if (bundle) download(bundle); });
}

if (typeof document !== 'undefined' && document.getElementById('validation-app')) {
    init().catch((error) => {
        console.error('[Validate] page failed to initialize:', error);
        const status = document.getElementById('gpu-status');
        if (status) { status.className = 'status not-supported'; status.textContent = `Validation page failed: ${error.message}`; }
    });
}
