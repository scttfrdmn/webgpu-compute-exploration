// Physical-GPU validation (#78): the fixed validation protocol that validate.html runs, the environment it
// records, and the validation bundle it downloads. Pure (no DOM, no GPU, no node: imports), so the page,
// tools/validation-summary.mjs and test/validation.test.mjs share it.
//
// The protocol is VALIDATION_PLAN: the three flagship investigations (matrix multiplication, N-body,
// parallel sorting), each experiment with fixed settings taken from its own option lists (so every cap
// still holds), run one at a time through the Measure tab's runner (js/teach/measure.js: makeContext,
// runExperiment, budget and step(), device-loss handling). Each run becomes a schema v1 result record; the
// records, the plan version and the environment form one bundle (schema/validation-bundle.v1.json):
//   { bundleVersion: 1, plan: VALIDATION_PLAN.version, smoke, generatedAt, environment, records: [...] }
// Bundles are committed as data/validation/<environment>/<date>-<commit>.json (data/validation/matrix.json
// lists the environments) and summarized into the README by tools/validation-summary.mjs.
//
// Change the plan's entries or settings → bump VALIDATION_PLAN.version, so bundles from different plans are
// never read as the same protocol.

import { validateResultRecord } from './result-record.js';

export const BUNDLE_VERSION = 1;
export const BUNDLE_SCHEMA_PATH = 'schema/validation-bundle.v1.json';
export const MATRIX_PATH = 'data/validation/matrix.json';
export const RECORD_SCHEMA_REF = 'experiment-result.v1.json';
export const FLAGSHIPS = ['matrix-multiplication', 'nbody', 'parallel-sort'];

// expectedSeconds: a rough estimate of one run on an Apple-silicon Mac (M1 or later), shown before running.
// The hard bound is the experiment's budget (20 s each), checked between steps.
export const VALIDATION_PLAN = Object.freeze({
    version: 1,
    entries: Object.freeze([
        { key: 'matmul', exampleId: 'matrix-multiplication', experimentId: 'cpu-naive-tiled',
            settings: { maxN: 1024 }, expectedSeconds: 5,
            why: 'all five sizes to the GPU cap; CPU baseline to n = 512' },
        { key: 'nbody-theta', exampleId: 'nbody', experimentId: 'theta-error',
            settings: { n: 16384, preset: 'plummer' }, expectedSeconds: 4,
            why: 'six opening angles against the float64 direct sum and the CPU replica medians' },
        { key: 'nbody-scaling', exampleId: 'nbody', experimentId: 'scaling',
            settings: { preset: 'plummer' }, expectedSeconds: 7,
            why: 'Barnes-Hut to N = 65,536, brute force to 16,384: the crossover' },
        { key: 'nbody-drift-brute', exampleId: 'nbody', experimentId: 'energy-drift',
            settings: { n: 2048, mode: 'brute', preset: 'plummer' }, expectedSeconds: 4,
            why: 'second-order energy error with exact forces' },
        { key: 'nbody-drift-bh', exampleId: 'nbody', experimentId: 'energy-drift',
            settings: { n: 2048, mode: 'bh', preset: 'plummer' }, expectedSeconds: 5,
            why: 'the Barnes-Hut energy floor' },
        { key: 'sort-u32', exampleId: 'parallel-sort', experimentId: 'scaling',
            settings: { dataType: 'u32', maxN: 1048576 }, expectedSeconds: 6,
            why: 'seven sizes to 1,048,576 u32 keys, every sort checked bit for bit' },
        { key: 'sort-f32', exampleId: 'parallel-sort', experimentId: 'scaling',
            settings: { dataType: 'f32', maxN: 1048576 }, expectedSeconds: 7,
            why: 'the same with f32 keys (encode → sort → decode) and the key-map checks' },
    ].map(e => Object.freeze({ ...e, settings: Object.freeze(e.settings) }))),
});

// validate.html?smoke=1: the smallest option of every setting and a short budget, for the CI browser test on
// a software GPU. Its bundles say smoke: true and are rejected by tools/validation-summary.mjs.
export const SMOKE_BUDGET_MS = 4000;
export const SMOKE_PLAN = Object.freeze({
    version: VALIDATION_PLAN.version,
    smoke: true,
    entries: Object.freeze([
        { key: 'matmul', exampleId: 'matrix-multiplication', experimentId: 'cpu-naive-tiled', settings: { maxN: 256 }, expectedSeconds: 2 },
        { key: 'nbody-theta', exampleId: 'nbody', experimentId: 'theta-error', settings: { n: 4096, preset: 'plummer' }, expectedSeconds: 2 },
        { key: 'nbody-drift-brute', exampleId: 'nbody', experimentId: 'energy-drift', settings: { n: 2048, mode: 'brute', preset: 'plummer' }, expectedSeconds: 2 },
        { key: 'sort-u32', exampleId: 'parallel-sort', experimentId: 'scaling', settings: { dataType: 'u32', maxN: 262144 }, expectedSeconds: 2 },
    ].map(e => Object.freeze({ ...e, budgetMs: SMOKE_BUDGET_MS, settings: Object.freeze(e.settings) }))),
});

export const planFor = (search = '') => (new URLSearchParams(search).get('smoke') === '1' ? SMOKE_PLAN : VALIDATION_PLAN);

// The experiment object of a plan entry from { exampleId: [experiments] }; throws when it does not exist
export function experimentOf(entry, experimentsByExample) {
    const experiment = experimentsByExample[entry.exampleId]?.find(e => e.id === entry.experimentId);
    if (!experiment) throw new Error(`plan entry ${entry.key}: no experiment ${entry.exampleId}/${entry.experimentId}`);
    return experiment;
}

// Budget a plan entry runs with: its own (smoke) budget, never above the experiment's
export const entryBudgetMs = (entry, experiment) => Math.min(entry.budgetMs ?? experiment.budgetMs, experiment.budgetMs);

// { expectedSeconds, worstCaseSeconds } for a plan; worst case = sum of budgets (a run stops at the first step
// boundary after its budget, so allow a little more)
export function planDuration(plan, experimentsByExample) {
    let expected = 0, budget = 0;
    for (const entry of plan.entries) {
        expected += entry.expectedSeconds;
        budget += entryBudgetMs(entry, experimentOf(entry, experimentsByExample)) / 1000;
    }
    return { expectedSeconds: expected, worstCaseSeconds: budget };
}

export const settingsText = (settings) => Object.entries(settings ?? {}).map(([k, v]) => `${k}=${v}`).join(', ');

// ---- Environment -------------------------------------------------------------------------------------------

// Browser and OS from the user agent and, where the browser has it (Chromium), User-Agent Client Hints
// (navigator.userAgentData.getHighEntropyValues(['platformVersion', 'fullVersionList', 'architecture'])).
// Safari and Firefox freeze the macOS version in the UA at 10_15_7, so the OS version is only known from
// client hints; otherwise it stays null and the page asks for it.
export function detectBrowser(userAgent = '', hints = null) {
    const ua = String(userAgent);
    const brand = (names) => hints?.fullVersionList?.find(b => names.includes(b.brand))?.version ?? null;
    let name = 'unknown', version = null;
    let m;
    if ((m = ua.match(/Edg\/([\d.]+)/))) { name = 'edge'; version = brand(['Microsoft Edge']) ?? m[1]; }
    else if ((m = ua.match(/Firefox\/([\d.]+)/))) { name = 'firefox'; version = m[1]; }
    else if ((m = ua.match(/(?:Chrome|CriOS)\/([\d.]+)/))) { name = 'chrome'; version = brand(['Google Chrome', 'Chromium']) ?? m[1]; }
    else if (/Safari\//.test(ua) && (m = ua.match(/Version\/([\d.]+)/))) { name = 'safari'; version = m[1]; }

    let os = 'unknown';
    if (/iPhone|iPad|iPod/.test(ua)) os = 'ios';
    else if (/Mac OS X|Macintosh/.test(ua)) os = 'macos';
    else if (/Windows/.test(ua)) os = 'windows';
    else if (/Android/.test(ua)) os = 'android';
    else if (/CrOS/.test(ua)) os = 'chromeos';
    else if (/Linux/.test(ua)) os = 'linux';
    const hinted = typeof hints?.platformVersion === 'string' && hints.platformVersion ? hints.platformVersion : null;
    return {
        browser: { name, version, source: version ? (brand(['Google Chrome', 'Chromium', 'Microsoft Edge']) ? 'client-hints' : 'user-agent') : 'unknown' },
        os: { name: os, version: hinted, source: hinted ? 'client-hints' : 'unknown' },
        architecture: typeof hints?.architecture === 'string' && hints.architecture ? hints.architecture : null,
    };
}

const major = (version) => Number.parseInt(String(version ?? ''), 10);

// The matrix environment (data/validation/matrix.json) a detected browser/OS belongs to, or null
export function matchEnvironment(matrix, detected) {
    return matrix?.environments?.find(env => env.browser === detected.browser.name && env.os === detected.os.name
        && (!env.minBrowserMajor || major(detected.browser.version) >= env.minBrowserMajor)) ?? null;
}

const text = (v) => (v === undefined || v === null ? '' : String(v));
const ADAPTER_FIELDS = ['vendor', 'architecture', 'device', 'description'];

// The bundle's environment. gpuDevice: { adapter, device, timestamps } from getGPUDevice; detected: from
// detectBrowser; entered: what the person running it typed { id, browserVersion, osVersion, hardware, notes }
export function validationEnvironment({ gpuDevice = null, detected, entered = {}, userAgent = '', source = null }) {
    const safe = (fn, fallback) => { try { return fn() ?? fallback; } catch { return fallback; } };
    const info = safe(() => gpuDevice.adapter.info, {});
    const fallback = safe(() => info.isFallbackAdapter ?? gpuDevice.adapter.isFallbackAdapter, null);
    const browserVersion = text(entered.browserVersion).trim() || detected.browser.version;
    const osVersion = text(entered.osVersion).trim() || detected.os.version;
    return {
        id: text(entered.id).trim() || 'unlisted',
        browser: { name: detected.browser.name, version: browserVersion || null,
            source: text(entered.browserVersion).trim() && entered.browserVersion !== detected.browser.version ? 'entered' : detected.browser.source },
        os: { name: detected.os.name, version: osVersion || null,
            source: text(entered.osVersion).trim() && entered.osVersion !== detected.os.version ? 'entered' : detected.os.source },
        architecture: detected.architecture ?? null,
        hardware: text(entered.hardware).trim() || null,
        notes: text(entered.notes).trim() || null,
        adapter: {
            ...Object.fromEntries(ADAPTER_FIELDS.map(k => [k, text(safe(() => info[k], ''))])),
            isFallbackAdapter: typeof fallback === 'boolean' ? fallback : null,
        },
        timestampQuery: {
            adapter: safe(() => gpuDevice.adapter.features.has('timestamp-query'), false) === true,
            device: gpuDevice?.timestamps === true,
        },
        userAgent: text(userAgent),
        source: {
            version: source?.version || 'unknown',
            commit: source?.commit || 'local',
            builtAt: source?.builtAt ?? null,
        },
    };
}

// The fields data/validation/matrix.json says must be recorded ("record": ["browser.version", …]) that a
// bundle environment leaves empty
export function missingFields(environment, matrix) {
    return (matrix?.record ?? []).map(r => r.field).filter(path => {
        const v = path.split('.').reduce((o, k) => (o === null || o === undefined ? undefined : o[k]), environment);
        return v === undefined || v === null || v === '';
    });
}

// ---- Bundle ------------------------------------------------------------------------------------------------

export function buildValidationBundle({ plan, environment, records, generatedAt = new Date().toISOString() }) {
    return {
        bundleVersion: BUNDLE_VERSION,
        plan: plan.version,
        smoke: plan.smoke === true,
        generatedAt,
        environment,
        records,
    };
}

// <date>-<commit>.json, the name under data/validation/<environment>/
export function bundleFileName(bundle) {
    const commit = bundle.environment?.source?.commit || 'local';
    const short = /^[0-9a-f]{7,40}$/.test(commit) ? commit.slice(0, 7) : 'local';
    return `${String(bundle.generatedAt).slice(0, 10)}-${short}${bundle.smoke ? '-smoke' : ''}.json`;
}

export const bundlePath = (bundle) => `data/validation/${bundle.environment?.id || 'unlisted'}/${bundleFileName(bundle)}`;

// The bundle schema with every { "$ref": "experiment-result.v1.json" } replaced by the record schema, so the
// same keyword-subset validator as result-record.js checks a whole bundle
export function resolveBundleSchema(bundleSchema, recordSchema) {
    const resolve = (node) => {
        if (Array.isArray(node)) return node.map(resolve);
        if (!node || typeof node !== 'object') return node;
        if (node.$ref !== undefined) {
            if (node.$ref !== RECORD_SCHEMA_REF && node.$ref !== recordSchema.$id) throw new Error(`unsupported $ref ${node.$ref}`);
            return recordSchema;
        }
        return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, resolve(v)]));
    };
    return resolve(bundleSchema);
}

// { valid, errors } — the bundle against schema/validation-bundle.v1.json, its records against
// schema/experiment-result.v1.json (errors name the record: "records[2].protocol: …")
export function validateValidationBundle(bundle, bundleSchema, recordSchema) {
    return validateResultRecord(bundle, resolveBundleSchema(bundleSchema, recordSchema));
}

// ---- Per-record summaries (page and summary tool) -----------------------------------------------------------

// 'pass' when no check failed and at least one passed, 'fail' when any failed, else 'not-checked'
export function verificationVerdict(record) {
    const v = record.verification ?? [];
    const count = (s) => v.filter(c => c.status === s).length;
    const verdict = count('fail') ? 'fail' : count('pass') ? 'pass' : 'not-checked';
    return { verdict, pass: count('pass'), fail: count('fail'), notChecked: count('not-checked') };
}

export const VERDICT_TEXT = { pass: 'PASS', fail: 'FAIL', 'not-checked': 'NOT CHECKED' };
export const agreesText = (a) => (a === true ? 'agrees' : a === false ? 'disagrees' : 'undetermined');

export function interpretationCounts(record) {
    const claims = record.interpretation ?? [];
    return {
        agrees: claims.filter(c => c.agrees === true).length,
        disagrees: claims.filter(c => c.agrees === false).length,
        undetermined: claims.filter(c => c.agrees !== true && c.agrees !== false).length,
    };
}

// A record's plan-independent label: example / experiment (settings)
export const recordLabel = (record) => `${record.experiment.exampleId} / ${record.experiment.id}`
    + (Object.keys(record.settings ?? {}).length ? ` (${settingsText(record.settings)})` : '');
