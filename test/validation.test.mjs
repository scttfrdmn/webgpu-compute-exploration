// Physical-GPU validation (#78), CPU only: the validation plan against the flagship experiments' own setting
// options and caps, environment detection, runPlan (js/teach/validate.js) on a mock device and with fake
// experiments (budget stops, errors, cancel), the bundle against schema/validation-bundle.v1.json and the
// record schema, and tools/validation-summary.mjs on synthetic bundles (pass / fail / undetermined, missing
// timestamps, several environments, misplaced or smoke bundles) plus the README block replacement.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, cpSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
    VALIDATION_PLAN, SMOKE_PLAN, SMOKE_BUDGET_MS, FLAGSHIPS, planFor, experimentOf, entryBudgetMs, planDuration,
    detectBrowser, matchEnvironment, validationEnvironment, missingFields, buildValidationBundle, validateValidationBundle,
    resolveBundleSchema, bundleFileName, bundlePath, verificationVerdict, interpretationCounts, recordLabel,
    BUNDLE_SCHEMA_PATH, MATRIX_PATH, RECORD_SCHEMA_REF,
} from '../js/teach/validation.js';
import { runPlan, EXPERIMENTS } from '../js/teach/validate.js';
import { makeContext, MAX_BUDGET_MS } from '../js/teach/measure.js';
import { buildResultRecord, check, SCHEMA_PATH, SUPPORTED_KEYWORDS, ANNOTATIONS } from '../js/teach/result-record.js';
import { MAX_SIZE, CPU_MAX_N } from '../js/teach/experiments/matrix-multiplication.js';
import { BH_MAX_N, BRUTE_MAX_N } from '../js/teach/experiments/nbody.js';
import { MAX_N as SORT_MAX_N } from '../js/teach/experiments/parallel-sort.js';
import { loadBundles, renderSummary, replaceBlock, checkBundle, formatNumber, START, END } from '../tools/validation-summary.mjs';
import { mockDevice } from './helpers/mock-gpu.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const readJSON = (p) => JSON.parse(readFileSync(join(root, p), 'utf8'));
const recordSchema = readJSON(SCHEMA_PATH);
const bundleSchema = readJSON(BUNDLE_SCHEMA_PATH);
const matrix = readJSON(MATRIX_PATH);
const SHA = 'b'.repeat(40);
const SOURCE = { version: '0.6.0', commit: SHA, url: null, builtAt: '2026-11-01T00:00:00.000Z' };

const CHROME_UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36';
const SAFARI_UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15';
const CHROME_HINTS = { platformVersion: '26.1.0', architecture: 'arm',
    fullVersionList: [{ brand: 'Not_A Brand', version: '99.0.0.0' }, { brand: 'Google Chrome', version: '142.0.7444.60' }] };

function fakeGpu({ timestamps = true, device = mockDevice() } = {}) {
    return { device, timestamps,
        adapter: { info: { vendor: 'apple', architecture: 'metal-3', device: '', description: '' }, features: new Set(timestamps ? ['timestamp-query'] : []) } };
}
const quietCtx = (gpuDevice, { budgetMs, signal }) => makeContext(gpuDevice, { budgetMs, signal, waitForUI: async () => {}, deviceLost: () => false });

// ---- The plan ---------------------------------------------------------------------------------------------

test('validation plan: every entry is a flagship experiment with settings from its own options (within caps)', (t) => {
    for (const plan of [VALIDATION_PLAN, SMOKE_PLAN]) {
        assert.ok(Number.isInteger(plan.version) && plan.version >= 1);
        assert.ok(Object.isFrozen(plan) && Object.isFrozen(plan.entries));
        assert.equal(new Set(plan.entries.map(e => e.key)).size, plan.entries.length, 'unique keys');
        for (const entry of plan.entries) {
            const experiment = experimentOf(entry, EXPERIMENTS);
            assert.ok(FLAGSHIPS.includes(entry.exampleId));
            const declared = (experiment.settings ?? []).map(s => s.key);
            assert.deepEqual(Object.keys(entry.settings).sort(), [...declared].sort(), `${entry.key}: every setting fixed, no extra ones`);
            for (const s of experiment.settings ?? []) {
                assert.ok(s.options.some(o => o.value === entry.settings[s.key]), `${entry.key}: ${s.key}=${entry.settings[s.key]} is one of the options`);
            }
            const budget = entryBudgetMs(entry, experiment);
            assert.ok(budget > 0 && budget <= experiment.budgetMs && budget <= MAX_BUDGET_MS);
            assert.ok(entry.expectedSeconds > 0 && entry.expectedSeconds * 1000 <= budget / 2,
                `${entry.key}: expected ${entry.expectedSeconds} s is well within the ${budget / 1000} s budget`);
        }
    }
    for (const id of FLAGSHIPS) assert.ok(VALIDATION_PLAN.entries.some(e => e.exampleId === id), `${id} is in the plan`);
    // every experiment of every flagship is in the full plan
    for (const [id, list] of Object.entries(EXPERIMENTS)) for (const e of list) {
        assert.ok(VALIDATION_PLAN.entries.some(x => x.exampleId === id && x.experimentId === e.id), `${id}/${e.id} is in the plan`);
    }
    // explicit caps
    const s = (key) => VALIDATION_PLAN.entries.find(e => e.key === key).settings;
    assert.ok(s('matmul').maxN <= MAX_SIZE && CPU_MAX_N <= MAX_SIZE);
    assert.ok(s('nbody-theta').n <= Math.min(BH_MAX_N, BRUTE_MAX_N));
    assert.ok(s('sort-u32').maxN <= SORT_MAX_N && s('sort-f32').maxN <= SORT_MAX_N);

    const full = planDuration(VALIDATION_PLAN, EXPERIMENTS), smoke = planDuration(SMOKE_PLAN, EXPERIMENTS);
    t.diagnostic(`plan v${VALIDATION_PLAN.version}: ${VALIDATION_PLAN.entries.length} runs, ~${full.expectedSeconds} s expected, `
        + `${full.worstCaseSeconds} s of budgets; smoke: ${SMOKE_PLAN.entries.length} runs, ${smoke.worstCaseSeconds} s of budgets`);
    assert.equal(full.worstCaseSeconds, VALIDATION_PLAN.entries.length * 20, 'each run keeps its experiment\'s 20 s budget');
    assert.ok(full.expectedSeconds <= 60, 'about a minute on an Apple-silicon Mac');
    assert.ok(full.worstCaseSeconds <= 180, 'at most three minutes even if every run hits its budget');
    // smoke: a subset of the plan's keys, short budgets, small enough for one 90 s browser test
    assert.equal(SMOKE_PLAN.smoke, true);
    assert.ok(SMOKE_PLAN.entries.every(e => VALIDATION_PLAN.entries.some(x => x.key === e.key) && e.budgetMs === SMOKE_BUDGET_MS));
    assert.ok(smoke.worstCaseSeconds <= 20);
    assert.equal(planFor(''), VALIDATION_PLAN);
    assert.equal(planFor('?smoke=1'), SMOKE_PLAN);
    assert.equal(planFor('?smoke=0'), VALIDATION_PLAN);
});

// ---- Environment ------------------------------------------------------------------------------------------

test('detectBrowser and matchEnvironment: Chrome (client hints), Safari 26 (frozen macOS version), others', () => {
    const chrome = detectBrowser(CHROME_UA, CHROME_HINTS);
    assert.deepEqual(chrome, { browser: { name: 'chrome', version: '142.0.7444.60', source: 'client-hints' },
        os: { name: 'macos', version: '26.1.0', source: 'client-hints' }, architecture: 'arm' });
    assert.equal(matchEnvironment(matrix, chrome).id, 'apple-silicon-chrome');
    assert.deepEqual(detectBrowser(CHROME_UA).browser, { name: 'chrome', version: '142.0.0.0', source: 'user-agent' });

    const safari = detectBrowser(SAFARI_UA, null);
    assert.deepEqual(safari, { browser: { name: 'safari', version: '26.1', source: 'user-agent' },
        os: { name: 'macos', version: null, source: 'unknown' }, architecture: null });
    assert.equal(matchEnvironment(matrix, safari).id, 'apple-silicon-safari26');
    assert.equal(matchEnvironment(matrix, detectBrowser(SAFARI_UA.replace('Version/26.1', 'Version/18.6'))), null, 'Safari 18 is not Safari 26');

    assert.equal(detectBrowser('Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:147.0) Gecko/20100101 Firefox/147.0').browser.name, 'firefox');
    assert.equal(detectBrowser(`${CHROME_UA} Edg/142.0.0.0`).browser.name, 'edge');
    assert.equal(detectBrowser('Mozilla/5.0 (X11; Linux x86_64) Chrome/142.0.0.0').os.name, 'linux');
    assert.equal(detectBrowser('').browser.name, 'unknown');
});

test('validationEnvironment: entered values override detection; missingFields follows the matrix', () => {
    const detected = detectBrowser(SAFARI_UA, null);
    const env = validationEnvironment({ gpuDevice: fakeGpu({ timestamps: false }), detected, userAgent: SAFARI_UA, source: SOURCE,
        entered: { id: 'apple-silicon-safari26', browserVersion: '26.1', osVersion: ' 26.1 ', hardware: 'MacBook Air, M2', notes: '' } });
    assert.equal(env.os.version, '26.1');
    assert.equal(env.os.source, 'entered');
    assert.equal(env.browser.source, 'user-agent', 'unchanged detected version keeps its source');
    assert.deepEqual(env.timestampQuery, { adapter: false, device: false });
    assert.equal(env.notes, null);
    assert.deepEqual(missingFields(env, matrix), []);
    const bare = validationEnvironment({ gpuDevice: fakeGpu(), detected, userAgent: SAFARI_UA, source: null });
    assert.equal(bare.id, 'unlisted');
    assert.equal(bare.source.commit, 'local');
    assert.deepEqual(missingFields(bare, matrix), ['os.version', 'hardware']);
    for (const r of matrix.record) assert.ok(r.field && r.how, 'each recorded field says how to get it');
    for (const e of matrix.environments) assert.ok(e.id && e.label && e.os && e.browser && e.minBrowserMajor);
    assert.deepEqual(matrix.environments.map(e => e.id), ['apple-silicon-chrome', 'apple-silicon-safari26']);
    assert.equal(matrix.plan, VALIDATION_PLAN.version, 'matrix and plan versions in step');
});

// ---- Running the plan -------------------------------------------------------------------------------------

test('runPlan: the smoke plan with the real flagship experiments on a mock device yields one valid record each', async (t) => {
    const gpu = fakeGpu({ timestamps: false });
    const progress = [];
    const results = await runPlan(SMOKE_PLAN, { gpuDevice: gpu, source: SOURCE, userAgent: CHROME_UA, newContext: quietCtx,
        onProgress: (p) => progress.push(p.index) });
    assert.equal(results.length, SMOKE_PLAN.entries.length);
    for (const r of results) {
        t.diagnostic(`${r.entry.key}: ${r.status}, ${r.record.outcome.steps} steps, ${r.seconds.toFixed(2)} s on the mock`);
        assert.ok(r.record.outcome.steps > 0);
        assert.deepEqual(r.record.settings, { ...r.entry.settings });
        assert.equal(r.record.protocol.budgetMs, SMOKE_BUDGET_MS);
        assert.equal(r.record.protocol.timing, 'wall-clock');
    }
    assert.ok(progress.length > 0);
    const bundle = buildValidationBundle({ plan: SMOKE_PLAN, environment: validationEnvironment({ gpuDevice: gpu, detected: detectBrowser(CHROME_UA, CHROME_HINTS),
        userAgent: CHROME_UA, source: SOURCE, entered: { id: 'apple-silicon-chrome' } }), records: results.map(r => r.record), generatedAt: '2026-11-01T10:00:00.000Z' });
    assert.deepEqual(validateValidationBundle(bundle, bundleSchema, recordSchema), { valid: true, errors: [] });
    assert.equal(bundle.smoke, true);
    assert.equal(bundleFileName(bundle), '2026-11-01-bbbbbbb-smoke.json');
});

const fake = (id, run, extra = {}) => ({ id, title: id, budgetMs: 20000, columns: [{ key: 'n', label: 'n' }], run, ...extra });
const fakePlan = (ids) => ({ version: 9, entries: ids.map(id => ({ key: id, exampleId: 'demo', experimentId: id, settings: {}, expectedSeconds: 1 })) });

test('runPlan: a budget stop or an error ends only that entry; a cancel stops the rest (not-run)', async () => {
    let clock = 0;
    const controller = new AbortController();
    const experiments = { demo: [
        fake('ok', async function* () { yield { n: 1 }; }, { verify: (rows) => [check('n', rows[0].n, { max: 0 })] }),   // fails its check
        fake('slow', async function* (ctx) { yield { n: 1 }; clock += 30000; await ctx.step(); yield { n: 2 }; }),
        fake('broken', async function* () { throw new Error('kaboom'); }),
        fake('cancelling', async function* (ctx) { yield { n: 1 }; controller.abort(); await ctx.step(); yield { n: 2 }; }),
        fake('after', async function* () { yield { n: 1 }; }),
    ] };
    const errors = [];
    const original = console.error;
    console.error = (...a) => errors.push(a.join(' '));
    let results;
    try {
        results = await runPlan(fakePlan(['ok', 'slow', 'broken', 'cancelling', 'after']), {
            gpuDevice: fakeGpu(), source: SOURCE, experiments, signal: controller.signal, now: () => clock,
            newContext: (g, o) => makeContext(g, { ...o, now: () => clock, waitForUI: async () => {}, deviceLost: () => false }),
        });
    } finally {
        console.error = original;
    }
    assert.deepEqual(results.map(r => r.status), ['done', 'budget', 'error', 'cancelled', 'not-run']);
    assert.equal(verificationVerdict(results[0].record).verdict, 'fail');
    assert.equal(results[1].record.outcome.steps, 1);
    assert.equal(results[2].record.outcome.steps, 0, 'a failed entry still has a record');
    assert.match(results[2].record.outcome.message, /kaboom/);
    assert.ok(errors.some(e => e.startsWith('[Validate] demo/broken')));
    assert.equal(results[4].record, null);
    for (const r of results.slice(0, 4)) assert.deepEqual(validateResultRecordShape(r.record), []);
});

const validateResultRecordShape = (record) => validateValidationBundle(
    buildValidationBundle({ plan: { version: 1 }, environment: sampleEnvironment(), records: [record], generatedAt: 'x' }), bundleSchema, recordSchema).errors;

// ---- Bundle schema ----------------------------------------------------------------------------------------

function sampleEnvironment(overrides = {}) {
    return { ...validationEnvironment({ gpuDevice: fakeGpu(), detected: detectBrowser(CHROME_UA, CHROME_HINTS), userAgent: CHROME_UA, source: SOURCE,
        entered: { id: 'apple-silicon-chrome', hardware: 'Mac mini, M4', notes: 'developer features off' } }), ...overrides };
}

test('bundle schema: keyword subset plus $ref to the record schema; the validator enforces it', () => {
    const seen = new Set();
    const walk = (node) => {
        for (const [k, v] of Object.entries(node)) {
            assert.ok(SUPPORTED_KEYWORDS.has(k) || ANNOTATIONS.has(k) || k === '$ref', `unsupported keyword "${k}"`);
            seen.add(k);
            if (k === 'properties') Object.values(v).forEach(walk);
            if ((k === 'items' || k === 'additionalProperties') && typeof v === 'object') walk(v);
        }
    };
    walk(bundleSchema);
    assert.ok(seen.has('$ref'));
    assert.equal(bundleSchema.properties.records.items.$ref, RECORD_SCHEMA_REF);
    assert.equal(bundleSchema.properties.bundleVersion.const, 1);
    assert.equal(resolveBundleSchema(bundleSchema, recordSchema).properties.records.items, recordSchema);
    assert.throws(() => resolveBundleSchema({ items: { $ref: 'other.json' } }, recordSchema), /unsupported \$ref/);

    const record = buildResultRecord({ exampleId: 'nbody', experiment: fake('demo'), rows: [{ n: 1 }], source: SOURCE, generatedAt: '2026-11-01T00:00:00.000Z' });
    const good = buildValidationBundle({ plan: VALIDATION_PLAN, environment: sampleEnvironment(), records: [record], generatedAt: '2026-11-02T08:00:00.000Z' });
    assert.deepEqual(Object.keys(good), ['bundleVersion', 'plan', 'smoke', 'generatedAt', 'environment', 'records']);
    assert.deepEqual(validateValidationBundle(good, bundleSchema, recordSchema), { valid: true, errors: [] });
    assert.equal(bundleFileName(good), '2026-11-02-bbbbbbb.json');
    assert.equal(bundlePath(good), 'data/validation/apple-silicon-chrome/2026-11-02-bbbbbbb.json');
    assert.equal(bundleFileName({ ...good, environment: { ...good.environment, source: { ...good.environment.source, commit: 'local' } } }), '2026-11-02-local.json');

    const bad = (mutate) => { const b = structuredClone(good); mutate(b); return validateValidationBundle(b, bundleSchema, recordSchema); };
    assert.match(bad(b => { b.bundleVersion = 2; }).errors.join(), /bundleVersion: must be 1/);
    assert.match(bad(b => { delete b.environment.timestampQuery; }).errors.join(), /missing required "timestampQuery"/);
    assert.match(bad(b => { b.environment.browser.name = 'netscape'; }).errors.join(), /environment\.browser\.name/);
    assert.match(bad(b => { b.records = []; }).errors.join(), /records: needs at least 1/);
    assert.match(bad(b => { b.records[0].protocol.timing = 'sundial'; }).errors.join(), /records\[0\]\.protocol\.timing/, 'records are checked against the record schema');
    assert.match(bad(b => { b.extra = 1; }).errors.join(), /unexpected "extra"/);
    assert.match(bad(b => { b.plan = 0; }).errors.join(), /plan: must be >= 1/);
});

// ---- Summary tool -----------------------------------------------------------------------------------------

function recordFor(exampleId, id, { settings = {}, checks, claims, timing = 'timestamp-query', status = 'done' }) {
    const experiment = fake(id, null, {
        verify: checks ? () => checks : undefined,
        interpret: claims ? () => claims : undefined,
    });
    return buildResultRecord({ exampleId, experiment, settings, rows: [{ n: 1 }], outcome: { status }, source: SOURCE,
        gpuDevice: { ...fakeGpu({ timestamps: timing === 'timestamp-query' }), device: { features: new Set(), limits: {} } }, generatedAt: '2026-11-01T00:00:00.000Z' });
}

function fakeRoot() {
    const dir = mkdtempSync(join(tmpdir(), 'validation-summary-'));
    for (const p of [MATRIX_PATH, SCHEMA_PATH, BUNDLE_SCHEMA_PATH]) {
        mkdirSync(join(dir, dirname(p)), { recursive: true });
        cpSync(join(root, p), join(dir, p));
    }
    return dir;
}

function writeBundle(dir, envId, bundle, name = bundleFileName(bundle)) {
    mkdirSync(join(dir, 'data/validation', envId), { recursive: true });
    writeFileSync(join(dir, 'data/validation', envId, name), JSON.stringify(bundle));
}

test('summary tool: correctness, tolerances, timing and interpretation tables over two environments', (t) => {
    const dir = fakeRoot();
    try {
        const matmul = (checks, claims, timing) => recordFor('matrix-multiplication', 'cpu-naive-tiled', { settings: { maxN: 1024 }, checks, claims, timing });
        const chromeRecords = [
            matmul([check('tiled error / √n·u', 0.8, { max: 4 }), check('naive error / √n·u', 0.7, { max: 4 })],
                [{ claim: 'tiled: n³', predicted: '3', measured: '2.98', agrees: true }, { claim: 'GPU beats CPU', predicted: 'yes', measured: '?', agrees: null }]),
            recordFor('nbody', 'scaling', { settings: { preset: 'plummer' }, claims: [{ claim: 'BH crossover', predicted: 'yes', measured: 'no', agrees: false }] }),
            recordFor('parallel-sort', 'scaling', { settings: { dataType: 'u32', maxN: 1048576 }, checks: [check('radix keys ≠ CPU', 3, { max: 0, tolerance: '0' })], status: 'budget' }),
        ];
        const safariRecords = [
            matmul([check('tiled error / √n·u', 5.5, { max: 4 }), check('naive error / √n·u', 0.7, { max: 4 })],
                [{ claim: 'tiled: n³', predicted: '3', measured: '2.5', agrees: false }], 'wall-clock'),
        ];
        const env = (id, overrides) => sampleEnvironment({ id, ...overrides });
        const older = buildValidationBundle({ plan: VALIDATION_PLAN, environment: env('apple-silicon-chrome'), records: [chromeRecords[1]], generatedAt: '2026-10-30T09:00:00.000Z' });
        const chrome = buildValidationBundle({ plan: VALIDATION_PLAN, environment: env('apple-silicon-chrome'), records: chromeRecords, generatedAt: '2026-11-01T09:00:00.000Z' });
        const safari = buildValidationBundle({ plan: VALIDATION_PLAN, records: safariRecords, generatedAt: '2026-11-01T10:00:00.000Z',
            environment: env('apple-silicon-safari26', { browser: { name: 'safari', version: '26.1', source: 'user-agent' },
                os: { name: 'macos', version: null, source: 'unknown' }, timestampQuery: { adapter: false, device: false } }) });
        writeBundle(dir, 'apple-silicon-chrome', older);
        writeBundle(dir, 'apple-silicon-chrome', chrome);
        writeBundle(dir, 'apple-silicon-safari26', safari);

        const loaded = loadBundles(dir);
        assert.deepEqual(loaded.errors, []);
        assert.equal(loaded.bundles.length, 3);
        const md = renderSummary(loaded);
        t.diagnostic(`\n${md}`);
        assert.match(md, /from 3 bundles/);
        assert.match(md, /\| apple-silicon-chrome \(Apple silicon Mac · Chrome\) \| 2026-11-01-bbbbbbb\.json \| 2 \| v1 \| 0\.6\.0 @ bbbbbbb \| chrome 142\.0\.7444\.60 \| macos 26\.1\.0 \| Mac mini, M4 \| apple \/ metal-3 \| yes \| — \|/);
        assert.match(md, /\| apple-silicon-safari26 [^\n]*\| safari 26\.1 \| macos \? \|[^\n]*\| no \(wall-clock\) \| os\.version \|/, 'missing timestamps and OS version');
        // correctness: pass / fail / not checked / stopped; '—' where an environment has no record
        assert.match(md, /\| matrix-multiplication \| cpu-naive-tiled \(maxN=1024\) \| PASS 2\/2 \| FAIL 1\/2 \|/);
        assert.match(md, /\| nbody \| scaling \(preset=plummer\) \| NOT CHECKED \| — \|/);
        assert.match(md, /\| parallel-sort \| scaling \(dataType=u32, maxN=1048576\) \| FAIL 0\/1 · stopped: budget after 1 steps \| — \|/);
        // tolerances: values vs bounds per environment
        assert.match(md, /\| matrix-multiplication \| cpu-naive-tiled \(maxN=1024\) \| tiled error \/ √n·u \| <= 4 \| 0\.8 pass \| 5\.5 fail \|/);
        assert.match(md, /\| parallel-sort \| [^\n]* \| radix keys ≠ CPU \| 0 \| 3 fail \| — \|/);
        // timing availability
        assert.match(md, /\*\*Timing availability\*\*[\s\S]*\| matrix-multiplication \| cpu-naive-tiled \(maxN=1024\) \| timestamp-query \| wall-clock \|/);
        // interpretation: agrees / disagrees / undetermined
        assert.match(md, /\| tiled: n³ \| agrees \| disagrees \|/);
        assert.match(md, /\| GPU beats CPU \| undetermined \| — \|/);
        assert.match(md, /\| BH crossover \| disagrees \| — \|/);
        // flagship order: matrix multiplication, N-body, sorting
        const correctness = md.slice(md.indexOf('**Correctness**'), md.indexOf('**Numerical'));
        assert.ok(correctness.indexOf('matrix-multiplication |') < correctness.indexOf('| nbody |') && correctness.indexOf('| nbody |') < correctness.indexOf('| parallel-sort |'));
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('summary tool: rejects smoke bundles, misplaced files, wrong names, unknown environments, invalid JSON', () => {
    const dir = fakeRoot();
    try {
        const records = [recordFor('nbody', 'scaling', {})];
        const good = buildValidationBundle({ plan: VALIDATION_PLAN, environment: sampleEnvironment(), records, generatedAt: '2026-11-01T09:00:00.000Z' });
        writeBundle(dir, 'apple-silicon-chrome', buildValidationBundle({ plan: SMOKE_PLAN, environment: sampleEnvironment(), records, generatedAt: '2026-11-01T09:00:00.000Z' }), '2026-11-01-bbbbbbb.json');
        writeBundle(dir, 'apple-silicon-safari26', good);                                  // a Chrome bundle under safari26/
        writeBundle(dir, 'apple-silicon-chrome', good, 'latest.json');                     // wrong file name
        writeBundle(dir, 'intel-mac-chrome', good);                                        // not in the matrix
        mkdirSync(join(dir, 'data/validation/apple-silicon-safari26'), { recursive: true });
        writeFileSync(join(dir, 'data/validation/apple-silicon-safari26/broken.json'), '{');
        writeBundle(dir, 'apple-silicon-safari26', { ...good, records: [] }, '2026-11-01-ccccccc.json');
        const { errors, bundles } = loadBundles(dir);
        assert.equal(bundles.length, 0);
        const all = errors.join('\n');
        assert.match(all, /apple-silicon-chrome\/2026-11-01-bbbbbbb\.json: a smoke bundle/);
        assert.match(all, /apple-silicon-safari26\/2026-11-01-bbbbbbb\.json: environment\.id is "apple-silicon-chrome"/);
        assert.match(all, /apple-silicon-chrome\/latest\.json: file name should be 2026-11-01-bbbbbbb\.json/);
        assert.match(all, /intel-mac-chrome\/: not an environment/);
        assert.match(all, /broken\.json: not JSON/);
        assert.match(all, /2026-11-01-ccccccc\.json: schema: records: needs at least 1/);
        assert.deepEqual(checkBundle(good, { envId: 'apple-silicon-chrome', file: bundleFileName(good), bundleSchema, recordSchema }), []);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('summary tool: README block replacement, and the committed README block is up to date', () => {
    const readme = `# T\n\nintro\n${START}\nold\n${END}\n\nafter\n`;
    const once = replaceBlock(readme, 'NEW');
    assert.equal(once, `# T\n\nintro\n${START}\nNEW\n${END}\n\nafter\n`);
    assert.equal(replaceBlock(once, 'NEW'), once, 'idempotent');
    assert.throws(() => replaceBlock('# no markers', 'x'), /exactly one/);
    assert.throws(() => replaceBlock(`${END}\n${START}`, 'x'), /exactly one/);
    assert.throws(() => replaceBlock(`${START}${END}${START}${END}`, 'x'), /exactly one/);

    const loaded = loadBundles(root);
    assert.deepEqual(loaded.errors, [], 'committed bundles are valid and in place');
    const current = readFileSync(join(root, 'README.md'), 'utf8');
    assert.equal(replaceBlock(current, renderSummary(loaded)), current, 'README is current: run node tools/validation-summary.mjs --write');
    assert.match(current, /## Validation on physical GPUs/);
    if (!loaded.bundles.length) assert.match(renderSummary(loaded), /no bundle yet[\s\S]*no bundle yet[\s\S]*No bundles yet/);
    assert.equal(formatNumber(0), '0');
    assert.equal(formatNumber(1.2345e-5), '1.23e-5');
    assert.equal(formatNumber(0.81234), '0.812');
    assert.equal(formatNumber(null), '—');
});

// ---- Page wiring ------------------------------------------------------------------------------------------

test('validate.js imports cleanly; validate.html has every element it uses; Pages deploys the page', async () => {
    assert.deepEqual(Object.keys(EXPERIMENTS).sort(), [...FLAGSHIPS].sort());
    const html = readFileSync(join(root, 'validate.html'), 'utf8');
    const js = readFileSync(join(root, 'js/teach/validate.js'), 'utf8');
    const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
    const used = [...js.matchAll(/\$\('([^']+)'\)/g), ...js.matchAll(/getElementById\('([^']+)'\)/g)].map(m => m[1]);
    assert.ok(used.length > 10);
    for (const id of used) assert.ok(ids.has(id), `validate.html has #${id}`);
    for (const m of js.matchAll(/'(validation-[a-z-]+)'/g)) assert.ok(ids.has(m[1]), `validate.html has #${m[1]}`);
    assert.match(html, /<script type="module" src="\.\/js\/teach\/validate\.js"><\/script>/);
    assert.match(html, /href="css\/site\.css"/);
    const yml = readFileSync(join(root, '.github/workflows/pages.yml'), 'utf8');
    assert.match(yml, /cp -r [^\n]*\bvalidate\.html\b[^\n]*\bdata\b[^\n]*\bschema\b[^\n]*_site\//);
    assert.equal(interpretationCounts({ interpretation: [{ agrees: true }, { agrees: null }, { agrees: false }, { agrees: null }] }).undetermined, 2);
    assert.equal(recordLabel({ experiment: { exampleId: 'nbody', id: 'scaling' }, settings: { preset: 'plummer' } }), 'nbody / scaling (preset=plummer)');
});
