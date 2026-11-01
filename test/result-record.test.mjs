// Result records, schema v1 (schema/experiment-result.v1.json, js/teach/result-record.js): building a record
// from a run, the dependency-free validator checked against the schema file itself (every required field,
// enum, const, type, bound and closed object in the schema must be enforced), comparison of two records,
// the deployed-revision lookup (js/source-info.js, tools/write-version.mjs) and the device report embedding.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
    buildResultRecord, validateResultRecord, compareRecords, ratioTable, verificationOf, check, environmentOf,
    resultFileName, SUPPORTED_KEYWORDS, ANNOTATIONS, SCHEMA_PATH, NOT_CHECKED,
} from '../js/teach/result-record.js';
import { runExperiment, makeContext, outcomeOf, ExperimentStopped } from '../js/teach/measure.js';
import { resolveSourceInfo, loadSourceInfo, REPO_URL } from '../js/source-info.js';
import { versionRecord } from '../tools/write-version.mjs';
import { buildDeviceReport } from '../js/device-report.js';
import nbodyExperiments from '../js/teach/experiments/nbody.js';
import { mockDevice } from './helpers/mock-gpu.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const schema = JSON.parse(readFileSync(join(root, SCHEMA_PATH), 'utf8'));
const SHA = 'a'.repeat(40);

function gpuDevice({ timestamps = true, vendor = 'acme', device: deviceName = 'gpu-1' } = {}) {
    return {
        timestamps,
        adapter: { info: { vendor, architecture: 'arch-x', device: deviceName, description: 'Acme GPU', isFallbackAdapter: false } },
        device: { features: new Set(timestamps ? ['timestamp-query'] : []),
            limits: { maxBufferSize: 268435456, maxComputeWorkgroupStorageSize: 16384, maxStorageBufferBindingSize: 134217728,
                maxComputeInvocationsPerWorkgroup: 256, notRecorded: 1 } },
    };
}

const experiment = {
    id: 'demo', title: 'Demo', budgetMs: 20000,
    columns: [{ key: 'n', label: 'N' }, { key: 'ms', label: 'wall ms', unit: 'ms' }, { key: 'err', label: 'error' }],
    verify: (rows) => [check('max error', Math.max(...rows.map(r => r.err)), { max: 1e-3, reference: 'CPU' })],
};
const rows = [
    { n: 1024, ms: 1.5, msLo: 1.2, msHi: 1.9, err: 1e-4 },
    { n: 2048, ms: 3, msLo: 2.5, msHi: 3.5, err: 2e-4 },
];
const source = { version: '0.6.0', commit: SHA, url: `${REPO_URL}/tree/${SHA}`, builtAt: '2026-11-01T00:00:00.000Z' };

function sampleRecord(overrides = {}) {
    return buildResultRecord({
        exampleId: 'nbody', experiment, settings: { preset: 'plummer', n: 4096 }, rows, outcome: { status: 'done' },
        measured: { calls: 2, warmup: 1, repeats: 5 }, gpuDevice: gpuDevice(), source, userAgent: 'TestBrowser/1.0',
        generatedAt: '2026-11-01T12:34:56.789Z', ...overrides,
    });
}

test('buildResultRecord: fields from the run, protocol, verification, environment, ranges', () => {
    const r = sampleRecord();
    assert.equal(r.schemaVersion, 1);
    assert.deepEqual(r.source, source);
    assert.deepEqual(r.experiment, { exampleId: 'nbody', id: 'demo', title: 'Demo' });
    assert.deepEqual(r.settings, { preset: 'plummer', n: 4096 });
    assert.deepEqual(r.protocol, { warmup: 1, repeats: 5, timing: 'timestamp-query', budgetMs: 20000, statistic: 'median', spread: 'iqr' });
    assert.deepEqual(r.verification, [{ metric: 'max error', value: 2e-4, tolerance: '<= 0.001', reference: 'CPU', status: 'pass' }]);
    assert.deepEqual(r.environment.adapter, { vendor: 'acme', architecture: 'arch-x', device: 'gpu-1', description: 'Acme GPU', isFallbackAdapter: false });
    assert.deepEqual(r.environment.features, ['timestamp-query']);
    assert.equal(r.environment.limits.notRecorded, undefined, 'only RECORDED_LIMITS');
    assert.equal(r.environment.limits.maxBufferSize, 268435456);
    assert.deepEqual(r.outcome, { status: 'done', steps: 2 });
    assert.deepEqual(r.columns, [{ key: 'n', label: 'N' }, { key: 'ms', label: 'wall ms', unit: 'ms', range: true }, { key: 'err', label: 'error' }]);
    assert.deepEqual(r.rows, rows);
    assert.equal(resultFileName(r), 'nbody-demo-2026-11-01T12-34-56Z.json');
    assert.deepEqual(validateResultRecord(r, schema), { valid: true, errors: [] });

    // no ctx.measure, no timestamps, no spread, no verify; non-finite cells become null; stopped run
    const plain = buildResultRecord({ exampleId: 'x', experiment: { ...experiment, verify: undefined }, rows: [{ n: 1, ms: NaN, err: Infinity }],
        outcome: outcomeOf(new ExperimentStopped('budget', 20000)), gpuDevice: gpuDevice({ timestamps: false }), source: resolveSourceInfo(null, { version: '0.6.0' }) });
    assert.deepEqual(plain.protocol, { warmup: null, repeats: null, timing: 'wall-clock', budgetMs: 20000, statistic: 'as-reported', spread: 'none' });
    assert.deepEqual(plain.verification, [{ ...NOT_CHECKED }]);
    assert.deepEqual(plain.rows, [{ n: 1, ms: null, err: null }]);
    assert.equal(plain.outcome.status, 'budget');
    assert.match(plain.outcome.message, /first step boundary/);
    assert.deepEqual(plain.source, { version: '0.6.0', commit: 'local', url: null, builtAt: null });
    assert.deepEqual(validateResultRecord(plain, schema).errors, []);

    // no GPU at all: environment still well-formed
    const env = environmentOf(null, undefined);
    assert.deepEqual(env, { userAgent: '', adapter: { vendor: '', architecture: '', device: '', description: '', isFallbackAdapter: null }, features: [], limits: {} });
});

test('verificationOf: normalizes, survives a throwing verify, check() bounds', () => {
    assert.deepEqual(verificationOf({}, [], {}), [{ ...NOT_CHECKED }]);
    assert.deepEqual(verificationOf({ verify: () => [] }, [], {}), [{ ...NOT_CHECKED }]);
    const thrown = verificationOf({ verify: () => { throw new Error('oops'); } }, [], {});
    assert.equal(thrown[0].status, 'not-checked');
    assert.match(thrown[0].reference, /oops/);
    const odd = verificationOf({ verify: () => [{ metric: 'm', value: NaN, status: 'maybe', tolerance: 3 }] }, [], {});
    assert.deepEqual(odd, [{ metric: 'm', value: null, tolerance: '3', reference: null, status: 'not-checked' }]);
    assert.equal(check('x', 5, { min: 1, max: 10 }).status, 'pass');
    assert.equal(check('x', 5, { min: 1, max: 10 }).tolerance, '1 … 10');
    assert.equal(check('x', 11, { max: 10 }).status, 'fail');
    assert.equal(check('x', 0, { min: 1 }).tolerance, '>= 1');
    assert.equal(check('x', undefined, { max: 1 }).status, 'not-checked');
});

// ---- The validator against the schema file -------------------------------------------------------------

test('schema file: only keywords the validator implements', () => {
    const seen = new Set();
    const walk = (node) => {
        for (const [k, v] of Object.entries(node)) {
            assert.ok(SUPPORTED_KEYWORDS.has(k) || ANNOTATIONS.has(k), `unsupported keyword "${k}" in the schema`);
            seen.add(k);
            if (k === 'properties') Object.values(v).forEach(walk);
            if ((k === 'items' || k === 'additionalProperties') && typeof v === 'object') walk(v);
        }
    };
    walk(schema);
    for (const k of SUPPORTED_KEYWORDS) assert.ok(seen.has(k), `validator keyword "${k}" unused by the schema (drift?)`);
    assert.equal(schema.properties.schemaVersion.const, 1);
});

// A value of none of the given types
function wrongType(types) {
    const ok = (v, t) => (t === 'integer' ? Number.isInteger(v) : t === 'number' ? typeof v === 'number'
        : t === 'array' ? Array.isArray(v) : t === 'null' ? v === null
            : t === 'object' ? v !== null && typeof v === 'object' && !Array.isArray(v) : typeof v === t);
    return [{ x: 1 }, [1], 'text', 1.5, true, null].find(v => !types.some(t => ok(v, t)));
}

// Every constraint in the schema, applied at every place the sample record has a value for it:
// [description, mutate(record)] pairs. Arrays are visited at every element, maps at every key.
function mutations(node, get, path, out) {
    const at = (rec) => get(rec);
    if (node.type !== undefined) {
        const types = Array.isArray(node.type) ? node.type : [node.type];
        const bad = wrongType(types);
        out.push([`${path}: wrong type`, (rec, set) => set(rec, bad)]);
    }
    if ('const' in node) out.push([`${path}: const`, (rec, set) => set(rec, node.const + 1)]);
    if (node.enum) out.push([`${path}: enum`, (rec, set) => set(rec, '__not-in-enum__')]);
    if (node.minimum !== undefined) out.push([`${path}: minimum`, (rec, set) => set(rec, node.minimum - 1)]);
    if (node.minLength !== undefined) out.push([`${path}: minLength`, (rec, set) => set(rec, '')]);
    if (node.minItems !== undefined) out.push([`${path}: minItems`, (rec, set) => set(rec, [])]);
    for (const key of node.required ?? []) {
        out.push([`${path}.${key}: required`, (rec) => { delete at(rec)[key]; }]);
    }
    if (node.additionalProperties === false) out.push([`${path}: closed`, (rec) => { at(rec).__extra__ = 1; }]);
    return out;
}

function collect(node, value, get, set, path, out) {
    for (const m of mutations(node, get, path, [])) { m.set = set; out.push(m); }
    if (Array.isArray(value) && node.items) {
        value.forEach((item, i) => collect(node.items, item, (r) => get(r)[i], (r, v) => { get(r)[i] = v; }, `${path}[${i}]`, out));
    }
    if (value && typeof value === 'object' && !Array.isArray(value)) {
        for (const [key, v] of Object.entries(value)) {
            const sub = node.properties?.[key] ?? (typeof node.additionalProperties === 'object' ? node.additionalProperties : null);
            if (sub) collect(sub, v, (r) => get(r)[key], (r, x) => { get(r)[key] = x; }, `${path}.${key}`, out);
        }
    }
    return out;
}

test('validator enforces every constraint of the schema file (walks the schema over a full record)', (t) => {
    const base = sampleRecord({ outcome: { status: 'cancelled', message: 'Cancelled' } });
    assert.ok(validateResultRecord(base, schema).valid);
    // the sample must exercise every optional property the schema declares, so the walk reaches it
    const declared = [];
    const listProps = (node, path) => {
        for (const [k, sub] of Object.entries(node.properties ?? {})) { declared.push(`${path}.${k}`); listProps(sub, `${path}.${k}`); }
        if (node.items) listProps(node.items, `${path}[0]`);
    };
    listProps(schema, '$');
    const out = collect(schema, base, (r) => r, null, '$', []);
    const reached = new Set(out.map(([p]) => p.replace(/: .*$/, '').replace(/\[\d+\]/g, '[0]')));
    for (const p of declared) assert.ok(reached.has(p) || reached.has(`${p}`), `sample record never reaches ${p}`);

    let count = 0;
    for (const m of out) {
        const [description, mutate] = m;
        let copy = structuredClone(base);
        mutate(copy, m.set ?? ((_, v) => { copy = v; }));    // the root itself is replaced
        const { valid, errors } = validateResultRecord(copy, schema);
        assert.equal(valid, false, `${description} is not enforced`);
        assert.ok(errors.length > 0);
        count++;
    }
    const required = out.filter(([d]) => d.endsWith(': required')).length;
    const enums = out.filter(([d]) => d.endsWith(': enum')).length;
    t.diagnostic(`${count} violations rejected (${required} required fields, ${enums} enums)`);
    assert.ok(required >= 30 && enums >= 5, 'the walk covers the required fields and enums');
});

test('validator: messages name the path; non-objects and strangers are rejected', () => {
    const r = sampleRecord();
    r.protocol.timing = 'sundial';
    delete r.environment.adapter.vendor;
    r.rows[0].extra = { nested: true };
    const { valid, errors } = validateResultRecord(r, schema);
    assert.equal(valid, false);
    assert.ok(errors.some(e => e.startsWith('protocol.timing: must be one of')), errors.join('\n'));
    assert.ok(errors.some(e => e === 'environment.adapter: missing required "vendor"'), errors.join('\n'));
    assert.ok(errors.some(e => e.startsWith('rows[0].extra: expected')), errors.join('\n'));
    for (const bad of [null, [], 'record', 42]) assert.equal(validateResultRecord(bad, schema).valid, false);
    assert.equal(validateResultRecord({}, schema).errors.length, schema.required.length);
});

test('N-body experiments on a mock device produce valid records', async () => {
    for (const [id, settings] of [['theta-error', { n: 4096, preset: 'plummer' }], ['scaling', { preset: 'spiral' }],
        ['energy-drift', { n: 2048, mode: 'brute', preset: 'plummer' }]]) {
        const e = nbodyExperiments.find(x => x.id === id);
        const device = mockDevice();
        const ctx = makeContext({ device, timestamps: false }, { budgetMs: e.budgetMs, waitForUI: async () => {}, deviceLost: () => false });
        const runRows = await runExperiment(e, settings, ctx);
        const record = buildResultRecord({ exampleId: 'nbody', experiment: e, settings, rows: runRows, outcome: { status: 'done' },
            measured: ctx.measured, gpuDevice: { device, timestamps: false }, source, budgetMs: ctx.budgetMs });
        assert.deepEqual(validateResultRecord(record, schema).errors, [], id);
        assert.equal(record.protocol.statistic, id === 'energy-drift' ? 'as-reported' : 'median');
        assert.equal(record.protocol.spread, id === 'energy-drift' ? 'none' : 'iqr');
        if (id === 'scaling') assert.deepEqual(record.verification.map(v => v.status), ['not-checked']);
        else assert.ok(record.verification.length > 1);
        // survives a JSON round trip unchanged (what Download writes and Compare reads)
        assert.deepEqual(JSON.parse(JSON.stringify(record)), record);
    }
});

// ---- Comparison -----------------------------------------------------------------------------------------

test('compareRecords: comparability checks and shared rows with ratios', () => {
    const mine = sampleRecord();
    const theirs = sampleRecord({
        gpuDevice: gpuDevice({ vendor: 'other', device: 'gpu-2' }), userAgent: 'OtherBrowser/2.0',
        rows: [{ n: 2048, ms: 6, msLo: 5, msHi: 7, err: 1e-4 }, { n: 4096, ms: 12, err: 1e-4 }],
    });
    const c = compareRecords(mine, theirs);
    const byLabel = Object.fromEntries(c.checks.map(x => [x.label, x]));
    assert.deepEqual(Object.keys(byLabel), ['Experiment', 'Settings', 'Protocol', 'Source commit', 'Adapter', 'Browser', 'Outcome', 'Verification']);
    assert.equal(byLabel.Experiment.same, true);
    assert.equal(byLabel.Settings.same, true);
    assert.equal(byLabel.Protocol.same, true);
    assert.equal(byLabel['Source commit'].same, true);
    assert.equal(byLabel.Adapter.same, false);
    assert.equal(byLabel.Adapter.theirs, 'other / arch-x / gpu-2 / Acme GPU');
    assert.equal(byLabel.Browser.same, false);
    assert.equal(byLabel.Settings.mine, 'n=4096, preset=plummer');
    assert.deepEqual(c.key, { key: 'n', label: 'N' });
    assert.deepEqual(c.quantities, [{ key: 'ms', label: 'wall ms', unit: 'ms' }, { key: 'err', label: 'error' }]);
    assert.deepEqual(c.rows.map(r => r.x), [2048], 'only the sweep points both have');
    assert.deepEqual(ratioTable(c, 'ms'), [{ x: 2048, mine: 3, theirs: 6, ratio: 2 }]);

    // different settings / protocol / local commits are flagged; a different sweep key shares no rows
    const local = { version: '0.6.0', commit: 'local', url: null };
    const other = sampleRecord({ settings: { preset: 'spiral', n: 4096 }, measured: { calls: 1, warmup: 0, repeats: 3 }, source: local });
    const c2 = compareRecords(sampleRecord({ source: local }), other);
    const by2 = Object.fromEntries(c2.checks.map(x => [x.label, x.same]));
    assert.deepEqual([by2.Settings, by2.Protocol, by2['Source commit']], [false, false, false], 'two local checkouts are not the same commit');
    const renamed = sampleRecord({ experiment: { ...experiment, id: 'other', columns: [{ key: 'm', label: 'M' }, ...experiment.columns.slice(1)] },
        rows: [{ m: 1024, ms: 1, err: 0 }] });
    const c3 = compareRecords(mine, renamed);
    assert.equal(c3.checks[0].same, false);
    assert.deepEqual(c3.rows, []);
    // a zero or missing value gives no ratio
    const zero = sampleRecord({ rows: [{ n: 1024, ms: 0, err: 0 }] });
    assert.deepEqual(ratioTable(compareRecords(zero, mine), 'ms'), [{ x: 1024, mine: 0, theirs: 1.5, ratio: null }]);
});

// ---- Source revision ------------------------------------------------------------------------------------

function fakeFetch(files) {
    return async (path) => {
        const body = files[path];
        if (body instanceof Error) throw body;
        if (body === undefined) return { ok: false, status: 404, json: async () => { throw new Error('404'); } };
        return { ok: true, status: 200, json: async () => (typeof body === 'string' ? JSON.parse(body) : body) };
    };
}

test('source info: version.json at deploy, package.json fallback when serving locally', async () => {
    const deployed = versionRecord({ version: '0.6.0' }, SHA, new Date('2026-11-01T00:00:00Z'));
    assert.deepEqual(deployed, { version: '0.6.0', commit: SHA, builtAt: '2026-11-01T00:00:00.000Z' });
    assert.throws(() => versionRecord({ version: '0.6.0' }, undefined), /GITHUB_SHA/);
    assert.throws(() => versionRecord({ version: '0.6.0' }, 'main'), /GITHUB_SHA/);

    const expected = { version: '0.6.0', commit: SHA, builtAt: '2026-11-01T00:00:00.000Z', url: `${REPO_URL}/tree/${SHA}` };
    assert.deepEqual(resolveSourceInfo(deployed, null), expected);
    assert.deepEqual(await loadSourceInfo(fakeFetch({ 'version.json': JSON.stringify(deployed), 'package.json': { version: '9.9.9' } })), expected);

    const local = { version: '0.6.0', commit: 'local', builtAt: null, url: null };
    assert.deepEqual(await loadSourceInfo(fakeFetch({ 'package.json': { version: '0.6.0' } })), local, '404 for version.json');
    assert.deepEqual(await loadSourceInfo(fakeFetch({ 'version.json': new TypeError('offline'), 'package.json': { version: '0.6.0' } })), local);
    assert.deepEqual(await loadSourceInfo(fakeFetch({ 'version.json': { version: '0.6.0', commit: 'not a sha' }, 'package.json': { version: '0.6.0' } })), local);
    assert.deepEqual(await loadSourceInfo(fakeFetch({ 'version.json': '{"version":', 'package.json': { version: '0.6.0' } })), local, 'bad JSON');
    assert.deepEqual(await loadSourceInfo(fakeFetch({})), { ...local, version: 'unknown' });
    assert.deepEqual(await loadSourceInfo(null), { ...local, version: 'unknown' });
    // the real package.json is what a local checkout reports
    const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    assert.match(pkg.version, /^0\./, 'stay on 0.x');
    assert.equal(resolveSourceInfo(null, pkg).version, pkg.version);
});

test('Pages workflow deploys the schema and writes version.json', () => {
    const yml = readFileSync(join(root, '.github/workflows/pages.yml'), 'utf8');
    assert.match(yml, /cp -r [^\n]*\bschema\b[^\n]*_site\//);
    assert.match(yml, /node tools\/write-version\.mjs _site/);
});

// ---- Device report --------------------------------------------------------------------------------------

test('device report embeds each record as a compact JSON block', async () => {
    const record = sampleRecord();
    const md = await buildDeviceReport({
        experiments: [record],
        env: { navigator: { userAgent: 'TestBrowser/1.0' }, window: {}, document: null, location: { origin: 'http://localhost:8000' },
            fetch: fakeFetch({ 'package.json': { version: '0.6.0' } }), now: new Date('2026-11-01T00:00:00Z') },
    });
    assert.match(md, /\| Source commit \| local \|/);
    assert.match(md, /\*\*nbody · Demo\*\* \(done; verified \(1 checks pass\); preset=plummer, n=4096\)/);
    const blocks = [...md.matchAll(/```json\n(.*)\n```/g)].map(m => JSON.parse(m[1]));
    assert.equal(blocks.length, 1);
    assert.deepEqual(blocks[0], record);
    assert.ok(validateResultRecord(blocks[0], schema).valid);
});
