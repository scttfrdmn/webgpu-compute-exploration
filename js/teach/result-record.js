// Experiment result records, schema v1 (schema/experiment-result.v1.json): building one from a Measure
// run, validating one (e.g. an imported file) against the schema, and comparing two.
// Pure apart from the inputs it is given; tested in test/result-record.test.mjs.

import { SPREAD, hasRange } from './stats.js';

export const SCHEMA_VERSION = 1;
export const SCHEMA_PATH = 'schema/experiment-result.v1.json';

// Device limits recorded with a result (the ones that bound compute work)
export const RECORDED_LIMITS = [
    'maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupsPerDimension',
    'maxStorageBuffersPerShaderStage',
];
const ADAPTER_FIELDS = ['vendor', 'architecture', 'device', 'description'];

const safe = (fn, fallback) => { try { return fn() ?? fallback; } catch { return fallback; } };

function cellValue(v) {
    if (v === null || v === undefined) return null;
    if (typeof v === 'number') return Number.isFinite(v) ? v : null;
    if (typeof v === 'string' || typeof v === 'boolean') return v;
    return String(v);
}

export function environmentOf(gpuDevice, userAgent = '') {
    const adapter = gpuDevice?.adapter, device = gpuDevice?.device;
    const info = safe(() => adapter.info, null);
    const fallback = safe(() => info?.isFallbackAdapter ?? adapter?.isFallbackAdapter, null);
    const limits = {};
    for (const name of RECORDED_LIMITS) {
        const v = safe(() => device.limits[name], undefined);
        if (typeof v === 'number' && Number.isFinite(v)) limits[name] = v;
    }
    return {
        userAgent: String(userAgent ?? ''),
        adapter: {
            ...Object.fromEntries(ADAPTER_FIELDS.map(k => [k, String(safe(() => info[k], '') ?? '')])),
            isFallbackAdapter: typeof fallback === 'boolean' ? fallback : null,
        },
        features: safe(() => Array.from(device.features).map(String).sort(), []),
        limits,
    };
}

export const NOT_CHECKED = Object.freeze({ metric: 'none', value: null, tolerance: null,
    reference: 'This experiment has no automatic check', status: 'not-checked' });

// The experiment's verify(rows, settings) output, normalized; a single not-checked entry without verify
export function verificationOf(experiment, rows, settings) {
    if (typeof experiment.verify !== 'function') return [{ ...NOT_CHECKED }];
    let checks;
    try {
        checks = experiment.verify(rows, settings);
    } catch (error) {
        return [{ ...NOT_CHECKED, metric: 'verify', reference: `verify() failed: ${error.message}` }];
    }
    if (!Array.isArray(checks) || !checks.length) return [{ ...NOT_CHECKED }];
    return checks.map(c => ({
        metric: String(c.metric || 'check'),
        value: typeof c.value === 'number' && Number.isFinite(c.value) ? c.value : null,
        tolerance: c.tolerance === undefined || c.tolerance === null ? null : String(c.tolerance),
        reference: c.reference === undefined || c.reference === null ? null : String(c.reference),
        status: ['pass', 'fail', 'not-checked'].includes(c.status) ? c.status : 'not-checked',
    }));
}

// Measured vs predicted claims from an experiment's optional interpret(rows, settings); never throws
export function interpretationOf(experiment, rows, settings) {
    if (typeof experiment.interpret !== 'function') return {};
    let claims;
    try {
        claims = experiment.interpret(rows, settings) ?? [];
    } catch (error) {
        claims = [{ claim: 'interpretation', predicted: '—', measured: '—', agrees: null, note: `interpret() failed: ${error.message}` }];
    }
    if (!claims.length) return {};
    return { interpretation: claims.map(c => ({
        claim: String(c.claim || 'claim'),
        predicted: String(c.predicted ?? '—') || '—',
        measured: String(c.measured ?? '—') || '—',
        agrees: typeof c.agrees === 'boolean' ? c.agrees : null,
        note: c.note === undefined || c.note === null ? null : String(c.note),
    })) };
}

// One check from a value and an inclusive bound, for experiments' verify()
export function check(metric, value, { max = Infinity, min = -Infinity, tolerance, reference = null } = {}) {
    const ok = typeof value === 'number' && Number.isFinite(value);
    return {
        metric, value: ok ? value : null, reference,
        tolerance: tolerance ?? (Number.isFinite(min) && Number.isFinite(max) ? `${min} … ${max}`
            : Number.isFinite(max) ? `<= ${max}` : `>= ${min}`),
        status: !ok ? 'not-checked' : value >= min && value <= max ? 'pass' : 'fail',
    };
}

// run: { exampleId, experiment, settings, rows, outcome: { status, message }, measured, verification,
//        gpuDevice, source, userAgent, generatedAt, budgetMs }
export function buildResultRecord({ exampleId, experiment, settings = {}, rows = [], outcome = { status: 'done' },
    measured = null, verification, gpuDevice = null, source, userAgent = '', generatedAt = new Date().toISOString(),
    budgetMs }) {
    const protocolUsed = measured && measured.calls > 0;
    const columns = experiment.columns.map(c => {
        const col = { key: c.key, label: String(c.label) };
        if (typeof c.unit === 'string') col.unit = c.unit;
        if (hasRange(rows, c.key)) col.range = true;
        return col;
    });
    return {
        schemaVersion: SCHEMA_VERSION,
        generatedAt,
        source: {
            version: source?.version || 'unknown',
            commit: source?.commit || 'local',
            url: source?.url ?? null,
            builtAt: source?.builtAt ?? null,
        },
        experiment: { exampleId, id: experiment.id, title: String(experiment.title ?? experiment.id) },
        settings: Object.fromEntries(Object.entries(settings).map(([k, v]) =>
            [k, typeof v === 'number' || typeof v === 'boolean' ? v : String(v)])),
        protocol: {
            warmup: protocolUsed ? measured.warmup : null,
            repeats: protocolUsed ? measured.repeats : null,
            timing: gpuDevice?.timestamps === true ? 'timestamp-query' : 'wall-clock',
            budgetMs: budgetMs ?? experiment.budgetMs,
            statistic: protocolUsed ? 'median' : 'as-reported',
            spread: columns.some(c => c.range) ? SPREAD : 'none',
        },
        verification: verification ?? verificationOf(experiment, rows, settings),
        ...interpretationOf(experiment, rows, settings),
        environment: environmentOf(gpuDevice, userAgent),
        outcome: { status: outcome.status, steps: rows.length, ...(outcome.message ? { message: String(outcome.message) } : {}) },
        columns,
        rows: rows.map(r => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, cellValue(v)]))),
    };
}

// ---- Validation: the subset of JSON Schema the v1 schema uses -----------------------------------------

export const SUPPORTED_KEYWORDS = new Set(['type', 'properties', 'required', 'additionalProperties', 'items', 'enum',
    'const', 'minimum', 'minLength', 'minItems']);
export const ANNOTATIONS = new Set(['$schema', '$id', 'title', 'description']);

function typeOf(v) {
    if (v === null) return 'null';
    if (Array.isArray(v)) return 'array';
    return typeof v;
}

function matchesType(v, type) {
    switch (type) {
        case 'integer': return typeof v === 'number' && Number.isInteger(v);
        case 'number': return typeof v === 'number' && Number.isFinite(v);
        case 'object': return typeOf(v) === 'object';
        default: return typeOf(v) === type;
    }
}

function validateNode(schema, value, path, errors) {
    const at = path || '(root)';
    if (schema.type !== undefined) {
        const types = Array.isArray(schema.type) ? schema.type : [schema.type];
        if (!types.some(t => matchesType(value, t))) {
            errors.push(`${at}: expected ${types.join(' or ')}, got ${typeOf(value)}`);
            return;
        }
    }
    if ('const' in schema && value !== schema.const) errors.push(`${at}: must be ${JSON.stringify(schema.const)}`);
    if (schema.enum && !schema.enum.includes(value)) errors.push(`${at}: must be one of ${schema.enum.join(', ')}`);
    if (typeof value === 'number' && schema.minimum !== undefined && value < schema.minimum) {
        errors.push(`${at}: must be >= ${schema.minimum}`);
    }
    if (typeof value === 'string' && schema.minLength !== undefined && value.length < schema.minLength) {
        errors.push(`${at}: must not be empty`);
    }
    if (Array.isArray(value)) {
        if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${at}: needs at least ${schema.minItems} item(s)`);
        if (schema.items) value.forEach((item, i) => validateNode(schema.items, item, `${path}[${i}]`, errors));
    }
    if (typeOf(value) === 'object') {
        for (const key of schema.required ?? []) {
            if (!(key in value)) errors.push(`${at}: missing required "${key}"`);
        }
        const props = schema.properties ?? {};
        for (const [key, v] of Object.entries(value)) {
            const sub = path ? `${path}.${key}` : key;
            if (key in props) validateNode(props[key], v, sub, errors);
            else if (schema.additionalProperties === false) errors.push(`${at}: unexpected "${key}"`);
            else if (schema.additionalProperties && typeof schema.additionalProperties === 'object') {
                validateNode(schema.additionalProperties, v, sub, errors);
            }
        }
    }
}

// { valid, errors: ['path: problem', …] } — schema is the parsed schema/experiment-result.v1.json
export function validateResultRecord(record, schema) {
    const errors = [];
    validateNode(schema, record, '', errors);
    return { valid: errors.length === 0, errors };
}

// ---- Comparison ------------------------------------------------------------------------------------

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sortedEntries = (o) => Object.entries(o ?? {}).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
const adapterText = (env) => ADAPTER_FIELDS.map(k => env?.adapter?.[k]).filter(Boolean).join(' / ') || '(not reported)';
const protocolText = (p) => `${p.warmup ?? '—'} warm-up, ${p.repeats ?? '—'} repeats, ${p.timing}, ${p.statistic}, spread ${p.spread}`;
const settingsText = (s) => sortedEntries(s).map(([k, v]) => `${k}=${v}`).join(', ') || '(none)';
const verificationText = (v) => {
    const count = (status) => v.filter(c => c.status === status).length;
    return `${count('pass')} pass, ${count('fail')} fail, ${count('not-checked')} not checked`;
};

// mine, theirs: valid records. Returns { checks: [{ label, same, mine, theirs }], key: { key, label },
// quantities: [{ key, label, unit? }], rows: [{ x, mine: row, theirs: row }] } where rows are the sweep
// points both records have, matched on the first column.
export function compareRecords(mine, theirs) {
    const checks = [
        ['Experiment', `${mine.experiment.exampleId}/${mine.experiment.id}`, `${theirs.experiment.exampleId}/${theirs.experiment.id}`],
        ['Settings', settingsText(mine.settings), settingsText(theirs.settings), same(sortedEntries(mine.settings), sortedEntries(theirs.settings))],
        ['Protocol', protocolText(mine.protocol), protocolText(theirs.protocol),
            same(['warmup', 'repeats', 'timing', 'statistic', 'spread'].map(k => mine.protocol[k]),
                ['warmup', 'repeats', 'timing', 'statistic', 'spread'].map(k => theirs.protocol[k]))],
        ['Source commit', `${mine.source.version} @ ${mine.source.commit}`, `${theirs.source.version} @ ${theirs.source.commit}`,
            mine.source.commit === theirs.source.commit && mine.source.commit !== 'local'],
        ['Adapter', adapterText(mine.environment), adapterText(theirs.environment),
            same(ADAPTER_FIELDS.map(k => mine.environment.adapter[k]), ADAPTER_FIELDS.map(k => theirs.environment.adapter[k]))],
        ['Browser', mine.environment.userAgent || '(not reported)', theirs.environment.userAgent || '(not reported)'],
        ['Outcome', `${mine.outcome.status}, ${mine.outcome.steps} steps`, `${theirs.outcome.status}, ${theirs.outcome.steps} steps`],
        ['Verification', verificationText(mine.verification), verificationText(theirs.verification)],
    ].map(([label, a, b, eq]) => ({ label, mine: a, theirs: b, same: eq ?? a === b }));

    const key = mine.columns[0];
    const theirKeys = new Set(theirs.columns.map(c => c.key));
    const numeric = (record, k) => record.rows.some(r => typeof r[k] === 'number');
    const quantities = mine.columns.slice(1)
        .filter(c => theirKeys.has(c.key) && numeric(mine, c.key) && numeric(theirs, c.key))
        .map(c => ({ key: c.key, label: c.label, ...(c.unit ? { unit: c.unit } : {}) }));
    const rows = [];
    if (theirs.columns[0]?.key === key.key) {
        for (const row of mine.rows) {
            const other = theirs.rows.find(r => String(r[key.key]) === String(row[key.key]));
            if (other) rows.push({ x: row[key.key], mine: row, theirs: other });
        }
    }
    return { checks, key: { key: key.key, label: key.label }, quantities, rows };
}

// Side-by-side values of one quantity: [{ x, mine, theirs, ratio = theirs / mine }] (null when not computable)
export function ratioTable(comparison, quantity) {
    return comparison.rows.map(({ x, mine, theirs }) => {
        const a = typeof mine[quantity] === 'number' ? mine[quantity] : null;
        const b = typeof theirs[quantity] === 'number' ? theirs[quantity] : null;
        return { x, mine: a, theirs: b, ratio: a !== null && b !== null && a !== 0 ? b / a : null };
    });
}

export function resultFileName(record) {
    const stamp = record.generatedAt.replace(/[:.]/g, '-').replace(/-\d{3}Z$/, 'Z');
    return `${record.experiment.exampleId}-${record.experiment.id}-${stamp}.json`;
}
