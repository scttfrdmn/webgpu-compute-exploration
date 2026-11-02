// Summarizes the committed physical-GPU validation bundles (#78) as Markdown tables.
//   node tools/validation-summary.mjs            # print the summary
//   node tools/validation-summary.mjs --write    # replace the README block between the markers below
//   node tools/validation-summary.mjs --check    # exit 1 if the README block is out of date
// Reads data/validation/matrix.json and every data/validation/<environment>/*.json, validates each bundle
// against schema/validation-bundle.v1.json (records against schema/experiment-result.v1.json) and checks that
// it sits where it belongs (directory = environment id from the matrix, file name = <date>-<commit>.json, not
// a smoke bundle). Any problem is printed and the tool exits 1 without writing. The summary uses the latest
// bundle of each environment and keeps correctness, numerical tolerances, timing availability and
// interpretation verdicts in separate tables.

import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    validateValidationBundle, bundleFileName, verificationVerdict, agreesText, settingsText, missingFields,
    VERDICT_TEXT, FLAGSHIPS, MATRIX_PATH, BUNDLE_SCHEMA_PATH,
} from '../js/teach/validation.js';
import { SCHEMA_PATH } from '../js/teach/result-record.js';

export const START = '<!-- validation-summary:start -->';
export const END = '<!-- validation-summary:end -->';
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const readJSON = (path) => JSON.parse(readFileSync(path, 'utf8'));

// { matrix, bundles: [{ path, envId, bundle }], errors: [string] } from a repository root
export function loadBundles(root = ROOT) {
    const matrix = readJSON(join(root, MATRIX_PATH));
    const bundleSchema = readJSON(join(root, BUNDLE_SCHEMA_PATH));
    const recordSchema = readJSON(join(root, SCHEMA_PATH));
    const dir = join(root, dirname(MATRIX_PATH));
    const ids = new Set(matrix.environments.map(e => e.id));
    const bundles = [], errors = [];
    for (const name of readdirSync(dir).sort()) {
        const path = join(dir, name);
        if (!statSync(path).isDirectory()) continue;
        const rel = `${dirname(MATRIX_PATH)}/${name}`;
        if (!ids.has(name)) { errors.push(`${rel}/: not an environment of ${MATRIX_PATH}`); continue; }
        for (const file of readdirSync(path).filter(f => f.endsWith('.json')).sort()) {
            const relFile = `${rel}/${file}`;
            let bundle;
            try { bundle = readJSON(join(path, file)); } catch (error) { errors.push(`${relFile}: not JSON (${error.message})`); continue; }
            const problems = checkBundle(bundle, { envId: name, file, bundleSchema, recordSchema });
            if (problems.length) errors.push(...problems.map(p => `${relFile}: ${p}`));
            else bundles.push({ path: relFile, envId: name, bundle });
        }
    }
    return { matrix, bundles, errors };
}

// Problems with one bundle at data/validation/<envId>/<file> ([] when it is fine)
export function checkBundle(bundle, { envId, file, bundleSchema, recordSchema }) {
    const { valid, errors } = validateValidationBundle(bundle, bundleSchema, recordSchema);
    if (!valid) return errors.slice(0, 10).map(e => `schema: ${e}`).concat(errors.length > 10 ? [`schema: … ${errors.length - 10} more`] : []);
    const problems = [];
    if (bundle.smoke) problems.push('a smoke bundle (validate.html?smoke=1) is not part of the validation matrix');
    if (bundle.environment.id !== envId) problems.push(`environment.id is "${bundle.environment.id}" but the file is under ${envId}/`);
    if (file !== bundleFileName(bundle)) problems.push(`file name should be ${bundleFileName(bundle)} (<date>-<commit>.json)`);
    return problems;
}

// ---- Rendering --------------------------------------------------------------------------------------------

const cell = (v) => String(v ?? '—').replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
const table = (headers, rows) => [
    `| ${headers.map(cell).join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map(r => `| ${r.map(cell).join(' | ')} |`),
].join('\n');

export function formatNumber(v) {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
    if (v === 0 || Number.isInteger(v)) return String(v);
    const a = Math.abs(v);
    return a < 1e-3 || a >= 1e5 ? v.toExponential(2) : String(+v.toPrecision(3));
}

const experimentKey = (r) => `${r.experiment.exampleId}\u0000${r.experiment.id}\u0000${settingsText(r.settings)}`;
const experimentName = (r) => `${r.experiment.id}${Object.keys(r.settings).length ? ` (${settingsText(r.settings)})` : ''}`;

// The latest bundle (by generatedAt) of each environment, and how many each has
export function latestByEnvironment(bundles) {
    const latest = new Map(), counts = new Map();
    for (const b of bundles) {
        counts.set(b.envId, (counts.get(b.envId) ?? 0) + 1);
        const cur = latest.get(b.envId);
        if (!cur || b.bundle.generatedAt > cur.bundle.generatedAt) latest.set(b.envId, b);
    }
    return { latest, counts };
}

// Experiments across the given records, flagship order first, then first appearance
function experimentRows(recordLists) {
    const seen = new Map();
    for (const records of recordLists) for (const r of records) if (!seen.has(experimentKey(r))) seen.set(experimentKey(r), r);
    const order = (r) => { const i = FLAGSHIPS.indexOf(r.experiment.exampleId); return i < 0 ? FLAGSHIPS.length : i; };
    return [...seen.values()].map((r, i) => ({ r, i })).sort((a, b) => order(a.r) - order(b.r) || a.i - b.i).map(x => x.r);
}

function correctnessCell(record) {
    if (!record) return '—';
    const v = verificationVerdict(record);
    const total = v.pass + v.fail + v.notChecked;
    const text = v.verdict === 'not-checked' ? VERDICT_TEXT['not-checked']
        : `${VERDICT_TEXT[v.verdict]} ${v.pass}/${total}${v.notChecked ? ` (${v.notChecked} not checked)` : ''}`;
    const o = record.outcome;
    return o.status === 'done' ? text : `${text} · stopped: ${o.status} after ${o.steps} steps`;
}

// Markdown for the README block (without the markers)
export function renderSummary({ matrix, bundles }) {
    const { latest, counts } = latestByEnvironment(bundles);
    const envs = matrix.environments;
    const out = [
        `_Generated by \`node tools/validation-summary.mjs --write\` from ${bundles.length} bundle${bundles.length === 1 ? '' : 's'} in `
            + '`data/validation/` (the latest of each environment). Do not edit by hand._',
        '',
        table(['Environment', 'Latest bundle', 'Bundles', 'Plan', 'Site commit', 'Browser', 'OS', 'Hardware', 'Adapter', 'timestamp-query', 'Not recorded'],
            envs.map(env => {
                const b = latest.get(env.id);
                if (!b) return [`${env.id} (${env.label})`, 'no bundle yet', '0', '—', '—', '—', '—', '—', '—', '—', '—'];
                const e = b.bundle.environment;
                const adapter = ['vendor', 'architecture', 'device', 'description'].map(k => e.adapter[k]).filter(Boolean).join(' / ') || '(not reported)';
                const ts = e.timestampQuery.device ? 'yes' : e.timestampQuery.adapter ? 'offered, not enabled' : 'no (wall-clock)';
                const commit = /^[0-9a-f]{40}$/.test(e.source.commit) ? e.source.commit.slice(0, 7) : e.source.commit;
                return [`${env.id} (${env.label})`, basename(b.path), String(counts.get(env.id)), `v${b.bundle.plan}`, `${e.source.version} @ ${commit}`,
                    `${e.browser.name} ${e.browser.version ?? '?'}`, `${e.os.name} ${e.os.version ?? '?'}`, e.hardware ?? '—', adapter, ts,
                    missingFields(e, matrix).join(', ') || '—'];
            })),
    ];
    const present = envs.filter(env => latest.has(env.id));
    if (!present.length) {
        out.push('', 'No bundles yet. Once one is committed, this block shows correctness, numerical tolerances, timing '
            + 'availability and interpretation verdicts per environment and flagship experiment.');
        return out.join('\n');
    }
    const recordsOf = (env) => latest.get(env.id).bundle.records;
    const find = (env, key) => recordsOf(env).find(r => experimentKey(r) === key);
    const experiments = experimentRows(present.map(recordsOf));
    const envHeads = present.map(env => env.id);

    out.push('', '**Correctness** — verification checks against CPU references (PASS = no check failed; passed/total).', '',
        table(['Flagship', 'Experiment', ...envHeads], experiments.map(x =>
            [x.experiment.exampleId, experimentName(x), ...present.map(env => correctnessCell(find(env, experimentKey(x))))])));

    const tolRows = [];
    for (const x of experiments) {
        const metrics = [];
        for (const env of present) for (const c of find(env, experimentKey(x))?.verification ?? []) {
            if (c.metric !== 'none' && !metrics.some(m => m.metric === c.metric)) metrics.push(c);
        }
        for (const m of metrics) {
            tolRows.push([x.experiment.exampleId, experimentName(x), m.metric, m.tolerance ?? '—', ...present.map(env => {
                const c = find(env, experimentKey(x))?.verification.find(v => v.metric === m.metric);
                if (!c) return '—';
                const bound = c.tolerance !== m.tolerance ? ` (bound ${c.tolerance ?? '—'})` : '';
                return c.status === 'not-checked' ? `not checked${bound}` : `${formatNumber(c.value)} ${c.status}${bound}`;
            })]);
        }
    }
    out.push('', '**Numerical tolerances** — each check\'s measured value against its bound.', '',
        tolRows.length ? table(['Flagship', 'Experiment', 'Check', 'Bound', ...envHeads], tolRows) : '_No checks with values._');

    out.push('', '**Timing availability** — how GPU times were measured (timestamp-query, or wall-clock submit → done).', '',
        table(['Flagship', 'Experiment', ...envHeads], experiments.map(x => [x.experiment.exampleId, experimentName(x),
            ...present.map(env => find(env, experimentKey(x))?.protocol.timing ?? '—')])));

    const claimRows = [];
    for (const x of experiments) {
        const claims = [];
        for (const env of present) for (const c of find(env, experimentKey(x))?.interpretation ?? []) if (!claims.includes(c.claim)) claims.push(c.claim);
        for (const claim of claims) {
            claimRows.push([x.experiment.exampleId, experimentName(x), claim, ...present.map(env => {
                const c = find(env, experimentKey(x))?.interpretation?.find(k => k.claim === claim);
                return c ? agreesText(c.agrees) : '—';
            })]);
        }
    }
    out.push('', '**Interpretation** — measured vs predicted (agrees / disagrees / undetermined within the stated uncertainty).', '',
        claimRows.length ? table(['Flagship', 'Experiment', 'Claim', ...envHeads], claimRows) : '_No interpretation claims._');
    return out.join('\n');
}

// The README with the block between START and END replaced by `summary`; throws if the markers are missing
export function replaceBlock(readme, summary) {
    const a = readme.indexOf(START), b = readme.indexOf(END);
    if (a < 0 || b < 0 || b < a || readme.indexOf(START, a + 1) >= 0) throw new Error(`README needs exactly one ${START} … ${END} block`);
    return `${readme.slice(0, a + START.length)}\n${summary}\n${readme.slice(b)}`;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
    const args = process.argv.slice(2);
    const { matrix, bundles, errors } = loadBundles(ROOT);
    if (errors.length) {
        console.error(`Problems with the validation bundles:\n${errors.map(e => `  ${e}`).join('\n')}`);
        process.exit(1);
    }
    const summary = renderSummary({ matrix, bundles });
    const readmePath = join(ROOT, 'README.md');
    if (args.includes('--write') || args.includes('--check')) {
        const readme = readFileSync(readmePath, 'utf8');
        const updated = replaceBlock(readme, summary);
        if (args.includes('--check')) {
            if (updated !== readme) { console.error('README validation summary is out of date: run node tools/validation-summary.mjs --write'); process.exit(1); }
            console.log('README validation summary is up to date.');
        } else {
            writeFileSync(readmePath, updated);
            console.log(`README.md: validation summary updated from ${bundles.length} bundle(s).`);
        }
    } else {
        console.log(summary);
    }
}
