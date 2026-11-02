// Teaching content lint: every example listed in js/teach/registry.js has well-formed content.
//
// Understand pages (content/understand/<id>.md): exist, render without lint problems (bad math, unclosed
// fences, unknown callouts, empty links), contain the required outline, and every internal link
// (#<id>/<tab>) and repository link (js/…, test/…) points at something real.
// Measure experiments (js/teach/experiments/<id>.js): import cleanly in Node, and every experiment
// declares the fields the runner relies on, with a budget within the runner's cap.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { UNDERSTAND, MEASURE, understandPath } from '../js/teach/registry.js';
import { lintMarkdown, headings } from '../js/teach/markdown.js';
import { MAX_BUDGET_MS } from '../js/teach/measure.js';
import { parseRoute } from '../js/teach/routes.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pageIds = new Set([...readFileSync(join(root, 'index.html'), 'utf8')
    .matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));

// The fixed outline of an Understand page (headings must start with these, case-insensitive)
export const REQUIRED_SECTIONS = [
    'Overview', 'The model', 'Mapping to the GPU', 'Numerical behaviour', 'Why it maps',
    'Try this', 'References', 'Read the code',
];

test('registry only lists examples that exist on the page', () => {
    for (const id of [...UNDERSTAND, ...MEASURE]) assert.ok(pageIds.has(id), `${id} is not a section id in index.html`);
});

for (const id of UNDERSTAND) {
    test(`Understand page for ${id}: lint, outline, links`, (t) => {
        const file = join(root, understandPath(id));
        assert.ok(existsSync(file), `missing ${understandPath(id)}`);
        const src = readFileSync(file, 'utf8');
        assert.deepEqual(lintMarkdown(src), [], 'lintMarkdown problems');

        const titles = headings(src).map(h => h.text.toLowerCase());
        for (const section of REQUIRED_SECTIONS) {
            assert.ok(titles.some(h => h.startsWith(section.toLowerCase())), `missing section "${section}"`);
        }

        const links = [...src.matchAll(/\]\(([^)\s]+)\)/g)].map(m => m[1]);
        for (const href of links) {
            if (href.startsWith('#')) {
                assert.ok(parseRoute(href, pageIds), `internal link ${href} does not name an example tab`);
            } else if (/^(js|test|tools|wasm|content|data)\//.test(href)) {
                const path = href.split('#')[0];
                assert.ok(existsSync(join(root, path)), `repository link ${href} does not exist`);
            } else {
                assert.match(href, /^https:\/\//, `link ${href} should be https`);
            }
        }
        t.diagnostic(`${src.split('\n').length} lines, ${titles.length} headings, ${links.length} links`);
    });
}

for (const id of MEASURE) {
    test(`Measure experiments for ${id}: declarations`, async (t) => {
        const url = pathToFileURL(join(root, 'js/teach/experiments', `${id}.js`)).href;
        const { default: experiments } = await import(url);
        assert.ok(Array.isArray(experiments) && experiments.length > 0, 'default export must be a non-empty array');
        const seen = new Set();
        for (const e of experiments) {
            const where = `${id}/${e.id}`;
            for (const field of ['id', 'title', 'question']) assert.equal(typeof e[field], 'string', `${where}: ${field}`);
            assert.ok(!seen.has(e.id), `${where}: duplicate id`); seen.add(e.id);
            assert.ok(Number.isFinite(e.budgetMs) && e.budgetMs > 0 && e.budgetMs <= MAX_BUDGET_MS, `${where}: budgetMs ≤ ${MAX_BUDGET_MS}`);
            assert.ok(e.run && Object.prototype.toString.call(e.run) === '[object AsyncGeneratorFunction]', `${where}: run must be async function*`);
            assert.ok(Array.isArray(e.columns) && e.columns.length > 0, `${where}: columns`);
            if (e.verify !== undefined) assert.equal(typeof e.verify, 'function', `${where}: verify must be a function`);
            for (const c of e.columns) {
                if (c.unit !== undefined) assert.equal(typeof c.unit, 'string', `${where}: column ${c.key} unit`);
                assert.ok(!/(Lo|Hi)$/.test(c.key) || !e.columns.some(o => `${o.key}Lo` === c.key || `${o.key}Hi` === c.key),
                    `${where}: column ${c.key} collides with the spread keys of another column`);
            }
            const keys = new Set(e.columns.map(c => c.key));
            for (const chart of e.charts ?? []) {
                assert.ok(keys.has(chart.x), `${where}: chart x "${chart.x}" is not a column`);
                assert.ok(chart.y.length >= 1 && chart.y.length <= 4, `${where}: 1–4 series per chart`);
                for (const y of chart.y) assert.ok(keys.has(y), `${where}: chart y "${y}" is not a column`);
                assert.equal(typeof chart.xLabel, 'string'); assert.equal(typeof chart.yLabel, 'string');
            }
            if (e.interpret !== undefined) assert.equal(typeof e.interpret, 'function', `${where}: interpret must be a function`);
            if (e.scope !== undefined) {
                for (const k of Object.keys(e.scope)) assert.ok(['shows', 'doesNotShow'].includes(k), `${where}: scope.${k}`);
                for (const list of Object.values(e.scope)) assert.ok(Array.isArray(list) && list.every(x => typeof x === 'string' && x.length), `${where}: scope lists of strings`);
            }
            for (const s of e.settings ?? []) {
                assert.ok(s.options.length > 0, `${where}: setting ${s.key} has options`);
                if (s.default !== undefined) assert.ok(s.options.some(o => o.value === s.default), `${where}: ${s.key} default is an option`);
            }
        }
        t.diagnostic(`${experiments.length} experiments: ${experiments.map(e => e.id).join(', ')}`);
    });
}
