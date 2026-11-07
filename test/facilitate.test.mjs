// Facilitator guides (#107), CPU only: every guide in content/facilitate/ is listed in js/teach/facilitate.js
// and on facilitate.html, renders without lint problems, has the fixed outline and the four rubric dimensions
// at three levels, links only to things that exist, names controls (in **bold**) exactly as the site labels
// them, and keeps the words read out to a learner (> **Say:** callouts) within the reading rules of the
// session's level. Also: the page wiring (learn.html link, Pages copy list, feedback template).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { SESSIONS, facilitatePath, sessionFromHash, resolveLink } from '../js/teach/facilitate.js';
import { lintMarkdown, headings, renderMarkdown } from '../js/teach/markdown.js';
import { LEVEL_RULES, plainText, readability, avoidedTerms } from '../js/teach/readability.js';
import { parseRoute } from '../js/teach/routes.js';
import { LEVELS, parseLevel, levelText } from '../js/teach/levels.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (path) => readFileSync(join(root, path), 'utf8');
const index = read('index.html');
const pageIds = new Set([...index.matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));
const html = read('facilitate.html');
const DIR = 'content/facilitate';

// The outline of a guide (h2 headings must start with these, in this order)
export const GUIDE_SECTIONS = ['Goal', 'Setup', 'What to say', 'What not to explain first', 'Plan', 'Prompts if stuck',
    'Ending', 'Observation rubric', 'Privacy and consent'];
// The rubric: one h3 per dimension, each with the three levels
export const RUBRIC_DIMENSIONS = ['Forms a prediction', 'Changes one thing at a time', 'Reads the measurement correctly',
    'Explains the conclusion in their own words'];
export const RUBRIC_LEVELS = ['Not yet', 'Partly', 'Clearly'];

// Visible text of an HTML page (tags dropped, the few entities index.html uses decoded)
const textOf = (src) => src.replace(/<script[\s\S]*?<\/script>/g, ' ').replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&nbsp;/g, ' ').replace(/&#39;/g, "'").replace(/\s+/g, ' ');

// Where a bold name may come from: the example's markup in index.html first, then the strings the scripts put on
// the page (readouts, the Measure tab, the tabs, the level menu, the status line, the flagship cards)
function labelSources(example) {
    const section = index.match(new RegExp(`<section class="example" id="${example}">[\\s\\S]*?</section>`))?.[0] ?? '';
    return [
        ['index.html (example)', textOf(section)],
        ['index.html', textOf(index)],
        [`js/examples/${example}.js`, read(`js/examples/${example}.js`)],
        [`js/teach/experiments/${example}.js`, read(`js/teach/experiments/${example}.js`)],
        ...['js/teach/measure.js', 'js/teach/measure-levels.js', 'js/teach/tabs.js', 'js/teach/flagships.js', 'js/main.js',
            'js/gpu-timer.js'].map(p => [p, read(p)]),
        ['levels.js levelText', LEVELS.map(l => levelText(l)).join('\n') + '\nLevel '],
    ];
}

// **bold** spans, except callout labels at the start of a blockquote
const boldNames = (src) => [...src.replace(/^>\s*\*\*[^*]+:\*\*/gm, '>').matchAll(/\*\*([^*]+)\*\*/g)].map(m => m[1].replace(/\s+/g, ' '));

// The text of every > **Say:** callout (the blockquote lines that follow the label)
export function sayBlocks(src) {
    const blocks = [];
    let current = null;
    for (const line of src.split('\n')) {
        const m = /^>\s?(.*)$/.exec(line);
        if (!m) { current = null; continue; }
        const label = /^\*\*Say:\*\*\s*(.*)$/.exec(m[1]);
        if (label) { current = [label[1]]; blocks.push(current); } else if (current) current.push(m[1]);
    }
    return blocks.map(b => b.join('\n').trim());
}

test('every guide file is a session, and every session has a file and a link on facilitate.html', () => {
    const files = readdirSync(join(root, DIR)).filter(f => f.endsWith('.md')).map(f => `${DIR}/${f}`).sort();
    assert.deepEqual(SESSIONS.map(facilitatePath).sort(), files, 'SESSIONS and content/facilitate/*.md differ');
    assert.equal(new Set(SESSIONS.map(s => s.id)).size, SESSIONS.length, 'duplicate session ids');
    for (const s of SESSIONS) {
        assert.equal(s.id, `${s.example}-${s.level}`, `${s.id}: id is <example>-<level>`);
        assert.ok(pageIds.has(s.example), `${s.id}: ${s.example} is not an example on index.html`);
        assert.equal(parseLevel(s.level), s.level, `${s.id}: unknown level`);
        assert.ok(html.includes(`<a href="#${s.id}" data-path="${facilitatePath(s)}">`), `facilitate.html lists ${s.id}`);
        assert.ok(html.includes(`<a href="${facilitatePath(s)}">`), `facilitate.html's noscript links ${facilitatePath(s)}`);
    }
    const listed = [...html.matchAll(/<a href="#([^"]+)" data-path=/g)].map(m => m[1]);
    assert.deepEqual(listed, SESSIONS.map(s => s.id), 'facilitate.html lists exactly the sessions, in order');
    assert.ok(SESSIONS.some(s => s.id === 'nbody-beginner'), 'the N-body Beginner session exists');
    for (const level of LEVELS) assert.ok(SESSIONS.some(s => s.level === level), `a session at ${level}`);
});

test('deep links: the hash selects a session, anything else the first', () => {
    assert.equal(sessionFromHash('#nbody-intermediate').id, 'nbody-intermediate');
    assert.equal(sessionFromHash('#nbody-beginner').id, 'nbody-beginner');
    for (const h of ['', '#', '#nope', null, undefined]) assert.equal(sessionFromHash(h).id, SESSIONS[0].id);
    assert.equal(resolveLink('js/examples/nbody.js'), 'https://github.com/scttfrdmn/webgpu-compute-exploration/blob/main/js/examples/nbody.js');
    assert.equal(resolveLink('index.html?level=beginner#nbody'), 'index.html?level=beginner#nbody');
});

for (const session of SESSIONS) {
    const path = facilitatePath(session);
    test(`facilitator guide ${session.id}: lint, outline, rubric, links, control names, Say text`, (t) => {
        assert.ok(existsSync(join(root, path)), `missing ${path}`);
        const src = read(path);
        assert.deepEqual(lintMarkdown(src), [], 'lintMarkdown problems');

        // outline, in order
        const h2 = headings(src).filter(h => h.level === 2).map(h => h.text.toLowerCase());
        let at = -1;
        for (const section of GUIDE_SECTIONS) {
            const i = h2.findIndex(h => h.startsWith(section.toLowerCase()));
            assert.ok(i >= 0, `missing section "${section}"`);
            assert.ok(i > at, `section "${section}" is out of order`);
            at = i;
        }
        assert.match(src, /20–30 minutes|2\d minutes|2\d–\d\d min/, 'the guide states its length');
        assert.match(src, /collects nothing/, 'the privacy note says the site collects nothing');
        assert.match(src, /consent/i);
        assert.match(src, /anonymi[sz]ed/i);

        // rubric: each dimension is an h3 with a table row per level, and each row has an example
        const rubric = src.slice(src.indexOf('## Observation rubric'), src.indexOf('## Privacy and consent'));
        const h3 = headings(rubric).filter(h => h.level === 3).map(h => h.text);
        assert.deepEqual(h3, RUBRIC_DIMENSIONS, 'rubric dimensions');
        for (const [k, dim] of RUBRIC_DIMENSIONS.entries()) {
            const start = rubric.indexOf(`### ${dim}`);
            const end = k + 1 < RUBRIC_DIMENSIONS.length ? rubric.indexOf(`### ${RUBRIC_DIMENSIONS[k + 1]}`) : rubric.length;
            const rows = rubric.slice(start, end).split('\n').filter(l => /^\|/.test(l)).map(l => l.split('|').slice(1, -1).map(c => c.trim()));
            assert.deepEqual(rows[0], ['Level', 'What you see', 'For example'], `${dim}: table header`);
            assert.deepEqual(rows.slice(2).map(r => r[0]), RUBRIC_LEVELS, `${dim}: one row per level`);
            for (const r of rows.slice(2)) assert.ok(r[1].length > 20 && r[2].length > 10, `${dim} / ${r[0]}: behaviour and example`);
        }

        // links
        const links = [...src.matchAll(/\]\(([^)\s]+)\)/g)].map(m => m[1]);
        for (const href of links) {
            const page = /^(index|learn|facilitate|validate)\.html(\?[^#]*)?(#.*)?$/.exec(href);
            if (page) {
                const level = new URLSearchParams(page[2] ?? '').get('level');
                if (level !== null) assert.ok(parseLevel(level), `${href}: unknown level`);
                if (page[1] === 'index' && page[3]) {
                    assert.ok(parseRoute(page[3], pageIds) || index.includes(`id="${page[3].slice(1)}"`), `${href} names no example tab or section`);
                }
                if (page[1] === 'facilitate' && page[3]) assert.ok(SESSIONS.some(s => `#${s.id}` === page[3]), `${href} is not a session`);
            } else if (href.startsWith('#')) {
                assert.ok(SESSIONS.some(s => `#${s.id}` === href), `${href} is not a session`);
            } else if (/^(js|test|tools|wasm|content|data)\//.test(href)) {
                assert.ok(existsSync(join(root, href.split('#')[0])), `${href} does not exist`);
            } else {
                assert.match(href, /^https:\/\//, `link ${href} should be https`);
                const template = /issues\/new\?template=([\w.-]+)$/.exec(href);
                if (template) assert.ok(existsSync(join(root, '.github/ISSUE_TEMPLATE', template[1])), `${href}: no such issue template`);
            }
        }
        assert.ok(links.some(h => h.startsWith(`index.html?level=${session.level}#${session.example}`) || (session.level === 'advanced' && h.startsWith(`index.html#${session.example}`))),
            'the setup links to the example at the session level');

        // control names in bold are written exactly as the page labels them
        const sources = labelSources(session.example);
        const names = [...new Set(boldNames(src))];
        const missing = names.filter(name => !sources.some(([, text]) => text.includes(name)));
        assert.deepEqual(missing, [], 'bold names not found on the page (index.html, readouts, Measure tab)');
        const inIndex = names.filter(name => sources[0][1].includes(name)).length;
        t.diagnostic(`${names.length} bold names, ${inIndex} of them in the example's markup in index.html`);

        // the words for the learner
        const says = sayBlocks(src);
        assert.ok(says.length >= 3, 'at least three Say callouts');
        const rendered = renderMarkdown(src);
        assert.equal(rendered.match(/<aside class="callout callout-say">/g)?.length, says.length, 'every Say block renders as a callout');
        const rules = LEVEL_RULES[session.level];
        const text = plainText(says.join('\n\n'));
        const r = readability(text);
        if (rules) {
            t.diagnostic(`Say text: ${r.words} words, grade ${r.grade.toFixed(1)} (≤ ${rules.maxGrade}), long words ${(100 * r.longWords).toFixed(1)}% (≤ ${Math.round(100 * rules.maxLongWords)}%)`);
            assert.ok(r.grade <= rules.maxGrade, `Say text: Flesch–Kincaid grade ${r.grade.toFixed(1)} > ${rules.maxGrade}`);
            assert.ok(r.longWords <= rules.maxLongWords, `Say text: long words ${(100 * r.longWords).toFixed(1)}% > ${100 * rules.maxLongWords}%`);
            if (!rules.math) assert.doesNotMatch(says.join('\n'), /\$/, 'no TeX math in the Say text at this level');
            assert.deepEqual(avoidedTerms(text, rules.avoid), [], 'terms to avoid in the Say text');
        } else {
            t.diagnostic(`Say text: ${r.words} words, grade ${r.grade.toFixed(1)}, long words ${(100 * r.longWords).toFixed(1)}% (no rules at ${session.level})`);
        }
    });
}

test('learn.html links to the guides; Pages deploys facilitate.html; the page loads the renderer, not WebGPU', async () => {
    assert.match(read('learn.html'), /<a href="facilitate\.html">For teachers and parents<\/a>/);
    const workflow = read('.github/workflows/pages.yml');
    const cp = workflow.split('\n').find(line => /^\s*cp -r .*_site\/\s*$/.test(line));
    assert.ok(cp.split(/\s+/).includes('facilitate.html'), 'facilitate.html is not copied');
    assert.ok(cp.split(/\s+/).includes('content'), 'content/ is not copied');
    assert.match(html, /<link rel="stylesheet" href="css\/site\.css">/);
    assert.match(html, /<script type="module" src="\.\/js\/teach\/facilitate\.js"><\/script>/);
    assert.match(html, /id="facilitate-guide"/);
    assert.match(html, /id="session-list"/);
    assert.ok(!/main\.js|webgpu-utils/.test(html + read('js/teach/facilitate.js')), 'no WebGPU on the guide page');
    await import(pathToFileURL(join(root, 'js/teach/facilitate.js')).href);   // imports cleanly in Node (no DOM at import)
});

test('the learning-session feedback template asks for anonymised results only', () => {
    const tpl = read('.github/ISSUE_TEMPLATE/learning-session.md');
    assert.match(tpl, /^---\nname: .+\nabout: .+\ntitle: .+\n(labels: .+\n)?---\n/);
    for (const s of SESSIONS) assert.ok(tpl.includes(s.id), `the template lists session ${s.id}`);
    for (const d of RUBRIC_DIMENSIONS) assert.ok(tpl.includes(d), `the template has the rubric row "${d}"`);
    for (const l of RUBRIC_LEVELS) assert.ok(tpl.includes(l), `the template has the level "${l}"`);
    assert.match(tpl, /no names/i);
    assert.match(tpl, /age range/i);
});
