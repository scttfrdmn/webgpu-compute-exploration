// Learning path (learn.html): the track data in js/teach/path.js covers every example, every step is
// complete, and every link the renderer (js/teach/learn.js) produces lands on a real example tab.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { TRACKS, EXAMPLES } from '../js/teach/path.js';
import { exampleLinks, trackFromHash, emphasisParts } from '../js/teach/learn.js';
import { parseRoute, TABS } from '../js/teach/routes.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (path) => readFileSync(join(root, path), 'utf8');
const index = read('index.html');
const pageIds = new Set([...index.matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));

const decode = (s) => s.replace(/&amp;/g, '&').replace(/&#39;|&apos;/g, "'").replace(/&quot;/g, '"');
const pageTitles = new Map([...index.matchAll(
    /<section class="example" id="([^"]+)">\s*<h3><span class="num">(\d+)<\/span>(.*?)<\/h3>/g)]
    .map(m => [m[1], { num: Number(m[2]), title: decode(m[3]) }]));

test('index.html has the 25 example sections', () => {
    assert.equal(pageIds.size, 25);
    assert.equal(pageTitles.size, 25);
});

test('every example id in the tracks is a section in index.html', () => {
    for (const track of TRACKS) {
        for (const step of track.steps) {
            assert.ok(step.examples.length > 0, `${track.id}: "${step.title}" has no examples`);
            for (const id of step.examples) assert.ok(pageIds.has(id), `${track.id}: ${id} is not a section id`);
        }
    }
});

test('every example appears in at least one track', () => {
    const used = new Set(TRACKS.flatMap(t => t.steps.flatMap(s => s.examples)));
    for (const id of pageIds) assert.ok(used.has(id), `${id} is in no track`);
});

test('example titles and numbers match index.html', () => {
    assert.deepEqual(new Set(Object.keys(EXAMPLES)), pageIds);
    for (const [id, info] of pageTitles) assert.deepEqual(EXAMPLES[id], info, id);
});

test('tracks and steps are complete', () => {
    assert.deepEqual(TRACKS.map(t => t.id), ['gpu', 'science']);
    for (const track of TRACKS) {
        assert.ok(track.title.trim() && track.intro.trim(), `${track.id} needs a title and intro`);
        assert.ok(track.steps.length >= 5, `${track.id} has too few steps`);
        for (const step of track.steps) {
            const where = `${track.id}: "${step.title}"`;
            assert.ok(step.title?.trim(), `${track.id}: step without a title`);
            assert.ok(step.why?.trim(), `${where} needs a why`);
            assert.ok(step.tryThis?.trim(), `${where} needs a tryThis`);
            assert.equal(new Set(step.examples).size, step.examples.length, `${where} repeats an example`);
            assert.equal((step.tryThis.match(/\*/g) ?? []).length % 2, 0, `${where}: unbalanced * in tryThis`);
            assert.ok(!/[<>]/.test(step.why + step.tryThis + step.title), `${where}: no markup in step text`);
        }
    }
});

test('every link the renderer produces parses as a route to that example', () => {
    for (const track of TRACKS) {
        for (const step of track.steps) {
            for (const id of step.examples) {
                const links = exampleLinks(id);
                assert.deepEqual(links.map(l => l.tab), TABS);
                for (const { tab, href } of links) {
                    const [page, hash] = href.split('#');
                    assert.equal(page, 'index.html', href);
                    assert.deepEqual(parseRoute(`#${hash}`, pageIds), { id, tab }, href);
                }
                assert.equal(links[0].href, `index.html#${id}`);
            }
        }
    }
});

test('track deep links', () => {
    assert.equal(trackFromHash('#gpu'), 'gpu');
    assert.equal(trackFromHash('#science'), 'science');
    assert.equal(trackFromHash(''), 'gpu');
    assert.equal(trackFromHash('#nope'), 'gpu');
    assert.equal(trackFromHash(undefined), 'gpu');
});

test('emphasis in tryThis', () => {
    assert.deepEqual(emphasisParts('Run *A b* now'),
        [{ text: 'Run ', em: false }, { text: 'A b', em: true }, { text: ' now', em: false }]);
    assert.deepEqual(emphasisParts('*X*'), [{ text: 'X', em: true }]);
});

test('learn.js builds the page with textContent only and without WebGPU', () => {
    const src = read('js/teach/learn.js');
    assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML/.test(src), 'no HTML injection of data');
    assert.ok(!/main\.js|navigator\.gpu|webgpu-utils/.test(src), 'no WebGPU on the learning path');
});

test('learn.html uses the shared stylesheet and the renderer, and links back to the catalog', () => {
    const html = read('learn.html');
    assert.match(html, /<link rel="stylesheet" href="css\/site\.css">/);
    assert.match(html, /<script type="module" src="\.\/js\/teach\/learn\.js"><\/script>/);
    assert.match(html, /<link rel="canonical" href="[^"]+\/learn\.html">/);
    assert.match(html, /<meta property="og:url" content="[^"]+\/learn\.html">/);
    assert.match(html, /href="index\.html"/);
    assert.match(html, /id="learning-path"/);
    assert.ok(!/main\.js/.test(html), 'learn.html must not load main.js');
});

test('index.html links to the learning path', () => {
    assert.match(index, /<a href="learn\.html">Learning path<\/a>/);
});

test('the Pages workflow copies learn.html', () => {
    const workflow = read('.github/workflows/pages.yml');
    const cp = workflow.split('\n').find(line => /^\s*cp -r .*_site\/\s*$/.test(line));
    assert.ok(cp, 'no cp -r … _site/ line');
    assert.ok(cp.split(/\s+/).includes('learn.html'), 'learn.html is not copied');
});
