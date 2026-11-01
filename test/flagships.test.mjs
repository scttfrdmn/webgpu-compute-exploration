// Reference investigations (js/teach/flagships.js): the three landing-page cards point at real examples with
// teaching content, every link is a valid route, and each prediction a card lists is a claim the experiment's
// interpret() actually makes (matched by the claim's key, with the card's terms present in its text).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { FLAGSHIPS, flagshipLinks } from '../js/teach/flagships.js';
import { MEASURE, UNDERSTAND } from '../js/teach/registry.js';
import { parseRoute, TABS } from '../js/teach/routes.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (path) => readFileSync(join(root, path), 'utf8');
const index = read('index.html');
const learn = read('learn.html');
const pageIds = new Set([...index.matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));

// Settings each experiment's interpret() is called with (the defaults a visitor would run)
const SETTINGS = { 'theta-error': { n: 4096, preset: 'plummer' }, 'energy-drift': { mode: 'brute' } };

test('the three flagships are matrix multiplication, N-body and parallel sorting', () => {
    assert.deepEqual(FLAGSHIPS.map(f => f.id), ['matrix-multiplication', 'nbody', 'parallel-sort']);
});

test('every flagship is an example section with Understand and Measure content', () => {
    for (const f of FLAGSHIPS) {
        assert.ok(pageIds.has(f.id), `${f.id} is not a section id in index.html`);
        assert.ok(MEASURE.has(f.id), `${f.id} is not in MEASURE`);
        assert.ok(UNDERSTAND.has(f.id), `${f.id} is not in UNDERSTAND`);
    }
});

test('every card is complete', () => {
    for (const f of FLAGSHIPS) {
        for (const field of ['title', 'question', 'protocol', 'verified', 'caveat']) {
            assert.ok(typeof f[field] === 'string' && f[field].trim(), `${f.id}: ${field}`);
        }
        assert.ok(f.question.endsWith('?'), `${f.id}: the question should be a question`);
        assert.match(f.protocol, /repeats/, `${f.id}: protocol names the repeats`);
        assert.match(f.protocol, /median/, `${f.id}: protocol names the statistic`);
        assert.match(f.caveat, /your|one device/i, `${f.id}: caveat says the results are device-specific`);
        assert.ok(f.predictions.length >= 1 && f.predictions.length <= 2, `${f.id}: one or two predictions`);
    }
});

test('every link parses as a route to the right tab', () => {
    for (const f of FLAGSHIPS) {
        const links = flagshipLinks(f);
        assert.deepEqual(links.map(l => l.label), ['Run it', 'How it works', 'Explore']);
        assert.deepEqual(links.map(l => l.tab), ['measure', 'understand', 'explore']);
        for (const link of links) {
            assert.deepEqual(parseRoute(link.href, pageIds), { id: f.id, tab: link.tab }, link.href);
            assert.ok(TABS.includes(link.tab));
            assert.ok(link.name.includes(f.title), `${link.href}: accessible name names the example`);
        }
        assert.equal(flagshipLinks(f)[0].href, `#${f.id}/measure`);
        for (const link of flagshipLinks(f, 'index.html')) {
            assert.ok(link.href.startsWith('index.html#'));
            assert.deepEqual(parseRoute(link.href.slice('index.html'.length), pageIds), { id: f.id, tab: link.tab });
        }
    }
});

test('every prediction on a card is a claim of that experiment\'s interpret()', async () => {
    for (const f of FLAGSHIPS) {
        const experiments = (await import(`../js/teach/experiments/${f.id}.js`)).default;
        for (const p of f.predictions) {
            const experiment = experiments.find(e => e.id === p.experiment);
            assert.ok(experiment, `${f.id}: no experiment ${p.experiment}`);
            assert.equal(typeof experiment.interpret, 'function', `${f.id}/${p.experiment} has no interpret()`);
            const claims = experiment.interpret([], SETTINGS[p.experiment] ?? {});
            assert.ok(p.claims.length > 0, `${f.id}: "${p.text}" cites no claim`);
            const cited = p.claims.map((key) => {
                const claim = claims.find(c => c.key === key);
                assert.ok(claim, `${f.id}/${p.experiment}: no claim with key ${key} (keys: ${claims.map(c => c.key).join(', ')})`);
                return `${claim.claim} ${claim.predicted}`;
            }).join(' ');
            for (const term of p.terms) {
                assert.ok(cited.includes(term), `${f.id}: "${term}" (from "${p.text}") is not in the cited claims: ${cited}`);
            }
        }
    }
});

test('claim keys are unique within each flagship experiment', async () => {
    for (const f of FLAGSHIPS) {
        for (const experiment of (await import(`../js/teach/experiments/${f.id}.js`)).default) {
            const keys = experiment.interpret([], SETTINGS[experiment.id] ?? {}).map(c => c.key);
            assert.ok(keys.every(k => typeof k === 'string' && k), `${f.id}/${experiment.id}: every claim has a key`);
            assert.equal(new Set(keys).size, keys.length, `${f.id}/${experiment.id}: duplicate keys ${keys}`);
        }
    }
});

test('index.html has the section above the catalog groups, learn.html the pointer', () => {
    const section = index.indexOf('id="reference-investigations"');
    assert.ok(section > 0, 'index.html: no #reference-investigations section');
    assert.ok(section > index.indexOf('</header>') && section < index.indexOf('class="group-title">Compute basics'),
        'the section sits between the hero and the first catalog group');
    assert.match(index, /<div id="flagship-cards"/);
    assert.match(index, /<h2 class="group-title" id="reference-investigations-title">Reference investigations<\/h2>/);
    assert.match(learn, /<div id="flagship-pointer">/);
    assert.match(learn, /Start with a reference investigation/);
});
