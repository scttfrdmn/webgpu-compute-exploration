// Learning path (learn.html): the track data in js/teach/path.js covers every example at Advanced, the
// Beginner and Intermediate tracks (#94) use the right examples and read at their level, every step is
// complete, every control or experiment it names exists, and every link the renderer (js/teach/learn.js)
// produces lands on a real example tab at the visitor's level.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
    TRACKS, EXAMPLES, TRACKS_BY_LEVEL, BEGINNER_TRACKS, INTERMEDIATE_TRACKS, LEVEL_INTROS, BEGINNER_START, tracksForLevel,
} from '../js/teach/path.js';
import { exampleLinks, trackFromHash, emphasisParts, levelView } from '../js/teach/learn.js';
import { parseRoute, TABS } from '../js/teach/routes.js';
import { LEVELS, DEFAULT_LEVEL, searchWithLevel } from '../js/teach/levels.js';
import { UNDERSTAND_LEVELS } from '../js/teach/registry.js';
import { LEVEL_RULES, readability, avoidedTerms } from '../js/teach/readability.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (path) => readFileSync(join(root, path), 'utf8');
const index = read('index.html');
const pageIds = new Set([...index.matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));
const experimentSources = readdirSync(join(root, 'js/teach/experiments')).filter(f => f.endsWith('.js'))
    .map(f => read(`js/teach/experiments/${f}`)).join('\n');

const decode = (s) => s.replace(/&amp;/g, '&').replace(/&#39;|&apos;/g, "'").replace(/&quot;/g, '"');
const pageTitles = new Map([...index.matchAll(
    /<section class="example" id="([^"]+)">\s*<h3><span class="num">(\d+)<\/span>(.*?)<\/h3>/g)]
    .map(m => [m[1], { num: Number(m[2]), title: decode(m[3]) }]));

// Every level's tracks, as [level, track]
const ALL = LEVELS.flatMap(level => TRACKS_BY_LEVEL[level].map(track => [level, track]));

test('index.html has the 25 example sections', () => {
    assert.equal(pageIds.size, 25);
    assert.equal(pageTitles.size, 25);
});

test('every example id in the tracks of every level is a section in index.html', () => {
    for (const [level, track] of ALL) {
        for (const step of track.steps) {
            assert.ok(step.examples.length > 0, `${level}/${track.id}: "${step.title}" has no examples`);
            for (const id of step.examples) assert.ok(pageIds.has(id), `${level}/${track.id}: ${id} is not a section id`);
        }
    }
});

test('every example appears in at least one Advanced track', () => {
    assert.equal(TRACKS_BY_LEVEL.advanced, TRACKS);
    assert.equal(tracksForLevel('advanced'), TRACKS);
    assert.equal(tracksForLevel('junk'), TRACKS);
    const used = new Set(TRACKS.flatMap(t => t.steps.flatMap(s => s.examples)));
    for (const id of pageIds) assert.ok(used.has(id), `${id} is in no track`);
});

test('example titles and numbers match index.html', () => {
    assert.deepEqual(new Set(Object.keys(EXAMPLES)), pageIds);
    for (const [id, info] of pageTitles) assert.deepEqual(EXAMPLES[id], info, id);
});

test('tracks per level', () => {
    assert.deepEqual(Object.keys(TRACKS_BY_LEVEL).sort(), [...LEVELS].sort());
    assert.deepEqual(TRACKS.map(t => t.id), ['gpu', 'science']);
    assert.equal(TRACKS_BY_LEVEL.intermediate, INTERMEDIATE_TRACKS);
    assert.equal(TRACKS_BY_LEVEL.beginner, BEGINNER_TRACKS);
    assert.equal(INTERMEDIATE_TRACKS.length, 1, 'one core track at Intermediate');
    assert.equal(BEGINNER_TRACKS.length, 1, 'one short track at Beginner');
    const steps = (tracks) => tracks.reduce((n, t) => n + t.steps.length, 0);
    assert.ok(steps(INTERMEDIATE_TRACKS) >= 6 && steps(INTERMEDIATE_TRACKS) <= 8, 'Intermediate: 6–8 steps');
    assert.ok(steps(BEGINNER_TRACKS) >= 4 && steps(BEGINNER_TRACKS) <= 6, 'Beginner: about 5 steps');
    const ids = ALL.map(([, t]) => t.id);
    assert.equal(new Set(ids).size, ids.length, 'track ids are unique across levels');
});

test('Beginner steps use only examples with a Beginner Understand page', () => {
    for (const track of BEGINNER_TRACKS) {
        for (const step of track.steps) {
            for (const id of step.examples) {
                assert.ok(UNDERSTAND_LEVELS.beginner.has(id), `${step.title}: ${id} has no Beginner page`);
            }
        }
    }
});

test('Intermediate steps mostly use examples with an Intermediate Understand page', (t) => {
    const ids = INTERMEDIATE_TRACKS.flatMap(tr => tr.steps.flatMap(s => s.examples));
    const withPage = ids.filter(id => UNDERSTAND_LEVELS.intermediate.has(id));
    t.diagnostic(`${withPage.length} of ${ids.length} Intermediate examples have an Intermediate page; `
        + 'the others fall back to the Advanced page');
    assert.ok(withPage.length / ids.length >= 0.5);
});

// The text a visitor reads at a level: the level's intro, each track's title and intro, and every step
function textAtLevel(level) {
    const parts = [LEVEL_INTROS[level], level === 'beginner' ? BEGINNER_START : ''];
    for (const track of TRACKS_BY_LEVEL[level]) {
        parts.push(`${track.title}.`, track.intro);
        for (const step of track.steps) parts.push(`${step.title}.`, step.why, step.tryThis);
    }
    return parts.filter(Boolean).join(' ').replace(/\*/g, '');
}

for (const level of ['beginner', 'intermediate']) {
    test(`${level} learning path reads at its level`, (t) => {
        const rules = LEVEL_RULES[level];
        const text = textAtLevel(level);
        const r = readability(text);
        t.diagnostic(`${r.words} words, grade ${r.grade.toFixed(1)} (≤ ${rules.maxGrade}), `
            + `long words ${(100 * r.longWords).toFixed(1)}% (≤ ${Math.round(100 * rules.maxLongWords)}%)`);
        assert.ok(r.grade <= rules.maxGrade, `Flesch–Kincaid grade ${r.grade.toFixed(1)} > ${rules.maxGrade}`);
        assert.ok(r.longWords <= rules.maxLongWords, `long words ${(100 * r.longWords).toFixed(1)}% > ${100 * rules.maxLongWords}%`);
        assert.deepEqual(avoidedTerms(text, rules.avoid), [], 'terms to avoid at this level');
        // a step is short, so its own grade is a loose check; the whole path is held to the rule above
        for (const track of TRACKS_BY_LEVEL[level]) {
            for (const step of track.steps) {
                const g = readability(`${step.title}. ${step.why} ${step.tryThis}`.replace(/\*/g, '')).grade;
                assert.ok(g <= rules.maxGrade + 2, `${step.title}: grade ${g.toFixed(1)}`);
            }
        }
    });
}

test('per-level page: Advanced keeps the static intro and the flagship pointer, Beginner points at step 1', () => {
    assert.deepEqual(levelView('advanced'), { tracks: TRACKS, tabs: true, intro: null, flagships: true, start: null });
    assert.deepEqual(levelView('intermediate'),
        { tracks: INTERMEDIATE_TRACKS, tabs: false, intro: LEVEL_INTROS.intermediate, flagships: true, start: null });
    assert.deepEqual(levelView('beginner'),
        { tracks: BEGINNER_TRACKS, tabs: false, intro: LEVEL_INTROS.beginner, flagships: false, start: BEGINNER_START });
    assert.match(BEGINNER_START, /step 1/);
    for (const level of ['beginner', 'intermediate']) assert.ok(LEVEL_INTROS[level]?.trim(), `${level} needs an intro`);
});

test('every control or experiment a step names exists in index.html or an experiment file', () => {
    for (const [level, track] of ALL) {
        for (const step of track.steps) {
            for (const [, name] of step.tryThis.matchAll(/\*([^*]+)\*/g)) {
                assert.ok(index.includes(name) || experimentSources.includes(name),
                    `${level}/${track.id}: "${name}" (in "${step.title}") is not in index.html or js/teach/experiments/`);
            }
        }
    }
});

test('tracks and steps are complete', () => {
    for (const [level, track] of ALL) {
        assert.ok(track.title.trim() && track.intro.trim(), `${track.id} needs a title and intro`);
        assert.ok(track.steps.length >= (level === 'advanced' ? 5 : 4), `${track.id} has too few steps`);
        for (const step of track.steps) {
            const where = `${track.id}: "${step.title}"`;
            assert.ok(step.title?.trim(), `${track.id}: step without a title`);
            assert.ok(step.why?.trim(), `${where} needs a why`);
            assert.ok(step.tryThis?.trim(), `${where} needs a tryThis`);
            assert.equal(new Set(step.examples).size, step.examples.length, `${where} repeats an example`);
            assert.equal((step.tryThis.match(/\*/g) ?? []).length % 2, 0, `${where}: unbalanced * in tryThis`);
            assert.ok(!/[<>]/.test(step.why + step.tryThis + step.title), `${where}: no markup in step text`);
            // step text goes in through textContent: TeX would show as raw $…$
            assert.ok(!/\$/.test(step.why + step.tryThis + step.title + track.intro), `${where}: no TeX in step text`);
        }
    }
});

test('every link the renderer produces parses as a route to that example and carries the level', () => {
    for (const [level, track] of ALL) {
        for (const step of track.steps) {
            for (const id of step.examples) {
                const links = exampleLinks(id, level);
                assert.deepEqual(links.map(l => l.tab), TABS);
                const page = `index.html${searchWithLevel('', level)}`;
                for (const { tab, href } of links) {
                    const [before, hash] = href.split('#');
                    assert.equal(before, page, href);
                    assert.equal(new URLSearchParams(before.split('?')[1] ?? '').get('level'),
                        level === DEFAULT_LEVEL ? null : level, href);
                    assert.deepEqual(parseRoute(`#${hash}`, pageIds), { id, tab }, href);
                }
                assert.equal(links[0].href, `${page}#${id}`);
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
    // at the lower levels an Advanced track hash, or any unknown one, selects the level's first track
    for (const level of ['beginner', 'intermediate']) {
        const tracks = TRACKS_BY_LEVEL[level];
        for (const hash of ['#gpu', '#science', '', '#nope']) {
            assert.equal(trackFromHash(hash, tracks), tracks[0].id, `${level} ${hash}`);
        }
        assert.equal(trackFromHash(`#${tracks[0].id}`, tracks), tracks[0].id);
    }
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

test('learn.js registers its window listeners once, not on every render', () => {
    const src = read('js/teach/learn.js');
    const render = src.slice(src.indexOf('export function renderLearningPath'), src.indexOf("if (typeof document !== 'undefined')"));
    assert.ok(render.length > 0, 'renderLearningPath not found');
    assert.doesNotMatch(render, /addEventListener\('(hashchange|popstate)'/, 'renderLearningPath must not add window listeners');
    assert.equal(src.match(/addEventListener\('hashchange'/g)?.length, 1);
    assert.equal(src.match(/addEventListener\('popstate'/g)?.length, 1);
});

test('learn.html uses the shared stylesheet and the renderer, and links back to the catalog', () => {
    const html = read('learn.html');
    assert.match(html, /<link rel="stylesheet" href="css\/site\.css">/);
    assert.match(html, /<script type="module" src="\.\/js\/teach\/learn\.js"><\/script>/);
    assert.match(html, /<link rel="canonical" href="[^"]+\/learn\.html">/);
    assert.match(html, /<meta property="og:url" content="[^"]+\/learn\.html">/);
    assert.match(html, /href="index\.html"/);
    assert.match(html, /id="learning-path"/);
    assert.match(html, /<p id="learn-intro">Two ordered routes/, 'the static intro is the Advanced one');
    assert.match(html, /<section class="flagship-pointer[^"]*"/, 'learn.js hides the flagship pointer by this class');
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
