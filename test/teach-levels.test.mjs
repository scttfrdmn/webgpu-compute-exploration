// Visitor levels (#89, #90), CPU only: the pure level functions in js/teach/levels.js, the readability
// measures in js/teach/readability.js, and a lint of every Beginner and Intermediate Understand page listed
// in js/teach/registry.js (file exists, renders cleanly, has its level's outline, links resolve, stays within
// the level's reading rules). The Advanced pages are linted by test/teach-content.test.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
    LEVELS, DEFAULT_LEVEL, parseLevel, resolveLevel, searchWithLevel, fallbackChain, levelText,
} from '../js/teach/levels.js';
import { UNDERSTAND, UNDERSTAND_LEVELS, understandPath, understandSource, hasUnderstand } from '../js/teach/registry.js';
import { LEVEL_RULES, plainText, readability, countSyllables, avoidedTerms } from '../js/teach/readability.js';
import { lintMarkdown, headings } from '../js/teach/markdown.js';
import { parseRoute } from '../js/teach/routes.js';
import { exampleLinks } from '../js/teach/learn.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pageIds = new Set([...readFileSync(join(root, 'index.html'), 'utf8')
    .matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));

// The fixed outline of an Understand page at each lower level (headings must start with these)
export const LEVEL_SECTIONS = {
    beginner: ['What you see', 'What is happening', 'Why a GPU helps', 'Try this'],
    intermediate: ['Overview', 'The idea', 'How the GPU does it', 'How good is the answer', 'Try this', 'Going further'],
};

test('levels: parse, resolve (URL over storage over default), query strings, fallback chain', () => {
    assert.deepEqual(LEVELS, ['beginner', 'intermediate', 'advanced']);
    assert.equal(DEFAULT_LEVEL, 'advanced');
    assert.equal(parseLevel(' Beginner '), 'beginner');
    for (const bad of ['', null, undefined, 'expert', 'advanced!']) assert.equal(parseLevel(bad), null);
    assert.equal(resolveLevel(), 'advanced');
    assert.equal(resolveLevel({ stored: 'intermediate' }), 'intermediate');
    assert.equal(resolveLevel({ search: '?level=beginner', stored: 'intermediate' }), 'beginner');
    assert.equal(resolveLevel({ search: '?level=nonsense', stored: 'intermediate' }), 'intermediate');
    assert.equal(resolveLevel({ search: '?level=nonsense', stored: 'junk' }), 'advanced');
    assert.equal(searchWithLevel('', 'beginner'), '?level=beginner');
    assert.equal(searchWithLevel('?level=beginner', 'advanced'), '', 'the default level leaves no parameter');
    assert.equal(searchWithLevel('?smoke=1', 'intermediate'), '?smoke=1&level=intermediate');
    assert.equal(searchWithLevel('?smoke=1&level=beginner', 'advanced'), '?smoke=1');
    assert.deepEqual(fallbackChain('beginner'), ['beginner', 'intermediate', 'advanced']);
    assert.deepEqual(fallbackChain('intermediate'), ['intermediate', 'advanced']);
    assert.deepEqual(fallbackChain('advanced'), ['advanced']);
    assert.deepEqual(fallbackChain('junk'), ['advanced']);
    assert.equal(levelText('beginner'), 'Beginner (middle school)');
});

test('registry: per-level paths and the fallback to the next level up', () => {
    assert.equal(understandPath('nbody'), 'content/understand/nbody.md');
    assert.equal(understandPath('nbody', 'beginner'), 'content/understand/nbody.beginner.md');
    // an example with only the Advanced page: every level falls back to it
    const advancedOnly = [...UNDERSTAND].find(id => !LEVELS.slice(0, 2).some(l => hasUnderstand(id, l)));
    assert.ok(advancedOnly, 'some example has only the Advanced page');
    assert.deepEqual(understandSource(advancedOnly, fallbackChain('beginner')), { level: 'advanced', path: understandPath(advancedOnly) });
    for (const level of ['beginner', 'intermediate']) {
        for (const id of UNDERSTAND_LEVELS[level]) {
            assert.deepEqual(understandSource(id, fallbackChain(level)), { level, path: understandPath(id, level) });
            assert.ok(UNDERSTAND.has(id), `${id}: a ${level} page needs the Advanced page above it`);
        }
    }
    assert.equal(understandSource('no-such-example', fallbackChain('beginner')), null);
});

test('learning-path links carry the level (none for the default)', () => {
    assert.deepEqual(exampleLinks('nbody').map(l => l.href), ['index.html#nbody', 'index.html#nbody/understand', 'index.html#nbody/measure']);
    assert.equal(exampleLinks('nbody', 'beginner')[1].href, 'index.html?level=beginner#nbody/understand');
});

test('readability: syllables, plain text, Flesch–Kincaid grade and long words', () => {
    for (const [word, n] of [['the', 1], ['cat', 1], ['simple', 2], ['make', 1], ['computer', 3], ['simulation', 4],
        ['gravitational', 5], ['GPU', 1]]) assert.equal(countSyllables(word), n, word);
    const text = plainText('# Title\n\nSee `code()` and $x^2$ [here](#nbody).\n\n```js\nlet a = 1;\n```\n\n- one item\n- two items\n');
    assert.doesNotMatch(text, /code|x\^2|let a|#nbody|```/);
    assert.match(text, /here/);
    assert.match(text, /one item\./, 'list items become sentences');
    assert.equal(readability(plainText('A value of 2.5 is fine, e.g. here. Next one.')).sentences, 2, 'decimals and e.g. do not end sentences');
    const easy = readability('The cat sat on the mat. The dog ran to the park. We like to play.');
    const hard = readability('Hierarchical approximation substantially accelerates gravitational simulations, '
        + 'notwithstanding considerable implementation complexity and numerical sensitivity.');
    assert.ok(easy.grade < 2 && easy.longWords === 0, JSON.stringify(easy));
    assert.ok(hard.grade > 16 && hard.longWords > 0.5, JSON.stringify(hard));
    assert.deepEqual(avoidedTerms('It is asymptotically fast, O(N log N).', LEVEL_RULES.beginner.avoid), ['asymptotically', 'O(N']);
    assert.deepEqual(avoidedTerms('Symptoms and logs.', LEVEL_RULES.beginner.avoid), [], 'whole words only');
});

// The rules have to separate the levels: most Advanced pages fail the Beginner rules
test('readability rules reject most Advanced pages at Beginner', (t) => {
    const rows = [...UNDERSTAND].map(id => {
        const src = readFileSync(join(root, understandPath(id)), 'utf8');
        const r = readability(plainText(src));
        const ok = r.grade <= LEVEL_RULES.beginner.maxGrade && r.longWords <= LEVEL_RULES.beginner.maxLongWords
            && !/\$/.test(src) && avoidedTerms(plainText(src), LEVEL_RULES.beginner.avoid).length === 0;
        return { id, grade: r.grade, longWords: r.longWords, ok };
    });
    const passing = rows.filter(r => r.ok).map(r => r.id);
    t.diagnostic(`Advanced pages: grade ${Math.min(...rows.map(r => r.grade)).toFixed(1)} … ${Math.max(...rows.map(r => r.grade)).toFixed(1)}, `
        + `long words ${(100 * Math.min(...rows.map(r => r.longWords))).toFixed(1)} … ${(100 * Math.max(...rows.map(r => r.longWords))).toFixed(1)}%`);
    assert.deepEqual(passing, [], 'no Advanced page passes as Beginner');
});

for (const level of ['beginner', 'intermediate']) {
    for (const id of UNDERSTAND_LEVELS[level]) {
        test(`${level} Understand page for ${id}: lint, outline, links, reading level`, (t) => {
            const path = understandPath(id, level);
            assert.ok(existsSync(join(root, path)), `missing ${path}`);
            const src = readFileSync(join(root, path), 'utf8');
            assert.deepEqual(lintMarkdown(src), [], 'lintMarkdown problems');
            const titles = headings(src).map(h => h.text.toLowerCase());
            for (const section of LEVEL_SECTIONS[level]) {
                assert.ok(titles.some(h => h.startsWith(section.toLowerCase())), `missing section "${section}"`);
            }
            for (const href of [...src.matchAll(/\]\(([^)\s]+)\)/g)].map(m => m[1])) {
                if (href.startsWith('#')) assert.ok(parseRoute(href, pageIds), `internal link ${href} does not name an example tab`);
                else if (/^(js|test|tools|wasm|content|data)\//.test(href)) assert.ok(existsSync(join(root, href.split('#')[0])), `${href} does not exist`);
                else assert.match(href, /^https:\/\//, `link ${href} should be https`);
            }
            const rules = LEVEL_RULES[level];
            const text = plainText(src);
            const r = readability(text);
            t.diagnostic(`${r.words} words, grade ${r.grade.toFixed(1)} (≤ ${rules.maxGrade}), long words ${(100 * r.longWords).toFixed(1)}% (≤ ${Math.round(100 * rules.maxLongWords)}%)`);
            assert.ok(r.grade <= rules.maxGrade, `Flesch–Kincaid grade ${r.grade.toFixed(1)} > ${rules.maxGrade}`);
            assert.ok(r.longWords <= rules.maxLongWords, `long words ${(100 * r.longWords).toFixed(1)}% > ${100 * rules.maxLongWords}%`);
            if (!rules.math) assert.doesNotMatch(src, /\$/, 'no TeX math at this level');
            assert.deepEqual(avoidedTerms(text, rules.avoid), [], 'terms to avoid at this level');
        });
    }
}
