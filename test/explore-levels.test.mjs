// Level-aware Explore (#92), CPU only: index.html is parsed as text (no DOM). Controls marked with
// data-level show from that level up, "What to look for" prompts (p.look-for, data-levels) show at the
// levels they list; the pure rules live in js/teach/levels.js. This test checks the marks are valid, that
// nothing an example needs (Start / Pause / Reset / primary buttons, presets, canvases, outputs) is ever
// level-hidden, that a label is hidden together with its input and value readout, that every example has
// a prompt for Beginner and Intermediate, and that the wrappers kept every id the example code looks up.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { LEVELS, visibleAt, shownFor, applyExploreLevel } from '../js/teach/levels.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const index = readFileSync(join(root, 'index.html'), 'utf8');

// Every id in index.html before #92 (origin/main at f1c6993). Example code looks these up by id, so the
// control wrappers must keep each one, exactly once. A frozen list: CI checks out one commit only.
const IDS_BEFORE = `
    gpu-status level-picker btn-device-report device-report-feedback device-report-details device-report-preview
    reference-investigations reference-investigations-title flagship-cards vector-addition btn-vector-add
    output-vector-add matrix-multiplication btn-matrix-mul output-matrix-mul gaussian-blur canvas-original
    canvas-blurred btn-image-blur output-image-blur game-of-life btn-gol-start btn-gol-pause btn-gol-step
    btn-gol-clear btn-gol-random gol-speed gol-speed-value canvas-gol output-gol molecular-dynamics btn-md-start
    btn-md-pause btn-md-reset md-atoms md-atoms-value md-temp md-temp-value canvas-md output-md sph-fluid
    btn-sph-start btn-sph-pause btn-sph-reset btn-sph-spawn sph-gravity sph-gravity-value sph-viscosity
    sph-viscosity-value canvas-sph output-sph ray-marching fractal-type fractal-power fractal-power-value
    fractal-quality fractal-quality-value canvas-fractal output-fractal boids btn-boids-start btn-boids-pause
    btn-boids-reset boids-count boids-count-value boids-speed boids-speed-value canvas-boids output-boids dla
    btn-dla-start btn-dla-pause btn-dla-reset dla-rate dla-rate-value dla-stickiness dla-stickiness-value
    canvas-dla output-dla reaction-diffusion btn-rd-start btn-rd-pause btn-rd-reset rd-feed rd-feed-value rd-kill
    rd-kill-value rd-speed rd-speed-value rd-palette canvas-rd output-rd wave-equation btn-wave-start
    btn-wave-pause btn-wave-reset wave-speed wave-speed-value wave-damping wave-damping-value wave-steps
    wave-steps-value btn-wave-rain btn-wave-absorb canvas-wave output-wave falling-sand btn-sand-start
    btn-sand-pause btn-sand-clear btn-sand-sandbox sand-brush sand-brush-value sand-speed sand-speed-value
    canvas-sand output-sand noise btn-noise-animate btn-noise-reset-view noise-type noise-color noise-ridged
    noise-frequency noise-frequency-value noise-octaves noise-octaves-value noise-lacunarity noise-lacunarity-value
    noise-gain noise-gain-value noise-warp noise-warp-value canvas-noise output-noise fft-audio btn-fft-synth
    btn-fft-mic btn-fft-file fft-file btn-fft-start btn-fft-pause btn-fft-mute fft-mode fft-size fft-smoothing
    fft-smoothing-value fft-range fft-range-value canvas-fft output-fft parallel-sort sort-algorithm sort-type
    sort-n btn-sort-bench btn-sort-play btn-sort-pause btn-sort-step btn-sort-shuffle sort-bars sort-pattern
    sort-speed sort-speed-value canvas-sort output-sort kmeans btn-kmeans-start btn-kmeans-pause btn-kmeans-step
    btn-kmeans-reset btn-kmeans-regenerate kmeans-init kmeans-voronoi kmeans-k kmeans-k-value kmeans-n
    kmeans-n-value kmeans-blobs kmeans-blobs-value kmeans-noise kmeans-noise-value kmeans-speed kmeans-speed-value
    canvas-kmeans output-kmeans voronoi btn-voronoi-animate btn-voronoi-random btn-voronoi-clear voronoi-count
    voronoi-count-value voronoi-metric voronoi-view voronoi-jfa1 btn-voronoi-check canvas-voronoi output-voronoi
    maze btn-maze-solve btn-maze-pause btn-maze-clear maze-speed maze-speed-value btn-maze-generate maze-algorithm
    maze-size maze-braid maze-braid-value canvas-maze output-maze cloth btn-cloth-start btn-cloth-pause
    btn-cloth-reset cloth-grid cloth-iterations cloth-iterations-value cloth-stiffness cloth-stiffness-value
    cloth-wind cloth-wind-value cloth-gravity cloth-gravity-value cloth-tear cloth-tear-threshold
    cloth-tear-threshold-value cloth-wireframe cloth-particles canvas-cloth output-cloth path-tracer btn-pt-start
    btn-pt-reset pt-resolution pt-target pt-target-value pt-bounces pt-bounces-value pt-nee pt-exposure
    pt-exposure-value pt-tonemap pt-clamp pt-clamp-value pt-fov pt-fov-value pt-dof pt-aperture pt-aperture-value
    pt-focus pt-focus-value canvas-pt output-pt mnist-train btn-mnist-train-start btn-mnist-train-pause
    btn-mnist-train-reset btn-mnist-export btn-mnist-use mnist-lr mnist-lr-value mnist-batch mnist-epochs
    mnist-epochs-value mnist-speed mnist-speed-value mnist-augment canvas-mnist-train canvas-mnist-filters
    output-mnist-train mnist-recognize btn-mnist-clear btn-mnist-random btn-mnist-cached canvas-mnist-draw
    mnist-bars canvas-mnist-maps output-mnist-recognize nbody btn-nbody-start btn-nbody-pause btn-nbody-reset
    nbody-count nbody-mode nbody-color nbody-trails nbody-theta nbody-theta-value nbody-softening
    nbody-softening-value nbody-dt nbody-dt-value nbody-speed nbody-speed-value canvas-nbody output-nbody
    convolution-reverb btn-reverb-start btn-reverb-stop btn-reverb-synth btn-reverb-mic btn-reverb-file reverb-file
    btn-reverb-bypass btn-reverb-ab reverb-decay reverb-decay-value reverb-predelay reverb-predelay-value
    reverb-mix reverb-mix-value reverb-gain reverb-gain-value reverb-block reverb-depth btn-reverb-ir
    reverb-ir-file canvas-reverb output-reverb wasm-mandelbrot canvas-mandelbrot btn-wasm-mandelbrot
    output-wasm-mandelbrot
`.trim().split(/\s+/);

// ---- a small, strict HTML tokenizer: every element with its attributes and its ancestors ------------
const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);
const RAW = new Set(['script', 'style']);

function parseAttrs(text) {
    const attrs = {};
    for (const m of text.matchAll(/([^\s=/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
        attrs[m[1].toLowerCase()] = m[2] ?? m[3] ?? m[4] ?? '';
    }
    return attrs;
}

function parseHTML(html) {
    const elements = [];
    const stack = [];
    const re = /<!--[\s\S]*?-->|<!DOCTYPE[^>]*>|<(\/?)([a-zA-Z][a-zA-Z0-9-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/gi;
    let m;
    while ((m = re.exec(html))) {
        if (!m[2]) continue;                                   // comment or doctype
        const tag = m[2].toLowerCase();
        if (m[1]) {
            const top = stack.pop();
            assert.equal(top?.tag, tag, `</${tag}> closes <${top?.tag}> at offset ${m.index}`);
            continue;
        }
        const el = { tag, attrs: parseAttrs(m[3]), ancestors: [...stack], offset: m.index };
        el.classes = new Set((el.attrs.class ?? '').split(/\s+/).filter(Boolean));
        elements.push(el);
        if (VOID.has(tag) || m[3].trimEnd().endsWith('/')) continue;
        if (RAW.has(tag)) {                                    // skip to the closing tag
            const end = html.toLowerCase().indexOf(`</${tag}>`, re.lastIndex);
            re.lastIndex = end + tag.length + 3;
            continue;
        }
        stack.push(el);
    }
    assert.equal(stack.length, 0, `unclosed: ${stack.map(e => e.tag).join(', ')}`);
    return elements;
}

const elements = parseHTML(index);
const self = (el) => [el, ...el.ancestors];
const marked = (el) => 'data-level' in el.attrs || 'data-levels' in el.attrs;
const levelMarks = (el) => self(el).filter(marked);
const sectionOf = (el) => self(el).find(e => e.tag === 'section' && e.classes.has('example'));
const sections = elements.filter(e => e.tag === 'section' && e.classes.has('example'));
const inSection = (section) => elements.filter(e => e.ancestors.includes(section));
const describe = (el) => `<${el.tag}${el.attrs.id ? ` id="${el.attrs.id}"` : ''}${el.attrs.class ? ` class="${el.attrs.class}"` : ''}>`;

// Buttons and areas an example needs to be usable or to be paused from outside (js/example-control.js)
function isProtected(el) {
    if (el.tag === 'button' && (el.classes.has('primary')
        || /-(start|pause|stop|reset|resume|play)$/.test(el.attrs.id ?? '')
        || el.ancestors.at(-1)?.classes.has('example'))) return true;    // a one-shot example's Run button
    if (el.tag === 'canvas' || el.classes.has('output') || el.classes.has('seed-buttons')) return true;
    return el.ancestors.some(a => a.classes.has('seed-buttons'));
}

// ---- the pure rules ---------------------------------------------------------------------------------
test('visibleAt: a control shows from its level up; no mark or an unknown mark shows everywhere', () => {
    const table = {
        beginner: [true, true, true],
        intermediate: [false, true, true],
        advanced: [false, false, true],
    };
    for (const [mark, row] of Object.entries(table)) {
        LEVELS.forEach((level, i) => assert.equal(visibleAt(mark, level), row[i], `${mark} at ${level}`));
    }
    for (const level of LEVELS) {
        assert.equal(visibleAt(undefined, level), true);
        assert.equal(visibleAt('expert', level), true);
    }
    // an unknown level counts as the default (Advanced): everything shows
    assert.equal(visibleAt('advanced', 'nonsense'), true);
    assert.equal(visibleAt(' Intermediate ', 'INTERMEDIATE'), true);
});

test('shownFor: a prompt shows at exactly the levels it lists', () => {
    assert.equal(shownFor('beginner intermediate', 'beginner'), true);
    assert.equal(shownFor('beginner intermediate', 'intermediate'), true);
    assert.equal(shownFor('beginner intermediate', 'advanced'), false);
    assert.equal(shownFor('beginner', 'intermediate'), false);
    assert.equal(shownFor('', 'beginner'), false);
    assert.equal(shownFor(undefined, 'beginner'), false);
    assert.equal(shownFor('beginner intermediate', 'nonsense'), false);   // the default level, Advanced
});

test('applyExploreLevel toggles hidden on marked elements and skips the Understand and Measure panels', () => {
    const fake = (attrs, panel = null) => ({
        attrs, hidden: null,
        hasAttribute: (name) => name in attrs,
        getAttribute: (name) => attrs[name] ?? null,
        closest: (selector) => (panel && selector.includes(panel) ? {} : null),
    });
    const control = fake({ 'data-level': 'intermediate' });
    const prompt = fake({ 'data-levels': 'beginner intermediate' });
    const article = fake({ 'data-level': 'beginner' }, '.tab-panel-understand');
    const root = { querySelectorAll: () => [control, prompt, article] };
    const seen = {};
    for (const level of LEVELS) {
        applyExploreLevel(level, root);
        seen[level] = [control.hidden, prompt.hidden];
    }
    assert.deepEqual(seen, { beginner: [true, false], intermediate: [false, false], advanced: [false, true] });
    assert.equal(article.hidden, null, 'the Understand article is never touched');
});

// ---- index.html -------------------------------------------------------------------------------------
test('every data-level and data-levels value names a level; marks sit in an example\'s Explore controls', () => {
    let controls = 0;
    for (const el of elements.filter(marked)) {
        assert.ok(sectionOf(el), `${describe(el)} is outside every section.example`);
        if ('data-level' in el.attrs) {
            controls++;
            assert.ok(LEVELS.includes(el.attrs['data-level']), `${describe(el)}: data-level="${el.attrs['data-level']}"`);
            assert.notEqual(el.attrs['data-level'], 'beginner', `${describe(el)}: beginner is the same as no mark`);
            assert.ok(self(el).some(e => e.classes.has('controls')), `${describe(el)} is not in a .controls row`);
            assert.ok(!('hidden' in el.attrs), `${describe(el)}: applyExploreLevel sets hidden, not the markup`);
        }
        if ('data-levels' in el.attrs) {
            const list = el.attrs['data-levels'].split(/\s+/).filter(Boolean);
            assert.ok(list.length > 0, `${describe(el)}: empty data-levels`);
            for (const level of list) assert.ok(LEVELS.includes(level), `${describe(el)}: data-levels "${level}"`);
            assert.ok(el.classes.has('look-for'), `${describe(el)}: data-levels is for look-for prompts`);
        }
    }
    console.log(`  ${controls} level-marked controls or rows`);
    assert.ok(controls > 0);
});

test('Start / Pause / Reset / primary buttons, presets, canvases and outputs are never level-hidden', () => {
    const guarded = elements.filter(e => sectionOf(e) && isProtected(e));
    assert.ok(guarded.length > 100);
    for (const el of guarded) {
        assert.deepEqual(levelMarks(el).map(describe), [], `${describe(el)} carries or sits inside a level mark`);
    }
    // the off-screen pause and the Explore tab stop click these: each example with buttons still has one
    // (ray marching has none: it renders while on screen)
    for (const section of sections) {
        const inside = inSection(section);
        if (!inside.some(e => e.tag === 'button')) continue;
        const handles = inside.filter(e => e.tag === 'button' && isProtected(e));
        assert.ok(handles.length > 0, `${section.attrs.id} has no unmarked Start / Pause / primary button`);
    }
});

test('a label, its input and its value readout are hidden together', () => {
    const nearestMark = (el) => self(el).find(e => 'data-level' in e.attrs) ?? null;
    const byId = new Map(elements.filter(e => e.attrs.id).map(e => [e.attrs.id, e]));
    let pairs = 0;
    for (const label of elements.filter(e => e.tag === 'label' && e.attrs.for && sectionOf(e))) {
        const target = byId.get(label.attrs.for);
        assert.ok(target, `label for="${label.attrs.for}" has no target`);
        const mark = nearestMark(label);
        assert.equal(nearestMark(target), mark, `#${label.attrs.for} and its label are marked differently`);
        const value = byId.get(`${label.attrs.for}-value`);
        if (value) assert.equal(nearestMark(value), mark, `#${label.attrs.for}-value is marked differently`);
        pairs++;
    }
    for (const wrapper of elements.filter(e => e.tag === 'span' && e.classes.has('control'))) {
        assert.ok('data-level' in wrapper.attrs, `span.control without data-level at offset ${wrapper.offset}`);
        assert.equal(elements.filter(e => e.tag === 'label' && e.ancestors.includes(wrapper)).length, 1,
            `span.control at offset ${wrapper.offset} must hold exactly one label`);
    }
    console.log(`  ${pairs} labelled controls checked`);
});

test('every example has a "What to look for" prompt for Beginner and Intermediate, above its canvas or output', () => {
    assert.equal(sections.length, 25);
    for (const section of sections) {
        const id = section.attrs.id;
        const inside = inSection(section);
        const prompts = inside.filter(e => e.classes.has('look-for'));
        assert.ok(prompts.length > 0, `${id} has no look-for prompt`);
        const covered = new Set();
        for (const p of prompts) {
            assert.equal(p.tag, 'p', `${id}: a look-for prompt is a <p>`);
            assert.ok('hidden' in p.attrs, `${id}: a prompt starts hidden (Advanced, the default, has none)`);
            assert.ok(!p.attrs.id, `${id}: prompts need no id`);
            assert.deepEqual(levelMarks(p), [p], `${id}: a prompt sits inside a level-marked control`);
            const levels = (p.attrs['data-levels'] ?? '').split(/\s+/).filter(Boolean);
            assert.ok(!levels.includes('advanced'), `${id}: Advanced has no prompts`);
            for (const level of levels) {
                assert.ok(!covered.has(level), `${id}: two prompts for ${level}`);
                covered.add(level);
            }
        }
        assert.deepEqual([...covered].sort(), ['beginner', 'intermediate'], `${id}: prompt levels`);
        const target = inside.find(e => e.tag === 'canvas') ?? inside.find(e => e.classes.has('output'));
        for (const p of prompts) assert.ok(p.offset < target.offset, `${id}: prompt after ${describe(target)}`);
    }
});

test('the prompts are short plain text', () => {
    for (const m of index.matchAll(/<p class="look-for"[^>]*><strong>What to look for:<\/strong> ([^<]*)<\/p>/g)) {
        const sentences = m[1].split(/(?<=[.!?])\s+/).filter(Boolean);
        assert.ok(sentences.length >= 1 && sentences.length <= 2, `${sentences.length} sentences: ${m[1]}`);
        assert.ok(m[1].length <= 240, `${m[1].length} characters: ${m[1]}`);
    }
    const all = [...index.matchAll(/class="look-for"/g)].length;
    const plain = [...index.matchAll(/<p class="look-for"[^>]*><strong>What to look for:<\/strong> [^<]*<\/p>/g)].length;
    assert.equal(plain, all, 'every prompt is <strong>What to look for:</strong> plus plain text');
});

test('every id from before #92 still exists exactly once, and all ids are unique', () => {
    const counts = new Map();
    for (const el of elements) if (el.attrs.id) counts.set(el.attrs.id, (counts.get(el.attrs.id) ?? 0) + 1);
    assert.equal(new Set(IDS_BEFORE).size, IDS_BEFORE.length);
    for (const id of IDS_BEFORE) assert.equal(counts.get(id), 1, `#${id}`);
    for (const [id, n] of counts) assert.equal(n, 1, `#${id} appears ${n} times`);
});
