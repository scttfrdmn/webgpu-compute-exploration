// Renders the learning path (learn.html) from the data in path.js for the visitor's level (#94): a track
// switcher (tabs, deep-linkable as learn.html#gpu / learn.html#science at Advanced; omitted when the level
// has a single track) and one card per step, plus a short per-level intro. No WebGPU here, and all data
// goes in through textContent. The pure helpers are exported for test/learn-path.test.mjs; the page is
// only built when this module runs in a browser.

import { TRACKS, EXAMPLES, tracksForLevel, LEVEL_INTROS, BEGINNER_START } from './path.js';
import { formatRoute } from './routes.js';
import { renderFlagshipPointer } from './flagships.js';
import { getLevel, mountLevelSelector, onLevelChange, searchWithLevel, DEFAULT_LEVEL } from './levels.js';

const CATALOG = 'index.html';
const VIEWS = [['explore', 'Explore'], ['understand', 'Understand'], ['measure', 'Measure']];

// The three links of an example at a visitor level: [{ tab, label, href }]
export function exampleLinks(id, level = DEFAULT_LEVEL) {
    return VIEWS.map(([tab, label]) => ({ tab, label, href: `${CATALOG}${searchWithLevel('', level)}${formatRoute(id, tab)}` }));
}

// The track a hash names ("#science" -> 'science'); anything else selects the first track
export function trackFromHash(hash, tracks = TRACKS) {
    const id = decodeURIComponent(String(hash ?? '').replace(/^#/, ''));
    return tracks.some(t => t.id === id) ? id : tracks[0].id;
}

// Splits "Run *Name* now" into [{ text, em }] parts; *…* marks an experiment or control name
export function emphasisParts(text) {
    return String(text).split('*').map((part, i) => ({ text: part, em: i % 2 === 1 })).filter(p => p.text);
}

// What the page shows at a level: the tracks, whether there is a tab bar, the intro text (null: keep the
// static Advanced intro of learn.html), and whether the reference-investigation pointer is shown or replaced
// by a line pointing at the first step
export function levelView(level) {
    const tracks = tracksForLevel(level);
    return {
        tracks,
        tabs: tracks.length > 1,
        intro: LEVEL_INTROS[level] ?? null,
        flagships: level !== 'beginner',
        start: level === 'beginner' ? BEGINNER_START : null,
    };
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function renderStep(step, index, level) {
    const card = el('li', 'step-card');
    const head = el('div', 'step-head');
    head.append(el('span', 'step-num', `Step ${index + 1}`), el('h3', 'step-title', step.title));
    card.append(head);

    const examples = el('ul', 'step-examples');
    for (const id of step.examples) {
        const info = EXAMPLES[id];
        const item = el('li', 'step-example');
        const name = el('span', 'step-example-name');
        name.append(el('span', 'num', String(info.num)), document.createTextNode(info.title));
        const links = el('span', 'step-example-links');
        for (const link of exampleLinks(id, level)) {
            const a = el('a', null, link.label);
            a.href = link.href;
            a.dataset.example = id;
            a.dataset.tab = link.tab;
            a.setAttribute('aria-label', `${info.title}: ${link.label}`);
            links.append(a);
        }
        item.append(name, links);
        examples.append(item);
    }
    card.append(examples);

    const why = el('p', 'step-why');
    why.append(el('strong', null, index === 0 ? 'Why first: ' : 'Why next: '), document.createTextNode(step.why));
    const tryThis = el('p', 'step-try callout callout-try');
    tryThis.append(el('strong', null, 'Try this: '));
    for (const part of emphasisParts(step.tryThis)) {
        tryThis.append(part.em ? el('em', null, part.text) : document.createTextNode(part.text));
    }
    card.append(why, tryThis);
    return card;
}

// One track; with a tab bar it is a tabpanel labelled by its tab, alone it is a region labelled by its title
function renderTrack(track, level, tabbed) {
    const panel = el(tabbed ? 'div' : 'section', 'track-panel');
    panel.id = `track-${track.id}-panel`;
    const title = el('h2', 'track-title', track.title);
    title.id = `track-${track.id}-title`;
    if (tabbed) {
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', `track-${track.id}-tab`);
        panel.tabIndex = 0;
    } else {
        panel.setAttribute('aria-labelledby', title.id);
    }
    panel.append(title, el('p', 'track-intro', track.intro));
    const steps = el('ol', 'steps');
    track.steps.forEach((step, i) => steps.append(renderStep(step, i, level)));
    panel.append(steps);
    return panel;
}

// Builds the tracks into root and returns { select(id, options), fromHash() }. It registers no window
// listeners: the caller does that once and forwards to the current view, so re-rendering on a level change
// does not stack them.
export function renderLearningPath(root, tracks = TRACKS, level = DEFAULT_LEVEL) {
    const tabbed = tracks.length > 1;
    const tabs = new Map(), panels = new Map();

    function select(id, { push = false, focus = false } = {}) {
        for (const [tid, panel] of panels) {
            const on = tid === id;
            panel.hidden = !on;
            const tab = tabs.get(tid);
            if (tab) {
                tab.setAttribute('aria-selected', String(on));
                tab.tabIndex = on ? 0 : -1;
            }
        }
        if (focus) tabs.get(id)?.focus();
        if (push && location.hash !== `#${id}`) history.pushState(null, '', `#${id}`);
    }

    const tablist = el('div', 'tabs track-tabs');
    tablist.setAttribute('role', 'tablist');
    tablist.setAttribute('aria-label', 'Learning tracks');
    tracks.forEach((track, i) => {
        panels.set(track.id, renderTrack(track, level, tabbed));
        if (!tabbed) return;
        const tab = el('button', 'tab', track.title);
        tab.type = 'button';
        tab.id = `track-${track.id}-tab`;
        tab.setAttribute('role', 'tab');
        tab.setAttribute('aria-controls', `track-${track.id}-panel`);
        tab.addEventListener('click', () => select(track.id, { push: true }));
        tab.addEventListener('keydown', (event) => {
            const last = tracks.length - 1;
            const next = { ArrowRight: i === last ? 0 : i + 1, ArrowLeft: i === 0 ? last : i - 1,
                Home: 0, End: last }[event.key];
            if (next === undefined) return;
            event.preventDefault();
            select(tracks[next].id, { push: true, focus: true });
        });
        tablist.append(tab);
        tabs.set(track.id, tab);
    });

    root.replaceChildren(...(tabbed ? [tablist] : []), ...panels.values());
    const fromHash = () => select(trackFromHash(location.hash, tracks));
    fromHash();
    return { select, fromHash };
}

if (typeof document !== 'undefined') {
    const root = document.getElementById('learning-path');
    const picker = document.getElementById('level-picker');
    const intro = document.getElementById('learn-intro');
    const staticIntro = intro ? [...intro.childNodes].map(n => n.cloneNode(true)) : [];
    const pointer = document.getElementById('flagship-pointer');
    const flagshipSection = pointer?.closest('.flagship-pointer') ?? null;
    let start = null;
    let view = null;

    // Lays out the page for a level: intro, flagship pointer or the Beginner start line, and the tracks
    const render = (level) => {
        const v = levelView(level);
        if (intro) intro.replaceChildren(...(v.intro === null ? staticIntro.map(n => n.cloneNode(true)) : [v.intro]));
        if (flagshipSection) flagshipSection.hidden = !v.flagships;
        if (root && v.start && !start) {
            start = el('p', 'learn-start callout callout-try');
            start.id = 'learn-start';
            root.before(start);
        }
        if (start) {
            start.hidden = !v.start;
            start.textContent = v.start ?? '';
        }
        if (root) view = renderLearningPath(root, v.tracks, level);
    };

    if (picker) mountLevelSelector(picker);
    if (pointer) renderFlagshipPointer(pointer);
    render(getLevel());
    // Registered once; they act on whichever view the current level built
    const fromHash = () => view?.fromHash();
    window.addEventListener('hashchange', fromHash);
    window.addEventListener('popstate', fromHash);
    onLevelChange(render);
}
