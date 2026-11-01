// Renders the learning path (learn.html) from the data in path.js: a track switcher (tabs, deep-linkable
// as learn.html#gpu / learn.html#science) and one card per step. No WebGPU here, and all data goes in
// through textContent. The pure helpers are exported for test/learn-path.test.mjs; the page is only built
// when this module runs in a browser.

import { TRACKS, EXAMPLES } from './path.js';
import { formatRoute } from './routes.js';

const CATALOG = 'index.html';
const VIEWS = [['explore', 'Explore'], ['understand', 'Understand'], ['measure', 'Measure']];

// The three links of an example: [{ tab, label, href }]
export function exampleLinks(id) {
    return VIEWS.map(([tab, label]) => ({ tab, label, href: `${CATALOG}${formatRoute(id, tab)}` }));
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

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function renderStep(step, index) {
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
        for (const link of exampleLinks(id)) {
            const a = el('a', null, link.label);
            a.href = link.href;
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

function renderTrack(track) {
    const panel = el('div', 'track-panel');
    panel.id = `track-${track.id}-panel`;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', `track-${track.id}-tab`);
    panel.tabIndex = 0;
    panel.append(el('h2', 'track-title', track.title), el('p', 'track-intro', track.intro));
    const steps = el('ol', 'steps');
    track.steps.forEach((step, i) => steps.append(renderStep(step, i)));
    panel.append(steps);
    return panel;
}

export function renderLearningPath(root, tracks = TRACKS) {
    const tablist = el('div', 'tabs track-tabs');
    tablist.setAttribute('role', 'tablist');
    tablist.setAttribute('aria-label', 'Learning tracks');
    const tabs = new Map(), panels = new Map();

    function select(id, { push = false, focus = false } = {}) {
        for (const [tid, tab] of tabs) {
            const on = tid === id;
            tab.setAttribute('aria-selected', String(on));
            tab.tabIndex = on ? 0 : -1;
            panels.get(tid).hidden = !on;
        }
        if (focus) tabs.get(id).focus();
        if (push && location.hash !== `#${id}`) history.pushState(null, '', `#${id}`);
    }

    tracks.forEach((track, i) => {
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
        panels.set(track.id, renderTrack(track));
    });

    root.replaceChildren(tablist, ...panels.values());
    const fromHash = () => select(trackFromHash(location.hash, tracks));
    window.addEventListener('hashchange', fromHash);
    window.addEventListener('popstate', fromHash);
    fromHash();
}

if (typeof document !== 'undefined') {
    const root = document.getElementById('learning-path');
    if (root) renderLearningPath(root);
}
