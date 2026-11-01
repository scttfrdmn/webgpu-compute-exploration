// Explore / Understand / Measure tabs for every example section (progressive enhancement).
//
// Each <section class="example"> keeps its <h3> and intro <p>; everything after them moves, unchanged,
// into the Explore panel, so example code that looks elements up by id keeps working. Understand and
// Measure panels are filled the first time they open. Leaving Explore stops the example's loop (via its
// own Pause/Stop button, so its UI stays in sync); nothing restarts on its own when you come back.

import { parseRoute, formatRoute, TABS } from './routes.js';
import { UNDERSTAND, MEASURE, understandPath, SOURCE_BASE } from './registry.js';
import { pauseExample } from '../example-control.js';

const LABELS = { explore: 'Explore', understand: 'Understand', measure: 'Measure' };
const sections = new Map();   // id -> { section, tabs: {tab: button}, panels: {tab: div}, current, loaded:Set }

export function setupTabs() {
    for (const section of document.querySelectorAll('section.example[id]')) {
        sections.set(section.id, buildTabs(section));
    }
    window.addEventListener('hashchange', () => applyRoute(location.hash, { scroll: true }));
    window.addEventListener('popstate', () => applyRoute(location.hash, { scroll: false }));
    applyRoute(location.hash, { scroll: true });
}

export function knownExampleIds() {
    return new Set(sections.keys());
}

function buildTabs(section) {
    const id = section.id;
    const intro = section.querySelector(':scope > p');
    const anchor = intro ?? section.querySelector(':scope > h3');

    const tablist = document.createElement('div');
    tablist.className = 'tabs';
    tablist.setAttribute('role', 'tablist');
    tablist.setAttribute('aria-label', `${section.querySelector('h3')?.textContent ?? id} views`);

    const tabs = {}, panels = {};
    for (const tab of TABS) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'tab';
        button.id = `${id}-tab-${tab}`;
        button.setAttribute('role', 'tab');
        button.setAttribute('aria-controls', `${id}-panel-${tab}`);
        button.textContent = LABELS[tab];
        if (tab === 'understand' && !UNDERSTAND.has(id)) button.classList.add('tab-soon');
        if (tab === 'measure' && !MEASURE.has(id)) button.classList.add('tab-soon');
        button.addEventListener('click', () => selectTab(id, tab, { push: true }));
        button.addEventListener('keydown', (event) => onTabKey(event, id, tab));
        tablist.append(button);
        tabs[tab] = button;

        const panel = document.createElement('div');
        panel.className = `tab-panel tab-panel-${tab}`;
        panel.id = `${id}-panel-${tab}`;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', button.id);
        panel.tabIndex = tab === 'explore' ? -1 : 0;
        panels[tab] = panel;
    }

    // Move everything after the intro into the Explore panel, preserving order and identity
    const rest = [];
    for (let node = anchor.nextSibling; node; node = node.nextSibling) rest.push(node);
    for (const node of rest) panels.explore.append(node);
    anchor.after(tablist, panels.explore, panels.understand, panels.measure);

    const entry = { section, tabs, panels, current: null, loaded: new Set(['explore']) };
    show(entry, 'explore');
    return entry;
}

function onTabKey(event, id, tab) {
    const i = TABS.indexOf(tab);
    const next = { ArrowRight: (i + 1) % TABS.length, ArrowLeft: (i + TABS.length - 1) % TABS.length,
        Home: 0, End: TABS.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    selectTab(id, TABS[next], { push: true, focus: true });
}

export function selectTab(id, tab, { push = false, focus = false } = {}) {
    const entry = sections.get(id);
    if (!entry || entry.current === tab) return;
    if (entry.current === 'explore') pauseExplore(entry.panels.explore);
    show(entry, tab);
    if (focus) entry.tabs[tab].focus();
    if (push) history.pushState(null, '', formatRoute(id, tab));
    if (!entry.loaded.has(tab)) {
        entry.loaded.add(tab);
        if (tab === 'understand') loadUnderstand(id, entry.panels.understand);
        if (tab === 'measure') loadMeasure(id, entry.panels.measure);
    }
}

function show(entry, tab) {
    for (const t of TABS) {
        const active = t === tab;
        entry.tabs[t].setAttribute('aria-selected', String(active));
        entry.tabs[t].tabIndex = active ? 0 : -1;
        entry.panels[t].hidden = !active;
    }
    entry.current = tab;
}

function applyRoute(hash, { scroll }) {
    const route = parseRoute(hash, knownExampleIds());
    if (!route) return;
    selectTab(route.id, route.tab);
    if (scroll && route.tab !== 'explore') sections.get(route.id).section.scrollIntoView({ block: 'start' });
}

// Stop a running example when its Explore panel is hidden (it is not resumed automatically when the
// user comes back to Explore; that is their choice).
export function pauseExplore(panel) {
    pauseExample(panel);
    // Lets the off-screen observer in main.js forget any pending auto-resume for this example
    panel.dispatchEvent(new CustomEvent('explorehidden', { bubbles: true }));
}

async function loadUnderstand(id, panel) {
    if (!UNDERSTAND.has(id)) {
        panel.append(soonNote('An explanation of this example is being written.'));
        return;
    }
    panel.append(statusLine('Loading…'));
    try {
        const [{ renderMarkdown }, response] = await Promise.all([
            import('./markdown.js'),
            fetch(understandPath(id)),
        ]);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        // idPrefix keeps heading ids unique across the 25 Understand pages on one page
        const html = renderMarkdown(await response.text(), { resolveLink, idPrefix: `${id}-` });
        panel.replaceChildren();
        const article = document.createElement('article');
        article.className = 'understand';
        article.innerHTML = html;   // renderMarkdown escapes all text and allowlists URLs
        panel.append(article);
    } catch (error) {
        panel.replaceChildren(statusLine(`Could not load the explanation: ${error.message}`, 'error'));
    }
}

async function loadMeasure(id, panel) {
    if (!MEASURE.has(id)) {
        panel.append(soonNote('Experiments for this example are being written.'));
        return;
    }
    panel.append(statusLine('Loading…'));
    try {
        const { mountMeasure } = await import('./measure.js');
        panel.replaceChildren();
        await mountMeasure(panel, id);
    } catch (error) {
        panel.replaceChildren(statusLine(`Could not load the experiments: ${error.message}`, 'error'));
    }
}

// Repository paths in Understand pages ("js/examples/nbody.js#L120") link to the source on GitHub
function resolveLink(href) {
    if (/^(js|test|tools|wasm|content|data)\//.test(href)) return SOURCE_BASE + href;
    return href;
}

function soonNote(text) {
    const p = document.createElement('p');
    p.className = 'hint tab-soon-note';
    p.append(`${text} Until then, the example's `);
    const a = document.createElement('a');
    a.href = `${SOURCE_BASE}js/examples/`;
    a.textContent = 'source code and comments';
    p.append(a, ' describe how it works.');
    return p;
}

function statusLine(text, cls = 'hint') {
    const p = document.createElement('p');
    p.className = cls;
    p.textContent = text;
    return p;
}
