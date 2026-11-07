// Visitor levels (#89): Beginner (middle school), Intermediate (high school) and Advanced (college, the
// original content and the default). The level picks which Understand page is shown (js/teach/tabs.js), and
// which Explore controls and "What to look for" prompts are shown (#92), and later which experiments and
// learning path. It is stored in localStorage and in the URL
// (?level=beginner), so a teacher can share a link fixed at one level; the URL wins over storage.
//
// The functions above the line are pure (tested in Node); the ones below it touch the browser only when
// called, never at import time.

export const LEVELS = ['beginner', 'intermediate', 'advanced'];
export const DEFAULT_LEVEL = 'advanced';
export const LEVEL_INFO = {
    beginner: { label: 'Beginner', audience: 'middle school' },
    intermediate: { label: 'Intermediate', audience: 'high school' },
    advanced: { label: 'Advanced', audience: 'college' },
};
export const STORAGE_KEY = 'webgpu-level';
export const PARAM = 'level';

// A level name, or null for anything else (case-insensitive, surrounding spaces ignored)
export function parseLevel(value) {
    const v = String(value ?? '').trim().toLowerCase();
    return LEVELS.includes(v) ? v : null;
}

// The level for a page: ?level= in the query string, else the stored one, else the default
export function resolveLevel({ search = '', stored = null } = {}) {
    return parseLevel(new URLSearchParams(search).get(PARAM)) ?? parseLevel(stored) ?? DEFAULT_LEVEL;
}

// The query string for a level: the default level leaves no parameter, other parameters are kept
export function searchWithLevel(search, level) {
    const params = new URLSearchParams(search);
    if (level === DEFAULT_LEVEL) params.delete(PARAM);
    else params.set(PARAM, level);
    const s = params.toString();
    return s ? `?${s}` : '';
}

// Levels to try for content at `level`, the level itself first, then each level above it
// (beginner → intermediate → advanced): a page missing at one level falls back to the next one up
export function fallbackChain(level) {
    const i = LEVELS.indexOf(parseLevel(level) ?? DEFAULT_LEVEL);
    return LEVELS.slice(i);
}

// Explore controls (#92): a control's wrapper in index.html carries data-level="intermediate" or
// data-level="advanced", the lowest level that shows it; controls without the attribute show at every
// level. True if a control marked `controlLevel` is shown at `level` (an unknown mark shows everywhere,
// an unknown level counts as the default).
export function visibleAt(controlLevel, level) {
    const need = LEVELS.indexOf(parseLevel(controlLevel) ?? LEVELS[0]);
    return LEVELS.indexOf(parseLevel(level) ?? DEFAULT_LEVEL) >= need;
}

// "What to look for" prompts carry data-levels="beginner intermediate", the exact levels that show them
export function shownFor(levelList, level) {
    const at = parseLevel(level) ?? DEFAULT_LEVEL;
    return String(levelList ?? '').split(/\s+/).some(v => parseLevel(v) === at);
}

// The first-visit choice (#108) is offered on the catalog only to someone with no level yet: no ?level=, nothing
// stored, and no deep link to an example (#nbody/measure: they came for something specific)
export function shouldOfferLevelChoice({ search = '', stored = null, hash = '' } = {}) {
    if (parseLevel(new URLSearchParams(search).get(PARAM)) || parseLevel(stored)) return false;
    return !String(hash).replace(/^#/, '');
}

export const levelText = (level) => `${LEVEL_INFO[level].label} (${LEVEL_INFO[level].audience})`;

// ---- browser ------------------------------------------------------------------------------------------
let current = null;
const listeners = new Set();

function readStored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

// A level from the URL is also stored, so a shared ?level= link keeps its level on the pages it leads to
export function getLevel() {
    if (current === null) {
        current = resolveLevel({ search: location.search, stored: readStored() });
        if (parseLevel(new URLSearchParams(location.search).get(PARAM))) {
            try { localStorage.setItem(STORAGE_KEY, current); } catch { /* not stored: links still carry it */ }
        }
    }
    return current;
}

// Sets the level, stores it, rewrites ?level= in place (no new history entry) and tells the listeners
export function setLevel(level) {
    const next = parseLevel(level);
    if (!next || next === getLevel()) return;
    current = next;
    try { localStorage.setItem(STORAGE_KEY, next); } catch { /* private mode: the URL still carries it */ }
    history.replaceState(history.state, '', `${location.pathname}${searchWithLevel(location.search, next)}${location.hash}`);
    for (const callback of listeners) callback(next);
}

export function onLevelChange(callback) {
    listeners.add(callback);
    return () => listeners.delete(callback);
}

// The first-visit choice: Learn (Beginner or Intermediate, opening the learning path) or the technical view
// (Advanced, staying here). Any choice, or dismissing it, stores a level, so it is offered once.
export function mountFirstVisitChoice(container, { learnPage = 'learn.html' } = {}) {
    if (!shouldOfferLevelChoice({ search: location.search, stored: readStored(), hash: location.hash })) return false;
    const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text) n.textContent = text; return n; };
    const box = el('section', 'first-visit');
    box.setAttribute('aria-labelledby', 'first-visit-title');
    const title = el('h2', null, 'How would you like to start?');
    title.id = 'first-visit-title';
    const choices = el('div', 'first-visit-choices');
    const choose = (level, go) => {
        setLevel(level);
        try { localStorage.setItem(STORAGE_KEY, level); } catch { /* the URL carries it */ }
        container.hidden = true;
        box.remove();
        if (go) location.href = `${learnPage}${searchWithLevel('', level)}`;
    };
    for (const [level, text, go] of [
        ['beginner', 'Learn — Beginner (middle school)', true],
        ['intermediate', 'Learn — Intermediate (high school)', true],
        ['advanced', 'Technical exploration (college and beyond)', false],
    ]) {
        const button = el('button', level === 'advanced' ? null : 'primary', text);
        button.type = 'button';
        button.dataset.choice = level;
        button.addEventListener('click', () => choose(level, go));
        choices.append(button);
    }
    const close = el('button', 'first-visit-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', 'Close and use the technical view');
    close.addEventListener('click', () => choose('advanced', false));
    box.append(close, title, el('p', 'hint', 'Learning starts with a guided path and simpler explanations. The technical view shows '
        + 'every control and the full measurement detail. You can change this at any time with the Level menu at the top.'), choices);
    container.append(box);
    container.hidden = false;
    return true;
}

// A labelled <select> for the hero links of index.html and learn.html
export function mountLevelSelector(container) {
    const label = document.createElement('label');
    label.className = 'level-select';
    label.append('Level ');
    const select = document.createElement('select');
    select.id = 'level-select';
    for (const level of LEVELS) {
        const option = document.createElement('option');
        option.value = level;
        option.textContent = levelText(level);
        select.append(option);
    }
    select.value = getLevel();
    select.addEventListener('change', () => setLevel(select.value));
    onLevelChange((level) => { select.value = level; });
    label.append(select);
    container.append(label);
    return select;
}

// Shows or hides the level-marked Explore controls and prompts of every example. It only toggles
// `hidden`, so hidden controls keep their values and listeners. The Understand and Measure panels are
// skipped: tabs.js marks the rendered Understand article with data-level for its own use.
export function applyExploreLevel(level = getLevel(), root = document) {
    for (const el of root.querySelectorAll('section.example [data-level], section.example [data-levels]')) {
        if (el.closest('.tab-panel-understand, .tab-panel-measure')) continue;
        el.hidden = el.hasAttribute('data-levels')
            ? !shownFor(el.getAttribute('data-levels'), level)
            : !visibleAt(el.getAttribute('data-level'), level);
    }
}
