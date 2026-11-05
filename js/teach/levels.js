// Visitor levels (#89): Beginner (middle school), Intermediate (high school) and Advanced (college, the
// original content and the default). The level picks which Understand page is shown (js/teach/tabs.js), and
// later which controls, experiments and learning path. It is stored in localStorage and in the URL
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
