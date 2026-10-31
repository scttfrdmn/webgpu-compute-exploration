// Deep links for the example tabs: #<example-id> (Explore), #<example-id>/understand, #<example-id>/measure.
// Pure functions so they can be tested in Node.

export const TABS = ['explore', 'understand', 'measure'];

// Returns { id, tab } for a hash like "#nbody/measure", or null if it doesn't name a known example.
export function parseRoute(hash, knownIds) {
    const raw = decodeURIComponent(String(hash ?? '').replace(/^#/, ''));
    if (!raw) return null;
    const [id, tab = 'explore', ...rest] = raw.split('/');
    if (rest.length || !knownIds.has(id) || !TABS.includes(tab)) return null;
    return { id, tab };
}

// Explore is the default, so it gets the plain anchor that the page navigation already uses.
export function formatRoute(id, tab) {
    return tab === 'explore' ? `#${id}` : `#${id}/${tab}`;
}
