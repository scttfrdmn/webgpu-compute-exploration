// Facilitator guides (#107): facilitate.html lists the sessions below and renders the selected guide
// (content/facilitate/<example>.<level>.md) with renderMarkdown. Each guide is for a teacher, parent or
// researcher running one session with one learner: goal, setup, what to say, prompts, a 20–30 minute plan,
// an observation rubric and a privacy note. No WebGPU here, nothing is sent anywhere, and the page is only
// built when this module runs in a browser. The pure helpers are exported for test/facilitate.test.mjs.

import { renderMarkdown } from './markdown.js';
import { SOURCE_BASE } from './registry.js';

// One entry per guide; `id` is the deep link (facilitate.html#nbody-beginner) and must be `${example}-${level}`
export const SESSIONS = [
    { id: 'nbody-beginner', example: 'nbody', level: 'beginner', minutes: '25–30',
        title: 'N-body: shortcuts and how to tell if they are worth it' },
    { id: 'nbody-intermediate', example: 'nbody', level: 'intermediate', minutes: '25–30',
        title: 'N-body: does a smaller time step always help?' },
    { id: 'matrix-multiplication-advanced', example: 'matrix-multiplication', level: 'advanced', minutes: '25–30',
        title: 'Matrix multiplication: testing the predictions of a reference investigation' },
];

export const LEVEL_LABELS = { beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced' };

export const facilitatePath = (session) => `content/facilitate/${session.example}.${session.level}.md`;

// The session a hash names ("#nbody-beginner"); anything else selects the first session
export function sessionFromHash(hash, sessions = SESSIONS) {
    const id = decodeURIComponent(String(hash ?? '').replace(/^#/, ''));
    return sessions.find(s => s.id === id) ?? sessions[0];
}

// Repository paths in the guides ("js/examples/nbody.js") link to the source on GitHub, as on the Understand tab
export function resolveLink(href) {
    return /^(js|test|tools|wasm|content|data)\//.test(href) ? SOURCE_BASE + href : href;
}

if (typeof document !== 'undefined') {
    const list = document.getElementById('session-list');
    const guide = document.getElementById('facilitate-guide');
    let loading = 0;

    const show = async () => {
        const session = sessionFromHash(location.hash);
        for (const a of list?.querySelectorAll('a[href^="#"]') ?? []) {
            if (a.getAttribute('href') === `#${session.id}`) a.setAttribute('aria-current', 'page');
            else a.removeAttribute('aria-current');
        }
        if (!guide) return;
        const ticket = ++loading;
        guide.setAttribute('aria-busy', 'true');
        try {
            const response = await fetch(facilitatePath(session));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const html = renderMarkdown(await response.text(), { resolveLink, idPrefix: `${session.id}-` });
            if (ticket !== loading) return;   // another session was picked while this one loaded
            const article = document.createElement('article');
            article.className = 'understand facilitate-guide';
            article.dataset.session = session.id;
            const title = document.createElement('h2');
            title.textContent = `${session.title} (${LEVEL_LABELS[session.level]}, ${session.minutes} minutes)`;
            article.append(title);
            article.insertAdjacentHTML('beforeend', html);   // renderMarkdown escapes all text and allowlists URLs
            guide.replaceChildren(article);
        } catch (error) {
            if (ticket !== loading) return;
            const p = document.createElement('p');
            p.className = 'error';
            p.textContent = `Could not load the guide: ${error.message}`;
            guide.replaceChildren(p);
        } finally {
            if (ticket === loading) guide.removeAttribute('aria-busy');
        }
    };

    window.addEventListener('hashchange', show);
    show();
}
