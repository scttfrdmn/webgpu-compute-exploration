// The three reference investigations featured on the landing page (index.html, "Reference investigations")
// and pointed to from the learning path (learn.html). Pure data plus a DOM renderer; no WebGPU, nothing runs at
// import time, and all text goes in through textContent.
//
// Each prediction cites claims of the experiment's interpret() by `key` (see the claim objects in
// js/teach/experiments/<id>.js). test/flagships.test.mjs runs interpret() and checks that every cited key exists
// and that each of `terms` appears in the text or prediction of those claims, so a card cannot drift from what the
// Measure tab actually tests.

import { formatRoute } from './routes.js';

export const FLAGSHIPS = [
    {
        id: 'matrix-multiplication',
        title: 'Matrix multiplication',
        question: 'How much do workgroup-memory tiling and the GPU itself buy over one CPU core for C = AB, '
            + 'and do all three get the same answer?',
        protocol: 'Sweeps n = 64 … 1,024 (or 2,048) for a naive and a tiled WGSL kernel and a single-threaded JavaScript loop on '
            + 'the same seeded matrices: 1 warm-up and 5 repeats per size, reported as medians with interquartile ranges; small GPU '
            + 'products are timed several to a pass so the timer resolves them.',
        verified: 'Every product is checked against a float64 reference on 16 sampled rows, within 4·√n·u.',
        predictions: [
            { text: 'Time grows like n³ (log-log slope 3).', experiment: 'cpu-naive-tiled', claims: ['naive-n3', 'tiled-n3'], terms: ['n³'] },
            { text: 'Tiling beats the naive kernel.', experiment: 'cpu-naive-tiled', claims: ['tiling-speedup'], terms: ['Tiling', 'faster'] },
        ],
        caveat: 'The numbers are for your device and browser; on a software GPU (as in CI) tiling can lose.',
    },
    {
        id: 'nbody',
        title: 'N-body gravity (Barnes-Hut)',
        question: 'How much accuracy does the Barnes-Hut tree give up for speed, and at what N does it beat the '
            + 'O(N²) direct sum on your GPU?',
        protocol: 'Sweeps the opening angle θ = 0.3 … 1.2, N = 1,024 … 65,536 for Barnes-Hut and brute force '
            + '(1 warm-up and 5 repeats, medians with interquartile ranges), and the time step for the energy error.',
        verified: 'Forces are compared with a float64 direct sum and with a float64 CPU replica of the tree shader on '
            + 'the same bodies.',
        predictions: [
            { text: 'Force error grows like θ².', experiment: 'theta-error', claims: ['theta-error-order'], terms: ['θ²'] },
            { text: 'Barnes-Hut overtakes brute force at some N.', experiment: 'scaling', claims: ['bh-crossover'], terms: ['Crossover', 'O(N log N)'] },
        ],
        caveat: 'Where the crossover falls depends on your GPU, driver and browser; it is not a fixed number.',
    },
    {
        id: 'parallel-sort',
        title: 'Parallel sorting',
        question: 'How do O(N log² N) bitonic sort and O(N) radix sort scale on your GPU, and at what N do they '
            + 'overtake the browser\'s built-in sort?',
        protocol: 'Sweeps N = 16,384 … 4,194,304 u32 or f32 keys; each trial sorts fresh seeded keys with three GPU '
            + 'sorts and the CPU sort: 1 warm-up and 5 repeats (3 from 2,097,152), medians with interquartile ranges.',
        verified: 'Every sort, warm-ups included, must match TypedArray.prototype.sort bit for bit.',
        predictions: [
            { text: 'Radix time ∝ N, bitonic ∝ N log² N.', experiment: 'scaling', claims: ['radix-slope', 'bitonic-slope'], terms: ['O(N · 8 passes)', 'O(N log² N)'] },
            { text: 'A crossover with the CPU sort, if your GPU has one (none in range is a valid outcome).', experiment: 'scaling',
                claims: ['crossover-radix'], terms: ['Crossover', 'CPU sort'] },
        ],
        caveat: 'One device, one browser, one run: not a ranking of GPUs. In CI, on a software GPU, the CPU sort wins.',
    },
];

// The three links of a card: [{ tab, label, href, name }]; `base` is '' on index.html, 'index.html' elsewhere
const LINKS = [['measure', 'Run it', 'Measure tab'], ['understand', 'How it works', 'Understand tab'], ['explore', 'Explore', 'interactive example']];
export function flagshipLinks(flagship, base = '') {
    return LINKS.map(([tab, label, what]) => ({
        tab, label, href: `${base}${formatRoute(flagship.id, tab)}`, name: `${label}: ${flagship.title} (${what})`,
    }));
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function linkList(flagship, base, className) {
    const nav = el('p', className);
    for (const link of flagshipLinks(flagship, base)) {
        const a = el('a', link.tab === 'measure' ? 'flagship-run' : null, link.label);
        a.href = link.href;
        a.dataset.tab = link.tab;
        a.setAttribute('aria-label', link.name);
        nav.append(a);
    }
    return nav;
}

function labelled(className, label, text) {
    const p = el('p', className);
    p.append(el('strong', null, `${label} `), document.createTextNode(text));
    return p;
}

// Full cards (landing page): <article class="flagship"> with an <h3> each
export function renderFlagships(root, { base = '', flagships = FLAGSHIPS } = {}) {
    const cards = flagships.map((f) => {
        const card = el('article', 'flagship');
        card.id = `flagship-${f.id}`;
        const heading = el('h3', null, f.title);
        heading.id = `flagship-${f.id}-title`;
        card.setAttribute('aria-labelledby', heading.id);
        card.append(heading, el('p', 'flagship-question', f.question));
        card.append(labelled('flagship-protocol', 'Protocol:', `${f.protocol} ${f.verified}`));
        const preds = el('ul', 'flagship-predictions');
        preds.setAttribute('aria-label', `Predictions tested: ${f.title}`);
        for (const p of f.predictions) preds.append(el('li', null, p.text));
        card.append(el('p', 'flagship-label', 'Tests:'), preds);
        card.append(el('p', 'hint flagship-caveat', f.caveat));
        card.append(linkList(f, base, 'flagship-links'));
        return card;
    });
    root.replaceChildren(...cards);
    return cards;
}

// Compact pointer (learning path): one line per investigation with its three links
export function renderFlagshipPointer(root, { base = 'index.html', flagships = FLAGSHIPS } = {}) {
    const list = el('ul', 'flagship-pointer-list');
    for (const f of flagships) {
        const item = el('li', 'flagship-pointer-item');
        item.append(el('strong', null, f.title), document.createTextNode(` — ${f.question}`));
        item.append(linkList(f, base, 'flagship-links'));
        list.append(item);
    }
    root.replaceChildren(list);
    return list;
}
