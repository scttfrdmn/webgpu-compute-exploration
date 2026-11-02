// Which examples have teaching content. Titles and numbers come from the page itself; this file only
// records what exists, so the Understand/Measure tabs can say "coming soon" instead of fetching a 404.
// test/teach-content.test.mjs checks every id listed here has its files, and that they render cleanly.

// Examples with an Understand page at content/understand/<id>.md
export const UNDERSTAND = new Set([
    'boids',
    'cloth',
    'convolution-reverb',
    'dla',
    'falling-sand',
    'fft-audio',
    'game-of-life',
    'gaussian-blur',
    'kmeans',
    'matrix-multiplication',
    'maze',
    'mnist-recognize',
    'mnist-train',
    'molecular-dynamics',
    'nbody',
    'noise',
    'parallel-sort',
    'path-tracer',
    'ray-marching',
    'reaction-diffusion',
    'sph-fluid',
    'vector-addition',
    'voronoi',
    'wasm-mandelbrot',
    'wave-equation',
]);

// Examples with experiments at js/teach/experiments/<id>.js (default export: array of experiments)
export const MEASURE = new Set([
    'boids',
    'cloth',
    'convolution-reverb',
    'dla',
    'falling-sand',
    'fft-audio',
    'game-of-life',
    'gaussian-blur',
    'kmeans',
    'matrix-multiplication',
    'maze',
    'mnist-recognize',
    'mnist-train',
    'molecular-dynamics',
    'nbody',
    'noise',
    'parallel-sort',
    'path-tracer',
    'ray-marching',
    'reaction-diffusion',
    'sph-fluid',
    'vector-addition',
    'voronoi',
    'wasm-mandelbrot',
    'wave-equation',
]);

// Understand pages at the lower visitor levels (js/teach/levels.js): content/understand/<id>.<level>.md. The
// pages above (<id>.md) are the Advanced level. A level without its page falls back to the next level up.
export const UNDERSTAND_LEVELS = {
    beginner: new Set([
        'boids',
        'falling-sand',
        'game-of-life',
        'matrix-multiplication',
        'nbody',
        'parallel-sort',
        'wasm-mandelbrot',
        'wave-equation',
    ]),
    intermediate: new Set([
        'boids',
        'falling-sand',
        'game-of-life',
        'matrix-multiplication',
        'nbody',
        'parallel-sort',
        'wasm-mandelbrot',
        'wave-equation',
    ]),
};

export function hasUnderstand(id, level = 'advanced') {
    return level === 'advanced' ? UNDERSTAND.has(id) : UNDERSTAND_LEVELS[level]?.has(id) === true;
}

export function understandPath(id, level = 'advanced') {
    return level === 'advanced' ? `content/understand/${id}.md` : `content/understand/${id}.${level}.md`;
}

// The page to show for `id` at `level`: { level, path } of the first level in fallbackChain(level) that has
// one, or null when none does
export function understandSource(id, chain) {
    const level = chain.find(l => hasUnderstand(id, l));
    return level ? { level, path: understandPath(id, level) } : null;
}

export function experimentsModule(id) {
    return `./experiments/${id}.js`;
}

// Where an example's code lives, for "Read the code" links when no Understand page exists yet
export const SOURCE_BASE = 'https://github.com/scttfrdmn/webgpu-compute-exploration/blob/main/';
