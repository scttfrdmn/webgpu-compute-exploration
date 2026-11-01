// Which examples have teaching content. Titles and numbers come from the page itself; this file only
// records what exists, so the Understand/Measure tabs can say "coming soon" instead of fetching a 404.
// test/teach-content.test.mjs checks every id listed here has its files, and that they render cleanly.

// Examples with an Understand page at content/understand/<id>.md
export const UNDERSTAND = new Set([
    'cloth',
    'convolution-reverb',
    'dla',
    'falling-sand',
    'fft-audio',
    'kmeans',
    'maze',
    'mnist-recognize',
    'mnist-train',
    'nbody',
    'noise',
    'parallel-sort',
    'path-tracer',
    'reaction-diffusion',
    'voronoi',
    'wasm-mandelbrot',
    'wave-equation',
]);

// Examples with experiments at js/teach/experiments/<id>.js (default export: array of experiments)
export const MEASURE = new Set([
    'cloth',
    'convolution-reverb',
    'dla',
    'falling-sand',
    'fft-audio',
    'kmeans',
    'maze',
    'mnist-recognize',
    'mnist-train',
    'nbody',
    'noise',
    'parallel-sort',
    'path-tracer',
    'reaction-diffusion',
    'voronoi',
    'wasm-mandelbrot',
    'wave-equation',
]);

export function understandPath(id) {
    return `content/understand/${id}.md`;
}

export function experimentsModule(id) {
    return `./experiments/${id}.js`;
}

// Where an example's code lives, for "Read the code" links when no Understand page exists yet
export const SOURCE_BASE = 'https://github.com/scttfrdmn/webgpu-compute-exploration/blob/main/';
