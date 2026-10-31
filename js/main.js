import { checkWebGPUSupport, getGPUDevice } from './webgpu-utils.js';
import { runVectorAddition } from './examples/vector-addition.js';
import { runMatrixMultiplication } from './examples/matrix-multiplication.js';
import { runImageBlur } from './examples/image-blur.js';
import { initGameOfLife } from './examples/game-of-life.js';
import { initMolecularDynamics } from './examples/molecular-dynamics.js';
import { initSPHFluid } from './examples/sph-fluid.js';
import { initRayMarching } from './examples/ray-marching.js';
import { initBoids } from './examples/boids.js';
import { initDLA3D } from './examples/dla-3d.js';
import { initReactionDiffusion } from './examples/reaction-diffusion.js';
import { runWASMMandelbrot } from './examples/wasm-mandelbrot.js';

let gpuDevice = null;

async function init() {
    const statusDiv = document.getElementById('gpu-status');

    try {
        const isSupported = await checkWebGPUSupport();

        if (!isSupported) {
            statusDiv.className = 'status not-supported';
            statusDiv.textContent = 'WebGPU not available — use Chrome/Edge 113+, Safari 26+, or Firefox 141+ (Windows) / 147+ (Apple silicon Mac)';
            disableAllButtons();
            return;
        }

        gpuDevice = await getGPUDevice(handleDeviceLost);
        statusDiv.className = 'status supported';
        const info = gpuDevice.adapter.info || {};
        const adapterName = [info.vendor, info.architecture].filter(Boolean).join(' ') || info.description;
        statusDiv.textContent = adapterName ? `WebGPU ready · ${adapterName}` : 'WebGPU ready';

        setupEventListeners();
        initInteractiveExamplesOnVisible();
    } catch (error) {
        statusDiv.className = 'status not-supported';
        statusDiv.textContent = `WebGPU failed to initialize: ${error.message}`;
        disableAllButtons();
    }
}

function setupEventListeners() {
    document.getElementById('btn-vector-add').addEventListener('click', async () => {
        const output = document.getElementById('output-vector-add');
        const btn = document.getElementById('btn-vector-add');
        btn.disabled = true;
        try {
            const result = await runVectorAddition(gpuDevice);
            output.innerHTML = result;
        } catch (error) {
            output.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        }
        btn.disabled = false;
    });

    document.getElementById('btn-matrix-mul').addEventListener('click', async () => {
        const output = document.getElementById('output-matrix-mul');
        const btn = document.getElementById('btn-matrix-mul');
        btn.disabled = true;
        try {
            const result = await runMatrixMultiplication(gpuDevice);
            output.innerHTML = result;
        } catch (error) {
            output.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        }
        btn.disabled = false;
    });

    document.getElementById('btn-image-blur').addEventListener('click', async () => {
        const output = document.getElementById('output-image-blur');
        const btn = document.getElementById('btn-image-blur');
        btn.disabled = true;
        try {
            const result = await runImageBlur(gpuDevice);
            output.innerHTML = result;
            btn.closest('.stage').classList.add('ran');
        } catch (error) {
            output.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        }
        btn.disabled = false;
    });

    document.getElementById('btn-wasm-mandelbrot').addEventListener('click', async () => {
        const output = document.getElementById('output-wasm-mandelbrot');
        const btn = document.getElementById('btn-wasm-mandelbrot');
        btn.disabled = true;
        try {
            const result = await runWASMMandelbrot(gpuDevice);
            output.innerHTML = result;
            btn.closest('.stage').classList.add('ran');
        } catch (error) {
            output.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        }
        btn.disabled = false;
    });
}

// Interactive examples are created only when their canvas nears the viewport, and none
// start simulating until the user presses Start. Ray marching has no Start button, so it
// renders only while its canvas is visible.
function initInteractiveExamplesOnVisible() {
    const examples = {
        'canvas-gol': initGameOfLife,
        'canvas-md': initMolecularDynamics,
        'canvas-sph': initSPHFluid,
        'canvas-fractal': initRayMarching,
        'canvas-boids': initBoids,
        'canvas-dla': initDLA3D,
        'canvas-rd': initReactionDiffusion,
    };
    const instances = new Map();

    const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            const id = entry.target.id;
            if (entry.isIntersecting && !instances.has(id)) {
                instances.set(id, examples[id](gpuDevice).catch((error) => {
                    console.error(`Failed to initialize ${id}:`, error);
                }));
            }
            if (id === 'canvas-fractal' && instances.has(id)) {
                instances.get(id).then((renderer) => {
                    if (!renderer) return;
                    if (entry.isIntersecting) renderer.start(); else renderer.pause();
                });
            }
        }
    }, { rootMargin: '200px' });

    for (const id of Object.keys(examples)) {
        observer.observe(document.getElementById(id));
    }
}

// Examples hold references to the old device and have already bound their DOM listeners, so
// rebuilding them in place would duplicate handlers. Stop interaction and offer a clean reload.
function handleDeviceLost(info) {
    const statusDiv = document.getElementById('gpu-status');
    statusDiv.className = 'status not-supported';
    statusDiv.textContent = info.message ? `GPU device lost: ${info.message}` : 'GPU device lost';

    document.querySelectorAll('.example button, .example input, .example select')
        .forEach(control => control.disabled = true);

    if (!document.getElementById('btn-reload')) {
        const reload = document.createElement('button');
        reload.id = 'btn-reload';
        reload.className = 'primary';
        reload.textContent = 'Reload page';
        reload.addEventListener('click', () => location.reload());
        statusDiv.after(reload);
    }
}

function disableAllButtons() {
    document.querySelectorAll('button').forEach(btn => btn.disabled = true);
}

init();
