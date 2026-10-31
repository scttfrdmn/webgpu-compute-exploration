import { checkWebGPUSupport, getGPUDevice } from './webgpu-utils.js';
import { GPU_TIME_NOTE } from './gpu-timer.js';
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
import { initWaveEquation } from './examples/wave-equation.js';
import { initFallingSand } from './examples/falling-sand.js';
import { initNoise } from './examples/noise.js';
import { initFFTAudio } from './examples/fft-audio.js';
import { initParallelSort } from './examples/parallel-sort.js';
import { initKMeans } from './examples/kmeans.js';
import { initVoronoiJFA } from './examples/voronoi-jfa.js';
import { initMaze } from './examples/maze.js';
import { initCloth } from './examples/cloth.js';
import { initPathTracer } from './examples/path-tracer.js';
import { initMNISTTrain } from './examples/mnist-train.js';
import { initMNISTRecognize } from './examples/mnist-recognize.js';
import { initNBody } from './examples/nbody.js';
import { initConvolutionReverb } from './examples/convolution-reverb.js';
import { runWASMMandelbrot } from './examples/wasm-mandelbrot.js';
import { buildDeviceReport } from './device-report.js';
import { setupTabs } from './teach/tabs.js';
import { setGPUDevice, setExampleInstance, markDeviceLost, experimentResults } from './teach/app-state.js';

let gpuDevice = null;

// State for the opt-in device report. Nothing here leaves the page unless the user copies it.
const resolvedInstances = new Map();   // canvas id -> example instance (null if init failed)
let deviceLostInfo = null;
let unavailableReason = '';
const recentWebGPUErrors = [];

// Keep the last 20 WebGPU-related console errors for the report; still log them as usual.
const originalConsoleError = console.error.bind(console);
console.error = (...args) => {
    originalConsoleError(...args);
    try {
        const text = args.map(a => (a instanceof Error ? `${a.name}: ${a.message}` : String(a))).join(' ');
        if (text.startsWith('[WebGPU]') || text.includes('WebGPU')) {
            recentWebGPUErrors.push(text);
            if (recentWebGPUErrors.length > 20) recentWebGPUErrors.shift();
        }
    } catch {
        // never let report bookkeeping break logging
    }
};

async function init() {
    const statusDiv = document.getElementById('gpu-status');
    setupDeviceReport();
    // Tabs first: a hidden Explore panel keeps its canvas from being initialized off screen
    setupTabs();

    try {
        const isSupported = await checkWebGPUSupport();

        if (!isSupported) {
            statusDiv.className = 'status not-supported';
            statusDiv.textContent = 'WebGPU not available — use Chrome/Edge 113+, Safari 26+, or Firefox 141+ (Windows) / 147+ (Apple silicon Mac)';
            unavailableReason = 'navigator.gpu is not available.';
            disableAllButtons();
            return;
        }

        gpuDevice = await getGPUDevice(handleDeviceLost);
        setGPUDevice(gpuDevice);
        statusDiv.className = 'status supported';
        const info = gpuDevice.adapter.info || {};
        const adapterName = [info.vendor, info.architecture].filter(Boolean).join(' ') || info.description;
        const timing = gpuDevice.timestamps
            ? 'GPU timestamps on (may be quantized to ~100 µs)'
            : 'GPU time not available (timestamp-query unsupported)';
        statusDiv.textContent = `${adapterName ? `WebGPU ready · ${adapterName}` : 'WebGPU ready'} · ${timing}`;
        statusDiv.title = GPU_TIME_NOTE;

        setupEventListeners();
        initInteractiveExamplesOnVisible();
    } catch (error) {
        statusDiv.className = 'status not-supported';
        statusDiv.textContent = `WebGPU failed to initialize: ${error.message}`;
        unavailableReason = `WebGPU failed to initialize: ${error.message}`;
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
        'canvas-wave': initWaveEquation,
        'canvas-sand': initFallingSand,
        'canvas-noise': initNoise,
        'canvas-fft': initFFTAudio,
        'canvas-sort': initParallelSort,
        'canvas-kmeans': initKMeans,
        'canvas-voronoi': initVoronoiJFA,
        'canvas-maze': initMaze,
        'canvas-cloth': initCloth,
        'canvas-pt': initPathTracer,
        'canvas-mnist-train': initMNISTTrain,
        'canvas-mnist-draw': initMNISTRecognize,
        'canvas-nbody': initNBody,
        'canvas-reverb': initConvolutionReverb,
    };
    const instances = new Map();

    const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            const id = entry.target.id;
            if (entry.isIntersecting && !instances.has(id)) {
                instances.set(id, examples[id](gpuDevice).catch((error) => {
                    console.error(`Failed to initialize ${id}:`, error);
                }));
                // The device report reads plain fields from resolved instances only
                instances.get(id).then((instance) => {
                    resolvedInstances.set(id, instance ?? null);
                    setExampleInstance(id, instance ?? null);
                });
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
    deviceLostInfo = { reason: info.reason, message: info.message };
    markDeviceLost();
    const statusDiv = document.getElementById('gpu-status');
    statusDiv.className = 'status not-supported';
    statusDiv.textContent = info.message ? `GPU device lost: ${info.message}` : 'GPU device lost';

    // Tabs stay usable so explanations can still be read
    document.querySelectorAll('.example button:not(.tab), .example input, .example select')
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

// The device report button stays usable: a report from a browser without WebGPU is still useful.
function disableAllButtons() {
    document.querySelectorAll('button:not(#btn-device-report):not(.tab)').forEach(btn => btn.disabled = true);
}

function currentDeviceReport() {
    return buildDeviceReport({
        adapter: gpuDevice?.adapter ?? null,
        device: gpuDevice?.device ?? null,
        instances: resolvedInstances,
        errors: recentWebGPUErrors,
        deviceLost: deviceLostInfo,
        unavailableReason,
        experiments: experimentResults(),
    });
}

// "Copy device report": builds the Markdown on demand, shows it in the <details> preview and
// copies it to the clipboard. If the clipboard API is unavailable or refused, the preview is
// opened with its text selected so the user can copy it by hand.
function setupDeviceReport() {
    const button = document.getElementById('btn-device-report');
    const details = document.getElementById('device-report-details');
    const preview = document.getElementById('device-report-preview');
    const feedback = document.getElementById('device-report-feedback');
    let feedbackTimer = 0;
    let skipToggleRefresh = false;   // opening the preview ourselves must not replace (and deselect) it

    const refreshPreview = async () => {
        try {
            preview.textContent = await currentDeviceReport();
        } catch (error) {
            preview.textContent = `Could not build the device report: ${error.message}`;
        }
        return preview.textContent;
    };

    const showFeedback = (text) => {
        feedback.textContent = text;
        clearTimeout(feedbackTimer);
        feedbackTimer = setTimeout(() => { feedback.textContent = ''; }, 2500);
    };

    details.addEventListener('toggle', () => {
        if (skipToggleRefresh) { skipToggleRefresh = false; return; }
        if (details.open) refreshPreview();
    });

    button.addEventListener('click', async () => {
        const text = await refreshPreview();
        try {
            await navigator.clipboard.writeText(text);
            showFeedback('Copied');
        } catch {
            if (!details.open) { skipToggleRefresh = true; details.open = true; }
            const range = document.createRange();
            range.selectNodeContents(preview);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            showFeedback('Clipboard unavailable — report selected, press Ctrl/Cmd+C');
        }
    });
}

init();
