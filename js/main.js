import { checkWebGPUSupport, getGPUDevice } from './webgpu-utils.js';
import { runVectorAddition } from './examples/vector-addition.js';
import { runMatrixMultiplication } from './examples/matrix-multiplication.js';
import { runImageBlur } from './examples/image-blur.js';
import { initGameOfLife } from './examples/game-of-life.js';
import { initMolecularDynamics } from './examples/molecular-dynamics.js';
import { initSPHFluid } from './examples/sph-fluid.js';
import { initRayMarching } from './examples/ray-marching.js';
import { initBoids } from './examples/boids.js';
import { runWASMMandelbrot } from './examples/wasm-mandelbrot.js';

let gpuDevice = null;

async function init() {
    const statusDiv = document.getElementById('gpu-status');

    try {
        const isSupported = await checkWebGPUSupport();

        if (!isSupported) {
            statusDiv.className = 'status not-supported';
            statusDiv.innerHTML = '❌ WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, or another compatible browser.';
            disableAllButtons();
            return;
        }

        gpuDevice = await getGPUDevice();
        statusDiv.className = 'status supported';
        statusDiv.innerHTML = `✅ WebGPU is supported! Using adapter: ${gpuDevice.adapter.name || 'Unknown'}`;

        setupEventListeners();
        await initGameOfLife(gpuDevice);
        await initMolecularDynamics(gpuDevice);
        await initSPHFluid(gpuDevice);
        await initRayMarching(gpuDevice);
        await initBoids(gpuDevice);
    } catch (error) {
        statusDiv.className = 'status not-supported';
        statusDiv.innerHTML = `❌ Error initializing WebGPU: ${error.message}`;
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
        } catch (error) {
            output.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        }
        btn.disabled = false;
    });
}

function disableAllButtons() {
    document.querySelectorAll('button').forEach(btn => btn.disabled = true);
}

init();
