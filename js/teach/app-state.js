// Shared page state between js/main.js (which owns the GPU device and example instances) and the
// teaching layer (tabs, Measure). Kept in its own module so neither side imports the other.

const state = {
    gpuDevice: null,          // { device, adapter, timestamps } from getGPUDevice, once available
    deviceLost: false,
    instances: new Map(),     // canvas id -> resolved example instance (null if init failed)
    experimentResults: [],    // last results per experiment, for the device report
};
const listeners = new Map();  // event name -> Set of callbacks

export function setGPUDevice(gpuDevice) {
    state.gpuDevice = gpuDevice;
    emit('device', gpuDevice);
}

export function getGPUDevice() {
    return state.gpuDevice;
}

export function markDeviceLost() {
    state.deviceLost = true;
    emit('device-lost');
}

export function isDeviceLost() {
    return state.deviceLost;
}

export function setExampleInstance(canvasId, instance) {
    state.instances.set(canvasId, instance);
}

export function getExampleInstance(canvasId) {
    return state.instances.get(canvasId);
}

export function exampleInstances() {
    return state.instances;
}

// Keeps the most recent result of each experiment (keyed by example + experiment id)
export function recordExperimentResult(result) {
    const key = `${result.exampleId}/${result.experimentId}`;
    state.experimentResults = state.experimentResults.filter(r => `${r.exampleId}/${r.experimentId}` !== key);
    state.experimentResults.push(result);
}

export function experimentResults() {
    return state.experimentResults;
}

export function on(event, callback) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(callback);
    return () => listeners.get(event).delete(callback);
}

function emit(event, payload) {
    for (const callback of listeners.get(event) ?? []) callback(payload);
}
