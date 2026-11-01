// Shared page state between js/main.js (which owns the GPU device and example instances) and the
// teaching layer (tabs, Measure). Kept in its own module so neither side imports the other.

const state = {
    gpuDevice: null,          // { device, adapter, timestamps } from getGPUDevice, once available
    deviceLost: false,
    instances: new Map(),     // canvas id -> resolved example instance (null if init failed)
    experimentResults: [],    // last result record per experiment, for the device report
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

// Keeps the most recent result record (schema v1, js/teach/result-record.js) of each experiment, keyed by
// example + experiment id
export function recordExperimentResult(record) {
    const keyOf = (r) => `${r.experiment.exampleId}/${r.experiment.id}`;
    const key = keyOf(record);
    state.experimentResults = state.experimentResults.filter(r => keyOf(r) !== key);
    state.experimentResults.push(record);
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
