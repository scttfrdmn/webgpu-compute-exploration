// fft-audio references shared by the CPU suite (test/fft-audio.test.mjs) and the GPU kernel
// tests (test/gpu/fft-audio.gpu.test.mjs). Nothing here touches a GPU.
// The pure references (naiveDFT, magnitudes, bands and their constants) live in js/refs/fft.js
// (deployable, for in-browser comparisons); makeViz is test-only.

import { mockDevice } from './mock-gpu.mjs';
import * as fft from '../../js/examples/fft-audio.js';

export {
    NUM_ROWS, F_LO, F_HI, SILENT_DB, DB_PER_LOG2, naiveDFT, magnitudes, bands,
} from '../../js/refs/fft.js';
export { mulberry32 } from '../../js/refs/random.js';

// A visualizer object with just the state the uniform/twiddle writers need (no canvas, no audio),
// on a recording mock device. Returns the shipped twiddle table that setFFTSize uploaded.
export function makeViz(n) {
    const device = mockDevice();
    const viz = Object.create(fft.FFTAudioVisualizer.prototype);
    viz.device = device;
    viz.paramsStride = 256;
    viz.params = new ArrayBuffer(viz.paramsStride * 12);
    viz.paramsF32 = new Float32Array(viz.params);
    viz.paramsU32 = new Uint32Array(viz.params);
    viz.twiddleBuffer = device.createBuffer({ size: 4096 * 4 });
    viz.spectrumBuffer = device.createBuffer({ size: 2049 * 8 });
    viz.samplesBuffer = device.createBuffer({ size: 4096 * 4 });
    viz.paramsBuffer = device.createBuffer({ size: 256 * 12 });
    viz.audio = { setFFTSize() {}, sampleRate: 44100 };
    viz.canvas = { width: 1234, height: 567 };
    viz.mode = 2; viz.smoothing = 0.6; viz.dbRange = 90; viz.maxDb = -3;
    viz.column = 311; viz.trigger = 77; viz.waveGain = 2.5;
    viz.setFFTSize(n);
    const tw = new Float32Array(device.lastWrite(viz.twiddleBuffer).bytes.buffer);
    return { viz, device, tw };
}
