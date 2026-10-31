// convolution-reverb references shared by the CPU suite (test/convolution-reverb.test.mjs) and the
// GPU kernel tests (test/gpu/convolution-reverb.gpu.test.mjs). Nothing here touches a GPU.
// The references themselves live in js/refs/convolution.js (deployable, for in-browser comparisons).

export { mulberry32, convolve, noiseIR } from '../../js/refs/convolution.js';
