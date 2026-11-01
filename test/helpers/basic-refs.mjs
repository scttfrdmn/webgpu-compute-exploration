// Float64 references shared by the CPU suite (test/*.test.mjs) and the GPU kernel tests (test/gpu/).
// The references themselves live in js/refs/basic.js (deployable, for in-browser comparisons).

export { matmulF64, separableBlurPixel } from '../../js/refs/basic.js';
