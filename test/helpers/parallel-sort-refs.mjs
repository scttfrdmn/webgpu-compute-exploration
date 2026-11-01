// Parallel-sort references shared by the CPU suite (test/parallel-sort.test.mjs) and the GPU kernel
// tests (test/gpu/parallel-sort.gpu.test.mjs).
// The references themselves live in js/refs/sort.js (deployable, for in-browser comparisons).

export {
    rng, encodeF32Key as encode, decodeF32Key as decode, edgeCases, cpuSort,
} from '../../js/refs/sort.js';
