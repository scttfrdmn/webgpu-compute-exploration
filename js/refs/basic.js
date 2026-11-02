// CPU references for the matrix-multiplication and image-blur examples.
// Precision: float64 throughout (inputs may be f32 arrays; accumulation is in JS doubles).
// Cost: matmulF64 is O(M·N·K) (512³ takes about a second in a browser; keep it ≤ 512 or check a
// subset of rows with matmulRowsF64, O(rows·N·K)); matmulBlockedF32 is the Measure tab's timed CPU
// baseline (float32 storage, see its comment), not a reference; separableBlurPixel is O((2·half+1)²) per pixel, so
// spot-check pixels rather than whole images for large radii.

/** Textbook triple loop in float64, i-k-j order: A is MxN, B is NxK, both row-major; returns MxK. */
export function matmulF64(A, B, M, N, K) {
    return matmulRowsF64(A, B, M, N, K, Array.from({ length: M }, (_, i) => i));
}

/** The same product for the listed rows of A only: returns rows.length x K (row r of the result is row rows[r]). */
export function matmulRowsF64(A, B, M, N, K, rows) {
    const C = new Float64Array(rows.length * K);
    rows.forEach((i, r) => {
        for (let k = 0; k < N; k++) {
            const a = A[i * N + k];
            for (let j = 0; j < K; j++) C[r * K + j] += a * B[k * K + j];
        }
    });
    return C;
}

export const CPU_BLOCK = 64;

/**
 * The Measure tab's CPU baseline (not a reference): a cache-blocked, single-threaded JavaScript product with
 * float32 storage. A is MxN, B is NxK, both row-major Float32Arrays; returns C (MxK Float32Array, or `out`).
 *
 * Loop order: blocks of k (k0) and j (j0) of size `block`, then every row i, then k, then j. The inner j
 * loop streams contiguous memory of B and C, and the block × block piece of B (16 KB for block = 64) stays
 * in the L1/L2 cache while all M rows use it, instead of streaming all of B from memory once per row.
 *
 * Precision: each product a·b of two f32 values is exact in a JS double (24 + 24 significant bits ≤ 53);
 * the sum c + a·b is formed in double and rounded to f32 when stored into C. So every term costs one
 * rounding to f32 (like a fused multiply-add; a double rounding f32 → f64 → f32 can differ in the last
 * bit in rare ties), and each c_ij is summed in the order k = 0, 1, …, N − 1, the same order as the GPU
 * kernels. Error vs the float64 product: about √N·u typical, N·u worst case (u = 2⁻²⁴).
 *
 * What it is not: an optimized BLAS. It uses one core, no SIMD (JS has none), no register blocking and
 * no packing; a tuned multithreaded BLAS on the same CPU is typically 10–100× faster.
 */
export function matmulBlockedF32(A, B, M, N, K, { block = CPU_BLOCK, out } = {}) {
    const C = out ?? new Float32Array(M * K);
    C.fill(0);
    for (let k0 = 0; k0 < N; k0 += block) {
        const k1 = Math.min(k0 + block, N);
        for (let j0 = 0; j0 < K; j0 += block) {
            const j1 = Math.min(j0 + block, K);
            for (let i = 0; i < M; i++) {
                const ci = i * K, ai = i * N;
                for (let k = k0; k < k1; k++) {
                    const a = A[ai + k];
                    const bk = k * K;
                    for (let j = j0; j < j1; j++) C[ci + j] += a * B[bk + j];
                }
            }
        }
    }
    return C;
}

/**
 * Separable float64 Gaussian blur of RGBA pixel (x, y) of `img` (w x h, 4 floats per pixel):
 * horizontal 1-D pass, then vertical, each renormalized over the taps inside the image. Because the
 * clipped window is a rectangle, the 2-D renormalized Gaussian factors exactly into two 1-D passes.
 */
export function separableBlurPixel(img, w, h, x, y, sigma, half) {
    const g = k => Math.exp(-(k * k) / (2 * sigma * sigma));
    const out = [0, 0, 0, 0];
    let wy = 0;
    for (let ky = -half; ky <= half; ky++) {
        const sy = y + ky;
        if (sy < 0 || sy >= h) continue;
        const row = [0, 0, 0, 0];
        let wx = 0;
        for (let kx = -half; kx <= half; kx++) {
            const sx = x + kx;
            if (sx < 0 || sx >= w) continue;
            for (let c = 0; c < 4; c++) row[c] += img[(sy * w + sx) * 4 + c] * g(kx);
            wx += g(kx);
        }
        for (let c = 0; c < 4; c++) out[c] += (row[c] / wx) * g(ky);
        wy += g(ky);
    }
    return out.map(v => v / wy);
}
