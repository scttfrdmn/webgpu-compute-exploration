// Float64 references shared by the CPU suite (test/*.test.mjs) and the GPU kernel tests (test/gpu/).

/** Textbook triple loop in float64, i-k-j order: A is MxN, B is NxK, both row-major; returns MxK. */
export function matmulF64(A, B, M, N, K) {
    const C = new Float64Array(M * K);
    for (let i = 0; i < M; i++)
        for (let k = 0; k < N; k++) {
            const a = A[i * N + k];
            for (let j = 0; j < K; j++) C[i * K + j] += a * B[k * K + j];
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
