// CPU reference for the reaction-diffusion (Gray-Scott) example: one explicit Euler step that
// convolves an explicit 3x3 Laplacian kernel with toroidally wrapped indices.
// Precision: float64 (U, V inputs may be f32; outputs are Float64Array, clamped to [0, 1]).
// Cost: O(w·h) per step (9 taps per cell); 512² × 100 steps is ~2.4e8 taps (about a second in a
// browser).
//
// The model constants are a parameter, not baked in: `c` = { DU, DV, DT, W_EDGE, W_CORNER }, where the
// Laplacian is W_CENTER·centre + W_EDGE·edges + W_CORNER·corners with W_CENTER = -1.
// js/examples/reaction-diffusion.js exports them (DU, DV, DT, W_EDGE, W_CORNER) and splices them into
// its WGSL.

export const W_CENTER = -1;

/** The 3x3 Laplacian kernel for the given edge / corner weights. */
export const laplacianKernel = (wEdge, wCorner) =>
    [[wCorner, wEdge, wCorner], [wEdge, W_CENTER, wEdge], [wCorner, wEdge, wCorner]];

/** One float64 Gray-Scott step of U, V (w x h) with feed F and kill k; returns [U2, V2]. */
export function stepReference(U, V, w, h, F, k, c) {
    const { DU, DV, DT } = c;
    const KERNEL = laplacianKernel(c.W_EDGE, c.W_CORNER);
    const U2 = new Float64Array(U.length), V2 = new Float64Array(V.length);
    const mod = (a, n) => ((a % n) + n) % n;
    // Wrapped neighbour indices, computed once per call: rows[j + 1][y] = mod(y + j, h) · w, cols[i + 1][x] = mod(x + i, w)
    const rows = [-1, 0, 1].map(j => Int32Array.from({ length: h }, (_, y) => mod(y + j, h) * w));
    const cols = [-1, 0, 1].map(i => Int32Array.from({ length: w }, (_, x) => mod(x + i, w)));
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let lu = 0, lv = 0;
        for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
            const n = rows[j][y] + cols[i][x];
            lu += KERNEL[j][i] * U[n];
            lv += KERNEL[j][i] * V[n];
        }
        const ci = y * w + x, r = U[ci] * V[ci] * V[ci];
        U2[ci] = Math.min(1, Math.max(0, U[ci] + DT * (DU * lu - r + F * (1 - U[ci]))));
        V2[ci] = Math.min(1, Math.max(0, V[ci] + DT * (DV * lv + r - (F + k) * V[ci])));
    }
    return [U2, V2];
}
