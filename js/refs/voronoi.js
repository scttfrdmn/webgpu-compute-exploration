// CPU reference for the Voronoi / jump-flood example: brute-force nearest seed per pixel, used to
// count pixels the JFA assigned to a farther seed, plus the error-rate bounds the suites assert.
// Precision: float64 distances; a pixel counts as wrong only if its seed is farther than the true
// nearest by more than f32 rounding (relative 1e-6).
// Cost: O(w·h·seeds): 512² pixels × 256 seeds is ~6.7e7 distance evaluations (under a second in a
// browser); for larger images or seed counts, sample pixels.

/** float64 distance for metric 0 = Euclidean (squared), 1 = Manhattan, 2 = Chebyshev. */
export function metric64(cx, cy, sx, sy, metric) {
    const dx = Math.abs(cx - sx), dy = Math.abs(cy - sy);
    if (metric === 1) return dx + dy;
    if (metric === 2) return Math.max(dx, dy);
    return dx * dx + dy * dy;
}

/**
 * Independent reference: float64 brute force. A pixel is an error if its JFA seed is farther than
 * the true nearest by more than f32 rounding (relative 1e-6). Returns { wrong, maxExcess }.
 */
export function errorsVsBrute(ids, w, h, seeds, metric) {
    const count = seeds.length / 2;
    let wrong = 0, maxExcess = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const cx = x + 0.5, cy = y + 0.5;
        let best = Infinity;
        for (let i = 0; i < count; i++) best = Math.min(best, metric64(cx, cy, seeds[2 * i], seeds[2 * i + 1], metric));
        const j = ids[y * w + x];
        const d = j < count ? metric64(cx, cy, seeds[2 * j], seeds[2 * j + 1], metric) : Infinity;
        if (d > best * (1 + 1e-6) + 1e-9) {
            wrong++;
            const lin = metric === 0 ? Math.sqrt(d) - Math.sqrt(best) : d - best;
            maxExcess = Math.max(maxExcess, lin);
        }
    }
    return { wrong, maxExcess };
}

// Error-rate bounds asserted by both suites (fraction of pixels farther than the true nearest seed).
// Rong & Tan report well under 1% errors for JFA, fewer with JFA+1; measured rates are ~0.01% (JFA)
// and ~0.002% (JFA+1), so these leave ~5-10x headroom while index-math bugs break a large fraction.
export const MAX_ERROR_RATE = 0.001;
export const MAX_ERROR_RATE_PLUS_ONE = 0.0005;
