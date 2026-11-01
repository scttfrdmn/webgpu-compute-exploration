// CPU reference for the k-means example: one Lloyd iteration (assign, then update means).
// Precision: float64 distances, sums and inertia (inputs may be f32).
// Cost: O(n·k) per step; 1e6 points × 16 clusters is ~1.6e7 distance evaluations (well under a
// second in a browser).

// Independent float64 Lloyd's step from centroids `cent` (interleaved x, y): nearest centroid by
// squared distance (ties to the lower index), then means; empty clusters keep their centroid.
// Returns { labels, next, inertia, counts }.
export function lloyd64Step(points, n, k, cent) {
    const labels = new Uint32Array(n);
    const sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Float64Array(k);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
        const x = points[2 * i], y = points[2 * i + 1];
        let best = 0, bestD = Infinity;
        for (let j = 0; j < k; j++) {
            const d = (x - cent[2 * j]) ** 2 + (y - cent[2 * j + 1]) ** 2;
            if (d < bestD) { bestD = d; best = j; }
        }
        labels[i] = best; sx[best] += x; sy[best] += y; cnt[best]++; inertia += bestD;
    }
    const next = Float64Array.from(cent);
    for (let j = 0; j < k; j++) if (cnt[j] > 0) { next[2 * j] = sx[j] / cnt[j]; next[2 * j + 1] = sy[j] / cnt[j]; }
    return { labels, next, inertia, counts: cnt };
}
