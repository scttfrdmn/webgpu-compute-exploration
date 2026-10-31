// K-means references shared by test/kmeans.test.mjs and test/gpu/.

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
