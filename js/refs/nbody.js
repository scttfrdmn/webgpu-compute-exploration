// CPU references for the N-body example: direct-sum gravitational forces and an error summary for
// comparing an approximate (Barnes-Hut or f32) force field against them.
// Precision: float64 throughout (positions/masses may be f32, packed x, y, z, m per body).
// Cost: directForces is O(n²): 16k bodies is ~2.7e8 pair interactions (a few seconds in a
// browser), so for larger N compare a random subset of target bodies; relErrors is O(n log n).

// Direct-sum reference (float64): softened acceleration acc (3 per body) and potential phi.
export function directForces(pos, n, eps2) {
    return directForcesAt(pos, n, eps2, Array.from({ length: n }, (_, i) => i));
}

// The same direct sum for a subset of target bodies only (O(targets · n)): acc (3 per target, in the
// order of `targets`) and phi. The Measure tab uses it to check large N without an O(n²) wait.
export function directForcesAt(pos, n, eps2, targets) {
    const m = targets.length;
    const acc = new Float64Array(3 * m), phi = new Float64Array(m);
    for (let t = 0; t < m; t++) {
        const i = targets[t];
        const xi = pos[4 * i], yi = pos[4 * i + 1], zi = pos[4 * i + 2];
        let ax = 0, ay = 0, az = 0, p = 0;
        for (let j = 0; j < n; j++) {
            if (j === i) continue;
            const dx = pos[4 * j] - xi, dy = pos[4 * j + 1] - yi, dz = pos[4 * j + 2] - zi;
            const invR = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
            const mInvR = pos[4 * j + 3] * invR;
            p -= mInvR;
            const s = mInvR * invR * invR;
            ax += dx * s; ay += dy * s; az += dz * s;
        }
        acc[3 * t] = ax; acc[3 * t + 1] = ay; acc[3 * t + 2] = az; phi[t] = p;
    }
    return { acc, phi };
}

// Per-body relative force error |acc - ref| / |ref| (both 3 per body): median, p99 and max.
export function relErrors(acc, ref, n) {
    const e = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const dx = acc[3 * i] - ref[3 * i], dy = acc[3 * i + 1] - ref[3 * i + 1], dz = acc[3 * i + 2] - ref[3 * i + 2];
        e[i] = Math.hypot(dx, dy, dz) / Math.hypot(ref[3 * i], ref[3 * i + 1], ref[3 * i + 2]);
    }
    e.sort();
    return { median: e[n >> 1], p99: e[Math.floor(0.99 * n)], max: e[n - 1] };
}
