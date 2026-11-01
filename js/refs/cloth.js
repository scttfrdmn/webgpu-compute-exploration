// CPU references for the cloth example: how far a particle state is from the rest lengths of its
// distance constraints (an inextensible cloth has every ratio |x_b − x_a| / rest equal to 1).
// Precision: float64 lengths from the f32 positions and rest lengths.
// Cost: O(number of constraints), about 2·10⁵ for the 128 × 128 grid.

/**
 * Stretch ratios |x_b − x_a| / rest of the constraints whose kind is in `kinds` (0 structural,
 * 1 shear, 2 bend). positions: n × 4 floats (x, y, z, inverse mass); constraintData: the
 * ArrayBuffer of 32-byte Constraint records from buildCloth (a u32, b u32, rest f32, kind u32, …).
 * Returns { count, max, mean, p99, rms } of (ratio − 1); max is the largest overstretch.
 */
export function stretchStats(positions, constraintData, kinds = [0]) {
    const u = new Uint32Array(constraintData);
    const f = new Float32Array(constraintData);
    const strains = [];
    for (let o = 0; o + 8 <= u.length; o += 8) {
        if (!kinds.includes(u[o + 3])) continue;
        const a = 4 * u[o], b = 4 * u[o + 1];
        const len = Math.hypot(positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]);
        strains.push(len / f[o + 2] - 1);
    }
    if (!strains.length) return { count: 0, max: 0, mean: 0, p99: 0, rms: 0 };
    const sorted = Float64Array.from(strains).sort();
    let sum = 0, sum2 = 0;
    for (const s of strains) { sum += s; sum2 += s * s; }
    return {
        count: strains.length,
        max: sorted[sorted.length - 1],
        mean: sum / strains.length,
        p99: sorted[Math.min(sorted.length - 1, Math.floor(0.99 * sorted.length))],
        rms: Math.sqrt(sum2 / strains.length),
    };
}
