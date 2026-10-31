// CPU references for the noise example: 3-D Perlin and simplex noise computed by formulations
// independent of the shader's (Perlin as a weighted corner sum, simplex as a brute-force sum over a
// 5x5x5 block of skewed-lattice points), sharing only the corner hash and gradient selection.
// Precision: float64.
// Cost: perlinRef is 8 corners per sample, simplexRef 125 lattice points per sample; a 256² image
// is ~8e6 lattice evaluations for simplex (well under a second in a browser).
//
// The simplex output scale and kernel radius² (`falloff`) are parameters: the example keeps them in
// its WGSL only (the tests parse them from there).

/** PCG3D hash (Jarzynski & Olano 2020) on u32 lanes, as the shader's pcg3d. */
function pcg3d(x, y, z) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    y = (Math.imul(y, 1664525) + 1013904223) >>> 0;
    z = (Math.imul(z, 1664525) + 1013904223) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0;
    y = (y + Math.imul(z, x)) >>> 0;
    z = (z + Math.imul(x, y)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0; z = (z ^ (z >>> 16)) >>> 0;
    x = (x + Math.imul(y, z)) >>> 0;
    y = (y + Math.imul(z, x)) >>> 0;
    z = (z + Math.imul(x, y)) >>> 0;
    return [x, y, z];
}
/** Hash of integer lattice corner (i, j, k): bitcast<vec3<u32>>(vec3<i32>), first lane. */
export const hashCorner = (i, j, k) => pcg3d(i >>> 0, j >>> 0, k >>> 0)[0];

/** Improved-Perlin gradient dot product: the top 4 hash bits pick one of 12 edge gradients. */
export function grad(h, dx, dy, dz) {
    const g = h >>> 28;
    const u = g < 8 ? dx : dy;
    const v = g < 4 ? dy : (g === 12 || g === 14 ? dx : dz);
    return ((g & 1) === 0 ? u : -u) + ((g & 2) === 0 ? v : -v);
}
/** Quintic fade 6t⁵ − 15t⁴ + 10t³. */
export const fade = t => t * t * t * (t * (t * 6 - 15) + 10);

// Perlin: sum over the 8 corners, each weighted by the separable falloff w(t) = 1 - fade(|t|)
export function perlinRef(x, y, z) {
    const cx = Math.floor(x), cy = Math.floor(y), cz = Math.floor(z);
    let sum = 0;
    for (let a = 0; a <= 1; a++) for (let b = 0; b <= 1; b++) for (let c = 0; c <= 1; c++) {
        const dx = x - (cx + a), dy = y - (cy + b), dz = z - (cz + c);
        const w = (1 - fade(Math.abs(dx))) * (1 - fade(Math.abs(dy))) * (1 - fade(Math.abs(dz)));
        sum += w * grad(hashCorner(cx + a, cy + b, cz + c), dx, dy, dz);
    }
    return sum;
}

// Simplex: brute force over every skewed-lattice point in a 5x5x5 block around p; each point within
// the kernel radius contributes (falloff − r²)⁴ · grad.
export function simplexRef(x, y, z, scale, falloff) {
    const corner = (h, dx, dy, dz) => {
        const t = falloff - (dx * dx + dy * dy + dz * dz);
        if (t <= 0) return 0;
        const t2 = t * t;
        return t2 * t2 * grad(h, dx, dy, dz);
    };
    const s = (x + y + z) / 3;
    const bx = Math.floor(x + s), by = Math.floor(y + s), bz = Math.floor(z + s);
    let sum = 0;
    for (let i = bx - 2; i <= bx + 2; i++) for (let j = by - 2; j <= by + 2; j++) for (let k = bz - 2; k <= bz + 2; k++) {
        const t = (i + j + k) / 6;   // unskew the lattice point
        sum += corner(hashCorner(i, j, k), x - (i - t), y - (j - t), z - (k - t));
    }
    return scale * sum;
}
