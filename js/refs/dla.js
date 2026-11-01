// CPU reference for the 3D DLA example: geometry of an aggregate on a size³ grid of u32 cells
// (nonzero = occupied, index z·size² + y·size + x, as in js/examples/dla-3d.js).
// Precision: float64. Cost: O(size³) for the scan plus O(cells · fit points) for the mass-radius fit;
// a 64³ grid with 5,000 cells takes a few milliseconds.
//
// Mass-radius dimension: count the cells N(<r) within distance r of the seed and fit log N = D log r + c
// by least squares over r = 2, 2.5, … up to the radius of gyration R_g = √(mean r²). A compact ball gives
// D = 3; 3D DLA is about 2.5 (off-lattice simulations: 2.50 ± 0.05).

/** { cells, rg, rmax, dimension, connected } of the occupied cells, measured from `center` ([x, y, z]). */
export function aggregateStats(grid, size, center = [size >> 1, size >> 1, size >> 1]) {
    const radii = [];
    for (let i = 0; i < grid.length; i++) {
        if (!grid[i]) continue;
        const x = i % size, y = Math.floor(i / size) % size, z = Math.floor(i / (size * size));
        radii.push(Math.hypot(x - center[0], y - center[1], z - center[2]));
    }
    const cells = radii.length;
    if (cells === 0) return { cells, rg: 0, rmax: 0, dimension: null, connected: 0 };
    radii.sort((a, b) => a - b);
    const rg = Math.sqrt(radii.reduce((s, r) => s + r * r, 0) / cells);
    const xs = [], ys = [];
    let k = 0;
    for (let r = 2; r <= rg; r += 0.5) {
        while (k < cells && radii[k] < r) k++;
        if (k > 0) { xs.push(Math.log(r)); ys.push(Math.log(k)); }
    }
    let dimension = null;
    if (xs.length >= 3) {
        const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
        dimension = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
    }
    return { cells, rg, rmax: radii[cells - 1], dimension, connected: connectedFraction(grid, size, center) };
}

/** Fraction of occupied cells 26-connected to the cell at `center` (1 for a valid single-seed aggregate). */
export function connectedFraction(grid, size, center) {
    const start = center[2] * size * size + center[1] * size + center[0];
    let total = 0;
    for (let i = 0; i < grid.length; i++) if (grid[i]) total++;
    if (!grid[start]) return 0;
    const seen = new Uint8Array(grid.length);
    const stack = [start];
    seen[start] = 1;
    let reached = 0;
    while (stack.length) {
        const i = stack.pop();
        reached++;
        const x = i % size, y = Math.floor(i / size) % size, z = Math.floor(i / (size * size));
        for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx, ny = y + dy, nz = z + dz;
            if (nx < 0 || ny < 0 || nz < 0 || nx >= size || ny >= size || nz >= size) continue;
            const j = nz * size * size + ny * size + nx;
            if (grid[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
        }
    }
    return reached / total;
}
