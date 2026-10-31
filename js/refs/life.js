// CPU reference for the Game of Life example: one generation of Conway's B3/S23 rule on a torus.
// Precision: exact (0/1 cells).
// Cost: O(w·h) per generation (8 neighbour reads per cell); a 2048² grid is ~3.4e7 reads (well
// under a second in a browser).

/** One generation of grid (w·h cells, row-major, 1 = alive, wrap-around edges); returns a new Uint32Array. */
export function lifeStep(grid, width, height) {
    const out = new Uint32Array(width * height);
    const getCell = (x, y) => {
        const wx = (x + width) % width, wy = (y + height) % height;
        return grid[wy * width + wx];
    };
    for (let gy = 0; gy < height; gy++)
        for (let gx = 0; gx < width; gx++) {
            const current = getCell(gx, gy);
            let n = 0;
            for (const [dx, dy] of [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]) n += getCell(gx + dx, gy + dy);
            out[gy * width + gx] = current === 1 ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
        }
    return out;
}
