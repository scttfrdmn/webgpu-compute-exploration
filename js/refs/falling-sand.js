// CPU reference for the falling-sand example: what the Margolus update must conserve.
// Movement only swaps cells inside a 2x2 block, so a material's cell count changes only through a
// reaction (fire, acid, plants growing into water). These helpers count materials in a grid of u32
// cells (bits 0-3 = material id, as in js/examples/falling-sand.js) and compare two grids.
// Precision: exact (integers). Cost: O(cells).

export const MATERIAL_NAMES = ['empty', 'stone', 'sand', 'water', 'oil', 'fire', 'plant', 'acid'];

/** Cell count per material id 0 … 15 (ids above 7 should never appear). */
export function materialCounts(cells) {
    const counts = new Array(16).fill(0);
    for (let i = 0; i < cells.length; i++) counts[cells[i] & 15]++;
    return counts;
}

/** Number of cells whose material differs between two grids of the same size. */
export function changedCells(a, b) {
    let n = 0;
    for (let i = 0; i < a.length; i++) if ((a[i] & 15) !== (b[i] & 15)) n++;
    return n;
}
