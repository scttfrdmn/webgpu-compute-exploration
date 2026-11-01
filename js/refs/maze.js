// CPU reference for the maze example: breadth-first distances from a start cell over open passages.
// Precision: exact (integer distances; INF = unreachable).
// Cost: O(w·h) time and memory; a 2048² maze (4M cells) takes well under a second in a browser.

// Cell bit flags: the single definition of the cell encoding. js/examples/maze.js imports these
// (and generates its WGSL constants from them), so the example and this reference cannot drift apart.
export const OPEN_N = 1, OPEN_E = 2, OPEN_S = 4, OPEN_W = 8, ON_PATH = 16;
export const INF = 0xffffffff;

/** Neighbours reachable through open passages from cell c. */
export function openNeighbors(cells, w, h, c) {
    const out = [];
    if (cells[c] & OPEN_N) out.push(c - w);
    if (cells[c] & OPEN_E) out.push(c + 1);
    if (cells[c] & OPEN_S) out.push(c + w);
    if (cells[c] & OPEN_W) out.push(c - 1);
    return out;
}

/** Independent reference: queue BFS over open passages; INF = unreachable. */
export function queueBFS(cells, w, h, start) {
    const n = w * h, dist = new Uint32Array(n).fill(INF), q = new Int32Array(n);
    let head = 0, tail = 0;
    dist[start] = 0; q[tail++] = start;
    while (head < tail) {
        const c = q[head++];
        for (const nb of openNeighbors(cells, w, h, c)) {
            if (dist[nb] === INF) { dist[nb] = dist[c] + 1; q[tail++] = nb; }
        }
    }
    return dist;
}
