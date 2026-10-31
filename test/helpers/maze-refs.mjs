// Maze references shared by test/maze.test.mjs (CPU replica) and test/gpu/maze.gpu.test.mjs
// (shipped cs_bfs_step on a software GPU).

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

/** Minimal stand-in for the DOM canvas MazeSolver's constructor touches. */
export function fakeCanvas() {
    return {
        width: 0, height: 0,
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
        addEventListener() {}, removeEventListener() {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 1, height: 1 }),
    };
}
