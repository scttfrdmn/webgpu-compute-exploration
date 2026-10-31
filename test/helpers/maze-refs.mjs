// Maze references shared by test/maze.test.mjs (CPU replica) and test/gpu/maze.gpu.test.mjs
// (shipped cs_bfs_step on a software GPU). The BFS reference and cell flags live in js/refs/maze.js
// (deployable, for in-browser comparisons); fakeCanvas is test-only.

export { OPEN_N, OPEN_E, OPEN_S, OPEN_W, ON_PATH, INF, openNeighbors, queueBFS } from '../../js/refs/maze.js';

/** Minimal stand-in for the DOM canvas MazeSolver's constructor touches. */
export function fakeCanvas() {
    return {
        width: 0, height: 0,
        getContext: () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
        addEventListener() {}, removeEventListener() {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 1, height: 1 }),
    };
}
