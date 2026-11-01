// Maze generation (CPU) and breadth-first solving (GPU).
//
// The point of this example is which graph algorithms map onto a GPU and which do not:
//   * Recursive backtracker (DFS) generation walks one cell at a time and each move depends on
//     every earlier move (the stack and the visited set), so it is inherently sequential.
//   * Kruskal's algorithm processes walls in one random order and asks a union-find "are these
//     two cells already connected?" before each removal; the answer depends on all earlier
//     removals. Parallel variants exist (Borůvka-style rounds with a GPU union-find and atomic
//     linking), but they no longer follow Kruskal's order and need several synchronized passes.
//     For one maze of at most ~785k cells the sequential CPU version takes tens of milliseconds,
//     so this example runs Kruskal's (and Prim's) on the CPU too.
//   * Breadth-first search, on the other hand, is a sequence of wavefronts: every cell at
//     distance d+1 can be found independently from the cells at distance d. One compute dispatch
//     per BFS level, one invocation per cell, no queue and no locks.
//   * DFS and A* solving pop one cell at a time from a stack or a priority queue, so they expose
//     no parallelism per step and are not run on the GPU here.
//
// Maze storage: one u32 per cell (cells, not a pixel grid). Bits 0-3 mark OPEN passages to the
// north, east, south and west neighbor; bit 4 marks a cell on the extracted shortest path. Both
// sides of a passage carry its bit. The render shader draws the maze as the classic
// (2w+1)×(2h+1) pixel lattice (odd/odd = cell, odd/even = wall slot, even/even = post) and
// derives every lattice pixel from the cell bitmasks, scaled by an integer factor.
//
// Distances: one u32 per cell, 0xFFFFFFFF = not reached yet, the start cell is 0.
//
// Uniform layout (struct Params, 32 bytes; shared by the compute and render shaders):
//   offset  0  size         vec2<u32>  maze width, height in cells             -> Uint32 view [0], [1]
//   offset  8  startCell    u32        index of the start cell                  -> Uint32 view [2]
//   offset 12  goalCell     u32        index of the goal cell                   -> Uint32 view [3]
//   offset 16  bfsStep      u32        BFS level expanded by this dispatch      -> Uint32 view [4]
//   offset 20  scale        u32        canvas pixels per lattice pixel          -> Uint32 view [5]
//   offset 24  solved       u32        1 = path extracted, dim the flood        -> Uint32 view [6]
//   offset 28  markerRadius f32        start/goal marker radius, canvas pixels  -> Float32 view [7]
// One copy per BFS dispatch sits at a dynamic offset of paramsStride bytes: all writeBuffer calls
// land before the submitted dispatches run, so each dispatch needs its own slot for its bfsStep.
// The render pass binds slot 0 (non-dynamic); only the per-dispatch bfsStep differs between slots.

// Compute-only use (the Measure tab, js/teach/experiments/maze.js, and test/gpu/): a MazeSolver with
// canvas = null creates no render pipeline, canvas context or mouse handlers and never draws.
// measureBFS() uploads a given maze and runs the shipped cs_bfs_step dispatches in batches (one
// command buffer each, timed), with the example's 12-byte termination readback after every batch.

import { GpuTimer } from '../gpu-timer.js';
// The cell encoding (passage bits, path bit, unreached distance) is defined once, next to the CPU
// reference that reads the same buffers; the WGSL constants below are generated from it.
import { INF, OPEN_N, OPEN_E, OPEN_S, OPEN_W, ON_PATH } from '../refs/maze.js';

export { INF, OPEN_N, OPEN_E, OPEN_S, OPEN_W, ON_PATH };

const PARAMS_SIZE = 32;
const MAX_STEPS_PER_FRAME = 512;
const READBACK_INTERVAL = 4;   // frames between termination checks while solving

const paramsStruct = `
struct Params {
    size: vec2<u32>,
    startCell: u32,
    goalCell: u32,
    bfsStep: u32,
    scale: u32,
    solved: u32,
    markerRadius: f32,
}

const INF: u32 = 0x${INF.toString(16)}u;
const OPEN_N: u32 = ${OPEN_N}u;
const OPEN_E: u32 = ${OPEN_E}u;
const OPEN_S: u32 = ${OPEN_S}u;
const OPEN_W: u32 = ${OPEN_W}u;
const ON_PATH: u32 = ${ON_PATH}u;
`;

const computeShader = paramsStruct + `
@group(0) @binding(0) var<storage, read> walls: array<u32>;
@group(0) @binding(1) var<storage, read_write> dist: array<atomic<u32>>;
// stats[0]: cells reached so far (start excluded), stats[1]: largest distance assigned so far
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 2>;
@group(0) @binding(3) var<uniform> params: Params;

// One BFS level ("pull" formulation). The dispatch for level L finds the cells at distance L+1:
// an unreached cell with an open passage to a neighbor at distance exactly L takes L+1.
// Each invocation writes only its own cell, and the only values written during this dispatch are
// L+1, never L, so reading a neighbor that is being written concurrently cannot change the result:
// the outcome is independent of invocation order and identical to a queue-based BFS.
// Dispatches in one compute pass are ordered, so level L+1 sees everything level L wrote.
// Once the goal holds a distance <= L (it was reached by an earlier dispatch) the flood stops,
// which makes the last level complete and deterministic. Atomics make the concurrent loads and
// stores well defined; stats feed the termination check (read back every few frames).
@compute @workgroup_size(8, 8)
fn cs_bfs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.size.x;
    let h = params.size.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let level = params.bfsStep;
    if (atomicLoad(&dist[params.goalCell]) <= level) {
        return;
    }
    let idx = gid.y * w + gid.x;
    if (atomicLoad(&dist[idx]) != INF) {
        return;
    }
    let open = walls[idx];
    var reached = false;
    if ((open & OPEN_N) != 0u && gid.y > 0u && atomicLoad(&dist[idx - w]) == level) {
        reached = true;
    }
    if ((open & OPEN_E) != 0u && gid.x + 1u < w && atomicLoad(&dist[idx + 1u]) == level) {
        reached = true;
    }
    if ((open & OPEN_S) != 0u && gid.y + 1u < h && atomicLoad(&dist[idx + w]) == level) {
        reached = true;
    }
    if ((open & OPEN_W) != 0u && gid.x > 0u && atomicLoad(&dist[idx - 1u]) == level) {
        reached = true;
    }
    if (reached) {
        atomicStore(&dist[idx], level + 1u);
        atomicAdd(&stats[0], 1u);
        atomicMax(&stats[1], level + 1u);
    }
}
`;

const renderShader = paramsStruct + `
@group(0) @binding(0) var<storage, read> walls: array<u32>;
@group(0) @binding(1) var<storage, read> dist: array<u32>;
@group(0) @binding(2) var<storage, read> stats: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

// One triangle that covers the whole screen; the fragment shader works in canvas pixels
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(corners[vi], 0.0, 1.0);
}

const WALL_COLOR = vec3<f32>(0.06, 0.07, 0.10);
const FLOOR_COLOR = vec3<f32>(0.78, 0.80, 0.85);
const FRONT_COLOR = vec3<f32>(0.97, 0.97, 1.00);
const PATH_COLOR = vec3<f32>(1.00, 0.84, 0.10);
const START_COLOR = vec3<f32>(0.20, 0.90, 0.40);
const GOAL_COLOR = vec3<f32>(1.00, 0.30, 0.30);

// Distance gradient: deep blue (start) -> violet -> rose (wavefront); no yellow, so the path stands out
fn distRamp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.10, 0.16, 0.55);
    let c1 = vec3<f32>(0.52, 0.16, 0.68);
    let c2 = vec3<f32>(0.93, 0.36, 0.48);
    if (t < 0.5) {
        return mix(c0, c1, t * 2.0);
    }
    return mix(c1, c2, (t - 0.5) * 2.0);
}

fn floodColor(d: u32) -> vec3<f32> {
    let maxDist = stats[1];
    if (params.solved == 0u && d == maxDist && maxDist > 0u) {
        return FRONT_COLOR;   // the current wavefront
    }
    var c = distRamp(f32(d) / f32(max(maxDist, 1u)));
    if (params.solved != 0u) {
        c *= 0.7;
    }
    return c;
}

fn cellColor(i: u32) -> vec3<f32> {
    if ((walls[i] & ON_PATH) != 0u) {
        return PATH_COLOR;
    }
    let d = dist[i];
    if (d == INF) {
        return FLOOR_COLOR;
    }
    return floodColor(d);
}

// An open passage between cells a and b: path if both are on the path, flood if both are reached
fn passageColor(a: u32, b: u32) -> vec3<f32> {
    if ((walls[a] & walls[b] & ON_PATH) != 0u) {
        return PATH_COLOR;
    }
    let da = dist[a];
    let db = dist[b];
    if (da == INF || db == INF) {
        return FLOOR_COLOR;
    }
    return floodColor(min(da, db));
}

// Disk marker with a dark outline; returns alpha in .w
fn marker(p: vec2<f32>, cell: u32, color: vec3<f32>) -> vec4<f32> {
    let w = params.size.x;
    let s = f32(params.scale);
    let cx = f32(2u * (cell % w) + 1u) + 0.5;
    let cy = f32(2u * (cell / w) + 1u) + 0.5;
    let r = length(p - vec2<f32>(cx, cy) * s);
    let radius = params.markerRadius;
    if (r < radius) {
        return vec4<f32>(color, 1.0);
    }
    if (r < radius + max(1.5, radius * 0.25)) {
        return vec4<f32>(WALL_COLOR, 1.0);
    }
    return vec4<f32>(0.0);
}

// Color of one lattice pixel: odd/odd = cell, odd/even = wall slot (open -> passage), even/even = post
fn latticeColor(lat: vec2<u32>) -> vec3<f32> {
    let w = params.size.x;
    let h = params.size.y;
    let oddX = (lat.x & 1u) == 1u;
    let oddY = (lat.y & 1u) == 1u;
    if (oddX && oddY) {
        return cellColor(((lat.y - 1u) / 2u) * w + (lat.x - 1u) / 2u);
    }
    if (oddY) {
        // Vertical wall slot between cells (cx - 1, cy) and (cx, cy); the outer border stays closed
        let cx = lat.x / 2u;
        let cy = (lat.y - 1u) / 2u;
        if (cx > 0u && cx < w) {
            let a = cy * w + cx - 1u;
            if ((walls[a] & OPEN_E) != 0u) {
                return passageColor(a, a + 1u);
            }
        }
    } else if (oddX) {
        // Horizontal wall slot between cells (cx, cy - 1) and (cx, cy)
        let cx = (lat.x - 1u) / 2u;
        let cy = lat.y / 2u;
        if (cy > 0u && cy < h) {
            let a = (cy - 1u) * w + cx;
            if ((walls[a] & OPEN_S) != 0u) {
                return passageColor(a, a + w);
            }
        }
    }
    return WALL_COLOR;
}

// Whether a lattice pixel is part of the drawn path (a path cell, or an open slot between two)
fn latticeOnPath(lat: vec2<u32>) -> bool {
    let w = params.size.x;
    let oddX = (lat.x & 1u) == 1u;
    let oddY = (lat.y & 1u) == 1u;
    if (oddX && oddY) {
        return (walls[((lat.y - 1u) / 2u) * w + (lat.x - 1u) / 2u] & ON_PATH) != 0u;
    }
    if (oddY && lat.x > 0u && lat.x < 2u * w) {
        let a = ((lat.y - 1u) / 2u) * w + lat.x / 2u - 1u;
        return (walls[a] & OPEN_E) != 0u && (walls[a] & walls[a + 1u] & ON_PATH) != 0u;
    }
    if (oddX && lat.y > 0u && lat.y < 2u * params.size.y) {
        let a = (lat.y / 2u - 1u) * w + (lat.x - 1u) / 2u;
        return (walls[a] & OPEN_S) != 0u && (walls[a] & walls[a + w] & ON_PATH) != 0u;
    }
    return false;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let latW = 2u * params.size.x + 1u;
    let latH = 2u * params.size.y + 1u;
    let lat = vec2<u32>(pos.xy) / params.scale;
    if (lat.x >= latW || lat.y >= latH) {
        return vec4<f32>(WALL_COLOR, 1.0);
    }

    let goalMark = marker(pos.xy, params.goalCell, GOAL_COLOR);
    if (goalMark.w > 0.0) {
        return vec4<f32>(goalMark.rgb, 1.0);
    }
    let startMark = marker(pos.xy, params.startCell, START_COLOR);
    if (startMark.w > 0.0) {
        return vec4<f32>(startMark.rgb, 1.0);
    }

    // At one canvas pixel per lattice pixel the 1-pixel path would vanish when the browser scales
    // the canvas down, so a solved path is drawn 3 lattice pixels wide (any 3x3 neighbor on it)
    if (params.solved != 0u && params.scale == 1u) {
        for (var dy = 0u; dy < 3u; dy++) {
            for (var dx = 0u; dx < 3u; dx++) {
                let q = lat + vec2<u32>(dx, dy);   // q - 1 is the neighbor; skip those off the lattice
                if (q.x >= 1u && q.y >= 1u && q.x - 1u < latW && q.y - 1u < latH &&
                    latticeOnPath(q - vec2<u32>(1u))) {
                    return vec4<f32>(PATH_COLOR, 1.0);
                }
            }
        }
    }
    return vec4<f32>(latticeColor(lat), 1.0);
}
`;

export const MAZE_SIZES = {
    small:  { w: 63,   h: 47,  scale: 8 },
    medium: { w: 255,  h: 191, scale: 2 },
    large:  { w: 511,  h: 383, scale: 1 },
    huge:   { w: 1023, h: 767, scale: 1 },
};

export const GENERATORS = {
    dfs:     'Recursive backtracker (DFS)',
    kruskal: "Kruskal's (union-find)",
    prim:    "Prim's (randomized)",
};

// mulberry32: a small seeded PRNG, so a maze can be rebuilt with a different braid amount
function makeRandom(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Opens the passage between neighboring cells a and b (b = a+1 or a+w)
function carve(cells, w, a, b) {
    // Same row means a horizontal neighbor; checking rows (not just b - a) keeps w = 1 correct
    const sameRow = Math.floor(a / w) === Math.floor(b / w);
    if (sameRow && b === a + 1) {
        cells[a] |= OPEN_E;
        cells[b] |= OPEN_W;
    } else if (sameRow && b === a - 1) {
        cells[a] |= OPEN_W;
        cells[b] |= OPEN_E;
    } else if (b === a + w) {
        cells[a] |= OPEN_S;
        cells[b] |= OPEN_N;
    } else {
        cells[a] |= OPEN_N;
        cells[b] |= OPEN_S;
    }
}

// Recursive backtracker with an explicit stack. Sequential by nature: each step extends the path
// from the cell the previous step reached, and backtracking depends on the whole history.
function generateDFS(cells, w, h, random) {
    const n = w * h;
    const visited = new Uint8Array(n);
    const stack = new Int32Array(n);
    const options = new Int32Array(4);
    let top = 0;
    const first = Math.floor(random() * n);
    stack[top++] = first;
    visited[first] = 1;
    while (top > 0) {
        const c = stack[top - 1];
        const x = c % w;
        const y = (c - x) / w;
        let count = 0;
        if (y > 0 && !visited[c - w]) options[count++] = c - w;
        if (x + 1 < w && !visited[c + 1]) options[count++] = c + 1;
        if (y + 1 < h && !visited[c + w]) options[count++] = c + w;
        if (x > 0 && !visited[c - 1]) options[count++] = c - 1;
        if (count === 0) {
            top--;   // dead end: backtrack
            continue;
        }
        const next = options[Math.floor(random() * count)];
        carve(cells, w, c, next);
        visited[next] = 1;
        stack[top++] = next;
    }
}

// Interior edges as (cell, direction): edge e < (w-1)·h joins cell to its east neighbor,
// the rest join a cell to its south neighbor
function edgeCells(e, w, h) {
    const eastCount = (w - 1) * h;
    if (e < eastCount) {
        const y = Math.floor(e / (w - 1));
        const a = y * w + (e - y * (w - 1));
        return [a, a + 1];
    }
    const a = e - eastCount;
    return [a, a + w];
}

// Kruskal's: shuffle all interior walls, remove a wall whenever its two cells are still in
// different trees. Union-find with path halving and union by size.
function generateKruskal(cells, w, h, random) {
    const n = w * h;
    const edgeCount = (w - 1) * h + w * (h - 1);
    const order = new Int32Array(edgeCount);
    for (let i = 0; i < edgeCount; i++) order[i] = i;
    for (let i = edgeCount - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const t = order[i]; order[i] = order[j]; order[j] = t;
    }
    const parent = new Int32Array(n);
    const size = new Int32Array(n).fill(1);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x) => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    let joined = 0;
    for (let k = 0; k < edgeCount && joined < n - 1; k++) {
        const [a, b] = edgeCells(order[k], w, h);
        let ra = find(a);
        let rb = find(b);
        if (ra === rb) continue;   // would close a loop
        if (size[ra] < size[rb]) { const t = ra; ra = rb; rb = t; }
        parent[rb] = ra;
        size[ra] += size[rb];
        carve(cells, w, a, b);
        joined++;
    }
}

// Randomized Prim's: grow one tree; repeatedly pick a random frontier cell and connect it to a
// random neighbor already in the tree
function generatePrim(cells, w, h, random) {
    const n = w * h;
    const inTree = new Uint8Array(n);
    const inFrontier = new Uint8Array(n);
    const frontier = new Int32Array(n);
    const options = new Int32Array(4);
    let count = 0;
    const addFrontier = (c) => {
        if (!inTree[c] && !inFrontier[c]) {
            inFrontier[c] = 1;
            frontier[count++] = c;
        }
    };
    const addCell = (c) => {
        inTree[c] = 1;
        const x = c % w;
        const y = (c - x) / w;
        if (y > 0) addFrontier(c - w);
        if (x + 1 < w) addFrontier(c + 1);
        if (y + 1 < h) addFrontier(c + w);
        if (x > 0) addFrontier(c - 1);
    };
    addCell(Math.floor(random() * n));
    while (count > 0) {
        const k = Math.floor(random() * count);
        const c = frontier[k];
        frontier[k] = frontier[--count];
        const x = c % w;
        const y = (c - x) / w;
        let m = 0;
        if (y > 0 && inTree[c - w]) options[m++] = c - w;
        if (x + 1 < w && inTree[c + 1]) options[m++] = c + 1;
        if (y + 1 < h && inTree[c + w]) options[m++] = c + w;
        if (x > 0 && inTree[c - 1]) options[m++] = c - 1;
        carve(cells, w, c, options[Math.floor(random() * m)]);
        addCell(c);
    }
}

// Removes each remaining interior wall with probability `braid`, adding loops (a braided maze has
// many paths, so BFS finds the shortest of several). Returns the number of walls removed.
export function braidMaze(cells, w, h, braid, seed) {
    if (braid <= 0) return 0;
    const random = makeRandom(seed ^ 0x9e3779b9);
    let removed = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const c = y * w + x;
            if (x + 1 < w && !(cells[c] & OPEN_E) && random() < braid) {
                carve(cells, w, c, c + 1);
                removed++;
            }
            if (y + 1 < h && !(cells[c] & OPEN_S) && random() < braid) {
                carve(cells, w, c, c + w);
                removed++;
            }
        }
    }
    return removed;
}

// Builds a perfect maze (a spanning tree of the grid) with the chosen algorithm, then braids it
export function generateMaze(algorithm, w, h, braid, seed) {
    const cells = new Uint32Array(w * h);
    const random = makeRandom(seed);
    if (algorithm === 'kruskal') generateKruskal(cells, w, h, random);
    else if (algorithm === 'prim') generatePrim(cells, w, h, random);
    else generateDFS(cells, w, h, random);
    const removed = braidMaze(cells, w, h, braid, seed);
    return { cells, removed };
}

// Walks from the goal back to the start, each step to an open neighbor exactly one closer.
// Every cell at distance d > 0 has such a neighbor (the one BFS reached it from), so this yields
// a shortest path of dist[goal] + 1 cells. Returns the cell indices from goal to start.
export function extractPath(cells, dist, w, h, startCell, goalCell) {
    if (dist[goalCell] === INF) return [];
    const path = [goalCell];
    let c = goalCell;
    let d = dist[c];
    while (d > 0) {
        const x = c % w;
        const open = cells[c];
        let next = -1;
        if ((open & OPEN_N) && c >= w && dist[c - w] === d - 1) next = c - w;
        else if ((open & OPEN_E) && x + 1 < w && dist[c + 1] === d - 1) next = c + 1;
        else if ((open & OPEN_S) && c + w < w * h && dist[c + w] === d - 1) next = c + w;
        else if ((open & OPEN_W) && x > 0 && dist[c - 1] === d - 1) next = c - 1;
        if (next < 0) return [];   // not a BFS distance field
        c = next;
        d--;
        path.push(c);
    }
    return c === startCell ? path : [];
}

class MazeSolver {
    constructor(device, canvas) {
        this.device = device;
        this.gpuTimer = new GpuTimer(device, { maxPasses: 2 });
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;

        this.algorithm = 'dfs';
        this.sizeKey = 'medium';
        this.braid = 0;
        this.seed = (Math.random() * 0xffffffff) >>> 0;
        this.stepsPerFrame = 16;

        this.width = 0;
        this.height = 0;
        this.cells = null;
        this.generationMs = 0;
        this.wallsRemoved = 0;

        this.resetSolveState();
        this.solveId = 0;
        this.readbackPending = false;

        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;

        // One 256-byte-aligned slot of Params per BFS dispatch in a frame (dynamic uniform offsets)
        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.params = new ArrayBuffer(this.paramsStride * MAX_STEPS_PER_FRAME);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);
        this.paramsBuffer = device.createBuffer({
            size: this.paramsStride * MAX_STEPS_PER_FRAME,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.statsBuffer = device.createBuffer({
            size: 16, // stats[0], stats[1] (+ padding)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        // Termination readback: stats[0], stats[1], dist[goal]
        this.statsStaging = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        this.setupComputePipeline();
        if (canvas) {
            this.setupRenderPipeline();
            this.setupCanvas();
            this.setupMouseControls();
        }
    }

    resetSolveState() {
        this.phase = 'idle';      // idle | solving | extracting | solved | unreachable
        this.dispatched = 0;      // BFS dispatches issued (may overshoot the goal level slightly)
        this.visited = 1;         // cells reached, from the last readback
        this.maxDist = 0;         // largest distance, from the last readback
        this.pathLength = 0;      // cells on the extracted path
        this.goalDist = INF;
        this.framesSinceReadback = 0;
    }

    setupComputePipeline() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });
        this.computeLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE,
                  buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE } }
            ]
        });
        this.bfsPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }),
            compute: { module: shaderModule, entryPoint: 'cs_bfs_step' }
        });
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });
        this.renderLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });
        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });
    }

    setupCanvas() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });
    }

    // (Re)creates the per-cell buffers and bind groups when the maze size changes
    setupBuffers(w, h) {
        if (this.wallBuffer) {
            this.wallBuffer.destroy();
            this.distBuffer.destroy();
        }
        this.width = w;
        this.height = h;
        this.cellCount = w * h;
        this.wallBuffer = this.device.createBuffer({
            size: this.cellCount * 4, // u32 open-passage bitmask per cell
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.distBuffer = this.device.createBuffer({
            size: this.cellCount * 4, // u32 BFS distance per cell
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computeLayout,
            entries: [
                { binding: 0, resource: { buffer: this.wallBuffer } },
                { binding: 1, resource: { buffer: this.distBuffer } },
                { binding: 2, resource: { buffer: this.statsBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        });
        if (!this.renderLayout) return;   // compute-only instance
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderLayout,
            entries: [
                { binding: 0, resource: { buffer: this.wallBuffer } },
                { binding: 1, resource: { buffer: this.distBuffer } },
                { binding: 2, resource: { buffer: this.statsBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        });
    }

    // Generates a new maze on the CPU, uploads it, resets the solution and draws it
    generate(newSeed = true) {
        this.pause();
        if (newSeed) this.seed = (Math.random() * 0xffffffff) >>> 0;
        const { w, h, scale } = MAZE_SIZES[this.sizeKey];
        if (w !== this.width || h !== this.height) {
            this.setupBuffers(w, h);
            this.startCell = 0;
            this.goalCell = w * h - 1;
        }
        this.scale = scale;
        this.canvas.width = (2 * w + 1) * scale;
        this.canvas.height = (2 * h + 1) * scale;

        const t0 = performance.now();
        const { cells, removed } = generateMaze(this.algorithm, w, h, this.braid, this.seed);
        this.generationMs = performance.now() - t0;
        this.cells = cells;
        this.wallsRemoved = removed;
        this.clearSolution();
    }

    // Forgets the flood and the path: dist = INF except the start, stats = 0
    clearSolution() {
        this.pause();
        this.solveId++;
        for (let i = 0; i < this.cellCount; i++) this.cells[i] &= ~ON_PATH;
        this.device.queue.writeBuffer(this.wallBuffer, 0, this.cells);
        const dist = new Uint32Array(this.cellCount).fill(INF);
        dist[this.startCell] = 0;
        this.device.queue.writeBuffer(this.distBuffer, 0, dist);
        this.device.queue.writeBuffer(this.statsBuffer, 0, new Uint32Array(4));
        this.resetSolveState();
        this.render();
    }

    setEndpoint(cell, isGoal) {
        if (isGoal) this.goalCell = cell; else this.startCell = cell;
        const wasSolving = this.running;
        this.clearSolution();
        if (wasSolving) this.solve();
    }

    setupMouseControls() {
        // The canvas is CSS-scaled, so convert client coordinates to canvas pixels, then to cells
        const toCell = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const px = (e.clientX - rect.left) * this.canvas.width / rect.width;
            const py = (e.clientY - rect.top) * this.canvas.height / rect.height;
            const lx = px / this.scale;
            const ly = py / this.scale;
            const cx = Math.min(Math.max(Math.floor((lx - 1) / 2), 0), this.width - 1);
            const cy = Math.min(Math.max(Math.floor((ly - 1) / 2), 0), this.height - 1);
            return cy * this.width + cx;
        };
        this.onMouseDown = (e) => {
            if (e.button !== 0 && e.button !== 2) return;
            e.preventDefault();
            this.setEndpoint(toCell(e), e.button === 2 || e.shiftKey);
            if (this.onStats) this.onStats();
        };
        this.onContextMenu = (e) => e.preventDefault();
        this.canvas.addEventListener('mousedown', this.onMouseDown);
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
    }

    // Writes `slots` copies of Params (one per BFS dispatch); see the layout table at the top
    writeParams(slots) {
        const wordsPerSlot = this.paramsStride / 4;
        const solved = this.phase === 'solved' ? 1 : 0;
        const markerRadius = this.canvas ? Math.max(this.scale * 1.2, this.canvas.width / 70) : 0;
        for (let k = 0; k < slots; k++) {
            const o = k * wordsPerSlot;
            this.paramsU32[o + 0] = this.width;
            this.paramsU32[o + 1] = this.height;
            this.paramsU32[o + 2] = this.startCell;
            this.paramsU32[o + 3] = this.goalCell;
            this.paramsU32[o + 4] = (this.dispatched + k) >>> 0;
            this.paramsU32[o + 5] = this.scale;
            this.paramsU32[o + 6] = solved;
            this.paramsF32[o + 7] = markerRadius;
        }
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params, 0, slots * this.paramsStride);
    }

    // Runs `steps` BFS levels (one dispatch each), then draws
    frame(steps) {
        const passes = Math.min(Math.max(steps, 0), MAX_STEPS_PER_FRAME);
        this.writeParams(Math.max(passes, 1));
        const commandEncoder = this.device.createCommandEncoder();
        if (passes > 0) {
            const computePass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('bfs'));
            computePass.setPipeline(this.bfsPipeline);
            const wgX = Math.ceil(this.width / 8);   // at most 128 x 96 workgroups
            const wgY = Math.ceil(this.height / 8);
            for (let i = 0; i < passes; i++) {
                computePass.setBindGroup(0, this.computeBindGroup, [i * this.paramsStride]);
                computePass.dispatchWorkgroups(wgX, wgY);
            }
            computePass.end();
        }
        this.encodeRender(commandEncoder);
        this.gpuTimer.resolve(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        this.gpuTimer.read();
        this.dispatched += passes;
    }

    encodeRender(commandEncoder) {
        const renderPass = commandEncoder.beginRenderPass(this.gpuTimer.renderPassDescriptor({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.06, g: 0.07, b: 0.10, a: 1.0 }
            }]
        }, 'render'));
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();
    }

    render() {
        this.frame(0);
    }

    // Termination check: copies stats and dist[goal] (12 bytes) to a mappable buffer. The flood is
    // done when the goal has a distance, or when the last dispatched level reached no new cell
    // (stats[1] < dispatches issued before the copy), i.e. the goal is unreachable.
    pollTermination() {
        if (this.readbackPending) return;
        this.readbackPending = true;
        const solveId = this.solveId;
        const dispatched = this.dispatched;
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.statsBuffer, 0, this.statsStaging, 0, 8);
        commandEncoder.copyBufferToBuffer(this.distBuffer, this.goalCell * 4, this.statsStaging, 8, 4);
        this.device.queue.submit([commandEncoder.finish()]);
        this.statsStaging.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Uint32Array(this.statsStaging.getMappedRange(0, 12).slice(0));
            this.statsStaging.unmap();
            this.readbackPending = false;
            if (this.destroyed || solveId !== this.solveId || this.phase !== 'solving') return;
            this.visited = data[0] + 1;
            this.maxDist = data[1];
            if (data[2] !== INF) {
                this.goalDist = data[2];
                this.finishSolve(true);
            } else if (data[1] < dispatched) {
                this.finishSolve(false);
            }
            if (this.onStats) this.onStats();
        }).catch(() => {
            this.readbackPending = false;
        });
    }

    // Reads the distance field back once and extracts the path on the CPU (a sequential walk of
    // a few thousand cells; doing it on the GPU would be one thread working alone)
    async finishSolve(found) {
        this.running = false;
        this.fps = 0;
        if (!found) {
            this.phase = 'unreachable';
            this.render();
            return;
        }
        this.phase = 'extracting';
        const solveId = this.solveId;
        const size = this.cellCount * 4;
        const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.distBuffer, 0, staging, 0, size);
        this.device.queue.submit([commandEncoder.finish()]);
        try {
            await staging.mapAsync(GPUMapMode.READ);
        } catch {
            staging.destroy();
            return;
        }
        const dist = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        if (this.destroyed || solveId !== this.solveId) return;

        const path = extractPath(this.cells, dist, this.width, this.height, this.startCell, this.goalCell);
        for (const c of path) this.cells[c] |= ON_PATH;
        this.device.queue.writeBuffer(this.wallBuffer, 0, this.cells);
        this.pathLength = path.length;
        this.phase = 'solved';
        this.render();
        if (this.onStats) this.onStats();
    }

    // Solve: restarts after a finished solve, otherwise starts or resumes the flood
    solve() {
        if (this.phase === 'solved' || this.phase === 'unreachable') this.clearSolution();
        if (this.phase === 'extracting') return;
        this.phase = 'solving';
        this.start();
    }

    start() {
        if (this.running) return;
        if (this.phase !== 'solving') return;
        this.running = true;
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
    }

    animate() {
        if (!this.running || this.destroyed) return;

        this.frame(this.stepsPerFrame);
        if (++this.framesSinceReadback >= READBACK_INTERVAL) {
            this.framesSinceReadback = 0;
            this.pollTermination();
        }

        this.framesSinceSample++;
        const now = performance.now();
        if (now - this.lastSampleTime >= 500) {
            this.fps = this.framesSinceSample * 1000 / (now - this.lastSampleTime);
            this.framesSinceSample = 0;
            this.lastSampleTime = now;
            if (this.onStats) this.onStats();
        }

        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.running = false;
        this.destroyed = true;
        this.gpuTimer.destroy();
        if (this.canvas) {
            this.canvas.removeEventListener('mousedown', this.onMouseDown);
            this.canvas.removeEventListener('contextmenu', this.onContextMenu);
        }
        this.wallBuffer?.destroy();
        this.distBuffer?.destroy();
        this.statsBuffer.destroy();
        this.statsStaging.destroy();
        this.paramsBuffer.destroy();
    }
}

// ---- Compute-only API for the Measure tab (js/teach/experiments/maze.js) and test/gpu/ ----

export const MAX_MEASURE_CELLS = 1024 * 768;     // the largest maze size (1023 × 767) fits
export const MAX_MEASURE_LEVELS = 100000;        // BFS dispatches per measureBFS call

// Breadth-first search of the maze `cells` (Uint32Array of passage bits, width × height) from `start` to
// `goal` with the shipped cs_bfs_step pipeline: batches of `levelsPerSubmit` (1 … 512) level dispatches
// in one compute pass and command buffer, each followed by the example's termination readback (cells
// reached, largest distance, dist[goal]: 12 bytes). Stops when the goal has a distance, when a level
// adds no cell (unreachable), or after MAX_MEASURE_LEVELS dispatches. between(), if given, is awaited
// after each batch (the Measure runner passes ctx.step, which throws to cancel). Resolves to
// { width, height, start, goal, found, levels (dist[goal], or null), visited, dispatches, submits,
// gpuMs, wallMs, dist }: GPU ms summed over the batches (null without timestamp-query), wall ms summed
// over submit → readback, and the full distance field read back once at the end.
export async function measureBFS(gpuDevice, { cells, width, height, start = 0, goal = width * height - 1,
    levelsPerSubmit = 256, between = null }) {
    const n = width * height;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || n > MAX_MEASURE_CELLS) {
        throw new RangeError(`${width} × ${height} is outside 1 … ${MAX_MEASURE_CELLS} cells`);
    }
    if (cells?.length !== n) throw new RangeError(`cells has ${cells?.length} entries, expected ${n}`);
    for (const c of [start, goal]) if (!Number.isInteger(c) || c < 0 || c >= n) throw new RangeError(`cell ${c} is outside the maze`);
    const batch = Math.min(Math.max(Math.round(levelsPerSubmit), 1), MAX_STEPS_PER_FRAME);
    const device = gpuDevice.device ?? gpuDevice;
    const solver = new MazeSolver(device, null);
    try {
        solver.setupBuffers(width, height);
        Object.assign(solver, { startCell: start, goalCell: goal, scale: 1 });
        solver.cells = Uint32Array.from(cells, c => c & ~ON_PATH);
        device.queue.writeBuffer(solver.wallBuffer, 0, solver.cells);
        const init = new Uint32Array(n).fill(INF);
        init[start] = 0;
        device.queue.writeBuffer(solver.distBuffer, 0, init);
        device.queue.writeBuffer(solver.statsBuffer, 0, new Uint32Array(4));
        solver.resetSolveState();
        solver.phase = 'solving';
        await device.queue.onSubmittedWorkDone();

        let found = false, goalDist = null, data = new Uint32Array(3), submits = 0, wallMs = 0, gpuMs = 0;
        const wgX = Math.ceil(width / 8), wgY = Math.ceil(height / 8);
        while (solver.dispatched < MAX_MEASURE_LEVELS) {
            const passes = Math.min(batch, MAX_MEASURE_LEVELS - solver.dispatched);
            solver.writeParams(passes);
            const t0 = performance.now();
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass(solver.gpuTimer.computePassDescriptor('bfs'));
            pass.setPipeline(solver.bfsPipeline);
            for (let i = 0; i < passes; i++) {
                pass.setBindGroup(0, solver.computeBindGroup, [i * solver.paramsStride]);
                pass.dispatchWorkgroups(wgX, wgY);
            }
            pass.end();
            encoder.copyBufferToBuffer(solver.statsBuffer, 0, solver.statsStaging, 0, 8);
            encoder.copyBufferToBuffer(solver.distBuffer, goal * 4, solver.statsStaging, 8, 4);
            solver.gpuTimer.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            const timing = solver.gpuTimer.read();
            solver.dispatched += passes;
            submits++;
            await solver.statsStaging.mapAsync(GPUMapMode.READ);
            data = new Uint32Array(solver.statsStaging.getMappedRange(0, 12).slice(0));
            solver.statsStaging.unmap();
            wallMs += performance.now() - t0;
            const t = await timing;
            gpuMs = gpuMs === null || typeof t?.passes.bfs !== 'number' ? null : gpuMs + t.passes.bfs;
            if (data[2] !== INF) {
                found = true;
                goalDist = data[2];
                break;
            }
            if (data[1] < solver.dispatched) break;   // the last level reached no new cell: unreachable
            if (between) await between();
        }
        const staging = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(solver.distBuffer, 0, staging, 0, 4 * n);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const dist = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return {
            width, height, start, goal, found, levels: goalDist, visited: data[0] + 1,
            dispatches: solver.dispatched, submits, gpuMs, wallMs, dist,
        };
    } finally {
        solver.destroy();
    }
}

export async function initMaze(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-maze');
    const output = document.getElementById('output-maze');

    const maze = new MazeSolver(device, canvas);
    maze.generate();
    maze.onStats = () => updateOutput();

    const algorithmSelect = document.getElementById('maze-algorithm');
    const sizeSelect = document.getElementById('maze-size');
    algorithmSelect.value = maze.algorithm;
    sizeSelect.value = maze.sizeKey;

    document.getElementById('btn-maze-generate').addEventListener('click', () => {
        maze.algorithm = GENERATORS[algorithmSelect.value] ? algorithmSelect.value : 'dfs';
        maze.sizeKey = MAZE_SIZES[sizeSelect.value] ? sizeSelect.value : 'medium';
        maze.generate();
        updateOutput();
    });

    document.getElementById('btn-maze-solve').addEventListener('click', () => {
        maze.solve();
        updateOutput();
    });

    document.getElementById('btn-maze-pause').addEventListener('click', () => {
        maze.pause();
        updateOutput();
    });

    document.getElementById('btn-maze-clear').addEventListener('click', () => {
        maze.clearSolution();
        updateOutput();
    });

    // Braid: rebuilds the same base maze (same seed) with a different share of walls removed
    const braidSlider = document.getElementById('maze-braid');
    const braidValue = document.getElementById('maze-braid-value');
    braidSlider.addEventListener('input', (e) => {
        braidValue.textContent = `${Math.round(parseFloat(e.target.value))}%`;
    });
    braidSlider.addEventListener('change', (e) => {
        maze.braid = Math.min(Math.max(parseFloat(e.target.value), 0), 30) / 100;
        maze.generate(false);
        updateOutput();
    });

    // Speed slider k -> 2^k BFS levels (dispatches) per frame
    const speedSlider = document.getElementById('maze-speed');
    const speedValue = document.getElementById('maze-speed-value');
    speedSlider.addEventListener('input', (e) => {
        const k = Math.min(Math.max(parseInt(e.target.value), 0), Math.log2(MAX_STEPS_PER_FRAME));
        maze.stepsPerFrame = 2 ** k;
        speedValue.textContent = maze.stepsPerFrame;
        updateOutput();
    });

    function statusText() {
        if (maze.phase === 'solving') {
            return maze.running ? '<span class="success">Flooding (BFS)</span>' : '<span class="info">Paused</span>';
        }
        if (maze.phase === 'extracting') return '<span class="info">Extracting path</span>';
        if (maze.phase === 'solved') return '<span class="success">Solved</span>';
        if (maze.phase === 'unreachable') return '<span class="error">Goal unreachable</span>';
        return '<span class="info">Ready: press Solve</span>';
    }

    function updateOutput() {
        const w = maze.width;
        const h = maze.height;
        const solved = maze.phase === 'solved';
        const levels = solved ? maze.goalDist : maze.maxDist;

        output.innerHTML = `<span class="success">✓ Maze Ready</span>

<span class="info">Maze:</span>
• Size: ${w} × ${h} cells (${(w * h).toLocaleString()} cells, drawn as a ${2 * w + 1} × ${2 * h + 1} lattice)
• Generator: ${GENERATORS[maze.algorithm]} on the CPU in ${maze.generationMs.toFixed(1)} ms
• Braid: ${Math.round(maze.braid * 100)}% of the remaining walls removed (${maze.wallsRemoved.toLocaleString()} walls, ${maze.wallsRemoved ? 'with loops' : 'a perfect maze: exactly one path between any two cells'})
• Status: ${statusText()}

<span class="info">Solving (GPU breadth-first wavefront):</span>
• BFS levels: ${levels.toLocaleString()} (${maze.dispatched.toLocaleString()} dispatches issued, ${maze.stepsPerFrame} per frame)
• Cells visited: ${maze.visited.toLocaleString()} of ${(w * h).toLocaleString()} (${(100 * maze.visited / (w * h)).toFixed(1)}%)
• Path length: ${solved ? `${maze.pathLength.toLocaleString()} cells (${maze.goalDist.toLocaleString()} moves, a shortest path)` : '—'}

<span class="info">Which algorithms parallelize:</span>
• BFS: one dispatch per level; an unreached cell next to a cell at distance d takes d+1. One thread per cell, no queue, no locks
• Recursive backtracker: each move continues from the previous one (stack + visited set), so it is sequential — CPU
• Kruskal's: each wall removal depends on the union-find state left by all earlier ones — CPU (parallel spanning-tree builders use Borůvka-style rounds instead)
• Prim's: grows one tree from a shared frontier list, one cell at a time — CPU
• DFS / A* solving: pop one cell at a time from a stack / priority queue — no per-step parallelism, not run here
• Path: one readback of the distance field, then the CPU walks from the goal to a neighbor one step closer

<span class="info">Performance:</span>
• Frame rate: ${maze.running ? maze.fps.toFixed(0) : 0} FPS
• ${maze.gpuTimer.frameText(maze.running ? maze.fps : 0, ['bfs', 'render'])}
• Termination: 12-byte readback (visited count, max distance, goal distance) every ${READBACK_INTERVAL} frames`;
    }

    updateOutput();
    return maze;
}

export { MazeSolver };   // for test/maze.test.mjs

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}
