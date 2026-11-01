// Falling sand: a cellular automaton with several interacting materials.
//
// Parallel update scheme (the Margolus neighborhood)
// --------------------------------------------------
// In a naive "one thread per cell" sand simulation, two grains can decide to fall into the
// same empty cell at the same time, and one of them is lost. The Margolus neighborhood avoids
// this: each step the grid is cut into non-overlapping 2×2 blocks, and ONE invocation owns a
// block. It reads its 4 cells, applies the rules, and writes the same 4 cells back. Particles
// only move by swapping places inside the block, so nothing is ever duplicated or lost, and no
// two invocations ever touch the same cell.
//
// The block grid alternates every step between offset (0,0) and (1,1). Each letter below marks
// which block owns a cell:
//
//   even step (offset 0,0)        odd step (offset 1,1)
//      x: 0 1 2 3 4 5                x: 0 1 2 3 4 5
//   y 0   A A B B C C             y 0   A B B C C D
//     1   A A B B C C               1   E F F G G H
//     2   D D E E F F               2   E F F G G H
//     3   D D E E F F               3   I J J K K L
//
// Without the shift a grain could never leave its block. On odd steps the blocks along the
// top and left edges (and right and bottom) hang off the grid; cells outside the grid read as
// solid wall and are never written. Within a block the cells are numbered
// 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right.
//
// Because blocks are exclusive, the grid is updated IN PLACE in a single storage buffer (no
// ping-pong): within one dispatch every cell is read and written by exactly one invocation, and
// WebGPU makes each dispatch's writes visible to the next dispatch. The render pass reads the
// same buffer after the last step.
//
// Randomness comes from a hash of (block coordinates, step counter): no Math.random on the GPU,
// and the same step always produces the same result.
//
// Cell encoding (one u32 per cell):
//   bits 0-3   material id
//   bits 4-7   shade: random 0-15, chosen when the cell is created; gives grains their texture
//   bits 8-15  state: fire's remaining life in steps; for water, oil and acid bit 8 is the
//              flow direction (1 = right); unused by the other materials
//
// Uniform layout (struct Params, 48 bytes). The uniform buffer holds MAX_STEPS copies of it,
// one per 256-byte slot, so each step of a frame sees its own step counter (selected with a
// dynamic offset); the render pass reads slot 0.
//   offset  0  size          vec2<u32>  grid width, height           -> Uint32 view [0], [1]
//   offset  8  brushPos      vec2<f32>  brush position in cells      -> Float32 view [2], [3]
//   offset 16  brushPrev     vec2<f32>  previous brush position      -> Float32 view [4], [5]
//   offset 24  brushRadius   f32        in cells                     -> Float32 view [6]
//   offset 28  brushMaterial u32        material id to paint         -> Uint32 view [7]
//   offset 32  brushOn       u32        1 while painting             -> Uint32 view [8]
//   offset 36  simulate      u32        0 = only apply the brush     -> Uint32 view [9]
//   offset 40  step          u32        step counter (offset, hash)  -> Uint32 view [10]
//   offset 44  paintSeed     u32        frame counter for the brush  -> Uint32 view [11]
//
// Compute-only use (the Measure tab, js/teach/experiments/falling-sand.js, and test/gpu/): a FallingSand
// with canvas = null creates no render pipeline, canvas context or mouse handlers and never draws.
// measureSand() loads a grid, runs cs_main with the shipped pipeline (the dispatches of frame(),
// without the brush) and reads the cells back, so material counts can be checked on the CPU.

import { GpuTimer } from '../gpu-timer.js';

const PARAMS_SIZE = 48;
const PARAMS_STRIDE = 256; // minUniformBufferOffsetAlignment is at most 256
const MAX_STEPS = 8;

const EMPTY = 0, STONE = 1, SAND = 2, WATER = 3, OIL = 4, FIRE = 5, PLANT = 6, ACID = 7;

const MATERIALS = [
    { id: SAND, name: 'Sand' },
    { id: WATER, name: 'Water' },
    { id: STONE, name: 'Stone' },
    { id: OIL, name: 'Oil' },
    { id: FIRE, name: 'Fire' },
    { id: PLANT, name: 'Plant' },
    { id: ACID, name: 'Acid' },
    { id: EMPTY, name: 'Eraser' },
];

const paramsStruct = `
struct Params {
    size: vec2<u32>,
    brushPos: vec2<f32>,
    brushPrev: vec2<f32>,
    brushRadius: f32,
    brushMaterial: u32,
    brushOn: u32,
    simulate: u32,
    step: u32,
    paintSeed: u32,
}

const EMPTY: u32 = 0u;
const STONE: u32 = 1u;
const SAND: u32 = 2u;
const WATER: u32 = 3u;
const OIL: u32 = 4u;
const FIRE: u32 = 5u;
const PLANT: u32 = 6u;
const ACID: u32 = 7u;
const WALL: u32 = 15u;   // outside the grid; never stored

fn materialOf(c: u32) -> u32 { return c & 15u; }
fn shadeOf(c: u32) -> u32 { return (c >> 4u) & 15u; }
fn stateOf(c: u32) -> u32 { return (c >> 8u) & 255u; }
fn makeCell(m: u32, shade: u32, state: u32) -> u32 {
    return m | ((shade & 15u) << 4u) | ((state & 255u) << 8u);
}

// PCG-style integer hash
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
`;

const computeShader = paramsStruct + `
@group(0) @binding(0) var<storage, read_write> cells: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

// Density decides who sinks: a heavier movable cell above a lighter one swaps with it.
//   FIRE 0 < EMPTY 1 < OIL 2 < WATER 3 < ACID 4 < SAND 5        STONE, PLANT, WALL never move
fn density(m: u32) -> u32 {
    switch m {
        case FIRE: { return 0u; }
        case EMPTY: { return 1u; }
        case OIL: { return 2u; }
        case WATER: { return 3u; }
        case ACID: { return 4u; }
        case SAND: { return 5u; }
        default: { return 255u; }
    }
}

fn isStatic(m: u32) -> bool { return m == STONE || m == PLANT || m == WALL; }

fn isLiquid(m: u32) -> bool { return m == WATER || m == OIL || m == ACID; }

// Chances are out of 256: (random & 255) < chance
fn nextRandom(r: ptr<function, u32>) -> u32 {
    *r = pcg(*r);
    return *r;
}
fn chance(r: ptr<function, u32>, outOf256: u32) -> bool {
    return (nextRandom(r) & 255u) < outOf256;
}

// Reaction table. When cell A shares a block with cell B, with the given chance per step
// A becomes newA and B becomes newB. A new fire gets life = fireLife + random(0..63) steps.
// Burning needs air: those rules apply only if the block also holds an empty cell, so a pool
// of oil burns from its surface down instead of all at once.
//   A      B       chance  newA    newB    fireLife  air?
//   FIRE   WATER   256     EMPTY   WATER   -         -     water puts fire out
//   FIRE   OIL      24     FIRE    FIRE    120       yes   oil burns long
//   FIRE   PLANT    32     FIRE    FIRE     60       yes   plants burn
//   ACID   STONE     6     EMPTY   EMPTY   -         -     acid eats stone slowly,
//   ACID   SAND     16     EMPTY   EMPTY   -         -     sand faster
//   ACID   PLANT    24     EMPTY   EMPTY   -         -     and plants fastest; the acid is used up
//   PLANT  WATER     1     PLANT   PLANT   -         -     plants grow slowly into water
struct Reaction {
    chance: u32,
    newA: u32,
    newB: u32,
    fireLife: u32,
    needsAir: bool,
}

fn reaction(a: u32, b: u32) -> Reaction {
    if (a == FIRE && b == WATER)  { return Reaction(256u, EMPTY, WATER, 0u,   false); }
    if (a == FIRE && b == OIL)    { return Reaction(24u,  FIRE,  FIRE,  120u, true); }
    if (a == FIRE && b == PLANT)  { return Reaction(32u,  FIRE,  FIRE,  60u,  true); }
    if (a == ACID && b == STONE)  { return Reaction(6u,   EMPTY, EMPTY, 0u,   false); }
    if (a == ACID && b == SAND)   { return Reaction(16u,  EMPTY, EMPTY, 0u,   false); }
    if (a == ACID && b == PLANT)  { return Reaction(24u,  EMPTY, EMPTY, 0u,   false); }
    if (a == PLANT && b == WATER) { return Reaction(1u,   PLANT, PLANT, 0u,   false); }
    return Reaction(0u, a, b, 0u, false);
}

// Replaces cell c by material m; keeps c if the material is unchanged (a fire that spreads
// keeps its own life). Empty cells are always stored as 0. New fire gets a random life,
// new liquids a random flow direction.
fn transform(c: u32, m: u32, fireLife: u32, r: ptr<function, u32>) -> u32 {
    if (materialOf(c) == m) { return c; }
    if (m == EMPTY) { return 0u; }
    let rnd = nextRandom(r);
    var state = 0u;
    if (m == FIRE) { state = fireLife + ((rnd >> 8u) & 63u); }
    if (isLiquid(m)) { state = (rnd >> 8u) & 1u; }
    return makeCell(m, rnd & 15u, state);
}

// Chance that "upper" (heavier) swaps with "lower" (lighter) below it
fn fallChance(upper: u32, lower: u32) -> u32 {
    if (upper == EMPTY) { return 48u; }        // fire rising slowly, so it lingers on its fuel
    if (lower == EMPTY) {
        // Falling through air. Liquids sometimes hesitate, which lets them spread sideways
        // instead; with a 100% fall chance a column of water keeps a stable zigzag edge.
        if (isLiquid(upper)) { return 192u; }
        return 256u;
    }
    return 128u;                               // sinking through a fluid is slower
}

// Chance of sliding diagonally down when the cell below is blocked
fn slideChance(m: u32) -> u32 {
    if (m == SAND) { return 128u; }
    return 256u;
}

// Chance of a sideways step (viscosity)
fn spreadChance(m: u32) -> u32 {
    switch m {
        case WATER: { return 240u; }
        case ACID: { return 200u; }
        case OIL: { return 144u; }
        default: { return 96u; }   // fire
    }
}

// The block being updated: 4 cells plus which of them already moved this step
struct Block {
    c: array<u32, 4>,
    moved: array<bool, 4>,
}

fn swapCells(b: ptr<function, Block>, i: u32, j: u32) {
    let t = (*b).c[i];
    (*b).c[i] = (*b).c[j];
    (*b).c[j] = t;
    (*b).moved[i] = true;
    (*b).moved[j] = true;
}

// Straight down: upper cell u over lower cell l
fn tryFall(b: ptr<function, Block>, u: u32, l: u32, r: ptr<function, u32>) {
    let mu = materialOf((*b).c[u]);
    let ml = materialOf((*b).c[l]);
    if (isStatic(mu) || isStatic(ml) || density(mu) <= density(ml)) { return; }
    if (!chance(r, fallChance(mu, ml))) { return; }
    swapCells(b, u, l);
}

// Diagonally down: top cell t to the opposite bottom cell g. s is below t, u is above g.
// Only when t cannot fall straight, and not through a gap between two solid corners.
// Air never slides, so fire only rises straight up.
fn trySlide(b: ptr<function, Block>, t: u32, s: u32, g: u32, u: u32, r: ptr<function, u32>) {
    if ((*b).moved[t] || (*b).moved[g]) { return; }
    let mt = materialOf((*b).c[t]);
    let ms = materialOf((*b).c[s]);
    let mg = materialOf((*b).c[g]);
    let mu = materialOf((*b).c[u]);
    if (mt == EMPTY || isStatic(mt) || isStatic(mg) || isStatic(mu)) { return; }
    if (density(mt) <= density(mg)) { return; }
    if (!isStatic(ms) && density(ms) < density(mt)) { return; }
    if (!chance(r, slideChance(mt))) { return; }
    swapCells(b, t, g);
}

// Sideways within a row (l = left cell, rt = right cell), for cells that did not fall.
// Liquids keep a flow direction (state bit 0) and keep moving that way, turning around when
// blocked; this levels pools far faster than a random walk. A liquid flows into air or under
// a lighter liquid (water pushes sideways under oil). Fire jitters randomly.
fn flowsRight(c: u32) -> bool { return (stateOf(c) & 1u) == 1u; }
fn turned(c: u32) -> u32 { return c ^ (1u << 8u); }
fn canEnter(m: u32, other: u32) -> bool { return !isStatic(other) && density(other) < density(m); }

fn trySpread(b: ptr<function, Block>, l: u32, rt: u32, r: ptr<function, u32>) {
    if ((*b).moved[l] || (*b).moved[rt]) { return; }
    let cl = (*b).c[l];
    let cr = (*b).c[rt];
    let ml = materialOf(cl);
    let mr = materialOf(cr);
    if (isLiquid(ml) && flowsRight(cl)) {
        if (!canEnter(ml, mr)) {
            (*b).c[l] = turned(cl);
        } else if (chance(r, spreadChance(ml))) {
            swapCells(b, l, rt);
            return;
        }
    }
    if (isLiquid(mr) && !flowsRight(cr)) {
        if (!canEnter(mr, ml)) {
            (*b).c[rt] = turned(cr);
        } else if (chance(r, spreadChance(mr))) {
            swapCells(b, l, rt);
            return;
        }
    }
    if ((ml == FIRE && mr == EMPTY) || (mr == FIRE && ml == EMPTY)) {
        if (chance(r, spreadChance(FIRE))) {
            swapCells(b, l, rt);
        }
    }
}

// Distance from p to the segment a-b (the brush stroke since the last frame)
fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let len2 = dot(ab, ab);
    var t = 0.0;
    if (len2 > 0.0) {
        t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    }
    return length(p - (a + ab * t));
}

// One invocation per 2x2 block. Block cells: 0 = top-left, 1 = top-right,
// 2 = bottom-left, 3 = bottom-right (y grows downward).
@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.size.x);
    let h = i32(params.size.y);
    let blocksX = params.size.x / 2u + 1u;
    let blocksY = params.size.y / 2u + 1u;
    if (gid.x >= blocksX || gid.y >= blocksY) {
        return;
    }
    let shift = i32(params.step & 1u);
    let x0 = i32(gid.x) * 2 - shift;
    let y0 = i32(gid.y) * 2 - shift;

    var rng = pcg(gid.x + pcg(gid.y + pcg(params.step)));

    // Read the block; cells outside the grid are walls
    var b: Block;
    var inside: array<bool, 4>;
    for (var k = 0u; k < 4u; k++) {
        let x = x0 + i32(k & 1u);
        let y = y0 + i32(k >> 1u);
        inside[k] = x >= 0 && x < w && y >= 0 && y < h;
        if (inside[k]) {
            b.c[k] = cells[u32(y * w + x)];
        } else {
            b.c[k] = WALL;
        }
        b.moved[k] = false;
    }

    // 1. Brush. Grains and liquids are sprayed at 50% density; stone, plant and eraser are solid.
    // The brush has its own random stream that changes every frame, so holding the brush still
    // while paused keeps filling in new cells.
    if (params.brushOn != 0u) {
        var paintRng = pcg(rng + params.paintSeed);
        let m = params.brushMaterial;
        var spray = 128u;
        if (m == STONE || m == PLANT || m == EMPTY) { spray = 256u; }
        for (var k = 0u; k < 4u; k++) {
            let p = vec2<f32>(f32(x0 + i32(k & 1u)) + 0.5, f32(y0 + i32(k >> 1u)) + 0.5);
            if (inside[k] && segmentDistance(p, params.brushPrev, params.brushPos) < params.brushRadius) {
                if (chance(&paintRng, spray)) {
                    b.c[k] = transform(0u, m, 48u, &paintRng);
                }
            }
        }
    }

    if (params.simulate != 0u) {
        // 2. Reactions between every ordered pair in the block; a cell reacts at most once
        var reacted = array<bool, 4>(false, false, false, false);
        var hasAir = false;
        for (var k = 0u; k < 4u; k++) {
            if (materialOf(b.c[k]) == EMPTY) { hasAir = true; }
        }
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                if (i == j || reacted[i] || reacted[j]) { continue; }
                let rule = reaction(materialOf(b.c[i]), materialOf(b.c[j]));
                if (rule.chance == 0u || (rule.needsAir && !hasAir) || !chance(&rng, rule.chance)) { continue; }
                b.c[i] = transform(b.c[i], rule.newA, rule.fireLife, &rng);
                b.c[j] = transform(b.c[j], rule.newB, rule.fireLife, &rng);
                reacted[i] = true;
                reacted[j] = true;
            }
        }

        // 3. Fire burns down and goes out
        for (var k = 0u; k < 4u; k++) {
            if (materialOf(b.c[k]) == FIRE) {
                let life = stateOf(b.c[k]);
                if (life <= 1u) {
                    b.c[k] = 0u;
                } else {
                    b.c[k] = makeCell(FIRE, shadeOf(b.c[k]), life - 1u);
                }
            }
        }

        // 4. Movement, by swaps only, so nothing is created or destroyed. Columns go in random
        // order to avoid a left/right bias: fall straight, else slide diagonally; cells that
        // did neither may then move sideways.
        if ((nextRandom(&rng) & 1u) == 0u) {
            tryFall(&b, 0u, 2u, &rng);
            tryFall(&b, 1u, 3u, &rng);
            trySlide(&b, 0u, 2u, 3u, 1u, &rng);
            trySlide(&b, 1u, 3u, 2u, 0u, &rng);
        } else {
            tryFall(&b, 1u, 3u, &rng);
            tryFall(&b, 0u, 2u, &rng);
            trySlide(&b, 1u, 3u, 2u, 0u, &rng);
            trySlide(&b, 0u, 2u, 3u, 1u, &rng);
        }
        trySpread(&b, 2u, 3u, &rng);
        trySpread(&b, 0u, 1u, &rng);
    }

    // Write the block back (only the cells inside the grid)
    for (var k = 0u; k < 4u; k++) {
        if (inside[k]) {
            let x = x0 + i32(k & 1u);
            let y = y0 + i32(k >> 1u);
            cells[u32(y * w + x)] = b.c[k];
        }
    }
}
`;

const renderShader = paramsStruct + `
@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle that covers the whole screen; uv (0,0) is the top-left of the grid
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let p = corners[vi];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

const GLOW_RADIUS: i32 = 2;

fn cellAt(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(params.size.x) - 1);
    let cy = clamp(y, 0, i32(params.size.y) - 1);
    return cells[u32(cy) * params.size.x + u32(cx)];
}

// Base color of a cell; shade (0-15) adds per-grain variation, fire flickers per step
fn cellColor(c: u32, x: i32, y: i32) -> vec3<f32> {
    let v = f32(shadeOf(c)) / 15.0;
    switch materialOf(c) {
        case STONE: { return mix(vec3<f32>(0.30, 0.32, 0.36), vec3<f32>(0.42, 0.43, 0.47), v); }
        case SAND:  { return mix(vec3<f32>(0.80, 0.66, 0.40), vec3<f32>(0.93, 0.80, 0.54), v); }
        case WATER: { return mix(vec3<f32>(0.11, 0.33, 0.78), vec3<f32>(0.18, 0.44, 0.88), v); }
        case OIL:   { return mix(vec3<f32>(0.28, 0.17, 0.10), vec3<f32>(0.40, 0.26, 0.14), v); }
        case PLANT: { return mix(vec3<f32>(0.14, 0.52, 0.20), vec3<f32>(0.26, 0.68, 0.27), v); }
        case ACID:  { return mix(vec3<f32>(0.52, 0.88, 0.10), vec3<f32>(0.70, 1.00, 0.30), v); }
        case FIRE: {
            let flicker = f32(pcg(u32(y) * 4096u + u32(x) + params.step * 7919u) & 255u) / 255.0;
            let heat = clamp(f32(stateOf(c)) / 120.0 + flicker * 0.45, 0.0, 1.0);
            return mix(vec3<f32>(0.80, 0.12, 0.02), vec3<f32>(1.00, 0.88, 0.40), heat);
        }
        default: {
            // Empty: a dark backdrop that fades slightly toward the bottom
            let t = f32(y) / f32(params.size.y);
            return mix(vec3<f32>(0.035, 0.040, 0.060), vec3<f32>(0.060, 0.065, 0.085), t);
        }
    }
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let cellXY = min(vec2<u32>(frag.uv * vec2<f32>(params.size)), params.size - vec2<u32>(1u));
    let x = i32(cellXY.x);
    let y = i32(cellXY.y);
    let c = cells[cellXY.y * params.size.x + cellXY.x];
    var color = cellColor(c, x, y);

    // Fire glow: non-fire cells pick up an orange tint from fire within GLOW_RADIUS cells
    if (materialOf(c) != FIRE) {
        var glow = 0.0;
        for (var dy = -GLOW_RADIUS; dy <= GLOW_RADIUS; dy++) {
            for (var dx = -GLOW_RADIUS; dx <= GLOW_RADIUS; dx++) {
                if (materialOf(cellAt(x + dx, y + dy)) == FIRE) {
                    glow += 1.0 / f32(1 + dx * dx + dy * dy);
                }
            }
        }
        color += vec3<f32>(1.0, 0.45, 0.10) * min(glow * 0.12, 0.6);
    }
    return vec4<f32>(min(color, vec3<f32>(1.0)), 1.0);
}
`;

// Sandbox scene: two stone ledges, a stone basin holding water with a slick of oil, a heap
// of sand waiting above the left ledge, a blob of water above the right one, and a plant
// patch on the basin rim. Returns one u32 per cell (row-major).
function buildSandbox(width, height) {
    const grid = new Uint32Array(width * height);
    let seed = 12345;
    const random = () => {
        seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
        return seed >>> 16;
    };
    const put = (x, y, m) => {
        if (x < 0 || x >= width || y < 0 || y >= height) return;
        if (m === EMPTY) { grid[y * width + x] = 0; return; }
        const r = random();
        const flow = (m === WATER || m === OIL || m === ACID) ? (r >> 4) & 1 : 0;
        grid[y * width + x] = m | ((r & 15) << 4) | (flow << 8);
    };
    const rect = (x0, y0, x1, y1, m) => {
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) put(x, y, m);
    };

    // Coordinates are for the 320×240 grid
    // Ledges sloping gently down toward the middle
    for (let i = 0; i < 110; i++) {
        const y = 80 + Math.floor(i / 6);
        rect(i, y, i + 1, y + 5, STONE);
        rect(width - 1 - i, y + 10, width - i, y + 15, STONE);
    }
    // Basin: walls at x 95-100 and 219-224, floor at y 226-231
    rect(95, 150, 101, 232, STONE);
    rect(219, 150, 225, 232, STONE);
    rect(101, 226, 219, 232, STONE);
    rect(101, 186, 219, 226, WATER);
    rect(101, 180, 219, 186, OIL);
    // Sand heap above the left ledge, water above the right ledge
    rect(20, 30, 70, 70, SAND);
    rect(245, 40, 295, 80, WATER);
    // Plants on the basin rims
    rect(95, 146, 101, 150, PLANT);
    rect(219, 146, 225, 150, PLANT);
    return grid;
}

class FallingSand {
    constructor(device, canvas, width, height) {
        this.device = device;
        this.gpuTimer = new GpuTimer(device, { maxPasses: 2 });
        this.canvas = canvas;
        this.width = width;
        this.height = height;
        this.cellCount = width * height;
        this.running = false;
        this.destroyed = false;

        this.stepsPerFrame = 3;
        this.material = SAND;
        this.brushPos = [0, 0];
        this.brushPrev = [0, 0];
        this.brushRadius = 6;
        this.painting = false;
        this.paintMaterial = SAND;
        this.paintFramePending = false;

        this.steps = 0;
        this.paintSeed = 0;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;

        this.params = new ArrayBuffer(PARAMS_STRIDE * MAX_STEPS);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);

        this.setupBuffers();
        this.setupComputePipeline();
        if (canvas) {   // canvas = null: compute only (see the top of the file)
            this.setupRenderPipeline();
            this.setupCanvas();
            this.setupMouseControls();
        }
    }

    setupBuffers() {
        this.cellBuffer = this.device.createBuffer({
            size: this.cellCount * 4, // one u32 per cell
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC   // COPY_SRC: measureSand() readback
        });

        this.paramsBuffer = this.device.createBuffer({
            size: PARAMS_STRIDE * MAX_STEPS,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    setupComputePipeline() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'cs_main' }
        });

        // A single cell buffer updated in place (see the Margolus notes at the top)
        this.computeBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cellBuffer } },
                { binding: 1, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        });
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });

        this.renderBindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cellBuffer } },
                { binding: 1, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        });
    }

    setupCanvas() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });
    }

    setupMouseControls() {
        // The canvas is CSS-scaled, so convert client coordinates to canvas pixels, then to cells
        const toCells = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const px = (e.clientX - rect.left) * this.canvas.width / rect.width;
            const py = (e.clientY - rect.top) * this.canvas.height / rect.height;
            return [px * this.width / this.canvas.width, py * this.height / this.canvas.height];
        };

        // Left button paints the selected material, right button erases
        this.onMouseDown = (e) => {
            if (e.button !== 0 && e.button !== 2) return;
            e.preventDefault();
            this.painting = true;
            this.paintMaterial = e.button === 2 ? EMPTY : this.material;
            this.brushPos = toCells(e);
            this.brushPrev = this.brushPos.slice();
            this.requestPaintFrame();
        };
        this.onMouseMove = (e) => {
            if (!this.painting) return;
            this.brushPos = toCells(e);
            this.requestPaintFrame();
        };
        this.onMouseUp = () => {
            this.painting = false;
        };
        this.onContextMenu = (e) => e.preventDefault();

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        this.canvas.addEventListener('mousemove', this.onMouseMove);
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
        window.addEventListener('mouseup', this.onMouseUp);
    }

    // While paused, painting still updates the grid: one brush-only pass per animation frame
    requestPaintFrame() {
        if (this.running || this.paintFramePending) return;
        this.paintFramePending = true;
        requestAnimationFrame(() => {
            this.paintFramePending = false;
            if (!this.running && !this.destroyed) this.frame(0);
        });
    }

    // Writes `count` uniform slots; see the layout table at the top of the file.
    // Slot k drives step k of this frame; only the first step applies the brush.
    writeParams(count, simulate, brushOn) {
        for (let k = 0; k < count; k++) {
            const base = k * PARAMS_STRIDE / 4;
            this.paramsU32[base + 0] = this.width;
            this.paramsU32[base + 1] = this.height;
            this.paramsF32[base + 2] = this.brushPos[0];
            this.paramsF32[base + 3] = this.brushPos[1];
            this.paramsF32[base + 4] = this.brushPrev[0];
            this.paramsF32[base + 5] = this.brushPrev[1];
            this.paramsF32[base + 6] = this.brushRadius;
            this.paramsU32[base + 7] = this.paintMaterial;
            this.paramsU32[base + 8] = brushOn && k === 0 ? 1 : 0;
            this.paramsU32[base + 9] = simulate ? 1 : 0;
            this.paramsU32[base + 10] = (this.steps + (simulate ? k : 0)) >>> 0;
            this.paramsU32[base + 11] = this.paintSeed;
        }
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params, 0, count * PARAMS_STRIDE);
    }

    loadGrid(grid) {
        this.device.queue.writeBuffer(this.cellBuffer, 0, grid);
        this.steps = 0;
        this.render();
    }

    sandbox() {
        this.loadGrid(buildSandbox(this.width, this.height));
    }

    clear() {
        this.loadGrid(new Uint32Array(this.cellCount));
    }

    // Runs `stepCount` simulation steps (or one brush-only pass if 0), then draws
    frame(stepCount) {
        const brushOn = this.painting;
        const simulate = stepCount > 0;
        const passes = simulate ? Math.min(stepCount, MAX_STEPS) : (brushOn ? 1 : 0);
        this.writeParams(Math.max(passes, 1), simulate, brushOn);

        const commandEncoder = this.device.createCommandEncoder();
        if (passes > 0) {
            const computePass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('sim'));
            computePass.setPipeline(this.computePipeline);
            const wgX = Math.ceil((this.width / 2 + 1) / 8);
            const wgY = Math.ceil((this.height / 2 + 1) / 8);
            for (let k = 0; k < passes; k++) {
                computePass.setBindGroup(0, this.computeBindGroup, [k * PARAMS_STRIDE]);
                computePass.dispatchWorkgroups(wgX, wgY);
            }
            computePass.end();
        }
        this.encodeRender(commandEncoder, true);
        this.gpuTimer.resolve(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        this.gpuTimer.read();

        if (simulate) this.steps += passes;
        this.paintSeed = (this.paintSeed + 1) >>> 0;
        this.brushPrev = this.brushPos.slice();
    }

    step() {
        this.frame(this.stepsPerFrame);
    }

    // Compute only: `count` (≤ MAX_STEPS) simulation steps in one timed compute pass, the dispatches
    // of frame() without the brush or the draw. Resolves when the GPU is done to
    // { gpuMs (null without timestamp-query), wallMs }, wall = submit → onSubmittedWorkDone.
    async timeSteps(count) {
        const passes = Math.min(Math.max(count, 1), MAX_STEPS);
        this.writeParams(passes, true, false);
        this.gpuTimer.beginFrame();
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass(this.gpuTimer.computePassDescriptor('sim'));
        computePass.setPipeline(this.computePipeline);
        const wgX = Math.ceil((this.width / 2 + 1) / 8);
        const wgY = Math.ceil((this.height / 2 + 1) / 8);
        for (let k = 0; k < passes; k++) {
            computePass.setBindGroup(0, this.computeBindGroup, [k * PARAMS_STRIDE]);
            computePass.dispatchWorkgroups(wgX, wgY);
        }
        computePass.end();
        this.gpuTimer.resolve(commandEncoder);
        const t0 = performance.now();
        this.device.queue.submit([commandEncoder.finish()]);
        const timing = this.gpuTimer.read();
        await this.device.queue.onSubmittedWorkDone();
        const wallMs = performance.now() - t0;
        const result = await timing;
        this.steps += passes;
        return { gpuMs: result ? result.total : null, wallMs };
    }

    // Compute only: the current cells (one u32 each), copied into a temporary MAP_READ buffer
    async readCells() {
        const size = this.cellCount * 4;
        const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.cellBuffer, 0, staging, 0, size);
        this.device.queue.submit([encoder.finish()]);
        try {
            await staging.mapAsync(GPUMapMode.READ);
            return new Uint32Array(staging.getMappedRange().slice(0));
        } finally {
            staging.destroy();
        }
    }

    // timed: add timestamp writes (only frame() resolves the timer)
    encodeRender(commandEncoder, timed = false) {
        const desc = {
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.035, g: 0.04, b: 0.06, a: 1.0 }
            }]
        };
        const renderPass = commandEncoder.beginRenderPass(timed ? this.gpuTimer.renderPassDescriptor(desc, 'render') : desc);
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();
    }

    render() {
        if (!this.canvas) return;
        this.writeParams(1, false, false);
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        if (this.running) return;
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
        if (!this.running) return;

        this.step();

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
            this.canvas.removeEventListener('mousemove', this.onMouseMove);
            this.canvas.removeEventListener('contextmenu', this.onContextMenu);
            window.removeEventListener('mouseup', this.onMouseUp);
        }
        this.cellBuffer.destroy();
        this.paramsBuffer.destroy();
    }
}

export async function initFallingSand(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-sand');
    const output = document.getElementById('output-sand');

    const width = 320;
    const height = 240;

    const simulation = new FallingSand(device, canvas, width, height);
    simulation.sandbox();
    simulation.onStats = () => updateOutput();

    document.getElementById('btn-sand-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });

    document.getElementById('btn-sand-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });

    document.getElementById('btn-sand-clear').addEventListener('click', () => {
        simulation.clear();
        updateOutput();
    });

    document.getElementById('btn-sand-sandbox').addEventListener('click', () => {
        simulation.sandbox();
        updateOutput();
    });

    const materialButtons = document.querySelectorAll('.sand-material');
    function showMaterial() {
        materialButtons.forEach(btn => btn.classList.toggle('active', parseInt(btn.dataset.material) === simulation.material));
    }
    materialButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const id = parseInt(btn.dataset.material);
            if (!MATERIALS.some(m => m.id === id)) return;
            simulation.material = id;
            showMaterial();
            updateOutput();
        });
    });

    const brushSlider = document.getElementById('sand-brush');
    const brushValue = document.getElementById('sand-brush-value');
    brushSlider.addEventListener('input', (e) => {
        simulation.brushRadius = Math.min(Math.max(parseInt(e.target.value), 1), 30);
        brushValue.textContent = simulation.brushRadius;
        updateOutput();
    });

    const speedSlider = document.getElementById('sand-speed');
    const speedValue = document.getElementById('sand-speed-value');
    speedSlider.addEventListener('input', (e) => {
        simulation.stepsPerFrame = Math.min(Math.max(parseInt(e.target.value), 1), MAX_STEPS);
        speedValue.textContent = simulation.stepsPerFrame;
        updateOutput();
    });

    function updateOutput() {
        const materialName = MATERIALS.find(m => m.id === simulation.material).name;
        const stepsPerSecond = simulation.running ? Math.round(simulation.fps * simulation.stepsPerFrame) : 0;
        const blocks = (width / 2 + 1) * (height / 2 + 1);

        output.innerHTML = `<span class="success">✓ Falling Sand Active</span>

<span class="info">Configuration:</span>
• Grid size: ${width} × ${height} (${(width * height).toLocaleString()} cells, one u32 each)
• Material: ${materialName}, brush radius ${simulation.brushRadius} cells
• Steps per frame: ${simulation.stepsPerFrame}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Algorithm (Margolus neighborhood):</span>
• Each step splits the grid into 2×2 blocks, offset (0,0) and (1,1) on alternate steps
• One invocation per block: read 4 cells, react, move by swapping, write 4 cells back
• Blocks never overlap, so the grid updates in place with no conflicts and no lost grains
• Density order: fire < empty < oil < water < acid < sand; stone and plant never move
• Randomness: a PCG hash of (block, step counter), never Math.random

<span class="info">Performance:</span>
• Frame rate: ${simulation.running ? simulation.fps.toFixed(0) : 0} FPS
• ${simulation.gpuTimer.frameText(simulation.running ? simulation.fps : 0, ['sim', 'render'])}
• Simulation steps: ${simulation.steps.toLocaleString()} (${stepsPerSecond.toLocaleString()} / s)
• Blocks per step: ${blocks.toLocaleString()}; one compute dispatch per step, no CPU readback`;
    }

    showMaterial();
    updateOutput();
    return simulation;
}

export { FallingSand, buildSandbox, MATERIALS, MAX_STEPS };   // for test/falling-sand.test.mjs
export const MATERIAL_IDS = { EMPTY, STONE, SAND, WATER, OIL, FIRE, PLANT, ACID };

// ---- Compute-only API for the Measure tab (js/teach/experiments/falling-sand.js) and test/gpu/ ----

export const MAX_MEASURE_CELLS = 2560 * 1920;   // cells per measureSand call (20 MB of u32)
export const MAX_MEASURE_STEPS = 4096;          // steps per call

// `steps` steps of the cellular automaton on a width x height grid (one u32 per cell, as buildSandbox
// returns), `stepsPerSubmit` (1 … MAX_STEPS) dispatches per command buffer. `warmup` untimed steps run
// first and are not counted. The cells are read back after each step count in `readAt` (≤ steps).
// between(), if given, is awaited after every submit (the Measure runner passes ctx.step there, which
// throws to cancel). Resolves to { width, height, steps, gpuMs, wallMs, gpuMsPerStep, wallMsPerStep,
// samples: [{ step, cells }] } (GPU times null without timestamp-query; wall = submit → done, summed).
export async function measureSand(gpuDevice, { width, height, grid, steps, stepsPerSubmit = MAX_STEPS, warmup = 0,
    readAt = [], between = null }) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 4 || height < 4 || width % 2 || height % 2) {
        throw new RangeError(`grid ${width} x ${height} must be even and at least 4 x 4`);
    }
    if (width * height > MAX_MEASURE_CELLS) throw new RangeError(`${width * height} cells exceed ${MAX_MEASURE_CELLS}`);
    if (!grid || grid.length !== width * height) throw new RangeError('grid must hold width · height cells');
    if (!Number.isInteger(steps) || steps < 1 || steps > MAX_MEASURE_STEPS) {
        throw new RangeError(`steps = ${steps} is outside 1 … ${MAX_MEASURE_STEPS}`);
    }
    if (!Number.isInteger(stepsPerSubmit) || stepsPerSubmit < 1 || stepsPerSubmit > MAX_STEPS) {
        throw new RangeError(`stepsPerSubmit = ${stepsPerSubmit} is outside 1 … ${MAX_STEPS}`);
    }
    if (!Number.isInteger(warmup) || warmup < 0 || warmup > MAX_STEPS) throw new RangeError(`warmup = ${warmup} is outside 0 … ${MAX_STEPS}`);
    const marks = [...new Set(readAt)].filter(k => Number.isInteger(k) && k >= 0 && k <= steps).sort((a, b) => a - b);
    const sim = new FallingSand(gpuDevice.device ?? gpuDevice, null, width, height);
    try {
        sim.loadGrid(grid);
        if (warmup > 0) await sim.timeSteps(warmup);
        const samples = [];
        let done = 0, gpuMs = 0, wallMs = 0, timed = true;
        for (const target of [...marks, steps]) {
            while (done < target) {
                const count = Math.min(target - done, stepsPerSubmit);
                const r = await sim.timeSteps(count);
                done += count;
                wallMs += r.wallMs;
                if (r.gpuMs === null) timed = false; else gpuMs += r.gpuMs;
                if (between) await between();
            }
            if (marks.includes(target) && !samples.some(s => s.step === target)) samples.push({ step: target, cells: await sim.readCells() });
        }
        return {
            width, height, steps, wallMs, gpuMs: timed ? gpuMs : null,
            wallMsPerStep: wallMs / steps, gpuMsPerStep: timed ? gpuMs / steps : null, samples,
        };
    } finally {
        sim.destroy();
    }
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}
