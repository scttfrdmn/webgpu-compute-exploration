// N-body gravity on the GPU: brute force O(N²) and Barnes-Hut O(N log N) with the tree rebuilt
// on the GPU every step.
//
// Units: G = 1, a galaxy or cluster has total mass 1 and a scale length of about 1. Every body has
// the same mass m = 1/N. Gravity is Plummer-softened: the force between two bodies a distance r
// apart is m₁m₂ r / (r² + ε²)^{3/2} and the pair potential −m₁m₂ / √(r² + ε²).
//
// Integration: leapfrog in kick-drift-kick form (velocity Verlet), fixed time step dt:
//   cs_kick_drift   v += a·dt/2, then x += v·dt
//   (forces)        a = a(x) by brute force or Barnes-Hut; also the potential φ per body
//   cs_kick         v += a·dt/2
// It is symplectic and time-reversible, so the energy error oscillates instead of drifting
// (for a fixed dt, until close encounters below the softening scale break that).
//
// Brute force (cs_brute_force): Nyland, Harris & Prins, "Fast N-Body Simulation with CUDA",
// GPU Gems 3, ch. 31. Each thread owns one body; the workgroup walks over all bodies in tiles of
// 256, every thread loading one body of the tile into workgroup memory, so each position is read
// from global memory once per workgroup instead of once per thread. N ≤ 32,768.
//
// Barnes-Hut (Barnes & Hut 1986, "A hierarchical O(N log N) force-calculation algorithm") with a
// linear BVH built from scratch every step, all on the GPU (31 dispatches):
//   1. cs_bounds_partial, cs_bounds_final: min/max reduction of the positions to a bounding box
//   2. cs_morton: each body gets a 30-bit Morton code (Morton 1966): its position normalized to
//      the bounding cube and quantized to 1024 cells per axis, with the bits of x, y, z
//      interleaved. Sorting by the code orders the bodies along a Z-order space-filling curve, so
//      bodies close in the order are close in space and every subtree is a compact cluster.
//   3. Radix sort of (code, body index): LSD radix sort with 4-bit digits, 8 passes of
//      histogram / scan / stable scatter (the same algorithm as example 15, carrying the index)
//   4. cs_init_nodes, cs_build_tree: the binary radix tree of Karras 2012, "Maximizing
//      Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (HPG 2012). With N sorted
//      leaves there are exactly N − 1 internal nodes, and internal node i can find its own key
//      range and split position independently from the codes alone (algorithm in cs_build_tree),
//      so the whole tree is built in one dispatch. Duplicate codes are made unique by appending
//      the sorted index (section 4 of the paper).
//   5. cs_bottom_up: mass, center of mass and bounding box of every internal node, bottom-up
//      (Karras section 4 / Apetrei 2014): one thread per leaf walks toward the root; at each
//      node an atomic visit counter makes the first arriving thread stop and the second one,
//      which knows both children are finished, combine them and continue upward.
//   6. cs_bh_force: one thread per body, in Morton order so neighboring threads walk similar
//      paths, traverses the tree depth-first with an explicit stack of 64 entries. A node of size
//      l (longest side of its bounding box) whose center of mass is a distance d away is used as a
//      single point mass when l/d < θ, i.e. d > l/θ; otherwise it is opened. Like Barnes (1994)
//      we add the offset δ between the center of mass and the box center to the opening radius
//      (open when d < l/θ + δ), which protects against the bad case of a node whose mass sits in
//      one corner. A node whose box contains the body is always opened (so a body never feels a
//      monopole that includes its own mass). Leaves are summed exactly.
//
// Bounds on the GPU work (no loop depends on data without a constant cap):
//   • Tree depth: each Karras node splits at the first bit where its (extended) keys differ, which
//     is strictly deeper than its parent's split. The keys are 30 code bits plus the sorted index
//     (18 bits for N ≤ 2^18), so the depth is at most 49 and the 64-entry stack (which holds at
//     most depth + 1 entries in a depth-first walk) never overflows. If it ever did, the node is
//     taken as a monopole instead of being opened. cs_bottom_up's upward walk is capped at 64.
//   • Traversal: at most MAX_VISITS node visits per body. If the cap were hit, the nodes still on
//     the stack are added as monopoles (an approximation, not a hang).
//   • Karras searches: the exponential and binary searches are capped at 32 iterations.
//   • N ≤ 262,144 for Barnes-Hut, ≤ 32,768 for brute force (128 tiles), at most 4 steps per frame,
//     θ ≥ 0.3. The CPU replica of the traversal at N = 262,144, θ = 0.3 needs at most ~12,500
//     visits per body (Plummer), well under MAX_VISITS.
//   • Per frame: steps are dropped (down to 1) when steps × interactions per step (N(N − 1) for
//     brute force, the measured count for Barnes-Hut) would exceed WORK_BUDGET = 2·10⁹, and
//     every step after the first is submitted as its own command buffer.
//
// Memory model note for cs_bottom_up: WGSL atomics are relaxed, and there is no device-scope fence,
// so formally the visit counter does not order the sibling's node data before the second visitor
// reads it. The node data is therefore written and read with atomicStore / atomicLoad (bitcast
// f32), which keeps every access at the device-coherent level, the practical way this pattern
// fails (a stale non-coherent cache line). Internal node masses are cleared to 0 before each build,
// and the traversal skips zero-mass nodes, so even a missed update costs one force contribution
// for one step, never a hang or a NaN.
//
// Energy (readout): every second, after a full step, cs_energy_partial / cs_energy_final sum the
// kinetic energy K = Σ ½ m v² (exact) and the potential energy W = ½ Σ m φ, where φ is the
// potential each body got from the force pass in the same step: exact for brute force, the same
// tree approximation as the forces for Barnes-Hut. The drift (E − E₀)/|E₀| with E = K + W is
// shown relative to the first sample after a reset or a change of mode, θ or ε. The 16-byte
// result is the only readback.
//
// Trails: a ring buffer of TRAIL_LEN = 8 past positions per body (16 bytes each, 32 MB at
// 262,144 bodies, allocated only while trails are on), recorded every TRAIL_EVERY frames, drawn as
// fading line segments.
//
// ---- Uniform layouts ----
//
// struct SimParams (32 bytes, compute):
//   offset  0  count        u32  number of bodies N                         -> Uint32 view [0]
//   offset  4  numTiles     u32  brute-force tiles ceil(N / 256)            -> Uint32 view [1]
//   offset  8  dt           f32  time step                                  -> Float32 view [2]
//   offset 12  eps2         f32  softening ε²                               -> Float32 view [3]
//   offset 16  theta        f32  Barnes-Hut opening angle θ                 -> Float32 view [4]
//   offset 20  numPartials  u32  ceil(N / 1024) reduction workgroups        -> Uint32 view [5]
//   offset 24  trailSlot    u32  ring-buffer slot recorded this frame       -> Uint32 view [6]
//   offset 28  pad0         u32                                             -> Uint32 view [7]
//
// struct RadixParams (16 bytes; one copy per digit pass at a dynamic offset of radixStride):
//   offset  0  shift     u32  bit position of this pass's 4-bit digit       -> Uint32 view [0]
//   offset  4  count     u32  number of keys N                              -> Uint32 view [1]
//   offset  8  numTiles  u32  ceil(N / 1024)                                -> Uint32 view [2]
//   offset 12  pad0      u32                                                -> Uint32 view [3]
//
// struct Camera (112 bytes, render):
//   offset   0  viewProj    mat4x4<f32>  column-major, depth [0, 1]         -> Float32 view [0..15]
//   offset  64  viewport    vec2<f32>    canvas size in pixels              -> Float32 view [16], [17]
//   offset  72  pointSize   f32          sprite radius in pixels            -> Float32 view [18]
//   offset  76  brightness  f32          per-body intensity                 -> Float32 view [19]
//   offset  80  colorMode   u32          0 = speed, 1 = origin              -> Uint32 view [20]
//   offset  84  count       u32          N                                  -> Uint32 view [21]
//   offset  88  speedScale  f32          speed mapped to the top of the ramp -> Float32 view [22]
//   offset  92  trailHead   u32          newest trail slot                  -> Uint32 view [23]
//   offset  96  trailLen    u32          TRAIL_LEN                          -> Uint32 view [24]
//   offset 100  pad0..pad2  u32                                             -> Uint32 view [25..27]
//
// Storage: positions array<vec4<f32>> (xyz, w = mass), velocities array<vec4<f32>> (xyz, w = origin
// tag), accel array<vec4<f32>> (xyz acceleration, w = potential φ), interactions array<f32>, keys and
// values array<u32> (two of each for the sort's ping-pong), radix offsets, tree: children
// array<vec2<u32>> and visit counters (N − 1 internal nodes), parents array<u32>, nodeMass
// array<vec4<f32>> (center of mass, mass) and nodeBox (two vec4 per node: min.xyz + l, max.xyz + δ)
// for 2N − 1 nodes (internal nodes 0 … N − 2, leaf of sorted body k at N − 1 + k, root 0).

export const WG = 256;
export const MAX_BH = 262144;
export const MAX_BRUTE = 32768;
export const MAX_STEPS_PER_FRAME = 4;
export const STACK_SIZE = 64;
export const MAX_DEPTH = 64;
export const MAX_VISITS = 32768;
export const MIN_THETA = 0.3;
export const WORK_BUDGET = 2e9;      // interactions per frame; more steps per frame are dropped
export const RADIX_PASSES = 8;
export const RADIX_TILE = 1024;
export const REDUCE_TILE = 1024;     // bodies per workgroup in the bounds and energy reductions
export const TRAIL_LEN = 8;
export const TRAIL_EVERY = 3;
export const NONE = 0xffffffff;
export const COUNTS = [4096, 16384, 32768, 65536, 131072, 262144];

const SIM_PARAMS_SIZE = 32;
const RADIX_PARAMS_SIZE = 16;
const CAMERA_SIZE = 112;
const ENERGY_INTERVAL_MS = 1000;

const simStruct = `
struct SimParams {
    count: u32,
    numTiles: u32,
    dt: f32,
    eps2: f32,
    theta: f32,
    numPartials: u32,
    trailSlot: u32,
    pad0: u32,
}

const WG: u32 = ${WG}u;
const NONE: u32 = 0xffffffffu;

// Workgroups are laid out on a 2D grid when a job needs more than maxComputeWorkgroupsPerDimension
fn flatIndex(wid: vec3<u32>, nwg: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (wid.y * nwg.x + wid.x) * WG + lid.x;
}
`;

// ---- Leapfrog (kick-drift-kick) ----
export const integrateShader = simStruct + `
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> accel: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SimParams;

// First half: v(t + dt/2) = v(t) + a(t)·dt/2, then x(t + dt) = x(t) + v(t + dt/2)·dt
@compute @workgroup_size(${WG})
fn cs_kick_drift(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flatIndex(wid, nwg, lid);
    if (i >= params.count) {
        return;
    }
    let vel = velocities[i];
    let v = vel.xyz + accel[i].xyz * (0.5 * params.dt);
    velocities[i] = vec4<f32>(v, vel.w);
    let p = positions[i];
    positions[i] = vec4<f32>(p.xyz + v * params.dt, p.w);
}

// Second half, after the forces at x(t + dt): v(t + dt) = v(t + dt/2) + a(t + dt)·dt/2
@compute @workgroup_size(${WG})
fn cs_kick(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flatIndex(wid, nwg, lid);
    if (i >= params.count) {
        return;
    }
    let vel = velocities[i];
    velocities[i] = vec4<f32>(vel.xyz + accel[i].xyz * (0.5 * params.dt), vel.w);
}
`;

// ---- Brute force, GPU Gems 3 ch. 31 tiling ----
export const bruteShader = simStruct + `
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accel: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

const MAX_TILES: u32 = ${MAX_BRUTE / WG}u;

var<workgroup> tile: array<vec4<f32>, ${WG}>;

// a_i = Σ_j m_j (x_j − x_i) / (|x_j − x_i|² + ε²)^{3/2},  φ_i = −Σ_{j≠i} m_j / √(|x_j − x_i|² + ε²)
// Every thread loads one body of the tile (zero mass past the end), the workgroup syncs, and each
// thread sums its body against all 256. Threads past N take part in the loads and barriers but
// write nothing. The tile count is uniform and capped by MAX_TILES.
@compute @workgroup_size(${WG})
fn cs_brute_force(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flatIndex(wid, nwg, lid);
    let valid = i < params.count;
    var p = vec3<f32>(0.0);
    if (valid) {
        p = positions[i].xyz;
    }
    var acc = vec3<f32>(0.0);
    var phi = 0.0;
    let tiles = min(params.numTiles, MAX_TILES);
    for (var t = 0u; t < tiles; t++) {
        let j = t * WG + lid.x;
        var q = vec4<f32>(0.0);
        if (j < params.count) {
            q = positions[j];
        }
        tile[lid.x] = q;
        workgroupBarrier();
        for (var k = 0u; k < WG; k++) {
            let b = tile[k];
            let d = b.xyz - p;
            let invR = inverseSqrt(dot(d, d) + params.eps2);
            let isMe = t * WG + k == i;
            let mInvR = select(b.w * invR, 0.0, isMe);
            phi -= mInvR;
            acc += d * (mInvR * invR * invR);
        }
        workgroupBarrier();
    }
    if (valid) {
        accel[i] = vec4<f32>(acc, phi);
    }
}
`;

// ---- Barnes-Hut stage 1: bounding box ----
export const boundsShader = simStruct + `
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
// [0] = min, [1] = max of the whole set; [2 + 2g], [3 + 2g] = workgroup g's partial min / max
@group(0) @binding(1) var<storage, read_write> bounds: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

const BIG: f32 = 3.0e38;
const ROUNDS: u32 = ${REDUCE_TILE / WG}u;

var<workgroup> redMin: array<vec3<f32>, ${WG}>;
var<workgroup> redMax: array<vec3<f32>, ${WG}>;

fn reduceTree(lid: u32) {
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            redMin[lid] = min(redMin[lid], redMin[lid + s]);
            redMax[lid] = max(redMax[lid], redMax[lid + s]);
        }
        workgroupBarrier();
    }
}

// Each workgroup reduces 1024 bodies (4 per thread, then a tree in workgroup memory)
@compute @workgroup_size(${WG})
fn cs_bounds_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.y * nwg.x + wid.x;
    if (g >= params.numPartials) {
        return;   // the whole workgroup leaves: g is workgroup-uniform
    }
    var lo = vec3<f32>(BIG);
    var hi = vec3<f32>(-BIG);
    for (var r = 0u; r < ROUNDS; r++) {
        let i = g * ${REDUCE_TILE}u + r * WG + lid.x;
        if (i < params.count) {
            let p = positions[i].xyz;
            lo = min(lo, p);
            hi = max(hi, p);
        }
    }
    redMin[lid.x] = lo;
    redMax[lid.x] = hi;
    workgroupBarrier();
    reduceTree(lid.x);
    if (lid.x == 0u) {
        bounds[2u + 2u * g] = vec4<f32>(redMin[0], 0.0);
        bounds[3u + 2u * g] = vec4<f32>(redMax[0], 0.0);
    }
}

// One workgroup reduces the ≤ 256 partials
@compute @workgroup_size(${WG})
fn cs_bounds_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    var lo = vec3<f32>(BIG);
    var hi = vec3<f32>(-BIG);
    if (lid.x < params.numPartials) {
        lo = bounds[2u + 2u * lid.x].xyz;
        hi = bounds[3u + 2u * lid.x].xyz;
    }
    redMin[lid.x] = lo;
    redMax[lid.x] = hi;
    workgroupBarrier();
    reduceTree(lid.x);
    if (lid.x == 0u) {
        bounds[0] = vec4<f32>(redMin[0], 0.0);
        bounds[1] = vec4<f32>(redMax[0], 0.0);
    }
}
`;

// ---- Barnes-Hut stage 2: Morton codes ----
export const mortonShader = simStruct + `
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bounds: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals: array<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;

// Spreads the low 10 bits of v so there are two zero bits between each: b9 b8 … b0 ->
// b9 0 0 b8 0 0 … 0 0 b0 (30 bits), by the usual shift-and-mask steps
fn expandBits(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

// The bounding box is widened to a cube (its longest side) so that the cells are cubes, then each
// coordinate is quantized to 0 … 1023. code = x9 y9 z9 x8 y8 z8 … x0 y0 z0.
@compute @workgroup_size(${WG})
fn cs_morton(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flatIndex(wid, nwg, lid);
    if (i >= params.count) {
        return;
    }
    let lo = bounds[0].xyz;
    let ext = bounds[1].xyz - lo;
    let side = max(max(ext.x, ext.y), max(ext.z, 1e-30));
    let scale = 1024.0 / side;
    let q = min(floor(max((positions[i].xyz - lo) * scale, vec3<f32>(0.0))), vec3<f32>(1023.0));
    let c = vec3<u32>(q);
    keys[i] = (expandBits(c.x) << 2u) | (expandBits(c.y) << 1u) | expandBits(c.z);
    vals[i] = i;
}
`;

// ---- Barnes-Hut stage 3: LSD radix sort of (code, index) pairs ----
export const radixShader = `
struct RadixParams {
    shift: u32,
    count: u32,
    numTiles: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> keysIn: array<u32>;
@group(0) @binding(1) var<storage, read> valsIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(3) var<storage, read_write> valsOut: array<u32>;
// digit-major: entry d·numTiles + tile. Histogram counts, then (after the scan) global offsets.
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: RadixParams;

const WG: u32 = ${WG}u;
const BUCKETS: u32 = 16u;
const ROUNDS: u32 = ${RADIX_TILE / WG}u;
const TILE: u32 = ${RADIX_TILE}u;
const MAX_CHUNK: u32 = ${(16 * (MAX_BH / RADIX_TILE)) / WG}u;   // scan entries per thread at N = ${MAX_BH}

// Per pass (4-bit digit at params.shift), a stable counting sort:
//   1. cs_radix_histogram: each workgroup counts the digits of its 1024-key tile
//   2. cs_radix_scan: exclusive prefix sum over the digit-major histogram, so entry (d, tile)
//      becomes "keys with a smaller digit anywhere + keys with digit d in earlier tiles"
//   3. cs_radix_scatter: each pair goes to offset(d, tile) + its rank among the equal digits
//      before it in the tile, which keeps equal digits in input order (stable)
// 8 passes cover the 30-bit codes (the last pass sees bits 28-31).

var<workgroup> digitCounts: array<atomic<u32>, 16>;
var<workgroup> partialSums: array<u32, ${WG}>;
// Scatter: 16 digit counters of 16 bits packed in 8 words per thread (lo = digits 0-7)
var<workgroup> scanLo: array<vec4<u32>, ${WG}>;
var<workgroup> scanHi: array<vec4<u32>, ${WG}>;
var<workgroup> tileBase: array<u32, 16>;

fn flatGroup(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.y * nwg.x + wid.x;
}

fn digitOf(key: u32) -> u32 {
    return (key >> params.shift) & (BUCKETS - 1u);
}

@compute @workgroup_size(${WG})
fn cs_radix_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>) {
    let tileIndex = flatGroup(wid, nwg);
    if (tileIndex >= params.numTiles) {
        return;
    }
    if (lid.x < BUCKETS) {
        atomicStore(&digitCounts[lid.x], 0u);
    }
    workgroupBarrier();
    for (var r = 0u; r < ROUNDS; r++) {
        let idx = tileIndex * TILE + r * WG + lid.x;
        if (idx < params.count) {
            atomicAdd(&digitCounts[digitOf(keysIn[idx])], 1u);
        }
    }
    workgroupBarrier();
    if (lid.x < BUCKETS) {
        offsets[lid.x * params.numTiles + tileIndex] = atomicLoad(&digitCounts[lid.x]);
    }
}

// Exclusive scan of the 16·numTiles (≤ 4096) entries by one workgroup: each thread sums a
// contiguous chunk, a Hillis-Steele scan combines the 256 chunk sums, each thread rewrites its chunk
@compute @workgroup_size(${WG})
fn cs_radix_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let total = BUCKETS * params.numTiles;
    let chunk = (total + WG - 1u) / WG;
    let start = lid.x * chunk;

    var sum = 0u;
    for (var i = 0u; i < MAX_CHUNK; i++) {
        if (i >= chunk) {
            break;
        }
        let idx = start + i;
        if (idx < total) {
            sum += offsets[idx];
        }
    }
    partialSums[lid.x] = sum;
    workgroupBarrier();

    for (var o = 1u; o < WG; o = o << 1u) {
        var v = partialSums[lid.x];
        if (lid.x >= o) {
            v += partialSums[lid.x - o];
        }
        workgroupBarrier();
        partialSums[lid.x] = v;
        workgroupBarrier();
    }

    var running = partialSums[lid.x] - sum;   // inclusive -> exclusive
    for (var i = 0u; i < MAX_CHUNK; i++) {
        if (i >= chunk) {
            break;
        }
        let idx = start + i;
        if (idx < total) {
            let c = offsets[idx];
            offsets[idx] = running;
            running += c;
        }
    }
}

fn wordOf(lo: vec4<u32>, hi: vec4<u32>, w: u32) -> u32 {
    if (w < 4u) {
        return lo[w];
    }
    return hi[w - 4u];
}

// Stable scatter, 4 rounds of 256 keys in index order. Each thread one-hot encodes its digit as a
// 16-bit counter; a Hillis-Steele scan of the packed counters gives every thread the number of
// equal digits held by lower threads (≤ 256, so the 16-bit fields never carry).
@compute @workgroup_size(${WG})
fn cs_radix_scatter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
    let tileIndex = flatGroup(wid, nwg);
    if (tileIndex >= params.numTiles) {
        return;
    }
    if (lid.x < BUCKETS) {
        tileBase[lid.x] = offsets[lid.x * params.numTiles + tileIndex];
    }
    workgroupBarrier();

    let lanes = vec4<u32>(0u, 1u, 2u, 3u);
    for (var r = 0u; r < ROUNDS; r++) {
        let idx = tileIndex * TILE + r * WG + lid.x;
        let valid = idx < params.count;
        var key = 0u;
        var val = 0u;
        var digit = 0u;
        var ownLo = vec4<u32>(0u);
        var ownHi = vec4<u32>(0u);
        if (valid) {
            key = keysIn[idx];
            val = valsIn[idx];
            digit = digitOf(key);
            let bit = 1u << ((digit & 1u) * 16u);
            let w = digit >> 1u;
            ownLo = select(vec4<u32>(0u), vec4<u32>(bit), lanes == vec4<u32>(w));
            ownHi = select(vec4<u32>(0u), vec4<u32>(bit), lanes + vec4<u32>(4u) == vec4<u32>(w));
        }

        var lo = ownLo;
        var hi = ownHi;
        scanLo[lid.x] = lo;
        scanHi[lid.x] = hi;
        workgroupBarrier();
        for (var o = 1u; o < WG; o = o << 1u) {
            if (lid.x >= o) {
                lo += scanLo[lid.x - o];
                hi += scanHi[lid.x - o];
            }
            workgroupBarrier();
            scanLo[lid.x] = lo;
            scanHi[lid.x] = hi;
            workgroupBarrier();
        }

        if (valid) {
            let word = wordOf(lo - ownLo, hi - ownHi, digit >> 1u);
            let rank = (word >> ((digit & 1u) * 16u)) & 0xffffu;
            let dst = tileBase[digit] + rank;
            keysOut[dst] = key;
            valsOut[dst] = val;
        }
        workgroupBarrier();   // everyone has read tileBase for this round

        // Thread 255's inclusive counters are the round's digit totals
        if (lid.x < BUCKETS) {
            let totalWord = wordOf(scanLo[WG - 1u], scanHi[WG - 1u], lid.x >> 1u);
            tileBase[lid.x] += (totalWord >> ((lid.x & 1u) * 16u)) & 0xffffu;
        }
        workgroupBarrier();
    }
}
`;

// ---- Barnes-Hut stage 4a: leaf data and cleared internal nodes ----
export const initNodesShader = simStruct + `
@group(0) @binding(0) var<storage, read> vals: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> nodeMass: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nodeBox: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

// Leaf N − 1 + k holds sorted body k: center of mass = its position, a point box (l = δ = 0).
// Internal node masses are zeroed so the traversal can tell an unfinished node from a real one.
@compute @workgroup_size(${WG})
fn cs_init_nodes(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let k = flatIndex(wid, nwg, lid);
    let n = params.count;
    if (k >= n) {
        return;
    }
    let p = positions[vals[k]];
    let leaf = n - 1u + k;
    nodeMass[leaf] = p;
    nodeBox[2u * leaf] = vec4<f32>(p.xyz, 0.0);
    nodeBox[2u * leaf + 1u] = vec4<f32>(p.xyz, 0.0);
    if (k + 1u < n) {
        nodeMass[k] = vec4<f32>(0.0);
    }
}
`;

// ---- Barnes-Hut stage 4b: Karras 2012 binary radix tree ----
export const buildShader = simStruct + `
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> children: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> parents: array<u32>;
@group(0) @binding(3) var<storage, read_write> visits: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

const SEARCH_CAP: u32 = 32u;

// δ(i, j): length of the longest common prefix of the keys of sorted leaves i and j, or −1 when j
// is outside [0, N − 1]. Equal codes are told apart by their indices: the key is (code, index), so
// δ = 32 + clz(i xor j) (Karras section 4), which is larger than any prefix of differing codes.
fn delta(i: i32, j: i32) -> i32 {
    if (j < 0 || j >= i32(params.count)) {
        return -1;
    }
    let ki = keys[u32(i)];
    let kj = keys[u32(j)];
    if (ki == kj) {
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(ki ^ kj));
}

// One thread per internal node i (Karras 2012, figure 4):
//   1. Direction d = ±1: the neighbor with the longer common prefix is inside the node's range.
//   2. Range end: δ_min = δ(i, i − d); every key in the range shares more than δ_min bits with key i.
//      Exponential search for an upper bound l_max, then binary search for the exact length l,
//      so the other end is j = i + l·d.
//   3. Split: binary search for the last key that shares more than δ_node = δ(i, j) bits with key
//      i, giving γ = i + s·d + min(d, 0). Left child covers [min(i, j), γ], right [γ + 1, max(i, j)];
//      a child whose range is a single key is that leaf.
// Each node's range comes from the keys only, so all N − 1 nodes are built in parallel. Node
// indices: internal 0 … N − 2 (root 0), leaf k at N − 1 + k. The visit counters for cs_bottom_up
// are reset here.
@compute @workgroup_size(${WG})
fn cs_build_tree(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = flatIndex(wid, nwg, lid);
    let n = params.count;
    if (idx == 0u) {
        parents[0] = NONE;   // the root (or, for N = 1, the only leaf) has no parent
    }
    if (idx + 1u >= n) {
        return;
    }
    let i = i32(idx);
    atomicStore(&visits[idx], 0u);

    let d = select(-1, 1, delta(i, i + 1) > delta(i, i - 1));
    let deltaMin = delta(i, i - d);
    var lMax = 2;
    for (var it = 0u; it < SEARCH_CAP; it++) {
        if (delta(i, i + lMax * d) <= deltaMin) {
            break;
        }
        lMax = lMax * 2;
    }
    var l = 0;
    var t = lMax / 2;
    for (var it = 0u; it < SEARCH_CAP; it++) {
        if (t < 1) {
            break;
        }
        if (delta(i, i + (l + t) * d) > deltaMin) {
            l = l + t;
        }
        t = t / 2;
    }
    let j = i + l * d;
    let deltaNode = delta(i, j);

    var s = 0;
    var div = 2;
    for (var it = 0u; it < SEARCH_CAP; it++) {
        let stride = (l + div - 1) / div;   // ⌈l/2⌉, ⌈l/4⌉, …, 1
        if (delta(i, i + (s + stride) * d) > deltaNode) {
            s = s + stride;
        }
        if (stride <= 1) {
            break;
        }
        div = div * 2;
    }
    let gamma = i + s * d + min(d, 0);

    let first = min(i, j);
    let last = max(i, j);
    var left = u32(gamma);
    if (first == gamma) {
        left = n - 1u + u32(gamma);
    }
    var right = u32(gamma + 1);
    if (last == gamma + 1) {
        right = n - 1u + u32(gamma + 1);
    }
    children[idx] = vec2<u32>(left, right);
    parents[left] = idx;
    parents[right] = idx;
}
`;

// ---- Barnes-Hut stage 5: bottom-up mass, center of mass and bounds ----
export const bottomUpShader = simStruct + `
@group(0) @binding(0) var<storage, read> parents: array<u32>;
@group(0) @binding(1) var<storage, read> children: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> visits: array<atomic<u32>>;
// The same buffers as nodeMass / nodeBox, seen as words for atomic (device-coherent) access
@group(0) @binding(3) var<storage, read_write> massWords: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> boxWords: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: SimParams;

const MAX_DEPTH: u32 = ${MAX_DEPTH}u;

fn loadMass(node: u32) -> vec4<f32> {
    let b = 4u * node;
    return vec4<f32>(bitcast<f32>(atomicLoad(&massWords[b])), bitcast<f32>(atomicLoad(&massWords[b + 1u])),
                     bitcast<f32>(atomicLoad(&massWords[b + 2u])), bitcast<f32>(atomicLoad(&massWords[b + 3u])));
}

fn storeMass(node: u32, v: vec4<f32>) {
    let b = 4u * node;
    atomicStore(&massWords[b], bitcast<u32>(v.x));
    atomicStore(&massWords[b + 1u], bitcast<u32>(v.y));
    atomicStore(&massWords[b + 2u], bitcast<u32>(v.z));
    atomicStore(&massWords[b + 3u], bitcast<u32>(v.w));
}

// nodeBox entry e (two per node) as a vec4
fn loadBox(e: u32) -> vec4<f32> {
    let b = 4u * e;
    return vec4<f32>(bitcast<f32>(atomicLoad(&boxWords[b])), bitcast<f32>(atomicLoad(&boxWords[b + 1u])),
                     bitcast<f32>(atomicLoad(&boxWords[b + 2u])), bitcast<f32>(atomicLoad(&boxWords[b + 3u])));
}

fn storeBox(e: u32, v: vec4<f32>) {
    let b = 4u * e;
    atomicStore(&boxWords[b], bitcast<u32>(v.x));
    atomicStore(&boxWords[b + 1u], bitcast<u32>(v.y));
    atomicStore(&boxWords[b + 2u], bitcast<u32>(v.z));
    atomicStore(&boxWords[b + 3u], bitcast<u32>(v.w));
}

// One thread per leaf climbs toward the root. At each parent, atomicAdd on its visit counter:
// the first thread to arrive (old value 0) stops, because the other subtree may not be done; the
// second one combines both children:
//   M = M_L + M_R,  com = (M_L·com_L + M_R·com_R) / M,  box = box_L ∪ box_R,
//   l = longest side of the box,  δ = |com − box center|  (for the opening test)
// and continues with the grandparent. Every internal node is completed exactly once, by the
// second visitor, after both of its subtrees.
@compute @workgroup_size(${WG})
fn cs_bottom_up(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let k = flatIndex(wid, nwg, lid);
    let n = params.count;
    if (k >= n) {
        return;
    }
    var node = n - 1u + k;
    for (var level = 0u; level < MAX_DEPTH; level++) {
        let p = parents[node];
        if (p == NONE) {
            break;
        }
        if (atomicAdd(&visits[p], 1u) == 0u) {
            break;
        }
        let c = children[p];
        let a = loadMass(c.x);
        let b = loadMass(c.y);
        let m = a.w + b.w;
        var com = 0.5 * (a.xyz + b.xyz);
        if (m > 0.0) {
            com = (a.xyz * a.w + b.xyz * b.w) / m;
        }
        let lo = min(loadBox(2u * c.x).xyz, loadBox(2u * c.y).xyz);
        let hi = max(loadBox(2u * c.x + 1u).xyz, loadBox(2u * c.y + 1u).xyz);
        let ext = hi - lo;
        let size = max(max(ext.x, ext.y), ext.z);
        let offset = length(com - 0.5 * (lo + hi));
        storeBox(2u * p, vec4<f32>(lo, size));
        storeBox(2u * p + 1u, vec4<f32>(hi, offset));
        storeMass(p, vec4<f32>(com, m));
        node = p;
    }
}
`;

// ---- Barnes-Hut stage 6: tree traversal ----
export const forceShader = simStruct + `
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vals: array<u32>;
@group(0) @binding(2) var<storage, read> nodeMass: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> nodeBox: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> children: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> accel: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> interactions: array<f32>;
@group(0) @binding(7) var<uniform> params: SimParams;

const STACK_SIZE: u32 = ${STACK_SIZE}u;
const MAX_VISITS: u32 = ${MAX_VISITS}u;

struct Accum {
    acc: vec3<f32>,
    phi: f32,
}

// Plummer-softened point mass m at offset d from the body
fn addMass(sum: Accum, d: vec3<f32>, m: f32) -> Accum {
    let invR = inverseSqrt(dot(d, d) + params.eps2);
    let mInvR = m * invR;
    return Accum(sum.acc + d * (mInvR * invR * invR), sum.phi - mInvR);
}

// Thread k handles sorted body k (Morton order keeps neighboring threads on similar paths).
// Depth-first walk from the root with a fixed-size stack. A leaf is summed exactly (except the
// body's own leaf); an internal node of size l, COM offset δ and mass M at distance d is accepted as
// a point mass when d ≥ l/θ + δ and the body is outside its box, otherwise its children are pushed.
@compute @workgroup_size(${WG})
fn cs_bh_force(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let k = flatIndex(wid, nwg, lid);
    let n = params.count;
    if (k >= n) {
        return;
    }
    let body = vals[k];
    let p = positions[body].xyz;
    let myLeaf = n - 1u + k;
    let firstLeaf = n - 1u;
    let invTheta = 1.0 / params.theta;

    var stack: array<u32, ${STACK_SIZE}>;
    stack[0] = 0u;   // root (internal node 0; for N = 1 the only leaf, which is skipped)
    var sp = 1u;
    var sum = Accum(vec3<f32>(0.0), 0.0);
    var count = 0u;
    for (var it = 0u; it < MAX_VISITS; it++) {
        if (sp == 0u) {
            break;
        }
        sp -= 1u;
        let node = stack[sp];
        let nm = nodeMass[node];
        if (nm.w <= 0.0) {
            continue;
        }
        let d = nm.xyz - p;
        if (node >= firstLeaf) {
            if (node != myLeaf) {
                sum = addMass(sum, d, nm.w);
                count += 1u;
            }
            continue;
        }
        let lo = nodeBox[2u * node];
        let hi = nodeBox[2u * node + 1u];
        let rOpen = lo.w * invTheta + hi.w;
        let inside = all(p >= lo.xyz) && all(p <= hi.xyz);
        if ((dot(d, d) < rOpen * rOpen || inside) && sp + 2u <= STACK_SIZE) {
            let c = children[node];
            stack[sp] = c.y;
            stack[sp + 1u] = c.x;
            sp += 2u;
        } else {
            sum = addMass(sum, d, nm.w);
            count += 1u;
        }
    }
    // Only if MAX_VISITS ran out: what is left on the stack counts as point masses
    for (var r = 0u; r < STACK_SIZE; r++) {
        if (r >= sp) {
            break;
        }
        let node = stack[r];
        let nm = nodeMass[node];
        if (node != myLeaf && nm.w > 0.0) {
            sum = addMass(sum, nm.xyz - p, nm.w);
            count += 1u;
        }
    }
    accel[body] = vec4<f32>(sum.acc, sum.phi);
    interactions[body] = f32(count);
}
`;

// ---- Energy readout ----
export const energyShader = simStruct + `
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> accel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> interactions: array<f32>;
// [0] = totals (K, W, interactions, 0); [1 + g] = workgroup g's partial sums
@group(0) @binding(4) var<storage, read_write> energy: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: SimParams;

const ROUNDS: u32 = ${REDUCE_TILE / WG}u;

var<workgroup> red: array<vec4<f32>, ${WG}>;

fn reduceTree(lid: u32) {
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            red[lid] = red[lid] + red[lid + s];
        }
        workgroupBarrier();
    }
}

// K = Σ ½ m v², W = ½ Σ m φ (each pair counted from both sides), plus the interaction counts
@compute @workgroup_size(${WG})
fn cs_energy_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.y * nwg.x + wid.x;
    if (g >= params.numPartials) {
        return;
    }
    var s = vec4<f32>(0.0);
    for (var r = 0u; r < ROUNDS; r++) {
        let i = g * ${REDUCE_TILE}u + r * WG + lid.x;
        if (i < params.count) {
            let m = positions[i].w;
            let v = velocities[i].xyz;
            s += vec4<f32>(0.5 * m * dot(v, v), 0.5 * m * accel[i].w, interactions[i], 0.0);
        }
    }
    red[lid.x] = s;
    workgroupBarrier();
    reduceTree(lid.x);
    if (lid.x == 0u) {
        energy[1u + g] = red[0];
    }
}

@compute @workgroup_size(${WG})
fn cs_energy_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    var s = vec4<f32>(0.0);
    if (lid.x < params.numPartials) {
        s = energy[1u + lid.x];
    }
    red[lid.x] = s;
    workgroupBarrier();
    reduceTree(lid.x);
    if (lid.x == 0u) {
        energy[0] = red[0];
    }
}
`;

// ---- Trails ----
export const trailShader = simStruct + `
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> trail: array<vec4<f32>>;   // slot s of body i at s·N + i
@group(0) @binding(2) var<uniform> params: SimParams;

const TRAIL_LEN: u32 = ${TRAIL_LEN}u;

@compute @workgroup_size(${WG})
fn cs_trail_record(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flatIndex(wid, nwg, lid);
    if (i >= params.count) {
        return;
    }
    trail[params.trailSlot * params.count + i] = positions[i];
}

// Fills every slot with the current position (after a reset), so old trails do not flash
@compute @workgroup_size(${WG})
fn cs_trail_fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flatIndex(wid, nwg, lid);
    if (i >= params.count) {
        return;
    }
    let p = positions[i];
    for (var s = 0u; s < TRAIL_LEN; s++) {
        trail[s * params.count + i] = p;
    }
}
`;

// ---- Rendering: additive point sprites (and trails) into an HDR target, then tone mapping ----
export const renderShader = `
struct Camera {
    viewProj: mat4x4<f32>,
    viewport: vec2<f32>,
    pointSize: f32,
    brightness: f32,
    colorMode: u32,
    count: u32,
    speedScale: f32,
    trailHead: u32,
    trailLen: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> trail: array<vec4<f32>>;

const HIDDEN = vec4<f32>(2.0, 2.0, 2.0, 1.0);   // outside the clip volume: dropped

// Speed ramp: deep blue (slow) -> pale blue-white -> gold -> red-orange (fast)
fn speedColor(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.20, 0.32, 1.00);
    let c1 = vec3<f32>(0.75, 0.85, 1.00);
    let c2 = vec3<f32>(1.00, 0.78, 0.35);
    let c3 = vec3<f32>(1.00, 0.35, 0.18);
    let x = clamp(t, 0.0, 1.0) * 3.0;
    if (x < 1.0) { return mix(c0, c1, x); }
    if (x < 2.0) { return mix(c1, c2, x - 1.0); }
    return mix(c2, c3, x - 2.0);
}

// Origin tags: 0 disk (or inner half) of the first system, 1 its bulge, 2 disk (or outer half) of
// the second, 3 its bulge
fn originColor(tag: f32) -> vec3<f32> {
    var colors = array<vec3<f32>, 4>(
        vec3<f32>(0.45, 0.65, 1.00), vec3<f32>(1.00, 0.92, 0.70),
        vec3<f32>(1.00, 0.55, 0.25), vec3<f32>(1.00, 0.80, 0.55)
    );
    return colors[min(u32(max(tag, 0.0)), 3u)];
}

fn bodyColor(i: u32) -> vec3<f32> {
    let v = velocities[i];
    if (cam.colorMode == 1u) {
        return originColor(v.w);
    }
    return speedColor(length(v.xyz) / cam.speedScale);
}

struct SpriteOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
}

// One instanced screen-space quad per body
@vertex
fn vs_body(@builtin(vertex_index) vi: u32, @builtin(instance_index) body: u32) -> SpriteOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let q = quad[vi];
    let clip = cam.viewProj * vec4<f32>(positions[body].xyz, 1.0);
    var out: SpriteOut;
    out.position = select(HIDDEN, clip + vec4<f32>(q * (2.0 * cam.pointSize) / cam.viewport * clip.w, 0.0, 0.0),
                          clip.w > 1e-3);
    out.local = q;
    out.color = bodyColor(body) * cam.brightness;
    return out;
}

// Soft glow: a Gaussian core with a faint halo, added into the HDR target
@fragment
fn fs_body(in: SpriteOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.local, in.local);
    if (r2 > 1.0) {
        discard;
    }
    let glow = exp(-r2 * 6.0) + 0.12 * (1.0 - r2);
    return vec4<f32>(in.color * glow, 0.0);
}

struct TrailOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

// Point k = 0 is the body now, k = 1 … trailLen the recorded slots from newest to oldest;
// segment s joins points s and s + 1 and fades with age
@vertex
fn vs_trail(@builtin(vertex_index) vi: u32, @builtin(instance_index) body: u32) -> TrailOut {
    let k = vi / 2u + (vi & 1u);
    var p = positions[body].xyz;
    if (k > 0u) {
        let slot = (cam.trailHead + cam.trailLen - (k - 1u)) % cam.trailLen;
        p = trail[slot * cam.count + body].xyz;
    }
    let clip = cam.viewProj * vec4<f32>(p, 1.0);
    var out: TrailOut;
    out.position = select(HIDDEN, clip, clip.w > 1e-3);
    let fade = 1.0 - f32(k) / f32(cam.trailLen + 1u);
    out.color = bodyColor(body) * (cam.brightness * 0.35 * fade);
    return out;
}

@fragment
fn fs_trail(in: TrailOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 0.0);
}
`;

export const tonemapShader = `
@group(0) @binding(0) var hdr: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(corners[vi], 0.0, 1.0);
}

// Exponential tone map 1 − e^{−x}: dense cores saturate smoothly to white instead of clipping
@fragment
fn fs_tonemap(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(hdr, vec2<i32>(pos.xy), 0).rgb;
    let background = vec3<f32>(0.008, 0.010, 0.022);
    return vec4<f32>(background + (vec3<f32>(1.0) - exp(-c)), 1.0);
}
`;

// ---- Initial conditions ----

// Mulberry32: small seeded PRNG so presets are reproducible
export function makeRng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function gaussian(rng) {
    const u = Math.max(rng(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function randomDirection(rng) {
    const z = 2 * rng() - 1;
    const phi = 2 * Math.PI * rng();
    const s = Math.sqrt(1 - z * z);
    return [s * Math.cos(phi), z, s * Math.sin(phi)];
}

// Plummer (1911) sphere sampling after Aarseth, Hénon & Wielen 1974 (A&A 37, 183), mass M, scale b,
// truncated at rMax: r = b / √(u^{−2/3} − 1); speed q·v_esc with q drawn from g(q) = q²(1 − q²)^{7/2}
// by rejection, v_esc = √(2GM) (r² + b²)^{−1/4}. Isotropic, in virial equilibrium.
function plummerRadius(rng, b, rMax) {
    for (;;) {
        const u = Math.max(rng(), 1e-12);
        const r = b / Math.sqrt(Math.pow(u, -2 / 3) - 1);
        if (r < rMax) return r;
    }
}

function plummerSpeed(rng, r, mass, b) {
    let q = 0;
    for (;;) {
        q = rng();
        if (0.1 * rng() < q * q * Math.pow(1 - q * q, 3.5)) break;
    }
    return q * Math.sqrt(2 * mass) * Math.pow(r * r + b * b, -0.25);
}

// A disk galaxy of n bodies and total mass `mass` in the xz plane, rotating so that its angular
// momentum points along +y:
//   • bulge: a Plummer sphere of mass fraction `bulgeFrac` and scale `bulgeScale`
//   • disk: exponential surface density Σ ∝ e^{−R/R_d} (R from the Gamma(2) distribution,
//     R = −R_d ln(u₁u₂)), truncated at rMax, vertical sech² profile of scale height 0.1·R_d
//   • disk velocities: circular speed from the mass enclosed within the body's spherical radius
//     (counted from the sampled bodies, a spherical approximation of the disk's own field), with
//     the same Plummer softening the forces use: v_c² = G M(<r) r² / (r² + ε²)^{3/2}; plus a radial
//     velocity dispersion from Toomre's stability criterion at Q = TOOMRE_Q (a colder disk
//     fragments into clumps), tangential σ_R/√2 and vertical σ_R/2. The disk is still unstable
//     enough to grow spiral arms and a bar.
//   • bulge velocities: the Plummer distribution of the bulge alone, scaled by √(M(<r)/M_bulge(<r))
//     (at most 2) to account for the disk mass inside its radius
// Tags: 0 + tagBase for disk bodies, 1 + tagBase for bulge bodies.
const TOOMRE_Q = 1.1;

function makeGalaxy(rng, n, mass, opts, eps, tagBase) {
    const { rd, rMax, bulgeFrac, bulgeScale } = opts;
    const nb = Math.round(n * bulgeFrac);
    const m = mass / n;
    const pos = new Float64Array(n * 3);
    const vel = new Float64Array(n * 3);
    const tag = new Uint8Array(n);
    const h = 0.1 * rd;
    for (let i = 0; i < n; i++) {
        if (i < nb) {
            const r = plummerRadius(rng, bulgeScale, 10 * bulgeScale);
            const dir = randomDirection(rng);
            pos[3 * i] = dir[0] * r; pos[3 * i + 1] = dir[1] * r; pos[3 * i + 2] = dir[2] * r;
            tag[i] = tagBase + 1;
        } else {
            let R;
            do { R = -rd * Math.log(Math.max(rng() * rng(), 1e-12)); } while (R > rMax);
            const phi = 2 * Math.PI * rng();
            const u = Math.min(Math.max(rng(), 1e-6), 1 - 1e-6);
            pos[3 * i] = R * Math.cos(phi);
            pos[3 * i + 1] = h * Math.atanh(2 * u - 1);
            pos[3 * i + 2] = R * Math.sin(phi);
            tag[i] = tagBase;
        }
    }
    // Enclosed mass from the sampled bodies, by spherical radius
    const radius = new Float64Array(n);
    for (let i = 0; i < n; i++) radius[i] = Math.hypot(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => radius[a] - radius[b]);
    const enclosed = new Float64Array(n);
    order.forEach((i, rank) => { enclosed[i] = rank * m; });
    const bulgeMass = mass * nb / n;
    const diskMass = mass - bulgeMass;

    for (let i = 0; i < n; i++) {
        const r = radius[i];
        if (i < nb) {
            const b = bulgeScale;
            const mbEnc = bulgeMass * r * r * r / Math.pow(r * r + b * b, 1.5);
            const boost = Math.min(Math.sqrt(Math.max(enclosed[i] / Math.max(mbEnc, 1e-12), 1)), 2);
            const speed = plummerSpeed(rng, r, bulgeMass, b) * boost;
            const dir = randomDirection(rng);
            vel[3 * i] = dir[0] * speed; vel[3 * i + 1] = dir[1] * speed; vel[3 * i + 2] = dir[2] * speed;
        } else {
            const x = pos[3 * i], z = pos[3 * i + 2];
            const R = Math.max(Math.hypot(x, z), 1e-6);
            const vc = Math.sqrt(enclosed[i] * r * r / Math.pow(r * r + eps * eps, 1.5));
            const eR = [x / R, 0, z / R];
            const ePhi = [z / R, 0, -x / R];   // L_y = z·v_x − x·v_z > 0
            // Toomre (1964) stability: σ_R = Q·3.36 G Σ(R) / κ, with κ ≈ √2·v_c/R (flat rotation
            // curve) and the disk's exponential Σ(R); capped at half of v_c
            const sigmaSurface = diskMass / (2 * Math.PI * rd * rd) * Math.exp(-R / rd);
            const kappa = Math.SQRT2 * Math.max(vc, 1e-6) / R;
            const sigmaR = Math.min(TOOMRE_Q * 3.36 * sigmaSurface / kappa, 0.5 * vc);
            const vR = sigmaR * gaussian(rng);
            const vPhi = vc + sigmaR * Math.SQRT1_2 * gaussian(rng);   // epicyclic ratio σ_φ/σ_R = κ/2Ω
            const vY = 0.5 * sigmaR * gaussian(rng);
            vel[3 * i] = eR[0] * vR + ePhi[0] * vPhi;
            vel[3 * i + 1] = vY;
            vel[3 * i + 2] = eR[2] * vR + ePhi[2] * vPhi;
        }
    }
    return { n, pos, vel, tag };
}

function rotateX(g, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    for (const a of [g.pos, g.vel]) {
        for (let i = 0; i < g.n; i++) {
            const y = a[3 * i + 1], z = a[3 * i + 2];
            a[3 * i + 1] = c * y - s * z;
            a[3 * i + 2] = s * y + c * z;
        }
    }
}

function shift(g, dp, dv) {
    for (let i = 0; i < g.n; i++) {
        for (let k = 0; k < 3; k++) {
            g.pos[3 * i + k] += dp[k];
            g.vel[3 * i + k] += dv[k];
        }
    }
}

export const PRESETS = {
    spiral: {
        name: 'Spiral galaxy', distance: 13, elevation: 0.75, speedScale: 1.0,
        build(rng, n, eps) {
            return [makeGalaxy(rng, n, 1, { rd: 1, rMax: 5, bulgeFrac: 0.2, bulgeScale: 0.25 }, eps, 0)];
        },
    },
    collision: {
        name: 'Galaxy collision', distance: 22, elevation: 0.8, speedScale: 1.0,
        // Two equal disks (mass 0.5 each) on a parabolic orbit with pericenter q = 2, starting
        // D = 8 apart. Relative speed √(2GM/D); angle α from the radial direction with
        // sin α = √(q/D) gives the orbital angular momentum √(2GMq). Disk A is tilted 15° (nearly
        // prograde, so it throws long tidal tails, as in Toomre & Toomre 1972), disk B 60°.
        build(rng, n, eps) {
            const opts = { rd: 0.8, rMax: 4, bulgeFrac: 0.2, bulgeScale: 0.2 };
            const na = Math.floor(n / 2);
            const a = makeGalaxy(rng, na, 0.5, opts, eps, 0);
            const b = makeGalaxy(rng, n - na, 0.5, opts, eps, 2);
            rotateX(a, 15 * Math.PI / 180);
            rotateX(b, 60 * Math.PI / 180);
            const D = 8, q = 2, M = 1;
            const v = Math.sqrt(2 * M / D);
            const sinA = Math.sqrt(q / D), cosA = Math.sqrt(1 - q / D);
            const vRel = [-v * cosA, 0, -v * sinA];   // r = (D, 0, 0): L_y = −D·v_z > 0, prograde for A
            shift(a, [-D / 2, 0, 0], vRel.map(c => -c / 2));
            shift(b, [D / 2, 0, 0], vRel.map(c => c / 2));
            return [a, b];
        },
    },
    plummer: {
        name: 'Plummer sphere', distance: 12, elevation: 0.35, speedScale: 0.8,
        build(rng, n) {
            const pos = new Float64Array(n * 3), vel = new Float64Array(n * 3), tag = new Uint8Array(n);
            const radii = new Float64Array(n);
            for (let i = 0; i < n; i++) {
                const r = plummerRadius(rng, 1, 10);
                radii[i] = r;
                const dir = randomDirection(rng);
                const speed = plummerSpeed(rng, r, 1, 1);
                const vdir = randomDirection(rng);
                for (let k = 0; k < 3; k++) {
                    pos[3 * i + k] = dir[k] * r;
                    vel[3 * i + k] = vdir[k] * speed;
                }
            }
            const median = Float64Array.from(radii).sort()[n >> 1];
            for (let i = 0; i < n; i++) tag[i] = radii[i] < median ? 0 : 2;
            return [{ n, pos, vel, tag }];
        },
    },
    collapse: {
        name: 'Cold collapse', distance: 9, elevation: 0.35, speedScale: 1.5,
        // Uniform sphere of radius 2 at rest; free-fall time t_ff = (π/2)·√(R³/2GM) = π
        build(rng, n) {
            const R = 2;
            const pos = new Float64Array(n * 3), vel = new Float64Array(n * 3), tag = new Uint8Array(n);
            for (let i = 0; i < n; i++) {
                const u = rng();
                const r = R * Math.cbrt(u);
                const dir = randomDirection(rng);
                for (let k = 0; k < 3; k++) pos[3 * i + k] = dir[k] * r;
                tag[i] = u < 0.5 ? 0 : 2;   // inner / outer half of the mass
            }
            return [{ n, pos, vel, tag }];
        },
    },
};

// Bodies for a preset: positions (xyz, w = mass 1/N) and velocities (xyz, w = origin tag), with the
// center of mass at rest at the origin
export function generateBodies(presetKey, n, eps, seed = 1) {
    const rng = makeRng(seed);
    const parts = PRESETS[presetKey].build(rng, n, eps);
    const positions = new Float32Array(n * 4);
    const velocities = new Float32Array(n * 4);
    const cm = [0, 0, 0], cv = [0, 0, 0];
    for (const g of parts) {
        for (let i = 0; i < g.n; i++) {
            for (let k = 0; k < 3; k++) {
                cm[k] += g.pos[3 * i + k] / n;
                cv[k] += g.vel[3 * i + k] / n;
            }
        }
    }
    let o = 0;
    for (const g of parts) {
        for (let i = 0; i < g.n; i++, o++) {
            for (let k = 0; k < 3; k++) {
                positions[4 * o + k] = g.pos[3 * i + k] - cm[k];
                velocities[4 * o + k] = g.vel[3 * i + k] - cv[k];
            }
            positions[4 * o + 3] = 1 / n;
            velocities[4 * o + 3] = g.tag[i];
        }
    }
    return { positions, velocities };
}

// Workgroup grid for a 1D job: one row while it fits, else rows of up to maxPerDim
export function dispatchGrid(workgroups, maxPerDim) {
    if (workgroups <= maxPerDim) return [Math.max(workgroups, 1), 1];
    const y = Math.ceil(workgroups / maxPerDim);
    const x = Math.ceil(workgroups / y);
    if (y > maxPerDim) throw new Error(`${workgroups} workgroups exceed the dispatch limit`);
    return [x, y];
}

// ---- Matrix math (column-major, element [col * 4 + row], WebGPU clip depth [0, 1]) ----
export class Mat4 {
    static perspective(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * nf, -1,
            0, 0, far * near * nf, 0
        ]);
    }

    static lookAt(eye, center, up) {
        const z = normalize(subtract(eye, center));
        const x = normalize(cross(up, z));
        const y = cross(z, x);
        return new Float32Array([
            x[0], y[0], z[0], 0,
            x[1], y[1], z[1], 0,
            x[2], y[2], z[2], 0,
            -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
        ]);
    }

    static multiply(a, b) {
        const result = new Float32Array(16);
        for (let col = 0; col < 4; col++) {
            for (let row = 0; row < 4; row++) {
                result[col * 4 + row] =
                    a[0 * 4 + row] * b[col * 4 + 0] +
                    a[1 * 4 + row] * b[col * 4 + 1] +
                    a[2 * 4 + row] * b[col * 4 + 2] +
                    a[3 * 4 + row] * b[col * 4 + 3];
            }
        }
        return result;
    }
}

function normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
}

function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export const FOV = Math.PI / 4;

export class OrbitCamera {
    constructor() {
        this.distance = 13;
        this.rotation = { x: 0.6, y: 0.9 };   // elevation, azimuth
        this.target = [0, 0, 0];
    }

    getEye() {
        const c = Math.cos(this.rotation.x);
        return [
            this.target[0] + this.distance * Math.cos(this.rotation.y) * c,
            this.target[1] + this.distance * Math.sin(this.rotation.x),
            this.target[2] + this.distance * Math.sin(this.rotation.y) * c
        ];
    }

    getBasis() {
        const eye = this.getEye();
        const forward = normalize(subtract(this.target, eye));
        const right = normalize(cross(forward, [0, 1, 0]));
        const up = cross(right, forward);
        return { eye, forward, right, up };
    }

    getViewProj(aspect) {
        const proj = Mat4.perspective(FOV, aspect, 0.05, 500);
        const view = Mat4.lookAt(this.getEye(), this.target, [0, 1, 0]);
        return Mat4.multiply(proj, view);
    }
}

// Sprite size and brightness by N, so the total light stays about the same
export function spriteStyle(n) {
    return {
        pointSize: n <= 4096 ? 3.5 : n <= 16384 ? 3.0 : n <= 65536 ? 2.5 : 2.0,
        brightness: 0.55 * Math.sqrt(16384 / n),
    };
}

export class NBodySimulation {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;

        this.count = 16384;
        this.mode = 'bh';             // 'bh' | 'brute'
        this.presetKey = 'spiral';
        this.theta = 0.5;
        this.softening = 0.05;
        this.dt = 0.01;
        this.stepsPerFrame = 2;
        this.colorMode = 0;           // 0 speed, 1 origin
        this.trails = false;
        this.seed = 1;

        this.time = 0;
        this.stepsTotal = 0;
        this.lastSteps = 0;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;
        this.frameIndex = 0;
        this.trailHead = 0;
        this.generation = 0;

        this.accelDirty = true;
        this.energy = null;           // { K, W, E, interactions, time }
        this.energy0 = null;
        this.baselineTime = 0;
        this.rebaseline = true;
        this.energyPending = false;
        this.lastEnergyTime = 0;

        this.camera = new OrbitCamera();
        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.radixStride = Math.ceil(RADIX_PARAMS_SIZE / align) * align;
        this.maxPerDim = device.limits.maxComputeWorkgroupsPerDimension || 65535;

        this.simData = new ArrayBuffer(SIM_PARAMS_SIZE);
        this.simF32 = new Float32Array(this.simData);
        this.simU32 = new Uint32Array(this.simData);
        this.camData = new ArrayBuffer(CAMERA_SIZE);
        this.camF32 = new Float32Array(this.camData);
        this.camU32 = new Uint32Array(this.camData);

        const U = GPUBufferUsage;
        this.simBuffer = device.createBuffer({ size: SIM_PARAMS_SIZE, usage: U.UNIFORM | U.COPY_DST });
        this.radixBuffer = device.createBuffer({ size: this.radixStride * RADIX_PASSES, usage: U.UNIFORM | U.COPY_DST });
        this.cameraBuffer = device.createBuffer({ size: CAMERA_SIZE, usage: U.UNIFORM | U.COPY_DST });
        this.energyReadback = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });
        this.dummyTrail = device.createBuffer({ size: 16, usage: U.STORAGE });

        this.setupPipelines();
        this.setupTargets();
        this.setupMouseControls();
        this.applyPresetCamera();
        this.allocate();
    }

    setupPipelines() {
        const device = this.device;
        const C = GPUShaderStage.COMPUTE;
        const layout = (types) => device.createBindGroupLayout({
            entries: types.map((type, binding) => ({ binding, visibility: C, buffer: { type } }))
        });
        const S = 'storage', R = 'read-only-storage', UNI = 'uniform';
        this.layouts = {
            integrate: layout([S, S, R, UNI]),
            brute: layout([R, S, UNI]),
            bounds: layout([R, S, UNI]),
            morton: layout([R, R, S, S, UNI]),
            radix: device.createBindGroupLayout({
                entries: [R, R, S, S, S].map((type, binding) => ({ binding, visibility: C, buffer: { type } })).concat([
                    { binding: 5, visibility: C, buffer: { type: UNI, hasDynamicOffset: true, minBindingSize: RADIX_PARAMS_SIZE } }
                ])
            }),
            initNodes: layout([R, R, S, S, UNI]),
            build: layout([R, S, S, S, UNI]),
            bottomUp: layout([R, R, S, S, S, UNI]),
            force: layout([R, R, R, R, R, S, S, UNI]),
            energy: layout([R, R, R, R, S, UNI]),
            trail: layout([R, S, UNI]),
        };
        const make = (code, layoutName, entries) => {
            const module = device.createShaderModule({ code });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layouts[layoutName]] });
            const out = {};
            for (const entryPoint of entries) {
                out[entryPoint] = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
            }
            return out;
        };
        this.pipelines = {
            ...make(integrateShader, 'integrate', ['cs_kick_drift', 'cs_kick']),
            ...make(bruteShader, 'brute', ['cs_brute_force']),
            ...make(boundsShader, 'bounds', ['cs_bounds_partial', 'cs_bounds_final']),
            ...make(mortonShader, 'morton', ['cs_morton']),
            ...make(radixShader, 'radix', ['cs_radix_histogram', 'cs_radix_scan', 'cs_radix_scatter']),
            ...make(initNodesShader, 'initNodes', ['cs_init_nodes']),
            ...make(buildShader, 'build', ['cs_build_tree']),
            ...make(bottomUpShader, 'bottomUp', ['cs_bottom_up']),
            ...make(forceShader, 'force', ['cs_bh_force']),
            ...make(energyShader, 'energy', ['cs_energy_partial', 'cs_energy_final']),
            ...make(trailShader, 'trail', ['cs_trail_record', 'cs_trail_fill']),
        };

        const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
        const V = GPUShaderStage.VERTEX;
        this.renderLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
                { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } },
            ]
        });
        const renderModule = device.createShaderModule({ code: renderShader });
        const additive = {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        };
        const renderPipeline = (vs, fs, topology) => device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
            vertex: { module: renderModule, entryPoint: vs },
            fragment: { module: renderModule, entryPoint: fs, targets: [{ format: 'rgba16float', blend: additive }] },
            primitive: { topology },
        });
        this.bodyPipeline = renderPipeline('vs_body', 'fs_body', 'triangle-list');
        this.trailPipeline = renderPipeline('vs_trail', 'fs_trail', 'line-list');

        this.tonemapLayout = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }]
        });
        const tonemapModule = device.createShaderModule({ code: tonemapShader });
        this.tonemapPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.tonemapLayout] }),
            vertex: { module: tonemapModule, entryPoint: 'vs_fullscreen' },
            fragment: { module: tonemapModule, entryPoint: 'fs_tonemap', targets: [{ format: 'bgra8unorm' }] },
            primitive: { topology: 'triangle-list' },
        });
    }

    setupTargets() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: 'bgra8unorm', alphaMode: 'opaque' });
        this.hdrTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.tonemapBindGroup = this.device.createBindGroup({
            layout: this.tonemapLayout,
            entries: [{ binding: 0, resource: this.hdrTexture.createView() }]
        });
    }

    // (Re)creates every N-sized buffer and bind group, then loads the preset
    allocate() {
        this.generation++;
        this.destroyBodyBuffers();
        const device = this.device;
        const n = this.count;
        if (n < 1 || n > MAX_BH) throw new Error(`N = ${n} is outside 1 … ${MAX_BH}`);
        const nodes = 2 * n - 1;
        const internal = Math.max(n - 1, 1);
        this.numPartials = Math.ceil(n / REDUCE_TILE);
        this.sortTiles = Math.ceil(n / RADIX_TILE);
        const U = GPUBufferUsage;
        const storage = (size, extra = 0) => device.createBuffer({ size: Math.max(size, 16), usage: U.STORAGE | extra });
        const b = {
            positions: storage(16 * n, U.COPY_DST),
            velocities: storage(16 * n, U.COPY_DST),
            accel: storage(16 * n),
            interactions: storage(4 * n),
            keysA: storage(4 * n), valsA: storage(4 * n),
            keysB: storage(4 * n), valsB: storage(4 * n),
            offsets: storage(4 * 16 * this.sortTiles),
            children: storage(8 * internal),
            parents: storage(4 * nodes),
            visits: storage(4 * internal),
            nodeMass: storage(16 * nodes),
            nodeBox: storage(32 * nodes),
            bounds: storage(16 * (2 + 2 * this.numPartials)),
            energy: storage(16 * (1 + this.numPartials), U.COPY_SRC),
        };
        this.buffers = b;
        this.allocateTrail();

        // Radix passes: one 16-byte RadixParams per dynamic-offset slot (layout at the top of the file)
        const radixData = new ArrayBuffer(this.radixStride * RADIX_PASSES);
        const radixU32 = new Uint32Array(radixData);
        for (let p = 0; p < RADIX_PASSES; p++) {
            const o = p * this.radixStride / 4;
            radixU32[o + 0] = 4 * p;
            radixU32[o + 1] = n;
            radixU32[o + 2] = this.sortTiles;
            radixU32[o + 3] = 0;
        }
        device.queue.writeBuffer(this.radixBuffer, 0, radixData);

        const group = (layoutName, buffers, dynamic) => device.createBindGroup({
            layout: this.layouts[layoutName],
            entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })).concat(dynamic ? [
                { binding: buffers.length, resource: { buffer: this.radixBuffer, offset: 0, size: RADIX_PARAMS_SIZE } }
            ] : [])
        });
        const sim = this.simBuffer;
        this.groups = {
            integrate: group('integrate', [b.positions, b.velocities, b.accel, sim]),
            brute: group('brute', [b.positions, b.accel, sim]),
            bounds: group('bounds', [b.positions, b.bounds, sim]),
            morton: group('morton', [b.positions, b.bounds, b.keysA, b.valsA, sim]),
            radixAB: group('radix', [b.keysA, b.valsA, b.keysB, b.valsB, b.offsets], true),
            radixBA: group('radix', [b.keysB, b.valsB, b.keysA, b.valsA, b.offsets], true),
            initNodes: group('initNodes', [b.valsA, b.positions, b.nodeMass, b.nodeBox, sim]),
            build: group('build', [b.keysA, b.children, b.parents, b.visits, sim]),
            bottomUp: group('bottomUp', [b.parents, b.children, b.visits, b.nodeMass, b.nodeBox, sim]),
            force: group('force', [b.positions, b.valsA, b.nodeMass, b.nodeBox, b.children, b.accel, b.interactions, sim]),
            energy: group('energy', [b.positions, b.velocities, b.accel, b.interactions, b.energy, sim]),
        };
        this.reset();
    }

    // The trail ring buffer exists only while trails are on (32 MB at the largest N)
    allocateTrail() {
        if (this.trailBuffer) {
            this.trailBuffer.destroy();
            this.trailBuffer = null;
        }
        const b = this.buffers;
        if (this.trails) {
            this.trailBuffer = this.device.createBuffer({ size: 16 * this.count * TRAIL_LEN, usage: GPUBufferUsage.STORAGE });
            this.trailGroup = this.device.createBindGroup({
                layout: this.layouts.trail,
                entries: [b.positions, this.trailBuffer, this.simBuffer].map((buffer, binding) => ({ binding, resource: { buffer } }))
            });
            this.trailFillPending = true;
        } else {
            this.trailGroup = null;
        }
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderLayout,
            entries: [this.cameraBuffer, b.positions, b.velocities, this.trailBuffer || this.dummyTrail]
                .map((buffer, binding) => ({ binding, resource: { buffer } }))
        });
    }

    destroyBodyBuffers() {
        if (this.buffers) {
            for (const buffer of Object.values(this.buffers)) buffer.destroy();
            this.buffers = null;
        }
        if (this.trailBuffer) {
            this.trailBuffer.destroy();
            this.trailBuffer = null;
        }
    }

    // Puts the bodies back to the preset's initial state; renders one frame (which computes the
    // initial forces and measures E₀)
    reset() {
        this.generation++;
        const { positions, velocities } = generateBodies(this.presetKey, this.count, this.softening, this.seed);
        const q = this.device.queue;
        q.writeBuffer(this.buffers.positions, 0, positions);
        q.writeBuffer(this.buffers.velocities, 0, velocities);
        this.time = 0;
        this.stepsTotal = 0;
        this.lastSteps = 0;
        this.accelDirty = true;
        this.energy = null;
        this.energy0 = null;
        this.rebaseline = true;
        this.trailFillPending = this.trails;
        this.render();
    }

    applyPresetCamera() {
        const p = PRESETS[this.presetKey];
        this.camera.distance = p.distance;
        this.camera.rotation = { x: p.elevation, y: 0.9 };
        this.camera.target = [0, 0, 0];
    }

    loadPreset(key) {
        if (!PRESETS[key]) return;
        this.presetKey = key;
        this.applyPresetCamera();
        this.reset();
    }

    setCount(n) {
        if (!COUNTS.includes(n)) return;
        this.count = n;
        this.allocate();
    }

    // Mode, θ or ε changed: forces are recomputed before the next kick and E₀ is measured again
    paramsChanged() {
        this.accelDirty = true;
        this.rebaseline = true;
        if (!this.running) this.render();
    }

    setTrails(on) {
        this.trails = on;
        this.allocateTrail();
        if (!this.running) this.render();
    }

    // ---- Mouse: drag rotates, right-drag pans, wheel zooms ----
    setupMouseControls() {
        let mode = null;
        let lastX = 0;
        let lastY = 0;
        // Canvas pixels per CSS pixel, for drag distances
        const scale = () => {
            const rect = this.canvas.getBoundingClientRect();
            return [this.canvas.width / rect.width, this.canvas.height / rect.height];
        };
        this.onMouseDown = (e) => {
            e.preventDefault();
            lastX = e.clientX;
            lastY = e.clientY;
            mode = e.button === 2 ? 'pan' : e.button === 0 ? 'rotate' : null;
        };
        this.onMouseMove = (e) => {
            if (!mode) return;
            const [sx, sy] = scale();
            const dx = (e.clientX - lastX) * sx;
            const dy = (e.clientY - lastY) * sy;
            lastX = e.clientX;
            lastY = e.clientY;
            if (mode === 'pan') {
                const { right, up } = this.camera.getBasis();
                const s = this.camera.distance * 2 * Math.tan(FOV / 2) / this.canvas.height;
                for (let k = 0; k < 3; k++) this.camera.target[k] += (-right[k] * dx + up[k] * dy) * s;
            } else {
                this.camera.rotation.y += dx * 0.006;
                this.camera.rotation.x = Math.max(-1.5, Math.min(1.5, this.camera.rotation.x + dy * 0.006));
            }
            if (!this.running) this.requestRender();
        };
        this.onMouseUp = () => { mode = null; };
        this.onWheel = (e) => {
            e.preventDefault();
            this.camera.distance = Math.max(1, Math.min(150, this.camera.distance * (1 + e.deltaY * 0.001)));
            if (!this.running) this.requestRender();
        };
        this.onContextMenu = (e) => e.preventDefault();
        this.canvas.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
    }

    requestRender() {
        if (this.renderPending) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            if (!this.destroyed && !this.running) this.drawOnly();
        });
    }

    effectiveMode() {
        return this.mode === 'brute' && this.count <= MAX_BRUTE ? 'brute' : 'bh';
    }

    // Writes SimParams (layout table at the top of the file)
    writeSimParams() {
        const f = this.simF32;
        const u = this.simU32;
        u[0] = this.count;
        u[1] = Math.ceil(this.count / WG);
        f[2] = this.dt;
        f[3] = this.softening * this.softening;
        f[4] = this.theta;
        u[5] = this.numPartials;
        u[6] = this.trailHead;
        u[7] = 0;
        this.device.queue.writeBuffer(this.simBuffer, 0, this.simData);
    }

    // Writes Camera (layout table at the top of the file)
    writeCamera() {
        const f = this.camF32;
        const u = this.camU32;
        f.set(this.camera.getViewProj(this.canvas.width / this.canvas.height), 0);
        f[16] = this.canvas.width;
        f[17] = this.canvas.height;
        const style = spriteStyle(this.count);
        f[18] = style.pointSize;
        f[19] = style.brightness;
        u[20] = this.colorMode;
        u[21] = this.count;
        f[22] = PRESETS[this.presetKey].speedScale;
        u[23] = this.trailHead;
        u[24] = TRAIL_LEN;
        u[25] = 0; u[26] = 0; u[27] = 0;
        this.device.queue.writeBuffer(this.cameraBuffer, 0, this.camData);
    }

    dispatch(pass, pipeline, group, workgroups, offsets) {
        const [x, y] = dispatchGrid(workgroups, this.maxPerDim);
        pass.setPipeline(pipeline);
        if (offsets) pass.setBindGroup(0, group, offsets); else pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(x, y);
    }

    bodyGroups() {
        return Math.ceil(this.count / WG);
    }

    // a(x) and φ(x) for the current positions: 1 dispatch (brute force) or 31 (Barnes-Hut)
    encodeForces(pass) {
        const P = this.pipelines;
        const G = this.groups;
        const bodies = this.bodyGroups();
        if (this.effectiveMode() === 'brute') {
            this.dispatch(pass, P.cs_brute_force, G.brute, bodies);
            return;
        }
        this.dispatch(pass, P.cs_bounds_partial, G.bounds, this.numPartials);
        this.dispatch(pass, P.cs_bounds_final, G.bounds, 1);
        this.dispatch(pass, P.cs_morton, G.morton, bodies);
        for (let p = 0; p < RADIX_PASSES; p++) {
            const group = p % 2 === 0 ? G.radixAB : G.radixBA;
            const offset = [p * this.radixStride];
            this.dispatch(pass, P.cs_radix_histogram, group, this.sortTiles, offset);
            this.dispatch(pass, P.cs_radix_scan, group, 1, offset);
            this.dispatch(pass, P.cs_radix_scatter, group, this.sortTiles, offset);
        }
        // An even number of passes leaves the sorted pairs in keysA / valsA
        this.dispatch(pass, P.cs_init_nodes, G.initNodes, bodies);
        this.dispatch(pass, P.cs_build_tree, G.build, Math.ceil(Math.max(this.count - 1, 1) / WG));
        this.dispatch(pass, P.cs_bottom_up, G.bottomUp, bodies);
        this.dispatch(pass, P.cs_bh_force, G.force, bodies);
    }

    dispatchesPerStep() {
        return this.effectiveMode() === 'brute' ? 3 : 33;
    }

    // Leapfrog steps; each step after the first goes into its own command buffer, so no single
    // submission carries more than one step of force work. Returns the encoder of the last one.
    encodeCompute(encoder, steps, sampleEnergy, recordTrail) {
        const P = this.pipelines;
        const G = this.groups;
        const bodies = this.bodyGroups();
        let pass = encoder.beginComputePass();
        if (this.accelDirty) {
            this.encodeForces(pass);
            this.accelDirty = false;
        }
        for (let s = 0; s < steps; s++) {
            if (s > 0) {
                pass.end();
                this.device.queue.submit([encoder.finish()]);
                encoder = this.device.createCommandEncoder();
                pass = encoder.beginComputePass();
            }
            this.dispatch(pass, P.cs_kick_drift, G.integrate, bodies);
            this.encodeForces(pass);
            this.dispatch(pass, P.cs_kick, G.integrate, bodies);
        }
        if (this.trailGroup && this.trailFillPending) {
            this.dispatch(pass, P.cs_trail_fill, this.trailGroup, bodies);
            this.trailFillPending = false;
        } else if (recordTrail) {
            this.dispatch(pass, P.cs_trail_record, this.trailGroup, bodies);
        }
        if (sampleEnergy) {
            this.dispatch(pass, P.cs_energy_partial, G.energy, this.numPartials);
            this.dispatch(pass, P.cs_energy_final, G.energy, 1);
        }
        pass.end();
        return encoder;
    }

    // Body-body (or body-node) interactions of one force evaluation: exact for brute force, the
    // last measured count for Barnes-Hut (an estimate of 2,000 per body before the first sample)
    interactionsPerStep() {
        const n = this.count;
        if (this.effectiveMode() === 'brute') return n * (n - 1);
        return this.energy && this.energy.interactions > 0 ? this.energy.interactions : 2000 * n;
    }

    // Steps this frame: the Speed setting, reduced (to at least 1) when it would exceed
    // WORK_BUDGET interactions per frame
    stepsThisFrame() {
        const wanted = Math.min(Math.max(this.stepsPerFrame, 1), MAX_STEPS_PER_FRAME);
        return Math.max(1, Math.min(wanted, Math.floor(WORK_BUDGET / this.interactionsPerStep())));
    }

    encodeRender(encoder) {
        const hdrPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.hdrTexture.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 0 }
            }]
        });
        hdrPass.setBindGroup(0, this.renderBindGroup);
        if (this.trailGroup) {
            hdrPass.setPipeline(this.trailPipeline);
            hdrPass.draw(2 * TRAIL_LEN, this.count);
        }
        hdrPass.setPipeline(this.bodyPipeline);
        hdrPass.draw(6, this.count);
        hdrPass.end();

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });
        pass.setPipeline(this.tonemapPipeline);
        pass.setBindGroup(0, this.tonemapBindGroup);
        pass.draw(3);
        pass.end();
    }

    // `steps` leapfrog steps, then (maybe) a trail record and an energy sample, then the render
    frame(steps, forceSample = false) {
        const now = performance.now();
        const recordTrail = !!this.trailGroup && steps > 0 && (this.frameIndex++ % TRAIL_EVERY === 0);
        if (recordTrail) this.trailHead = (this.trailHead + 1) % TRAIL_LEN;
        const sampleEnergy = !this.energyPending &&
            (forceSample || this.rebaseline || now - this.lastEnergyTime >= ENERGY_INTERVAL_MS);
        this.writeSimParams();
        this.writeCamera();
        const encoder = this.encodeCompute(this.device.createCommandEncoder(), steps, sampleEnergy, recordTrail);
        if (sampleEnergy) encoder.copyBufferToBuffer(this.buffers.energy, 0, this.energyReadback, 0, 16);
        this.encodeRender(encoder);
        this.device.queue.submit([encoder.finish()]);
        this.time += steps * this.dt;
        this.stepsTotal += steps;
        this.lastSteps = steps;
        if (sampleEnergy) this.readEnergy(now);
    }

    // The 16-byte energy readback (K, W, interaction count); at most one in flight
    readEnergy(now) {
        this.energyPending = true;
        this.lastEnergyTime = now;
        const generation = this.generation;
        const time = this.time;
        const baseline = this.rebaseline;
        this.rebaseline = false;
        const readback = this.energyReadback;
        readback.mapAsync(GPUMapMode.READ).then(() => {
            const [K, W, interactions] = new Float32Array(readback.getMappedRange().slice(0));
            readback.unmap();
            this.energyPending = false;
            if (this.destroyed) return;
            if (generation === this.generation) {
                this.energy = { K, W, E: K + W, interactions, time };
                if (baseline || !this.energy0) {
                    this.energy0 = this.energy;
                    this.baselineTime = time;
                }
                if (this.onStats) this.onStats();
            }
            // A reset or parameter change while this read was in flight skipped its own sample
            if (this.rebaseline && !this.running) this.render();
        }).catch(() => { this.energyPending = false; });
    }

    render() {
        this.frame(0, true);
    }

    // Redraw without touching the simulation (camera moves while paused)
    drawOnly() {
        this.writeCamera();
        const encoder = this.device.createCommandEncoder();
        this.encodeRender(encoder);
        this.device.queue.submit([encoder.finish()]);
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
        if (!this.running || this.destroyed) return;
        const now = performance.now();
        this.frame(this.stepsThisFrame());
        this.framesSinceSample++;
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
        this.canvas.removeEventListener('mousedown', this.onMouseDown);
        window.removeEventListener('mousemove', this.onMouseMove);
        window.removeEventListener('mouseup', this.onMouseUp);
        this.canvas.removeEventListener('wheel', this.onWheel);
        this.canvas.removeEventListener('contextmenu', this.onContextMenu);
        this.destroyBodyBuffers();
        for (const b of [this.simBuffer, this.radixBuffer, this.cameraBuffer, this.energyReadback,
            this.dummyTrail, this.hdrTexture]) b.destroy();
    }
}

export async function initNBody(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-nbody');
    const output = document.getElementById('output-nbody');

    const simulation = new NBodySimulation(device, canvas);
    simulation.onStats = () => updateOutput();
    let notice = '';

    document.getElementById('btn-nbody-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });
    document.getElementById('btn-nbody-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });
    document.getElementById('btn-nbody-reset').addEventListener('click', () => {
        simulation.reset();
        updateOutput();
    });

    const countSelect = document.getElementById('nbody-count');
    const modeSelect = document.getElementById('nbody-mode');

    // Brute force is O(N²): keep it at N ≤ 32,768 by switching whichever control was not just changed
    countSelect.addEventListener('change', () => {
        const n = parseInt(countSelect.value);
        if (!COUNTS.includes(n)) return;
        notice = '';
        if (simulation.mode === 'brute' && n > MAX_BRUTE) {
            simulation.mode = 'bh';
            modeSelect.value = 'bh';
            notice = `Brute force is limited to ${MAX_BRUTE.toLocaleString()} bodies (O(N²)); switched to Barnes-Hut.`;
        }
        simulation.setCount(n);
        updateOutput();
    });
    modeSelect.addEventListener('change', () => {
        const mode = modeSelect.value === 'brute' ? 'brute' : 'bh';
        notice = '';
        simulation.mode = mode;
        if (mode === 'brute' && simulation.count > MAX_BRUTE) {
            countSelect.value = String(MAX_BRUTE);
            notice = `Brute force is limited to ${MAX_BRUTE.toLocaleString()} bodies (O(N²)); N set to ${MAX_BRUTE.toLocaleString()}.`;
            simulation.setCount(MAX_BRUTE);
        } else {
            simulation.paramsChanged();
        }
        updateOutput();
    });

    // The browser may have restored earlier control values
    const restoredCount = parseInt(countSelect.value);
    if (COUNTS.includes(restoredCount) && restoredCount !== simulation.count) simulation.setCount(restoredCount);
    if (modeSelect.value === 'brute') {
        simulation.mode = 'brute';
        if (simulation.count > MAX_BRUTE) {
            countSelect.value = String(MAX_BRUTE);
            simulation.setCount(MAX_BRUTE);
        } else {
            simulation.paramsChanged();
        }
    }

    const slider = (id, parse, apply, format, changesForces) => {
        const input = document.getElementById(id);
        const label = document.getElementById(`${id}-value`);
        const update = () => {
            apply(parse(input.value));
            label.textContent = format();
            if (changesForces) simulation.paramsChanged();
            updateOutput();
        };
        input.addEventListener('input', update);
        apply(parse(input.value));
        label.textContent = format();
    };
    slider('nbody-theta', parseFloat, (v) => { simulation.theta = Math.min(Math.max(v, MIN_THETA), 1.0); },
        () => simulation.theta.toFixed(2), true);
    slider('nbody-softening', parseFloat, (v) => { simulation.softening = Math.min(Math.max(v, 0.005), 0.2); },
        () => simulation.softening.toFixed(3), true);
    slider('nbody-dt', parseFloat, (v) => { simulation.dt = Math.min(Math.max(v, 0.001), 0.02); },
        () => simulation.dt.toFixed(3), false);
    slider('nbody-speed', parseInt, (v) => { simulation.stepsPerFrame = Math.min(Math.max(v, 1), MAX_STEPS_PER_FRAME); },
        () => `${simulation.stepsPerFrame}×`, false);

    const colorSelect = document.getElementById('nbody-color');
    const applyColor = () => {
        simulation.colorMode = colorSelect.value === 'origin' ? 1 : 0;
        if (!simulation.running) simulation.drawOnly();
    };
    colorSelect.addEventListener('change', applyColor);
    applyColor();

    const trailsBox = document.getElementById('nbody-trails');
    trailsBox.addEventListener('change', () => {
        simulation.setTrails(trailsBox.checked);
        updateOutput();
    });
    if (trailsBox.checked) simulation.setTrails(true);

    const presetButtons = document.querySelectorAll('.nbody-preset');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!PRESETS[btn.dataset.preset]) return;
            simulation.loadPreset(btn.dataset.preset);
            showPreset();
            updateOutput();
        });
    });

    function showPreset() {
        presetButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === simulation.presetKey));
    }

    function updateOutput() {
        const n = simulation.count;
        const mode = simulation.effectiveMode();
        const wanted = Math.min(Math.max(simulation.stepsPerFrame, 1), MAX_STEPS_PER_FRAME);
        const steps = simulation.stepsThisFrame();
        const e = simulation.energy;
        const e0 = simulation.energy0;
        const perStep = mode === 'brute' || (e && e.interactions > 0) ? simulation.interactionsPerStep() : null;
        const perFrame = perStep === null ? 'measuring…'
            : `${(perStep * steps / 1e6).toFixed(1)} M (${(perStep / n).toFixed(0)} per body per step${mode === 'bh' ? ', measured' : ''})`;
        const limited = steps < wanted ? ` (Speed ${wanted}× limited to ${steps} by the ${WORK_BUDGET / 1e9}·10⁹ interactions/frame budget)` : '';
        let energyLine = 'measuring…';
        if (e && e0) {
            const drift = (e.E - e0.E) / Math.abs(e0.E);
            const span = e.time - simulation.baselineTime;
            energyLine = `${drift >= 0 ? '+' : ''}${(drift * 100).toFixed(3)}% over t = ${span.toFixed(1)} (E₀ = ${e0.E.toFixed(4)}, K = ${e.K.toFixed(4)}, W = ${e.W.toFixed(4)}, virial 2K/|W| = ${(2 * e.K / Math.abs(e.W)).toFixed(2)})`;
        }
        const fps = simulation.running ? simulation.fps.toFixed(0) : 0;
        const algo = mode === 'brute'
            ? `Brute force O(N²), 256-body tiles in workgroup memory`
            : `Barnes-Hut O(N log N), θ = ${simulation.theta.toFixed(2)}, tree rebuilt on the GPU every step`;

        output.innerHTML = `<span class="success">✓ N-Body Simulation Active</span>${notice ? `\n<span class="info">${notice}</span>` : ''}

<span class="info">Configuration:</span>
• Bodies: ${n.toLocaleString()} · Preset: ${PRESETS[simulation.presetKey].name}
• Mode: ${algo}
• Softening ε = ${simulation.softening.toFixed(3)} · dt = ${simulation.dt.toFixed(3)} · ${steps} leapfrog step${steps > 1 ? 's' : ''} per frame${limited}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'} · t = ${simulation.time.toFixed(2)} (${simulation.stepsTotal.toLocaleString()} steps)

<span class="info">Algorithm:</span>
• Leapfrog kick-drift-kick with fixed dt; Plummer-softened gravity, G = 1, total mass 1
• Barnes-Hut: bounding box reduction → 30-bit Morton codes → 8-pass radix sort → Karras (2012) radix tree → bottom-up mass/COM with atomic visit counters → stack traversal, open when d &lt; l/θ + δ
• Brute force: GPU Gems 3 ch. 31 tiling

<span class="info">Performance:</span>
• Frame rate: ${fps} FPS
• Interactions per frame: ${perFrame}
• Compute dispatches per step: ${simulation.dispatchesPerStep()}${simulation.trails ? ` · trails: ${TRAIL_LEN} positions per body` : ''}
• Energy drift (E − E₀)/|E₀|: ${energyLine}
• Readbacks: 16 bytes (K, W, interactions) once per second`;
    }

    showPreset();
    updateOutput();
    return simulation;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'integrate', code: integrateShader },
        { name: 'brute', code: bruteShader },
        { name: 'bounds', code: boundsShader },
        { name: 'morton', code: mortonShader },
        { name: 'radix', code: radixShader },
        { name: 'initNodes', code: initNodesShader },
        { name: 'build', code: buildShader },
        { name: 'bottomUp', code: bottomUpShader },
        { name: 'force', code: forceShader },
        { name: 'energy', code: energyShader },
        { name: 'trail', code: trailShader },
        { name: 'render', code: renderShader },
        { name: 'tonemap', code: tonemapShader },
    ];
}
