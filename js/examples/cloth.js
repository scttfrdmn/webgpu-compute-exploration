// Cloth simulation with position-based dynamics (PBD) and Verlet integration on the GPU.
//
// A rectangular n × n grid of particles (n = 32 … 128) is held together by three kinds of distance
// constraints: structural (horizontal and vertical neighbors), shear (both cell diagonals) and bend
// (two particles apart). Each fixed substep of h = 1/240 s runs, all in one compute pass:
//   cs_integrate  Verlet: x' = x + damping·(x − x_prev) + a·h², a = gravity + wind/air drag
//   cs_tether     long-range attachments (see below)
//   cs_solve      `iterations` × NUM_COLORS dispatches, one per constraint color (see below)
//   cs_collide    push particles out of the sphere and above the floor, with friction
// and once per frame cs_normals computes per-vertex normals for lighting and the next frame's wind.
//
// Race-free parallel solving: graph coloring. A thread projects one constraint and writes both of
// its particles, so two constraints that share a particle must never run in the same dispatch.
// The grid structure gives an exact 12-coloring: every constraint kind (horizontal, vertical,
// diagonal, anti-diagonal, horizontal bend, vertical bend) splits into two independent sets by the
// parity of its start column/row (bend links span two cells, so they split by floor(i / 2) parity).
// Constraints are stored sorted by color; each dispatch reads its [start, count) range through a
// dynamic uniform offset. This is a Gauss-Seidel solve: every color sees the corrections of the
// colors before it, so it converges much faster than a Jacobi solve (accumulate corrections per
// particle, then average), which also needs atomics or a second gather pass and under-relaxation.
//
// Long-range attachments (tethers, Kim et al. 2012): a few solver iterations cannot carry the
// weight of 128 rows up to the pins, so links next to a pin would stretch to about twice their
// length. Each particle therefore also gets a unilateral tether to its two nearest pins: it may
// not be farther from a pin than in the flat rest layout. A thread only writes its own particle
// (pins never move in the solver), so one dispatch per substep is race-free. Once anything has
// torn the tethers switch off (tornCount > 0), since a torn-off piece is no longer held by them.
//
// Tearing: when a structural or shear link is stretched beyond tearRatio × its rest length it sets
// its flag in `broken`, and every constraint skips broken links from then on. Bend links (and the
// anti-diagonal of a cell) depend on "parent" links they span and break with them, so nothing holds
// a tear shut invisibly. The mesh draws a triangle only where its three edges are intact.
//
// Picking: shift-mousedown copies the particle positions once to a mappable buffer; the CPU then
// finds the particle closest to the mouse ray (within a few pixels, nearest to the camera). While
// the button is held, that particle and the ones within grabRadius grid cells of it (a small
// "grip", like fingers pinching fabric) move kinematically (inverse mass 0) with the mouse ray on
// a camera-parallel plane through the grab point: each substep translates them by grabDelta. A
// single kinematic particle would just rip out of the cloth; the grip spreads the pull so a
// tear opens in the fabric around it. A 4-byte torn-link counter is read back every few frames
// while tearing is on. There are no other readbacks.
//
// Simulation uniform layout (struct SimParams, 128 bytes):
//   offset   0  gravity     vec4<f32>  xyz acceleration (m/s²)               -> Float32 view [0..3]
//   offset  16  wind        vec4<f32>  xyz mean wind velocity, w gustiness   -> Float32 view [4..7]
//   offset  32  sphere      vec4<f32>  xyz center, w radius                  -> Float32 view [8..11]
//   offset  48  grabDelta   vec4<f32>  xyz grip translation per substep      -> Float32 view [12..15]
//   offset  64  stiffness   vec4<f32>  per-iteration k: structural, shear, bend -> Float32 view [16..19]
//   offset  80  dt          f32        substep h (s)                         -> Float32 view [20]
//   offset  84  damping     f32        velocity kept per substep             -> Float32 view [21]
//   offset  88  floorY      f32        floor height                          -> Float32 view [22]
//   offset  92  friction    f32        Coulomb friction coefficient μ        -> Float32 view [23]
//   offset  96  time        f32        simulated time (s), drives the gusts  -> Float32 view [24]
//   offset 100  tearRatio   f32        break when length > tearRatio × rest  -> Float32 view [25]
//   offset 104  tearOn      u32        1 = tearing enabled                   -> Uint32 view [26]
//   offset 108  grabIndex   u32        grabbed particle (grip center) or NONE -> Uint32 view [27]
//   offset 112  gridN       u32        particles per side                    -> Uint32 view [28]
//   offset 116  sphereOn    u32        1 = sphere collider present           -> Uint32 view [29]
//   offset 120  thickness   f32        collision margin (m)                  -> Float32 view [30]
//   offset 124  grabRadius  u32        grip half-size in grid cells          -> Uint32 view [31]
//
// Per-color uniform (struct ColorRange, 16 bytes, one slot per color at a dynamic offset):
//   offset 0 start u32 [0] · 4 count u32 [1] · 8 pad0 u32 [2] · 12 pad1 u32 [3]
//
// Render uniform layout (struct Camera, 144 bytes):
//   offset   0  viewProj   mat4x4<f32>  column-major, depth [0, 1]           -> Float32 view [0..15]
//   offset  64  eye        vec4<f32>    camera position                      -> Float32 view [16..19]
//   offset  80  sphere     vec4<f32>    xyz center, w radius                 -> Float32 view [20..23]
//   offset  96  lightDir   vec4<f32>    xyz unit vector toward the light     -> Float32 view [24..27]
//   offset 112  viewport   vec2<f32>    canvas size in pixels                -> Float32 view [28], [29]
//   offset 120  floorY     f32                                               -> Float32 view [30]
//   offset 124  pointSize  f32          particle marker size in pixels       -> Float32 view [31]
//   offset 128  gridN      u32                                               -> Uint32 view [32]
//   offset 132  sphereOn   u32                                               -> Uint32 view [33]
//   offset 136  grabIndex  u32                                               -> Uint32 view [34]
//   offset 140  pad0       u32                                               -> Uint32 view [35]
//
// Storage: positions/previous array<vec4<f32>> (xyz, w = inverse mass, 0 = pinned), constraints
// array<Constraint> (32 bytes: a, b, rest, kind, parentA, parentB, 2 pad), broken array<u32>,
// links array<Links> (48 bytes per particle: the constraint indices of its four structural links
// and the cell diagonal, its two tether pins and their rest distances, 3 pad), normals
// array<vec4<f32>>, tornCount atomic<u32>. Seven storage buffers in the compute stage (limit 8).
//
// Compute-only use (the Measure tab, js/teach/experiments/cloth.js, and test/gpu/): a ClothSimulation
// with canvas = null creates no render targets or mouse handlers and never draws. measureCloth() runs
// the shipped compute pass (encodeCompute, MAX_SUBSTEPS substeps per command buffer, as in a frame),
// times a few chunks and reads the positions back once at the end.

import { GpuTimer } from '../gpu-timer.js';

export const SUBSTEP_DT = 1 / 240;
export const MAX_SUBSTEPS = 8;           // per animation frame; slower frames drop simulated time
export const NUM_COLORS = 12;
export const NONE = 0xffffffff;
export const KIND_STRUCTURAL = 0;
export const KIND_SHEAR = 1;
export const KIND_BEND = 2;
export const SHEAR_FACTOR = 0.5;         // shear stiffness relative to the stiffness slider (fabric shears easily)
export const BEND_FACTOR = 0.2;          // bend stiffness relative to the stiffness slider
export const DAMPING_RATE = 0.4;         // 1/s, velocity decay on top of air drag
export const FRICTION = 0.6;             // Coulomb friction coefficient μ for sphere and floor contacts
export const THICKNESS = 0.015;          // m, particles stay this far from colliders
export const FLOOR_Y = -1.2;
export const MAX_SPEED = 25.0;           // m/s, safety clamp in cs_integrate
export const WIND_NORMAL_DRAG = 1.0;     // 1/s, pressure drag along the cloth normal
export const WIND_TANGENT_DRAG = 0.5;    // 1/s, drag along the cloth surface (stands in for flutter form drag)
export const WORKGROUP = 64;

const SIM_PARAMS_SIZE = 128;
const COLOR_RANGE_SIZE = 16;
const CAMERA_SIZE = 144;
const CONSTRAINT_WORDS = 8;
const LINK_WORDS = 12;
const MSAA = 4;

export const simStructs = `
struct SimParams {
    gravity: vec4<f32>,
    wind: vec4<f32>,
    sphere: vec4<f32>,
    grabDelta: vec4<f32>,
    stiffness: vec4<f32>,
    dt: f32,
    damping: f32,
    floorY: f32,
    friction: f32,
    time: f32,
    tearRatio: f32,
    tearOn: u32,
    grabIndex: u32,
    gridN: u32,
    sphereOn: u32,
    thickness: f32,
    grabRadius: u32,
}

struct ColorRange {
    start: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
}
`;

export const sharedStructs = `
const NONE: u32 = 0xffffffffu;
const KIND_BEND: u32 = ${KIND_BEND}u;

struct Constraint {
    a: u32,
    b: u32,
    rest: f32,
    kind: u32,
    parentA: u32,   // links this one depends on (NONE if none); it breaks when they break
    parentB: u32,
    pad0: u32,
    pad1: u32,
}

struct Links {
    xNeg: u32,      // structural link to (i - 1, j)
    xPos: u32,      // structural link to (i + 1, j)
    yNeg: u32,      // structural link to (i, j - 1)
    yPos: u32,      // structural link to (i, j + 1)
    diag: u32,      // diagonal (i, j)-(i + 1, j + 1) of the cell with this particle as corner 00
    pinA: u32,      // tether pins (NONE if the cloth has none) and their rest-layout distances
    pinB: u32,
    pad0: u32,
    restA: f32,
    restB: f32,
    pad1: u32,
    pad2: u32,
}
`;

export const computeShader = simStructs + sharedStructs + `
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(3) var<storage, read_write> broken: array<u32>;
@group(0) @binding(4) var<storage, read_write> tornCount: atomic<u32>;
@group(0) @binding(5) var<storage, read> links: array<Links>;
@group(0) @binding(6) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params: SimParams;
@group(1) @binding(0) var<uniform> colorRange: ColorRange;

const MAX_SPEED: f32 = ${MAX_SPEED.toFixed(1)};
const WIND_NORMAL_DRAG: f32 = ${WIND_NORMAL_DRAG.toFixed(3)};
const WIND_TANGENT_DRAG: f32 = ${WIND_TANGENT_DRAG.toFixed(3)};

// Gusty wind: the mean velocity scaled by slow traveling waves, plus a crosswind of the same
// size times the gustiness. The crosswind has a component along any cloth normal, which is what
// makes a flag in a steady wind flutter.
fn windAt(x: vec3<f32>) -> vec3<f32> {
    let t = params.time;
    let g = params.wind.w;
    let mean = params.wind.xyz;
    let strength = length(mean);
    let surge = 1.0 + g * 0.5 * (sin(1.7 * t + 1.3 * x.x + 0.7 * x.y) + 0.5 * sin(2.9 * t - 0.9 * x.z + 1.1 * x.y));
    let crossWind = vec3<f32>(
        sin(2.3 * t + 2.1 * x.x + 0.5 * x.z),
        0.4 * sin(1.9 * t + 1.7 * x.z + 0.8 * x.x),
        cos(1.6 * t + 2.4 * x.y + 1.3 * x.x)
    );
    return mean * surge + crossWind * (strength * g * 0.35);
}

// Particles held by the mouse: within grabRadius grid cells of the grabbed particle
fn gripped(i: u32) -> bool {
    if (params.grabIndex == NONE) {
        return false;
    }
    let n = i32(params.gridN);
    let di = abs(i32(i) % n - i32(params.grabIndex) % n);
    let dj = abs(i32(i) / n - i32(params.grabIndex) / n);
    let r = i32(params.grabRadius);
    return di <= r && dj <= r;
}

fn inverseMass(i: u32, w: f32) -> f32 {
    return select(w, 0.0, gripped(i));
}

// Verlet step with damping. Air drag uses the wind velocity relative to the particle: a pressure
// term along the (previous frame's) normal and a weaker term along the surface. With no wind this
// is plain air resistance, so a swinging cloth settles.
@compute @workgroup_size(${WORKGROUP})
fn cs_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.gridN * params.gridN) {
        return;
    }
    let p = positions[idx];
    if (gripped(idx)) {
        // Kinematic: follows the mouse; prev = pos so it is at rest when released
        let g = vec4<f32>(p.xyz + params.grabDelta.xyz, p.w);
        positions[idx] = g;
        previous[idx] = g;
        return;
    }
    if (p.w == 0.0) {
        previous[idx] = p;   // pinned
        return;
    }

    let h = params.dt;
    var vel = (p.xyz - previous[idx].xyz) * (params.damping / h);
    let speed = length(vel);
    if (speed > MAX_SPEED) {
        vel = vel * (MAX_SPEED / speed);
    }

    let n = normals[idx].xyz;   // unit length, or zero where every neighbor link is torn
    let rel = windAt(p.xyz) - vel;
    let accel = params.gravity.xyz + WIND_NORMAL_DRAG * dot(n, rel) * n + WIND_TANGENT_DRAG * rel;

    previous[idx] = vec4<f32>(p.xyz, p.w);
    positions[idx] = vec4<f32>(p.xyz + vel * h + accel * (h * h), p.w);
}

fn tether(x: vec3<f32>, pin: u32, rest: f32) -> vec3<f32> {
    if (pin == NONE) {
        return x;
    }
    let anchor = positions[pin].xyz;
    let d = x - anchor;
    let len = length(d);
    if (len <= rest) {
        return x;
    }
    return anchor + d * (rest / len);
}

// Long-range attachments: |x − pin| ≤ rest distance, for the particle's two nearest pins
@compute @workgroup_size(${WORKGROUP})
fn cs_tether(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.gridN * params.gridN) {
        return;
    }
    if (atomicLoad(&tornCount) != 0u) {
        return;
    }
    let p = positions[idx];
    if (p.w == 0.0 || gripped(idx)) {
        return;
    }
    let l = links[idx];
    var x = tether(p.xyz, l.pinA, l.restA);
    x = tether(x, l.pinB, l.restB);
    positions[idx] = vec4<f32>(x, p.w);
}

fn breakConstraint(ci: u32) {
    broken[ci] = 1u;
    atomicAdd(&tornCount, 1u);
}

// One PBD distance-constraint projection per invocation, for the constraints of one color.
// C = |xb − xa| − rest; the correction moves both ends along the link, weighted by inverse mass:
//   Δxa = +k·wa/(wa + wb)·C·d̂,  Δxb = −k·wb/(wa + wb)·C·d̂
// No two constraints of a color share a particle, so the read-modify-writes never race.
@compute @workgroup_size(${WORKGROUP})
fn cs_solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= colorRange.count) {
        return;
    }
    let ci = colorRange.start + gid.x;
    if (broken[ci] != 0u) {
        return;
    }
    let c = constraints[ci];
    // Parents are always in other colors, so their flags are stable during this dispatch
    var parentTorn = false;
    if (c.parentA != NONE) {
        parentTorn = broken[c.parentA] != 0u;
    }
    if (c.parentB != NONE) {
        parentTorn = parentTorn || broken[c.parentB] != 0u;
    }
    if (parentTorn) {
        breakConstraint(ci);
        return;
    }

    let pa = positions[c.a];
    let pb = positions[c.b];
    let d = pb.xyz - pa.xyz;
    let len = length(d);
    if (len < 1e-7) {
        return;
    }
    if (params.tearOn != 0u && c.kind < KIND_BEND && len > c.rest * params.tearRatio) {
        breakConstraint(ci);
        return;
    }
    let wa = inverseMass(c.a, pa.w);
    let wb = inverseMass(c.b, pb.w);
    let wsum = wa + wb;
    if (wsum <= 0.0) {
        return;
    }
    let corr = d * (params.stiffness[c.kind] * (len - c.rest) / (len * wsum));
    positions[c.a] = vec4<f32>(pa.xyz + corr * wa, pa.w);
    positions[c.b] = vec4<f32>(pb.xyz - corr * wb, pb.w);
}

// Contact response. x is the particle already projected onto the collider surface, v its
// displacement this substep (from the previous position, before projection). The approaching
// normal part of v is removed (no bounce). Coulomb friction acts on the sliding part, with the
// penetration depth standing in for the normal impulse (Macklin et al. 2014): a slide shorter
// than μ·depth is undone (static friction), a longer one is shortened by μ·depth (kinetic). The
// undone slide is taken back out of the position too, not just the velocity; otherwise the
// constraint solver's tangential corrections would let resting cloth creep.
struct Contact {
    x: vec3<f32>,
    v: vec3<f32>,
}

fn contact(x: vec3<f32>, v: vec3<f32>, n: vec3<f32>, depth: f32) -> Contact {
    let vn = dot(v, n);
    let vt = v - vn * n;
    let slide = length(vt);
    let limit = params.friction * depth;
    var kept = vec3<f32>(0.0);
    if (slide > limit) {
        kept = vt * (1.0 - limit / slide);
    }
    return Contact(x - (vt - kept), kept + max(vn, 0.0) * n);
}

@compute @workgroup_size(${WORKGROUP})
fn cs_collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.gridN * params.gridN) {
        return;
    }
    let p = positions[idx];
    if (p.w == 0.0 || gripped(idx)) {
        return;
    }
    var x = p.xyz;
    var v = x - previous[idx].xyz;   // displacement this substep
    var hit = false;
    if (params.sphereOn != 0u) {
        let d = x - params.sphere.xyz;
        let r = params.sphere.w + params.thickness;
        let dist = length(d);
        if (dist < r && dist > 1e-6) {
            let n = d / dist;
            let c = contact(params.sphere.xyz + n * r, v, n, r - dist);
            x = c.x;
            v = c.v;
            hit = true;
        }
    }
    let floorTop = params.floorY + params.thickness;
    if (x.y < floorTop) {
        let depth = floorTop - x.y;
        let c = contact(vec3<f32>(x.x, floorTop, x.z), v, vec3<f32>(0.0, 1.0, 0.0), depth);
        x = c.x;
        v = c.v;
        hit = true;
    }
    if (hit) {
        positions[idx] = vec4<f32>(x, p.w);
        previous[idx] = vec4<f32>(x - v, p.w);
    }
}

// Vertex normal from central differences along the grid, n = (x[i+1] − x[i−1]) × (x[j+1] − x[j−1]),
// using the particle itself in place of a neighbor across a torn (or missing) structural link
@compute @workgroup_size(${WORKGROUP})
fn cs_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.gridN;
    if (idx >= n * n) {
        return;
    }
    let l = links[idx];
    let p = positions[idx].xyz;
    var xp = p;
    var xn = p;
    var yp = p;
    var yn = p;
    if (l.xPos != NONE && broken[l.xPos] == 0u) { xp = positions[idx + 1u].xyz; }
    if (l.xNeg != NONE && broken[l.xNeg] == 0u) { xn = positions[idx - 1u].xyz; }
    if (l.yPos != NONE && broken[l.yPos] == 0u) { yp = positions[idx + n].xyz; }
    if (l.yNeg != NONE && broken[l.yNeg] == 0u) { yn = positions[idx - n].xyz; }
    let c = cross(xp - xn, yp - yn);
    let len = length(c);
    var out = vec4<f32>(0.0);
    if (len > 1e-12) {
        out = vec4<f32>(c / len, 0.0);
    }
    normals[idx] = out;
}
`;

export const cameraStruct = `
struct Camera {
    viewProj: mat4x4<f32>,
    eye: vec4<f32>,
    sphere: vec4<f32>,
    lightDir: vec4<f32>,
    viewport: vec2<f32>,
    floorY: f32,
    pointSize: f32,
    gridN: u32,
    sphereOn: u32,
    grabIndex: u32,
    pad0: u32,
}
`;

export const renderShader = cameraStruct + sharedStructs + `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> links: array<Links>;
@group(0) @binding(4) var<storage, read> broken: array<u32>;
@group(0) @binding(5) var<storage, read> constraints: array<Constraint>;

const PI: f32 = 3.14159265;
const SPHERE_SEG_U: u32 = 48u;
const SPHERE_SEG_V: u32 = 24u;
const FLOOR_HALF: f32 = 8.0;
const HIDDEN = vec4<f32>(2.0, 2.0, 2.0, 1.0);   // outside the clip volume: the primitive is dropped

fn intact(ci: u32) -> bool {
    return ci == NONE || broken[ci] == 0u;
}

// Shared lighting: hemisphere ambient + wrapped diffuse + soft specular, n facing the viewer
fn shade(base: vec3<f32>, n: vec3<f32>, worldPos: vec3<f32>, shininess: f32, specAmount: f32) -> vec3<f32> {
    let l = cam.lightDir.xyz;
    let v = normalize(cam.eye.xyz - worldPos);
    let hemi = mix(vec3<f32>(0.20, 0.18, 0.17), vec3<f32>(0.34, 0.38, 0.46), n.y * 0.5 + 0.5);
    let diffuse = max((dot(n, l) + 0.25) / 1.25, 0.0);
    let hv = normalize(l + v);
    let spec = pow(max(dot(n, hv), 0.0), shininess) * specAmount;
    return base * (hemi + vec3<f32>(1.0, 0.97, 0.92) * diffuse * 0.85) + vec3<f32>(spec);
}

// ---- Cloth: two triangles per grid cell, drawn only if their three edges are intact ----
struct ClothOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_cloth(@builtin(vertex_index) vi: u32) -> ClothOut {
    let n = cam.gridN;
    let cells = n - 1u;
    let cell = vi / 6u;
    let corner = vi % 6u;
    let ci = cell % cells;
    let cj = cell / cells;
    let p00 = cj * n + ci;
    let p10 = p00 + 1u;
    let p01 = p00 + n;
    let p11 = p01 + 1u;
    // Triangle 0: p00 p10 p11 (edges H(i, j), V(i + 1, j), diagonal)
    // Triangle 1: p00 p11 p01 (edges diagonal, H(i, j + 1), V(i, j))
    var corners = array<u32, 6>(p00, p10, p11, p00, p11, p01);
    let particle = corners[corner];
    var visible = intact(links[p00].diag);
    if (corner < 3u) {
        visible = visible && intact(links[p00].xPos) && intact(links[p10].yPos);
    } else {
        visible = visible && intact(links[p01].xPos) && intact(links[p00].yPos);
    }

    var out: ClothOut;
    let world = positions[particle].xyz;
    out.worldPos = world;
    out.normal = normals[particle].xyz;
    out.uv = vec2<f32>(f32(particle % n), f32(particle / n)) / f32(cells);
    out.position = select(HIDDEN, cam.viewProj * vec4<f32>(world, 1.0), visible);
    return out;
}

// Gingham on the front (two crossing stripe sets, darker where they overlap), a plain woven
// blue on the back; a fine thread pattern on both
@fragment
fn fs_cloth(in: ClothOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
    let weave = 0.94 + 0.06 * sin(in.uv.x * 900.0) * sin(in.uv.y * 900.0);
    var base: vec3<f32>;
    if (frontFacing) {
        let s = step(vec2<f32>(0.5), fract(in.uv * 12.0));
        let stripes = s.x + s.y;
        let cream = vec3<f32>(0.93, 0.89, 0.80);
        let red = vec3<f32>(0.72, 0.10, 0.12);
        base = mix(cream, red, stripes * 0.5);
    } else {
        base = vec3<f32>(0.16, 0.30, 0.55);
    }
    base = base * weave;

    var nrm = in.normal;
    let len = length(nrm);
    if (len < 1e-6) {
        nrm = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        nrm = nrm / len;
    }
    // Two-sided: light the side that faces the viewer
    if (dot(nrm, cam.eye.xyz - in.worldPos) < 0.0) {
        nrm = -nrm;
    }
    return vec4<f32>(shade(base, nrm, in.worldPos, 24.0, 0.08), 1.0);
}

// ---- Wireframe: intact structural links (they come first in the constraint buffer) ----
@vertex
fn vs_wire(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let ci = vi / 2u;
    let c = constraints[ci];
    let particle = select(c.a, c.b, (vi & 1u) == 1u);
    let world = positions[particle].xyz;
    // Pulled slightly toward the camera so the lines win the depth test against the surface
    let toEye = normalize(cam.eye.xyz - world);
    let clip = cam.viewProj * vec4<f32>(world + toEye * 0.004, 1.0);
    return select(HIDDEN, clip, broken[ci] == 0u);
}

@fragment
fn fs_wire() -> @location(0) vec4<f32> {
    return vec4<f32>(0.95, 0.95, 0.85, 1.0);
}

// ---- Particles: screen-space quads; pinned red, grabbed yellow ----
struct PointOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_point(@builtin(vertex_index) vi: u32, @builtin(instance_index) particle: u32) -> PointOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let q = quad[vi];
    let p = positions[particle];
    let toEye = normalize(cam.eye.xyz - p.xyz);
    let clip = cam.viewProj * vec4<f32>(p.xyz + toEye * 0.006, 1.0);
    var size = cam.pointSize;
    var color = vec3<f32>(0.92, 0.92, 0.95);
    if (p.w == 0.0) {
        color = vec3<f32>(1.0, 0.25, 0.2);
        size = size * 1.8;
    }
    if (particle == cam.grabIndex) {
        color = vec3<f32>(1.0, 0.9, 0.2);
        size = size * 2.5;
    }
    var out: PointOut;
    out.position = clip + vec4<f32>(q * size / cam.viewport * clip.w, 0.0, 0.0);
    out.local = q;
    out.color = color;
    return out;
}

@fragment
fn fs_point(in: PointOut) -> @location(0) vec4<f32> {
    if (dot(in.local, in.local) > 1.0) {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}

// ---- Sphere collider: a latitude/longitude mesh generated from the vertex index ----
struct SolidOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_sphere(@builtin(vertex_index) vi: u32) -> SolidOut {
    var offsets = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u),
        vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 1u)
    );
    let quadIdx = vi / 6u;
    let o = offsets[vi % 6u];
    let u = quadIdx % SPHERE_SEG_U + o.x;
    let v = quadIdx / SPHERE_SEG_U + o.y;
    let theta = f32(v) / f32(SPHERE_SEG_V) * PI;
    let phi = f32(u) / f32(SPHERE_SEG_U) * 2.0 * PI;
    let dir = vec3<f32>(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    let world = cam.sphere.xyz + dir * cam.sphere.w;
    var out: SolidOut;
    out.position = cam.viewProj * vec4<f32>(world, 1.0);
    out.worldPos = world;
    out.normal = dir;
    return out;
}

@fragment
fn fs_sphere(in: SolidOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    return vec4<f32>(shade(vec3<f32>(0.55, 0.58, 0.62), n, in.worldPos, 60.0, 0.35), 1.0);
}

// ---- Floor: a large quad with an anti-aliased 0.5 m grid and a soft blob shadow under the sphere ----
@vertex
fn vs_floor(@builtin(vertex_index) vi: u32) -> SolidOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let c = corners[vi] * FLOOR_HALF;
    let world = vec3<f32>(c.x, cam.floorY, c.y);
    var out: SolidOut;
    out.position = cam.viewProj * vec4<f32>(world, 1.0);
    out.worldPos = world;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    return out;
}

@fragment
fn fs_floor(in: SolidOut) -> @location(0) vec4<f32> {
    let coord = in.worldPos.xz * 2.0;
    let width = fwidth(coord);
    let grid = abs(fract(coord - 0.5) - 0.5) / max(width, vec2<f32>(1e-4));
    let line = 1.0 - min(min(grid.x, grid.y), 1.0);
    let dist = length(in.worldPos.xz);
    let fade = 1.0 - smoothstep(3.0, FLOOR_HALF, dist);
    var base = mix(vec3<f32>(0.11, 0.12, 0.15), vec3<f32>(0.30, 0.33, 0.40), line * fade);
    if (cam.sphereOn != 0u) {
        let r = cam.sphere.w;
        let ds = length(in.worldPos.xz - cam.sphere.xz) / max(r, 0.01);
        let height = max(cam.sphere.y - cam.floorY, r);
        base = base * (1.0 - 0.55 * exp(-ds * ds * 1.5) * clamp(1.5 * r / height, 0.0, 1.0));
    }
    let lit = shade(base, vec3<f32>(0.0, 1.0, 0.0), in.worldPos, 16.0, 0.0);
    // Fade into the background color with distance
    return vec4<f32>(mix(vec3<f32>(0.035, 0.04, 0.06), lit, fade), 1.0);
}
`;

// ---- Presets ----
// place(u, v) gives the initial position of grid point (u, v) ∈ [0, 1]², rest(u, v) the flat layout
// the rest lengths come from (defaults to place), pinned(i, j, n) marks fixed particles.
export const PRESETS = {
    corners: {
        name: 'Hang from two corners',
        place: (u, v) => [(0.5 - u) * 2.0, 1.3, v * 2.0],
        pinned: (i, j, n) => j === 0 && (i === 0 || i === n - 1),
        wind: 0, windDir: [0, 0, 1], gust: 0.3, sphere: null,
    },
    edge: {
        name: 'Hang from top edge',
        place: (u, v) => [(0.5 - u) * 2.0, 1.3, v * 2.0],
        pinned: (i, j) => j === 0,
        wind: 0, windDir: [0, 0, 1], gust: 0.3, sphere: null,
    },
    curtain: {
        name: 'Curtain',
        // Hung from a straight rod 80% as wide as the fabric; a ripple below the rod seeds the folds
        rest: (u, v) => [(0.5 - u) * 2.4, 1.3 - v * 2.2, 0],
        place: (u, v) => [(0.5 - u) * 2.4 * 0.8, 1.3 - v * 2.2, 0.06 * Math.sin(u * Math.PI * 10) * Math.min(v * 8, 1)],
        pinned: (i, j) => j === 0,
        wind: 5, windDir: [0, 0, -1], gust: 0.6,
        sphere: { center: [0.35, -0.25, -0.75], radius: 0.4 },
    },
    drape: {
        name: 'Drop on sphere',
        place: (u, v) => [(0.5 - u) * 2.6, 0.9, (v - 0.5) * 2.6],
        pinned: () => false,
        wind: 0, windDir: [1, 0, 0], gust: 0.3,
        sphere: { center: [0.04, -0.35, 0.02], radius: 0.6 },
    },
    flag: {
        name: 'Flag',
        place: (u, v) => [-1.2 + u * 2.4, -0.1 + v * 1.5, 0],
        pinned: (i) => i === 0,
        wind: 12, windDir: [1, 0, 0], gust: 0.7, sphere: null,
    },
};

// Builds particles, color-sorted constraints, per-particle links and the color ranges.
// Color order: H even, H odd, V even, V odd, D even, D odd, A even, A odd, BH 0, BH 1, BV 0, BV 1,
// so the structural links are the first numStructural constraints (drawn by the wireframe).
export function buildCloth(n, presetKey) {
    const preset = PRESETS[presetKey];
    const count = n * n;
    const restFn = preset.rest || preset.place;
    const restPos = new Float64Array(count * 3);
    const positions = new Float32Array(count * 4);
    let pinnedCount = 0;
    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const idx = j * n + i;
            const u = i / (n - 1);
            const v = j / (n - 1);
            const r = restFn(u, v);
            const p = preset.place(u, v);
            restPos.set(r, idx * 3);
            positions.set(p, idx * 4);
            const pinned = preset.pinned(i, j, n);
            positions[idx * 4 + 3] = pinned ? 0 : 1;
            if (pinned) pinnedCount++;
        }
    }

    const id = (i, j) => j * n + i;
    const colors = Array.from({ length: NUM_COLORS }, () => []);
    // Each entry: [a, b, kind, parentKeyA, parentKeyB]; parents resolved after sorting
    const hKey = (i, j) => `H${i},${j}`;
    const vKey = (i, j) => `V${i},${j}`;
    const dKey = (i, j) => `D${i},${j}`;
    for (let j = 0; j < n; j++) {
        for (let i = 0; i + 1 < n; i++) colors[i & 1].push([id(i, j), id(i + 1, j), KIND_STRUCTURAL, null, null, hKey(i, j)]);
    }
    for (let j = 0; j + 1 < n; j++) {
        for (let i = 0; i < n; i++) colors[2 + (j & 1)].push([id(i, j), id(i, j + 1), KIND_STRUCTURAL, null, null, vKey(i, j)]);
    }
    for (let j = 0; j + 1 < n; j++) {
        for (let i = 0; i + 1 < n; i++) colors[4 + (i & 1)].push([id(i, j), id(i + 1, j + 1), KIND_SHEAR, null, null, dKey(i, j)]);
    }
    for (let j = 0; j + 1 < n; j++) {
        for (let i = 0; i + 1 < n; i++) colors[6 + (i & 1)].push([id(i + 1, j), id(i, j + 1), KIND_SHEAR, dKey(i, j), null, null]);
    }
    for (let j = 0; j < n; j++) {
        for (let i = 0; i + 2 < n; i++) colors[8 + ((i >> 1) & 1)].push([id(i, j), id(i + 2, j), KIND_BEND, hKey(i, j), hKey(i + 1, j), null]);
    }
    for (let j = 0; j + 2 < n; j++) {
        for (let i = 0; i < n; i++) colors[10 + ((j >> 1) & 1)].push([id(i, j), id(i, j + 2), KIND_BEND, vKey(i, j), vKey(i, j + 1), null]);
    }

    const ranges = [];
    const flat = [];
    for (const list of colors) {
        ranges.push({ start: flat.length, count: list.length });
        for (const c of list) flat.push(c);
    }
    const keyIndex = new Map();
    flat.forEach((c, k) => { if (c[5]) keyIndex.set(c[5], k); });

    const numConstraints = flat.length;
    const constraintData = new ArrayBuffer(numConstraints * CONSTRAINT_WORDS * 4);
    const cu = new Uint32Array(constraintData);
    const cf = new Float32Array(constraintData);
    const kindCounts = [0, 0, 0];
    flat.forEach(([a, b, kind, pa, pb], k) => {
        const o = k * CONSTRAINT_WORDS;
        const dx = restPos[b * 3] - restPos[a * 3];
        const dy = restPos[b * 3 + 1] - restPos[a * 3 + 1];
        const dz = restPos[b * 3 + 2] - restPos[a * 3 + 2];
        cu[o + 0] = a;
        cu[o + 1] = b;
        cf[o + 2] = Math.sqrt(dx * dx + dy * dy + dz * dz);
        cu[o + 3] = kind;
        cu[o + 4] = pa ? keyIndex.get(pa) : NONE;
        cu[o + 5] = pb ? keyIndex.get(pb) : NONE;
        kindCounts[kind]++;
    });

    const linkData = new ArrayBuffer(count * LINK_WORDS * 4);
    const links = new Uint32Array(linkData).fill(NONE);
    const linkF32 = new Float32Array(linkData);
    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const o = id(i, j) * LINK_WORDS;
            if (i > 0) links[o + 0] = keyIndex.get(hKey(i - 1, j));
            if (i + 1 < n) links[o + 1] = keyIndex.get(hKey(i, j));
            if (j > 0) links[o + 2] = keyIndex.get(vKey(i, j - 1));
            if (j + 1 < n) links[o + 3] = keyIndex.get(vKey(i, j));
            if (i + 1 < n && j + 1 < n) links[o + 4] = keyIndex.get(dKey(i, j));
        }
    }
    // Tethers: the two nearest pins in the rest layout (ties broken by index)
    const pins = [];
    for (let k = 0; k < count; k++) if (positions[k * 4 + 3] === 0) pins.push(k);
    for (let k = 0; k < count; k++) {
        const o = k * LINK_WORDS;
        links[o + 7] = 0;
        links[o + 10] = 0;
        links[o + 11] = 0;
        linkF32[o + 8] = 0;
        linkF32[o + 9] = 0;
        if (positions[k * 4 + 3] === 0) continue;
        let best = [NONE, Infinity], second = [NONE, Infinity];
        for (const pin of pins) {
            const d = Math.hypot(restPos[k * 3] - restPos[pin * 3], restPos[k * 3 + 1] - restPos[pin * 3 + 1],
                restPos[k * 3 + 2] - restPos[pin * 3 + 2]);
            if (d < best[1]) { second = best; best = [pin, d]; } else if (d < second[1]) second = [pin, d];
        }
        links[o + 5] = best[0];
        links[o + 6] = second[0];
        linkF32[o + 8] = Number.isFinite(best[1]) ? best[1] : 0;
        linkF32[o + 9] = Number.isFinite(second[1]) ? second[1] : 0;
    }

    return {
        n, count, positions, constraintData, numConstraints, ranges, links, linkData, pinnedCount,
        numStructural: kindCounts[KIND_STRUCTURAL], kindCounts,
    };
}

// Per-iteration stiffness so the total effect after `iterations` passes is k: 1 − (1 − k)^(1/iterations)
export function iterationStiffness(k, iterations) {
    return 1 - Math.pow(1 - Math.min(Math.max(k, 0), 1), 1 / iterations);
}

// ---- Matrix math (column-major, element [col * 4 + row], WebGPU clip depth [0, 1]) ----
class Mat4 {
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

    // a * b for column-major matrices
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
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export const FOV = Math.PI / 4;

export class OrbitCamera {
    constructor() {
        this.distance = 5.2;
        this.rotation = { x: 0.28, y: 1.2 };   // elevation, azimuth (π/2 looks along −z)
        this.target = [0, 0.1, 0];
    }

    getEye() {
        const c = Math.cos(this.rotation.x);
        return [
            this.target[0] + this.distance * Math.cos(this.rotation.y) * c,
            this.target[1] + this.distance * Math.sin(this.rotation.x),
            this.target[2] + this.distance * Math.sin(this.rotation.y) * c
        ];
    }

    // Orthonormal camera basis, matching Mat4.lookAt
    getBasis() {
        const eye = this.getEye();
        const forward = normalize(subtract(this.target, eye));
        const right = normalize(cross(forward, [0, 1, 0]));
        const up = cross(right, forward);
        return { eye, forward, right, up };
    }

    getViewProj(aspect) {
        const proj = Mat4.perspective(FOV, aspect, 0.05, 60);
        const view = Mat4.lookAt(this.getEye(), this.target, [0, 1, 0]);
        return Mat4.multiply(proj, view);
    }

    // World-space ray through canvas pixel (px, py)
    ray(px, py, width, height) {
        const { eye, forward, right, up } = this.getBasis();
        const t = Math.tan(FOV / 2);
        const sx = ((px / width) * 2 - 1) * t * (width / height);
        const sy = (1 - (py / height) * 2) * t;
        const dir = normalize([
            forward[0] + right[0] * sx + up[0] * sy,
            forward[1] + right[1] * sx + up[1] * sy,
            forward[2] + right[2] * sx + up[2] * sy
        ]);
        return { origin: eye, dir, forward };
    }
}

// Picks the particle nearest to the camera among those within `radius(t)` of the ray
export function pickParticle(positions, count, ray, pixelAngle) {
    let best = -1;
    let bestT = Infinity;
    let fallback = -1;
    let fallbackScore = Infinity;
    const [ox, oy, oz] = ray.origin;
    const [dx, dy, dz] = ray.dir;
    for (let i = 0; i < count; i++) {
        const px = positions[i * 4] - ox;
        const py = positions[i * 4 + 1] - oy;
        const pz = positions[i * 4 + 2] - oz;
        const t = px * dx + py * dy + pz * dz;
        if (t <= 0) continue;
        const qx = px - dx * t;
        const qy = py - dy * t;
        const qz = pz - dz * t;
        const perp = Math.sqrt(qx * qx + qy * qy + qz * qz);
        const angle = perp / t;
        if (angle < pixelAngle * 10 && t < bestT) {
            bestT = t;
            best = i;
        }
        if (angle < fallbackScore) {
            fallbackScore = angle;
            fallback = i;
        }
    }
    if (best >= 0) return best;
    return fallbackScore < pixelAngle * 30 ? fallback : -1;
}

export class ClothSimulation {
    // canvas = null: compute only ("Compute-only use" at the top of the file)
    constructor(device, canvas, { gridN = 64, presetKey = 'corners' } = {}) {
        this.device = device;
        this.gpuTimer = new GpuTimer(device, { maxPasses: 2 });
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;

        this.gridN = gridN;
        this.presetKey = presetKey;
        this.iterations = 8;
        this.stiffness = 1.0;
        this.windStrength = 0;
        this.gravity = 9.8;
        this.tearOn = false;
        this.tearRatio = 1.8;
        this.showWireframe = false;
        this.showParticles = false;

        this.time = 0;
        this.accumulator = 0;
        this.lastFrameTime = 0;
        this.substepsTotal = 0;
        this.lastSubsteps = 0;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;
        this.tornCount = 0;
        this.generation = 0;

        this.grabIndex = NONE;
        this.grabTarget = [0, 0, 0];   // where the mouse wants the grip center
        this.grabPos = [0, 0, 0];      // where the GPU has moved it so far
        this.grabDepth = 0;
        this.pickPending = false;
        this.tornReadPending = false;
        this.framesSinceTornRead = 0;

        this.camera = new OrbitCamera();

        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.colorStride = Math.ceil(COLOR_RANGE_SIZE / align) * align;

        this.simParams = new ArrayBuffer(SIM_PARAMS_SIZE);
        this.simF32 = new Float32Array(this.simParams);
        this.simU32 = new Uint32Array(this.simParams);
        this.cameraData = new ArrayBuffer(CAMERA_SIZE);
        this.camF32 = new Float32Array(this.cameraData);
        this.camU32 = new Uint32Array(this.cameraData);

        this.simParamsBuffer = device.createBuffer({ size: SIM_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.colorBuffer = device.createBuffer({ size: this.colorStride * NUM_COLORS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.cameraBuffer = device.createBuffer({ size: CAMERA_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.counterBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.counterReadback = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        this.setupPipelines();
        if (canvas) {
            this.setupTargets();
            this.setupMouseControls();
        }
        this.applyPresetSettings();
        this.buildGrid();
    }

    setupPipelines() {
        const device = this.device;
        const computeModule = device.createShaderModule({ code: computeShader });
        const C = GPUShaderStage.COMPUTE;
        this.computeLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'storage' } },
                { binding: 1, visibility: C, buffer: { type: 'storage' } },
                { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: C, buffer: { type: 'storage' } },
                { binding: 4, visibility: C, buffer: { type: 'storage' } },
                { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 6, visibility: C, buffer: { type: 'storage' } },
                { binding: 7, visibility: C, buffer: { type: 'uniform' } }
            ]
        });
        this.colorLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: COLOR_RANGE_SIZE } }
            ]
        });
        const particleLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
        const solveLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout, this.colorLayout] });
        const make = (layout, entryPoint) => device.createComputePipeline({ layout, compute: { module: computeModule, entryPoint } });
        this.integratePipeline = make(particleLayout, 'cs_integrate');
        this.tetherPipeline = make(particleLayout, 'cs_tether');
        this.collidePipeline = make(particleLayout, 'cs_collide');
        this.normalsPipeline = make(particleLayout, 'cs_normals');
        this.solvePipeline = make(solveLayout, 'cs_solve');
        this.colorBindGroup = device.createBindGroup({
            layout: this.colorLayout,
            entries: [{ binding: 0, resource: { buffer: this.colorBuffer, offset: 0, size: COLOR_RANGE_SIZE } }]
        });

        const renderModule = device.createShaderModule({ code: renderShader });
        const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
        const V = GPUShaderStage.VERTEX;
        this.renderLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
                { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: V, buffer: { type: 'read-only-storage' } }
            ]
        });
        const layout = device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] });
        const pipeline = (vs, fs, topology, cullMode, depthCompare = 'less') => device.createRenderPipeline({
            layout,
            vertex: { module: renderModule, entryPoint: vs },
            fragment: { module: renderModule, entryPoint: fs, targets: [{ format: 'bgra8unorm' }] },
            primitive: { topology, cullMode },
            depthStencil: { depthWriteEnabled: true, depthCompare, format: 'depth24plus' },
            multisample: { count: MSAA }
        });
        this.clothPipeline = pipeline('vs_cloth', 'fs_cloth', 'triangle-list', 'none');
        this.wirePipeline = pipeline('vs_wire', 'fs_wire', 'line-list', 'none', 'less-equal');
        this.pointPipeline = pipeline('vs_point', 'fs_point', 'triangle-list', 'none', 'less-equal');
        this.spherePipeline = pipeline('vs_sphere', 'fs_sphere', 'triangle-list', 'none');
        this.floorPipeline = pipeline('vs_floor', 'fs_floor', 'triangle-list', 'none');
    }

    setupTargets() {
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({ device: this.device, format: 'bgra8unorm', alphaMode: 'opaque' });
        const size = [this.canvas.width, this.canvas.height];
        this.msaaTexture = this.device.createTexture({
            size, format: 'bgra8unorm', sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.depthTexture = this.device.createTexture({
            size, format: 'depth24plus', sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }

    // (Re)creates every grid-sized buffer for this.gridN and loads the current preset
    buildGrid() {
        this.generation++;
        this.destroyGridBuffers();
        const cloth = buildCloth(this.gridN, this.presetKey);
        this.cloth = cloth;
        const device = this.device;
        const storage = (size, extra = 0) => device.createBuffer({
            size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra
        });
        this.positionBuffer = storage(cloth.count * 16, GPUBufferUsage.COPY_SRC);
        this.previousBuffer = storage(cloth.count * 16);
        this.normalBuffer = storage(cloth.count * 16);
        this.constraintBuffer = storage(cloth.numConstraints * CONSTRAINT_WORDS * 4);
        this.brokenBuffer = storage(cloth.numConstraints * 4);
        this.linkBuffer = storage(cloth.count * LINK_WORDS * 4);
        this.pickReadback = device.createBuffer({ size: cloth.count * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        device.queue.writeBuffer(this.constraintBuffer, 0, cloth.constraintData);
        device.queue.writeBuffer(this.linkBuffer, 0, cloth.linkData);

        // Color ranges: one 16-byte ColorRange per dynamic-offset slot (layout at the top of the file)
        const colorData = new ArrayBuffer(this.colorStride * NUM_COLORS);
        const colorU32 = new Uint32Array(colorData);
        cloth.ranges.forEach((r, c) => {
            const o = c * this.colorStride / 4;
            colorU32[o + 0] = r.start;
            colorU32[o + 1] = r.count;
            colorU32[o + 2] = 0;
            colorU32[o + 3] = 0;
        });
        device.queue.writeBuffer(this.colorBuffer, 0, colorData);

        this.computeBindGroup = device.createBindGroup({
            layout: this.computeLayout,
            entries: [
                { binding: 0, resource: { buffer: this.positionBuffer } },
                { binding: 1, resource: { buffer: this.previousBuffer } },
                { binding: 2, resource: { buffer: this.constraintBuffer } },
                { binding: 3, resource: { buffer: this.brokenBuffer } },
                { binding: 4, resource: { buffer: this.counterBuffer } },
                { binding: 5, resource: { buffer: this.linkBuffer } },
                { binding: 6, resource: { buffer: this.normalBuffer } },
                { binding: 7, resource: { buffer: this.simParamsBuffer } }
            ]
        });
        this.renderBindGroup = device.createBindGroup({
            layout: this.renderLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: { buffer: this.positionBuffer } },
                { binding: 2, resource: { buffer: this.normalBuffer } },
                { binding: 3, resource: { buffer: this.linkBuffer } },
                { binding: 4, resource: { buffer: this.brokenBuffer } },
                { binding: 5, resource: { buffer: this.constraintBuffer } }
            ]
        });
        this.reset();
    }

    destroyGridBuffers() {
        for (const name of ['positionBuffer', 'previousBuffer', 'normalBuffer', 'constraintBuffer',
            'brokenBuffer', 'linkBuffer', 'pickReadback']) {
            if (this[name]) {
                // A pick readback may still be mapping; destroying it rejects that mapAsync
                this[name].destroy();
                this[name] = null;
            }
        }
        this.pickPending = false;
    }

    // Puts the cloth back in the preset's initial state (same grid); renders one frame
    reset() {
        this.generation++;
        const cloth = buildCloth(this.gridN, this.presetKey);
        this.cloth.positions = cloth.positions;
        const q = this.device.queue;
        q.writeBuffer(this.positionBuffer, 0, cloth.positions);
        q.writeBuffer(this.previousBuffer, 0, cloth.positions);
        q.writeBuffer(this.normalBuffer, 0, new Float32Array(cloth.count * 4));
        q.writeBuffer(this.brokenBuffer, 0, new Uint32Array(cloth.numConstraints));
        q.writeBuffer(this.counterBuffer, 0, new Uint32Array(1));
        this.tornCount = 0;
        this.time = 0;
        this.accumulator = 0;
        this.substepsTotal = 0;
        this.grabIndex = NONE;
        this.render();
    }

    applyPresetSettings() {
        this.windStrength = PRESETS[this.presetKey].wind;
    }

    loadPreset(key) {
        if (!PRESETS[key]) return;
        this.presetKey = key;
        this.applyPresetSettings();
        this.buildGrid();   // the constraints' rest lengths depend on the preset
    }

    setGridSize(n) {
        this.gridN = n;
        this.buildGrid();
    }

    // ---- Mouse: drag rotates, right-drag pans, wheel zooms, shift-drag grabs a particle ----
    setupMouseControls() {
        let mode = null;   // 'rotate' | 'pan' | 'grab'
        let lastX = 0;
        let lastY = 0;
        const toCanvas = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            return [
                (e.clientX - rect.left) * this.canvas.width / rect.width,
                (e.clientY - rect.top) * this.canvas.height / rect.height
            ];
        };
        this.mouse = [0, 0];

        this.onMouseDown = (e) => {
            e.preventDefault();
            lastX = e.clientX;
            lastY = e.clientY;
            this.mouse = toCanvas(e);
            if (e.button === 0 && e.shiftKey) {
                mode = 'grab';
                this.beginGrab(this.mouse);
            } else if (e.button === 2) {
                mode = 'pan';
            } else if (e.button === 0) {
                mode = 'rotate';
            }
        };
        this.onMouseMove = (e) => {
            if (!mode) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            if (mode === 'grab') {
                this.mouse = toCanvas(e);
                this.updateGrabTarget();
            } else if (mode === 'pan') {
                const { right, up } = this.camera.getBasis();
                const s = this.camera.distance * 0.0015;
                for (let k = 0; k < 3; k++) this.camera.target[k] += (-right[k] * dx + up[k] * dy) * s;
            } else {
                this.camera.rotation.y += dx * 0.01;
                this.camera.rotation.x = Math.max(-0.2, Math.min(1.45, this.camera.rotation.x + dy * 0.01));
            }
            if (!this.running) this.requestRender();
        };
        this.onMouseUp = () => {
            if (mode === 'grab') this.endGrab();
            mode = null;
        };
        this.onWheel = (e) => {
            e.preventDefault();
            this.camera.distance = Math.max(1.5, Math.min(15, this.camera.distance * (1 + e.deltaY * 0.001)));
            if (!this.running) this.requestRender();
        };
        this.onContextMenu = (e) => e.preventDefault();

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
    }

    // One readback of the particle positions per grab; the pick itself runs on the CPU
    beginGrab(mousePos) {
        if (this.pickPending) return;
        this.pickPending = true;
        this.grabbing = true;
        const generation = this.generation;
        const readback = this.pickReadback;
        const ray = this.camera.ray(mousePos[0], mousePos[1], this.canvas.width, this.canvas.height);
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.positionBuffer, 0, readback, 0, this.cloth.count * 16);
        this.device.queue.submit([encoder.finish()]);
        readback.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Float32Array(readback.getMappedRange().slice(0));
            readback.unmap();
            this.pickPending = false;
            if (this.destroyed || generation !== this.generation || !this.grabbing) return;
            const pixelAngle = 2 * Math.tan(FOV / 2) / this.canvas.height;
            const idx = pickParticle(data, this.cloth.count, ray, pixelAngle);
            if (idx < 0) return;
            const p = [data[idx * 4], data[idx * 4 + 1], data[idx * 4 + 2]];
            const f = ray.forward;
            this.grabDepth = (p[0] - ray.origin[0]) * f[0] + (p[1] - ray.origin[1]) * f[1] + (p[2] - ray.origin[2]) * f[2];
            this.grabTarget = p;
            this.grabPos = p;
            this.grabIndex = idx;
            this.updateGrabTarget();
            if (!this.running) this.requestRender();
        }).catch(() => {
            // The buffer was destroyed by a grid change (which already cleared pickPending)
            if (readback === this.pickReadback) this.pickPending = false;
        });
    }

    // Grab target: the mouse ray's intersection with the camera-parallel plane through the grab point
    updateGrabTarget() {
        if (this.grabIndex === NONE) return;
        const ray = this.camera.ray(this.mouse[0], this.mouse[1], this.canvas.width, this.canvas.height);
        const f = ray.forward;
        const denom = ray.dir[0] * f[0] + ray.dir[1] * f[1] + ray.dir[2] * f[2];
        const t = this.grabDepth / Math.max(denom, 1e-3);
        this.grabTarget = [0, 1, 2].map(k => ray.origin[k] + ray.dir[k] * t);
        // The grip skips collisions, so keep it above the floor
        this.grabTarget[1] = Math.max(this.grabTarget[1], FLOOR_Y + 2 * THICKNESS);
    }

    endGrab() {
        this.grabbing = false;
        this.grabIndex = NONE;
        if (!this.running) this.requestRender();
    }

    requestRender() {
        if (this.renderPending) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            if (!this.destroyed && !this.running) this.render();
        });
    }

    // Grip half-size in grid cells: about 2.5% of the cloth width
    grabRadius() {
        return Math.max(1, Math.round(this.gridN / 40));
    }

    // Writes SimParams (layout table at the top of the file). The grip moves by the remaining
    // distance to the mouse target, spread evenly over this frame's substeps.
    writeSimParams(substeps) {
        const f = this.simF32;
        const u = this.simU32;
        const preset = PRESETS[this.presetKey];
        const wd = normalize(preset.windDir);
        f[0] = 0; f[1] = -this.gravity; f[2] = 0; f[3] = 0;
        f[4] = wd[0] * this.windStrength; f[5] = wd[1] * this.windStrength; f[6] = wd[2] * this.windStrength;
        f[7] = preset.gust;
        const s = preset.sphere;
        f[8] = s ? s.center[0] : 0; f[9] = s ? s.center[1] : 0; f[10] = s ? s.center[2] : 0; f[11] = s ? s.radius : 0;
        const delta = [0, 0, 0];
        if (this.grabIndex !== NONE && substeps > 0) {
            for (let k = 0; k < 3; k++) delta[k] = (this.grabTarget[k] - this.grabPos[k]) / substeps;
            this.grabPos = this.grabTarget.slice();
        }
        f[12] = delta[0]; f[13] = delta[1]; f[14] = delta[2]; f[15] = 0;
        f[16] = iterationStiffness(this.stiffness, this.iterations);
        f[17] = iterationStiffness(this.stiffness * SHEAR_FACTOR, this.iterations);
        f[18] = iterationStiffness(this.stiffness * BEND_FACTOR, this.iterations);
        f[19] = 0;
        f[20] = SUBSTEP_DT;
        f[21] = Math.exp(-DAMPING_RATE * SUBSTEP_DT);
        f[22] = FLOOR_Y;
        f[23] = FRICTION;
        f[24] = this.time;
        f[25] = this.tearRatio;
        u[26] = this.tearOn ? 1 : 0;
        u[27] = this.grabIndex >>> 0;
        u[28] = this.gridN;
        u[29] = s ? 1 : 0;
        f[30] = THICKNESS;
        u[31] = this.grabRadius();
        this.device.queue.writeBuffer(this.simParamsBuffer, 0, this.simParams);
    }

    // Writes Camera (layout table at the top of the file)
    writeCamera() {
        const f = this.camF32;
        const u = this.camU32;
        f.set(this.camera.getViewProj(this.canvas.width / this.canvas.height), 0);
        const eye = this.camera.getEye();
        f[16] = eye[0]; f[17] = eye[1]; f[18] = eye[2]; f[19] = 1;
        const s = PRESETS[this.presetKey].sphere;
        f[20] = s ? s.center[0] : 0; f[21] = s ? s.center[1] : 0; f[22] = s ? s.center[2] : 0; f[23] = s ? s.radius : 0;
        const l = normalize([0.45, 0.85, 0.55]);
        f[24] = l[0]; f[25] = l[1]; f[26] = l[2]; f[27] = 0;
        f[28] = this.canvas.width; f[29] = this.canvas.height;
        f[30] = FLOOR_Y;
        f[31] = this.gridN <= 48 ? 3.0 : 2.0;
        u[32] = this.gridN;
        u[33] = s ? 1 : 0;
        u[34] = this.grabIndex >>> 0;
        u[35] = 0;
        this.device.queue.writeBuffer(this.cameraBuffer, 0, this.cameraData);
    }

    // Encodes `substeps` substeps (integrate, tethers, iterations × colors, collide), then normals
    encodeCompute(encoder, substeps, descriptor = this.gpuTimer.computePassDescriptor('sim')) {
        const pass = encoder.beginComputePass(descriptor);
        const particleGroups = Math.ceil(this.cloth.count / WORKGROUP);
        pass.setBindGroup(0, this.computeBindGroup);
        for (let s = 0; s < substeps; s++) {
            pass.setPipeline(this.integratePipeline);
            pass.dispatchWorkgroups(particleGroups);
            if (this.cloth.pinnedCount > 0) {
                pass.setPipeline(this.tetherPipeline);
                pass.dispatchWorkgroups(particleGroups);
            }
            pass.setPipeline(this.solvePipeline);
            for (let it = 0; it < this.iterations; it++) {
                for (let c = 0; c < NUM_COLORS; c++) {
                    const range = this.cloth.ranges[c];
                    if (range.count === 0) continue;
                    pass.setBindGroup(1, this.colorBindGroup, [c * this.colorStride]);
                    pass.dispatchWorkgroups(Math.ceil(range.count / WORKGROUP));
                }
            }
            pass.setPipeline(this.collidePipeline);
            pass.dispatchWorkgroups(particleGroups);
        }
        pass.setPipeline(this.normalsPipeline);
        pass.dispatchWorkgroups(particleGroups);
        pass.end();
    }

    encodeRender(encoder) {
        const pass = encoder.beginRenderPass(this.gpuTimer.renderPassDescriptor({
            colorAttachments: [{
                view: this.msaaTexture.createView(),
                resolveTarget: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'discard',
                clearValue: { r: 0.035, g: 0.04, b: 0.06, a: 1.0 }
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'discard',
                depthClearValue: 1.0
            }
        }, 'render'));
        pass.setBindGroup(0, this.renderBindGroup);
        pass.setPipeline(this.floorPipeline);
        pass.draw(6);
        if (PRESETS[this.presetKey].sphere) {
            pass.setPipeline(this.spherePipeline);
            pass.draw(48 * 24 * 6);
        }
        pass.setPipeline(this.clothPipeline);
        pass.draw((this.gridN - 1) * (this.gridN - 1) * 6);
        if (this.showWireframe) {
            pass.setPipeline(this.wirePipeline);
            pass.draw(this.cloth.numStructural * 2);
        }
        if (this.showParticles || this.grabIndex !== NONE) {
            pass.setPipeline(this.pointPipeline);
            if (this.showParticles) {
                pass.draw(6, this.cloth.count);
            } else {
                pass.draw(6, 1, 0, this.grabIndex);
            }
        }
        pass.end();
    }

    frame(substeps) {
        this.writeSimParams(substeps);
        this.writeCamera();
        const encoder = this.device.createCommandEncoder();
        this.encodeCompute(encoder, substeps);
        this.encodeRender(encoder);
        const readTorn = this.tearOn && substeps > 0 && !this.tornReadPending && ++this.framesSinceTornRead >= 10;
        if (readTorn) {
            encoder.copyBufferToBuffer(this.counterBuffer, 0, this.counterReadback, 0, 4);
        }
        this.gpuTimer.resolve(encoder);
        this.device.queue.submit([encoder.finish()]);
        this.gpuTimer.read();
        if (readTorn) this.readTornCount();
        this.time += substeps * SUBSTEP_DT;
        this.substepsTotal += substeps;
        this.lastSubsteps = substeps;
    }

    readTornCount() {
        this.framesSinceTornRead = 0;
        this.tornReadPending = true;
        const generation = this.generation;
        this.counterReadback.mapAsync(GPUMapMode.READ).then(() => {
            const value = new Uint32Array(this.counterReadback.getMappedRange())[0];
            this.counterReadback.unmap();
            this.tornReadPending = false;
            if (generation === this.generation) this.tornCount = value;
        }).catch(() => { this.tornReadPending = false; });
    }

    // Draws the current state (normals are recomputed so a fresh preset is lit correctly)
    render() {
        if (!this.canvas) {
            this.stepCompute(0);   // compute only: just the normals, as frame(0) computes them
            return;
        }
        this.frame(0);
    }

    // ---- Compute-only use (canvas = null) ----
    // `substeps` substeps and the normals in one command buffer, like frame() without the render pass.
    // timer: a GpuTimer for this one pass (or null). Returns the submitted work's promise.
    stepCompute(substeps, timer = null) {
        this.writeSimParams(substeps);
        const encoder = this.device.createCommandEncoder();
        this.encodeCompute(encoder, substeps, timer ? timer.computePassDescriptor('sim') : {});
        timer?.resolve(encoder);
        this.device.queue.submit([encoder.finish()]);
        this.time += substeps * SUBSTEP_DT;
        this.substepsTotal += substeps;
        return timer ? timer.read() : null;
    }

    async readPositions() {
        const size = this.cloth.count * 16;
        const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.positionBuffer, 0, staging, 0, size);
        this.device.queue.submit([encoder.finish()]);
        try {
            await staging.mapAsync(GPUMapMode.READ);
            return new Float32Array(staging.getMappedRange().slice(0));
        } finally {
            staging.destroy();
        }
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.lastFrameTime = performance.now();
        this.framesSinceSample = 0;
        this.lastSampleTime = this.lastFrameTime;
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
    }

    animate() {
        if (!this.running || this.destroyed) return;
        const now = performance.now();
        // Fixed substeps of 1/240 s; at most MAX_SUBSTEPS per frame (longer frames drop time)
        this.accumulator += Math.min((now - this.lastFrameTime) / 1000, MAX_SUBSTEPS * SUBSTEP_DT);
        this.lastFrameTime = now;
        const substeps = Math.min(Math.floor(this.accumulator / SUBSTEP_DT + 1e-6), MAX_SUBSTEPS);
        this.accumulator = Math.max(this.accumulator - substeps * SUBSTEP_DT, 0);
        this.frame(substeps);

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
        this.gpuTimer.destroy();
        if (this.canvas) {
            this.canvas.removeEventListener('mousedown', this.onMouseDown);
            window.removeEventListener('mousemove', this.onMouseMove);
            window.removeEventListener('mouseup', this.onMouseUp);
            this.canvas.removeEventListener('wheel', this.onWheel);
            this.canvas.removeEventListener('contextmenu', this.onContextMenu);
        }
        this.destroyGridBuffers();
        for (const b of [this.simParamsBuffer, this.colorBuffer, this.cameraBuffer, this.counterBuffer,
            this.counterReadback, this.msaaTexture, this.depthTexture]) b?.destroy();
    }
}

// ---- Compute-only API for the Measure tab (js/teach/experiments/cloth.js) and test/gpu/ ----

export const GRID_SIZES = [32, 48, 64, 96, 128];   // the Grid select's choices
export const MAX_ITERATIONS = 30;                   // the Iterations slider's maximum
export const MAX_MEASURE_SUBSTEPS = 960;            // 4 s of simulated time per measureCloth call
export const MAX_MEASURE_TIMED = 64;                // timed substeps per call (in chunks of MAX_SUBSTEPS)

const median = (values) => {
    const v = values.filter(x => typeof x === 'number' && Number.isFinite(x)).sort((a, b) => a - b);
    return v.length ? v[v.length >> 1] : null;
};

// Runs the cloth of `preset` on an n × n grid for `substeps` substeps (wind as the preset sets it,
// tearing off), then `timed` more substeps in chunks of MAX_SUBSTEPS, each chunk one command buffer
// timed on its own; reads the positions back once. between(), if given, is awaited after every
// MAX_SUBSTEPS × 8 substeps (the Measure runner passes ctx.step there, which throws to cancel).
// Resolves to { n, count, numConstraints, iterations, dispatchesPerSubstep, gpuMsPerSubstep (null
// without timestamp-query), wallMsPerSubstep, simulatedSeconds, positions, constraintData } where
// positions are the final (x, y, z, inverse mass) per particle.
export async function measureCloth(gpuDevice, { n = 64, preset = 'corners', iterations = 8, stiffness = 1,
    substeps = 480, timed = 32, between = null }) {
    if (!GRID_SIZES.includes(n)) throw new RangeError(`grid ${n} is not one of ${GRID_SIZES.join(', ')}`);
    if (!PRESETS[preset]) throw new RangeError(`unknown preset ${preset}`);
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > MAX_ITERATIONS) {
        throw new RangeError(`iterations = ${iterations} is outside 1 … ${MAX_ITERATIONS}`);
    }
    if (!Number.isInteger(substeps) || substeps < 0 || substeps > MAX_MEASURE_SUBSTEPS) {
        throw new RangeError(`substeps = ${substeps} is outside 0 … ${MAX_MEASURE_SUBSTEPS}`);
    }
    if (!Number.isInteger(timed) || timed < MAX_SUBSTEPS || timed > MAX_MEASURE_TIMED) {
        throw new RangeError(`timed = ${timed} is outside ${MAX_SUBSTEPS} … ${MAX_MEASURE_TIMED}`);
    }
    const device = gpuDevice.device ?? gpuDevice;
    const sim = new ClothSimulation(device, null, { gridN: n, presetKey: preset });
    Object.assign(sim, { iterations, stiffness: Math.min(Math.max(stiffness, 0.05), 1), tearOn: false });
    const timer = new GpuTimer(device, { maxPasses: 1, poolSize: 1 });
    try {
        for (let done = 0; done < substeps;) {
            const k = Math.min(MAX_SUBSTEPS, substeps - done);
            sim.stepCompute(k);
            done += k;
            if (done % (8 * MAX_SUBSTEPS) === 0 || done === substeps) {
                await device.queue.onSubmittedWorkDone();
                if (between) await between();
            }
        }
        const gpu = [], wall = [];
        for (let done = 0; done + MAX_SUBSTEPS <= timed; done += MAX_SUBSTEPS) {
            const t0 = performance.now();
            const timing = sim.stepCompute(MAX_SUBSTEPS, timer);
            await device.queue.onSubmittedWorkDone();
            wall.push((performance.now() - t0) / MAX_SUBSTEPS);
            const r = await timing;
            gpu.push(r ? r.total / MAX_SUBSTEPS : null);
        }
        const positions = await sim.readPositions();
        const cloth = sim.cloth;
        return {
            n, count: cloth.count, numConstraints: cloth.numConstraints, iterations,
            dispatchesPerSubstep: (cloth.pinnedCount > 0 ? 3 : 2) + iterations * NUM_COLORS,
            gpuMsPerSubstep: median(gpu), wallMsPerSubstep: median(wall),
            simulatedSeconds: sim.time, positions, constraintData: cloth.constraintData,
        };
    } finally {
        timer.destroy();
        sim.destroy();
    }
}

export async function initCloth(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-cloth');
    const output = document.getElementById('output-cloth');

    const simulation = new ClothSimulation(device, canvas);
    simulation.onStats = () => updateOutput();

    document.getElementById('btn-cloth-start').addEventListener('click', () => {
        simulation.start();
        updateOutput();
    });
    document.getElementById('btn-cloth-pause').addEventListener('click', () => {
        simulation.pause();
        updateOutput();
    });
    document.getElementById('btn-cloth-reset').addEventListener('click', () => {
        simulation.reset();
        updateOutput();
    });

    const gridSelect = document.getElementById('cloth-grid');
    gridSelect.addEventListener('change', () => {
        const n = parseInt(gridSelect.value);
        if (![32, 48, 64, 96, 128].includes(n)) return;
        simulation.setGridSize(n);
        updateOutput();
    });
    if (parseInt(gridSelect.value) !== simulation.gridN && [32, 48, 64, 96, 128].includes(parseInt(gridSelect.value))) {
        simulation.setGridSize(parseInt(gridSelect.value));
    }

    const slider = (id, parse, apply, format) => {
        const input = document.getElementById(id);
        const label = document.getElementById(`${id}-value`);
        const update = () => {
            apply(parse(input.value));
            label.textContent = format();
            updateOutput();
        };
        input.addEventListener('input', update);
        apply(parse(input.value));   // the browser may have restored an earlier value
        label.textContent = format();
        return { input, label, format };
    };
    slider('cloth-iterations', parseInt, (v) => { simulation.iterations = Math.min(Math.max(v, 1), 30); },
        () => simulation.iterations);
    slider('cloth-stiffness', parseFloat, (v) => { simulation.stiffness = Math.min(Math.max(v, 0.05), 1); },
        () => simulation.stiffness.toFixed(2));
    // Wind comes from the preset, so the slider starts at the preset's value
    document.getElementById('cloth-wind').value = simulation.windStrength;
    const windSlider = slider('cloth-wind', parseFloat, (v) => { simulation.windStrength = Math.min(Math.max(v, 0), 20); },
        () => `${simulation.windStrength.toFixed(1)} m/s`);
    slider('cloth-gravity', parseFloat, (v) => { simulation.gravity = Math.min(Math.max(v, 0), 20); },
        () => `${simulation.gravity.toFixed(1)} m/s²`);
    slider('cloth-tear-threshold', parseFloat, (v) => { simulation.tearRatio = Math.min(Math.max(v, 1.1), 4); },
        () => `${simulation.tearRatio.toFixed(2)}×`);

    const checkbox = (id, apply) => {
        const input = document.getElementById(id);
        input.addEventListener('change', () => {
            apply(input.checked);
            if (!simulation.running) simulation.render();
            updateOutput();
        });
        apply(input.checked);
    };
    checkbox('cloth-tear', (on) => { simulation.tearOn = on; });
    checkbox('cloth-wireframe', (on) => { simulation.showWireframe = on; });
    checkbox('cloth-particles', (on) => { simulation.showParticles = on; });

    const presetButtons = document.querySelectorAll('.cloth-preset');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!PRESETS[btn.dataset.preset]) return;
            simulation.loadPreset(btn.dataset.preset);
            windSlider.input.value = simulation.windStrength;
            windSlider.label.textContent = windSlider.format();
            showPreset();
            updateOutput();
        });
    });

    function showPreset() {
        presetButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.preset === simulation.presetKey));
    }

    function updateOutput() {
        const cloth = simulation.cloth;
        const n = simulation.gridN;
        const preset = PRESETS[simulation.presetKey];
        const dispatchesPerSubstep = (cloth.pinnedCount > 0 ? 3 : 2) + simulation.iterations * NUM_COLORS;
        const substepsPerSecond = Math.round(1 / SUBSTEP_DT);
        const projections = cloth.numConstraints * simulation.iterations * substepsPerSecond;
        const torn = simulation.tearOn || simulation.tornCount > 0
            ? ` · torn: ${simulation.tornCount.toLocaleString()}` : '';

        output.innerHTML = `<span class="success">✓ Cloth Simulation Active</span>

<span class="info">Configuration:</span>
• Particles: ${n} × ${n} = ${cloth.count.toLocaleString()} (${cloth.pinnedCount} pinned)
• Constraints: ${cloth.numConstraints.toLocaleString()} (structural ${cloth.kindCounts[0].toLocaleString()}, shear ${cloth.kindCounts[1].toLocaleString()}, bend ${cloth.kindCounts[2].toLocaleString()})${torn}
• Preset: ${preset.name}${preset.sphere ? ' (sphere collider)' : ''}
• Solver: ${simulation.iterations} iterations × ${NUM_COLORS} colors per substep, substep h = 1/${substepsPerSecond} s
• Stiffness: ${simulation.stiffness.toFixed(2)} (bend ×${BEND_FACTOR}) · Gravity: ${simulation.gravity.toFixed(1)} m/s² · Wind: ${simulation.windStrength.toFixed(1)} m/s
• Tearing: ${simulation.tearOn ? `on, at ${simulation.tearRatio.toFixed(2)}× rest length` : 'off'}
• Status: ${simulation.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'} · simulated ${simulation.time.toFixed(1)} s

<span class="info">Algorithm (position-based dynamics):</span>
• Verlet integration with gravity, gusty wind and air drag along the cloth normals
• Distance constraints projected Gauss-Seidel style; graph coloring (12 independent sets) makes every dispatch race-free
• Long-range tethers to the nearest pins stop the hanging weight from overstretching the links at the pins (off once torn)
• Sphere and floor collisions with friction; links stretched past the threshold tear
• Normals from neighboring particles on the GPU; triangles with a torn edge are not drawn

<span class="info">Performance:</span>
• Frame rate: ${simulation.running ? simulation.fps.toFixed(0) : 0} FPS (${simulation.lastSubsteps} substeps last frame)
• ${simulation.gpuTimer.frameText(simulation.running ? simulation.fps : 0, ['sim', 'render'])}
• Compute dispatches per substep: ${dispatchesPerSubstep}
• Constraint projections: ${(projections / 1e6).toFixed(1)} M/s at ${substepsPerSecond} substeps/s
• Readbacks: positions once per grab (CPU pick), 4-byte torn counter while tearing is on`;
    }

    showPreset();
    updateOutput();
    return simulation;
}

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}
