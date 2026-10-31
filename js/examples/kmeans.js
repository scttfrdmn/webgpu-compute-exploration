// K-means clustering (Lloyd's algorithm) of a 2D point cloud, one iteration per step on the GPU.
//
// Each iteration is two compute dispatches:
//   cs_assign  one invocation per 4 points: find the nearest centroid, write the label, count label
//              changes, and reduce the per-cluster (sum x, sum y, count, squared distance) of its
//              workgroup's 1,024 points in workgroup memory into one partial per (workgroup, cluster)
//   cs_update  one workgroup sums the partials of all workgroups per cluster (second level of the
//              reduction), moves each centroid to the mean of its points and appends it to its trail
// The only CPU readback is the 80-byte Stats struct (changed-label count, inertia, empty clusters,
// per-cluster counts), copied to a mappable buffer once per iteration for the info panel and the
// convergence test. Only one readback is in flight, so iterations never queue up behind it.
//
// World coordinates: x in [-ASPECT, ASPECT], y in [-1, 1] (y up), so distances are isotropic on
// the 4:3 canvas.
//
// Compute uniform layout (struct Params, 16 bytes):
//   offset  0  n          u32  number of points                    -> Uint32 view [0]
//   offset  4  k          u32  number of clusters (2..MAX_K)       -> Uint32 view [1]
//   offset  8  numGroups  u32  assign workgroups = ceil(n / TILE)  -> Uint32 view [2]
//   offset 12  pad0       u32                                      -> Uint32 view [3]
// Both dispatches of an iteration read the same values, so one slot suffices.
//
// Render uniform layout (struct RenderParams, 40 bytes):
//   offset  0  scale        vec2<f32>  world -> clip scale             -> Float32 view [0], [1]
//   offset  8  pixelClip    vec2<f32>  one pixel in clip units          -> Float32 view [2], [3]
//   offset 16  k            u32        number of clusters               -> Uint32 view [4]
//   offset 20  trailCount   u32        valid trail entries per centroid -> Uint32 view [5]
//   offset 24  pointSize    f32        point quad size in pixels        -> Float32 view [6]
//   offset 28  markerSize   f32        centroid marker diameter, px     -> Float32 view [7]
//   offset 32  pixelWorld   f32        one pixel in world units         -> Float32 view [8]
//   offset 36  voronoiAlpha f32        opacity of the Voronoi fill      -> Float32 view [9]
//
// Stats (storage, 80 bytes, cleared with clearBuffer before every iteration):
//   offset  0 changed atomic<u32> · 4 inertia f32 · 8 empty u32 · 12 pad · 16 counts array<u32, 16>

const MAX_K = 16;
const MAX_N = 500000;
const ASSIGN_WG = 256;                    // invocations per assign workgroup
const POINTS_PER_THREAD = 4;
const TILE = ASSIGN_WG * POINTS_PER_THREAD;  // points per assign workgroup (1,024)
const UPDATE_WG = 256;
const MAX_GROUPS = 512;                   // ≥ ceil(MAX_N / TILE) = 489, a multiple of UPDATE_WG
const TRAIL_LEN = 24;                     // centroid positions kept per cluster (newest first)
const COMPUTE_PARAMS_SIZE = 16;
const RENDER_PARAMS_SIZE = 40;
const STATS_SIZE = 80;
const UNASSIGNED = 0xffffffff;
const MAX_ITERATIONS = 300;
const KPP_SAMPLE = 20000;                 // k-means++ seeding runs on at most this many points
const ASPECT = 4 / 3;

// 16 distinguishable colors (sRGB 0..1); index = cluster
const PALETTE = [
    [0.902, 0.294, 0.235], [0.204, 0.596, 0.859], [0.180, 0.800, 0.443], [0.945, 0.769, 0.059],
    [0.608, 0.349, 0.714], [0.902, 0.494, 0.133], [0.102, 0.737, 0.612], [0.925, 0.439, 0.627],
    [0.580, 0.780, 0.200], [0.400, 0.450, 0.950], [0.850, 0.650, 0.450], [0.000, 0.850, 0.950],
    [0.750, 0.200, 0.450], [0.600, 0.600, 0.150], [0.950, 0.950, 0.700], [0.450, 0.300, 0.200],
];

const wgslPalette = PALETTE
    .map(([r, g, b]) => `        vec3<f32>(${r.toFixed(3)}, ${g.toFixed(3)}, ${b.toFixed(3)})`)
    .join(',\n');

const computeShader = `
const MAX_K: u32 = ${MAX_K}u;
const ASSIGN_WG: u32 = ${ASSIGN_WG}u;
const PPT: u32 = ${POINTS_PER_THREAD}u;
const TILE: u32 = ${TILE}u;
const UPDATE_WG: u32 = ${UPDATE_WG}u;
const MAX_GROUPS: u32 = ${MAX_GROUPS}u;
const TRAIL_LEN: u32 = ${TRAIL_LEN}u;
const FAR: f32 = 3.0e38;

struct Params {
    n: u32,
    k: u32,
    numGroups: u32,
    pad0: u32,
}

struct Stats {
    changed: atomic<u32>,
    inertia: f32,
    empty: u32,
    pad0: u32,
    counts: array<u32, ${MAX_K}>,
}

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> centroids: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;   // [group * MAX_K + cluster]
@group(0) @binding(4) var<storage, read_write> stats: Stats;
@group(0) @binding(5) var<storage, read_write> trail: array<vec2<f32>>;      // [cluster * TRAIL_LEN + age]
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> assignSums: array<vec4<f32>, ASSIGN_WG>;
var<workgroup> assignChanged: atomic<u32>;
var<workgroup> updateSums: array<vec4<f32>, UPDATE_WG>;

// Lloyd step 1 (assignment) plus the first level of the update reduction.
// Workgroup g owns points [g * TILE, (g + 1) * TILE); invocation t handles g * TILE + i * ASSIGN_WG + t
// for i = 0..PPT-1 (coalesced). Each point takes the nearest centroid by squared distance (ties go to
// the lower index). Then, for each cluster c, every invocation sums its own points of cluster c as
// (x, y, 1, d²) and the workgroup tree-reduces those 256 vec4s in workgroup memory; invocation 0
// stores the result as partials[g * MAX_K + c]. Out-of-range points get the sentinel label MAX_K and
// contribute nothing, so every invocation reaches every barrier.
@compute @workgroup_size(ASSIGN_WG)
fn cs_assign(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let base = wid.x * TILE;
    var pts: array<vec2<f32>, PPT>;
    var labs: array<u32, PPT>;
    var dists: array<f32, PPT>;
    var changedHere = 0u;

    for (var i = 0u; i < PPT; i++) {
        let idx = base + i * ASSIGN_WG + t;
        labs[i] = MAX_K;
        if (idx < params.n) {
            let p = points[idx];
            var best = 0u;
            var bestD = FAR;
            for (var j = 0u; j < MAX_K; j++) {
                if (j >= params.k) {
                    break;
                }
                let delta = p - centroids[j];
                let dd = dot(delta, delta);
                if (dd < bestD) {
                    bestD = dd;
                    best = j;
                }
            }
            if (labels[idx] != best) {
                changedHere++;
                labels[idx] = best;
            }
            pts[i] = p;
            labs[i] = best;
            dists[i] = bestD;
        }
    }
    if (changedHere > 0u) {
        atomicAdd(&assignChanged, changedHere);
    }

    for (var c = 0u; c < MAX_K; c++) {
        if (c >= params.k) {
            break;
        }
        var acc = vec4<f32>(0.0);
        for (var i = 0u; i < PPT; i++) {
            if (labs[i] == c) {
                acc += vec4<f32>(pts[i], 1.0, dists[i]);
            }
        }
        assignSums[t] = acc;
        workgroupBarrier();
        for (var s = ASSIGN_WG / 2u; s > 0u; s >>= 1u) {
            if (t < s) {
                assignSums[t] += assignSums[t + s];
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            partials[wid.x * MAX_K + c] = assignSums[0];
        }
        workgroupBarrier();   // assignSums is reused for the next cluster
    }

    // The loop above ran at least two barriers (k >= 2), so every atomicAdd has landed
    if (t == 0u) {
        let groupChanged = atomicLoad(&assignChanged);
        if (groupChanged > 0u) {
            atomicAdd(&stats.changed, groupChanged);
        }
    }
}

// Lloyd step 2 (update): the second level of the reduction, in a single workgroup.
// For each cluster c, invocation t sums partials of groups t, t + 256 (fixed order), the workgroup
// tree-reduces the 256 results, and invocation 0 sets centroid c = (sum x, sum y) / count.
// Empty clusters (count 0) keep their previous centroid. The centroid is pushed onto its trail
// (age 0 = newest), and the counts and the inertia Σ d² (measured against the centroids that did
// the assignment, i.e. before this update) go to Stats.
@compute @workgroup_size(UPDATE_WG)
fn cs_update(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var inertia = 0.0;
    var empty = 0u;

    for (var c = 0u; c < MAX_K; c++) {
        if (c >= params.k) {
            break;
        }
        var acc = vec4<f32>(0.0);
        for (var j = 0u; j < MAX_GROUPS / UPDATE_WG; j++) {
            let g = j * UPDATE_WG + t;
            if (g < params.numGroups) {
                acc += partials[g * MAX_K + c];
            }
        }
        updateSums[t] = acc;
        workgroupBarrier();
        for (var s = UPDATE_WG / 2u; s > 0u; s >>= 1u) {
            if (t < s) {
                updateSums[t] += updateSums[t + s];
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            let total = updateSums[0];
            var next = centroids[c];
            if (total.z > 0.0) {
                next = total.xy / total.z;
            } else {
                empty++;
            }
            centroids[c] = next;
            stats.counts[c] = u32(total.z);
            inertia += total.w;
            for (var h = TRAIL_LEN - 1u; h > 0u; h--) {
                trail[c * TRAIL_LEN + h] = trail[c * TRAIL_LEN + h - 1u];
            }
            trail[c * TRAIL_LEN] = next;
        }
        workgroupBarrier();   // updateSums is reused for the next cluster
    }

    if (t == 0u) {
        stats.inertia = inertia;
        stats.empty = empty;
    }
}
`;

const renderShader = `
const MAX_K: u32 = ${MAX_K}u;
const TRAIL_LEN: u32 = ${TRAIL_LEN}u;
const TRAIL_WIDTH: f32 = 2.5;   // pixels
const FAR: f32 = 3.0e38;

struct RenderParams {
    scale: vec2<f32>,
    pixelClip: vec2<f32>,
    k: u32,
    trailCount: u32,
    pointSize: f32,
    markerSize: f32,
    pixelWorld: f32,
    voronoiAlpha: f32,
}

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read> centroids: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> trail: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> rp: RenderParams;

fn clusterColor(c: u32) -> vec3<f32> {
    var palette = array<vec3<f32>, 16>(
${wgslPalette}
    );
    return palette[c % 16u];
}

// Two triangles covering [-1, 1]²
fn quadCorner(vi: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    return corners[vi % 6u];
}

// ---- Voronoi overlay: full-screen triangle, each pixel shaded by its nearest centroid ----
struct ScreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec2<f32>,
}

@vertex
fn vs_voronoi(@builtin(vertex_index) vi: u32) -> ScreenOut {
    var tri = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: ScreenOut;
    out.position = vec4<f32>(tri[vi], 0.0, 1.0);
    out.world = tri[vi] / rp.scale;
    return out;
}

// Nearest and second-nearest centroid; the distance from the pixel to the bisector between them,
// (d2² - d1²) / (2·|c2 - c1|), draws the cell borders with a constant width in pixels.
@fragment
fn fs_voronoi(frag: ScreenOut) -> @location(0) vec4<f32> {
    var d1 = FAR;
    var d2 = FAR;
    var i1 = 0u;
    var i2 = 0u;
    for (var j = 0u; j < MAX_K; j++) {
        if (j >= rp.k) {
            break;
        }
        let delta = frag.world - centroids[j];
        let dd = dot(delta, delta);
        if (dd < d1) {
            d2 = d1;
            i2 = i1;
            d1 = dd;
            i1 = j;
        } else if (dd < d2) {
            d2 = dd;
            i2 = j;
        }
    }
    let sep = length(centroids[i2] - centroids[i1]);
    var edge = 0.0;
    if (sep > 1e-6) {
        let edgePx = (d2 - d1) / (2.0 * sep) / rp.pixelWorld;
        edge = 1.0 - smoothstep(0.5, 1.5, edgePx);
    }
    let color = mix(clusterColor(i1), vec3<f32>(0.9, 0.92, 0.95), edge);
    return vec4<f32>(color, mix(rp.voronoiAlpha, 0.45, edge));
}

// ---- Points: one instanced quad per point, colored by its label (gray before the first pass) ----
struct PointOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_points(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PointOut {
    let lab = labels[ii];
    var out: PointOut;
    out.color = vec4<f32>(0.62, 0.65, 0.72, 0.55);
    if (lab < rp.k) {
        out.color = vec4<f32>(clusterColor(lab), 0.8);
    }
    let offset = quadCorner(vi) * 0.5 * rp.pointSize * rp.pixelClip;
    out.position = vec4<f32>(points[ii] * rp.scale + offset, 0.0, 1.0);
    return out;
}

@fragment
fn fs_points(frag: PointOut) -> @location(0) vec4<f32> {
    return frag.color;
}

// ---- Trails: instance c * (TRAIL_LEN - 1) + j is the segment from trail age j to age j + 1 ----
struct TrailOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_trail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TrailOut {
    let c = ii / (TRAIL_LEN - 1u);
    let j = ii % (TRAIL_LEN - 1u);
    var out: TrailOut;   // zero: a degenerate (invisible) triangle unless filled in below
    if (c >= rp.k || j + 1u >= rp.trailCount) {
        return out;
    }
    let a = trail[c * TRAIL_LEN + j] * rp.scale;
    let b = trail[c * TRAIL_LEN + j + 1u] * rp.scale;
    let px = (b - a) / rp.pixelClip;
    let len = length(px);
    if (len < 1e-3) {
        return out;
    }
    let dir = px / len;
    let nrm = vec2<f32>(-dir.y, dir.x);
    let corner = quadCorner(vi);
    let along = select(a, b, corner.x > 0.0);
    out.position = vec4<f32>(along + nrm * corner.y * 0.5 * TRAIL_WIDTH * rp.pixelClip, 0.0, 1.0);
    let age = f32(j) + 0.5 * (corner.x + 1.0);
    let fade = 1.0 - age / f32(TRAIL_LEN - 1u);
    out.color = vec4<f32>(mix(clusterColor(c), vec3<f32>(1.0), 0.5), 0.9 * fade);
    return out;
}

@fragment
fn fs_trail(frag: TrailOut) -> @location(0) vec4<f32> {
    return frag.color;
}

// ---- Centroid markers: dark outline, white ring, cluster-colored center ----
struct MarkerOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) cluster: u32,
}

@vertex
fn vs_marker(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MarkerOut {
    let corner = quadCorner(vi);
    var out: MarkerOut;
    out.position = vec4<f32>(centroids[ii] * rp.scale + corner * 0.5 * rp.markerSize * rp.pixelClip, 0.0, 1.0);
    out.local = corner;
    out.cluster = ii;
    return out;
}

@fragment
fn fs_marker(frag: MarkerOut) -> @location(0) vec4<f32> {
    let radius = 0.5 * rp.markerSize;
    let r = length(frag.local) * radius;   // pixels from the center
    if (r > radius) {
        discard;
    }
    var color = clusterColor(frag.cluster);
    if (r > radius - 2.0) {
        color = vec3<f32>(0.04, 0.05, 0.08);
    } else if (r > radius - 5.0) {
        color = vec3<f32>(1.0, 1.0, 1.0);
    }
    return vec4<f32>(color, 1.0);
}
`;

// Small seeded PRNG (mulberry32) so a data set is reproducible from its seed
function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// n points as interleaved (x, y): `blobs` anisotropic Gaussian blobs (random center, widths, rotation
// and weight) plus a fraction `noise` of points spread uniformly over the whole domain
function generatePoints(n, blobs, noise, seed) {
    const rng = mulberry32(seed);
    const normal = () => {
        const u = Math.max(rng(), 1e-12);
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
    };

    const centers = [];
    for (let b = 0; b < blobs; b++) {
        // Best of a few candidates: keeps blob centers apart without a hard rule
        let best = null;
        let bestGap = -1;
        for (let tries = 0; tries < 12; tries++) {
            const cx = (rng() * 2 - 1) * ASPECT * 0.78;
            const cy = (rng() * 2 - 1) * 0.78;
            let gap = Infinity;
            for (const c of centers) gap = Math.min(gap, Math.hypot(cx - c.x, cy - c.y));
            if (gap > bestGap) {
                bestGap = gap;
                best = { x: cx, y: cy };
            }
        }
        const angle = rng() * Math.PI;
        centers.push({
            ...best,
            sx: 0.05 + rng() * 0.11,
            sy: 0.04 + rng() * 0.08,
            cos: Math.cos(angle),
            sin: Math.sin(angle),
            weight: 0.5 + rng(),
        });
    }
    const cumulative = [];
    let totalWeight = 0;
    for (const c of centers) {
        totalWeight += c.weight;
        cumulative.push(totalWeight);
    }

    const data = new Float32Array(n * 2);
    const noiseCount = Math.round(n * noise);
    for (let i = 0; i < n; i++) {
        let x;
        let y;
        if (i < noiseCount || centers.length === 0) {
            x = (rng() * 2 - 1) * ASPECT;
            y = rng() * 2 - 1;
        } else {
            const r = rng() * totalWeight;
            let b = 0;
            while (b < centers.length - 1 && cumulative[b] < r) b++;
            const c = centers[b];
            const u = normal() * c.sx;
            const v = normal() * c.sy;
            x = c.x + u * c.cos - v * c.sin;
            y = c.y + u * c.sin + v * c.cos;
        }
        data[2 * i] = Math.min(Math.max(x, -ASPECT), ASPECT);
        data[2 * i + 1] = Math.min(Math.max(y, -1), 1);
    }
    // Shuffle so noise and blobs are interleaved in memory (and in draw order)
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tx = data[2 * i];
        const ty = data[2 * i + 1];
        data[2 * i] = data[2 * j];
        data[2 * i + 1] = data[2 * j + 1];
        data[2 * j] = tx;
        data[2 * j + 1] = ty;
    }
    return data;
}

// Initial centroids as interleaved (x, y).
// 'random' (Forgy): k distinct data points chosen uniformly.
// 'kmeans++': D² seeding (Arthur & Vassilvitskii 2007) on a uniform sample of at most KPP_SAMPLE
// points, on the CPU: the first center is a random sample point, each next one is drawn with
// probability proportional to its squared distance to the nearest center chosen so far.
function initCentroids(data, n, k, method, rng) {
    const out = new Float32Array(k * 2);
    if (method === 'random') {
        const chosen = new Set();
        for (let c = 0; c < k; c++) {
            let idx;
            do {
                idx = Math.floor(rng() * n);
            } while (chosen.has(idx) && chosen.size < n);
            chosen.add(idx);
            out[2 * c] = data[2 * idx];
            out[2 * c + 1] = data[2 * idx + 1];
        }
        return out;
    }

    const m = Math.min(n, KPP_SAMPLE);
    const sample = new Float32Array(m * 2);
    for (let i = 0; i < m; i++) {
        const idx = m === n ? i : Math.floor(rng() * n);
        sample[2 * i] = data[2 * idx];
        sample[2 * i + 1] = data[2 * idx + 1];
    }
    const d2 = new Float64Array(m).fill(Infinity);
    let pick = Math.floor(rng() * m);
    for (let c = 0; c < k; c++) {
        const cx = sample[2 * pick];
        const cy = sample[2 * pick + 1];
        out[2 * c] = cx;
        out[2 * c + 1] = cy;
        if (c === k - 1) break;
        let total = 0;
        for (let i = 0; i < m; i++) {
            const dx = sample[2 * i] - cx;
            const dy = sample[2 * i + 1] - cy;
            const d = dx * dx + dy * dy;
            if (d < d2[i]) d2[i] = d;
            total += d2[i];
        }
        if (total <= 0) {
            pick = Math.floor(rng() * m);   // every sample point is already a center
            continue;
        }
        let r = rng() * total;
        pick = m - 1;
        for (let i = 0; i < m; i++) {
            r -= d2[i];
            if (r <= 0) {
                pick = i;
                break;
            }
        }
    }
    return out;
}

class KMeans {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;

        this.n = 100000;
        this.blobs = 5;
        this.noise = 0.05;
        this.k = 5;
        this.initMethod = 'kmeans++';
        this.speed = 5;             // iterations per second while running
        this.voronoi = true;
        this.dataSeed = (Math.random() * 0xffffffff) >>> 0;
        this.data = null;

        this.iteration = 0;         // iterations issued since the last reset
        this.changed = null;        // results of the latest finished iteration (readback)
        this.inertia = null;
        this.emptyClusters = 0;
        this.counts = [];
        this.status = 'ready';      // ready | running | paused | converged | limit
        this.readbackBusy = false;
        this.generation = 0;        // bumps on reset, so stale readbacks are ignored
        this.lastIterationTime = 0;

        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;

        this.computeParams = new ArrayBuffer(COMPUTE_PARAMS_SIZE);
        this.computeU32 = new Uint32Array(this.computeParams);
        this.renderParams = new ArrayBuffer(RENDER_PARAMS_SIZE);
        this.renderF32 = new Float32Array(this.renderParams);
        this.renderU32 = new Uint32Array(this.renderParams);

        this.checkLimits();
        this.setupBuffers();
        this.setupComputePipelines();
        this.setupRenderPipelines();
        this.setupCanvas();
    }

    checkLimits() {
        const limits = this.device.limits;
        const groups = Math.ceil(MAX_N / TILE);
        if (groups > MAX_GROUPS || groups > limits.maxComputeWorkgroupsPerDimension) {
            throw new Error(`K-means needs ${groups} workgroups per dispatch`);
        }
        if (limits.maxComputeWorkgroupStorageSize < Math.max(ASSIGN_WG, UPDATE_WG) * 16 + 4 ||
            limits.maxComputeInvocationsPerWorkgroup < Math.max(ASSIGN_WG, UPDATE_WG)) {
            throw new Error('K-means needs 256 invocations and 4 KB of workgroup memory per workgroup');
        }
    }

    setupBuffers() {
        const d = this.device;
        this.pointBuffer = d.createBuffer({ size: MAX_N * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.labelBuffer = d.createBuffer({ size: MAX_N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.centroidBuffer = d.createBuffer({ size: MAX_K * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.partialBuffer = d.createBuffer({ size: MAX_GROUPS * MAX_K * 16, usage: GPUBufferUsage.STORAGE });
        this.statsBuffer = d.createBuffer({
            size: STATS_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.trailBuffer = d.createBuffer({ size: MAX_K * TRAIL_LEN * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.readbackBuffer = d.createBuffer({ size: STATS_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.computeParamsBuffer = d.createBuffer({ size: COMPUTE_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.renderParamsBuffer = d.createBuffer({ size: RENDER_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.labelInit = new Uint32Array(MAX_N).fill(UNASSIGNED);
    }

    setupComputePipelines() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });
        const C = GPUShaderStage.COMPUTE;
        const layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: C, buffer: { type: 'storage' } },
                { binding: 2, visibility: C, buffer: { type: 'storage' } },
                { binding: 3, visibility: C, buffer: { type: 'storage' } },
                { binding: 4, visibility: C, buffer: { type: 'storage' } },
                { binding: 5, visibility: C, buffer: { type: 'storage' } },
                { binding: 6, visibility: C, buffer: { type: 'uniform' } }
            ]
        });
        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
        this.assignPipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'cs_assign' }
        });
        this.updatePipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'cs_update' }
        });
        this.computeBindGroup = this.device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.pointBuffer } },
                { binding: 1, resource: { buffer: this.labelBuffer } },
                { binding: 2, resource: { buffer: this.centroidBuffer } },
                { binding: 3, resource: { buffer: this.partialBuffer } },
                { binding: 4, resource: { buffer: this.statsBuffer } },
                { binding: 5, resource: { buffer: this.trailBuffer } },
                { binding: 6, resource: { buffer: this.computeParamsBuffer } }
            ]
        });
    }

    setupRenderPipelines() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });
        const V = GPUShaderStage.VERTEX;
        const F = GPUShaderStage.FRAGMENT;
        const layout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: V | F, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: V | F, buffer: { type: 'uniform' } }
            ]
        });
        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
        const blend = {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
        };
        const make = (vs, fs) => this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: { module: shaderModule, entryPoint: vs },
            fragment: { module: shaderModule, entryPoint: fs, targets: [{ format: 'bgra8unorm', blend }] },
            primitive: { topology: 'triangle-list' }
        });
        this.voronoiPipeline = make('vs_voronoi', 'fs_voronoi');
        this.pointPipeline = make('vs_points', 'fs_points');
        this.trailPipeline = make('vs_trail', 'fs_trail');
        this.markerPipeline = make('vs_marker', 'fs_marker');
        this.renderBindGroup = this.device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.pointBuffer } },
                { binding: 1, resource: { buffer: this.labelBuffer } },
                { binding: 2, resource: { buffer: this.centroidBuffer } },
                { binding: 3, resource: { buffer: this.trailBuffer } },
                { binding: 4, resource: { buffer: this.renderParamsBuffer } }
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

    get numGroups() {
        return Math.ceil(this.n / TILE);
    }

    // See the compute uniform layout table at the top of the file
    writeComputeParams() {
        this.computeU32[0] = this.n;
        this.computeU32[1] = this.k;
        this.computeU32[2] = this.numGroups;
        this.computeU32[3] = 0;
        this.device.queue.writeBuffer(this.computeParamsBuffer, 0, this.computeParams);
    }

    // See the render uniform layout table at the top of the file
    writeRenderParams() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.renderF32[0] = 1 / ASPECT;
        this.renderF32[1] = 1;
        this.renderF32[2] = 2 / w;
        this.renderF32[3] = 2 / h;
        this.renderU32[4] = this.k;
        this.renderU32[5] = Math.min(this.iteration + 1, TRAIL_LEN);
        this.renderF32[6] = this.n > 250000 ? 1.5 : 2;
        this.renderF32[7] = 20;
        this.renderF32[8] = 2 / h;
        this.renderF32[9] = 0.16;
        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, this.renderParams);
    }

    // New point cloud from a new seed, then new centroids
    regenerate() {
        this.dataSeed = (Math.random() * 0xffffffff) >>> 0;
        this.data = generatePoints(this.n, this.blobs, this.noise, this.dataSeed);
        this.device.queue.writeBuffer(this.pointBuffer, 0, this.data);
        this.reset();
    }

    // New initial centroids (all trail entries at the start position), every label unassigned
    reset() {
        this.generation++;
        const init = initCentroids(this.data, this.n, this.k, this.initMethod, Math.random);
        const centroids = new Float32Array(MAX_K * 2);
        centroids.set(init);
        const trail = new Float32Array(MAX_K * TRAIL_LEN * 2);
        for (let c = 0; c < this.k; c++) {
            for (let h = 0; h < TRAIL_LEN; h++) {
                trail[(c * TRAIL_LEN + h) * 2] = init[2 * c];
                trail[(c * TRAIL_LEN + h) * 2 + 1] = init[2 * c + 1];
            }
        }
        this.device.queue.writeBuffer(this.centroidBuffer, 0, centroids);
        this.device.queue.writeBuffer(this.trailBuffer, 0, trail);
        this.device.queue.writeBuffer(this.labelBuffer, 0, this.labelInit, 0, this.n);

        this.iteration = 0;
        this.changed = null;
        this.inertia = null;
        this.emptyClusters = 0;
        this.counts = [];
        this.status = this.running ? 'running' : 'ready';
        this.render();
    }

    canIterate() {
        return !this.readbackBusy && this.status !== 'converged' && this.status !== 'limit' && this.data !== null;
    }

    // Encodes one Lloyd iteration: clear Stats, assign + partial sums, final sums + centroid update,
    // then copy Stats to the mappable readback buffer
    encodeIteration(commandEncoder) {
        this.writeComputeParams();
        commandEncoder.clearBuffer(this.statsBuffer);
        const pass = commandEncoder.beginComputePass();
        pass.setBindGroup(0, this.computeBindGroup);
        pass.setPipeline(this.assignPipeline);
        pass.dispatchWorkgroups(this.numGroups);
        pass.setPipeline(this.updatePipeline);
        pass.dispatchWorkgroups(1);
        pass.end();
        commandEncoder.copyBufferToBuffer(this.statsBuffer, 0, this.readbackBuffer, 0, STATS_SIZE);
    }

    // Reads the 80-byte Stats of the iteration just submitted; stops when no label changed
    readStats() {
        const generation = this.generation;
        const iteration = this.iteration;
        this.readbackBusy = true;
        this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            if (this.destroyed) return;
            const bytes = this.readbackBuffer.getMappedRange().slice(0);
            this.readbackBuffer.unmap();
            this.readbackBusy = false;
            if (generation !== this.generation) return;
            const u32 = new Uint32Array(bytes);
            const f32 = new Float32Array(bytes);
            this.changed = u32[0];
            this.inertia = f32[1];
            this.emptyClusters = u32[2];
            this.counts = Array.from(u32.subarray(4, 4 + this.k));
            if (this.changed === 0) {
                this.status = 'converged';
                this.running = false;
            } else if (iteration >= MAX_ITERATIONS) {
                this.status = 'limit';
                this.running = false;
            }
            if (this.onStats) this.onStats();
        }).catch(() => {
            this.readbackBusy = false;
        });
    }

    // Runs one iteration if `iterate`, then draws
    frame(iterate) {
        const commandEncoder = this.device.createCommandEncoder();
        if (iterate) {
            this.iteration++;
            this.encodeIteration(commandEncoder);
        }
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        if (iterate) this.readStats();
    }

    step() {
        if (!this.canIterate()) return;
        this.frame(true);
        if (this.onStats) this.onStats();
    }

    encodeRender(commandEncoder) {
        this.writeRenderParams();
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.035, g: 0.04, b: 0.06, a: 1.0 }
            }]
        });
        pass.setBindGroup(0, this.renderBindGroup);
        if (this.voronoi) {
            pass.setPipeline(this.voronoiPipeline);
            pass.draw(3);
        }
        pass.setPipeline(this.pointPipeline);
        pass.draw(6, this.n);
        pass.setPipeline(this.trailPipeline);
        pass.draw(6, this.k * (TRAIL_LEN - 1));
        pass.setPipeline(this.markerPipeline);
        pass.draw(6, this.k);
        pass.end();
    }

    render() {
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        if (this.running) return;
        if (this.status === 'converged' || this.status === 'limit') return;
        this.running = true;
        this.status = 'running';
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.lastIterationTime = 0;
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
        if (this.status === 'running') this.status = 'paused';
    }

    animate() {
        if (!this.running || this.destroyed) {
            this.fps = 0;
            if (this.onStats) this.onStats();
            return;
        }

        const now = performance.now();
        const due = now - this.lastIterationTime >= 1000 / this.speed - 1;
        const iterate = due && this.canIterate();
        if (iterate) this.lastIterationTime = now;
        this.frame(iterate);

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
        [this.pointBuffer, this.labelBuffer, this.centroidBuffer, this.partialBuffer, this.statsBuffer,
         this.trailBuffer, this.readbackBuffer, this.computeParamsBuffer, this.renderParamsBuffer]
            .forEach(b => b.destroy());
    }
}

export async function initKMeans(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-kmeans');
    const output = document.getElementById('output-kmeans');

    const kmeans = new KMeans(device, canvas);
    kmeans.regenerate();
    kmeans.onStats = () => updateOutput();

    document.getElementById('btn-kmeans-start').addEventListener('click', () => {
        kmeans.start();
        updateOutput();
    });

    document.getElementById('btn-kmeans-pause').addEventListener('click', () => {
        kmeans.pause();
        updateOutput();
    });

    document.getElementById('btn-kmeans-step').addEventListener('click', () => {
        kmeans.step();
        updateOutput();
    });

    document.getElementById('btn-kmeans-reset').addEventListener('click', () => {
        kmeans.reset();
        updateOutput();
    });

    document.getElementById('btn-kmeans-regenerate').addEventListener('click', () => {
        kmeans.regenerate();
        updateOutput();
    });

    // Sliders show their value while dragging and apply on release (regenerating 500k points or
    // re-seeding on every input event would stall the page)
    const slider = (id, parse, show, apply) => {
        const input = document.getElementById(id);
        const label = document.getElementById(`${id}-value`);
        input.addEventListener('input', () => {
            label.textContent = show(parse(input.value));
        });
        input.addEventListener('change', () => {
            apply(parse(input.value));
            label.textContent = show(parse(input.value));
            updateOutput();
        });
    };
    const clampInt = (lo, hi) => (v) => Math.min(Math.max(parseInt(v, 10) || lo, lo), hi);

    slider('kmeans-k', clampInt(2, MAX_K), v => v, (v) => {
        kmeans.k = v;
        kmeans.reset();
    });
    slider('kmeans-n', clampInt(10000, MAX_N), v => `${Math.round(v / 1000)}k`, (v) => {
        kmeans.n = v;
        kmeans.regenerate();
    });
    slider('kmeans-blobs', clampInt(1, MAX_K), v => v, (v) => {
        kmeans.blobs = v;
        kmeans.regenerate();
    });
    slider('kmeans-noise', clampInt(0, 50), v => `${v}%`, (v) => {
        kmeans.noise = v / 100;
        kmeans.regenerate();
    });

    const speedInput = document.getElementById('kmeans-speed');
    const speedLabel = document.getElementById('kmeans-speed-value');
    speedInput.addEventListener('input', () => {
        kmeans.speed = clampInt(1, 60)(speedInput.value);
        speedLabel.textContent = kmeans.speed;
        updateOutput();
    });

    document.getElementById('kmeans-init').addEventListener('change', (e) => {
        kmeans.initMethod = e.target.value === 'random' ? 'random' : 'kmeans++';
        kmeans.reset();
        updateOutput();
    });

    document.getElementById('kmeans-voronoi').addEventListener('change', (e) => {
        kmeans.voronoi = e.target.checked;
        if (!kmeans.running) kmeans.render();
        updateOutput();
    });

    function updateOutput() {
        const statusText = {
            ready: '<span class="info">Ready: press Start or Step</span>',
            running: '<span class="success">Running</span>',
            paused: '<span class="info">Paused</span>',
            converged: `<span class="success">Converged after ${kmeans.iteration} iteration${kmeans.iteration === 1 ? '' : 's'} (no label changed)</span>`,
            limit: `<span class="info">Stopped at the ${MAX_ITERATIONS}-iteration limit</span>`,
        }[kmeans.status];
        const done = kmeans.changed !== null;
        const finished = kmeans.readbackBusy ? kmeans.iteration - 1 : kmeans.iteration;
        const sizes = kmeans.counts.length
            ? kmeans.counts.map(c => c.toLocaleString()).join(' · ')
            : '–';

        output.innerHTML = `<span class="success">✓ K-Means Clustering Active</span>

<span class="info">Configuration:</span>
• Points: ${kmeans.n.toLocaleString()} (${kmeans.blobs} Gaussian blobs, ${Math.round(kmeans.noise * 100)}% uniform noise)
• Clusters K: ${kmeans.k} · Initialization: ${kmeans.initMethod === 'random' ? 'random data points (Forgy)' : `k-means++ (CPU, on ${Math.min(kmeans.n, KPP_SAMPLE).toLocaleString()} sampled points)`}
• Speed: ${kmeans.speed} iterations / s (at most one per frame) · Voronoi overlay: ${kmeans.voronoi ? 'on' : 'off'}
• Status: ${statusText}

<span class="info">Convergence:</span>
• Iteration: ${finished}
• Points that changed cluster: ${done ? kmeans.changed.toLocaleString() : '–'}
• Inertia Σ‖x − c‖²: ${done ? kmeans.inertia.toPrecision(6) : '–'}${done ? ` (mean ${(kmeans.inertia / kmeans.n).toExponential(3)} per point)` : ''}
• Cluster sizes: ${sizes}${kmeans.emptyClusters ? ` (${kmeans.emptyClusters} empty, kept in place)` : ''}

<span class="info">Algorithm (Lloyd's k-means):</span>
• Assign: each point takes its nearest centroid (one invocation per 4 points, K ≤ ${MAX_K})
• Update: per-cluster sums and counts in two levels: workgroup-memory tree reduction per 1,024 points, then one workgroup sums the ${kmeans.numGroups} partials
• Each centroid moves to the mean of its points; an empty cluster keeps its old centroid
• Stops when no point changes cluster; inertia never increases from one iteration to the next

<span class="info">Performance:</span>
• Frame rate: ${kmeans.running ? kmeans.fps.toFixed(0) : 0} FPS
• Per iteration: ${kmeans.numGroups + 1} workgroups in 2 dispatches, ${(kmeans.n * kmeans.k).toLocaleString()} distance evaluations
• Readback: 80 bytes of statistics per iteration (changed count, inertia, cluster sizes); points and labels stay on the GPU`;
    }

    updateOutput();
    return kmeans;
}
