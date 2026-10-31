// Monte Carlo path tracer with progressive accumulation.
//
// Every frame a compute shader traces ONE path per pixel (camera ray, up to 8 bounces, a shadow
// ray toward a light at every diffuse bounce) and adds its radiance to a running per-pixel sum in
// a storage buffer. A full-screen fragment pass divides by the sample count, applies exposure, a
// tone curve (ACES or Reinhard) and the sRGB transfer curve. Changing the camera, scene or any
// light-transport parameter clears the sum; exposure and tone mapping only affect the display
// pass, so they keep the accumulated samples.
//
// GPU workload bounds (a too-long GPU job per frame can hang the machine):
//   - 1 sample per pixel per dispatch, 1 dispatch per animation frame, at most MAX_IN_FLIGHT
//     frames queued on the GPU.
//   - Path length: at most MAX_BOUNCES = 8 scattering events, so at most 9 path segments plus 8
//     shadow rays = 17 rays per pixel. Russian roulette ends paths early after the 3rd bounce.
//   - Scene intersection is brute force over at most MAX_PRIMS = 64 primitives; every loop in the
//     shaders has a constant bound.
//   - Resolution at most 800 × 600 (the select only offers 100%, 75% and 50%).
//   Worst case per frame: 480,000 pixels × 17 rays = 8.2 million rays × 64 primitive tests =
//   522 million ray/primitive tests. Typical (measured with a CPU port of this kernel): the
//   default Cornell box (9 primitives, 5 bounces) averages 3.7 rays per pixel, i.e. ~1.8 million
//   rays and ~16 million tests per frame; Spheres (62 primitives) 2.2 rays per pixel, ~1.1
//   million rays and ~65 million tests. The run stops by itself at the target sample count.
//
// Uniform layout (struct Params, 160 bytes, compute shader):
//   offset   0  size        vec2<u32>  render width, height                -> Uint32 view [0], [1]
//   offset   8  sampleCount u32        samples already in the accumulator  -> Uint32 view [2]
//   offset  12  maxBounces  u32        1..MAX_BOUNCES                      -> Uint32 view [3]
//   offset  16  primCount   u32        primitives in the scene buffer      -> Uint32 view [4]
//   offset  20  lightCount  u32        leading primitives that are lights  -> Uint32 view [5]
//   offset  24  envType     u32        0 black, 1 sky gradient, 2 uniform  -> Uint32 view [6]
//   offset  28  useNEE      u32        1 = next-event estimation + MIS     -> Uint32 view [7]
//   offset  32  camPos      vec4<f32>  eye (xyz), tan(fov/2) (w)           -> Float32 view [8..11]
//   offset  48  camForward  vec4<f32>  forward (xyz), lens radius (w)      -> Float32 view [12..15]
//   offset  64  camRight    vec4<f32>  right (xyz), focus distance (w)     -> Float32 view [16..19]
//   offset  80  camUp       vec4<f32>  up (xyz), radiance clamp, 0=off (w) -> Float32 view [20..23]
//   offset  96  skyHorizon  vec4<f32>  horizon color (xyz), intensity (w)  -> Float32 view [24..27]
//   offset 112  skyZenith   vec4<f32>  zenith color (xyz)                  -> Float32 view [28..31]
//   offset 128  skyGround   vec4<f32>  color below the horizon (xyz)       -> Float32 view [32..35]
//   offset 144  seed        u32        per-session random seed             -> Uint32 view [36]
//   offset 148  (padding to the struct's 16-byte alignment)
//
// Display uniform (struct Display, 24 bytes, render shader):
//   offset   0  renderSize  vec2<u32>  accumulator width, height           -> Uint32 view [0], [1]
//   offset   8  canvasSize  vec2<u32>  canvas width, height                -> Uint32 view [2], [3]
//   offset  16  toneMap     u32        0 = ACES, 1 = Reinhard              -> Uint32 view [4]
//   offset  20  exposure    f32        linear multiplier 2^EV              -> Float32 view [5]
//
// Scene storage buffer: array<Prim>, 96 bytes (6 × vec4) per primitive. All fields are vec4 so
// no vec3 alignment padding is involved:
//   offset  0  geo0      vec4<f32>  sphere: center, radius · quad: corner Q · box: center ·
//                                   plane: normal, offset d (dot(n, p) = d)
//   offset 16  geo1      vec4<f32>  quad: edge u · box: half extents · plane: checker color,
//                                   checker frequency (w, 0 = plain)
//   offset 32  geo2      vec4<f32>  quad: edge v · box: cos, sin of the rotation about +y
//   offset 48  albedo    vec4<f32>  color (xyz), metal roughness or glass IOR (w)
//   offset 64  emission  vec4<f32>  emitted radiance (xyz)
//   offset 80  kind      vec4<u32>  shape (x), material (y)       -> Uint32 view [20], [21]
// Emissive spheres and quads are sorted to the front: indices < lightCount are the lights that
// next-event estimation samples.

const MAX_PRIMS = 64;
const MAX_BOUNCES = 8;
const MAX_IN_FLIGHT = 2;
const FULL_WIDTH = 800;
const FULL_HEIGHT = 600;
const PARAMS_SIZE = 160;
const DISPLAY_SIZE = 24;
const PRIM_SIZE = 96;
const PRIM_WORDS = PRIM_SIZE / 4;

const SHAPE_SPHERE = 0;
const SHAPE_QUAD = 1;
const SHAPE_BOX = 2;
const SHAPE_PLANE = 3;
const MAT_DIFFUSE = 0;
const MAT_METAL = 1;
const MAT_GLASS = 2;

const RESOLUTION_SCALES = [1, 0.75, 0.5];
const TARGET_SPP = [64, 128, 256, 512, 1024, 2048, 4096];
const TONE_MAPS = ['ACES', 'Reinhard'];

const traceShader = `
struct Params {
    size: vec2<u32>,
    sampleCount: u32,
    maxBounces: u32,
    primCount: u32,
    lightCount: u32,
    envType: u32,
    useNEE: u32,
    camPos: vec4<f32>,
    camForward: vec4<f32>,
    camRight: vec4<f32>,
    camUp: vec4<f32>,
    skyHorizon: vec4<f32>,
    skyZenith: vec4<f32>,
    skyGround: vec4<f32>,
    seed: u32,
}

struct Prim {
    geo0: vec4<f32>,
    geo1: vec4<f32>,
    geo2: vec4<f32>,
    albedo: vec4<f32>,
    emission: vec4<f32>,
    kind: vec4<u32>,
}

struct RayCounter {
    count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> prims: array<Prim>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> rayCounter: RayCounter;

const PI: f32 = 3.14159265;
const INF: f32 = 1e30;
const T_MIN: f32 = 1e-4;      // closest accepted hit distance
const EPS: f32 = 1e-3;        // offset of secondary ray origins along the normal
const MAX_PRIMS: u32 = ${MAX_PRIMS}u;
const MAX_BOUNCES: u32 = ${MAX_BOUNCES}u;
const RR_START: u32 = 2u;     // segment index after which Russian roulette may end the path

const SHAPE_SPHERE: u32 = ${SHAPE_SPHERE}u;
const SHAPE_QUAD: u32 = ${SHAPE_QUAD}u;
const SHAPE_BOX: u32 = ${SHAPE_BOX}u;
const SHAPE_PLANE: u32 = ${SHAPE_PLANE}u;
const MAT_DIFFUSE: u32 = ${MAT_DIFFUSE}u;
const MAT_METAL: u32 = ${MAT_METAL}u;
const MAT_GLASS: u32 = ${MAT_GLASS}u;

// ---------------------------------------------------------------- random numbers
// PCG (O'Neill 2014), RXS-M-XS output: one u32 of state per pixel, seeded from the pixel index
// and the sample (frame) index, so every frame draws fresh, decorrelated numbers.
fn pcgHash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand(state: ptr<function, u32>) -> f32 {
    *state = *state * 747796405u + 2891336453u;
    let s = *state;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f32(((word >> 22u) ^ word) >> 8u) / 16777216.0;   // [0, 1) with 24 bits
}

// ---------------------------------------------------------------- intersection
struct Hit {
    t: f32,
    index: u32,
    normal: vec3<f32>,   // geometric normal; outward for spheres and boxes, cross(u, v) for quads
}

// Sphere |o + t·d - c|² = r²: t = -b ± sqrt(b² - (|oc|² - r²)) with b = oc·d (d is unit length)
fn hitSphere(p: Prim, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let oc = ro - p.geo0.xyz;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - p.geo0.w * p.geo0.w;
    let disc = b * b - c;
    if (disc < 0.0) {
        return INF;
    }
    let s = sqrt(disc);
    var t = -b - s;
    if (t < T_MIN) {
        t = -b + s;   // origin inside the sphere (glass)
    }
    if (t < T_MIN) {
        return INF;
    }
    return t;
}

// Parallelogram Q + a·u + b·v, 0 ≤ a, b ≤ 1: intersect its plane, then solve for (a, b)
fn hitQuad(p: Prim, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let u = p.geo1.xyz;
    let v = p.geo2.xyz;
    let n = cross(u, v);
    let denom = dot(n, rd);
    if (abs(denom) < 1e-9) {
        return INF;
    }
    let t = dot(n, p.geo0.xyz - ro) / denom;
    if (t < T_MIN) {
        return INF;
    }
    let w = n / dot(n, n);
    let q = ro + rd * t - p.geo0.xyz;
    let a = dot(w, cross(q, v));
    let b = dot(w, cross(u, q));
    if (a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0) {
        return INF;
    }
    return t;
}

// Rotation about +y by the angle with cosine c and sine s (box local -> world), and its inverse
fn rotY(v: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}
fn rotYInv(v: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

// Box rotated about +y: transform the ray into box space, then the slab test. Returns t in .w
// and the outward world-space normal in .xyz.
fn hitBox(p: Prim, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let cs = p.geo2.x;
    let sn = p.geo2.y;
    let h = p.geo1.xyz;
    let o = rotYInv(ro - p.geo0.xyz, cs, sn);
    let dRaw = rotYInv(rd, cs, sn);
    // Keep 1/d finite: a zero component would give inf and then NaN (0 · inf) in the slabs
    let d = select(dRaw, vec3<f32>(1e-9), abs(dRaw) < vec3<f32>(1e-9));
    let inv = 1.0 / d;
    let t0 = (-h - o) * inv;
    let t1 = (h - o) * inv;
    let tLo = min(t0, t1);
    let tHi = max(t0, t1);
    let tNear = max(tLo.x, max(tLo.y, tLo.z));
    let tFar = min(tHi.x, min(tHi.y, tHi.z));
    if (tNear > tFar || tFar < T_MIN) {
        return vec4<f32>(0.0, 0.0, 0.0, INF);
    }
    var t = tNear;
    var nLocal: vec3<f32>;
    if (tNear >= T_MIN) {
        // Entering: the face whose slab was entered last; outward normal opposes d
        nLocal = -sign(d) * vec3<f32>(vec3<bool>(tLo.x == tNear, tLo.y == tNear, tLo.z == tNear));
    } else {
        // Origin inside (glass box): leave through the face whose slab is exited first
        t = tFar;
        nLocal = sign(d) * vec3<f32>(vec3<bool>(tHi.x == tFar, tHi.y == tFar, tHi.z == tFar));
    }
    return vec4<f32>(normalize(rotY(nLocal, cs, sn)), t);
}

fn hitPlane(p: Prim, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let denom = dot(p.geo0.xyz, rd);
    if (abs(denom) < 1e-9) {
        return INF;
    }
    let t = (p.geo0.w - dot(p.geo0.xyz, ro)) / denom;
    if (t < T_MIN) {
        return INF;
    }
    return t;
}

// Closest hit by brute force over the (at most MAX_PRIMS) primitives
fn traceScene(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var best: Hit;
    best.t = INF;
    best.index = MAX_PRIMS;
    best.normal = vec3<f32>(0.0, 1.0, 0.0);
    let count = min(params.primCount, MAX_PRIMS);
    for (var i = 0u; i < count; i++) {
        let p = prims[i];
        let shape = p.kind.x;
        var t = INF;
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if (shape == SHAPE_SPHERE) {
            t = hitSphere(p, ro, rd);
            n = (ro + rd * t - p.geo0.xyz) / p.geo0.w;
        } else if (shape == SHAPE_QUAD) {
            t = hitQuad(p, ro, rd);
            n = normalize(cross(p.geo1.xyz, p.geo2.xyz));
        } else if (shape == SHAPE_BOX) {
            let b = hitBox(p, ro, rd);
            t = b.w;
            n = b.xyz;
        } else if (shape == SHAPE_PLANE) {
            t = hitPlane(p, ro, rd);
            n = p.geo0.xyz;
        }
        if (t < best.t) {
            best.t = t;
            best.index = i;
            best.normal = n;
        }
    }
    return best;
}

// ---------------------------------------------------------------- sampling
// Orthonormal basis around a unit normal (Duff et al. 2017, branchless)
fn basisT(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}
fn basisB(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(b, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere (Malley's method): a uniform disk point lifted onto the hemisphere.
// pdf(ω) = cos θ / π, so a Lambertian bounce weighs f·cos/pdf = albedo.
fn cosineSample(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let z = sqrt(max(0.0, 1.0 - u1));
    return normalize(basisT(n) * (r * cos(phi)) + basisB(n) * (r * sin(phi)) + n * z);
}

// Uniform point in the unit ball (direction from z and φ, radius from the cube root)
fn inUnitBall(u1: f32, u2: f32, u3: f32) -> vec3<f32> {
    let z = 1.0 - 2.0 * u1;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 2.0 * PI * u2;
    return vec3<f32>(r * cos(phi), r * sin(phi), z) * pow(u3, 1.0 / 3.0);
}

struct LightSample {
    wi: vec3<f32>,
    dist: f32,
    pdf: f32,    // solid-angle pdf of wi for this light; 0 = no sample
}

// Direction toward light i from x.
//   Quad: uniform point on the area A; pdf_ω = dist² / (cos θ_light · A). Lights emit from the
//         front side (along cross(u, v)) only, so back-facing samples are rejected.
//   Sphere: uniform direction inside the cone the sphere subtends (cos θ_max = sqrt(1 - r²/d²));
//           pdf_ω = 1 / (2π (1 - cos θ_max)). From inside the sphere there is no sample.
fn sampleLight(i: u32, x: vec3<f32>, u1: f32, u2: f32) -> LightSample {
    let p = prims[i];
    var ls: LightSample;
    ls.wi = vec3<f32>(0.0, 1.0, 0.0);
    ls.dist = 0.0;
    ls.pdf = 0.0;
    if (p.kind.x == SHAPE_QUAD) {
        let u = p.geo1.xyz;
        let v = p.geo2.xyz;
        let nArea = cross(u, v);
        let area = length(nArea);
        let pt = p.geo0.xyz + u * u1 + v * u2;
        let toL = pt - x;
        let d2 = dot(toL, toL);
        if (d2 < 1e-8 || area < 1e-12) {
            return ls;
        }
        let dist = sqrt(d2);
        let wi = toL / dist;
        let cosL = -dot(nArea / area, wi);
        if (cosL <= 1e-6) {
            return ls;
        }
        ls.wi = wi;
        ls.dist = dist;
        ls.pdf = d2 / (cosL * area);
    } else if (p.kind.x == SHAPE_SPHERE) {
        let toC = p.geo0.xyz - x;
        let d2 = dot(toC, toC);
        let r = p.geo0.w;
        if (d2 <= r * r * 1.0001) {
            return ls;
        }
        let dist = sqrt(d2);
        let axis = toC / dist;
        let cosMax = sqrt(max(0.0, 1.0 - r * r / d2));
        let cosT = 1.0 - u1 * (1.0 - cosMax);
        let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
        let phi = 2.0 * PI * u2;
        ls.wi = normalize(basisT(axis) * (sinT * cos(phi)) + basisB(axis) * (sinT * sin(phi)) + axis * cosT);
        ls.dist = dist;   // only used as a hint; the shadow ray checks which primitive it hits
        ls.pdf = 1.0 / max(2.0 * PI * (1.0 - cosMax), 1e-12);
    }
    return ls;
}

// Solid-angle pdf with which sampleLight would have produced direction wi from x, given that wi
// hits light i at distance t (used for the MIS weight of BSDF-sampled light hits)
fn lightPdf(i: u32, x: vec3<f32>, wi: vec3<f32>, t: f32, nHit: vec3<f32>) -> f32 {
    let p = prims[i];
    if (p.kind.x == SHAPE_QUAD) {
        let area = length(cross(p.geo1.xyz, p.geo2.xyz));
        let cosL = -dot(nHit, wi);
        if (cosL <= 1e-6 || area < 1e-12) {
            return 0.0;
        }
        return t * t / (cosL * area);
    }
    let toC = p.geo0.xyz - x;
    let d2 = dot(toC, toC);
    let r = p.geo0.w;
    if (d2 <= r * r * 1.0001) {
        return 0.0;
    }
    let cosMax = sqrt(max(0.0, 1.0 - r * r / d2));
    return 1.0 / max(2.0 * PI * (1.0 - cosMax), 1e-12);
}

// Veach's power heuristic (β = 2) for combining light sampling and BSDF sampling
fn powerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
    let a = pdfA * pdfA;
    let b = pdfB * pdfB;
    return select(0.0, a / (a + b), a + b > 0.0);
}

fn environment(d: vec3<f32>) -> vec3<f32> {
    if (params.envType == 0u) {
        return vec3<f32>(0.0);
    }
    let intensity = params.skyHorizon.w;
    if (params.envType == 2u) {
        return params.skyHorizon.xyz * intensity;   // uniform (furnace test)
    }
    if (d.y >= 0.0) {
        return mix(params.skyHorizon.xyz, params.skyZenith.xyz, sqrt(d.y)) * intensity;
    }
    return mix(params.skyHorizon.xyz, params.skyGround.xyz, sqrt(min(-d.y * 4.0, 1.0))) * intensity;
}

fn maxComponent(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

// Schlick's approximation of the Fresnel reflectance, cosine taken on the outside (n = 1) side
fn schlick(cosine: f32, ior: f32) -> f32 {
    var r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

struct PathResult {
    radiance: vec3<f32>,
    rays: u32,
}

// One path from the camera. Segment k is the ray leaving scattering event k (k = 0: camera ray).
// At each hit: add emission (MIS-weighted if the previous bounce was diffuse and NEE is on), stop
// once maxBounces scattering events have happened, otherwise scatter:
//   diffuse - next-event estimation: one shadow ray to a uniformly chosen light, weighted by the
//             power heuristic against the cosine pdf; then a cosine-weighted bounce
//   metal   - mirror reflection perturbed by roughness × a random point in the unit ball
//   glass   - Fresnel (Schlick) chooses reflection or Snell refraction; total internal reflection
//             always reflects. Reflect + refract probabilities are F and 1 - F, so no weight.
// Light hits after metal/glass bounces count fully: those paths cannot be produced by NEE
// (shadow rays stop at glass and mirrors), so nothing is counted twice.
fn tracePath(pixel: vec2<u32>, rng: ptr<function, u32>) -> PathResult {
    let size = vec2<f32>(params.size);
    let jitter = vec2<f32>(rand(rng), rand(rng));
    let ndc = (vec2<f32>(pixel) + jitter) / size * 2.0 - 1.0;
    let tanHalf = params.camPos.w;
    let sx = ndc.x * (size.x / size.y) * tanHalf;
    let sy = -ndc.y * tanHalf;   // pixel rows grow downward
    let forward = params.camForward.xyz;
    let right = params.camRight.xyz;
    let up = params.camUp.xyz;
    let dirPinhole = forward + right * sx + up * sy;

    var ro = params.camPos.xyz;
    var rd = normalize(dirPinhole);
    let lensRadius = params.camForward.w;
    if (lensRadius > 0.0) {
        // Thin lens: all rays through a pixel meet on the focal plane at focusDist along forward
        let focusPoint = ro + dirPinhole * params.camRight.w;
        let r = lensRadius * sqrt(rand(rng));
        let phi = 2.0 * PI * rand(rng);
        ro = ro + (right * cos(phi) + up * sin(phi)) * r;
        rd = normalize(focusPoint - ro);
    }

    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var rays = 0u;
    var prevSpecular = true;    // camera rays behave like specular bounces (no MIS)
    var prevPdf = 0.0;          // cosine pdf of the last diffuse bounce
    var prevPos = ro;
    let bounces = clamp(params.maxBounces, 1u, MAX_BOUNCES);
    let lightCount = min(params.lightCount, min(params.primCount, MAX_PRIMS));
    let useNEE = params.useNEE != 0u && lightCount > 0u;

    for (var seg = 0u; seg <= MAX_BOUNCES; seg++) {
        let hit = traceScene(ro, rd);
        rays++;
        if (hit.t >= INF) {
            radiance += throughput * environment(rd);
            break;
        }
        let p = prims[hit.index];
        let pos = ro + rd * hit.t;
        let frontFace = dot(hit.normal, rd) < 0.0;

        // Emission (front side only)
        if (frontFace && maxComponent(p.emission.xyz) > 0.0) {
            var w = 1.0;
            if (!prevSpecular && useNEE && hit.index < lightCount) {
                let pl = lightPdf(hit.index, prevPos, rd, hit.t, hit.normal) / f32(lightCount);
                w = powerHeuristic(prevPdf, pl);
            }
            radiance += throughput * p.emission.xyz * w;
        }
        if (seg >= bounces) {
            break;
        }

        let nf = select(-hit.normal, hit.normal, frontFace);   // faces the incoming ray
        let material = p.kind.y;
        var albedo = p.albedo.xyz;
        if (p.kind.x == SHAPE_PLANE && p.geo1.w > 0.0) {
            // Checkerboard in the plane's x/z coordinates (used for horizontal ground planes)
            let c = floor(pos.x * p.geo1.w) + floor(pos.z * p.geo1.w);
            if (c - 2.0 * floor(c * 0.5) > 0.5) {
                albedo = p.geo1.xyz;
            }
        }

        if (material == MAT_METAL) {
            let roughness = p.albedo.w;
            let dir = reflect(rd, nf) + inUnitBall(rand(rng), rand(rng), rand(rng)) * roughness;
            let len = length(dir);
            if (len < 1e-6 || dot(dir, nf) <= 0.0) {
                break;   // perturbed below the surface: absorbed
            }
            rd = dir / len;
            ro = pos + nf * EPS;
            throughput *= albedo;
            prevSpecular = true;
        } else if (material == MAT_GLASS) {
            let ior = max(p.albedo.w, 1.0001);
            let eta = select(ior, 1.0 / ior, frontFace);   // n_from / n_to
            let cosI = min(dot(-rd, nf), 1.0);
            let sin2T = eta * eta * (1.0 - cosI * cosI);
            var reflectIt = sin2T >= 1.0;                  // total internal reflection
            var cosT = 0.0;
            if (!reflectIt) {
                cosT = sqrt(max(0.0, 1.0 - sin2T));
                let fresnel = schlick(select(cosT, cosI, frontFace), ior);
                reflectIt = rand(rng) < fresnel;
            }
            if (reflectIt) {
                rd = reflect(rd, nf);
                ro = pos + nf * EPS;
            } else {
                rd = normalize(rd * eta + nf * (eta * cosI - cosT));
                ro = pos - nf * EPS;
            }
            throughput *= albedo;   // tint (1 for clear glass)
            prevSpecular = true;
        } else {
            // Diffuse: next-event estimation toward one light
            if (useNEE) {
                let li = min(u32(rand(rng) * f32(lightCount)), lightCount - 1u);
                let ls = sampleLight(li, pos, rand(rng), rand(rng));
                let cosS = dot(nf, ls.wi);
                if (ls.pdf > 0.0 && cosS > 0.0) {
                    let shadow = traceScene(pos + nf * EPS, ls.wi);
                    rays++;
                    if (shadow.index == li && dot(shadow.normal, ls.wi) < 0.0) {
                        let pdfL = ls.pdf / f32(lightCount);
                        let w = powerHeuristic(pdfL, cosS / PI);
                        radiance += throughput * albedo * (cosS / PI) * prims[li].emission.xyz * (w / pdfL);
                    }
                }
            }
            rd = cosineSample(nf, rand(rng), rand(rng));
            ro = pos + nf * EPS;
            prevPdf = max(dot(nf, rd), 0.0) / PI;
            throughput *= albedo;
            prevSpecular = false;
        }
        prevPos = pos;

        // Russian roulette after the 3rd bounce: continue with probability q, divide by q
        if (seg >= RR_START) {
            let q = clamp(maxComponent(throughput), 0.05, 0.95);
            if (rand(rng) >= q) {
                break;
            }
            throughput /= q;
        }
        if (maxComponent(throughput) <= 0.0) {
            break;
        }
    }

    var result: PathResult;
    result.radiance = radiance;
    result.rays = rays;
    return result;
}

var<workgroup> workgroupRays: atomic<u32>;

@compute @workgroup_size(8, 8)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_index) localIndex: u32) {
    var rays = 0u;
    if (gid.x < params.size.x && gid.y < params.size.y) {
        let idx = gid.y * params.size.x + gid.x;
        var rng = pcgHash(idx ^ pcgHash(params.sampleCount * 9781u + params.seed));
        let path = tracePath(gid.xy, &rng);
        rays = path.rays;

        var c = path.radiance;
        // Drop NaN/Inf samples instead of letting them poison the running sum forever
        if (any(c != c) || any(abs(c) > vec3<f32>(1e20))) {
            c = vec3<f32>(0.0);
        }
        c = max(c, vec3<f32>(0.0));
        // Optional firefly clamp on the sample's max component (biased, but converges faster)
        let clampMax = params.camUp.w;
        let peak = maxComponent(c);
        if (clampMax > 0.0 && peak > clampMax) {
            c *= clampMax / peak;
        }

        var sum = vec4<f32>(0.0);
        if (params.sampleCount > 0u) {
            sum = accum[idx];
        }
        accum[idx] = sum + vec4<f32>(c, 1.0);
    }
    // Ray statistics: sum per workgroup, then one global atomic per workgroup
    atomicAdd(&workgroupRays, rays);
    workgroupBarrier();
    if (localIndex == 0u) {
        atomicAdd(&rayCounter.count, atomicLoad(&workgroupRays));
    }
}
`;

const displayShader = `
struct Display {
    renderSize: vec2<u32>,
    canvasSize: vec2<u32>,
    toneMap: u32,
    exposure: f32,
}

@group(0) @binding(0) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> display: Display;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

// One triangle that covers the whole screen
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(corners[vi], 0.0, 1.0);
    return out;
}

// ACES filmic curve, Narkowicz 2015 fit
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// sRGB transfer function (linear -> display)
fn toSRGB(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    // The accumulator may be smaller than the canvas (resolution scale): nearest-pixel upscale
    let scale = vec2<f32>(display.renderSize) / vec2<f32>(display.canvasSize);
    let pixel = min(vec2<u32>(frag.position.xy * scale), display.renderSize - vec2<u32>(1u));
    let sum = accum[pixel.y * display.renderSize.x + pixel.x];
    if (sum.w <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let hdr = max(sum.xyz / sum.w, vec3<f32>(0.0)) * display.exposure;
    var mapped: vec3<f32>;
    if (display.toneMap == 1u) {
        mapped = hdr / (1.0 + hdr);   // Reinhard
    } else {
        mapped = aces(hdr);
    }
    return vec4<f32>(toSRGB(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
`;

// ---------------------------------------------------------------- scenes
// Materials: { type, albedo: [r,g,b], param (roughness or IOR), emission: [r,g,b] }
const diffuse = (albedo) => ({ type: MAT_DIFFUSE, albedo, param: 0, emission: [0, 0, 0] });
const metal = (albedo, roughness) => ({ type: MAT_METAL, albedo, param: roughness, emission: [0, 0, 0] });
const glass = (ior, tint = [1, 1, 1]) => ({ type: MAT_GLASS, albedo: tint, param: ior, emission: [0, 0, 0] });
const light = (emission) => ({ type: MAT_DIFFUSE, albedo: [0, 0, 0], param: 0, emission });

// Deterministic generator for scene layout (not used per pixel)
function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

class SceneBuilder {
    constructor() {
        this.prims = [];
    }
    sphere(center, radius, mat) {
        this.prims.push({ shape: SHAPE_SPHERE, g0: [...center, radius], g1: [0, 0, 0, 0], g2: [0, 0, 0, 0], mat });
        return this;
    }
    quad(corner, u, v, mat) {
        this.prims.push({ shape: SHAPE_QUAD, g0: [...corner, 0], g1: [...u, 0], g2: [...v, 0], mat });
        return this;
    }
    // Box with half extents `half`, rotated by `angleDeg` about +y around its center
    box(center, half, angleDeg, mat) {
        const a = angleDeg * Math.PI / 180;
        this.prims.push({ shape: SHAPE_BOX, g0: [...center, 0], g1: [...half, 0], g2: [Math.cos(a), Math.sin(a), 0, 0], mat });
        return this;
    }
    // Plane dot(normal, p) = offset; optional checker of a second color (frequency in cells/unit)
    plane(normal, offset, mat, checkerColor = null, checkerFreq = 0) {
        this.prims.push({ shape: SHAPE_PLANE, g0: [...normal, offset], g1: [...(checkerColor || [0, 0, 0]), checkerColor ? checkerFreq : 0], g2: [0, 0, 0, 0], mat });
        return this;
    }

    // Packs the scene buffer; see the Prim layout table at the top of the file
    build() {
        if (this.prims.length > MAX_PRIMS) throw new Error(`Scene has ${this.prims.length} primitives (max ${MAX_PRIMS})`);
        const isLight = (p) => (p.shape === SHAPE_SPHERE || p.shape === SHAPE_QUAD) && Math.max(...p.mat.emission) > 0;
        const sorted = [...this.prims.filter(isLight), ...this.prims.filter(p => !isLight(p))];
        const data = new ArrayBuffer(Math.max(sorted.length, 1) * PRIM_SIZE);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        sorted.forEach((p, i) => {
            const o = i * PRIM_WORDS;
            f32.set(p.g0, o);
            f32.set(p.g1, o + 4);
            f32.set(p.g2, o + 8);
            f32.set([...p.mat.albedo, p.mat.param], o + 12);
            f32.set([...p.mat.emission, 0], o + 16);
            u32[o + 20] = p.shape;
            u32[o + 21] = p.mat.type;
            u32[o + 22] = 0;
            u32[o + 23] = 0;
        });
        return { data, f32, u32, count: sorted.length, lightCount: sorted.filter(isLight).length };
    }
}

const SCENES = {
    cornell: {
        name: 'Cornell box',
        build() {
            const white = diffuse([0.73, 0.73, 0.73]);
            const s = new SceneBuilder()
                .quad([-1, 0, -1], [2, 0, 0], [0, 0, 2], white)            // floor
                .quad([-1, 2, -1], [2, 0, 0], [0, 0, 2], white)            // ceiling
                .quad([-1, 0, -1], [2, 0, 0], [0, 2, 0], white)            // back wall
                .quad([-1, 0, -1], [0, 0, 2], [0, 2, 0], diffuse([0.63, 0.06, 0.05]))  // left, red
                .quad([1, 0, -1], [0, 0, 2], [0, 2, 0], diffuse([0.14, 0.45, 0.09]))   // right, green
                // Ceiling light: cross(u, v) = -y, so it emits downward
                .quad([-0.25, 1.995, -0.2], [0.5, 0, 0], [0, 0, 0.4], light([17, 12, 4]))
                .box([0.42, 0.6, -0.45], [0.27, 0.6, 0.27], 18, white)       // tall box
                .sphere([-0.45, 0.33, -0.3], 0.33, metal([0.95, 0.95, 0.95], 0.02))
                .sphere([0.35, 0.3, 0.42], 0.3, glass(1.5));
            return s.build();
        },
        camera: { target: [0, 1, 0], yaw: 0, pitch: 0, distance: 3.9, fov: 40, focus: 3.5, aperture: 0.03 },
        env: { type: 0 },
        bounces: 5,
    },
    spheres: {
        name: 'Spheres',
        build() {
            const s = new SceneBuilder()
                .plane([0, 1, 0], 0, diffuse([0.5, 0.5, 0.5]), [0.25, 0.27, 0.3], 1)
                .sphere([0, 1, 0], 1, glass(1.5))
                .sphere([-4, 1, 0], 1, diffuse([0.4, 0.2, 0.1]))
                .sphere([4, 1, 0], 1, metal([0.7, 0.6, 0.5], 0));
            const rnd = mulberry32(42);
            const big = [[0, 0], [-4, 0], [4, 0]];
            for (let a = -6; a < 6 && s.prims.length < MAX_PRIMS; a++) {
                for (let b = -3; b < 3 && s.prims.length < MAX_PRIMS; b++) {
                    const x = a + 0.8 * rnd();
                    const z = b + 0.8 * rnd();
                    const pick = rnd();
                    if (big.some(([bx, bz]) => Math.hypot(x - bx, z - bz) < 1.25)) continue;
                    const r = 0.2;
                    if (pick < 0.62) {
                        s.sphere([x, r, z], r, diffuse([rnd() * rnd(), rnd() * rnd(), rnd() * rnd()]));
                    } else if (pick < 0.85) {
                        s.sphere([x, r, z], r, metal([0.5 + 0.5 * rnd(), 0.5 + 0.5 * rnd(), 0.5 + 0.5 * rnd()], 0.4 * rnd()));
                    } else {
                        s.sphere([x, r, z], r, glass(1.5));
                    }
                }
            }
            return s.build();
        },
        camera: { target: [0, 0.6, 0], yaw: 1.36, pitch: 0.16, distance: 13, fov: 24, focus: 12.5, aperture: 0.08 },
        env: { type: 1, horizon: [1, 1, 1], zenith: [0.45, 0.65, 1.0], ground: [0.35, 0.33, 0.3], intensity: 1 },
        bounces: 6,
    },
    caustics: {
        name: 'Caustics',
        build() {
            return new SceneBuilder()
                .plane([0, 1, 0], 0, diffuse([0.8, 0.8, 0.8]))
                .plane([0, 0, 1], -3, diffuse([0.5, 0.45, 0.4]))           // back wall z = -3
                .sphere([0, 1.5, 0], 0.8, glass(1.5))
                .sphere([-1.0, 5.2, -0.3], 0.3, light([160, 150, 130]))   // small light, up and left
                .sphere([1.6, 0.35, 0.9], 0.35, glass(1.5, [0.6, 0.85, 1.0]))
                .build();
        },
        camera: { target: [0.4, 0.6, 0], yaw: 0.25, pitch: 0.55, distance: 6.5, fov: 40, focus: 6.3, aperture: 0.04 },
        env: { type: 0 },
        bounces: 8,
    },
    materials: {
        name: 'Materials',
        build() {
            const s = new SceneBuilder()
                .plane([0, 1, 0], 0, diffuse([0.7, 0.7, 0.7]), [0.15, 0.15, 0.17], 2)
                // Soft area light above the row (cross(u, v) = -y: emits downward)
                .quad([-2.5, 3.2, -1.2], [5, 0, 0], [0, 0, 1.2], light([4, 3.9, 3.7]))
                // Small warm sphere light in front on the right, sampled by its subtended cone
                .sphere([2.0, 0.3, 1.4], 0.15, light([60, 30, 10]));
            const mats = [
                diffuse([0.8, 0.25, 0.2]),
                metal([0.95, 0.95, 0.95], 0.0),
                metal([1.0, 0.78, 0.34], 0.25),
                glass(1.5),
                glass(2.4, [0.9, 1.0, 0.95]),
            ];
            mats.forEach((m, i) => {
                const x = (i - 2) * 1.3;
                s.box([x, 0.2, 0], [0.4, 0.2, 0.4], 25 + 10 * i, diffuse([0.3, 0.32, 0.36]));  // pedestal
                s.sphere([x, 0.9, 0], 0.5, m);
            });
            return s.build();
        },
        camera: { target: [0, 0.8, 0], yaw: 0, pitch: 0.2, distance: 6.5, fov: 45, focus: 6.5, aperture: 0.05 },
        env: { type: 1, horizon: [0.35, 0.38, 0.45], zenith: [0.1, 0.12, 0.2], ground: [0.1, 0.1, 0.1], intensity: 0.5 },
        bounces: 6,
    },
};

// ---------------------------------------------------------------- CPU intersection (click to focus)
// Mirrors traceScene() in the shader on the same packed data; one ray, no allocation-heavy work.
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function intersectScene(scene, ro, rd) {
    const T_MIN = 1e-4;
    const f = scene.f32;
    let best = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < Math.min(scene.count, MAX_PRIMS); i++) {
        const o = i * PRIM_WORDS;
        const g0 = [f[o], f[o + 1], f[o + 2], f[o + 3]];
        const g1 = [f[o + 4], f[o + 5], f[o + 6], f[o + 7]];
        const g2 = [f[o + 8], f[o + 9], f[o + 10], f[o + 11]];
        const shape = scene.u32[o + 20];
        let t = Infinity;
        if (shape === SHAPE_SPHERE) {
            const oc = sub3(ro, g0);
            const b = dot3(oc, rd);
            const disc = b * b - (dot3(oc, oc) - g0[3] * g0[3]);
            if (disc >= 0) {
                const s = Math.sqrt(disc);
                t = -b - s >= T_MIN ? -b - s : (-b + s >= T_MIN ? -b + s : Infinity);
            }
        } else if (shape === SHAPE_QUAD) {
            const n = cross3(g1, g2);
            const denom = dot3(n, rd);
            if (Math.abs(denom) >= 1e-9) {
                const tq = dot3(n, sub3(g0, ro)) / denom;
                if (tq >= T_MIN) {
                    const nn = dot3(n, n);
                    const q = sub3([ro[0] + rd[0] * tq, ro[1] + rd[1] * tq, ro[2] + rd[2] * tq], g0);
                    const a = dot3(n, cross3(q, g2)) / nn;
                    const b = dot3(n, cross3(g1, q)) / nn;
                    if (a >= 0 && a <= 1 && b >= 0 && b <= 1) t = tq;
                }
            }
        } else if (shape === SHAPE_BOX) {
            const [c, s] = [g2[0], g2[1]];
            const inv = (v) => [c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]];
            const oL = inv(sub3(ro, g0));
            const dL = inv(rd).map(x => Math.abs(x) < 1e-9 ? 1e-9 : x);
            let tNear = -Infinity;
            let tFar = Infinity;
            for (let k = 0; k < 3; k++) {
                const t0 = (-g1[k] - oL[k]) / dL[k];
                const t1 = (g1[k] - oL[k]) / dL[k];
                tNear = Math.max(tNear, Math.min(t0, t1));
                tFar = Math.min(tFar, Math.max(t0, t1));
            }
            if (tNear <= tFar && tFar >= T_MIN) t = tNear >= T_MIN ? tNear : tFar;
        } else if (shape === SHAPE_PLANE) {
            const denom = dot3(g0, rd);
            if (Math.abs(denom) >= 1e-9) {
                const tp = (g0[3] - dot3(g0, ro)) / denom;
                if (tp >= T_MIN) t = tp;
            }
        }
        if (t < best) {
            best = t;
            bestIndex = i;
        }
    }
    return { t: best, index: bestIndex };
}

// Orbit camera basis: eye on a sphere around the target, built directly (no matrices)
function cameraBasis(cam) {
    const cp = Math.cos(cam.pitch);
    const eye = [
        cam.target[0] + cam.distance * cp * Math.sin(cam.yaw),
        cam.target[1] + cam.distance * Math.sin(cam.pitch),
        cam.target[2] + cam.distance * cp * Math.cos(cam.yaw),
    ];
    const forward = norm3(sub3(cam.target, eye));
    const right = norm3(cross3(forward, [0, 1, 0]));
    const up = cross3(right, forward);
    return { eye, forward, right, up };
}

class PathTracer {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;

        this.sceneKey = 'cornell';
        this.scene = null;
        this.env = { type: 0 };
        this.camera = { target: [0, 1, 0], yaw: 0, pitch: 0, distance: 4, fov: 40, focus: 3.5, aperture: 0.03 };
        this.dofOn = false;
        this.maxBounces = 5;
        this.useNEE = true;
        this.clampMax = 0;           // 0 = off
        this.exposureEV = 0;
        this.toneMap = 0;
        this.scaleIndex = 0;
        this.targetIndex = 4;        // 1024 spp
        this.seedValue = (Math.random() * 0xffffffff) >>> 0;   // once per session, not per pixel
        this.resizeRender();

        this.sampleCount = 0;
        this.inFlight = 0;
        this.renderPending = false;
        this.pendingReset = false;
        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;
        this.raysLastFrame = 0;      // from the ray counter readback
        this.readbackBusy = false;
        this.readbackWanted = false;
        this.accumStart = performance.now();
        this.accumTime = 0;

        this.params = new ArrayBuffer(PARAMS_SIZE);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);
        this.displayData = new ArrayBuffer(DISPLAY_SIZE);
        this.displayF32 = new Float32Array(this.displayData);
        this.displayU32 = new Uint32Array(this.displayData);

        this.setupBuffers();
        this.setupPipelines();
        this.setupCanvas();
        this.setupMouseControls();
    }

    get targetSpp() {
        return TARGET_SPP[this.targetIndex];
    }

    get converged() {
        return this.sampleCount >= this.targetSpp;
    }

    resizeRender() {
        const s = RESOLUTION_SCALES[this.scaleIndex];
        this.renderWidth = Math.round(FULL_WIDTH * s);
        this.renderHeight = Math.round(FULL_HEIGHT * s);
    }

    setupBuffers() {
        this.accumBuffer = this.device.createBuffer({
            size: FULL_WIDTH * FULL_HEIGHT * 16,   // vec4<f32> per pixel at the largest resolution
            usage: GPUBufferUsage.STORAGE
        });
        this.primBuffer = this.device.createBuffer({
            size: MAX_PRIMS * PRIM_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.paramsBuffer = this.device.createBuffer({
            size: PARAMS_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.displayBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.counterBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.readbackBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    setupPipelines() {
        const traceModule = this.device.createShaderModule({ code: traceShader });
        const traceLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
            ]
        });
        this.tracePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [traceLayout] }),
            compute: { module: traceModule, entryPoint: 'cs_trace' }
        });
        this.traceBindGroup = this.device.createBindGroup({
            layout: traceLayout,
            entries: [
                { binding: 0, resource: { buffer: this.accumBuffer } },
                { binding: 1, resource: { buffer: this.primBuffer } },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
                { binding: 3, resource: { buffer: this.counterBuffer } }
            ]
        });

        const displayModule = this.device.createShaderModule({ code: displayShader });
        const displayLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });
        this.displayPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [displayLayout] }),
            vertex: { module: displayModule, entryPoint: 'vs_main' },
            fragment: {
                module: displayModule,
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
            primitive: { topology: 'triangle-list' }
        });
        this.displayBindGroup = this.device.createBindGroup({
            layout: displayLayout,
            entries: [
                { binding: 0, resource: { buffer: this.accumBuffer } },
                { binding: 1, resource: { buffer: this.displayBuffer } }
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
        // The canvas is CSS-scaled, so convert client coordinates to canvas pixels
        const toPixels = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            return [
                (e.clientX - rect.left) * this.canvas.width / rect.width,
                (e.clientY - rect.top) * this.canvas.height / rect.height
            ];
        };

        this.dragMode = null;   // 'orbit' | 'pan'
        this.lastPointer = [0, 0];
        this.downPointer = [0, 0];
        this.dragDistance = 0;

        this.onMouseDown = (e) => {
            if (e.button !== 0 && e.button !== 2) return;
            e.preventDefault();
            this.dragMode = e.button === 2 ? 'pan' : 'orbit';
            this.lastPointer = toPixels(e);
            this.downPointer = this.lastPointer.slice();
            this.dragDistance = 0;
        };
        this.onMouseMove = (e) => {
            if (!this.dragMode) return;
            const p = toPixels(e);
            const dx = p[0] - this.lastPointer[0];
            const dy = p[1] - this.lastPointer[1];
            this.lastPointer = p;
            this.dragDistance = Math.max(this.dragDistance, Math.hypot(p[0] - this.downPointer[0], p[1] - this.downPointer[1]));
            if (this.dragDistance < 3) return;   // still a click
            const cam = this.camera;
            if (this.dragMode === 'orbit') {
                cam.yaw -= dx * 0.006;
                cam.pitch = Math.min(Math.max(cam.pitch + dy * 0.006, -1.45), 1.45);
            } else {
                // Pan in the view plane, scaled so the target tracks the cursor
                const { right, up } = cameraBasis(cam);
                const k = 2 * cam.distance * Math.tan(cam.fov * Math.PI / 360) / this.canvas.height;
                for (let i = 0; i < 3; i++) cam.target[i] += (-dx * right[i] + dy * up[i]) * k;
            }
            this.changed();
        };
        this.onMouseUp = (e) => {
            if (!this.dragMode) return;
            const wasClick = this.dragMode === 'orbit' && this.dragDistance < 3;
            this.dragMode = null;
            if (wasClick && e.target === this.canvas) this.focusAt(...toPixels(e));
        };
        this.onWheel = (e) => {
            e.preventDefault();
            this.camera.distance = Math.min(Math.max(this.camera.distance * Math.exp(e.deltaY * 0.001), 0.5), 40);
            this.changed();
        };
        this.onContextMenu = (e) => e.preventDefault();

        this.canvas.addEventListener('mousedown', this.onMouseDown);
        window.addEventListener('mousemove', this.onMouseMove);
        window.addEventListener('mouseup', this.onMouseUp);
        this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
        this.canvas.addEventListener('contextmenu', this.onContextMenu);
    }

    // Click to focus: one pinhole ray through the pixel center, intersected on the CPU with the
    // same packed scene data the GPU uses (no readback). Focus distance is measured along the
    // view axis, matching the focal plane in the shader.
    focusAt(px, py) {
        const { eye, forward, right, up } = cameraBasis(this.camera);
        const tanHalf = Math.tan(this.camera.fov * Math.PI / 360);
        const aspect = this.canvas.width / this.canvas.height;
        const sx = (px / this.canvas.width * 2 - 1) * aspect * tanHalf;
        const sy = -(py / this.canvas.height * 2 - 1) * tanHalf;
        const rd = norm3([0, 1, 2].map(i => forward[i] + right[i] * sx + up[i] * sy));
        const hit = intersectScene(this.scene, eye, rd);
        if (!Number.isFinite(hit.t)) return;
        this.camera.focus = Math.min(Math.max(hit.t * dot3(rd, forward), 0.1), 100);
        if (this.onFocusChange) this.onFocusChange();
        if (this.dofOn) this.changed();
    }

    loadScene(key) {
        const def = SCENES[key];
        this.sceneKey = key;
        this.scene = def.build();
        this.env = def.env;
        this.camera = { ...def.camera, target: def.camera.target.slice() };
        this.maxBounces = def.bounces;
        this.device.queue.writeBuffer(this.primBuffer, 0, this.scene.data);
        this.changed();
    }

    // Anything that changes the image: clear the accumulator and draw one fresh sample
    changed() {
        this.resetAccumulation();
        this.requestFrame();
    }

    resetAccumulation() {
        this.sampleCount = 0;   // the next dispatch overwrites instead of adding
        this.accumStart = performance.now();
        this.accumTime = 0;
    }

    // While paused, coalesce changes into one sample + display per animation frame
    requestFrame() {
        if (this.running || this.renderPending || this.destroyed) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            if (this.running || this.destroyed) return;
            if (this.converged) this.present(); else this.frame();
            if (this.onStats) this.onStats();
        });
    }

    // Writes the compute uniforms; see the Params layout table at the top of the file
    writeParams() {
        const { eye, forward, right, up } = cameraBasis(this.camera);
        const env = this.env;
        const u = this.paramsU32;
        const f = this.paramsF32;
        u[0] = this.renderWidth;
        u[1] = this.renderHeight;
        u[2] = this.sampleCount;
        u[3] = Math.min(Math.max(this.maxBounces, 1), MAX_BOUNCES);
        u[4] = this.scene.count;
        u[5] = this.scene.lightCount;
        u[6] = env.type;
        u[7] = this.useNEE ? 1 : 0;
        f.set([...eye, Math.tan(this.camera.fov * Math.PI / 360)], 8);
        f.set([...forward, this.dofOn ? this.camera.aperture : 0], 12);
        f.set([...right, this.camera.focus], 16);
        f.set([...up, this.clampMax], 20);
        f.set([...(env.horizon || [0, 0, 0]), env.intensity ?? 1], 24);
        f.set([...(env.zenith || [0, 0, 0]), 0], 28);
        f.set([...(env.ground || [0, 0, 0]), 0], 32);
        u[36] = this.seedValue;
        u[37] = 0;
        u[38] = 0;
        u[39] = 0;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params);
    }

    // Writes the display uniforms; see the Display layout table at the top of the file
    writeDisplay() {
        this.displayU32[0] = this.renderWidth;
        this.displayU32[1] = this.renderHeight;
        this.displayU32[2] = this.canvas.width;
        this.displayU32[3] = this.canvas.height;
        this.displayU32[4] = this.toneMap;
        this.displayF32[5] = Math.pow(2, this.exposureEV);
        this.device.queue.writeBuffer(this.displayBuffer, 0, this.displayData);
    }

    encodeDisplay(commandEncoder) {
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });
        renderPass.setPipeline(this.displayPipeline);
        renderPass.setBindGroup(0, this.displayBindGroup);
        renderPass.draw(3);
        renderPass.end();
    }

    // Display only (exposure / tone map changes keep the accumulated samples)
    present() {
        this.writeDisplay();
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeDisplay(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    requestPresent() {
        if (this.running || this.destroyed) return;   // the running loop presents every frame
        requestAnimationFrame(() => {
            if (!this.running && !this.destroyed) this.present();
        });
    }

    // One sample per pixel (one dispatch), then display
    frame() {
        if (this.converged) return;
        this.writeParams();
        this.writeDisplay();

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.clearBuffer(this.counterBuffer);
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.tracePipeline);
        computePass.setBindGroup(0, this.traceBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.renderWidth / 8), Math.ceil(this.renderHeight / 8));
        computePass.end();

        // 4-byte ray-count readback, at most one outstanding, only when stats are due
        const readback = this.readbackWanted && !this.readbackBusy;
        if (readback) commandEncoder.copyBufferToBuffer(this.counterBuffer, 0, this.readbackBuffer, 0, 4);

        this.encodeDisplay(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        this.sampleCount++;
        this.accumTime = performance.now() - this.accumStart;

        this.inFlight++;
        this.device.queue.onSubmittedWorkDone().then(() => { this.inFlight--; });

        if (readback) {
            this.readbackBusy = true;
            this.readbackWanted = false;
            this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
                this.raysLastFrame = new Uint32Array(this.readbackBuffer.getMappedRange())[0];
                this.readbackBuffer.unmap();
                this.readbackBusy = false;
            }).catch(() => { this.readbackBusy = false; });
        }
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.readbackWanted = true;
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
    }

    animate() {
        if (!this.running || this.destroyed) return;

        if (this.converged) {
            // Target reached: stop so the GPU idles
            this.running = false;
            this.fps = 0;
            if (this.onStateChange) this.onStateChange();
            if (this.onStats) this.onStats();
            return;
        }

        // Never queue more than MAX_IN_FLIGHT frames of path tracing on the GPU
        if (this.inFlight < MAX_IN_FLIGHT) {
            this.frame();
            this.framesSinceSample++;
        }

        const now = performance.now();
        if (now - this.lastSampleTime >= 500) {
            this.fps = this.framesSinceSample * 1000 / (now - this.lastSampleTime);
            this.framesSinceSample = 0;
            this.lastSampleTime = now;
            this.readbackWanted = true;
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
        this.accumBuffer.destroy();
        this.primBuffer.destroy();
        this.paramsBuffer.destroy();
        this.displayBuffer.destroy();
        this.counterBuffer.destroy();
        this.readbackBuffer.destroy();
    }
}

export async function initPathTracer(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-pt');
    const output = document.getElementById('output-pt');

    const tracer = new PathTracer(device, canvas);
    tracer.onStats = () => updateOutput();

    const startButton = document.getElementById('btn-pt-start');
    const sceneButtons = document.querySelectorAll('.pt-scene');
    const resolutionSelect = document.getElementById('pt-resolution');
    const toneSelect = document.getElementById('pt-tonemap');
    const neeBox = document.getElementById('pt-nee');
    const dofBox = document.getElementById('pt-dof');

    function showState() {
        startButton.textContent = tracer.running ? '⏸ Pause' : (tracer.converged ? '✓ Converged' : '▶ Start');
    }
    tracer.onStateChange = () => showState();

    startButton.addEventListener('click', () => {
        if (tracer.running) tracer.pause(); else tracer.start();
        showState();
        updateOutput();
    });

    document.getElementById('btn-pt-reset').addEventListener('click', () => {
        tracer.changed();
        showState();
        updateOutput();
    });

    // Sliders: id suffix, getter/setter, clamp range, display format, resets accumulation?
    const sliders = [
        { id: 'bounces',  get: () => tracer.maxBounces, set: v => { tracer.maxBounces = v; }, parse: v => parseInt(v, 10),
          min: 1, max: MAX_BOUNCES, show: v => String(v), resets: true },
        { id: 'target',   get: () => tracer.targetIndex, set: v => { tracer.targetIndex = v; }, parse: v => parseInt(v, 10),
          min: 0, max: TARGET_SPP.length - 1, show: v => String(TARGET_SPP[v]), resets: false },
        { id: 'exposure', get: () => tracer.exposureEV, set: v => { tracer.exposureEV = v; }, parse: parseFloat,
          min: -3, max: 3, show: v => (v > 0 ? '+' : '') + v.toFixed(1) + ' EV', resets: false },
        { id: 'clamp',    get: () => tracer.clampMax, set: v => { tracer.clampMax = v; }, parse: parseFloat,
          min: 0, max: 100, show: v => v === 0 ? 'off' : String(v), resets: true },
        { id: 'fov',      get: () => tracer.camera.fov, set: v => { tracer.camera.fov = v; }, parse: parseFloat,
          min: 15, max: 90, show: v => v.toFixed(0) + '°', resets: true },
        { id: 'aperture', get: () => tracer.camera.aperture, set: v => { tracer.camera.aperture = v; }, parse: parseFloat,
          min: 0.005, max: 0.3, show: v => v.toFixed(3), resets: () => tracer.dofOn },
        { id: 'focus',    get: () => tracer.camera.focus, set: v => { tracer.camera.focus = v; }, parse: parseFloat,
          min: 0.1, max: 30, show: v => v.toFixed(2), resets: () => tracer.dofOn },
    ].map(s => ({
        ...s,
        input: document.getElementById(`pt-${s.id}`),
        label: document.getElementById(`pt-${s.id}-value`)
    }));

    function showParams() {
        for (const s of sliders) {
            s.input.value = s.get();
            s.label.textContent = s.show(s.get());
        }
        resolutionSelect.value = String(tracer.scaleIndex);
        toneSelect.value = String(tracer.toneMap);
        neeBox.checked = tracer.useNEE;
        dofBox.checked = tracer.dofOn;
        sceneButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.scene === tracer.sceneKey));
    }
    tracer.onFocusChange = () => showParams();

    for (const s of sliders) {
        s.input.addEventListener('input', (e) => {
            const v = s.parse(e.target.value);
            if (Number.isNaN(v)) return;
            s.set(Math.min(Math.max(v, s.min), s.max));
            s.label.textContent = s.show(s.get());
            const resets = typeof s.resets === 'function' ? s.resets() : s.resets;
            if (resets) {
                tracer.changed();
            } else if (s.id === 'target') {
                // A higher target lets a converged image continue; a lower one stops sooner
                if (!tracer.running) tracer.requestPresent();
            } else {
                tracer.requestPresent();
            }
            showState();
            updateOutput();
        });
    }

    resolutionSelect.addEventListener('change', (e) => {
        const i = parseInt(e.target.value, 10);
        tracer.scaleIndex = Math.min(Math.max(Number.isNaN(i) ? 0 : i, 0), RESOLUTION_SCALES.length - 1);
        tracer.resizeRender();
        tracer.changed();
        updateOutput();
    });

    toneSelect.addEventListener('change', (e) => {
        tracer.toneMap = e.target.value === '1' ? 1 : 0;
        tracer.requestPresent();
        updateOutput();
    });

    neeBox.addEventListener('change', (e) => {
        tracer.useNEE = e.target.checked;
        tracer.changed();
        updateOutput();
    });

    dofBox.addEventListener('change', (e) => {
        tracer.dofOn = e.target.checked;
        tracer.changed();
        updateOutput();
    });

    sceneButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (!SCENES[btn.dataset.scene]) return;
            tracer.loadScene(btn.dataset.scene);
            showParams();
            showState();
            updateOutput();
        });
    });

    function updateOutput() {
        const pixels = tracer.renderWidth * tracer.renderHeight;
        const raysPerFrame = tracer.raysLastFrame;
        const raysPerSecond = tracer.running ? raysPerFrame * tracer.fps : 0;
        const seconds = tracer.accumTime / 1000;
        const status = tracer.running ? '<span class="success">Accumulating</span>'
            : tracer.converged ? `<span class="success">Converged at ${tracer.targetSpp} spp</span> (GPU idle)`
            : '<span class="info">Paused</span>';

        output.innerHTML = `<span class="success">✓ Path Tracer Active</span>

<span class="info">Configuration:</span>
• Scene: ${SCENES[tracer.sceneKey].name} (${tracer.scene.count} primitives, ${tracer.scene.lightCount} sampled light${tracer.scene.lightCount === 1 ? '' : 's'})
• Resolution: ${tracer.renderWidth} × ${tracer.renderHeight} (${pixels.toLocaleString()} paths per sample)
• Max bounces: ${tracer.maxBounces} · Russian roulette after bounce 3
• Light sampling: ${tracer.useNEE ? 'next-event estimation + MIS (power heuristic)' : 'off (BSDF sampling only)'}
• Camera: field of view ${tracer.camera.fov.toFixed(0)}°, depth of field ${tracer.dofOn ? `on (aperture ${tracer.camera.aperture.toFixed(3)}, focus ${tracer.camera.focus.toFixed(2)})` : 'off'}
• Display: ${TONE_MAPS[tracer.toneMap]} tone map, exposure ${tracer.exposureEV.toFixed(1)} EV, firefly clamp ${tracer.clampMax ? tracer.clampMax : 'off'}
• Status: ${status}

<span class="info">Denoise-free progress:</span>
• Samples per pixel: ${tracer.sampleCount.toLocaleString()} / ${tracer.targetSpp.toLocaleString()}
• Frame rate: ${tracer.running ? tracer.fps.toFixed(0) : 0} FPS (1 sample per pixel per frame)
• Rays in the last measured frame: ${raysPerFrame ? raysPerFrame.toLocaleString() + ` (${(raysPerFrame / pixels).toFixed(2)} per pixel)` : '—'}
• Rays per second: ${raysPerSecond ? (raysPerSecond / 1e6).toFixed(0) + ' million' : '—'}
• Accumulation time: ${seconds.toFixed(1)} s

<span class="info">Algorithm:</span>
• One compute invocation per pixel traces one path: jittered camera ray (thin lens with DoF), brute-force closest hit over ≤ ${MAX_PRIMS} spheres, quads, rotated boxes and planes
• Diffuse: cosine-weighted bounce + one shadow ray to a light (area sampling for quads, cone sampling for spheres), combined by multiple importance sampling
• Metal: reflection fuzzed by roughness · Glass: Schlick Fresnel picks reflection or refraction
• PCG random numbers seeded from pixel index and sample index; the radiance sum lives in a storage buffer
• Display: average = sum / count, exposure, ${TONE_MAPS[tracer.toneMap]}, sRGB; noise falls as 1/√spp`;
    }

    tracer.loadScene('cornell');
    showParams();
    showState();
    updateOutput();
    return tracer;
}

// Exported for the CPU reference port used in testing
export { SCENES, SceneBuilder, intersectScene, cameraBasis };
