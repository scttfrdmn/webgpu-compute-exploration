// Convolution reverb on the GPU: uniformly partitioned FFT convolution of a live stereo signal
// with a stereo impulse response (IR) of up to 6 seconds.
//
// Algorithm: uniformly partitioned overlap-save (UPOLS; Gardner 1995, "Efficient convolution
// without input-output delay", JAES 43(3); Wefers 2015, "Partitioned convolution algorithms for
// real-time auralization", ch. 5). The IR h of length L is cut into P = ceil(L / B) partitions
// h_j of B samples. Each is zero-padded to N = 2B and transformed once: H_j = FFT(h_j). For input
// block k (B new samples) the GPU
//   1. transforms the window [x_{k-1}, x_k] (previous and current block, N samples): X_k = FFT(.)
//   2. stores X_k in a frequency-domain delay line (a ring of spectra, newest at slot k mod R)
//   3. multiply-accumulates over all partitions in ONE dispatch: Y = sum_{j<P} X_{k-j} · H_j
//      (one invocation per frequency bin loops over the partitions)
//   4. y = IFFT(Y) / N; the last B samples of y are block k of the linear convolution x * h (the
//      first B are circular wrap-around and are discarded: that is overlap-save).
// Because the FFT is linear, summing the partition products before a single inverse FFT gives
// exactly sum_j (x * h_j) delayed by j·B. Cost per block: one N-point FFT, P·(N/2+1) complex
// multiply-adds per channel, one inverse FFT, instead of L multiply-adds per output sample.
//
// Stereo with one complex FFT (the "two real FFTs for the price of one" trick): the window is
// uploaded as z[n] = xL[n] + i·xR[n]. With Z = FFT(z) and Hermitian symmetry of real spectra,
//   XL[k] = (Z[k] + conj(Z[N-k])) / 2,   XR[k] = (Z[k] - conj(Z[N-k])) / (2i)
// so the split pass stores (XL, XR) for bins 0..N/2 only. The filters H_L, H_R are prepared the
// same way from h_L + i·h_R. The MAC builds Y = Y_L + i·Y_R (and Y[N-k] = conj(Y_L) + i·conj(Y_R)),
// so one inverse FFT returns y_L in the real part and y_R in the imaginary part.
//
// Processing model (streaming): an AudioWorklet (inline module from a Blob URL) collects B input
// frames and posts them to the main thread; the main thread uploads the window, runs the passes
// above in one compute pass, copies the B wet frames to a mappable buffer and posts them back
// when mapAsync resolves. The worklet plays wet block k at a fixed time, (1 + D)·B frames after
// block k started arriving: B frames for the partition (the inherent latency of a uniformly
// partitioned convolver; the input block must be complete before it can be transformed) plus D
// blocks of slack for the GPU round trip. A block that is not back in time is played as silence
// and counted as an underrun; the GPU state never skips a block, so the next one is correct. The
// dry signal and the browser's ConvolverNode (A/B reference) are delayed by the same (1 + D)·B,
// so switching A/B or the mix does not shift anything in time.
//
// Readbacks: the wet block every block (B·8 bytes; streaming needs it to reach the speakers), and
// one self-test readback per IR (5 blocks) that compares the GPU against a direct CPU convolution.
//
// Compute uniform layout (struct Params, 36 bytes; one 256-byte-aligned slot per FFT stage,
// selected with a dynamic offset, so each butterfly dispatch sees its own `stage`):
//   offset  0  n           u32  FFT size N = 2B                         -> Uint32 view [0]
//   offset  4  log2n       u32  log2(N), number of butterfly stages     -> Uint32 view [1]
//   offset  8  stage       u32  butterfly stage of this dispatch        -> Uint32 view [2]
//   offset 12  blockSize   u32  B (= N/2; also N/2 + 1 bins - 1)         -> Uint32 view [3]
//   offset 16  batch       u32  FFTs in one dispatch (P for the IR, 1)   -> Uint32 view [4]
//   offset 20  partitions  u32  P, the MAC loop count                    -> Uint32 view [5]
//   offset 24  head        u32  delay-line slot of the newest spectrum   -> Uint32 view [6]
//   offset 28  ring        u32  delay-line slots R                      -> Uint32 view [7]
//   offset 32  invN        f32  1/N, inverse FFT scaling                -> Float32 view [8]
// Every slot written for a submit carries the same values except `stage`. Each block (and the IR
// preparation) is its own writeBuffer + submit pair, so queue order keeps their params apart.

const BLOCK_SIZES = [1024, 2048];
const MAX_IR_SECONDS = 6;
const MAX_IR_FRAMES = 576000;            // 6 s at 96 kHz; longer IRs are truncated
const MAX_N = 4096;
const MAX_LOG2N = 12;
const WORKGROUP = 64;
const PARAMS_SIZE = 36;
const VIEW_SIZE = 80;
const MAX_IN_FLIGHT = 8;                  // wet blocks submitted but not yet read back
const MAX_BACKLOG = 64;                   // input blocks waiting: the GPU cannot keep up
const SELF_TEST_CHUNK = 32;               // self-test blocks per onSubmittedWorkDone
const SELF_TEST_SLOTS = 6;
const OUTPUT_LIMIT = 0.5;                 // master gain ceiling and soft-clip ceiling
const IR_NORM = 0.7;                      // RMS over channels of ||h||2 after normalization

// Delay-line slots for block size B: enough partitions for MAX_IR_FRAMES
const ringSlots = (b) => Math.ceil(MAX_IR_FRAMES / b);
const MAX_PARTITIONS = ringSlots(BLOCK_SIZES[0]);   // 563, the MAC loop bound
const maxOver = (f) => Math.max(...BLOCK_SIZES.map(f));
const CAPACITY = {
    irFrames: maxOver(b => ringSlots(b) * b),                // zero-padded IR, frames
    complex: maxOver(b => ringSlots(b) * 2 * b),             // batched FFT scratch, vec2 count
    spectra: maxOver(b => ringSlots(b) * (b + 1)),           // ring or filters, vec4 count
};

// Canvas layout (pixels, 1024 x 512) and visualization sizes
const IR_COLS = 640;
const SPEC_ROWS = 128;
const SCOPE_COLS = 336;
const SPEC_F_LO = 50;
const SPEC_F_HI = 16000;
const SPEC_FRAME = 1024;

export const computeShader = `
struct Params {
    n: u32,
    log2n: u32,
    stage: u32,
    blockSize: u32,
    batch: u32,
    partitions: u32,
    head: u32,
    ring: u32,
    invN: f32,
}

@group(0) @binding(0) var<storage, read> timeIn: array<vec2<f32>>;          // (L, R) frames: input window or IR
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;             // complex input of a pass
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;       // complex output of a pass
@group(0) @binding(3) var<storage, read> twiddles: array<vec2<f32>>;        // W_N^k = e^(-2πik/N), k < N/2
@group(0) @binding(4) var<storage, read_write> spectra: array<vec4<f32>>;   // (XL, XR) per bin: delay line, or filters during IR prep
@group(0) @binding(5) var<storage, read> filters: array<vec4<f32>>;         // (HL, HR) per bin, partition-major
@group(0) @binding(6) var<storage, read_write> wet: array<vec2<f32>>;       // B output frames (L, R)
@group(0) @binding(7) var<uniform> params: Params;

const MAX_PARTITIONS: u32 = ${MAX_PARTITIONS}u;

fn bitReverse(i: u32) -> u32 {
    return reverseBits(i) >> (32u - params.log2n);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Input window -> bit-reversed order. The window is already the packed complex sequence
// z[n] = xL[n] + i·xR[n]: frames 0..B-1 are the previous block, B..N-1 the current one.
// Iterative radix-2 decimation in time wants bit-reversed input and returns natural order.
@compute @workgroup_size(${WORKGROUP})
fn cs_pack_block(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    dst[bitReverse(i)] = timeIn[i];
}

// IR partitions, batched: FFT b holds h_b = h[b·B .. b·B + B) followed by B zeros (the zero
// padding makes the circular convolution of length N contain B valid linear outputs).
@compute @workgroup_size(${WORKGROUP})
fn cs_pack_ir(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.batch * params.n) {
        return;
    }
    let b = t >> params.log2n;
    let i = t & (params.n - 1u);
    var v = vec2<f32>(0.0, 0.0);
    if (i < params.blockSize) {
        v = timeIn[b * params.blockSize + i];
    }
    dst[(b << params.log2n) + bitReverse(i)] = v;
}

// Butterfly stage s of 'batch' independent N-point FFTs stored back to back. Invocation t:
//   b = t div (N/2), p = t mod (N/2), h = 2^s, j = p mod h,
//   i0 = b·N + (p div h)·2h + j, i1 = i0 + h, w = W_{2h}^j = W_N^(j·N/2h)
//   X[i0] = a + w·c, X[i1] = a - w·c   (a = src[i0], c = src[i1])
// The inverse transform uses conj(w); its 1/N scaling is applied in cs_output.
fn butterfly(t: u32, sign: f32) {
    let halfN = params.n >> 1u;
    if (t >= params.batch * halfN) {
        return;
    }
    let b = t >> (params.log2n - 1u);
    let p = t & (halfN - 1u);
    let h = 1u << params.stage;
    let j = p & (h - 1u);
    let i0 = (b << params.log2n) + ((p >> params.stage) << (params.stage + 1u)) + j;
    let i1 = i0 + h;
    let tw = twiddles[j << (params.log2n - 1u - params.stage)];
    let w = vec2<f32>(tw.x, sign * tw.y);
    let a = src[i0];
    let wc = cmul(w, src[i1]);
    dst[i0] = a + wc;
    dst[i1] = a - wc;
}

@compute @workgroup_size(${WORKGROUP})
fn cs_fft_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    butterfly(gid.x, 1.0);
}

@compute @workgroup_size(${WORKGROUP})
fn cs_fft_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    butterfly(gid.x, -1.0);
}

// Separates the packed spectrum Z of FFT b into the two real-signal spectra, bins 0..N/2:
//   XL = (Z[k] + conj(Z[N-k])) / 2,  XR = (Z[k] - conj(Z[N-k])) / (2i)
// and stores (XL, XR) at slot head + b (delay-line slot for a block, partition b for the IR).
@compute @workgroup_size(${WORKGROUP})
fn cs_split(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bins = params.blockSize + 1u;
    let t = gid.x;
    if (t >= params.batch * bins) {
        return;
    }
    let b = t / bins;
    let k = t - b * bins;
    let base = b << params.log2n;
    let z = src[base + k];
    let m = src[base + ((params.n - k) & (params.n - 1u))];
    let zc = vec2<f32>(m.x, -m.y);
    let xl = 0.5 * (z + zc);
    let d = z - zc;
    let xr = 0.5 * vec2<f32>(d.y, -d.x);
    spectra[(params.head + b) * bins + k] = vec4<f32>(xl, xr);
}

// Frequency-domain multiply-accumulate, one invocation per bin k = 0..N/2, one loop over all P
// partitions (bounded by MAX_PARTITIONS): Y_c[k] = sum_j X_{newest - j}[k] · H_j[k] for c = L, R.
// Writes Y = Y_L + i·Y_R at bit-reversed positions for the inverse FFT, plus the mirrored bin
// Y[N-k] = conj(Y_L[k]) + i·conj(Y_R[k]) (bins 0 and N/2 have no distinct mirror).
@compute @workgroup_size(${WORKGROUP})
fn cs_mac(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bins = params.blockSize + 1u;
    let k = gid.x;
    if (k >= bins) {
        return;
    }
    var accL = vec2<f32>(0.0, 0.0);
    var accR = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < MAX_PARTITIONS; j++) {
        if (j >= params.partitions) {
            break;
        }
        let slot = (params.head + params.ring - j) % params.ring;
        let x = spectra[slot * bins + k];
        let hj = filters[j * bins + k];
        accL += cmul(x.xy, hj.xy);
        accR += cmul(x.zw, hj.zw);
    }
    dst[bitReverse(k)] = vec2<f32>(accL.x - accR.y, accL.y + accR.x);
    if (k > 0u && k < params.blockSize) {
        dst[bitReverse(params.n - k)] = vec2<f32>(accL.x + accR.y, accR.x - accL.y);
    }
}

// Overlap-save: keep the last B samples of the circular convolution, scaled by 1/N.
// Real part = left channel, imaginary part = right channel.
@compute @workgroup_size(${WORKGROUP})
fn cs_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.blockSize) {
        return;
    }
    wet[i] = src[params.blockSize + i] * params.invN;
}
`;

// Render uniform layout (struct View, 80 bytes):
//   offset  0  canvasSize vec2<f32>  canvas width, height                 -> Float32 view [0], [1]
//   offset  8  irSeconds  f32        IR length in seconds                  -> Float32 view [2]
//   offset 12  rt60       f32        measured T30-based RT60 (0 = none)    -> Float32 view [3]
//   offset 16  levels     vec4<f32>  peak in L, in R, wet L, wet R         -> Float32 view [4..7]
//   offset 32  holds      vec4<f32>  peak-hold of the same                 -> Float32 view [8..11]
//   offset 48  outLevel   f32        final output peak                     -> Float32 view [12]
//   offset 52  outHold    f32        its peak hold                         -> Float32 view [13]
//   offset 56  gainIn     f32        scope gain, input lane                -> Float32 view [14]
//   offset 60  gainOut    f32        scope gain, wet lane                  -> Float32 view [15]
//   offset 64  preDelay   f32        seconds                               -> Float32 view [16]
//   offset 68  hasScope   u32        1 once a wet block was received       -> Uint32 view [17]
//   offset 72  (padding to the 16-byte struct alignment)
export const renderShader = `
struct View {
    canvasSize: vec2<f32>,
    irSeconds: f32,
    rt60: f32,
    levels: vec4<f32>,
    holds: vec4<f32>,
    outLevel: f32,
    outHold: f32,
    gainIn: f32,
    gainOut: f32,
    preDelay: f32,
    hasScope: u32,
}

@group(0) @binding(0) var<storage, read> irEnv: array<vec4<f32>>;   // per column: min L, max L, min R, max R (of peak)
@group(0) @binding(1) var<storage, read> irSpec: array<f32>;        // IR_COLS x SPEC_ROWS, dB re max
@group(0) @binding(2) var<storage, read> irEdc: array<f32>;         // Schroeder energy decay per column, dB
@group(0) @binding(3) var<storage, read> scope: array<vec4<f32>>;   // per column: in min, in max, wet min, wet max
@group(0) @binding(4) var<uniform> ui: View;

const IR_COLS: u32 = ${IR_COLS}u;
const SPEC_ROWS: u32 = ${SPEC_ROWS}u;
const SCOPE_COLS: u32 = ${SCOPE_COLS}u;
const WAVE_X0: f32 = 16.0;
const WAVE_W: f32 = ${IR_COLS}.0;
const WAVE_Y0: f32 = 16.0;
const WAVE_H: f32 = 160.0;
const SPEC_Y0: f32 = 192.0;
const SPEC_H: f32 = 304.0;
const SPEC_DB: f32 = 90.0;
const SCOPE_X0: f32 = 672.0;
const SCOPE_W: f32 = ${SCOPE_COLS}.0;
const SCOPE_Y0: f32 = 16.0;
const SCOPE_H: f32 = 240.0;
const METER_Y0: f32 = 272.0;
const METER_H: f32 = 224.0;
const METER_DB: f32 = 60.0;
const BACKGROUND: vec3<f32> = vec3<f32>(0.020, 0.027, 0.055);
const PANEL: vec3<f32> = vec3<f32>(0.035, 0.045, 0.085);
const GRID: vec3<f32> = vec3<f32>(0.10, 0.13, 0.20);
const LEFT_COLOR: vec3<f32> = vec3<f32>(0.30, 0.85, 1.0);
const RIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.62, 0.25);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

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

fn inside(p: vec2<f32>, x0: f32, y0: f32, w: f32, h: f32) -> bool {
    return p.x >= x0 && p.x < x0 + w && p.y >= y0 && p.y < y0 + h;
}

// Black -> purple -> red -> orange -> pale yellow
fn heat(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.0, 0.0, 0.02);
    let c1 = vec3<f32>(0.25, 0.04, 0.45);
    let c2 = vec3<f32>(0.75, 0.15, 0.35);
    let c3 = vec3<f32>(0.98, 0.55, 0.10);
    let c4 = vec3<f32>(1.0, 1.0, 0.75);
    if (t < 0.25) { return mix(c0, c1, t / 0.25); }
    if (t < 0.5) { return mix(c1, c2, (t - 0.25) / 0.25); }
    if (t < 0.75) { return mix(c2, c3, (t - 0.5) / 0.25); }
    return mix(c3, c4, (t - 0.75) / 0.25);
}

// 1 on a vertical grid line every half second of IR time
fn timeGrid(px: f32) -> bool {
    let t = (px - WAVE_X0) / WAVE_W * ui.irSeconds;
    return abs(t - round(t * 2.0) * 0.5) * WAVE_W / max(ui.irSeconds, 0.001) < 0.5;
}

// Min-max span [lo, hi] of a lane centred at cy with half-height amp, drawn with a 1 px minimum
fn span(py: f32, cy: f32, amp: f32, lo: f32, hi: f32) -> bool {
    let top = cy - clamp(hi, -1.0, 1.0) * amp - 0.5;
    let bottom = cy - clamp(lo, -1.0, 1.0) * amp + 0.5;
    return py >= top && py <= bottom;
}

// IR waveform: left channel in the upper lane, right in the lower, linear, scaled to the IR peak
fn drawWave(p: vec2<f32>) -> vec3<f32> {
    var color = PANEL;
    if (timeGrid(p.x)) { color = GRID; }
    let c = min(u32(p.x - WAVE_X0), IR_COLS - 1u);
    let e = irEnv[c];
    let amp = WAVE_H * 0.25 - 4.0;
    let upper = p.y < WAVE_Y0 + WAVE_H * 0.5;
    if (upper) {
        let cy = WAVE_Y0 + WAVE_H * 0.25;
        if (abs(p.y - cy) < 0.5) { color = GRID; }
        if (span(p.y, cy, amp, e.x, e.y)) { color = LEFT_COLOR; }
    } else {
        let cy = WAVE_Y0 + WAVE_H * 0.75;
        if (abs(p.y - cy) < 0.5) { color = GRID; }
        if (span(p.y, cy, amp, e.z, e.w)) { color = RIGHT_COLOR; }
    }
    return color;
}

// IR spectrogram (time left to right, log frequency 50 Hz - 16 kHz bottom to top, 0 to -90 dB)
// with the Schroeder energy decay curve (white), the -60 dB line and the measured RT60 (green)
fn drawSpectrogram(p: vec2<f32>) -> vec3<f32> {
    let c = min(u32(p.x - WAVE_X0), IR_COLS - 1u);
    let rowF = (SPEC_Y0 + SPEC_H - p.y) / SPEC_H * f32(SPEC_ROWS);
    let row = min(u32(max(rowF, 0.0)), SPEC_ROWS - 1u);
    let db = irSpec[c * SPEC_ROWS + row];
    var color = heat(clamp(1.0 + db / SPEC_DB, 0.0, 1.0)) * 0.85;

    let y60 = SPEC_Y0 + 60.0 / SPEC_DB * SPEC_H;
    if (abs(p.y - y60) < 0.6 && fract(p.x / 8.0) < 0.5) {
        color = vec3<f32>(0.7, 0.7, 0.75);
    }
    if (timeGrid(p.x) && fract(p.y / 6.0) < 0.5) {
        color = mix(color, vec3<f32>(0.6), 0.35);
    }
    let e0 = clamp(irEdc[c], -SPEC_DB - 2.0, 0.0);
    let e1 = clamp(irEdc[min(c + 1u, IR_COLS - 1u)], -SPEC_DB - 2.0, 0.0);
    let ya = SPEC_Y0 - e0 / SPEC_DB * SPEC_H;
    let yb = SPEC_Y0 - e1 / SPEC_DB * SPEC_H;
    if (p.y >= min(ya, yb) - 1.0 && p.y <= max(ya, yb) + 1.0) {
        color = vec3<f32>(1.0);
    }
    if (ui.rt60 > 0.0) {
        let x = WAVE_X0 + ui.rt60 / max(ui.irSeconds, 0.001) * WAVE_W;
        if (abs(p.x - x) < 1.0 && fract(p.y / 6.0) < 0.6) {
            color = vec3<f32>(0.35, 1.0, 0.45);
        }
    }
    return color;
}

// Latest block: input (upper lane) and GPU wet output (lower lane), mid channel, auto gain
fn drawScope(p: vec2<f32>) -> vec3<f32> {
    var color = PANEL;
    let upper = p.y < SCOPE_Y0 + SCOPE_H * 0.5;
    var cy = SCOPE_Y0 + SCOPE_H * 0.75;
    if (upper) { cy = SCOPE_Y0 + SCOPE_H * 0.25; }
    if (abs(p.y - cy) < 0.5) { color = GRID; }
    if (abs(p.y - (SCOPE_Y0 + SCOPE_H * 0.5)) < 0.5) { color = GRID; }
    if (ui.hasScope == 0u) { return color; }
    let c = min(u32(p.x - SCOPE_X0), SCOPE_COLS - 1u);
    let s = scope[c];
    let amp = SCOPE_H * 0.25 - 6.0;
    if (upper) {
        if (span(p.y, cy, amp, s.x * ui.gainIn, s.y * ui.gainIn)) { color = LEFT_COLOR; }
    } else {
        if (span(p.y, cy, amp, s.z * ui.gainOut, s.w * ui.gainOut)) { color = RIGHT_COLOR; }
    }
    return color;
}

fn meterHeight(v: f32) -> f32 {
    return clamp((20.0 * log2(max(v, 1e-6)) / log2(10.0) + METER_DB) / METER_DB, 0.0, 1.0);
}

// Five peak meters (-60..0 dBFS): input L, input R, wet L, wet R, final output; hold lines
fn drawMeters(p: vec2<f32>) -> vec3<f32> {
    var color = PANEL;
    let y = (METER_Y0 + METER_H - p.y) / METER_H;   // 0 at the bottom, 1 at 0 dBFS
    let tick = abs(y * METER_DB - round(y * METER_DB / 12.0) * 12.0) * METER_H / METER_DB;
    if (tick < 0.5) { color = GRID; }
    let lx = p.x - SCOPE_X0 - 16.0;
    if (lx < 0.0) { return color; }
    let i = u32(lx / 64.0);
    if (i > 4u || lx - f32(i) * 64.0 >= 44.0) { return color; }
    var v = ui.outLevel;
    var hold = ui.outHold;
    if (i < 4u) {
        v = ui.levels[i];
        hold = ui.holds[i];
    }
    var lit = select(LEFT_COLOR, RIGHT_COLOR, i == 2u || i == 3u);
    if (i == 4u) { lit = vec3<f32>(0.40, 0.95, 0.50); }
    if (y > 0.9) { lit = vec3<f32>(1.0, 0.30, 0.25); } else if (y > 0.8) { lit = vec3<f32>(1.0, 0.85, 0.30); }
    let h = meterHeight(v);
    if (y <= h) {
        color = lit;
    } else {
        color = mix(color, lit, 0.08);
    }
    if (abs(y - meterHeight(hold)) * METER_H < 1.0 && hold > 1e-6) {
        color = vec3<f32>(0.95);
    }
    return color;
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let p = frag.position.xy;
    var color = BACKGROUND;
    if (inside(p, WAVE_X0, WAVE_Y0, WAVE_W, WAVE_H)) {
        color = drawWave(p);
    } else if (inside(p, WAVE_X0, SPEC_Y0, WAVE_W, SPEC_H)) {
        color = drawSpectrogram(p);
    } else if (inside(p, SCOPE_X0, SCOPE_Y0, SCOPE_W, SCOPE_H)) {
        color = drawScope(p);
    } else if (inside(p, SCOPE_X0, METER_Y0, SCOPE_W, METER_H)) {
        color = drawMeters(p);
    }
    return vec4<f32>(color, 1.0);
}
`;

// ---------------------------------------------------------------------------------------------
// Procedural impulse responses (no IR files, so no licensing questions)
// ---------------------------------------------------------------------------------------------

// Each space: RT60 of the mid band (seconds), pre-delay, early reflections (count and time window
// after the pre-delay), the build-up time of the diffuse tail, and the RT60 ratios of the low
// (< ~400 Hz) and high (> ~4 kHz) bands: high < 1 lowpasses the tail more and more over time.
export const SPACES = {
    room:      { name: 'Small room',   rt60: 0.45, preDelay: 0.004, reflections: 8,  erStart: 0.002, erEnd: 0.030, erLevel: 2.5, buildUp: 0.010, lowRatio: 1.0,  highRatio: 0.45, seed: 11 },
    hall:      { name: 'Concert hall', rt60: 2.2,  preDelay: 0.022, reflections: 12, erStart: 0.008, erEnd: 0.090, erLevel: 2.0, buildUp: 0.060, lowRatio: 1.25, highRatio: 0.50, seed: 23 },
    cathedral: { name: 'Cathedral',    rt60: 4.4,  preDelay: 0.045, reflections: 8,  erStart: 0.020, erEnd: 0.150, erLevel: 1.6, buildUp: 0.140, lowRatio: 1.3,  highRatio: 0.40, seed: 37 },
    plate:     { name: 'Plate',        rt60: 1.8,  preDelay: 0.0,   reflections: 0,  erStart: 0,     erEnd: 0,     erLevel: 0,   buildUp: 0.002, lowRatio: 0.85, highRatio: 0.90, seed: 41 },
    reverse:   { name: 'Reverse',      rt60: 1.6,  preDelay: 0.0,   reflections: 0,  erStart: 0,     erEnd: 0,     erLevel: 0,   buildUp: 0.020, lowRatio: 1.1,  highRatio: 0.60, seed: 53, reverse: true },
};

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

// IR length in seconds for a decay time: until about -75 dB in the slowest band, 1..6 s
export function irSeconds(space, rt60, preDelay) {
    const slowest = rt60 * Math.max(space.lowRatio, 1);
    return Math.min(MAX_IR_SECONDS, Math.max(1, preDelay + 1.25 * slowest));
}

// Stereo IR: decorrelated white noise per channel split into three bands with one-pole filters,
// each band with its own exponential envelope 10^(-3t/RT60_band) (-60 dB after RT60), a smooth
// build-up 1 - e^(-t/buildUp), sparse early reflections panned between the channels, a pre-delay
// and a short fade-out. 'Reverse' plays the hall-like tail backwards. Normalized so the RMS over
// the channels of ||h||2 is IR_NORM (white noise in -> wet at 0.7x its RMS).
export function generateImpulseResponse(spaceKey, { sampleRate, rt60, preDelay }) {
    const space = SPACES[spaceKey];
    const fs = sampleRate;
    const seconds = irSeconds(space, rt60, preDelay);
    const length = Math.min(Math.round(seconds * fs), MAX_IR_FRAMES);
    const tailStart = space.reverse ? 0 : Math.round(preDelay * fs);
    const rand = mulberry32(space.seed);
    const coef = (fc) => 1 - Math.exp(-2 * Math.PI * fc / fs);
    const aLow = coef(400);
    const aHigh = coef(4000);
    const rtLow = rt60 * space.lowRatio;
    const rtHigh = rt60 * space.highRatio;
    // Per-sample amplitude decay factors: 10^(-3/(RT60·fs))
    const dLow = Math.pow(10, -3 / (rtLow * fs));
    const dMid = Math.pow(10, -3 / (rt60 * fs));
    const dHigh = Math.pow(10, -3 / (rtHigh * fs));
    const fade = Math.min(Math.round(0.05 * fs), length >> 2);
    const channels = [new Float32Array(length), new Float32Array(length)];

    for (let c = 0; c < 2; c++) {
        const h = channels[c];
        let lpLow = 0, lpHigh = 0;
        let eLow = 1, eMid = 1, eHigh = 1;
        for (let n = tailStart; n < length; n++) {
            const x = (rand() + rand() - 1) * 1.2247;   // triangular noise, unit variance
            lpLow += aLow * (x - lpLow);
            lpHigh += aHigh * (x - lpHigh);
            const low = lpLow;
            const mid = lpHigh - lpLow;
            const high = x - lpHigh;
            const t = (n - tailStart) / fs;
            const onset = 1 - Math.exp(-t / space.buildUp);
            h[n] = onset * (low * eLow + mid * eMid + high * eHigh);
            eLow *= dLow;
            eMid *= dMid;
            eHigh *= dHigh;
        }
    }

    // Early reflections: single-sample taps, amplitude falling with time, alternating sides
    for (let r = 0; r < space.reflections; r++) {
        const t = space.erStart + (space.erEnd - space.erStart) * (r + rand() * 0.8) / space.reflections;
        const n = tailStart + Math.round(t * fs);
        if (n >= length) continue;
        const level = space.erLevel * (1 - 0.6 * r / space.reflections) * (rand() < 0.5 ? -1 : 1);
        const pan = 0.5 + (r % 2 ? 0.35 : -0.35) * (0.5 + 0.5 * rand());
        channels[0][n] += level * Math.cos(pan * Math.PI / 2) * Math.SQRT2;
        channels[1][n] += level * Math.sin(pan * Math.PI / 2) * Math.SQRT2;
    }

    for (const h of channels) {
        for (let i = 0; i < fade; i++) {
            h[length - 1 - i] *= 0.5 - 0.5 * Math.cos(Math.PI * i / fade);
        }
    }

    if (space.reverse) {
        // Reverse the decaying tail, then delay the swell by the pre-delay
        const pre = Math.round(preDelay * fs);
        for (let c = 0; c < 2; c++) {
            const h = channels[c];
            const reversed = new Float32Array(length);
            for (let n = 0; n + pre < length; n++) reversed[n + pre] = h[length - 1 - n];
            channels[c] = reversed;
        }
    }

    normalize(channels[0], channels[1]);
    return { left: channels[0], right: channels[1], seconds: length / fs, sampleRate: fs };
}

function normalize(left, right) {
    let energy = 0;
    for (let i = 0; i < left.length; i++) energy += left[i] * left[i] + right[i] * right[i];
    const rms = Math.sqrt(energy / 2);
    if (!(rms > 0)) return;
    const g = IR_NORM / rms;
    for (let i = 0; i < left.length; i++) {
        left[i] *= g;
        right[i] *= g;
    }
}

// An uploaded IR (an AudioBuffer already at the context rate): first two channels (mono is
// duplicated), pre-delay prepended, truncated to 6 s, normalized like the generated ones
export function prepareUploadedIR(channelData, sampleRate, preDelay) {
    const pre = Math.round(preDelay * sampleRate);
    const limit = Math.min(Math.round(MAX_IR_SECONDS * sampleRate), MAX_IR_FRAMES);
    const srcLen = channelData[0].length;
    const length = Math.max(1, Math.min(srcLen + pre, limit));
    const left = new Float32Array(length);
    const right = new Float32Array(length);
    const l = channelData[0];
    const r = channelData[1] || channelData[0];
    for (let n = pre; n < length; n++) {
        left[n] = l[n - pre];
        right[n] = r[n - pre];
    }
    normalize(left, right);
    return { left, right, seconds: length / sampleRate, sampleRate, truncated: srcLen + pre > limit };
}

// Schroeder backward integration E(t) = sum_{τ >= t} h_L² + h_R², and RT60 from a least-squares
// line through the decay between -5 and -35 dB (T30, extrapolated to 60 dB). Null if the curve
// does not reach -35 dB.
export function schroeder(left, right) {
    const edc = new Float64Array(left.length + 1);
    for (let i = left.length - 1; i >= 0; i--) edc[i] = edc[i + 1] + left[i] * left[i] + right[i] * right[i];
    return edc;
}

export function measureRT60(left, right, sampleRate, edc = schroeder(left, right)) {
    const total = edc[0];
    if (!(total > 0)) return null;
    const step = Math.max(1, Math.round(sampleRate / 4000));
    let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < left.length; i += step) {
        const db = 10 * Math.log10(edc[i] / total);
        if (db > -5) continue;
        if (db < -35) break;
        const t = i / sampleRate;
        n++; sx += t; sy += db; sxx += t * t; sxy += t * db;
    }
    if (n < 8) return null;
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    return slope < 0 ? -60 / slope : null;
}

function fftInPlace(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wr = Math.cos(ang), wi = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let cr = 1, ci = 0;
            for (let k = 0; k < len / 2; k++) {
                const a = i + k, b = a + len / 2;
                const tr = re[b] * cr - im[b] * ci;
                const ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr; im[a] += ti;
                const nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
}

// Display data for the canvas: waveform envelope, spectrogram and energy decay per column
export function analyzeImpulseResponse(left, right, sampleRate) {
    const length = left.length;
    let peak = 1e-12;
    for (let i = 0; i < length; i++) peak = Math.max(peak, Math.abs(left[i]), Math.abs(right[i]));

    const env = new Float32Array(IR_COLS * 4);
    for (let c = 0; c < IR_COLS; c++) {
        const a = Math.floor(c * length / IR_COLS);
        const b = Math.max(Math.floor((c + 1) * length / IR_COLS), a + 1);
        let lo0 = Infinity, hi0 = -Infinity, lo1 = Infinity, hi1 = -Infinity;
        for (let i = a; i < Math.min(b, length); i++) {
            lo0 = Math.min(lo0, left[i]); hi0 = Math.max(hi0, left[i]);
            lo1 = Math.min(lo1, right[i]); hi1 = Math.max(hi1, right[i]);
        }
        if (lo0 === Infinity) lo0 = hi0 = lo1 = hi1 = 0;
        env.set([lo0 / peak, hi0 / peak, lo1 / peak, hi1 / peak], c * 4);
    }

    const edc = schroeder(left, right);
    const edcDb = new Float32Array(IR_COLS);
    for (let c = 0; c < IR_COLS; c++) {
        const i = Math.floor(c * length / IR_COLS);
        edcDb[c] = edc[0] > 0 ? Math.max(10 * Math.log10(Math.max(edc[i], 1e-30) / edc[0]), -200) : -200;
    }

    // Spectrogram: Hann-windowed SPEC_FRAME-point FFTs of the mid channel centred on each column
    const power = new Float32Array(IR_COLS * SPEC_ROWS);
    const re = new Float64Array(SPEC_FRAME);
    const im = new Float64Array(SPEC_FRAME);
    const binHz = sampleRate / SPEC_FRAME;
    const fHi = Math.min(SPEC_F_HI, sampleRate / 2);
    let maxPower = 1e-30;
    for (let c = 0; c < IR_COLS; c++) {
        const center = Math.floor((c + 0.5) * length / IR_COLS);
        for (let i = 0; i < SPEC_FRAME; i++) {
            const n = center - SPEC_FRAME / 2 + i;
            const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / SPEC_FRAME);
            re[i] = n >= 0 && n < length ? 0.5 * (left[n] + right[n]) * w : 0;
            im[i] = 0;
        }
        fftInPlace(re, im);
        for (let r = 0; r < SPEC_ROWS; r++) {
            const f0 = SPEC_F_LO * Math.pow(fHi / SPEC_F_LO, r / SPEC_ROWS);
            const f1 = SPEC_F_LO * Math.pow(fHi / SPEC_F_LO, (r + 1) / SPEC_ROWS);
            const k0 = Math.max(1, Math.floor(f0 / binHz));
            const k1 = Math.max(k0, Math.min(Math.ceil(f1 / binHz), SPEC_FRAME / 2));
            let p = 0;
            for (let k = k0; k <= k1; k++) p = Math.max(p, re[k] * re[k] + im[k] * im[k]);
            power[c * SPEC_ROWS + r] = p;
            maxPower = Math.max(maxPower, p);
        }
    }
    const spec = new Float32Array(IR_COLS * SPEC_ROWS);
    for (let i = 0; i < spec.length; i++) spec[i] = Math.max(10 * Math.log10(power[i] / maxPower + 1e-30), -200);

    return { env, spec, edc: edcDb, rt60: measureRT60(left, right, sampleRate, edc), seconds: length / sampleRate, peak };
}

// ---------------------------------------------------------------------------------------------
// GPU convolution engine (no DOM or Web Audio: also driven by the CPU tests through a mock device)
// ---------------------------------------------------------------------------------------------

export class GPUConvolver {
    constructor(device) {
        this.device = device;
        const limits = device.limits;
        const largest = CAPACITY.spectra * 16;
        if (limits.maxStorageBufferBindingSize < largest || limits.maxBufferSize < largest) {
            throw new Error(`Needs storage buffers of ${(largest / 1048576).toFixed(1)} MB (device limit ${(limits.maxStorageBufferBindingSize / 1048576).toFixed(0)} MB)`);
        }
        if (limits.maxStorageBuffersPerShaderStage < 7) {
            throw new Error('Needs 7 storage buffers per shader stage');
        }
        this.maxWorkgroups = limits.maxComputeWorkgroupsPerDimension;
        const align = limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.params = new ArrayBuffer(this.paramsStride * MAX_LOG2N);
        this.paramsU32 = new Uint32Array(this.params);
        this.paramsF32 = new Float32Array(this.params);

        this.blockSize = 0;
        this.partitions = 0;
        this.irLeft = null;
        this.irRight = null;
        this.sampleRate = 48000;
        this.irVersion = 0;
        this.blockIndex = 0;
        this.chain = Promise.resolve();
        this.setupBuffers();
        this.setupPipelines();
    }

    setupBuffers() {
        const d = this.device;
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.windowBuffer = d.createBuffer({ size: MAX_N * 8, usage: storage });
        this.irTimeBuffer = d.createBuffer({ size: CAPACITY.irFrames * 8, usage: storage });
        this.complexBuffers = [0, 1].map(() => d.createBuffer({ size: CAPACITY.complex * 8, usage: storage }));
        this.twiddleBuffer = d.createBuffer({ size: (MAX_N / 2) * 8, usage: storage });
        this.spectraBuffer = d.createBuffer({ size: CAPACITY.spectra * 16, usage: storage });
        this.filterBuffer = d.createBuffer({ size: CAPACITY.spectra * 16, usage: storage });
        this.dummyBuffer = d.createBuffer({ size: 16, usage: storage });
        this.wetBuffer = d.createBuffer({ size: (MAX_N / 2) * 8, usage: storage | GPUBufferUsage.COPY_SRC });
        this.paramsBuffer = d.createBuffer({
            size: this.paramsStride * MAX_LOG2N,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const readback = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
        this.staging = [];
        for (let i = 0; i < MAX_IN_FLIGHT; i++) this.staging.push(d.createBuffer({ size: (MAX_N / 2) * 8, usage: readback }));
        this.allStaging = this.staging.slice();
        this.testStaging = d.createBuffer({ size: SELF_TEST_SLOTS * (MAX_N / 2) * 8, usage: readback });
    }

    setupPipelines() {
        const d = this.device;
        const module = d.createShaderModule({ code: computeShader });
        const C = GPUShaderStage.COMPUTE;
        const layout = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: C, buffer: { type: 'storage' } },
                { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: C, buffer: { type: 'storage' } },
                { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 6, visibility: C, buffer: { type: 'storage' } },
                { binding: 7, visibility: C,
                  buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE } }
            ]
        });
        const pipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [layout] });
        const pipeline = (entryPoint) => d.createComputePipeline({
            layout: pipelineLayout,
            compute: { module, entryPoint }
        });
        this.pipelines = {
            packBlock: pipeline('cs_pack_block'),
            packIR: pipeline('cs_pack_ir'),
            forward: pipeline('cs_fft_forward'),
            inverse: pipeline('cs_fft_inverse'),
            split: pipeline('cs_split'),
            mac: pipeline('cs_mac'),
            output: pipeline('cs_output'),
        };
        // groups[i] reads complexBuffers[i] (src) and writes complexBuffers[1 - i] (dst).
        // Stream groups: window in, split -> delay line, MAC reads the filters.
        // IR groups: IR frames in, split -> filters (the MAC is not used, so a dummy is bound).
        const make = (timeIn, spectra, filters) => [0, 1].map(i => d.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: timeIn } },
                { binding: 1, resource: { buffer: this.complexBuffers[i] } },
                { binding: 2, resource: { buffer: this.complexBuffers[1 - i] } },
                { binding: 3, resource: { buffer: this.twiddleBuffer } },
                { binding: 4, resource: { buffer: spectra } },
                { binding: 5, resource: { buffer: filters } },
                { binding: 6, resource: { buffer: this.wetBuffer } },
                { binding: 7, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        }));
        this.streamGroups = make(this.windowBuffer, this.spectraBuffer, this.filterBuffer);
        this.irGroups = make(this.irTimeBuffer, this.filterBuffer, this.dummyBuffer);
    }

    // Serializes GPU tasks that must not interleave (IR uploads, self-tests, stream resets)
    enqueue(task) {
        const result = this.chain.then(task);
        this.chain = result.catch(() => {});
        return result;
    }

    // Block size B: N = 2B, R delay-line slots; uploads the twiddles and clears the stream
    configure(blockSize) {
        if (!BLOCK_SIZES.includes(blockSize)) throw new Error(`Unsupported block size ${blockSize}`);
        this.blockSize = blockSize;
        this.n = 2 * blockSize;
        this.log2n = Math.round(Math.log2(this.n));
        this.bins = blockSize + 1;
        this.ring = ringSlots(blockSize);
        this.windowData = new Float32Array(2 * this.n);
        this.prevLeft = new Float32Array(blockSize);
        this.prevRight = new Float32Array(blockSize);
        const tw = new Float32Array(this.n);
        for (let k = 0; k < this.n / 2; k++) {
            const angle = -2 * Math.PI * k / this.n;
            tw[2 * k] = Math.cos(angle);
            tw[2 * k + 1] = Math.sin(angle);
        }
        this.device.queue.writeBuffer(this.twiddleBuffer, 0, tw);
        this.partitions = 0;
        if (this.irLeft) this.uploadIR(this.irLeft, this.irRight, this.sampleRate);
        this.resetStream();
    }

    // Writes `slots` copies of Params (slot s has stage = s); layout table at the top of the file
    writeParams(slots, batch, head) {
        const words = this.paramsStride / 4;
        for (let s = 0; s < slots; s++) {
            const o = s * words;
            this.paramsU32[o + 0] = this.n;
            this.paramsU32[o + 1] = this.log2n;
            this.paramsU32[o + 2] = s;
            this.paramsU32[o + 3] = this.blockSize;
            this.paramsU32[o + 4] = batch;
            this.paramsU32[o + 5] = this.partitions;
            this.paramsU32[o + 6] = head;
            this.paramsU32[o + 7] = this.ring;
            this.paramsF32[o + 8] = 1 / this.n;
        }
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params, 0, slots * this.paramsStride);
    }

    workgroups(count) {
        const groups = Math.ceil(count / WORKGROUP);
        if (groups > this.maxWorkgroups) throw new Error(`Dispatch of ${groups} workgroups exceeds the device limit`);
        return groups;
    }

    // log2(N) butterfly dispatches; data starts in complexBuffers[0] and ends in [log2n % 2]
    encodeStages(pass, pipeline, groups, count) {
        pass.setPipeline(pipeline);
        const wg = this.workgroups(count);
        for (let s = 0; s < this.log2n; s++) {
            pass.setBindGroup(0, groups[s % 2], [s * this.paramsStride]);
            pass.dispatchWorkgroups(wg);
        }
    }

    // Partitions the IR, FFTs all P partitions in one batched set of dispatches and stores the
    // split (HL, HR) spectra. Called whenever the IR or the block size changes.
    uploadIR(left, right, sampleRate) {
        this.irLeft = left;
        this.irRight = right;
        this.sampleRate = sampleRate;
        this.irVersion++;
        if (!this.blockSize) return;
        const B = this.blockSize;
        const P = Math.min(Math.max(1, Math.ceil(left.length / B)), this.ring);
        this.partitions = P;
        const frames = new Float32Array(P * B * 2);
        const len = Math.min(left.length, P * B);
        for (let i = 0; i < len; i++) {
            frames[2 * i] = left[i];
            frames[2 * i + 1] = right[i];
        }
        this.device.queue.writeBuffer(this.irTimeBuffer, 0, frames);
        this.writeParams(this.log2n, P, 0);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        const g = this.irGroups;
        pass.setPipeline(this.pipelines.packIR);
        pass.setBindGroup(0, g[1], [0]);                  // dst = complexBuffers[0]
        pass.dispatchWorkgroups(this.workgroups(P * this.n));
        this.encodeStages(pass, this.pipelines.forward, g, P * this.n / 2);
        pass.setPipeline(this.pipelines.split);
        pass.setBindGroup(0, g[this.log2n % 2], [0]);
        pass.dispatchWorkgroups(this.workgroups(P * this.bins));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    // Empties the delay line and the overlap-save history; the next block is block 0
    resetStream() {
        this.blockIndex = 0;
        if (this.prevLeft) {
            this.prevLeft.fill(0);
            this.prevRight.fill(0);
        }
        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.spectraBuffer);
        this.device.queue.submit([encoder.finish()]);
    }

    // Uploads one input block and encodes its convolution; the wet block is copied to `target`
    // (a mappable buffer) unless it is null
    submitBlock(left, right, target, targetOffset) {
        const B = this.blockSize;
        const head = this.blockIndex % this.ring;
        this.blockIndex++;
        const win = this.windowData;
        for (let i = 0; i < B; i++) {
            win[2 * i] = this.prevLeft[i];
            win[2 * i + 1] = this.prevRight[i];
            win[2 * (B + i)] = left[i];
            win[2 * (B + i) + 1] = right[i];
        }
        this.prevLeft.set(left);
        this.prevRight.set(right);
        this.device.queue.writeBuffer(this.windowBuffer, 0, win);
        this.writeParams(this.log2n, 1, head);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        const g = this.streamGroups;
        const p = this.pipelines;
        pass.setPipeline(p.packBlock);
        pass.setBindGroup(0, g[1], [0]);                  // dst = complexBuffers[0]
        pass.dispatchWorkgroups(this.workgroups(this.n));
        this.encodeStages(pass, p.forward, g, this.n / 2);
        pass.setPipeline(p.split);
        pass.setBindGroup(0, g[this.log2n % 2], [0]);     // src = the spectrum
        pass.dispatchWorkgroups(this.workgroups(this.bins));
        pass.setPipeline(p.mac);
        pass.setBindGroup(0, g[1], [0]);                  // dst = complexBuffers[0]
        pass.dispatchWorkgroups(this.workgroups(this.bins));
        this.encodeStages(pass, p.inverse, g, this.n / 2);
        pass.setPipeline(p.output);
        pass.setBindGroup(0, g[this.log2n % 2], [0]);     // src = the time-domain result
        pass.dispatchWorkgroups(this.workgroups(B));
        pass.end();
        if (target) encoder.copyBufferToBuffer(this.wetBuffer, 0, target, targetOffset, B * 8);
        this.device.queue.submit([encoder.finish()]);
        return head;
    }

    get freeStaging() {
        return this.staging.length;
    }

    // Convolves one stereo block; resolves to the wet block (same index order as submitted)
    processBlock(left, right) {
        const staging = this.staging.pop();
        if (!staging) return Promise.reject(new Error('No free readback buffer'));
        const index = this.blockIndex;
        const B = this.blockSize;
        this.submitBlock(left, right, staging, 0);
        return staging.mapAsync(GPUMapMode.READ, 0, B * 8).then(() => {
            const data = new Float32Array(staging.getMappedRange(0, B * 8));
            const outL = new Float32Array(B);
            const outR = new Float32Array(B);
            for (let i = 0; i < B; i++) {
                outL[i] = data[2 * i];
                outR[i] = data[2 * i + 1];
            }
            staging.unmap();
            this.staging.push(staging);
            return { index, left: outL, right: outR };
        });
    }

    // Accuracy and throughput check of the whole pipeline for the current IR: block 0 is seeded
    // noise, blocks 1..P are silence, so the output spans every partition. Blocks 0, the two after
    // the IR's peak sample (the pre-delay can make block 0 silent), P/2, P-1 and P are read back
    // (one readback) and compared with a direct CPU convolution, which costs only B·B
    // multiply-adds per block because the input is a single block. Leaves the stream reset.
    async selfTest() {
        const B = this.blockSize;
        const P = this.partitions;
        const total = P + 1;
        let peakAt = 0;
        for (let i = 0, best = -1; i < this.irLeft.length; i++) {
            const v = Math.max(Math.abs(this.irLeft[i]), Math.abs(this.irRight[i]));
            if (v > best) { best = v; peakAt = i; }
        }
        const peakBlock = Math.floor(peakAt / B);
        const checks = [...new Set([0, peakBlock, peakBlock + 1, Math.floor(P / 2), P - 1, P])]
            .filter(k => k >= 0 && k < total).sort((a, b) => a - b);
        const rand = mulberry32(1234);
        const xl = new Float32Array(B);
        const xr = new Float32Array(B);
        for (let i = 0; i < B; i++) {
            xl[i] = rand() * 2 - 1;
            xr[i] = rand() * 2 - 1;
        }
        const zero = new Float32Array(B);
        this.resetStream();
        const t0 = performance.now();
        for (let k = 0; k < total; k++) {
            const slot = checks.indexOf(k);
            this.submitBlock(k === 0 ? xl : zero, k === 0 ? xr : zero, slot >= 0 ? this.testStaging : null, Math.max(slot, 0) * B * 8);
            if ((k + 1) % SELF_TEST_CHUNK === 0) await this.device.queue.onSubmittedWorkDone();
        }
        const bytes = checks.length * B * 8;
        await this.testStaging.mapAsync(GPUMapMode.READ, 0, bytes);
        const elapsed = performance.now() - t0;
        const gpu = new Float32Array(this.testStaging.getMappedRange(0, bytes)).slice();
        this.testStaging.unmap();
        this.resetStream();

        const h = [this.irLeft, this.irRight];
        const x = [xl, xr];
        const L = h[0].length;
        let maxError = 0, peak = 0;
        checks.forEach((k, slot) => {
            for (let m = 0; m < B; m++) {
                const n = k * B + m;
                for (let c = 0; c < 2; c++) {
                    let ref = 0;
                    const lo = Math.max(0, n - L + 1);
                    const hi = Math.min(B - 1, n);
                    for (let i = lo; i <= hi; i++) ref += x[c][i] * h[c][n - i];
                    const v = gpu[slot * B * 2 + 2 * m + c];
                    maxError = Math.max(maxError, Math.abs(v - ref));
                    peak = Math.max(peak, Math.abs(ref));
                }
            }
        });
        const seconds = total * B / this.sampleRate;
        return {
            maxError, peak, relative: peak > 0 ? maxError / peak : 0, blocks: total, checked: checks,
            ms: elapsed, realtime: seconds / (elapsed / 1000), irVersion: this.irVersion
        };
    }

    destroy() {
        [this.windowBuffer, this.irTimeBuffer, ...this.complexBuffers, this.twiddleBuffer, this.spectraBuffer,
         this.filterBuffer, this.dummyBuffer, this.wetBuffer, this.paramsBuffer, ...this.allStaging,
         this.testStaging].forEach(b => b.destroy());
    }
}

// Direct convolution reference for output block m of a signal made of `blocks` (B samples each,
// preceded only by silence): y[n] = sum_i x[i]·h[n - i]. About (m + 1)·B² multiply-adds.
export function directBlock(blocks, h, m) {
    const B = blocks[0].length;
    const y = new Float64Array(B);
    for (let t = 0; t < B; t++) {
        const n = m * B + t;
        let s = 0;
        for (let i = Math.max(0, n - h.length + 1); i <= n; i++) s += blocks[Math.floor(i / B)][i % B] * h[n - i];
        y[t] = s;
    }
    return y;
}
const LIVE_CHECK_BLOCKS = 12;    // give up if the wet signal is still silent (long pre-delay)

// ---------------------------------------------------------------------------------------------
// AudioWorklet bridge: capture blocks, play wet blocks at a fixed delay, count underruns
// ---------------------------------------------------------------------------------------------

export const WORKLET_SOURCE = `
class ReverbBridgeProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const o = options.processorOptions;
        this.blockSize = o.blockSize;
        this.delay = (1 + o.depth) * o.blockSize;    // wet block k starts playing at k·B + delay
        this.ringSize = o.ringSize;
        this.capL = new Float32Array(this.blockSize);
        this.capR = new Float32Array(this.blockSize);
        this.fill = 0;
        this.captured = 0;
        this.slots = [];
        for (let i = 0; i < this.ringSize; i++) {
            this.slots.push({ index: -1, left: new Float32Array(this.blockSize), right: new Float32Array(this.blockSize) });
        }
        this.frame = 0;
        this.underruns = 0;
        this.late = 0;
        this.played = 0;
        this.sinceStats = 0;
        this.stopped = false;
        this.port.onmessage = (event) => this.receive(event.data);
    }

    receive(msg) {
        if (msg.type === 'stop') {
            this.stopped = true;
            return;
        }
        if (msg.type !== 'wet') return;
        if (msg.index * this.blockSize + this.delay < this.frame) {
            this.late++;     // its start has already been played as silence
            return;
        }
        const slot = this.slots[msg.index % this.ringSize];
        slot.index = msg.index;
        for (let i = 0; i < this.blockSize; i++) {
            const l = msg.left[i], r = msg.right[i];
            // Guard the speakers against NaN/Inf or runaway values
            slot.left[i] = l === l ? Math.max(-4, Math.min(4, l)) : 0;
            slot.right[i] = r === r ? Math.max(-4, Math.min(4, r)) : 0;
        }
    }

    process(inputs, outputs) {
        const input = inputs[0];
        const output = outputs[0];
        const n = output[0].length;
        const inL = input && input.length > 0 ? input[0] : null;
        const inR = input && input.length > 1 ? input[1] : inL;
        const B = this.blockSize;
        for (let i = 0; i < n; i++) {
            this.capL[this.fill] = inL ? inL[i] : 0;
            this.capR[this.fill] = inR ? inR[i] : 0;
            this.fill++;
            if (this.fill === B) {
                const left = this.capL, right = this.capR;
                this.port.postMessage({ type: 'block', index: this.captured, left, right }, [left.buffer, right.buffer]);
                this.captured++;
                this.capL = new Float32Array(B);
                this.capR = new Float32Array(B);
                this.fill = 0;
            }
        }
        const outL = output[0];
        const outR = output.length > 1 ? output[1] : null;
        for (let i = 0; i < n; i++) {
            const pos = this.frame + i - this.delay;
            let l = 0, r = 0;
            if (pos >= 0) {
                const blk = Math.floor(pos / B);
                const off = pos - blk * B;
                const slot = this.slots[blk % this.ringSize];
                const ready = slot.index === blk;
                if (off === 0) {
                    if (ready) this.played++; else this.underruns++;
                }
                if (ready) {
                    l = slot.left[off];
                    r = slot.right[off];
                }
            }
            outL[i] = l;
            if (outR) outR[i] = r;
        }
        this.frame += n;
        this.sinceStats += n;
        if (this.sinceStats >= 4096) {
            this.sinceStats = 0;
            this.port.postMessage({ type: 'stats', underruns: this.underruns, late: this.late, played: this.played, captured: this.captured });
        }
        return !this.stopped;
    }
}
registerProcessor('reverb-bridge', ReverbBridgeProcessor);
`;

// Main-thread side of the stream: input blocks from the worklet -> GPU -> wet blocks back.
// At most MAX_IN_FLIGHT blocks are on the GPU; later ones wait in order.
export class ReverbStream {
    constructor(convolver, port, onOutput) {
        this.convolver = convolver;
        this.port = port;
        this.onOutput = onOutput;
        this.queue = [];
        this.inFlight = 0;
        this.expected = 0;
        this.closed = false;
        this.error = null;
        this.processed = 0;
        this.roundTrip = { sum: 0, count: 0, max: 0 };
        this.stats = { underruns: 0, late: 0, played: 0, captured: 0 };
        this.maxQueue = 0;
        port.onmessage = (event) => this.receive(event.data);
    }

    receive(msg) {
        if (this.closed) return;
        if (msg.type === 'stats') {
            this.stats = msg;
            return;
        }
        if (msg.type !== 'block') return;
        if (msg.index !== this.expected) {
            this.fail(new Error(`Block ${msg.index} arrived, expected ${this.expected}`));
            return;
        }
        this.expected++;
        this.queue.push(msg);
        this.maxQueue = Math.max(this.maxQueue, this.queue.length);
        if (this.queue.length > MAX_BACKLOG) {
            this.fail(new Error('The GPU cannot keep up with the audio (backlog over 64 blocks)'));
            return;
        }
        this.pump();
    }

    pump() {
        while (!this.closed && this.queue.length && this.inFlight < MAX_IN_FLIGHT && this.convolver.freeStaging > 0) {
            const block = this.queue.shift();
            const irVersion = this.convolver.irVersion;
            const sent = performance.now();
            this.inFlight++;
            this.convolver.processBlock(block.left, block.right).then((wet) => {
                this.inFlight--;
                if (this.closed) return;
                const ms = performance.now() - sent;
                this.roundTrip.sum += ms;
                this.roundTrip.count++;
                this.roundTrip.max = Math.max(this.roundTrip.max, ms);
                this.processed++;
                if (this.onOutput) this.onOutput(block, wet, irVersion);
                this.port.postMessage({ type: 'wet', index: block.index, left: wet.left, right: wet.right },
                    [wet.left.buffer, wet.right.buffer]);
                this.pump();
            }, (error) => this.fail(error));
        }
    }

    fail(error) {
        if (this.closed) return;
        this.error = error;
        if (this.onError) this.onError(error);
    }

    close() {
        this.closed = true;
        this.queue = [];
        this.port.onmessage = null;
        try { this.port.postMessage({ type: 'stop' }); } catch { /* port already closed */ }
    }
}

// ---------------------------------------------------------------------------------------------
// Web Audio: sources, the dry / GPU / ConvolverNode paths, output protection
// ---------------------------------------------------------------------------------------------

// Synth demo: an 8-second loop of plucks, claps and chord stabs with gaps, so tails are audible
const SYNTH_LOOP = 8;
const SYNTH_EVENTS = [
    [0.00, 'pluck', 220.00], [0.25, 'pluck', 329.63], [0.50, 'pluck', 440.00],
    [1.75, 'clap'],
    [3.00, 'chord', [261.63, 329.63, 392.00]],
    [4.50, 'clap'], [4.75, 'clap'],
    [6.00, 'pluck', 587.33], [6.18, 'pluck', 493.88], [6.36, 'pluck', 392.00],
];
const SCHEDULE_AHEAD = 0.2;

class ReverbAudio {
    constructor() {
        this.ctx = null;
        this.kind = null;            // 'synth' | 'mic' | 'file'
        this.label = '';
        this.nodes = new Set();      // live source nodes; finished synth notes remove themselves
        this.stream = null;
        this.timer = null;
        this.request = 0;
        this.fileBuffer = null;
        this.fileName = '';
        this.workletReady = null;
        this.bridge = null;
        this.latency = 0;
        this.mix = { wet: 0.35, gain: 0.3, bypass: false, useGPU: true };
    }

    // Must run inside a click handler, before any await
    ensureContext() {
        if (!this.ctx) {
            const Ctor = window.AudioContext || window.webkitAudioContext;
            if (!Ctor) throw new Error('The Web Audio API is not available in this browser');
            const ctx = new Ctor();
            this.ctx = ctx;
            this.inputBus = ctx.createGain();
            this.dryDelay = ctx.createDelay(1);
            this.dryGain = ctx.createGain();
            this.convolver = ctx.createConvolver();
            this.convolver.normalize = false;           // same scaling as the GPU path
            this.convDelay = ctx.createDelay(1);
            this.convWet = ctx.createGain();
            this.gpuWet = ctx.createGain();
            this.master = ctx.createGain();
            // Output protection: a fast limiter, then a soft clip that never exceeds OUTPUT_LIMIT
            this.limiter = ctx.createDynamicsCompressor();
            this.limiter.threshold.value = -10;
            this.limiter.knee.value = 6;
            this.limiter.ratio.value = 12;
            this.limiter.attack.value = 0.003;
            this.limiter.release.value = 0.2;
            this.clipper = ctx.createWaveShaper();
            const curve = new Float32Array(1025);
            for (let i = 0; i < curve.length; i++) {
                const x = i / 512 - 1;
                curve[i] = OUTPUT_LIMIT * Math.tanh(x / OUTPUT_LIMIT);
            }
            this.clipper.curve = curve;
            this.analyser = ctx.createAnalyser();
            this.analyser.fftSize = 1024;
            this.meterData = new Float32Array(1024);

            this.inputBus.connect(this.dryDelay).connect(this.dryGain).connect(this.master);
            this.inputBus.connect(this.convolver).connect(this.convDelay).connect(this.convWet).connect(this.master);
            this.gpuWet.connect(this.master);
            this.master.connect(this.limiter).connect(this.clipper).connect(this.analyser).connect(ctx.destination);
            [this.dryGain, this.convWet, this.gpuWet, this.master].forEach(g => g.gain.value = 0);
        }
        this.resume();
        return this.ctx;
    }

    get sampleRate() {
        return this.ctx ? this.ctx.sampleRate : 48000;
    }

    resume() {
        if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    }

    suspend() {
        if (this.ctx && this.ctx.state === 'running') this.ctx.suspend().catch(() => {});
    }

    loadWorklet() {
        if (!this.ctx.audioWorklet) {
            return Promise.reject(new Error('AudioWorklet is not available (it needs a secure context: https or localhost)'));
        }
        if (!this.workletReady) {
            const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
            this.workletReady = this.ctx.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url));
        }
        return this.workletReady;
    }

    setImpulseResponse(left, right) {
        if (!this.ctx) return;
        const buffer = this.ctx.createBuffer(2, left.length, this.ctx.sampleRate);
        buffer.copyToChannel(left, 0);
        buffer.copyToChannel(right, 1);
        this.convolver.buffer = buffer;
    }

    // Worklet in, GPU wet out; dry and ConvolverNode paths delayed by the same latency
    openBridge(blockSize, depth) {
        this.closeBridge();
        const node = new AudioWorkletNode(this.ctx, 'reverb-bridge', {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            outputChannelCount: [2],
            channelCount: 2,
            channelCountMode: 'explicit',
            channelInterpretation: 'speakers',
            processorOptions: { blockSize, depth, ringSize: 16 }
        });
        this.inputBus.connect(node);
        node.connect(this.gpuWet);
        this.bridge = node;
        this.latency = (1 + depth) * blockSize / this.ctx.sampleRate;
        const now = this.ctx.currentTime;
        this.dryDelay.delayTime.setValueAtTime(this.latency, now);
        this.convDelay.delayTime.setValueAtTime(this.latency, now);
        this.updateMix();
        return node;
    }

    closeBridge() {
        if (!this.bridge) return;
        this.bridge.disconnect();
        try { this.inputBus.disconnect(this.bridge); } catch { /* not connected */ }
        this.bridge = null;
    }

    // dry = 1 - wet, wet to the GPU or ConvolverNode path, everything times the output gain
    updateMix() {
        if (!this.ctx) return;
        const t = this.ctx.currentTime;
        const { wet, gain, bypass, useGPU } = this.mix;
        const w = bypass ? 0 : wet;
        const running = !!this.bridge;
        this.dryGain.gain.setTargetAtTime(bypass ? 1 : 1 - wet, t, 0.02);
        this.gpuWet.gain.setTargetAtTime(useGPU ? w : 0, t, 0.02);
        this.convWet.gain.setTargetAtTime(useGPU ? 0 : w, t, 0.02);
        this.master.gain.setTargetAtTime(running ? Math.min(gain, OUTPUT_LIMIT) : 0, t, 0.02);
    }

    outputPeak() {
        if (!this.analyser) return 0;
        this.analyser.getFloatTimeDomainData(this.meterData);
        let peak = 0;
        for (let i = 0; i < this.meterData.length; i++) peak = Math.max(peak, Math.abs(this.meterData[i]));
        return peak;
    }

    stopSource() {
        this.request++;
        if (this.timer) clearInterval(this.timer);
        this.timer = null;
        for (const node of this.nodes) {
            try { if (node.stop) node.stop(); } catch { /* already stopped */ }
            node.disconnect();
        }
        this.nodes.clear();
        if (this.stream) this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
        this.kind = null;
        this.label = '';
    }

    noiseBuffer() {
        if (!this.noise) {
            const len = Math.round(0.25 * this.ctx.sampleRate);
            this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
            const data = this.noise.getChannelData(0);
            const rand = mulberry32(99);
            for (let i = 0; i < len; i++) data[i] = rand() * 2 - 1;
        }
        return this.noise;
    }

    scheduleEvent(type, value, t, bus) {
        const ctx = this.ctx;
        // The first node is the scheduled source; when it ends the whole note is released
        const track = (...nodes) => {
            nodes[0].onended = () => nodes.forEach((node) => {
                node.disconnect();
                this.nodes.delete(node);
            });
            nodes.forEach(node => this.nodes.add(node));
        };
        if (type === 'pluck') {
            const osc = ctx.createOscillator();
            osc.type = 'sawtooth';
            osc.frequency.value = value;
            const lp = ctx.createBiquadFilter();
            lp.type = 'lowpass';
            lp.frequency.setValueAtTime(4000, t);
            lp.frequency.exponentialRampToValueAtTime(500, t + 0.3);
            const env = ctx.createGain();
            env.gain.setValueAtTime(0.0001, t);
            env.gain.exponentialRampToValueAtTime(0.25, t + 0.004);
            env.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
            osc.connect(lp).connect(env).connect(bus);
            osc.start(t);
            osc.stop(t + 0.42);
            track(osc, lp, env);
        } else if (type === 'clap') {
            const src = ctx.createBufferSource();
            src.buffer = this.noiseBuffer();
            const bp = ctx.createBiquadFilter();
            bp.type = 'bandpass';
            bp.frequency.value = 1300;
            bp.Q.value = 0.9;
            const env = ctx.createGain();
            env.gain.setValueAtTime(0, t);
            for (const dt of [0, 0.011, 0.022]) {           // three quick bursts, then a decay
                env.gain.setValueAtTime(0.6, t + dt);
                env.gain.exponentialRampToValueAtTime(0.05, t + dt + 0.009);
            }
            env.gain.setValueAtTime(0.5, t + 0.033);
            env.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
            src.connect(bp).connect(env).connect(bus);
            src.start(t);
            src.stop(t + 0.22);
            track(src, bp, env);
        } else if (type === 'chord') {
            const env = ctx.createGain();
            env.gain.setValueAtTime(0.0001, t);
            env.gain.exponentialRampToValueAtTime(0.12, t + 0.006);
            env.gain.exponentialRampToValueAtTime(0.001, t + 0.55);
            env.connect(bus);
            const oscs = value.map((f) => {
                const osc = ctx.createOscillator();
                osc.type = 'triangle';
                osc.frequency.value = f;
                osc.connect(env);
                osc.start(t);
                osc.stop(t + 0.58);
                return osc;
            });
            track(...oscs, env);
        }
    }

    useSynth() {
        this.stopSource();
        const ctx = this.ctx;
        const bus = ctx.createGain();
        bus.connect(this.inputBus);
        this.nodes.add(bus);
        const start = ctx.currentTime + 0.1;
        let next = 0;                         // index of the next event, counting across loops
        const schedule = () => {
            const horizon = ctx.currentTime + SCHEDULE_AHEAD;
            for (let guard = 0; guard < 64; guard++) {
                const loop = Math.floor(next / SYNTH_EVENTS.length);
                const [dt, type, value] = SYNTH_EVENTS[next % SYNTH_EVENTS.length];
                const t = start + loop * SYNTH_LOOP + dt;
                if (t > horizon) break;
                if (t >= ctx.currentTime) this.scheduleEvent(type, value, t, bus);   // skip after a throttled timer
                next++;
            }
        };
        schedule();
        this.timer = setInterval(schedule, 25);
        this.kind = 'synth';
        this.label = 'Synth demo (plucks, claps and a chord with gaps, 8 s loop)';
    }

    async useMicrophone() {
        this.stopSource();
        const request = this.request;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Microphone access needs a secure context (https or localhost) and a browser with getUserMedia');
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
        if (request !== this.request) {
            stream.getTracks().forEach(track => track.stop());
            return false;
        }
        const source = this.ctx.createMediaStreamSource(stream);
        source.connect(this.inputBus);
        this.nodes.add(source);
        this.stream = stream;
        this.kind = 'mic';
        const track = stream.getAudioTracks()[0];
        this.label = `Microphone${track && track.label ? ` (${track.label})` : ''}`;
        return true;
    }

    async decodeFile(file) {
        const request = ++this.request;
        const buffer = await this.ctx.decodeAudioData(await file.arrayBuffer());
        if (request !== this.request) return false;
        this.fileBuffer = buffer;
        this.fileName = file.name;
        return true;
    }

    playFile() {
        this.stopSource();
        const source = this.ctx.createBufferSource();
        source.buffer = this.fileBuffer;
        source.loop = true;
        source.connect(this.inputBus);
        source.start();
        this.nodes.add(source);
        this.kind = 'file';
        this.label = `File "${this.fileName}" (${this.fileBuffer.duration.toFixed(1)} s, looping)`;
    }

    close() {
        this.stopSource();
        this.closeBridge();
        if (this.ctx) this.ctx.close().catch(() => {});
        this.ctx = null;
    }
}

// ---------------------------------------------------------------------------------------------
// Example: UI, visualization and the stream lifecycle
// ---------------------------------------------------------------------------------------------

class ConvolutionReverb {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.convolver = new GPUConvolver(device);
        this.audio = new ReverbAudio();
        this.running = false;
        this.starting = false;
        this.destroyed = false;
        this.stream = null;
        this.startToken = 0;
        this.lost = false;
        this.space = 'hall';
        this.rt60 = SPACES.hall.rt60;
        this.preDelay = SPACES.hall.preDelay;
        this.blockSize = 1024;
        this.depth = 2;
        this.ir = null;
        this.irInfo = null;
        this.uploaded = null;          // decoded IR file: { channels, name }
        this.selfTest = null;
        this.selfTestPending = false;
        this.liveCheck = null;         // first-block GPU vs CPU check of the current stream
        this.message = '';
        this.levels = new Float32Array(4);
        this.holds = new Float32Array(4);
        this.holdTimes = new Float64Array(5);
        this.outLevel = 0;
        this.outHold = 0;
        this.scopeData = new Float32Array(SCOPE_COLS * 4);
        this.scopeGains = [1, 1];
        this.hasScope = false;
        this.lastPeaks = [0, 0, 0, 0];

        // Audio keeps playing after a GPU device loss unless we stop it here
        device.lost.then((info) => {
            this.lost = true;
            if (info.reason !== 'destroyed') this.stop();
        });

        this.view = new ArrayBuffer(VIEW_SIZE);
        this.viewF32 = new Float32Array(this.view);
        this.viewU32 = new Uint32Array(this.view);
        this.setupRender();
        this.convolver.configure(this.blockSize);
    }

    setupRender() {
        const d = this.device;
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.envBuffer = d.createBuffer({ size: IR_COLS * 16, usage: storage });
        this.specBuffer = d.createBuffer({ size: IR_COLS * SPEC_ROWS * 4, usage: storage });
        this.edcBuffer = d.createBuffer({ size: IR_COLS * 4, usage: storage });
        this.scopeBuffer = d.createBuffer({ size: SCOPE_COLS * 16, usage: storage });
        this.viewBuffer = d.createBuffer({ size: VIEW_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const module = d.createShaderModule({ code: renderShader });
        const F = GPUShaderStage.FRAGMENT;
        const layout = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: F, buffer: { type: 'uniform' } }
            ]
        });
        this.renderPipeline = d.createRenderPipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
            vertex: { module, entryPoint: 'vs_main' },
            fragment: { module, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
            primitive: { topology: 'triangle-list' }
        });
        this.renderBindGroup = d.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.envBuffer } },
                { binding: 1, resource: { buffer: this.specBuffer } },
                { binding: 2, resource: { buffer: this.edcBuffer } },
                { binding: 3, resource: { buffer: this.scopeBuffer } },
                { binding: 4, resource: { buffer: this.viewBuffer } }
            ]
        });
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({ device: d, format: 'bgra8unorm' });
    }

    // Regenerates (or re-prepares) the IR at the current sample rate and uploads it everywhere
    rebuildIR() {
        const fs = this.audio.sampleRate;
        if (this.space === 'file' && this.uploaded) {
            this.ir = prepareUploadedIR(this.uploaded.channels, fs, this.preDelay);
        } else {
            this.ir = generateImpulseResponse(this.space, { sampleRate: fs, rt60: this.rt60, preDelay: this.preDelay });
        }
        this.irInfo = analyzeImpulseResponse(this.ir.left, this.ir.right, fs);
        const q = this.device.queue;
        q.writeBuffer(this.envBuffer, 0, this.irInfo.env);
        q.writeBuffer(this.specBuffer, 0, this.irInfo.spec);
        q.writeBuffer(this.edcBuffer, 0, this.irInfo.edc);
        this.audio.setImpulseResponse(this.ir.left, this.ir.right);
        const { left, right } = this.ir;
        this.convolver.enqueue(() => this.convolver.uploadIR(left, right, fs));
        this.selfTestPending = true;
        if (!this.running && !this.starting) this.runSelfTest();
        this.render();
    }

    runSelfTest() {
        if (!this.selfTestPending || this.lost) return this.convolver.chain;
        this.selfTestPending = false;
        return this.convolver.enqueue(() => this.convolver.selfTest()).then((result) => {
            this.selfTest = result;
            if (this.onStats) this.onStats();
        }, (error) => {
            this.message = `<span class="error">GPU self-test failed: ${error.message}</span>`;
            if (this.onStats) this.onStats();
        });
    }

    setBlockSize(b) {
        if (!BLOCK_SIZES.includes(b) || b === this.blockSize) return;
        const wasRunning = this.running;
        if (wasRunning) this.stopStream();
        this.blockSize = b;
        this.convolver.enqueue(() => this.convolver.configure(b));
        this.selfTestPending = true;
        if (wasRunning) this.startStream(); else this.runSelfTest();
    }

    // Opens the worklet and starts streaming; the AudioContext already exists (user gesture)
    async startStream() {
        if (this.running || this.starting || this.lost) return;
        this.starting = true;
        const token = this.startToken;
        try {
            await this.audio.loadWorklet();
            if (this.ir.sampleRate !== this.audio.sampleRate) this.rebuildIR();
            await this.runSelfTest();
            await this.convolver.enqueue(() => this.convolver.resetStream());
            if (this.destroyed || this.lost || token !== this.startToken) return;   // stopped meanwhile
            const node = this.audio.openBridge(this.blockSize, this.depth);
            this.stream = new ReverbStream(this.convolver, node.port, (block, wet, irVersion) => this.onBlock(block, wet, irVersion));
            this.stream.onError = (error) => {
                this.message = `<span class="error">${error.message}</span>`;
                this.stop();
            };
            this.liveCheck = { state: 'waiting' };
            this.running = true;
            this.audio.updateMix();
            this.animate();
        } finally {
            this.starting = false;
        }
    }

    stopStream() {
        if (this.stream) this.stream.close();
        this.stream = null;
        this.audio.closeBridge();
        this.running = false;
        this.audio.updateMix();
    }

    stop() {
        this.startToken++;
        this.stopStream();
        this.audio.stopSource();
        this.audio.suspend();
        this.levels.fill(0);
        this.outLevel = 0;
        this.render();
        if (this.selfTestPending && !this.destroyed) this.runSelfTest();
        if (this.onStats) this.onStats();
    }

    // A wet block came back: meters, scope, and the GPU-vs-CPU check of the first non-silent block
    onBlock(block, wet, irVersion) {
        const peak = (a) => { let p = 0; for (let i = 0; i < a.length; i++) p = Math.max(p, Math.abs(a[i])); return p; };
        this.lastPeaks = [peak(block.left), peak(block.right), peak(wet.left), peak(wet.right)];
        const B = block.left.length;
        let inPeak = 1e-6, outPeak = 1e-6;
        for (let c = 0; c < SCOPE_COLS; c++) {
            const a = Math.floor(c * B / SCOPE_COLS);
            const b = Math.max(Math.floor((c + 1) * B / SCOPE_COLS), a + 1);
            let lo0 = Infinity, hi0 = -Infinity, lo1 = Infinity, hi1 = -Infinity;
            for (let i = a; i < b; i++) {
                const x = 0.5 * (block.left[i] + block.right[i]);
                const y = 0.5 * (wet.left[i] + wet.right[i]);
                lo0 = Math.min(lo0, x); hi0 = Math.max(hi0, x);
                lo1 = Math.min(lo1, y); hi1 = Math.max(hi1, y);
            }
            this.scopeData.set([lo0, hi0, lo1, hi1], c * 4);
            inPeak = Math.max(inPeak, -lo0, hi0);
            outPeak = Math.max(outPeak, -lo1, hi1);
        }
        this.scopeGains = [Math.min(0.9 / inPeak, 50), Math.min(0.9 / outPeak, 50)];
        this.hasScope = true;
        this.scopeDirty = true;

        // Live check: from the first non-silent input block on (everything before it was exactly
        // zero, so the delay line holds nothing else), keep the input and compare the first wet
        // block that is not silent (the pre-delay can push it a few blocks later) with a direct
        // CPU convolution
        const check = this.liveCheck;
        if (!check || (check.state !== 'waiting' && check.state !== 'collecting')) return;
        if (check.state === 'waiting') {
            if (this.lastPeaks[0] === 0 && this.lastPeaks[1] === 0) return;
            Object.assign(check, { state: 'collecting', first: block.index, left: [], right: [], irVersion });
        }
        if (irVersion !== check.irVersion || irVersion !== this.convolver.irVersion ||
            block.index !== check.first + check.left.length) {
            this.liveCheck = { state: 'skipped' };
            return;
        }
        check.left.push(block.left.slice());
        check.right.push(block.right.slice());
        const m = check.left.length - 1;
        let maxError = 0, ref = 0;
        for (const [x, y, h] of [[check.left, wet.left, this.ir.left], [check.right, wet.right, this.ir.right]]) {
            const r = directBlock(x, h, m);
            for (let i = 0; i < B; i++) {
                maxError = Math.max(maxError, Math.abs(y[i] - r[i]));
                ref = Math.max(ref, Math.abs(r[i]));
            }
        }
        if (ref > 0) {
            this.liveCheck = { state: 'done', index: block.index, maxError, peak: ref, relative: maxError / ref };
        } else if (m + 1 >= LIVE_CHECK_BLOCKS) {
            this.liveCheck = { state: 'skipped' };
        }
    }

    updateMeters() {
        const now = performance.now();
        const decay = 0.88;
        for (let i = 0; i < 4; i++) {
            this.levels[i] = Math.max(this.lastPeaks[i], this.levels[i] * decay);
            if (this.levels[i] >= this.holds[i]) {
                this.holds[i] = this.levels[i];
                this.holdTimes[i] = now;
            } else if (now - this.holdTimes[i] > 1200) {
                this.holds[i] *= 0.94;
            }
        }
        this.lastPeaks = this.lastPeaks.map(v => v * 0.5);
        const out = this.audio.outputPeak();
        this.outLevel = Math.max(out, this.outLevel * decay);
        if (this.outLevel >= this.outHold) {
            this.outHold = this.outLevel;
            this.holdTimes[4] = now;
        } else if (now - this.holdTimes[4] > 1200) {
            this.outHold *= 0.94;
        }
    }

    // Writes struct View; layout table above renderShader
    writeView() {
        const f = this.viewF32;
        const info = this.irInfo;
        f[0] = this.canvas.width;
        f[1] = this.canvas.height;
        f[2] = info ? info.seconds : 1;
        f[3] = info && info.rt60 && this.space !== 'reverse' ? info.rt60 : 0;
        f.set(this.levels, 4);
        f.set(this.holds, 8);
        f[12] = this.outLevel;
        f[13] = this.outHold;
        f[14] = this.scopeGains[0];
        f[15] = this.scopeGains[1];
        f[16] = this.preDelay;
        this.viewU32[17] = this.hasScope ? 1 : 0;
        this.device.queue.writeBuffer(this.viewBuffer, 0, this.view);
    }

    render() {
        if (this.destroyed) return;
        if (this.scopeDirty) {
            this.device.queue.writeBuffer(this.scopeBuffer, 0, this.scopeData);
            this.scopeDirty = false;
        }
        this.writeView();
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.02, g: 0.027, b: 0.055, a: 1.0 }
            }]
        });
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.renderBindGroup);
        pass.draw(3);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    animate() {
        if (!this.running || this.destroyed) return;
        this.updateMeters();
        this.render();
        const now = performance.now();
        if (!this.lastStats || now - this.lastStats > 500) {
            this.lastStats = now;
            if (this.onStats) this.onStats();
        }
        requestAnimationFrame(() => this.animate());
    }

    destroy() {
        this.destroyed = true;
        this.stopStream();
        this.audio.close();
        this.convolver.destroy();
        [this.envBuffer, this.specBuffer, this.edcBuffer, this.scopeBuffer, this.viewBuffer].forEach(b => b.destroy());
    }
}

const DEPTHS = [1, 2, 3, 5];     // slack blocks D; total latency (1 + D)·B

export async function initConvolutionReverb(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-reverb');
    const output = document.getElementById('output-reverb');
    const reverb = new ConvolutionReverb(device, canvas);
    const audio = reverb.audio;
    reverb.onStats = () => updateOutput();

    const $ = (id) => document.getElementById(id);
    const startButton = $('btn-reverb-start');
    const stopButton = $('btn-reverb-stop');
    const synthButton = $('btn-reverb-synth');
    const micButton = $('btn-reverb-mic');
    const fileButton = $('btn-reverb-file');
    const fileInput = $('reverb-file');
    const bypassButton = $('btn-reverb-bypass');
    const abButton = $('btn-reverb-ab');
    const irButton = $('btn-reverb-ir');
    const irInput = $('reverb-ir-file');
    const decaySlider = $('reverb-decay');
    const preDelaySlider = $('reverb-predelay');
    const mixSlider = $('reverb-mix');
    const gainSlider = $('reverb-gain');
    const blockSelect = $('reverb-block');
    const depthSelect = $('reverb-depth');
    const spaceButtons = document.querySelectorAll('.reverb-space');
    let lastKind = 'synth';

    function fail(error) {
        reverb.message = `<span class="error">${error.message || error}</span>`;
        updateOutput();
    }

    // Creates/resumes the AudioContext synchronously inside the click gesture
    function withContext() {
        try {
            const fresh = !audio.ctx;
            audio.ensureContext();
            if (fresh) {
                // The IR was made at 48 kHz before the context existed; ConvolverNode needs its rate
                if (reverb.ir.sampleRate === audio.sampleRate) audio.setImpulseResponse(reverb.ir.left, reverb.ir.right);
                else reverb.rebuildIR();
            }
            return true;
        } catch (error) {
            fail(error);
            return false;
        }
    }

    async function startWith(kind) {
        reverb.message = '';
        try {
            if (kind === 'mic') {
                reverb.message = '<span class="info">Waiting for microphone permission…</span>';
                updateOutput();
                // Lower the output for a live mic (feedback through speakers): use headphones
                if (audio.mix.gain > 0.2) {
                    audio.mix.gain = 0.2;
                    showControls();
                }
                const ok = await audio.useMicrophone();
                if (!ok) return;
                reverb.message = '';
            } else if (kind === 'file' && audio.fileBuffer) {
                audio.playFile();
            } else {
                audio.useSynth();
                kind = 'synth';
            }
            lastKind = kind;
            await reverb.startStream();
        } catch (error) {
            const denied = error.name === 'NotAllowedError' || error.name === 'SecurityError';
            fail(denied ? new Error('Microphone permission was denied. Allow it in the site settings, or use the synth demo or a file.') : error);
        }
        showControls();
        updateOutput();
    }

    startButton.addEventListener('click', () => {
        if (reverb.running || reverb.starting) return;
        if (!withContext()) return;
        startWith(lastKind);
    });

    stopButton.addEventListener('click', () => {
        reverb.stop();
        showControls();
        updateOutput();
    });

    synthButton.addEventListener('click', () => {
        if (!withContext()) return;
        startWith('synth');
    });

    micButton.addEventListener('click', () => {
        if (!withContext()) return;
        startWith('mic');
    });

    fileButton.addEventListener('click', () => {
        if (!withContext()) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files && fileInput.files[0];
        fileInput.value = '';
        if (!file) return;
        reverb.message = `<span class="info">Decoding "${file.name}"…</span>`;
        updateOutput();
        try {
            if (!(await audio.decodeFile(file))) return;
            audio.resume();
            await startWith('file');
        } catch (error) {
            fail(new Error(`Could not decode "${file.name}": ${error.message || error}`));
        }
    });

    bypassButton.addEventListener('click', () => {
        audio.mix.bypass = !audio.mix.bypass;
        audio.updateMix();
        showControls();
        updateOutput();
    });

    abButton.addEventListener('click', () => {
        audio.mix.useGPU = !audio.mix.useGPU;
        audio.updateMix();
        showControls();
        updateOutput();
    });

    spaceButtons.forEach(btn => btn.addEventListener('click', () => {
        const key = btn.dataset.space;
        if (!SPACES[key]) return;
        reverb.space = key;
        reverb.rt60 = SPACES[key].rt60;
        reverb.preDelay = SPACES[key].preDelay;
        reverb.rebuildIR();
        showControls();
        updateOutput();
    }));

    irButton.addEventListener('click', () => {
        if (!withContext()) return;   // decoding needs the context (resamples to its rate)
        irInput.click();
    });

    irInput.addEventListener('change', async () => {
        const file = irInput.files && irInput.files[0];
        irInput.value = '';
        if (!file) return;
        try {
            const buffer = await audio.ctx.decodeAudioData(await file.arrayBuffer());
            const channels = [buffer.getChannelData(0)];
            if (buffer.numberOfChannels > 1) channels.push(buffer.getChannelData(1));
            reverb.uploaded = { channels, name: file.name, duration: buffer.duration, count: buffer.numberOfChannels };
            reverb.space = 'file';
            reverb.preDelay = 0;
            reverb.message = '';
            reverb.rebuildIR();
        } catch (error) {
            fail(new Error(`Could not decode the IR "${file.name}": ${error.message || error}`));
        }
        showControls();
        updateOutput();
    });

    decaySlider.addEventListener('input', () => {
        $('reverb-decay-value').textContent = `${parseFloat(decaySlider.value).toFixed(1)} s`;
    });
    decaySlider.addEventListener('change', () => {
        reverb.rt60 = Math.min(Math.max(parseFloat(decaySlider.value), 0.2), 5);
        reverb.rebuildIR();
        showControls();
        updateOutput();
    });
    preDelaySlider.addEventListener('input', () => {
        $('reverb-predelay-value').textContent = `${preDelaySlider.value} ms`;
    });
    preDelaySlider.addEventListener('change', () => {
        reverb.preDelay = Math.min(Math.max(parseFloat(preDelaySlider.value), 0), 100) / 1000;
        reverb.rebuildIR();
        showControls();
        updateOutput();
    });
    mixSlider.addEventListener('input', () => {
        audio.mix.wet = Math.min(Math.max(parseFloat(mixSlider.value), 0), 100) / 100;
        audio.updateMix();
        showControls();
    });
    gainSlider.addEventListener('input', () => {
        audio.mix.gain = Math.min(Math.max(parseFloat(gainSlider.value), 0), OUTPUT_LIMIT);
        audio.updateMix();
        showControls();
    });
    blockSelect.addEventListener('change', () => {
        reverb.setBlockSize(parseInt(blockSelect.value));
        updateOutput();
    });
    depthSelect.addEventListener('change', () => {
        const d = parseInt(depthSelect.value);
        if (!DEPTHS.includes(d)) return;
        reverb.depth = d;
        if (reverb.running) {
            reverb.stopStream();
            reverb.startStream().catch(fail);
        }
        updateOutput();
    });

    function showControls() {
        decaySlider.value = reverb.rt60;
        decaySlider.disabled = reverb.space === 'file';
        $('reverb-decay-value').textContent = `${reverb.rt60.toFixed(1)} s`;
        preDelaySlider.value = Math.round(reverb.preDelay * 1000);
        $('reverb-predelay-value').textContent = `${Math.round(reverb.preDelay * 1000)} ms`;
        mixSlider.value = Math.round(audio.mix.wet * 100);
        $('reverb-mix-value').textContent = `${Math.round(audio.mix.wet * 100)}% wet`;
        gainSlider.value = audio.mix.gain;
        $('reverb-gain-value').textContent = audio.mix.gain > 0 ? `${(20 * Math.log10(audio.mix.gain)).toFixed(1)} dB` : 'off';
        blockSelect.value = String(reverb.blockSize);
        depthSelect.value = String(reverb.depth);
        bypassButton.textContent = audio.mix.bypass ? 'Bypass: on' : 'Bypass: off';
        bypassButton.classList.toggle('active', audio.mix.bypass);
        abButton.textContent = audio.mix.useGPU ? 'A/B: GPU' : 'A/B: ConvolverNode';
        abButton.classList.toggle('active', !audio.mix.useGPU);
        synthButton.classList.toggle('active', audio.kind === 'synth');
        micButton.classList.toggle('active', audio.kind === 'mic');
        fileButton.classList.toggle('active', audio.kind === 'file');
        spaceButtons.forEach(b => b.classList.toggle('active', b.dataset.space === reverb.space));
        irButton.classList.toggle('active', reverb.space === 'file');
    }

    function updateOutput() {
        const c = reverb.convolver;
        const fs = audio.sampleRate;
        const B = reverb.blockSize;
        const info = reverb.irInfo;
        const s = reverb.stream;
        const st = s ? s.stats : null;
        const ms = (frames) => (1000 * frames / fs).toFixed(1);
        const spaceName = reverb.space === 'file' && reverb.uploaded
            ? `uploaded "${reverb.uploaded.name}" (${reverb.uploaded.count} ch${reverb.ir && reverb.ir.truncated ? ', truncated to 6 s' : ''})`
            : SPACES[reverb.space].name;
        let rt = '—';
        if (reverb.space === 'reverse') rt = 'n/a (reversed: the tail swells instead of decaying)';
        else if (info && info.rt60) rt = `${info.rt60.toFixed(2)} s (T30 of the Schroeder curve)`;
        const t = reverb.selfTest;
        const stale = t && t.irVersion !== c.irVersion;
        const selfLine = t
            ? `${t.maxError.toExponential(2)} max |GPU − CPU| = ${t.relative.toExponential(2)} of the peak over blocks ${t.checked.join(', ')} of ${t.blocks}${stale ? ' (previous IR; re-measured when stopped)' : ''}`
            : 'running…';
        const speed = t ? `${t.realtime.toFixed(0)}× realtime (${t.blocks} blocks in ${t.ms.toFixed(0)} ms incl. submit and readback)` : '—';
        const lc = reverb.liveCheck;
        let liveLine = 'after Start';
        if (lc && (lc.state === 'waiting' || lc.state === 'collecting')) liveLine = 'waiting for the first non-silent wet block';
        else if (lc && lc.state === 'skipped') liveLine = 'skipped (IR changed, or no wet signal within 12 blocks)';
        else if (lc && lc.state === 'done') liveLine = `block ${lc.index}: ${lc.maxError.toExponential(2)} max |GPU − CPU| = ${lc.relative.toExponential(2)} of the peak`;
        const rtAvg = s && s.roundTrip.count ? (s.roundTrip.sum / s.roundTrip.count).toFixed(1) : '—';
        const rtMax = s && s.roundTrip.count ? s.roundTrip.max.toFixed(1) : '—';
        const macs = c.partitions * (B + 1);

        output.innerHTML = `<span class="success">✓ Convolution Reverb Ready</span>
${reverb.message ? `\n${reverb.message}\n` : ''}
<span class="info">Configuration:</span>
• Source: ${audio.kind ? audio.label : 'none (press Start, Synth demo, Microphone or Audio file)'}
• Sample rate: ${audio.ctx ? `${fs.toLocaleString()} Hz` : '48,000 Hz assumed until audio starts'}
• Space: ${spaceName} · IR ${info ? info.seconds.toFixed(2) : '—'} s stereo · pre-delay ${Math.round(reverb.preDelay * 1000)} ms · RT60 ${rt}
• Partitions: P = ${c.partitions} × B = ${B} samples, FFT N = ${2 * B}, delay line ${c.ring} spectra
• Latency: ${ms((1 + reverb.depth) * B)} ms = ${B} (partition) + ${reverb.depth} × ${B} (GPU slack); dry and ConvolverNode paths are delayed to match
• Output: ${audio.mix.bypass ? 'bypass (dry only)' : `${Math.round(audio.mix.wet * 100)}% wet through ${audio.mix.useGPU ? 'the GPU convolver' : 'the browser ConvolverNode'}`} · gain ${audio.mix.gain.toFixed(2)} (max ${OUTPUT_LIMIT}), limiter + soft clip
• Status: ${reverb.running ? '<span class="success">Streaming</span>' : '<span class="info">Stopped</span>'}

<span class="info">Accuracy (GPU vs direct CPU convolution):</span>
• Self-test (noise block through every partition): ${selfLine}
• Live stream, first non-silent wet block: ${liveLine}

<span class="info">Performance:</span>
• Self-test throughput: ${speed}
• Stream: ${s ? s.processed : 0} blocks · round trip avg ${rtAvg} ms, max ${rtMax} ms (deadline ${ms(reverb.depth * B)} ms) · backlog max ${s ? s.maxQueue : 0}
• Underruns: ${st ? st.underruns : 0} blocks played as silence (${st ? st.late : 0} arrived late)
• GPU work per block: ${(2 * B).toLocaleString()}-point FFT + inverse, ${macs.toLocaleString()} bins × 2 channels of complex multiply-add, ${4 + 2 * c.log2n} dispatches; ${(B * 8 / 1024).toFixed(0)} KB up, ${(B * 8 / 1024).toFixed(0)} KB read back

<span class="info">Algorithm (uniformly partitioned overlap-save, Gardner 1995 / Wefers 2015):</span>
• IR split into P blocks of B samples, each zero-padded to 2B and FFT'd once (batched dispatches)
• Per block: FFT of [previous, current] block → delay line; Y = Σⱼ X₍ₖ₋ⱼ₎·Hⱼ in one dispatch; inverse FFT; keep the last B samples
• Stereo in one complex FFT: z = L + i·R, split with Z[k] and conj(Z[N−k]); Y = Y_L + i·Y_R`;
    }

    reverb.rebuildIR();
    showControls();
    updateOutput();
    return reverb;
}

// Exported for the CPU tests (test/convolution-reverb.test.mjs): the View uniform writer.
export { ConvolutionReverb };

// Every WGSL module this example compiles, exactly as passed to createShaderModule.
// tools/validate-wgsl.mjs validates these in CI.
export function wgslModules() {
    return [
        { name: 'compute', code: computeShader },
        { name: 'render', code: renderShader },
    ];
}
