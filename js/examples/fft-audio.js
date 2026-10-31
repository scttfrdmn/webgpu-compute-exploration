// Real-time audio spectrum analyzer: the Web Audio API supplies time-domain samples, and the GPU
// does everything else (Hann window, radix-2 FFT, magnitudes in dB, log-frequency bands, the
// spectrogram history and beat detection). The CPU only copies N samples per frame to the GPU.
//
// Uniform layout (struct Params, 56 bytes; shared by the compute and render shaders):
//   offset  0  n          u32        FFT size N (1024, 2048 or 4096)      -> Uint32 view [0]
//   offset  4  log2n      u32        log2(N), the number of FFT stages    -> Uint32 view [1]
//   offset  8  stage      u32        butterfly stage of this dispatch     -> Uint32 view [2]
//   offset 12  column     u32        spectrogram column written / newest  -> Uint32 view [3]
//   offset 16  sampleRate f32        Hz                                   -> Float32 view [4]
//   offset 20  smoothing  f32        0..0.95, exponential magnitude blend -> Float32 view [5]
//   offset 24  minDb      f32        bottom of the displayed dB range     -> Float32 view [6]
//   offset 28  maxDb      f32        top of the displayed dB range        -> Float32 view [7]
//   offset 32  mode       u32        0 bars, 1 waveform, 2 spectrogram    -> Uint32 view [8]
//   offset 36  trigger    u32        waveform start sample (zero crossing) -> Uint32 view [9]
//   offset 40  canvasSize vec2<f32>  canvas width, height in pixels        -> Float32 view [10], [11]
//   offset 48  waveGain   f32        waveform vertical gain                -> Float32 view [12]
//   offset 52  (padding to the struct's 8-byte alignment)
// One copy per FFT stage sits at a dynamic offset of paramsStride bytes, so each butterfly
// dispatch sees its own `stage` without a CPU round trip; the other passes use slot 0.
const PARAMS_SIZE = 56;
const MAX_N = 4096;
const MAX_LOG2N = 12;
const NUM_ROWS = 256;        // log-frequency rows (bands) from F_LO to F_HI
const SPEC_COLUMNS = 512;    // spectrogram history: one column per frame
const NUM_BARS = 64;         // bars mode: each bar is the max of NUM_ROWS / NUM_BARS rows
const ANALYSIS_HEADER = 4;   // f32 words before the bands in struct Analysis
const SILENT_DB = -200;      // "no data" level for empty bands and history
const F_LO = 20;
const F_HI = 20000;
const WORKGROUP = 64;

const commonWgsl = `
struct Params {
    n: u32,
    log2n: u32,
    stage: u32,
    column: u32,
    sampleRate: f32,
    smoothing: f32,
    minDb: f32,
    maxDb: f32,
    mode: u32,
    trigger: u32,
    canvasSize: vec2<f32>,
    waveGain: f32,
}

// Beat detector state followed by the log-frequency bands (one dB value per row)
struct Analysis {
    energy: f32,    // low-band energy of this frame
    average: f32,   // running average of the low-band energy
    pulse: f32,     // 1 on a detected beat, decays every frame
    above: f32,     // 1 while energy > BEAT_RATIO * average (beats fire on the rising edge)
    bands: array<f32, ${NUM_ROWS}>,
}

const NUM_ROWS: u32 = ${NUM_ROWS}u;
const SPEC_COLUMNS: u32 = ${SPEC_COLUMNS}u;
const F_LO: f32 = ${F_LO.toFixed(1)};
const F_HI: f32 = ${F_HI.toFixed(1)};
const SILENT_DB: f32 = ${SILENT_DB.toFixed(1)};
`;

const computeShader = commonWgsl + `
@group(0) @binding(0) var<storage, read> samples: array<f32>;          // N time-domain samples
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;         // complex input of a stage
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;   // complex output of a stage
@group(0) @binding(3) var<storage, read> twiddles: array<vec2<f32>>;    // W_N^k = e^(-2πik/N), k < N/2
@group(0) @binding(4) var<storage, read_write> spectrum: array<vec2<f32>>; // (smoothed |X|, dB) per bin
@group(0) @binding(5) var<storage, read_write> analysis: Analysis;
@group(0) @binding(6) var<storage, read_write> spectrogram: array<f32>;    // SPEC_COLUMNS x NUM_ROWS dB
@group(0) @binding(7) var<uniform> params: Params;

const PI: f32 = 3.14159265358979;
const DB_PER_LOG2: f32 = 6.02059991;   // 20·log10(x) = 6.0206·log2(x)
const MAX_ROW_BINS: u32 = 128u;        // widest row: 20 kHz·(1 - 2^(-log2(1000)/256)) ≈ 535 Hz ≈ 46 bins at N = 4096
const BEAT_LO_HZ: f32 = 40.0;          // kick-drum band
const BEAT_HI_HZ: f32 = 150.0;
const MAX_BEAT_BINS: u32 = 64u;
const BEAT_RATIO: f32 = 1.6;           // a beat is energy rising above 1.6x its running average...
const BEAT_MIN_ENERGY: f32 = 0.0001;   // ...and above -40 dB, so silence and hiss never trigger
const AVERAGE_RATE: f32 = 0.03;        // running average: about the last 33 frames (~0.5 s)
const PULSE_DECAY: f32 = 0.88;         // beat flash fades to 10% in ~18 frames

// Stage 0: Hann window w[n] = 0.5 - 0.5·cos(2πn/N) (periodic form; reduces spectral leakage),
// written to the bit-reversed index. Iterative radix-2 Cooley-Tukey (decimation in time) needs its
// input in bit-reversed order so that each butterfly stage can work on contiguous blocks; the
// output then comes out in natural order.
@compute @workgroup_size(${WORKGROUP})
fn cs_window(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let w = 0.5 - 0.5 * cos(2.0 * PI * f32(i) / f32(params.n));
    let j = reverseBits(i) >> (32u - params.log2n);
    dst[j] = vec2<f32>(samples[i] * w, 0.0);
}

// Butterfly stage s (one dispatch each, s = 0 .. log2(N) - 1, ping-ponging src/dst). Blocks of
// size 2h (h = 2^s) are combined from two DFTs of size h. Invocation t handles one pair:
//   j = t mod h, i0 = (t div h)·2h + j, i1 = i0 + h, w = W_{2h}^j = W_N^(j·N/2h)
//   X[i0] = a + w·b,  X[i1] = a - w·b      (a = src[i0], b = src[i1])
// The twiddles come from a table computed once on the CPU in double precision, which is more
// accurate than f32 sin/cos of large arguments.
@compute @workgroup_size(${WORKGROUP})
fn cs_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.n / 2u) {
        return;
    }
    let h = 1u << params.stage;
    let j = t & (h - 1u);
    let i0 = ((t >> params.stage) << (params.stage + 1u)) + j;
    let i1 = i0 + h;
    let w = twiddles[j << (params.log2n - 1u - params.stage)];
    let a = src[i0];
    let b = src[i1];
    let wb = vec2<f32>(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
    dst[i0] = a + wb;
    dst[i1] = a - wb;
}

// Magnitude of bins 0 .. N/2, scaled by 4/N (2/N for the one-sided spectrum, 2 for the Hann
// window's coherent gain of 0.5) so a full-scale sine reads 0 dB. Smoothing blends with the
// previous frame like AnalyserNode.smoothingTimeConstant does, then converts to dB.
@compute @workgroup_size(${WORKGROUP})
fn cs_magnitude(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k > params.n / 2u) {
        return;
    }
    let mag = length(src[k]) * 4.0 / f32(params.n);
    let blended = params.smoothing * spectrum[k].x + (1.0 - params.smoothing) * mag;
    spectrum[k] = vec2<f32>(blended, DB_PER_LOG2 * log2(max(blended, 1e-10)));
}

// Resamples the dB spectrum onto NUM_ROWS log-spaced rows from F_LO to F_HI: a row takes the
// loudest bin inside its frequency range, or, when the row is narrower than one bin (low
// frequencies), the dB spectrum interpolated at the row's center. Writes the bands (for the bars)
// and one column of the spectrogram ring buffer. Rows are fixed in Hz, so history survives FFT
// size changes.
@compute @workgroup_size(${WORKGROUP})
fn cs_bands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= NUM_ROWS) {
        return;
    }
    let half = params.n / 2u;
    let nyquistBin = f32(half);
    let binHz = params.sampleRate / f32(params.n);
    let ratio = F_HI / F_LO;
    let bLo = F_LO * pow(ratio, f32(r) / f32(NUM_ROWS)) / binHz;
    let bHi = min(F_LO * pow(ratio, f32(r + 1u) / f32(NUM_ROWS)) / binHz, nyquistBin);

    var db = SILENT_DB;
    if (bLo < nyquistBin) {
        let kStart = u32(ceil(bLo));
        let kEnd = u32(floor(bHi));
        if (kEnd < kStart) {
            let c = sqrt(bLo * bHi);
            let k0 = min(u32(floor(c)), half);
            let k1 = min(k0 + 1u, half);
            db = mix(spectrum[k0].y, spectrum[k1].y, c - floor(c));
        } else {
            for (var i = 0u; i < MAX_ROW_BINS; i++) {
                let k = kStart + i;
                if (k > kEnd) {
                    break;
                }
                db = max(db, spectrum[k].y);
            }
        }
    }
    analysis.bands[r] = db;
    spectrogram[params.column * NUM_ROWS + r] = db;
}

// Beat detection, one invocation: energy E = sum of |X|² over the 40-150 Hz bins of this frame
// (unsmoothed), compared against its running average. A beat fires when E crosses
// BEAT_RATIO x average upward (rising edge, so one kick gives one beat at any frame rate) and E
// is above BEAT_MIN_ENERGY. The flash intensity lives in analysis.pulse and is read directly by
// the fragment shader, so the CPU never reads anything back.
@compute @workgroup_size(1)
fn cs_beat() {
    let binHz = params.sampleRate / f32(params.n);
    let kStart = max(u32(ceil(BEAT_LO_HZ / binHz)), 1u);
    let kEnd = max(u32(floor(BEAT_HI_HZ / binHz)), kStart);
    var energy = 0.0;
    for (var i = 0u; i < MAX_BEAT_BINS; i++) {
        let k = kStart + i;
        if (k > kEnd) {
            break;
        }
        let m = length(src[k]) * 4.0 / f32(params.n);
        energy += m * m;
    }
    let avg = analysis.average;
    let isAbove = energy > BEAT_RATIO * avg && energy > BEAT_MIN_ENERGY;
    var pulse = analysis.pulse * PULSE_DECAY;
    if (isAbove && analysis.above == 0.0) {
        pulse = 1.0;
    }
    analysis.energy = energy;
    analysis.average = avg + AVERAGE_RATE * (energy - avg);
    analysis.pulse = pulse;
    analysis.above = select(0.0, 1.0, isAbove);
}
`;

const renderShader = commonWgsl + `
@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read> analysis: Analysis;
@group(0) @binding(2) var<storage, read> spectrogram: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const NUM_BARS: u32 = ${NUM_BARS}u;
const ROWS_PER_BAR: u32 = ${NUM_ROWS / NUM_BARS}u;
const MAX_WAVE_SPAN: u32 = 16u;    // samples per pixel column is at most N/2 / width = 2
const BACKGROUND: vec3<f32> = vec3<f32>(0.020, 0.027, 0.055);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle that covers the whole screen; uv (0,0) is the top-left corner
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

fn level(db: f32) -> f32 {
    return clamp((db - params.minDb) / (params.maxDb - params.minDb), 0.0, 1.0);
}

// Position of frequency f on the log axis, 0 at F_LO, 1 at F_HI
fn logPosition(f: f32) -> f32 {
    return log(f / F_LO) / log(F_HI / F_LO);
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

// Black -> purple -> red -> orange -> pale yellow (an inferno-like heat map)
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

// 1 within half a pixel of a vertical line at 100 Hz, 1 kHz and 10 kHz
fn decadeLine(x: f32) -> f32 {
    var hit = 0.0;
    for (var d = 0u; d < 3u; d++) {
        let f = 100.0 * pow(10.0, f32(d));
        if (abs(x - logPosition(f)) * params.canvasSize.x < 0.75) {
            hit = 1.0;
        }
    }
    return hit;
}

// Bars: NUM_BARS log-spaced bands, height = dB level, hue from red (bass) to violet (treble)
fn drawBars(uv: vec2<f32>) -> vec3<f32> {
    let px = uv * params.canvasSize;
    let y = 1.0 - uv.y;
    var color = BACKGROUND;

    // Grid: decade lines and a horizontal line every 20 dB
    let dbHere = params.minDb + y * (params.maxDb - params.minDb);
    let dbLine = abs(dbHere - round(dbHere / 20.0) * 20.0) / (params.maxDb - params.minDb) * params.canvasSize.y;
    if (decadeLine(uv.x) > 0.0 || dbLine < 0.6) {
        color = vec3<f32>(0.10, 0.13, 0.20);
    }

    let fb = uv.x * f32(NUM_BARS);
    let bar = min(u32(fb), NUM_BARS - 1u);
    let local = fb - f32(bar);
    var db = SILENT_DB;
    for (var i = 0u; i < ROWS_PER_BAR; i++) {
        db = max(db, analysis.bands[bar * ROWS_PER_BAR + i]);
    }
    let h = level(db);
    if (local > 0.12 && local < 0.88 && y <= h) {
        let hue = 0.8 * f32(bar) / f32(NUM_BARS - 1u);
        let shade = 0.45 + 0.55 * y / max(h, 0.001);
        color = hsv2rgb(hue, 0.85, shade);
    }
    return color;
}

// Waveform: N/2 samples from the trigger point (a rising zero crossing, so periodic signals stand
// still), drawn as the min-max span of the samples under each pixel column plus a soft glow
fn drawWaveform(uv: vec2<f32>) -> vec3<f32> {
    let px = uv * params.canvasSize;
    var color = BACKGROUND;
    if (abs(px.y - 0.5 * params.canvasSize.y) < 0.5) {
        color = vec3<f32>(0.10, 0.13, 0.20);
    }

    let span = f32(params.n / 2u);
    let perPixel = span / params.canvasSize.x;
    let column = floor(px.x);
    let a = u32(column * perPixel);
    let b = max(u32((column + 1.0) * perPixel), a + 1u);
    var lo = 1e9;
    var hi = -1e9;
    for (var i = 0u; i < MAX_WAVE_SPAN; i++) {
        let s = a + i;
        if (s > b) {
            break;
        }
        let v = samples[min(params.trigger + s, params.n - 1u)] * params.waveGain;
        lo = min(lo, v);
        hi = max(hi, v);
    }
    let yTop = (0.5 - 0.45 * clamp(hi, -1.1, 1.1)) * params.canvasSize.y;
    let yBottom = (0.5 - 0.45 * clamp(lo, -1.1, 1.1)) * params.canvasSize.y;
    let dist = max(max(yTop - px.y, px.y - yBottom), 0.0);
    let lineColor = vec3<f32>(0.30, 0.85, 1.0);
    let core = 1.0 - smoothstep(0.75, 1.75, dist);
    let glow = exp(-dist / 6.0) * 0.35;
    return color + lineColor * max(core, glow);
}

// Spectrogram: time runs left to right (newest column at the right edge), log frequency bottom to top
fn drawSpectrogram(uv: vec2<f32>) -> vec3<f32> {
    let offset = min(u32(uv.x * f32(SPEC_COLUMNS)), SPEC_COLUMNS - 1u);
    let col = (params.column + 1u + offset) % SPEC_COLUMNS;
    let row = min(u32((1.0 - uv.y) * f32(NUM_ROWS)), NUM_ROWS - 1u);
    var color = heat(level(spectrogram[col * NUM_ROWS + row]));
    // Decade ticks on the left edge
    let y = 1.0 - uv.y;
    for (var d = 0u; d < 3u; d++) {
        let f = 100.0 * pow(10.0, f32(d));
        if (abs(y - logPosition(f)) * params.canvasSize.y < 0.75 && uv.x * params.canvasSize.x < 14.0) {
            color = vec3<f32>(0.8);
        }
    }
    return color;
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let uv = frag.uv;
    var color: vec3<f32>;
    if (params.mode == 1u) {
        color = drawWaveform(uv);
    } else if (params.mode == 2u) {
        color = drawSpectrogram(uv);
    } else {
        color = drawBars(uv);
    }

    // Beat indicator: a disc in the top-right corner lit by the GPU beat detector, plus a faint flash
    let pulse = analysis.pulse;
    let px = uv * params.canvasSize;
    let center = vec2<f32>(params.canvasSize.x - 36.0, 36.0);
    let d = length(px - center);
    let beatColor = vec3<f32>(1.0, 0.35, 0.65);
    color += beatColor * pulse * 0.06;
    if (d < 16.0) {
        color = mix(vec3<f32>(0.12, 0.10, 0.16), beatColor, pulse);
    } else if (d < 18.0) {
        color = vec3<f32>(0.45, 0.30, 0.40);
    }
    return vec4<f32>(min(color, vec3<f32>(1.0)), 1.0);
}
`;

const MODES = ['Bars', 'Waveform', 'Spectrogram'];
const FFT_SIZES = [1024, 2048, 4096];

const PRESETS = {
    music:  { name: 'Music',        fftSize: 2048, smoothing: 0.6,  range: 90 },
    fast:   { name: 'Fast response', fftSize: 1024, smoothing: 0.2,  range: 80 },
    detail: { name: 'Fine detail',  fftSize: 4096, smoothing: 0.5,  range: 110 },
    calm:   { name: 'Smooth',       fftSize: 2048, smoothing: 0.9,  range: 90 },
};

// Synth demo: an A major chord, an exponential sweep (a straight diagonal on the log axis) and a
// 120 BPM kick drum. Levels are linear amplitudes before the output volume.
const SYNTH_CHORD = [220.0, 329.63, 554.37];   // A3, E4, C#5 (open voicing: resolvable at N = 2048)
const CHORD_LEVEL = 0.1;
const SWEEP_LO = 200;
const SWEEP_HI = 8000;
const SWEEP_PERIOD = 4;       // seconds per sweep
const SWEEP_LEVEL = 0.08;
const KICK_INTERVAL = 0.5;    // seconds (120 BPM)
const KICK_LEVEL = 0.7;
const OUTPUT_VOLUME = 0.3;    // speaker level for synth and file playback
const SCHEDULE_AHEAD = 0.1;   // kick scheduler lookahead in seconds

// Owns the AudioContext and the active source. Graph: source -> analyser -> monitor -> speakers;
// the monitor gain is 0 for the microphone (no feedback) and when muted, so the analyser still
// sees the signal.
class AudioEngine {
    constructor(fftSize) {
        this.fftSize = fftSize;
        this.ctx = null;
        this.kind = null;          // 'synth' | 'mic' | 'file'
        this.label = '';
        this.muted = false;
        this.nodes = [];
        this.stream = null;
        this.kickTimer = null;
        this.request = 0;          // invalidates a pending mic/file request when the source changes
    }

    // Must be called from a user gesture (click handler) before any await
    ensureContext() {
        if (!this.ctx) {
            const Ctor = window.AudioContext || window.webkitAudioContext;
            if (!Ctor) throw new Error('The Web Audio API is not available in this browser');
            this.ctx = new Ctor();
            this.analyser = this.ctx.createAnalyser();
            this.analyser.fftSize = this.fftSize;
            this.monitor = this.ctx.createGain();
            this.monitor.gain.value = 0;
            this.analyser.connect(this.monitor);
            this.monitor.connect(this.ctx.destination);
        }
        this.resume();
        return this.ctx;
    }

    get sampleRate() {
        return this.ctx ? this.ctx.sampleRate : 48000;
    }

    setFFTSize(n) {
        this.fftSize = n;
        if (this.analyser) this.analyser.fftSize = n;
    }

    setMuted(muted) {
        this.muted = muted;
        this.updateMonitor();
    }

    updateMonitor() {
        if (!this.monitor) return;
        const audible = (this.kind === 'synth' || this.kind === 'file') && !this.muted;
        this.monitor.gain.setTargetAtTime(audible ? OUTPUT_VOLUME : 0, this.ctx.currentTime, 0.02);
    }

    resume() {
        if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    }

    suspend() {
        if (this.ctx && this.ctx.state === 'running') this.ctx.suspend().catch(() => {});
    }

    stopSource() {
        this.request++;
        if (this.kickTimer) clearInterval(this.kickTimer);
        this.kickTimer = null;
        for (const node of this.nodes) {
            try { if (node.stop) node.stop(); } catch { /* already stopped */ }
            node.disconnect();
        }
        this.nodes = [];
        if (this.stream) this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
        this.kind = null;
        this.label = '';
        this.updateMonitor();
    }

    useSynth() {
        this.ensureContext();
        this.stopSource();
        const ctx = this.ctx;
        const now = ctx.currentTime;
        const bus = ctx.createGain();
        bus.connect(this.analyser);
        this.nodes.push(bus);

        for (const f of SYNTH_CHORD) {
            const osc = ctx.createOscillator();
            osc.frequency.value = f;
            const gain = ctx.createGain();
            gain.gain.value = CHORD_LEVEL;
            osc.connect(gain).connect(bus);
            osc.start(now);
            this.nodes.push(osc, gain);
        }

        // Exponential sweep: a sawtooth LFO drives the detune (in cents) around the geometric
        // center, so the frequency goes SWEEP_LO -> SWEEP_HI once per SWEEP_PERIOD and jumps back
        const sweep = ctx.createOscillator();
        sweep.frequency.value = Math.sqrt(SWEEP_LO * SWEEP_HI);
        const lfo = ctx.createOscillator();
        lfo.type = 'sawtooth';
        lfo.frequency.value = 1 / SWEEP_PERIOD;
        const depth = ctx.createGain();
        depth.gain.value = 1200 * Math.log2(Math.sqrt(SWEEP_HI / SWEEP_LO));
        lfo.connect(depth).connect(sweep.detune);
        const sweepGain = ctx.createGain();
        sweepGain.gain.value = SWEEP_LEVEL;
        sweep.connect(sweepGain).connect(bus);
        sweep.start(now);
        lfo.start(now);
        this.nodes.push(sweep, lfo, depth, sweepGain);

        // Kick drum: a sine dropping 150 -> 45 Hz with a fast decay, scheduled slightly ahead
        // of the audio clock (a timer alone would drift and jitter)
        let nextKick = now + 0.05;
        const scheduleKicks = () => {
            if (ctx.currentTime > nextKick) nextKick = ctx.currentTime + 0.05; // after a throttled timer
            while (nextKick < ctx.currentTime + SCHEDULE_AHEAD) {
                const osc = ctx.createOscillator();
                const env = ctx.createGain();
                osc.frequency.setValueAtTime(150, nextKick);
                osc.frequency.exponentialRampToValueAtTime(45, nextKick + 0.12);
                env.gain.setValueAtTime(KICK_LEVEL, nextKick);
                env.gain.exponentialRampToValueAtTime(0.001, nextKick + 0.3);
                osc.connect(env).connect(bus);
                osc.start(nextKick);
                osc.stop(nextKick + 0.32);
                osc.onended = () => env.disconnect();
                nextKick += KICK_INTERVAL;
            }
        };
        scheduleKicks();
        this.kickTimer = setInterval(scheduleKicks, 25);

        this.kind = 'synth';
        this.label = 'Synth demo (A major chord, 200 Hz-8 kHz sweep, 120 BPM kick)';
        this.updateMonitor();
    }

    // Call ensureContext() in the click handler first; throws on denial or missing support
    async useMicrophone() {
        this.ensureContext();
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
        source.connect(this.analyser);
        this.nodes.push(source);
        this.stream = stream;
        this.kind = 'mic';
        const track = stream.getAudioTracks()[0];
        this.label = `Microphone${track && track.label ? ` (${track.label})` : ''}, not played back`;
        this.updateMonitor();
        return true;
    }

    // Decodes the whole file and loops it
    async useFile(file) {
        this.ensureContext();
        this.stopSource();
        const request = this.request;
        const data = await file.arrayBuffer();
        const buffer = await this.ctx.decodeAudioData(data);
        if (request !== this.request) return false;
        const source = this.ctx.createBufferSource();
        source.buffer = buffer;
        source.loop = true;
        source.connect(this.analyser);
        source.start();
        this.nodes.push(source);
        this.kind = 'file';
        this.label = `File "${file.name}" (${buffer.duration.toFixed(1)} s, ${buffer.numberOfChannels} ch, looping)`;
        this.updateMonitor();
        return true;
    }

    close() {
        this.stopSource();
        if (this.ctx) this.ctx.close().catch(() => {});
        this.ctx = null;
    }
}

class FFTAudioVisualizer {
    constructor(device, canvas) {
        this.device = device;
        // Audio keeps playing after a GPU device loss unless we stop it here; the page offers a reload
        device.lost.then((info) => {
            if (info.reason !== 'destroyed') this.pause();
        });
        this.canvas = canvas;
        this.running = false;
        this.destroyed = false;

        this.mode = 0;
        this.smoothing = 0.6;
        this.dbRange = 90;
        this.maxDb = 0;
        this.column = 0;
        this.trigger = 0;
        this.waveGain = 1;
        this.peakLevel = 0;
        this.frames = 0;

        this.fps = 0;
        this.framesSinceSample = 0;
        this.lastSampleTime = 0;

        this.audio = new AudioEngine(2048);

        // One 256-byte-aligned slot of Params per FFT stage (dynamic uniform offsets)
        const align = device.limits.minUniformBufferOffsetAlignment || 256;
        this.paramsStride = Math.ceil(PARAMS_SIZE / align) * align;
        this.params = new ArrayBuffer(this.paramsStride * MAX_LOG2N);
        this.paramsF32 = new Float32Array(this.params);
        this.paramsU32 = new Uint32Array(this.params);

        this.setupBuffers();
        this.setupComputePipelines();
        this.setupRenderPipeline();
        this.setupCanvas();
        this.setFFTSize(2048);
    }

    setupBuffers() {
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.samplesBuffer = this.device.createBuffer({ size: MAX_N * 4, usage: storage });
        // Two complex (vec2<f32>) buffers that the FFT stages ping-pong between
        this.complexBuffers = [0, 1].map(() => this.device.createBuffer({ size: MAX_N * 8, usage: storage }));
        this.twiddleBuffer = this.device.createBuffer({ size: (MAX_N / 2) * 8, usage: storage });
        this.spectrumBuffer = this.device.createBuffer({ size: (MAX_N / 2 + 1) * 8, usage: storage });
        this.analysisBuffer = this.device.createBuffer({ size: (ANALYSIS_HEADER + NUM_ROWS) * 4, usage: storage });
        this.spectrogramBuffer = this.device.createBuffer({ size: SPEC_COLUMNS * NUM_ROWS * 4, usage: storage });
        this.paramsBuffer = this.device.createBuffer({
            size: this.paramsStride * MAX_LOG2N,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const analysis = new Float32Array(ANALYSIS_HEADER + NUM_ROWS).fill(SILENT_DB);
        analysis.fill(0, 0, ANALYSIS_HEADER);
        this.device.queue.writeBuffer(this.analysisBuffer, 0, analysis);
        this.device.queue.writeBuffer(this.spectrogramBuffer, 0, new Float32Array(SPEC_COLUMNS * NUM_ROWS).fill(SILENT_DB));
    }

    setupComputePipelines() {
        const shaderModule = this.device.createShaderModule({ code: computeShader });
        const C = GPUShaderStage.COMPUTE;
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: C, buffer: { type: 'storage' } },
                { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: C, buffer: { type: 'storage' } },
                { binding: 5, visibility: C, buffer: { type: 'storage' } },
                { binding: 6, visibility: C, buffer: { type: 'storage' } },
                { binding: 7, visibility: C,
                  buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAMS_SIZE } }
            ]
        });
        const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        const pipeline = (entryPoint) => this.device.createComputePipeline({
            layout,
            compute: { module: shaderModule, entryPoint }
        });
        this.windowPipeline = pipeline('cs_window');
        this.butterflyPipeline = pipeline('cs_butterfly');
        this.magnitudePipeline = pipeline('cs_magnitude');
        this.bandsPipeline = pipeline('cs_bands');
        this.beatPipeline = pipeline('cs_beat');

        // computeBindGroups[i] reads complexBuffers[i] (src) and writes complexBuffers[1 - i] (dst)
        this.computeBindGroups = [0, 1].map(i => this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.samplesBuffer } },
                { binding: 1, resource: { buffer: this.complexBuffers[i] } },
                { binding: 2, resource: { buffer: this.complexBuffers[1 - i] } },
                { binding: 3, resource: { buffer: this.twiddleBuffer } },
                { binding: 4, resource: { buffer: this.spectrumBuffer } },
                { binding: 5, resource: { buffer: this.analysisBuffer } },
                { binding: 6, resource: { buffer: this.spectrogramBuffer } },
                { binding: 7, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
            ]
        }));
    }

    setupRenderPipeline() {
        const shaderModule = this.device.createShaderModule({ code: renderShader });
        const F = GPUShaderStage.FRAGMENT;
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: F, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: F, buffer: { type: 'uniform' } }
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
                { binding: 0, resource: { buffer: this.samplesBuffer } },
                { binding: 1, resource: { buffer: this.analysisBuffer } },
                { binding: 2, resource: { buffer: this.spectrogramBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer, offset: 0, size: PARAMS_SIZE } }
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

    // Uploads the twiddle table for the new N and clears the smoothing state (bins changed meaning)
    setFFTSize(n) {
        if (!FFT_SIZES.includes(n)) return;
        this.n = n;
        this.log2n = Math.round(Math.log2(n));
        this.timeData = new Float32Array(n);
        this.audio.setFFTSize(n);

        const tw = new Float32Array(n);   // n/2 complex values
        for (let k = 0; k < n / 2; k++) {
            const angle = -2 * Math.PI * k / n;
            tw[2 * k] = Math.cos(angle);
            tw[2 * k + 1] = Math.sin(angle);
        }
        this.device.queue.writeBuffer(this.twiddleBuffer, 0, tw);
        this.device.queue.writeBuffer(this.spectrumBuffer, 0, new Float32Array(2 * (MAX_N / 2 + 1)));
        this.device.queue.writeBuffer(this.samplesBuffer, 0, this.timeData);
    }

    // Writes `slots` copies of Params (slot s has stage = s); see the layout table at the top of the file
    writeParams(slots) {
        const wordsPerSlot = this.paramsStride / 4;
        for (let s = 0; s < slots; s++) {
            const o = s * wordsPerSlot;
            this.paramsU32[o + 0] = this.n;
            this.paramsU32[o + 1] = this.log2n;
            this.paramsU32[o + 2] = s;
            this.paramsU32[o + 3] = this.column;
            this.paramsF32[o + 4] = this.audio.sampleRate;
            this.paramsF32[o + 5] = this.smoothing;
            this.paramsF32[o + 6] = this.maxDb - this.dbRange;
            this.paramsF32[o + 7] = this.maxDb;
            this.paramsU32[o + 8] = this.mode;
            this.paramsU32[o + 9] = this.trigger;
            this.paramsF32[o + 10] = this.canvas.width;
            this.paramsF32[o + 11] = this.canvas.height;
            this.paramsF32[o + 12] = this.waveGain;
        }
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params, 0, slots * this.paramsStride);
    }

    // CPU side of a frame: copy the samples and pick the waveform trigger and gain (display only)
    readSamples() {
        const data = this.timeData;
        if (this.audio.analyser && this.audio.analyser.fftSize === this.n) {
            this.audio.analyser.getFloatTimeDomainData(data);
        }
        let peak = 0;
        for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
        this.peakLevel = peak;
        // Auto gain for the waveform: quiet inputs are scaled up to 90% of the height (at most 20x)
        this.waveGain = peak > 0 ? Math.min(Math.max(0.9 / peak, 1), 20) : 1;
        // First rising zero crossing in the first half, so N/2 samples follow it
        this.trigger = 0;
        for (let i = 1; i < this.n / 2; i++) {
            if (data[i - 1] < 0 && data[i] >= 0) {
                this.trigger = i;
                break;
            }
        }
    }

    // One analysis frame: upload samples, window + bit reversal, log2(N) butterflies, magnitudes,
    // bands + spectrogram column, beat detection, then draw
    frame() {
        this.readSamples();
        this.column = (this.column + 1) % SPEC_COLUMNS;
        this.writeParams(this.log2n);
        this.device.queue.writeBuffer(this.samplesBuffer, 0, this.timeData);

        const n = this.n;
        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        // Bind group 1 has dst = complexBuffers[0], where stage 0 reads
        pass.setPipeline(this.windowPipeline);
        pass.setBindGroup(0, this.computeBindGroups[1], [0]);
        pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP));

        // Stage s reads complexBuffers[s % 2] and writes the other one
        pass.setPipeline(this.butterflyPipeline);
        for (let s = 0; s < this.log2n; s++) {
            pass.setBindGroup(0, this.computeBindGroups[s % 2], [s * this.paramsStride]);
            pass.dispatchWorkgroups(Math.ceil(n / 2 / WORKGROUP));
        }

        // After log2(N) stages the spectrum is in complexBuffers[log2n % 2]: read it as src
        const finalGroup = this.computeBindGroups[this.log2n % 2];
        pass.setBindGroup(0, finalGroup, [0]);
        pass.setPipeline(this.magnitudePipeline);
        pass.dispatchWorkgroups(Math.ceil((n / 2 + 1) / WORKGROUP));
        pass.setPipeline(this.bandsPipeline);
        pass.dispatchWorkgroups(Math.ceil(NUM_ROWS / WORKGROUP));
        pass.setPipeline(this.beatPipeline);
        pass.dispatchWorkgroups(1);
        pass.end();

        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        this.frames++;
    }

    encodeRender(commandEncoder) {
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.02, g: 0.027, b: 0.055, a: 1.0 }
            }]
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();
    }

    // Redraws the current data without analyzing (paused mode/range changes, first frame)
    render() {
        this.writeParams(1);
        const commandEncoder = this.device.createCommandEncoder();
        this.encodeRender(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.audio.resume();
        this.framesSinceSample = 0;
        this.lastSampleTime = performance.now();
        this.animate();
    }

    pause() {
        this.running = false;
        this.fps = 0;
        this.audio.suspend();
    }

    animate() {
        if (!this.running || this.destroyed) return;

        this.frame();

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
        this.audio.close();
        this.samplesBuffer.destroy();
        this.complexBuffers.forEach(b => b.destroy());
        this.twiddleBuffer.destroy();
        this.spectrumBuffer.destroy();
        this.analysisBuffer.destroy();
        this.spectrogramBuffer.destroy();
        this.paramsBuffer.destroy();
    }
}

export async function initFFTAudio(gpuDevice) {
    const { device } = gpuDevice;
    const canvas = document.getElementById('canvas-fft');
    const output = document.getElementById('output-fft');

    const viz = new FFTAudioVisualizer(device, canvas);
    const audio = viz.audio;
    viz.render();
    viz.onStats = () => updateOutput();

    let message = '';   // last source error or notice, shown in the output panel

    const synthButton = document.getElementById('btn-fft-synth');
    const micButton = document.getElementById('btn-fft-mic');
    const fileButton = document.getElementById('btn-fft-file');
    const fileInput = document.getElementById('fft-file');
    const muteButton = document.getElementById('btn-fft-mute');

    // Every handler creates/resumes the AudioContext synchronously, inside the click gesture
    function withContext() {
        try {
            audio.ensureContext();
            return true;
        } catch (error) {
            message = `<span class="error">${error.message}</span>`;
            updateOutput();
            return false;
        }
    }

    synthButton.addEventListener('click', () => {
        if (!withContext()) return;
        audio.useSynth();
        message = '';
        viz.start();
        showControls();
        updateOutput();
    });

    micButton.addEventListener('click', async () => {
        if (!withContext()) return;
        message = '<span class="info">Waiting for microphone permission…</span>';
        viz.start();
        showControls();
        updateOutput();
        try {
            await audio.useMicrophone();   // false if another source was chosen meanwhile
            message = '';
        } catch (error) {
            const denied = error.name === 'NotAllowedError' || error.name === 'SecurityError';
            message = denied
                ? '<span class="error">Microphone permission was denied. Allow it in the browser\'s site settings and press Microphone again, or use the synth demo or a file.</span>'
                : `<span class="error">Microphone unavailable: ${error.message}</span>`;
        }
        showControls();
        updateOutput();
    });

    fileButton.addEventListener('click', () => {
        // Create the context now: picking a file can outlast the click's user activation
        if (!withContext()) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files && fileInput.files[0];
        fileInput.value = '';
        if (!file) return;
        message = `<span class="info">Decoding "${file.name}"…</span>`;
        updateOutput();
        try {
            await audio.useFile(file);
            message = '';
            audio.resume();
            viz.start();
        } catch (error) {
            message = `<span class="error">Could not decode "${file.name}": ${error.message || error}</span>`;
        }
        showControls();
        updateOutput();
    });

    document.getElementById('btn-fft-start').addEventListener('click', () => {
        if (!withContext()) return;
        if (!audio.kind) audio.useSynth();
        viz.start();
        showControls();
        updateOutput();
    });

    document.getElementById('btn-fft-pause').addEventListener('click', () => {
        viz.pause();
        updateOutput();
    });

    muteButton.addEventListener('click', () => {
        audio.setMuted(!audio.muted);
        showControls();
        updateOutput();
    });

    const modeSelect = document.getElementById('fft-mode');
    modeSelect.addEventListener('change', (e) => {
        viz.mode = Math.min(Math.max(parseInt(e.target.value), 0), MODES.length - 1);
        if (!viz.running) viz.render();
        updateOutput();
    });

    const sizeSelect = document.getElementById('fft-size');
    sizeSelect.addEventListener('change', (e) => {
        viz.setFFTSize(parseInt(e.target.value));
        presetButtons.forEach(btn => btn.classList.remove('active'));
        updateOutput();
    });

    const smoothingSlider = document.getElementById('fft-smoothing');
    const smoothingValue = document.getElementById('fft-smoothing-value');
    smoothingSlider.addEventListener('input', (e) => {
        viz.smoothing = Math.min(Math.max(parseFloat(e.target.value), 0), 0.95);
        presetButtons.forEach(btn => btn.classList.remove('active'));
        showControls();
        updateOutput();
    });

    const rangeSlider = document.getElementById('fft-range');
    const rangeValue = document.getElementById('fft-range-value');
    rangeSlider.addEventListener('input', (e) => {
        viz.dbRange = Math.min(Math.max(parseFloat(e.target.value), 40), 140);
        presetButtons.forEach(btn => btn.classList.remove('active'));
        showControls();
        if (!viz.running) viz.render();
        updateOutput();
    });

    const presetButtons = document.querySelectorAll('.fft-preset');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const preset = PRESETS[btn.dataset.preset];
            if (!preset) return;
            viz.setFFTSize(preset.fftSize);
            viz.smoothing = preset.smoothing;
            viz.dbRange = preset.range;
            presetButtons.forEach(b => b.classList.toggle('active', b === btn));
            showControls();
            if (!viz.running) viz.render();
            updateOutput();
        });
    });

    function showControls() {
        sizeSelect.value = String(viz.n);
        smoothingSlider.value = viz.smoothing;
        smoothingValue.textContent = viz.smoothing.toFixed(2);
        rangeSlider.value = viz.dbRange;
        rangeValue.textContent = `${viz.dbRange} dB`;
        muteButton.textContent = audio.muted ? 'Mute: on' : 'Mute: off';
        muteButton.classList.toggle('active', audio.muted);
        muteButton.disabled = audio.kind === 'mic';
        synthButton.classList.toggle('active', audio.kind === 'synth');
        micButton.classList.toggle('active', audio.kind === 'mic');
        fileButton.classList.toggle('active', audio.kind === 'file');
    }

    function updateOutput() {
        const fs = audio.sampleRate;
        const n = viz.n;
        const level = viz.peakLevel > 0 ? `${(20 * Math.log10(viz.peakLevel)).toFixed(1)} dBFS peak` : 'silent';
        const dispatches = 1 + viz.log2n + 3;

        output.innerHTML = `<span class="success">✓ FFT Audio Visualizer Active</span>
${message ? `\n${message}\n` : ''}
<span class="info">Configuration:</span>
• Source: ${audio.kind ? audio.label : 'none yet (press Synth demo, Microphone or Audio file)'}
• Sample rate: ${audio.ctx ? `${fs.toLocaleString()} Hz` : 'set by the audio device once a source starts'}
• FFT size N = ${n}: ${n / 2} bins of ${(fs / n).toFixed(2)} Hz, window ${(1000 * n / fs).toFixed(1)} ms
• View: ${MODES[viz.mode]} · smoothing ${viz.smoothing.toFixed(2)} · range ${viz.maxDb - viz.dbRange} to ${viz.maxDb} dBFS
• Input level: ${level} · Output: ${audio.kind === 'mic' ? 'not played back' : (audio.muted ? 'muted' : 'playing')}
• Status: ${viz.running ? '<span class="success">Running</span>' : '<span class="info">Paused</span>'}

<span class="info">Algorithm (radix-2 Cooley-Tukey on the GPU):</span>
• Hann window w[n] = 0.5 − 0.5·cos(2πn/N), written at the bit-reversed index
• log2(N) = ${viz.log2n} butterfly stages, one dispatch each, ping-ponging two vec2&lt;f32&gt; buffers:
  X[i0] = a + W·b, X[i1] = a − W·b with twiddles W_N^k from a precomputed table
• |X[k]|·4/N → smoothed → dB; resampled onto ${NUM_ROWS} log-spaced rows (${F_LO} Hz-${F_HI / 1000} kHz)
• Spectrogram: ${SPEC_COLUMNS}-column ring buffer on the GPU, one column per frame
• Beats: 40-150 Hz energy rising above 1.6× its running average, detected on the GPU

<span class="info">Performance:</span>
• Frame rate: ${viz.running ? viz.fps.toFixed(0) : 0} FPS (one FFT per frame)
• Compute dispatches per frame: ${dispatches} (window, ${viz.log2n} stages, magnitude, bands, beat)
• Work per FFT: ${(n / 2 * viz.log2n).toLocaleString()} butterflies; CPU → GPU: ${n * 4} bytes of samples, no readback`;
    }

    showControls();
    updateOutput();
    return viz;
}
