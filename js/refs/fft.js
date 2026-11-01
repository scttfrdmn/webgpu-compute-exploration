// CPU references for the fft-audio example: a naive DFT (independent of any FFT factorization), a
// float64 radix-2 FFT (for CPU timing comparisons), a spectrum error measure, and the magnitude /
// log-band reductions the visualizer applies to a spectrum.
// Precision: naiveDFT and fft64 are float64 (window and twiddles in double); magnitudes and bands mirror
// the shader's f32 outputs (Float32Array results, Math.fround on the magnitude) so they compare tightly.
// Cost: naiveDFT is O(n²) (n = 4096 is ~1.7e7 complex multiply-adds, fine in a browser; avoid
// n ≫ 16384); fft64 is O(n log n); spectrumError, magnitudes are O(n); bands is O(NUM_ROWS + n).

export const NUM_ROWS = 256, F_LO = 20, F_HI = 20000, SILENT_DB = -200;
export const DB_PER_LOG2 = Math.fround(6.02059991);

// Independent reference: float64 naive DFT of the float64 Hann-windowed input. With `windowed`
// given, transforms that sequence as is (no window).
export function naiveDFT(samples, windowed = null) {
    const n = samples.length;
    const x = windowed ? Float64Array.from(windowed)
        : Float64Array.from(samples, (v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n)));
    const cos = new Float64Array(n), sin = new Float64Array(n);
    for (let i = 0; i < n; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    const out = new Float64Array(2 * n);
    for (let k = 0; k < n; k++) {
        let re = 0, im = 0;
        for (let i = 0, idx = 0; i < n; i++, idx = (idx + k) % n) { re += x[i] * cos[idx]; im -= x[i] * sin[idx]; }
        out[2 * k] = re; out[2 * k + 1] = im;
    }
    return { out, x };
}

// float64 iterative radix-2 FFT of a real sequence (no window), for CPU timing comparisons on the
// Measure tab: the same factorization as the shader, in double precision. O(n log n); n a power of 2.
// Returns the interleaved complex spectrum (Float64Array 2n).
export function fft64(x) {
    const n = x.length, bits = Math.round(Math.log2(n));
    const out = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
        let j = 0;
        for (let b = 0; b < bits; b++) j |= ((i >>> b) & 1) << (bits - 1 - b);
        out[2 * j] = x[i];
    }
    for (let h = 1; h < n; h *= 2) {
        const step = Math.PI / h;
        for (let j = 0; j < h; j++) {
            const wr = Math.cos(step * j), wi = -Math.sin(step * j);
            for (let i0 = j; i0 < n; i0 += 2 * h) {
                const i1 = i0 + h;
                const br = out[2 * i1], bi = out[2 * i1 + 1];
                const tr = wr * br - wi * bi, ti = wr * bi + wi * br;
                const ar = out[2 * i0], ai = out[2 * i0 + 1];
                out[2 * i0] = ar + tr; out[2 * i0 + 1] = ai + ti;
                out[2 * i1] = ar - tr; out[2 * i1 + 1] = ai - ti;
            }
        }
    }
    return out;
}

// Error of an interleaved complex spectrum X against a reference `ref` (both n bins), relative to the
// reference's largest bin: maxRel = max_k |X_k − ref_k| / max_k |ref_k|, rmsRel = rms_k |X_k − ref_k| /
// max_k |ref_k|. parseval compares Σ|X_k|² with n·Σ x_i² of the transformed sequence `x` (if given).
export function spectrumError(X, ref, n, x = null) {
    let maxErr = 0, sumSq = 0, maxRef = 0, energy = 0;
    for (let k = 0; k < n; k++) {
        const e = Math.hypot(X[2 * k] - ref[2 * k], X[2 * k + 1] - ref[2 * k + 1]);
        maxErr = Math.max(maxErr, e);
        sumSq += e * e;
        maxRef = Math.max(maxRef, Math.hypot(ref[2 * k], ref[2 * k + 1]));
        energy += X[2 * k] ** 2 + X[2 * k + 1] ** 2;
    }
    const result = { maxRel: maxErr / maxRef, rmsRel: Math.sqrt(sumSq / n) / maxRef };
    if (x) {
        let e = 0;
        for (let i = 0; i < n; i++) e += x[i] * x[i];
        result.parseval = Math.abs(energy / (n * e) - 1);
    }
    return result;
}

// ---- cs_magnitude and cs_bands ----
// Interleaved complex spectrum X (n bins) -> interleaved (magnitude, dB) for bins 0..n/2, with
// exponential smoothing against `prev`.
export function magnitudes(X, n, smoothing = 0, prev = null) {
    const spec = new Float32Array(2 * (n / 2 + 1));
    for (let k = 0; k <= n / 2; k++) {
        const mag = Math.fround(Math.hypot(X[2 * k], X[2 * k + 1]) * 4 / n);
        const blended = Math.fround(smoothing * (prev ? prev[2 * k] : 0) + (1 - smoothing) * mag);
        spec[2 * k] = blended;
        spec[2 * k + 1] = DB_PER_LOG2 * Math.log2(Math.max(blended, 1e-10));
    }
    return spec;
}
// NUM_ROWS log-spaced bands from F_LO to F_HI: peak dB of the bins inside each band, or the
// interpolated dB at the band's geometric centre when no bin falls inside.
export function bands(spec, n, sampleRate) {
    const half = n / 2, binHz = sampleRate / n, ratio = F_HI / F_LO;
    const out = new Float32Array(NUM_ROWS);
    for (let r = 0; r < NUM_ROWS; r++) {
        const bLo = F_LO * Math.pow(ratio, r / NUM_ROWS) / binHz;
        const bHi = Math.min(F_LO * Math.pow(ratio, (r + 1) / NUM_ROWS) / binHz, half);
        let db = SILENT_DB;
        if (bLo < half) {
            const kStart = Math.ceil(bLo), kEnd = Math.floor(bHi);
            if (kEnd < kStart) {
                const c = Math.sqrt(bLo * bHi);
                const k0 = Math.min(Math.floor(c), half), k1 = Math.min(k0 + 1, half);
                const f = c - Math.floor(c);
                db = spec[2 * k0 + 1] * (1 - f) + spec[2 * k1 + 1] * f;
            } else {
                for (let i = 0; i < 128 && kStart + i <= kEnd; i++) db = Math.max(db, spec[2 * (kStart + i) + 1]);
            }
        }
        out[r] = db;
    }
    return out;
}
