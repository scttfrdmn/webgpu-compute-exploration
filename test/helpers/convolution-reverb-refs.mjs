// convolution-reverb references shared by the CPU suite (test/convolution-reverb.test.mjs) and the
// GPU kernel tests (test/gpu/convolution-reverb.gpu.test.mjs). Nothing here touches a GPU.

export function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Independent reference: full linear convolution in float64, y[n] = sum_i x[i]·h[n - i].
export function convolve(x, h, length) {
    const y = new Float64Array(length);
    for (let i = 0; i < x.length; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        for (let k = 0; k < h.length && i + k < length; k++) y[i + k] += xi * h[k];
    }
    return y;
}

// Stereo exponentially decaying noise IR of `length` samples.
export function noiseIR(length, seed, decaySamples) {
    const rand = mulberry32(seed);
    const make = () => Float32Array.from({ length }, (_, i) => (rand() * 2 - 1) * Math.exp(-i / decaySamples));
    return [make(), make()];
}
