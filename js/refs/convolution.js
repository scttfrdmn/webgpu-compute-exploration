// CPU references for the convolution-reverb example: direct linear convolution and the seeded
// noise impulse response the tests feed it.
// Precision: convolve accumulates in float64; noiseIR returns f32 samples (what the example uploads).
// Cost: convolve is O(len(x)·len(h)); a 1 s stereo IR at 48 kHz against a 1 s block is ~2.3e9
// multiply-adds, so in a browser compare short blocks (a few thousand samples) or a truncated IR.

import { mulberry32 } from './random.js';

export { mulberry32 };

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
