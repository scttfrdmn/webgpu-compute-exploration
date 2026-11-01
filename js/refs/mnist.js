// CPU reference for the MNIST examples: the forward pass of the shared CNN (js/examples/mnist-model.js)
// written out directly from the architecture, in float64.
// Network: 28×28 input (byte / 255) → conv 5×5 ×8 → ReLU → 2×2 max-pool → conv 5×5 ×16 → ReLU → 2×2
// max-pool → dense 256 → 10 → softmax. Parameter order (5,994 floats): W1 [8][1][5][5], b1 [8],
// W2 [16][8][5][5], b2 [16], W3 [10][256] (j = c·16 + y·4 + x), b3 [10].
// Precision: float64 throughout (the GPU works in f32; probabilities agree to about 1e-6).
// Cost: about 3.3·10⁵ multiply-adds per image; the 2,000 test images take on the order of a second.

const W1 = 0, B1 = 200, W2 = 208, B2 = 3408, W3 = 3424, B3 = 5984;

/** Softmax probabilities (Float64Array(10)) of image `index` of `images` (uint8, 784 per image). */
export function mnistForward(weights, images, index) {
    const x = new Float64Array(784);
    for (let i = 0; i < 784; i++) x[i] = images[index * 784 + i] / 255;
    // conv1 + ReLU + pool: 8 × 12 × 12
    const p1 = new Float64Array(8 * 144);
    for (let f = 0; f < 8; f++) {
        for (let py = 0; py < 12; py++) for (let px = 0; px < 12; px++) {
            let best = -Infinity;
            for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
                const oy = 2 * py + dy, ox = 2 * px + dx;
                let s = weights[B1 + f];
                for (let ky = 0; ky < 5; ky++) for (let kx = 0; kx < 5; kx++) {
                    s += weights[W1 + f * 25 + ky * 5 + kx] * x[(oy + ky) * 28 + ox + kx];
                }
                best = Math.max(best, Math.max(s, 0));
            }
            p1[f * 144 + py * 12 + px] = best;
        }
    }
    // conv2 + ReLU + pool: 16 × 4 × 4
    const p2 = new Float64Array(256);
    for (let f = 0; f < 16; f++) {
        for (let py = 0; py < 4; py++) for (let px = 0; px < 4; px++) {
            let best = -Infinity;
            for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
                const oy = 2 * py + dy, ox = 2 * px + dx;
                let s = weights[B2 + f];
                for (let c = 0; c < 8; c++) {
                    for (let ky = 0; ky < 5; ky++) for (let kx = 0; kx < 5; kx++) {
                        s += weights[W2 + ((f * 8 + c) * 5 + ky) * 5 + kx] * p1[c * 144 + (oy + ky) * 12 + ox + kx];
                    }
                }
                best = Math.max(best, Math.max(s, 0));
            }
            p2[f * 16 + py * 4 + px] = best;
        }
    }
    // dense + softmax (log-sum-exp)
    const logits = new Float64Array(10);
    for (let k = 0; k < 10; k++) {
        let s = weights[B3 + k];
        for (let j = 0; j < 256; j++) s += weights[W3 + k * 256 + j] * p2[j];
        logits[k] = s;
    }
    const m = Math.max(...logits);
    let sum = 0;
    for (let k = 0; k < 10; k++) sum += Math.exp(logits[k] - m);
    return logits.map(l => Math.exp(l - m) / sum);
}

/** Index of the largest probability (the first one on ties). */
export function argmax(p, offset = 0, length = 10) {
    let best = 0;
    for (let k = 1; k < length; k++) if (p[offset + k] > p[offset + best]) best = k;
    return best;
}

/**
 * The first `count` images of { images, labels }: { count, correct, accuracy, loss (mean cross-entropy),
 * probs: Float64Array(count × 10) }.
 */
export function mnistEvaluate(weights, data, count = data.count) {
    const probs = new Float64Array(count * 10);
    let correct = 0, loss = 0;
    for (let i = 0; i < count; i++) {
        const p = mnistForward(weights, data.images, i);
        probs.set(p, i * 10);
        if (argmax(p) === data.labels[i]) correct++;
        loss -= Math.log(Math.max(p[data.labels[i]], 1e-300));
    }
    return { count, correct, accuracy: correct / count, loss: loss / count, probs };
}
