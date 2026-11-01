// CPU references for the path-tracer example: closed-form radiance of the furnace scenes, against
// which a Monte Carlo estimate (mean ± standard error over samples) can be checked for bias, and the
// RMS error of a rendered image against a reference image or value.
// Precision: float64, exact expressions.
// Cost: O(1); imageRmse is O(pixels).

/**
 * Closed box whose every wall has albedo rho and emission le, traced with `bounces` bounces: every
 * path vertex sees le, so L = le · Σ_{k=0..bounces} rho^k.
 */
export const furnaceRadiance = (rho, le, bounces) => le * (1 - rho ** (bounces + 1)) / (1 - rho);

/** White furnace: a convex diffuse object of albedo rho under a uniform environment env returns rho·env. */
export const whiteFurnaceRadiance = (rho, env) => rho * env;

/**
 * Root-mean-square difference of an image (any flat array, e.g. rgb per pixel) from a reference image
 * of the same length or from a constant, and the reference's mean: { rmse, mean, relative = rmse / mean }.
 * Float64 sums.
 */
export function imageRmse(image, reference) {
    const constant = typeof reference === 'number';
    let se = 0, sum = 0;
    for (let i = 0; i < image.length; i++) {
        const r = constant ? reference : reference[i];
        se += (image[i] - r) ** 2;
        sum += r;
    }
    const rmse = Math.sqrt(se / image.length);
    const mean = sum / image.length;
    return { rmse, mean, relative: mean !== 0 ? rmse / Math.abs(mean) : Infinity };
}
