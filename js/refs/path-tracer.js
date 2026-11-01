// CPU references for the path-tracer example: closed-form radiance of the furnace scenes, against
// which a Monte Carlo estimate (mean ± standard error over samples) can be checked for bias.
// Precision: float64, exact expressions.
// Cost: O(1).

/**
 * Closed box whose every wall has albedo rho and emission le, traced with `bounces` bounces: every
 * path vertex sees le, so L = le · Σ_{k=0..bounces} rho^k.
 */
export const furnaceRadiance = (rho, le, bounces) => le * (1 - rho ** (bounces + 1)) / (1 - rho);

/** White furnace: a convex diffuse object of albedo rho under a uniform environment env returns rho·env. */
export const whiteFurnaceRadiance = (rho, env) => rho * env;
