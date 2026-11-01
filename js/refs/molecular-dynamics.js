// CPU references for the molecular-dynamics example: the Lennard-Jones energy of a periodic box and one
// step of the shader's integrator, both in float64.
// Atoms are packed as the shader's array<Atom>: 8 floats per atom (position xyz, pad, velocity xyz, pad).
// Units: ε = σ = m = 1. The shader truncates the force at r_c = 2.5σ without shifting it, so the
// potential that this force is the gradient of is the truncated-and-shifted U(r) − U(r_c) for r < r_c;
// ljEnergy uses that one, which is what the dynamics conserves (an unshifted sum would jump by U(r_c)
// whenever a pair crosses the cutoff).
// Precision: float64 throughout. Cost: O(n²) pairs per call (5,000 atoms is 1.25e7 pairs, well under a
// second in a browser).

export const CUTOFF = 2.5;
const RC2 = CUTOFF * CUTOFF;
const U_RC = 4 * (RC2 ** -6 - RC2 ** -3);

// Minimum image of one component in a box of side L centered on 0 (the shader's minImage / applyPBC)
const wrap = (d, L) => (d > L / 2 ? d - L : d < -L / 2 ? d + L : d);

/** Kinetic, potential (truncated and shifted at 2.5σ, minimum image) and total energy of `atoms`. */
export function ljEnergy(atoms, n, boxSize) {
    let K = 0, U = 0;
    for (let i = 0; i < n; i++) {
        const vx = atoms[8 * i + 4], vy = atoms[8 * i + 5], vz = atoms[8 * i + 6];
        K += 0.5 * (vx * vx + vy * vy + vz * vz);
        for (let j = i + 1; j < n; j++) {
            const dx = wrap(atoms[8 * i] - atoms[8 * j], boxSize);
            const dy = wrap(atoms[8 * i + 1] - atoms[8 * j + 1], boxSize);
            const dz = wrap(atoms[8 * i + 2] - atoms[8 * j + 2], boxSize);
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > RC2 || r2 < 0.01) continue;   // the shader's cutoffs
            const sr6 = 1 / (r2 * r2 * r2);
            U += 4 * (sr6 * sr6 - sr6) - U_RC;
        }
    }
    return { K, U, E: K + U };
}

/**
 * One dispatch of the shader's `main` in float64: all-pairs LJ force with minimum image, semi-implicit
 * Euler (v += F·dt, then x += v·dt), periodic wrap, and (thermostat = true) the per-atom speed rescale
 * toward √(0.01·T). Returns new packed atoms (Float64Array, same layout).
 */
export function mdStep(atoms, n, boxSize, dt, { thermostat = false, temperature = 30 } = {}) {
    const out = new Float64Array(8 * n);
    for (let i = 0; i < n; i++) {
        let fx = 0, fy = 0, fz = 0;
        for (let j = 0; j < n; j++) {
            if (j === i) continue;
            const dx = wrap(atoms[8 * i] - atoms[8 * j], boxSize);
            const dy = wrap(atoms[8 * i + 1] - atoms[8 * j + 1], boxSize);
            const dz = wrap(atoms[8 * i + 2] - atoms[8 * j + 2], boxSize);
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > RC2 || r2 < 0.01) continue;
            const sr6 = 1 / (r2 * r2 * r2);
            const m = 24 * (2 * sr6 * sr6 - sr6) / r2;
            fx += m * dx; fy += m * dy; fz += m * dz;
        }
        let vx = atoms[8 * i + 4] + fx * dt, vy = atoms[8 * i + 5] + fy * dt, vz = atoms[8 * i + 6] + fz * dt;
        out[8 * i] = wrap(atoms[8 * i] + vx * dt, boxSize);
        out[8 * i + 1] = wrap(atoms[8 * i + 1] + vy * dt, boxSize);
        out[8 * i + 2] = wrap(atoms[8 * i + 2] + vz * dt, boxSize);
        const speed = Math.hypot(vx, vy, vz);
        if (thermostat && speed > 0) {
            const s = 1 - 0.01 * (1 - Math.sqrt(temperature * 0.01) / speed);
            vx *= s; vy *= s; vz *= s;
        }
        out[8 * i + 4] = vx; out[8 * i + 5] = vy; out[8 * i + 6] = vz;
    }
    return out;
}
