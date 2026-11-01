// CPU reference for the SPH fluid example: the density summation of its density pass, by brute force.
// Particles are packed as the shader's array<Particle>: 8 floats each (position xy, velocity xy, density,
// pressure, pad). Kernel: 2D Poly6, W(r, h) = 4 / (π h⁸) · (h² − r²)³ for r < h, with the particle itself
// included (r = 0), as in COMPUTE_DENSITY_SHADER.
// Precision: float64 (the shader's π is 3.14159265, 1e-9 relative from Math.PI, far below f32).
// Cost: O(n²) (4,000 particles is 1.6e7 pairs, a fraction of a second in a browser).

/** ρ_i = Σ_j m W(|x_i − x_j|, h) for every particle; returns a Float64Array of n densities. */
export function sphDensities(particles, n, h, mass) {
    const coef = 4 / (Math.PI * h ** 8);
    const rho = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const xi = particles[8 * i], yi = particles[8 * i + 1];
        let d = 0;
        for (let j = 0; j < n; j++) {
            const dx = xi - particles[8 * j], dy = yi - particles[8 * j + 1];
            const r2 = dx * dx + dy * dy;
            if (r2 < h * h) d += mass * coef * (h * h - r2) ** 3;
        }
        rho[i] = d;
    }
    return rho;
}
