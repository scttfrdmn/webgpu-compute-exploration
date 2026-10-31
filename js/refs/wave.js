// CPU reference for the wave-equation example: the leapfrog scheme's conserved discrete energy,
// used to measure drift (conserved by the undamped scheme with u = 0 outside the grid).
// Precision: float64 accumulation over f32 height fields.
// Cost: O(W·H) per evaluation (5 taps per cell); cheap for any grid the example runs.

// Discrete energy E(n+½) with u0 = u(n), u1 = u(n+1): ‖u1 − u0‖² − C²⟨u1, L u0⟩
export function energy(u0, u1, W, H, C2) {
    let kin = 0, pot = 0;
    const at = (u, x, y) => (x < 0 || y < 0 || x >= W || y >= H) ? 0 : u[y * W + x];
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const i = y * W + x;
            const d = u1[i] - u0[i];
            kin += d * d;
            const lap = at(u0, x - 1, y) + at(u0, x + 1, y) + at(u0, x, y - 1) + at(u0, x, y + 1) - 4 * u0[i];
            pot -= u1[i] * lap;
        }
    }
    return kin + C2 * pot;
}
