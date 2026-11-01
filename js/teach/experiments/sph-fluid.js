// Measure experiments for the SPH fluid (js/examples/sph-fluid.js). Interface: see the top of
// js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureSPHFrame() and runSPH(), which encode
// frames exactly as the interactive step() does (6 substeps of density, forces, integrate) on the
// example's dam break. The reference is the brute-force float64 density sum of js/refs/sph.js, applied
// to the positions the last density pass saw. Caps: N ≤ 4,000 (the example's maximum), ≤ 150 frames
// (2.5 simulated seconds) per stiffness run, one frame per submit.

import { measureSPHFrame, runSPH, MAX_SPH_PARTICLES } from '../../examples/sph-fluid.js';
import { sphDensities } from '../../refs/sph.js';

const ms = (v) => v.toFixed(3);

// ---- 1. Scaling ----------------------------------------------------------------------------------------
export const COUNTS = [250, 500, 1000, 2000, 4000];
export const SUBSTEPS = 6;   // per frame, as in the example

const scaling = {
    id: 'scaling',
    title: 'Particles vs time per substep',
    question: 'How does the cost of the brute-force SPH passes grow with N, and which of the three passes dominates?',
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'particles N' },
        { key: 'density', label: 'density GPU ms', format: ms },
        { key: 'forces', label: 'forces GPU ms', format: ms },
        { key: 'integrate', label: 'integrate GPU ms', format: ms },
        { key: 'gpuMsPerSubstep', label: 'GPU ms / substep', format: ms },
        { key: 'wallMsPerFrame', label: 'wall ms / frame (6 substeps)', format: ms },
        { key: 'pairsWall', label: 'G pair checks/s (wall)', format: (v) => v.toFixed(2) },
    ],
    charts: [
        {
            x: 'n', xLabel: 'particles N', logX: true, y: ['density', 'forces', 'integrate'], yLabel: 'GPU ms per substep', logY: true,
            labels: { density: 'density', forces: 'forces', integrate: 'integrate' },
            emptyNote: 'Per-pass GPU times need the timestamp-query feature, which this browser or GPU does not offer. The wall-time chart below still applies.',
        },
        {
            x: 'n', xLabel: 'particles N', logX: true, y: ['wallMsPerFrame'], yLabel: 'wall ms per frame (6 substeps)', logY: true,
            reference: (rows) => {
                const last = rows.filter(r => r.wallMsPerFrame > 0).at(-1);
                return last ? { label: 'N²', points: [COUNTS[0], last.n].map(n => [n, last.wallMsPerFrame * (n / last.n) ** 2]) } : { label: 'N²', points: [] };
            },
        },
    ],
    baseline: 'Density and forces each loop over all N particles (2 N² pair checks per substep); integrate is O(N). '
        + 'At 4,000 particles and 6 substeps a frame is 192 million pair checks, which is why the example caps N there. '
        + 'CI (CPU-emulated GPU): 7.8 ms per frame at N = 250, 45 ms at 1,000 and 406 ms at 4,000.',
    async *run(ctx) {
        for (const n of COUNTS) {
            if (n > MAX_SPH_PARTICLES) break;
            await ctx.step();
            const r = await measureSPHFrame(ctx.gpuDevice, { n, repeats: 3 });
            const per = (v) => (v === null ? null : v / SUBSTEPS);
            yield {
                n, density: per(r.passes.density), forces: per(r.passes.forces), integrate: per(r.passes.integrate),
                gpuMsPerSubstep: per(r.gpuMs), wallMsPerFrame: r.wallMs,
                pairsWall: r.wallMs > 0 ? 2 * n * n * SUBSTEPS / r.wallMs / 1e6 : null,
            };
        }
    },
};

// ---- 2. Stiffness vs compression and stability ----------------------------------------------------------
export const GAS_CONSTANTS = [2.5e5, 1e6, 4e6, 1.6e7, 6.4e7];   // the example uses 1e6
export const STIFFNESS_COUNTS = [500, 1000];
export const FRAMES = 150;       // 2.5 s: the block lands at about 1.2 s and splashes
export const SAMPLES = 50;       // a read-back every 3 frames, often enough to catch the impact peak

// Largest relative difference between the GPU densities in a snapshot and the float64 sum
export function densityError(particles, n, h, mass) {
    const ref = sphDensities(particles, n, h, mass);
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(particles[8 * i + 4] - ref[i]) / ref[i]);
    return worst;
}

const stiffness = {
    id: 'stiffness',
    title: 'Gas constant vs compression and stability',
    question: 'How stiff must the equation of state be to keep the fluid nearly incompressible, and where does the fixed time step stop being stable?',
    settings: [{ key: 'n', label: 'Particles', options: STIFFNESS_COUNTS.map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 1000 }],
    budgetMs: 20000,
    columns: [
        { key: 'k', label: 'gas constant k', format: (v) => v.toExponential(1) },
        { key: 'soundSpeed', label: 'c = √k (px/s)', format: (v) => v.toFixed(0) },
        { key: 'cfl', label: 'c·Δt / h', format: (v) => v.toFixed(2) },
        { key: 'peakCompression', label: 'peak ρ/ρ₀ − 1', format: (v) => v.toFixed(3) },
        { key: 'maxSpeed', label: 'max speed (px/s)', format: (v) => v.toFixed(0) },
        { key: 'densityError', label: 'density vs float64 (max rel.)', format: (v) => v.toExponential(1) },
        { key: 'wallMs', label: 'wall ms', format: (v) => v.toFixed(0) },
    ],
    charts: [
        { x: 'k', xLabel: 'gas constant k', logX: true, y: ['peakCompression'], yLabel: 'peak ρ/ρ₀ − 1 over 2.5 s', logY: true },
        { x: 'k', xLabel: 'gas constant k', logX: true, y: ['maxSpeed'], yLabel: 'max particle speed (px/s)', logY: true },
    ],
    baseline: 'CI (shipped shaders on a CPU-emulated GPU, N = 500): peak ρ/ρ₀ − 1 of 5.4, 1.8, 0.79 and 0.34 for '
        + 'k = 2.5e5 … 1.6e7 (c·Δt/h = 0.09 … 0.69), roughly halving per 4× in k; at k = 6.4e7 (c·Δt/h = 1.39) the run '
        + 'blows up: 27,000 px/s against 800–2,400 for the stable runs. The density pass matches the float64 sum to 2.5e-7. '
        + 'A float64 CPU replica at N = 1,000 gives 5.6, 2.0, 0.86 and 0.43. The shipped k = 1e6 lets the splash compress the fluid to about 3× rest density.',
    async *run(ctx, { n }) {
        const N = Math.min(n, MAX_SPH_PARTICLES);
        for (const k of GAS_CONSTANTS) {
            await ctx.step();
            const r = await runSPH(ctx.gpuDevice, { n: N, gasConstant: k, frames: FRAMES, samples: SAMPLES, between: () => ctx.step() });
            let peak = 0, vmax = 0;
            for (const { particles } of r.snapshots.slice(1)) {
                for (let i = 0; i < N; i++) {
                    peak = Math.max(peak, particles[8 * i + 4] / r.restDensity);
                    vmax = Math.max(vmax, Math.hypot(particles[8 * i + 2], particles[8 * i + 3]));
                }
            }
            const c = Math.sqrt(k);
            yield {
                k, soundSpeed: c, cfl: c * r.dt / r.smoothingRadius, peakCompression: peak - 1, maxSpeed: vmax,
                densityError: densityError(r.snapshots.at(-1).particles, N, r.smoothingRadius, r.mass), wallMs: r.wallMs,
            };
        }
    },
};

export default [scaling, stiffness];
