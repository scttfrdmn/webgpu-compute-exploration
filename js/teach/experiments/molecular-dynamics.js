// Measure experiments for molecular dynamics (js/examples/molecular-dynamics.js). Interface: see the top
// of js/teach/measure.js.
//
// The GPU work goes through the example's compute-only measureMD(), which runs the shipped all-pairs
// Lennard-Jones kernel (or its NVE variant without the thermostat) with the example's buffers. The
// shader computes forces but no potential, so the energy is computed on the CPU from read-back atoms with
// js/refs/molecular-dynamics.js (float64, the truncated-and-shifted potential the truncated force derives
// from). Caps: N ≤ 5,000 (the slider's maximum), ≤ 2,000 steps per run, ≤ 100 steps and ≤ 2.5e8 pair terms per submit.

import { measureMD, generateAtoms, MAX_MD_ATOMS } from '../../examples/molecular-dynamics.js';
import { ljEnergy } from '../../refs/molecular-dynamics.js';
import { mulberry32 } from '../../refs/random.js';

export const SEED = 11;
export const TEMPERATURE = 30;   // the example's default slider value

// ---- 1. Scaling ----------------------------------------------------------------------------------------
export const COUNTS = [250, 500, 1000, 2000, 4000, 5000];
export const SCALING_BOX = 30;   // the example's box
export const SCALING_STEPS = 20;

const ms = (v) => v.toFixed(3);
const sci = (v) => v.toExponential(2);

const scaling = {
    id: 'scaling',
    title: 'Atoms vs time per step',
    question: 'How does the cost of one all-pairs force step grow with N, and how many pair interactions per second does your GPU sustain?',
    budgetMs: 20000,
    columns: [
        { key: 'n', label: 'atoms N' },
        { key: 'gpuMsPerStep', label: 'GPU ms / step', format: ms },
        { key: 'wallMsPerStep', label: 'wall ms / step', format: ms },
        { key: 'pairsGpu', label: 'G pair terms/s (GPU)', format: (v) => v.toFixed(2) },
        { key: 'pairsWall', label: 'G pair terms/s (wall)', format: (v) => v.toFixed(2) },
    ],
    charts: [
        {
            x: 'n', xLabel: 'atoms N', logX: true, y: ['gpuMsPerStep', 'wallMsPerStep'], yLabel: 'ms per step', logY: true,
            labels: { gpuMsPerStep: 'GPU', wallMsPerStep: 'wall' },
            reference: (rows) => {
                const last = rows.filter(r => r.wallMsPerStep > 0).at(-1);
                return last ? { label: 'N²', points: [COUNTS[0], last.n].map(n => [n, last.wallMsPerStep * (n / last.n) ** 2]) } : { label: 'N²', points: [] };
            },
        },
        {
            x: 'n', xLabel: 'atoms N', logX: true, y: ['pairsGpu', 'pairsWall'], yLabel: 'billion pair terms per second (N(N−1) per step)', logY: true,
            labels: { pairsGpu: 'GPU time', pairsWall: 'wall' },
        },
    ],
    baseline: `Each atom loops over all N atoms (no neighbour list), so a step costs N(N − 1) pair terms: 25 million at `
        + `N = ${MAX_MD_ATOMS.toLocaleString('en-US')}. ${SCALING_STEPS} steps are timed (up to 20 per submit) in the example's 30-unit box, where `
        + 'only a few pairs fall inside the 2.5σ cutoff; the cost is the loop, not the force arithmetic. CI (CPU-emulated '
        + 'GPU): 0.37 ms per step at N = 250, 68 ms at 5,000, 0.17 → 0.37 billion pair terms per second.',
    async *run(ctx) {
        for (const n of COUNTS) {
            if (n > MAX_MD_ATOMS) break;
            const atoms = generateAtoms(n, SCALING_BOX, TEMPERATURE, mulberry32(SEED));
            await ctx.step();
            await measureMD(ctx.gpuDevice, { atoms, boxSize: SCALING_BOX, steps: 5 });   // warm-up
            await ctx.step();
            const r = await measureMD(ctx.gpuDevice, { atoms, boxSize: SCALING_BOX, steps: SCALING_STEPS });
            const pairs = n * (n - 1);
            const rate = (t) => (t > 0 ? pairs / t / 1e6 : null);
            yield { n, gpuMsPerStep: r.gpuMsPerStep, wallMsPerStep: r.wallMsPerStep, pairsGpu: rate(r.gpuMsPerStep), pairsWall: rate(r.wallMsPerStep) };
        }
    },
};

// ---- 2. Time step vs energy error -------------------------------------------------------------------
export const DTS = [0.016, 0.008, 0.004, 0.002];
export const DRIFT_TIME = 4;          // simulated time per run: steps = 4 / dt ≤ 2,000
export const DRIFT_COUNTS = [512, 1000];
export const DENSITY = 0.8;           // atoms per σ³: a dense liquid, so every atom has ~ 50 neighbours inside 2.5σ
export const DRIFT_SAMPLES = 20;
export const boxFor = (n) => Math.cbrt(n / DENSITY);

const energyDrift = {
    id: 'energy-drift',
    title: 'Time step vs energy error',
    question: 'How does the energy error of semi-implicit Euler scale with the time step, and what does the example\'s thermostat do to the energy?',
    settings: [
        { key: 'n', label: 'Atoms', options: DRIFT_COUNTS.map(n => ({ value: n, label: n.toLocaleString('en-US') })), default: 512 },
        { key: 'thermostat', label: 'Thermostat', options: [{ value: false, label: 'off (NVE)' }, { value: true, label: 'on (as in Explore)' }], default: false },
    ],
    budgetMs: 20000,
    columns: [
        { key: 'dt', label: 'dt', format: String },
        { key: 'steps', label: 'steps' },
        { key: 'maxDrift', label: 'max |ΔE/E₀|', format: sci },
        { key: 'finalDrift', label: 'final ΔE/E₀', format: sci },
        { key: 'kineticRatio', label: 'K(end) / K(0)', format: (v) => v.toFixed(1) },
        { key: 'wallMs', label: 'wall ms', format: (v) => v.toFixed(0) },
    ],
    charts: [{
        x: 'dt', xLabel: 'time step dt', logX: true, y: ['maxDrift'], yLabel: `max |ΔE/E₀| over t = ${DRIFT_TIME}`, logY: true,
        // first order: a line ∝ dt through the visitor's smallest-dt point
        reference: (rows) => {
            const last = rows.filter(r => r.maxDrift > 0).at(-1);
            return last ? { label: '∝ dt', points: [DTS.at(-1), DTS[0]].map(dt => [dt, last.maxDrift * dt / last.dt]) } : { label: '∝ dt', points: [] };
        },
    }],
    baseline: 'CI (NVE kernel on a CPU-emulated GPU, N = 512, ρ = 0.8, t = 4): max |ΔE/E₀| 4.8e-3, 2.6e-3, 1.3e-3 and '
        + '6.6e-4 at dt = 0.016 … 0.002, a factor 2 per halving (first order). The simple cubic start is unstable, so the '
        + 'lattice melts and K grows about 47×, while E stays put. With the thermostat (t = 2) the change is 6.7e-2 at '
        + 'dt = 0.016 and 1.0e-1 at 0.008: it follows the thermostat, not dt.',
    async *run(ctx, { n, thermostat }) {
        const N = Math.min(n, MAX_MD_ATOMS);
        const boxSize = boxFor(N);
        const atoms = generateAtoms(N, boxSize, TEMPERATURE, mulberry32(SEED));
        for (const dt of DTS) {
            await ctx.step();
            const steps = Math.round(DRIFT_TIME / dt);
            const r = await measureMD(ctx.gpuDevice, { atoms, boxSize, dt, temperature: TEMPERATURE, thermostat, steps,
                samples: DRIFT_SAMPLES, between: () => ctx.step() });
            const energies = r.snapshots.map(s => ljEnergy(s.atoms, N, boxSize));
            const E0 = energies[0].E;
            const drifts = energies.map(e => (e.E - E0) / Math.abs(E0));
            yield {
                dt, steps, maxDrift: Math.max(...drifts.map(Math.abs)), finalDrift: drifts.at(-1),
                kineticRatio: energies.at(-1).K / energies[0].K, wallMs: r.wallMs,
            };
        }
    },
};

export default [scaling, energyDrift];
