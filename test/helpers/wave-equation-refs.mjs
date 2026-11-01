// Wave-equation references shared by test/wave-equation.test.mjs and test/gpu/.
//
// makeSim / presetState: the shipped WaveEquation on a mock device and the walls + two time levels
// its loadPreset() uploads. stepReplica: replica of cs_step (5-point Laplacian, u = 0 outside the
// grid, damped leapfrog, sponge, walls, plane-wave source column; no rain), f32 storage.
// energy: the leapfrog scheme's conserved discrete energy E(n+1/2), from js/refs/wave.js.

import { mockDevice, stub } from './mock-gpu.mjs';
import * as wave from '../../js/examples/wave-equation.js';

export { energy } from '../../js/refs/wave.js';

const { WaveEquation } = wave;

// Shader constants (cs_step)
export const SOURCE_X = 30, SOURCE_AMP = 0.6, SPONGE = 24, SPONGE_STRENGTH = 0.08;

globalThis.window ??= { addEventListener() {} };

export function makeSim(width, height) {
    const device = mockDevice();
    const canvas = { width: 1024, height: 768, getContext: () => stub('ctx'), addEventListener() {} };
    const sim = new WaveEquation(device, canvas, width, height);
    return { sim, device };
}

// Walls and the two identical time levels uploaded by the shipped loadPreset()
export function presetState(key, width, height) {
    const { sim, device } = makeSim(width, height);
    device.writes.length = 0;
    sim.loadPreset(key);
    const bytes = buf => new Uint8Array(device.writes.find(w => w.buffer === buf).bytes);
    const walls = new Uint32Array(bytes(sim.wallBuffer).buffer.slice(0));
    const cur = new Float32Array(bytes(sim.heightBuffers[0]).buffer.slice(0));
    const prev = new Float32Array(bytes(sim.heightBuffers[1]).buffer.slice(0));
    return { sim, walls, cur, prev };
}

// One cs_step dispatch: reads cur = u(t), prev = u(t-1); writes u(t+1) into prev (in place).
export function stepReplica(s, o) {
    const { W, H, cur, prev, walls } = s;
    const { courant2: C2, damping, absorb, planeWave, step, period } = o;
    const at = (x, y) => (x < 0 || y < 0 || x >= W || y >= H) ? 0 : cur[y * W + x];
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const idx = y * W + x;
            if (walls && walls[idx] !== 0) { prev[idx] = 0; continue; }
            const c = cur[idx];
            const lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * c;
            let g = damping;
            if (absorb) {
                const edge = Math.min(Math.min(x, y), Math.min(W - 1 - x, H - 1 - y));
                if (edge < SPONGE) {
                    const t = (SPONGE - edge) / SPONGE;
                    g += SPONGE_STRENGTH * t * t;
                }
            }
            let next = (2 * c - (1 - g) * prev[idx] + C2 * lap) / (1 + g);
            if (planeWave && x === SOURCE_X) {
                next = SOURCE_AMP * Math.sin(2 * Math.PI * ((step % period) / period));
            }
            prev[idx] = next;   // Float32Array store rounds to f32
        }
    }
    // swap: u(t+1) becomes current
    const t = s.cur; s.cur = s.prev; s.prev = t;
}
