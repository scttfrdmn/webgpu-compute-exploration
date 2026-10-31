// WASM + WebGPU Mandelbrot (example #10).
// CPU replica of the `compute` shader (f32 via Math.fround); validates the algorithm and index math,
// not the WGSL itself — see #51.
//
// The independent reference is wasm/src/lib.rs `compute_mandelbrot_cpu` (float64), transcribed to JS
// below because js/wasm-pkg/ is a build artifact that is not present in CI's test job. The uniform
// bytes come from lib.rs `MandelbrotConfig::get_shader_params`, also transcribed (runWASMMandelbrot
// itself needs the built WASM package and a DOM canvas, so it is not driven here).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { wgslModules } from '../js/examples/wasm-mandelbrot.js';
import { structLayout, decodeStruct, moduleCode } from './helpers/wgsl-layout.mjs';

const mod = { wgslModules };
const f = Math.fround;

// Transcription of lib.rs MandelbrotConfig (defaults from MandelbrotConfig::new).
function config(width, height) {
    return { width, height, max_iterations: 256, zoom: 1.0, center_x: -0.5, center_y: 0.0 };
}
const shaderParams = c => new Float32Array([c.width, c.height, c.max_iterations, c.zoom, c.center_x, c.center_y]);

// Transcription of lib.rs compute_mandelbrot_cpu (float64).
function rustReference(c) {
    const { width, height, max_iterations } = c;
    const out = new Float32Array(width * height);
    const scale = 4.0 / (c.zoom * Math.min(width, height));
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const cx = c.center_x + (x - width / 2.0) * scale;
            const cy = c.center_y + (y - height / 2.0) * scale;
            let zx = 0, zy = 0, it = 0;
            while (zx * zx + zy * zy < 4.0 && it < max_iterations) {
                const t = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = t;
                it++;
            }
            out[y * width + x] = it;
        }
    return out;
}

// Replica of the WGSL main(): params are f32 uniforms, all arithmetic in f32.
function replica(params) {
    const p = decodeStruct(params, structLayout(moduleCode(mod, 'compute'), 'Params'));
    const width = Math.trunc(p.width) >>> 0, height = Math.trunc(p.height) >>> 0;
    const out = new Float32Array(width * height);
    const scale = f(4.0 / f(p.zoom * Math.min(width, height)));
    const maxIter = Math.trunc(p.max_iterations) >>> 0;
    for (let gy = 0; gy < Math.ceil(height / 8) * 8; gy++)
        for (let gx = 0; gx < Math.ceil(width / 8) * 8; gx++) {
            if (gx >= width || gy >= height) continue;
            const cx = f(p.center_x + f(f(gx - f(width / 2)) * scale));
            const cy = f(p.center_y + f(f(gy - f(height / 2)) * scale));
            let zx = 0, zy = 0, it = 0;
            for (let i = 0; i < maxIter; i++) {
                if (f(f(zx * zx) + f(zy * zy)) >= 4.0) break;
                const t = f(f(f(zx * zx) - f(zy * zy)) + cx);
                zy = f(f(f(2 * zx) * zy) + cy);
                zx = t;
                it++;
            }
            out[gy * width + gx] = it;
        }
    return out;
}

test('Params uniform: get_shader_params order matches the WGSL struct', (t) => {
    const layout = structLayout(moduleCode(mod, 'compute'), 'Params');
    t.diagnostic(`Params offsets ${JSON.stringify(layout.offsets)} size=${layout.size}`);
    assert.deepEqual(layout.offsets, { width: 0, height: 4, max_iterations: 8, zoom: 12, center_x: 16, center_y: 20 });
    const bytes = shaderParams({ width: 800, height: 600, max_iterations: 256, zoom: 2.5, center_x: -0.75, center_y: 0.1 });
    assert.equal(bytes.byteLength, layout.size);
    const d = decodeStruct(bytes, layout);
    assert.deepEqual(d, { width: 800, height: 600, max_iterations: 256, zoom: 2.5, center_x: -0.75, center_y: f(0.1) });
});

test('transcription and replica match wasm/src/lib.rs and the WGSL source', () => {
    const rs = readFileSync(new URL('../wasm/src/lib.rs', import.meta.url), 'utf8');
    assert.match(rs, /max_iterations: 256,\s*zoom: 1\.0,\s*center_x: -0\.5,\s*center_y: 0\.0,/);
    assert.match(rs, /self\.width as f32,\s*self\.height as f32,\s*self\.max_iterations as f32,\s*self\.zoom as f32,\s*self\.center_x as f32,\s*self\.center_y as f32,/);
    assert.match(rs, /let scale = 4\.0 \/ \(config\.zoom \* width\.min\(height\) as f64\);/);
    const wgsl = moduleCode(mod, 'compute');
    assert.match(wgsl, /let scale = 4\.0 \/ \(params\.zoom \* f32\(min\(width, height\)\)\);/);
    assert.match(wgsl, /if \(zx \* zx \+ zy \* zy >= 4\.0\)/);
    assert.match(rs, /while zx \* zx \+ zy \* zy < 4\.0 && iteration < max_iterations/);
});

test('f32 shader replica vs float64 Rust reference: mismatches only on the set boundary', (t) => {
    const c = config(200, 150);
    const gpu = replica(shaderParams(c));
    const ref = rustReference(c);
    const n = c.width * c.height;
    let mismatches = 0, maxDiff = 0, minRefAtMismatch = Infinity, inSet = 0;
    for (let i = 0; i < n; i++) {
        if (ref[i] === 256) inSet++;
        if (gpu[i] !== ref[i]) {
            mismatches++;
            maxDiff = Math.max(maxDiff, Math.abs(gpu[i] - ref[i]));
            minRefAtMismatch = Math.min(minRefAtMismatch, ref[i], gpu[i]);
        }
    }
    t.diagnostic(`${c.width}x${c.height}: in-set ${inSet}, mismatched pixels ${mismatches} (${(100 * mismatches / n).toFixed(3)}%), ` +
        `max |Δiter| ${maxDiff}, min iteration count at a mismatch ${minRefAtMismatch}`);
    // The center pixel c = -0.5 + 0i is inside the main cardioid.
    assert.equal(gpu[75 * 200 + 100], 256);
    assert.equal(ref[75 * 200 + 100], 256);
    // f32 vs f64 only diverges where the orbit is chaotic, i.e. near the boundary: few pixels, and
    // only pixels whose escape takes many iterations.
    assert.ok(mismatches / n < 0.01, 'fewer than 1% of pixels differ');
    assert.ok(minRefAtMismatch >= 10, 'quickly-escaping pixels always agree');
});
