// CPU reference for the ray-marching example: its four distance estimators, the camera ray of fs_main,
// and the sphere-tracing loop of rayMarch with its exit recorded (as the Measure tab's statistics kernel
// records it on the GPU).
// Precision: float64 throughout (the shader is f32, so pixels whose rays graze the surface can end
// differently; compare outcome rates, not single pixels). Cost: up to maxSteps distance evaluations per
// pixel, each up to 15 fractal iterations: spot-check a few hundred pixels, not a whole frame.

const len = v => Math.hypot(v[0], v[1], v[2]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm = a => scl(a, 1 / len(a));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function mandelbulbDE(pos, power) {
    let z = pos, dr = 1, r = 0;
    for (let i = 0; i < 15; i++) {
        r = len(z);
        if (r > 2) break;
        const theta = Math.acos(z[2] / r), phi = Math.atan2(z[1], z[0]);
        dr = Math.pow(r, power - 1) * power * dr + 1;
        const zr = Math.pow(r, power), nt = theta * power, np = phi * power;
        z = add([zr * Math.sin(nt) * Math.cos(np), Math.sin(np) * Math.sin(nt) * zr, zr * Math.cos(nt)], pos);
    }
    return 0.5 * Math.log(r) * r / dr;
}

const JC = [0.18, 0.88, 0.24, 0.16];
function juliaDE(pos) {
    let z = [pos[0], pos[1], pos[2], 0], md2 = 1;
    const dot4 = q => q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let mz2 = dot4(z);
    for (let i = 0; i < 12; i++) {
        md2 *= 4 * mz2;
        z = [z[0] * z[0] - (z[1] * z[1] + z[2] * z[2] + z[3] * z[3]) + JC[0],
            2 * z[0] * z[1] + JC[1], 2 * z[0] * z[2] + JC[2], 2 * z[0] * z[3] + JC[3]];
        mz2 = dot4(z);
        if (mz2 > 4) break;
    }
    return 0.25 * Math.sqrt(mz2 / md2) * Math.log(mz2);
}

function mengerDE(pos) {
    const q = pos.map(c => Math.abs(c) - 1);
    let d = len(q.map(c => Math.max(c, 0))) + Math.min(Math.max(q[0], q[1], q[2]), 0);
    let s = 1;
    for (let i = 0; i < 4; i++) {
        const a = pos.map(c => { const x = c * s; return x - 2 * Math.floor(x / 2) - 1; });
        s *= 3;
        const r = a.map(c => Math.abs(1 - 3 * Math.abs(c)));
        d = Math.max(d, (Math.min(Math.max(r[0], r[1]), Math.max(r[1], r[2]), Math.max(r[2], r[0])) - 1) / s);
    }
    return d;
}

function mandelboxDE(pos) {
    let p = pos, dr = 1;
    for (let i = 0; i < 12; i++) {
        p = p.map(c => Math.min(Math.max(c, -1), 1) * 2 - c);
        const r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if (r2 < 0.25) { p = scl(p, 4); dr *= 4; } else if (r2 < 1) { p = scl(p, 1 / r2); dr /= r2; }
        p = add(scl(p, -1.5), pos);
        dr = dr * 1.5 + 1;
    }
    return len(p) / Math.abs(dr);
}

/** sceneSDF for fractalType 0 … 3 (Mandelbulb with `power`, Julia, Menger, Mandelbox scaled by 1/3). */
export function sceneDE(fractalType, power = 8) {
    return [
        p => mandelbulbDE(p, power),
        juliaDE,
        mengerDE,
        p => mandelboxDE(scl(p, 3)) / 3,
    ][fractalType];
}

const rotX = (p, a) => [p[0], Math.cos(a) * p[1] - Math.sin(a) * p[2], Math.sin(a) * p[1] + Math.cos(a) * p[2]];
const rotY = (p, a) => [Math.cos(a) * p[0] + Math.sin(a) * p[2], p[1], -Math.sin(a) * p[0] + Math.cos(a) * p[2]];

/** fs_main's ray for pixel coordinates (px, py) (use pixel centers, x + 0.5) of a W × H frame; u: the camera uniforms. */
export function cameraRay(px, py, W, H, u) {
    const uv = [(px - W * 0.5) / H, -((py - H * 0.5) / H)];
    const ro = rotY(rotX([0, 0, u.cameraZoom], u.cameraRotX), u.cameraRotY + u.time * 0.1);
    const forward = norm(scl(ro, -1));
    const right = norm(cross([0, 1, 0], forward));
    const up = cross(forward, right);
    return { ro, rd: norm(add(add(forward, scl(right, uv[0])), scl(up, uv[1]))) };
}

/**
 * rayMarch's loop: at most maxSteps distance evaluations, a hit when d < 0.001, an escape when t > 50,
 * steps of 0.5·d. Returns { t (−1 unless hit), steps, outcome: 0 hit / 1 escaped / 2 exhausted, d }.
 */
export function marchRay(sdf, ro, rd, maxSteps) {
    let t = 0, d = 0;
    for (let i = 0; i < maxSteps; i++) {
        d = sdf(add(ro, scl(rd, t)));
        if (d < 0.001) return { t, steps: i + 1, outcome: 0, d };
        if (t > 50) return { t: -1, steps: i + 1, outcome: 1, d };
        t += d * 0.5;
    }
    return { t: -1, steps: maxSteps, outcome: 2, d };
}
