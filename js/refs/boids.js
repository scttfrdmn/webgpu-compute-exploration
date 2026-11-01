// CPU references for the boids example: one step of the compute shader for selected boids, and the
// flock's polarization (order parameter).
// Boids are packed as the shader's array<Boid>: 4 floats each (position xy, velocity xy); params are the
// Params uniform's fields (separationRadius … behaviorMode, boundsMin/Max and mousePos as [x, y]).
// Precision: float64 throughout. Cost: boidsStepAt is O(targets · n) (three distance tests per pair, as in
// the shader); polarization is O(n).

const length2 = (x, y) => Math.hypot(x, y);
function limit(x, y, maxLen) {
    const l = length2(x, y);
    return l > maxLen ? [x / l * maxLen, y / l * maxLen] : [x, y];
}
function safeNormalize(x, y) {
    const l = length2(x, y);
    return l < 1e-6 ? [0, 0] : [x / l, y / l];
}

/**
 * The shader's `main` for the boids listed in `targets`: separation, alignment and cohesion over all n
 * boids (brute force), the mouse force, the behavior mode, the speed limit, the position update and the
 * wrap-around. Returns a Float64Array of 4 floats per target (in the order of `targets`).
 */
export function boidsStepAt(boids, n, p, targets) {
    const out = new Float64Array(4 * targets.length);
    const center = [(p.boundsMin[0] + p.boundsMax[0]) * 0.5, (p.boundsMin[1] + p.boundsMax[1]) * 0.5];
    targets.forEach((idx, t) => {
        const px = boids[4 * idx], py = boids[4 * idx + 1], vx = boids[4 * idx + 2], vy = boids[4 * idx + 3];
        let sx = 0, sy = 0, sc = 0, ax = 0, ay = 0, ac = 0, cx = 0, cy = 0, cc = 0;
        for (let i = 0; i < n; i++) {
            const ox = boids[4 * i], oy = boids[4 * i + 1];
            const dx = px - ox, dy = py - oy, dist = length2(dx, dy);
            if (dist > 0 && dist < p.separationRadius) { sx += dx / dist / dist; sy += dy / dist / dist; sc++; }
            if (dist > 0 && dist < p.alignmentRadius) { ax += boids[4 * i + 2]; ay += boids[4 * i + 3]; ac++; }
            if (dist > 0 && dist < p.cohesionRadius) { cx += ox; cy += oy; cc++; }
        }
        let sep = [0, 0], ali = [0, 0], coh = [0, 0];
        if (sc > 0 && length2(sx / sc, sy / sc) > 0) {
            const [nx, ny] = safeNormalize(sx / sc, sy / sc);
            sep = limit(nx * p.maxSpeed - vx, ny * p.maxSpeed - vy, p.maxForce);
        }
        if (ac > 0) {
            const [nx, ny] = safeNormalize(ax / ac, ay / ac);
            ali = limit(nx * p.maxSpeed - vx, ny * p.maxSpeed - vy, p.maxForce);
        }
        if (cc > 0) {
            const [nx, ny] = safeNormalize(cx / cc - px, cy / cc - py);
            coh = limit(nx * p.maxSpeed - vx, ny * p.maxSpeed - vy, p.maxForce);
        }
        let mouse = [0, 0];
        const mdx = p.mousePos[0] - px, mdy = p.mousePos[1] - py, md = length2(mdx, mdy);
        if (md < p.mouseRadius && md > 0) {
            const k = p.mouseForce * (1 - md / p.mouseRadius);
            mouse = [mdx / md * k, mdy / md * k];
        }
        let acc = [
            sep[0] * p.separationWeight + ali[0] * p.alignmentWeight + coh[0] * p.cohesionWeight + mouse[0],
            sep[1] * p.separationWeight + ali[1] * p.alignmentWeight + coh[1] * p.cohesionWeight + mouse[1],
        ];
        if (p.behaviorMode === 1) {
            if (idx === 0) {
                const [nx, ny] = safeNormalize(center[0] - px, center[1] - py);
                acc = [nx * p.maxSpeed * 0.5, ny * p.maxSpeed * 0.5];
            } else {
                const dx = px - boids[0], dy = py - boids[1], dist = length2(dx, dy);
                if (dist < 150 && dist > 0) acc = [acc[0] + dx / dist * p.maxForce * 3, acc[1] + dy / dist * p.maxForce * 3];
            }
        } else if (p.behaviorMode === 2) {
            const [nx, ny] = safeNormalize(px - center[0], py - center[1]);
            acc = [acc[0] * 0.1 + nx * p.maxForce * 2, acc[1] * 0.1 + ny * p.maxForce * 2];
        } else if (p.behaviorMode === 3) {
            const dx = px - center[0], dy = py - center[1], dist = length2(dx, dy);
            if (dist > 10) {
                const [nx, ny] = safeNormalize(-dy - dx / dist * 0.3, dx - dy / dist * 0.3);
                acc = [acc[0] + nx * p.maxSpeed * 2, acc[1] + ny * p.maxSpeed * 2];
            }
        }
        const [nvx, nvy] = limit(vx + acc[0], vy + acc[1], p.maxSpeed);
        let x = px + nvx, y = py + nvy;
        if (x < p.boundsMin[0]) x = p.boundsMax[0];
        if (x > p.boundsMax[0]) x = p.boundsMin[0];
        if (y < p.boundsMin[1]) y = p.boundsMax[1];
        if (y > p.boundsMax[1]) y = p.boundsMin[1];
        out.set([x, y, nvx, nvy], 4 * t);
    });
    return out;
}

/** Polarization |Σ v̂| / n of the headings (1 = all boids fly the same way, ≈ 1/√n for random headings). */
export function polarization(boids, n) {
    let sx = 0, sy = 0, m = 0;
    for (let i = 0; i < n; i++) {
        const vx = boids[4 * i + 2], vy = boids[4 * i + 3], l = Math.hypot(vx, vy);
        if (l > 0) { sx += vx / l; sy += vy / l; m++; }
    }
    return m ? Math.hypot(sx, sy) / m : 0;
}
