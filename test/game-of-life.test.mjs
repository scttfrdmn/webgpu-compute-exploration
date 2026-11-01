// Conway's Game of Life (example #4).
// CPU replica of the `compute` shader (toroidal B3/S23 step); validates the algorithm and index math,
// not the WGSL itself — see #51.
//
// Shipped code exercised directly: GameOfLife constructed on a mock device (dimensions uniform bytes,
// buffer sizes) and its loadPattern()/SEED_PATTERNS (the grid bytes it uploads). The replica is checked
// against an independent set-of-live-coordinates Life implementation and against the known periods
// and displacements of the seed patterns.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GameOfLife, SEED_PATTERNS, wgslModules } from '../js/examples/game-of-life.js';
import { mockDevice, stub } from './helpers/mock-gpu.mjs';
import { typeLayout } from './helpers/wgsl-layout.mjs';
import { lifeStep } from '../js/refs/life.js';

const fakeCanvas = () => ({ width: 64, height: 64, getContext: () => stub('ctx') });

function makeGame(w, h) {
    const device = mockDevice();
    const game = new GameOfLife(device, fakeCanvas(), w, h);
    return { device, game };
}

// Replica of main() + getCell(): i32 wrap (x + width) % width, row-major y * width + x. It lives in
// js/refs/life.js as lifeStep (deployable, for in-browser comparisons); the 8x8 workgroup grid only
// orders the cells, so the reference iterates them directly.
const replicaStep = lifeStep;

// Independent reference: a Map of live cells, counting neighbor contributions per live cell.
function setStep(live, width, height) {
    const counts = new Map();
    for (const key of live) {
        const x = key % width, y = Math.floor(key / width);
        for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                const k = ((y + dy + height) % height) * width + ((x + dx + width) % width);
                counts.set(k, (counts.get(k) || 0) + 1);
            }
    }
    const next = new Set();
    for (const [k, c] of counts) if (c === 3 || (c === 2 && live.has(k))) next.add(k);
    return next;
}
const toSet = grid => new Set([...grid.keys()].filter(i => grid[i]));
const sameSet = (a, b) => a.size === b.size && [...a].every(k => b.has(k));

function loadedGrid(name, w, h) {
    const { device, game } = makeGame(w, h);
    game.loadPattern(name);
    const write = device.lastWrite(game.gridBuffers[game.currentBuffer]);
    return { grid: new Uint32Array(write.bytes.buffer), game };
}

test('dimensions uniform is vec2<u32> written as integers; grid buffers sized w*h*4', (t) => {
    const { device, game } = makeGame(100, 75);
    const dims = new Uint32Array(game.dimensionsBuffer.getMappedRange());
    t.diagnostic(`dimensions = [${[...dims]}], grid buffer bytes = ${game.gridBuffers[0].size}`);
    assert.equal(game.dimensionsBuffer.size, typeLayout('vec2<u32>').size);
    assert.deepEqual([...dims], [100, 75]);
    assert.equal(game.gridBuffers[0].size, 100 * 75 * 4);
    for (const { code } of wgslModules()) assert.match(code, /var<uniform> dimensions: vec2<u32>/);
    assert.equal(device.buffers.length, 3);
});

test('loadPattern centers every seed pattern and uploads exactly its live cells', (t) => {
    const w = 64, h = 48;
    for (const [name, p] of Object.entries(SEED_PATTERNS)) {
        assert.equal(p.cells.length, p.height, `${name} height`);
        for (const row of p.cells) assert.equal(row.length, p.width, `${name} width`);
        const { grid, game } = loadedGrid(name, w, h);
        const expected = p.cells.flat().reduce((a, b) => a + b, 0);
        assert.equal(game.totalCells, expected, `${name} live count`);
        const ox = Math.floor((w - p.width) / 2), oy = Math.floor((h - p.height) / 2);
        for (let y = 0; y < p.height; y++)
            for (let x = 0; x < p.width; x++) assert.equal(grid[(oy + y) * w + ox + x], p.cells[y][x]);
    }
    t.diagnostic(`${Object.keys(SEED_PATTERNS).length} patterns checked`);
});

test('replica matches the set-based reference for 200 generations of a random soup', (t) => {
    const w = 37, h = 29;   // odd sizes so the 8x8 workgroup padding and wrap both matter
    let s = 12345;
    const rand = () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 2 ** 32);
    let grid = Uint32Array.from({ length: w * h }, () => (rand() > 0.7 ? 1 : 0));
    let live = toSet(grid);
    for (let g = 0; g < 200; g++) {
        grid = replicaStep(grid, w, h);
        live = setStep(live, w, h);
        assert.ok(sameSet(toSet(grid), live), `generation ${g + 1} differs`);
    }
    t.diagnostic(`200 generations on ${w}x${h} torus agree; final population ${live.size}`);
});

// Known behavior: [period, dx, dy] — after `period` steps the pattern reappears shifted by (dx, dy).
const KNOWN = {
    glider: [4, 1, 1], lwss: [4, -2, 0], pulsar: [3, 0, 0], pentadecathlon: [15, 0, 0],
};

test('seed patterns from loadPattern have their known periods and displacements', (t) => {
    const w = 64, h = 48;
    for (const [name, [period, dx, dy]] of Object.entries(KNOWN)) {
        let { grid } = loadedGrid(name, w, h);
        const start = toSet(grid);
        for (let i = 0; i < period; i++) grid = replicaStep(grid, w, h);
        const shifted = new Set([...start].map(k => {
            const x = k % w, y = Math.floor(k / w);
            return ((y + dy + h) % h) * w + ((x + dx + w) % w);
        }));
        assert.ok(sameSet(toSet(grid), shifted), `${name}: period ${period}, shift (${dx},${dy})`);
        t.diagnostic(`${name}: period ${period}, displacement (${dx},${dy}) confirmed`);
    }
});

test('Gosper gun emits a glider every 30 generations; glider wraps around the torus', (t) => {
    const w = 64, h = 48;
    let { grid } = loadedGrid('gosper-gun', w, h);
    const pops = [];
    for (let g = 1; g <= 120; g++) { grid = replicaStep(grid, w, h); if (g % 30 === 0) pops.push(toSet(grid).size); }
    t.diagnostic(`gun population at gen 30/60/90/120: ${pops.join(', ')}`);
    // Each emitted glider adds 5 cells; the gun's own population is periodic with period 30.
    assert.ok(pops[1] - pops[0] === 5 && pops[2] - pops[1] === 5, 'one new glider (5 cells) per 30 generations');

    // A glider travels (w, h) = (64, 48) only via the wrap: after 4*48 generations it is shifted by
    // (48, 48) ≡ (48, 0) on a 64x48 torus.
    let { grid: g2 } = loadedGrid('glider', w, h);
    const start = toSet(g2);
    for (let i = 0; i < 4 * 48; i++) g2 = replicaStep(g2, w, h);
    const expect = new Set([...start].map(k => Math.floor(k / w) * w + ((k % w) + 48) % w));
    assert.ok(sameSet(toSet(g2), expect));
});
