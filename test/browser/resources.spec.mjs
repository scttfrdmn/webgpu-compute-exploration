// 11. No GPU resource growth (#106): repeating a Measure experiment or an Explore cycle must not leave GPU
// buffers, textures or query sets behind. An init script (trackGpuResources, test-only) counts live objects,
// created − destroyed, per device. The count after the first repetition is the baseline: it includes
// everything created once and kept on purpose (the example's own buffers, pipelines' uniforms, a lazily
// initialized neighbour). Every later repetition must end at or below it, with no slack: the site creates
// GPU objects only through createBuffer/createTexture/createQuerySet and frees them with destroy(), and
// pipeline or shader-module caches are not counted (they are not buffers, textures or query sets).
// Garbage collection does not count as freeing: an object dropped without destroy() stays live here.
import { test, expect, openSite, panel, experimentCard, trackGpuResources, gpuResources } from './fixtures.mjs';

const REPEATS = 3;
const FINISHED = /^Done: \d+ steps in [\d.]+ s\.$|time budget \(\d+ steps? completed\)\.$/;

test.beforeEach(async ({ page }) => { await page.addInitScript(trackGpuResources); });

const describe = (r) => `${r.buffer} buffers, ${r.texture} textures, ${r.querySet} query sets`;

// Fails with the objects created after the baseline that are still live
async function expectNoGrowth(page, what, baseline, after) {
    const grew = ['buffer', 'texture', 'querySet'].filter(k => after[k] > baseline[k]);
    if (!grew.length) return;
    const leaked = await page.evaluate((since) => window.__gpuResources.liveSince(since), baseline.seq);
    const lines = leaked.slice(0, 20).map(e => `  ${e.kind} ${e.label ? `"${e.label}" ` : ''}size ${e.size} from ${e.site}`);
    expect(grew, `${what}: live GPU objects grew from ${describe(baseline)} to ${describe(after)}; `
        + `${leaked.length} created since the first repetition are not destroyed:\n${lines.join('\n')}`).toEqual([]);
}

// The smallest setting of each reference investigation (js/teach/flagships.js), run REPEATS times
const EXPERIMENTS = [
    { id: 'matrix-multiplication', title: 'CPU vs naive GPU vs tiled GPU', setting: ['cpu-naive-tiled-maxN', '256'] },
    { id: 'nbody', title: 'Opening angle vs force error', setting: ['theta-error-n', '4096'] },
    { id: 'parallel-sort', title: 'Sorting N keys', setting: ['scaling-maxN', '262144'] },
];

for (const { id, title, setting: [key, value] } of EXPERIMENTS) {
    test(`${id}: repeating "${title}" (${value}) does not grow live GPU objects`, async ({ page }) => {
        test.setTimeout(REPEATS * 45_000);
        await openSite(page, `index.html#${id}/measure`);
        const card = experimentCard(page, id, title);
        await card.locator(`#${id}-${key}`).selectOption(value);
        const counts = [];
        for (let run = 1; run <= REPEATS; run++) {
            const started = Date.now();
            await card.locator('button.primary').click();
            await expect(card.locator('.experiment-status')).toHaveText(FINISHED, { timeout: 40_000 });
            await expect(card.locator('button.primary')).toBeEnabled();
            counts.push(await gpuResources(page));
            console.log(`[resources] ${id} run ${run}: ${describe(counts.at(-1))} live `
                + `(${counts.at(-1).seq} created in total), ${((Date.now() - started) / 1000).toFixed(1)} s`);
        }
        await expectNoGrowth(page, `${id} "${title}"`, counts[0], counts.at(-1));
        expect(await panel(page, id, 'measure').locator('.experiment-table tbody tr').count()).toBeGreaterThan(0);
    });
}

// Interactive examples: Explore cycles that start, pause, reset and reallocate
const CYCLES = [
    {
        id: 'game-of-life', canvas: '#canvas-gol', output: '#output-gol', ready: 'Game of Life Active',
        async cycle(page) {
            await page.locator('#btn-gol-random').click();
            await page.locator('#btn-gol-start').click();
            await expect(page.locator('#output-gol')).toContainText('Status: Running');
            await page.waitForTimeout(500);
            await page.locator('#btn-gol-pause').click();
            await expect(page.locator('#output-gol')).toContainText('Status: Paused');
            await page.locator('#btn-gol-step').click();
            await page.locator('#btn-gol-clear').click();
        },
    },
    {
        id: 'nbody', canvas: '#canvas-nbody', output: '#output-nbody', ready: 'N-Body Simulation Active',
        async cycle(page) {
            const output = page.locator('#output-nbody');
            await page.locator('#btn-nbody-start').click();
            await expect(output).toContainText('Status: Running');
            await page.waitForTimeout(500);
            await page.locator('#btn-nbody-pause').click();
            await expect(output).toContainText('Status: Paused');
            await page.locator('#btn-nbody-reset').click();
            // Reallocation paths: trails on and off (ring buffer), a different N and back (every N-sized buffer)
            await page.locator('#nbody-trails').check();
            await page.locator('#nbody-trails').uncheck();
            await page.locator('#nbody-count').selectOption('4096');
            await page.locator('#nbody-count').selectOption('16384');
            await expect(output).toContainText('Status: Paused');
        },
    },
];

for (const { id, canvas, output, ready, cycle } of CYCLES) {
    test(`${id}: Explore start/pause/reset cycles do not grow live GPU objects`, async ({ page }) => {
        await openSite(page, `index.html#${id}`);
        await page.locator(canvas).scrollIntoViewIfNeeded();
        await expect(page.locator(output)).toContainText(ready);
        const counts = [];
        for (let round = 1; round <= REPEATS; round++) {
            await cycle(page);
            counts.push(await gpuResources(page));
            console.log(`[resources] ${id} cycle ${round}: ${describe(counts.at(-1))} live (${counts.at(-1).seq} created in total)`);
        }
        await expectNoGrowth(page, `${id} Explore cycle`, counts[0], counts.at(-1));
    });
}
