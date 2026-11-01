// 6. Device loss. An init script gives the device a `lost` promise the test resolves (reason 'unknown'),
// which runs the site's real handleDeviceLost path without losing the software GPU itself.
import { test, expect, openSite, tab, panel, experimentCard, expectSelected, controllableDeviceLoss } from './fixtures.mjs';

// Enabled controls of the examples themselves (Explore panels). Measure panels are checked through their
// Run buttons: their setting selects and hidden Copy CSV buttons do no GPU work.
const enabledControls = (page) => page.evaluate(() =>
    [...document.querySelectorAll('.tab-panel-explore :is(button, input, select)')]
        .filter(c => !c.disabled).map(c => c.id || c.textContent.trim()));

test('a lost device disables the examples, keeps tabs usable and offers a reload', async ({ page }) => {
    await page.addInitScript(controllableDeviceLoss);
    await openSite(page, 'index.html#vector-addition/measure');
    const vectorCard = experimentCard(page, 'vector-addition', 'Size vs bandwidth');
    await expect(vectorCard.locator('button.primary')).toBeEnabled();
    await expect(page.locator('#btn-reload')).toHaveCount(0);

    await page.evaluate(() => window.__loseTestDevice('simulated by the browser tests'));
    await expect(page.locator('#gpu-status')).toHaveText('GPU device lost: simulated by the browser tests');
    await expect(page.locator('#gpu-status')).toHaveClass('status not-supported');
    await expect(page.locator('#btn-reload')).toBeVisible();
    await expect(page.locator('#btn-reload')).toBeEnabled();
    await expect(page.locator('#btn-device-report')).toBeEnabled();
    await expect(vectorCard.locator('button.primary')).toBeDisabled();
    expect(await enabledControls(page)).toEqual([]);

    // Tabs still work, and explanations still load
    await tab(page, 'game-of-life', 'understand').click();
    await expectSelected(page, 'game-of-life', 'understand');
    await expect(panel(page, 'game-of-life', 'understand').locator('article.understand')).toBeVisible();

    // A Measure panel mounted after the loss starts with Run disabled
    await tab(page, 'noise', 'measure').click();
    await expectSelected(page, 'noise', 'measure');
    const noiseRuns = panel(page, 'noise', 'measure').locator('.experiment button.primary');
    await expect(noiseRuns.first()).toBeVisible();
    for (const run of await noiseRuns.all()) await expect(run).toBeDisabled();

    // Examples that come into view after the loss must not re-enable their controls
    for (const id of ['canvas-boids', 'canvas-mnist-train', 'canvas-nbody']) {
        await page.locator(`#${id}`).scrollIntoViewIfNeeded();
        await page.waitForTimeout(500);
    }
    expect(await enabledControls(page)).toEqual([]);
    for (const run of await page.locator('.experiment button.primary').all()) await expect(run).toBeDisabled();
});
