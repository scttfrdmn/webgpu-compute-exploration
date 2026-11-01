// 4. Measure: a cheap experiment produces rows and charts; Cancel stops a run and re-enables Run; switching
//    tabs during a run doesn't break it.
// 5. No timestamp-query: GPU-ms cells show "—" and GPU-only charts show their empty-state note. The feature
//    is hidden from the page by an init script, so this holds whether or not the adapter offers it.
import { test, expect, openSite, tab, panel, experimentCard, hideTimestampQuery } from './fixtures.mjs';

const runButtons = (page) => page.locator('.experiment button.primary');

async function runToEnd(card, timeout = 60_000) {
    await card.locator('button.primary').click();
    await expect(card.locator('.experiment-status')).toHaveText(/^Done: \d+ steps\.$/, { timeout });
}

test('vector addition: Size vs bandwidth produces rows and charts', async ({ page }) => {
    await openSite(page, 'index.html#vector-addition/measure');
    const card = experimentCard(page, 'vector-addition', 'Size vs bandwidth');
    await runToEnd(card);
    const rows = card.locator('.experiment-table tbody tr');
    expect(await rows.count()).toBeGreaterThanOrEqual(5);
    expect(await card.locator('.experiment-charts svg').count()).toBeGreaterThan(0);
    // The last column counts sums that differ from the correctly rounded f32 sum: always 0
    const mismatches = await rows.evaluateAll(trs => trs.map(tr => tr.lastElementChild.textContent));
    expect(mismatches.every(v => v === '0')).toBe(true);
    await expect(card.locator('button', { hasText: 'Copy CSV' })).toBeVisible();
    await expect(runButtons(page).first()).toBeEnabled();
});

test('Cancel stops a running experiment and re-enables every Run button', async ({ page }) => {
    await openSite(page, 'index.html#noise/measure');
    const card = experimentCard(page, 'noise', 'Octaves and resolution');
    await page.locator('#noise-cost-side').selectOption('1024');
    await card.locator('button.primary').click();
    await expect(card.locator('.experiment-status')).toHaveText(/^Running/);
    await expect(runButtons(page).first()).toBeDisabled();   // one experiment at a time, page-wide
    await card.locator('button', { hasText: 'Cancel' }).click();
    await expect(card.locator('.experiment-status')).toHaveText(/^Cancelled/, { timeout: 60_000 });
    await expect(card.locator('button', { hasText: 'Cancel' })).toBeDisabled();
    for (const button of await runButtons(page).all()) await expect(button).toBeEnabled();
});

test('switching tabs during a run leaves the run and the page working', async ({ page }) => {
    await openSite(page, 'index.html#vector-addition/measure');
    const card = experimentCard(page, 'vector-addition', 'Size vs bandwidth');
    await card.locator('button.primary').click();
    await tab(page, 'vector-addition', 'explore').click();
    await tab(page, 'vector-addition', 'understand').click();
    await tab(page, 'vector-addition', 'measure').click();
    await tab(page, 'game-of-life', 'understand').click();
    await expect(card.locator('.experiment-status')).toHaveText(/^Done: \d+ steps\.$/, { timeout: 60_000 });
    for (const button of await runButtons(page).all()) await expect(button).toBeEnabled();
    expect(await card.locator('.experiment-table tbody tr').count()).toBeGreaterThanOrEqual(5);
    await expect(panel(page, 'vector-addition', 'understand').locator('article.understand')).toBeHidden();
    await expect(panel(page, 'game-of-life', 'understand').locator('article.understand')).toBeVisible();
    // The page still runs a second experiment after the switches
    await runToEnd(card);
});

test.describe('without timestamp-query', () => {
    test.beforeEach(async ({ page }) => { await page.addInitScript(hideTimestampQuery); });

    test('status says GPU time is unavailable and GPU-ms cells show —', async ({ page }) => {
        const status = await openSite(page, 'index.html#vector-addition/measure');
        expect(status).toContain('GPU time not available');
        const card = experimentCard(page, 'vector-addition', 'Size vs bandwidth');
        await runToEnd(card);
        const gpuCells = await card.locator('.experiment-table').evaluate((table) => {
            const col = [...table.querySelectorAll('thead th')].findIndex(th => th.textContent === 'kernel GPU ms');
            return [...table.querySelectorAll('tbody tr')].map(tr => tr.cells[col].textContent);
        });
        expect(gpuCells.length).toBeGreaterThan(0);
        expect(gpuCells.every(v => v === '—')).toBe(true);
    });

    test('a GPU-time-only chart shows its empty-state note', async ({ page }) => {
        await openSite(page, 'index.html#noise/measure');
        const card = experimentCard(page, 'noise', 'Octaves and resolution');
        await page.locator('#noise-cost-side').selectOption('512');
        await card.locator('button.primary').click();
        // On a slow software GPU the run may stop at its 20 s budget; the charts only need some rows
        await expect(card.locator('.experiment-status')).toHaveText(/^Done: \d+ steps\.$|time budget after \d+ steps\.$/, { timeout: 60_000 });
        const figures = card.locator('.experiment-charts figure.chart');
        await expect(figures.first().locator('.chart-empty')).toContainText('timestamp-query');
        await expect(figures.nth(1).locator('svg.chart-svg')).toHaveCount(1);   // the wall-time chart still plots
    });
});
