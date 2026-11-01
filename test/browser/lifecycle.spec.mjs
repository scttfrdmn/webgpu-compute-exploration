// 3. Example lifecycle (Game of Life, 200×150): leaving Explore stops it for good; scrolling it off screen
// pauses it and scrolling back resumes it, unless the user paused it.
import { test, expect, openSite, tab, expectSelected } from './fixtures.mjs';

const output = (page) => page.locator('#output-gol');

async function startLife(page) {
    await openSite(page, 'index.html#game-of-life');
    await page.locator('#canvas-gol').scrollIntoViewIfNeeded();
    await expect(output(page)).toContainText('Game of Life Active');   // initialized: buttons are wired
    await page.locator('#btn-gol-random').click();
    await page.locator('#btn-gol-start').click();
    await expect(output(page)).toContainText('Status: Running');
}

test('leaving Explore stops the example and returning does not restart it', async ({ page }) => {
    await startLife(page);
    await tab(page, 'game-of-life', 'understand').click();
    await expectSelected(page, 'game-of-life', 'understand');
    await expect(output(page)).toContainText('Status: Paused');

    await tab(page, 'game-of-life', 'explore').click();
    await expectSelected(page, 'game-of-life', 'explore');
    await page.locator('#canvas-gol').scrollIntoViewIfNeeded();
    await page.waitForTimeout(1000);
    await expect(output(page)).toContainText('Status: Paused');
});

test('scrolling off screen pauses and scrolling back resumes', async ({ page }) => {
    await startLife(page);
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    await expect(output(page)).toContainText('Status: Paused');
    await page.locator('#canvas-gol').scrollIntoViewIfNeeded();
    await expect(output(page)).toContainText('Status: Running');
    await page.locator('#btn-gol-pause').click();   // leave it stopped for the next page
    await expect(output(page)).toContainText('Status: Paused');
});

test('a user pause survives scrolling away and back', async ({ page }) => {
    await startLife(page);
    await page.locator('#btn-gol-pause').click();
    await expect(output(page)).toContainText('Status: Paused');
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    await page.waitForTimeout(500);
    await page.locator('#canvas-gol').scrollIntoViewIfNeeded();
    await page.waitForTimeout(1000);
    await expect(output(page)).toContainText('Status: Paused');
});
