// 7. Learning path (learn.html, no WebGPU): track deep links, keyboard switching, and links that land on
// real example tabs of index.html.
import { test, expect, openSite, expectSelected } from './fixtures.mjs';

const trackTab = (page, id) => page.locator(`#track-${id}-tab`);
const trackPanel = (page, id) => page.locator(`#track-${id}-panel`);

async function expectTrack(page, selected) {
    for (const id of ['gpu', 'science']) {
        await expect(trackTab(page, id)).toHaveAttribute('aria-selected', String(id === selected));
        if (id === selected) await expect(trackPanel(page, id)).toBeVisible();
        else await expect(trackPanel(page, id)).toBeHidden();
    }
}

test('learn.html#science opens the science track', async ({ page }) => {
    await page.goto('learn.html#science');
    await expectTrack(page, 'science');
    expect(await trackPanel(page, 'science').locator('li.step-card').count()).toBeGreaterThan(3);
});

test('arrow keys, Home and End switch tracks', async ({ page }) => {
    await page.goto('learn.html');
    await expectTrack(page, 'gpu');
    await trackTab(page, 'gpu').focus();
    for (const [key, expected] of [['ArrowRight', 'science'], ['ArrowRight', 'gpu'], ['ArrowLeft', 'science'],
        ['Home', 'gpu'], ['End', 'science']]) {
        await page.keyboard.press(key);
        await expectTrack(page, expected);
        await expect(trackTab(page, expected)).toBeFocused();
        expect(new URL(page.url()).hash).toBe(`#${expected}`);
    }
    await page.goBack();
    await expectTrack(page, 'gpu');
});

test('every step link points at an existing example tab, and following one opens it', async ({ page, request }) => {
    await page.goto('learn.html');
    const hrefs = await page.locator('.step-example-links a').evaluateAll(as => as.map(a => a.getAttribute('href')));
    expect(hrefs.length).toBeGreaterThanOrEqual(75);   // 25 examples × 3 views, at least once each
    const html = await (await request.get('index.html')).text();
    const ids = new Set([...html.matchAll(/<section class="example" id="([^"]+)">/g)].map(m => m[1]));
    const bad = hrefs.filter((href) => {
        const m = /^index\.html#([\w-]+)(?:\/(understand|measure))?$/.exec(href);
        return !m || !ids.has(m[1]);
    });
    expect(bad).toEqual([]);

    await page.locator('.step-example-links a[href="index.html#nbody/measure"]').first().click();
    await expect(page).toHaveURL(/index\.html#nbody\/measure$/);
    await expect(page.locator('#gpu-status')).toHaveClass('status supported', { timeout: 30_000 });
    await expectSelected(page, 'nbody', 'measure');
});

test('the catalog links to the learning path', async ({ page }) => {
    await openSite(page);
    await expect(page.locator('.hero-links a[href="learn.html"]')).toBeVisible();
});
