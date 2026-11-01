// 2. Explore / Understand / Measure tabs: structure, mouse and keyboard switching, deep links, history.
import { test, expect, openSite, tab, panel, expectSelected } from './fixtures.mjs';

test('every example section has three tabs wired to their panels', async ({ page }) => {
    await openSite(page);
    const problems = await page.evaluate(() => {
        const out = [];
        for (const section of document.querySelectorAll('section.example[id]')) {
            const tabs = [...section.querySelectorAll(':scope > .tabs > [role="tab"]')];
            if (tabs.map(t => t.textContent).join('|') !== 'Explore|Understand|Measure') out.push(`${section.id}: tabs ${tabs.length}`);
            for (const t of tabs) {
                const p = document.getElementById(t.getAttribute('aria-controls'));
                if (!p || p.getAttribute('role') !== 'tabpanel' || p.getAttribute('aria-labelledby') !== t.id) out.push(`${t.id}: panel`);
            }
            if (tabs[0]?.getAttribute('aria-selected') !== 'true') out.push(`${section.id}: Explore not selected`);
        }
        return out;
    });
    expect(problems).toEqual([]);
});

test('clicking a tab selects it and pushes a deep link', async ({ page }) => {
    await openSite(page);
    await tab(page, 'game-of-life', 'understand').click();
    await expectSelected(page, 'game-of-life', 'understand');
    expect(new URL(page.url()).hash).toBe('#game-of-life/understand');
    await tab(page, 'game-of-life', 'measure').click();
    await expectSelected(page, 'game-of-life', 'measure');
    expect(new URL(page.url()).hash).toBe('#game-of-life/measure');
    await tab(page, 'game-of-life', 'explore').click();
    await expectSelected(page, 'game-of-life', 'explore');
    expect(new URL(page.url()).hash).toBe('#game-of-life');
});

test('arrow keys, Home and End move between tabs and focus', async ({ page }) => {
    await openSite(page);
    const id = 'vector-addition';
    await tab(page, id, 'explore').focus();
    const steps = [['ArrowRight', 'understand'], ['ArrowRight', 'measure'], ['ArrowRight', 'explore'],
        ['ArrowLeft', 'measure'], ['Home', 'explore'], ['End', 'measure'], ['ArrowLeft', 'understand']];
    for (const [key, expected] of steps) {
        await page.keyboard.press(key);
        await expectSelected(page, id, expected);
        await expect(tab(page, id, expected)).toBeFocused();
    }
});

test('deep link #nbody/understand opens the Understand tab with rendered math', async ({ page }) => {
    await openSite(page, 'index.html#nbody/understand');
    await expectSelected(page, 'nbody', 'understand');
    const article = panel(page, 'nbody', 'understand').locator('article.understand');
    await expect(article).toBeVisible();
    expect(await article.locator('math').count()).toBeGreaterThan(0);
    expect((await article.textContent()).length).toBeGreaterThan(1000);
    // Other examples stay on Explore
    await expectSelected(page, 'boids', 'explore');
});

test('deep link #nbody/measure mounts the experiments', async ({ page }) => {
    await openSite(page, 'index.html#nbody/measure');
    await expectSelected(page, 'nbody', 'measure');
    const runs = panel(page, 'nbody', 'measure').locator('.experiment button.primary');
    expect(await runs.count()).toBeGreaterThan(0);
    await expect(runs.first()).toBeEnabled();
});

test('back and forward restore the tabs', async ({ page }) => {
    await openSite(page, 'index.html#nbody/understand');
    await expectSelected(page, 'nbody', 'understand');
    await tab(page, 'nbody', 'measure').click();
    await expectSelected(page, 'nbody', 'measure');
    await tab(page, 'nbody', 'explore').click();
    await expectSelected(page, 'nbody', 'explore');

    await page.goBack();
    expect(new URL(page.url()).hash).toBe('#nbody/measure');
    await expectSelected(page, 'nbody', 'measure');
    await page.goBack();
    expect(new URL(page.url()).hash).toBe('#nbody/understand');
    await expectSelected(page, 'nbody', 'understand');
    await page.goForward();
    expect(new URL(page.url()).hash).toBe('#nbody/measure');
    await expectSelected(page, 'nbody', 'measure');
    await page.goForward();
    expect(new URL(page.url()).hash).toBe('#nbody');
    await expectSelected(page, 'nbody', 'explore');
});
