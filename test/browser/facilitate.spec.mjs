// Facilitator guides (facilitate.html, #107; no WebGPU): the deep link renders its guide with the Say callouts
// and the rubric, the session list switches guides, and the learning path links to the page.
import { test, expect } from './fixtures.mjs';

const guide = (page) => page.locator('#facilitate-guide article.facilitate-guide');

test('facilitate.html#nbody-beginner shows the Beginner N-body guide', async ({ page }) => {
    await page.goto('facilitate.html#nbody-beginner');
    await expect(guide(page)).toHaveAttribute('data-session', 'nbody-beginner');
    await expect(guide(page).locator('h2').first()).toContainText('Beginner');
    await expect(guide(page).locator('h2', { hasText: 'Observation rubric' })).toBeVisible();
    expect(await guide(page).locator('aside.callout-say').count()).toBeGreaterThanOrEqual(3);
    expect(await guide(page).locator('table').count()).toBe(4);
    await expect(guide(page).locator('a[href="index.html?level=beginner#nbody"]').first()).toBeVisible();
    await expect(page.locator('#session-list a[href="#nbody-beginner"]')).toHaveAttribute('aria-current', 'page');

    await page.locator('#session-list a[href="#matrix-multiplication-advanced"]').click();
    await expect(guide(page)).toHaveAttribute('data-session', 'matrix-multiplication-advanced');
    await expect(page.locator('#session-list a[href="#nbody-beginner"]')).not.toHaveAttribute('aria-current', 'page');
    await page.goBack();
    await expect(guide(page)).toHaveAttribute('data-session', 'nbody-beginner');
});

test('the learning path links to the guides', async ({ page }) => {
    await page.goto('learn.html');
    await page.locator('.hero-links a[href="facilitate.html"]').click();
    await expect(page).toHaveURL(/facilitate\.html$/);
    await expect(guide(page)).toHaveAttribute('data-session', 'nbody-beginner');
});
