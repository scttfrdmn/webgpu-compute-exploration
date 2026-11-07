// 10. Visitor levels (#89–#95): the Level menu follows ?level= and remembers it, Understand shows the level's
// page or the fallback note, Explore hides controls by level and shows the look-for prompts, Measure lists the
// level's experiments. (The learning path per level is in learn.spec.mjs.)
import { test, expect, openSite, panel, expectSelected } from './fixtures.mjs';

const levelMenu = (page) => page.locator('#level-select');
const visibleExperiments = (page, id) => panel(page, id, 'measure').locator('.experiment:visible');

test('?level= selects the level, it is remembered, and choosing Advanced drops the parameter', async ({ page }) => {
    await openSite(page, 'index.html?level=beginner');
    await expect(levelMenu(page)).toHaveValue('beginner');
    // remembered on a page without the parameter
    await openSite(page, 'index.html');
    await expect(levelMenu(page)).toHaveValue('beginner');
    await levelMenu(page).selectOption('intermediate');
    await expect(page).toHaveURL(/\?level=intermediate/);
    await levelMenu(page).selectOption('advanced');
    await expect(page).not.toHaveURL(/level=/);
    expect(await page.evaluate(() => localStorage.getItem('webgpu-level'))).toBe('advanced');
});

test('Understand shows the page for the level and re-renders when the level changes', async ({ page }) => {
    await openSite(page, 'index.html?level=beginner#nbody/understand');
    await expectSelected(page, 'nbody', 'understand');
    const article = panel(page, 'nbody', 'understand').locator('article.understand');
    await expect(article).toHaveAttribute('data-level', 'beginner');
    await expect(article.getByRole('heading', { name: 'What you see' })).toBeVisible();
    await levelMenu(page).selectOption('intermediate');
    await expect(article).toHaveAttribute('data-level', 'intermediate');
    await levelMenu(page).selectOption('advanced');
    await expect(article).toHaveAttribute('data-level', 'advanced');
    await expect(panel(page, 'nbody', 'understand').locator('.level-fallback-note')).toHaveCount(0);
});

test('an example without a page at the level falls back to the next level up, with a note', async ({ page }) => {
    await openSite(page, 'index.html?level=beginner#cloth/understand');
    const understand = panel(page, 'cloth', 'understand');
    await expect(understand.locator('article.understand')).toHaveAttribute('data-level', 'advanced');
    await expect(understand.locator('.level-fallback-note')).toContainText('Beginner');
});

test('Explore hides controls below their level and shows the look-for prompts below Advanced', async ({ page }) => {
    await openSite(page, 'index.html?level=beginner#nbody');
    const nbody = panel(page, 'nbody', 'explore');
    await expect(page.locator('#btn-nbody-start')).toBeVisible();
    await expect(page.locator('#nbody-theta')).toBeVisible();            // Beginner: named in its Understand page
    await expect(page.locator('#nbody-softening')).toBeHidden();         // Intermediate and up
    await expect(page.locator('#nbody-trails')).toBeHidden();            // Advanced only
    await expect(nbody.locator('.look-for:visible')).toHaveCount(1);
    await levelMenu(page).selectOption('intermediate');
    await expect(page.locator('#nbody-softening')).toBeVisible();
    await expect(page.locator('#nbody-trails')).toBeHidden();
    await levelMenu(page).selectOption('advanced');
    await expect(page.locator('#nbody-trails')).toBeVisible();
    await expect(nbody.locator('.look-for:visible')).toHaveCount(0);
});

test('Measure lists the experiments for the level', async ({ page }) => {
    await openSite(page, 'index.html?level=beginner#nbody/measure');
    await expectSelected(page, 'nbody', 'measure');
    await expect(visibleExperiments(page, 'nbody')).toHaveCount(1);
    await levelMenu(page).selectOption('advanced');
    await expect(visibleExperiments(page, 'nbody')).toHaveCount(3);
    // an example with no Beginner experiment points to the next level up
    await openSite(page, 'index.html?level=beginner#cloth/measure');
    await expect(visibleExperiments(page, 'cloth')).toHaveCount(0);
    await expect(panel(page, 'cloth', 'measure').locator('.measure-level-note')).toBeVisible();
});

test('first visit (#108): the catalog offers Learn or the technical view once; Learn opens the path at that level', async ({ page }) => {
    await openSite(page, 'index.html');
    const choice = page.locator('#first-visit .first-visit');
    await expect(choice).toBeVisible();
    await choice.locator('[data-choice="beginner"]').click();
    await expect(page).toHaveURL(/learn\.html\?level=beginner/);
    await expect(levelMenu(page)).toHaveValue('beginner');
    await openSite(page, 'index.html');
    await expect(page.locator('#first-visit .first-visit')).toHaveCount(0);
});

test('first visit: closing it keeps the technical view and it is not offered again; deep links never show it', async ({ page }) => {
    await openSite(page, 'index.html#nbody/measure');
    await expect(page.locator('#first-visit .first-visit')).toHaveCount(0);
    await openSite(page, 'index.html');
    await page.locator('#first-visit .first-visit-close').click();
    await expect(page.locator('#first-visit')).toBeHidden();
    await expect(levelMenu(page)).toHaveValue('advanced');
    await page.reload();
    await expect(page.locator('#first-visit .first-visit')).toHaveCount(0);
});
