// 9. Reference investigations (#79): the landing-page section renders three cards from js/teach/flagships.js,
// each "Run it" link opens that example's Measure tab, and learn.html points at the same three.
import { test, expect, openSite, expectSelected, panel } from './fixtures.mjs';

const IDS = ['matrix-multiplication', 'nbody', 'parallel-sort'];

test('the landing page shows three reference investigation cards, above the catalog', async ({ page }) => {
    await openSite(page);
    const section = page.locator('#reference-investigations');
    await expect(section.getByRole('heading', { level: 2, name: 'Reference investigations' })).toBeVisible();
    const cards = section.locator('article.flagship');
    await expect(cards).toHaveCount(3);
    expect(await cards.evaluateAll(as => as.map(a => a.id))).toEqual(IDS.map(id => `flagship-${id}`));
    for (const id of IDS) {
        const card = page.locator(`#flagship-${id}`);
        await expect(card.locator('h3')).toBeVisible();
        expect(await card.locator('.flagship-predictions li').count()).toBeGreaterThan(0);
        const hrefs = await card.locator('.flagship-links a').evaluateAll(as => as.map(a => a.getAttribute('href')));
        expect(hrefs).toEqual([`#${id}/measure`, `#${id}/understand`, `#${id}`]);
    }
    // The section comes before the first example section
    const before = await page.evaluate(() => {
        const s = document.getElementById('reference-investigations');
        const first = document.querySelector('section.example[id]');
        return Boolean(s.compareDocumentPosition(first) & Node.DOCUMENT_POSITION_FOLLOWING);
    });
    expect(before).toBe(true);
});

for (const id of IDS) {
    test(`"Run it" on the ${id} card opens its Measure tab`, async ({ page }) => {
        await openSite(page);
        await page.locator(`#flagship-${id} a.flagship-run`).click();
        await expect(page).toHaveURL(new RegExp(`#${id}/measure$`));
        await expectSelected(page, id, 'measure');
        const runs = panel(page, id, 'measure').locator('.experiment button.primary');
        await expect(runs.first()).toBeEnabled();
    });
}

test('learn.html points at the three investigations', async ({ page }) => {
    await page.goto('learn.html');
    const pointer = page.locator('#flagship-pointer');
    await expect(page.getByRole('heading', { level: 2, name: 'Start with a reference investigation' })).toBeVisible();
    await expect(pointer.locator('li.flagship-pointer-item')).toHaveCount(3);
    const runs = await pointer.locator('a.flagship-run').evaluateAll(as => as.map(a => a.getAttribute('href')));
    expect(runs).toEqual(IDS.map(id => `index.html#${id}/measure`));
});
