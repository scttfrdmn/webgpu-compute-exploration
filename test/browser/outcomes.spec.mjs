// 11. Completion, verification and interpretation kept apart (#105), at Beginner: a run stopped by its time budget
// says so first, with how many of its steps finished and what they cover, before the check summary; a failed check
// is counted and listed at Beginner (not behind the "Show the numbers" disclosure) whatever the claims say.
// The budget stop is forced with a page clock that jumps an hour once the first step has finished; the failure
// with a CPU reference sort that swaps two keys. parallel-sort's "scaling" (Largest N 262,144) plans 5 steps.
import { test, expect, openSite, panel, experimentCard, controllableClock, corruptUint32Sort } from './fixtures.mjs';

const TITLE = 'Sorting N keys: bitonic, bitonic with workgroup memory, radix, CPU';

async function runStoppedAfterFirstStep(page) {
    await openSite(page, 'index.html?level=beginner#parallel-sort/measure');
    const card = experimentCard(page, 'parallel-sort', TITLE);
    await expect(card).toBeVisible();
    await page.locator('#parallel-sort-scaling-maxN').selectOption('262144');
    await card.locator('.experiment-status').evaluate((status) => window.__stopAfterSteps(status, 1));
    await card.locator('button.primary').click();
    await expect(card.locator('.experiment-status')).toHaveText(
        'Stopped early by the 20 s time limit after 1 of 5 steps: these results cover only N = 16,384.', { timeout: 80_000 });
    await expect(card.locator('.experiment-checks')).toBeVisible();
    return card;
}

// The status (completion) comes before the check summary, which comes before the claims, in the card
async function expectOrder(card) {
    const order = await card.evaluate((el) => {
        const at = (sel) => el.querySelector(sel);
        const before = (a, b) => !a || !b || Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
        const status = at('.experiment-status'), checks = at('.check-summary'), claims = at('.experiment-interpretation:not([hidden])');
        return { checks: Boolean(checks), statusFirst: before(status, checks), checksBeforeClaims: before(checks, claims) };
    });
    expect(order).toEqual({ checks: true, statusFirst: true, checksBeforeClaims: true });
}

test.describe('Beginner run summary', () => {
    test.beforeEach(async ({ page }) => { await page.addInitScript(controllableClock); });

    test('a run stopped by the time budget says so first, then counts passed and not-checked checks', async ({ page }) => {
        const card = await runStoppedAfterFirstStep(page);
        await expect(card.locator('.check-summary')).toHaveText(
            '3 of 4 checks passed, 1 could not be checked.');   // the f32 key map is not used with u32 keys
        await expect(card.locator('.experiment-checks .check-status')).toHaveCount(0);   // no failure to list
        await expectOrder(card);
        await expect(card.locator('.experiment-table details.experiment-numbers')).toHaveCount(1);
        // Advanced keeps its own wording for the same run
        await page.locator('#level-select').selectOption('advanced');
        await expect(card.locator('.experiment-status')).toHaveText(
            'Stopped at the first step boundary after the 20 s time budget (1 step completed).');
        await expect(card.locator('.check-summary')).toHaveCount(0);
        await expect(card.locator('.experiment-checks .check-status')).toHaveText(['PASS', 'PASS', 'PASS', 'NOT CHECKED']);
    });

    test.describe('with a wrong CPU reference', () => {
        test.beforeEach(async ({ page }) => { await page.addInitScript(corruptUint32Sort); });

        test('a failed check is counted and listed at Beginner, outside the disclosure', async ({ page }) => {
            const card = await runStoppedAfterFirstStep(page);
            await expect(card.locator('.check-summary')).toHaveText(
                '3 of 4 checks failed, 1 could not be checked: some of the GPU\'s answers do not match a careful CPU calculation.');
            await expect(card.locator('.check-summary')).toHaveClass(/check-summary-fail/);
            const failed = card.locator('.experiment-checks .check-status');
            await expect(failed).toHaveText(['FAIL', 'FAIL', 'FAIL']);
            for (const f of await failed.all()) await expect(f).toBeVisible();
            expect(await card.locator('details .experiment-checks').count()).toBe(0);
            await expectOrder(card);
            // Claims, if any are shown at Beginner for one step, carry the warning and no success marker
            const interpretation = card.locator('.experiment-interpretation');
            if (await interpretation.isVisible()) {
                await expect(interpretation.locator('.claims-note')).toHaveText(/^A check failed/);
            }
        });
    });
});
