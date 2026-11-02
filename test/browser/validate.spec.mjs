// 10. Physical-GPU validation page (#78), here on the software adapter with the reduced plan (?smoke=1):
//     the page shows the plan and the detected adapter, runs every entry one after another, shows the
//     outcomes, and offers a bundle that validates against both schemas. Cancel stops the remaining entries.
//     The full plan is for the maintainer's own browsers; CI never runs it.
import { test, expect, hideTimestampQuery } from './fixtures.mjs';
import { readFile } from 'node:fs/promises';
import { SMOKE_PLAN, validateValidationBundle } from '../../js/teach/validation.js';

const ROOT = new URL('../../', import.meta.url);
const readSchema = async (path) => JSON.parse(await readFile(new URL(path, ROOT), 'utf8'));

async function openValidate(page, query = '?smoke=1') {
    await page.goto(`validate.html${query}`);
    await expect(page.locator('#gpu-status')).toHaveClass('status supported', { timeout: 30_000 });
}

test('validate.html?smoke=1 shows the plan and adapter, runs every entry and offers a valid bundle', async ({ page }) => {
    test.setTimeout(180_000);
    await openValidate(page);
    await expect(page.locator('#validation-plan tbody tr')).toHaveCount(SMOKE_PLAN.entries.length);
    await expect(page.locator('#validation-plan-summary')).toContainText('SMOKE PLAN');
    const detected = page.locator('#validation-detected');
    await expect(detected).toContainText('Adapter:');
    await expect(detected).toContainText('timestamp-query:');
    console.log(`[test/browser] validate.html detected:\n${await detected.innerText()}`);
    await expect(page.locator('#validation-run')).toBeEnabled();

    await page.locator('#validation-run').click();
    await expect(page.locator('#validation-cancel')).toBeEnabled();
    const status = page.locator('#validation-status');
    await expect(status).toHaveText(new RegExp(`^Done: ${SMOKE_PLAN.entries.length} of ${SMOKE_PLAN.entries.length} experiments ran`), { timeout: 150_000 });
    console.log(`[test/browser] validate.html: ${await status.textContent()}`);
    const rows = page.locator('#validation-results tbody tr');
    await expect(rows).toHaveCount(SMOKE_PLAN.entries.length);
    for (const row of await rows.all()) {
        const text = await row.innerText();
        console.log(`[test/browser]   ${text.replace(/\s+/g, ' ')}`);
        expect(text).toMatch(/done|budget/);
        expect(text).toMatch(/PASS|FAIL|NOT CHECKED/);
        expect(text).toMatch(/wall-clock|timestamp-query/);
    }
    await expect(page.locator('#validation-download-hint')).toContainText('data/validation/');

    const download = page.locator('#validation-download');
    await expect(download).toBeEnabled();
    const [file] = await Promise.all([page.waitForEvent('download'), download.click()]);
    expect(file.suggestedFilename()).toMatch(/^\d{4}-\d{2}-\d{2}-[0-9a-z]+-smoke\.json$/);
    const bundle = JSON.parse(await readFile(await file.path(), 'utf8'));
    expect(bundle.bundleVersion).toBe(1);
    expect(bundle.plan).toBe(SMOKE_PLAN.version);
    expect(bundle.smoke).toBe(true);
    expect(bundle.records).toHaveLength(SMOKE_PLAN.entries.length);
    expect(bundle.records.map(r => `${r.experiment.exampleId}/${r.experiment.id}`))
        .toEqual(SMOKE_PLAN.entries.map(e => `${e.exampleId}/${e.experimentId}`));
    const result = validateValidationBundle(bundle, await readSchema('schema/validation-bundle.v1.json'), await readSchema('schema/experiment-result.v1.json'));
    expect(result.errors).toEqual([]);
    console.log(`[test/browser] bundle environment: ${JSON.stringify({ ...bundle.environment, userAgent: undefined })}`);
});

test('Cancel stops the validation run; the remaining entries are not run', async ({ page }) => {
    test.setTimeout(120_000);
    await openValidate(page);
    await page.locator('#validation-run').click();
    const status = page.locator('#validation-status');
    await expect(status).toHaveText(/^Running/);
    await page.locator('#validation-cancel').click();
    await expect(status).toHaveText(new RegExp(`^Cancelled: [0-${SMOKE_PLAN.entries.length - 1}] of ${SMOKE_PLAN.entries.length} experiments ran`), { timeout: 60_000 });
    await expect(page.locator('#validation-results tbody tr').last()).toContainText('not run');
    await expect(page.locator('#validation-run')).toBeEnabled();
    await expect(page.locator('#validation-cancel')).toBeDisabled();
});

test.describe('without timestamp-query', () => {
    test.beforeEach(async ({ page }) => { await page.addInitScript(hideTimestampQuery); });

    test('the page says GPU times fall back to wall-clock before running', async ({ page }) => {
        await openValidate(page, '');
        await expect(page.locator('#gpu-status')).toContainText('no timestamp-query');
        await expect(page.locator('#validation-detected')).toContainText('not available');
        await expect(page.locator('#validation-plan tbody tr')).toHaveCount(7);
        await expect(page.locator('#validation-plan-summary')).not.toContainText('SMOKE');
    });
});
