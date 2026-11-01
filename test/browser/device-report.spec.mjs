// 8. The opt-in device report: the button builds a report with the adapter section, shown in the preview
// (the clipboard may be unavailable in headless Chromium; then the preview is opened and selected).
import { test, expect, openSite } from './fixtures.mjs';
import { SOFTWARE } from './adapter.mjs';

test('Copy device report builds a report with the adapter section', async ({ page }) => {
    await openSite(page);
    await page.locator('#btn-device-report').click();
    const feedback = page.locator('#device-report-feedback');
    await expect(feedback).toHaveText(/Copied|Clipboard unavailable/);
    const preview = page.locator('#device-report-preview');
    await expect(preview).toContainText('## WebGPU device report');
    const text = await preview.textContent();
    expect(text).toContain('### Adapter');
    expect(text).toContain('#### Limits');
    console.log(`[test/browser] device report feedback: ${await feedback.textContent()}`);
    const adapter = text.slice(text.indexOf('### Adapter'), text.indexOf('#### Adapter features'));
    console.log(`[test/browser] report adapter section:\n${adapter.trim()}`);
    expect(SOFTWARE.test(adapter) || /isFallbackAdapter\s*\|\s*true/.test(adapter), adapter).toBe(true);
});

test('opening the preview builds the report', async ({ page }) => {
    await openSite(page);
    await page.locator('#device-report-details summary').click();
    await expect(page.locator('#device-report-preview')).toContainText('### Adapter');
});
