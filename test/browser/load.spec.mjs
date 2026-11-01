// 1. Page load: WebGPU ready on a software adapter, no [WebGPU] console errors while loading.
import { test, expect, openSite } from './fixtures.mjs';
import { probeAdapterInPage, isSoftwareAdapter, describeAdapter } from './adapter.mjs';

test('index.html loads and reports WebGPU ready on a software adapter', async ({ page }) => {
    const status = await openSite(page);
    console.log(`[test/browser] status badge: ${status}`);
    expect(status).toMatch(/^WebGPU ready/);
    expect(status).toMatch(/GPU timestamps on|GPU time not available/);

    const info = await page.evaluate(probeAdapterInPage);
    expect(isSoftwareAdapter(info), describeAdapter(info)).toBe(true);

    // Every example section got its tabs, and the examples near the top were created without errors
    await expect(page.locator('section.example[id] > .tabs[role="tablist"]')).toHaveCount(25);
    await page.waitForTimeout(1500);   // let the IntersectionObserver-driven inits finish; the fixture checks errors
});
