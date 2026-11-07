// Shared fixtures and helpers for test/browser/*.spec.mjs (#77). Importing this module runs the CI guard.
import './guard.mjs';   // first: refuses unless CI=true on Linux, before Playwright is loaded
import { test as base, expect } from '@playwright/test';

// Console output the site treats as a WebGPU problem, or an example failing to initialize
const BAD_CONSOLE = /^\[WebGPU\]|^Failed to initialize|^\[Measure\]/;

export const test = base.extend({
    // Every test fails on uncaught page exceptions and on [WebGPU] / init / Measure console errors
    pageErrors: [async ({ page }, use) => {
        const errors = [];
        page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
        page.on('console', (message) => {
            if (message.type() === 'error' && BAD_CONSOLE.test(message.text())) errors.push(message.text());
        });
        await use(errors);
        expect(errors, 'page errors and [WebGPU] console errors').toEqual([]);
    }, { auto: true }],
});
export { expect };

// Opens a page of the site and waits until main.js reports a WebGPU device
export async function openSite(page, path = 'index.html') {
    await page.goto(path);
    await expect(page.locator('#gpu-status')).toHaveClass('status supported', { timeout: 30_000 });
    return page.locator('#gpu-status').textContent();
}

export const tab = (page, id, name) => page.locator(`#${id}-tab-${name}`);
export const panel = (page, id, name) => page.locator(`#${id}-panel-${name}`);

// Asserts that `name` is the selected tab of example `id` and only its panel is shown
export async function expectSelected(page, id, name) {
    for (const t of ['explore', 'understand', 'measure']) {
        await expect(tab(page, id, t)).toHaveAttribute('aria-selected', String(t === name));
        await expect(tab(page, id, t)).toHaveAttribute('tabindex', t === name ? '0' : '-1');
        if (t === name) await expect(panel(page, id, t)).toBeVisible();
        else await expect(panel(page, id, t)).toBeHidden();
    }
}

// The card of one Measure experiment, found by its title
export function experimentCard(page, id, title) {
    return panel(page, id, 'measure').locator('.experiment').filter({ has: page.locator('h4', { hasText: title }) });
}

// Hides timestamp-query from the page (adapter and device), as on an adapter without the feature
export function hideTimestampQuery() {
    if (typeof GPUSupportedFeatures === 'undefined') return;
    const has = GPUSupportedFeatures.prototype.has;
    GPUSupportedFeatures.prototype.has = function (feature) {
        return feature === 'timestamp-query' ? false : has.call(this, feature);
    };
}

// Gives every device a `lost` promise the test can resolve: window.__loseTestDevice(message) runs the
// site's real device-lost path (reason 'unknown'; 'destroyed' is ignored by design) without a real loss.
export function controllableDeviceLoss() {
    if (typeof GPUAdapter === 'undefined') return;
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...args) {
        const device = await requestDevice.apply(this, args);
        let resolveLost;
        const lost = new Promise((resolve) => { resolveLost = resolve; });
        Object.defineProperty(device, 'lost', { configurable: true, get: () => lost });
        window.__loseTestDevice = (message) => resolveLost({ reason: 'unknown', message });
        return device;
    };
}

// A page clock the test can jump forward (#105): performance.now() gets an offset, and
// window.__stopAfterSteps(statusElement, k) adds an hour to it as soon as that Measure status reports k finished
// steps. The run's next step boundary then sees its time budget spent and stops with reason 'budget', after
// exactly k steps, however fast or slow the GPU is.
export function controllableClock() {
    const now = performance.now.bind(performance);
    let offset = 0;
    performance.now = () => now() + offset;
    window.__stopAfterSteps = (status, steps) => {
        const observer = new MutationObserver(() => {
            const m = /^Running… (\d+) step/.exec(status.textContent);
            if (m && Number(m[1]) >= steps) { offset += 3600_000; observer.disconnect(); }
        });
        observer.observe(status, { childList: true, characterData: true, subtree: true });
    };
}

// A wrong CPU reference: Uint32Array sort() swaps the first and last keys of its result, so every GPU sort that
// is compared with it (parallel-sort's verify()) fails its check
export function corruptUint32Sort() {
    const sort = Uint32Array.prototype.sort;
    Uint32Array.prototype.sort = function (...args) {
        const out = sort.apply(this, args);
        const last = this.length - 1;
        if (last > 0 && this[0] !== this[last]) [this[0], this[last]] = [this[last], this[0]];
        return out;
    };
}
