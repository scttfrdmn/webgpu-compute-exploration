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

// Counts live GPU objects per device (#106): wraps GPUDevice.prototype.createBuffer / createTexture /
// createQuerySet and the objects' destroy(), so live = created − destroyed. Test-only: injected with
// page.addInitScript, never shipped with the site. An object dropped without destroy() stays live here even
// after it is garbage-collected — on purpose: destroy() is what frees GPU memory promptly, GC may take
// arbitrarily long. Each live object keeps its kind, label, size and the site code that created it, so a
// failing test can say what leaked. Read with page.evaluate(() => window.__gpuResources.snapshot()).
export function trackGpuResources() {
    if (typeof GPUDevice === 'undefined' || window.__gpuResources) return;
    const KINDS = { createBuffer: 'buffer', createTexture: 'texture', createQuerySet: 'querySet' };
    const devices = [];                 // [{ created: { kind: n }, destroyed: { kind: n } }]
    const deviceIndex = new WeakMap();  // GPUDevice -> index in devices
    const info = new WeakMap();         // GPU object -> { device, kind, seq, label, size, site }
    const live = new Map();             // seq -> info, for objects not destroyed yet
    let seq = 0;
    const zero = () => ({ buffer: 0, texture: 0, querySet: 0 });
    const counts = (device) => {
        if (!deviceIndex.has(device)) {
            deviceIndex.set(device, devices.length);
            devices.push({ created: zero(), destroyed: zero() });
        }
        return devices[deviceIndex.get(device)];
    };
    // The innermost stack frame in the site's js/: the code that created the object
    const site = () => ((new Error().stack || '').split('\n').find(l => /\/js\//.test(l)) ?? '?')
        .trim().replace(/^at /, '').replace(/https?:\/\/[^/]+\//, '');
    for (const [method, kind] of Object.entries(KINDS)) {
        const create = GPUDevice.prototype[method];
        GPUDevice.prototype[method] = function (descriptor) {
            const object = create.call(this, descriptor);
            counts(this).created[kind]++;
            const size = kind === 'buffer' ? descriptor?.size : kind === 'querySet' ? descriptor?.count
                : [descriptor?.size].flat().map(v => (typeof v === 'object' ? Object.values(v).join('×') : v)).join('×');
            const entry = { device: deviceIndex.get(this), kind, seq: ++seq, label: descriptor?.label ?? '', size, site: site() };
            info.set(object, entry);
            live.set(entry.seq, entry);
            return object;
        };
    }
    for (const Class of [GPUBuffer, GPUTexture, GPUQuerySet]) {
        const destroy = Class.prototype.destroy;
        Class.prototype.destroy = function () {
            const entry = info.get(this);
            if (entry && live.delete(entry.seq)) devices[entry.device].destroyed[entry.kind]++;   // a second destroy() does not count
            return destroy.call(this);
        };
    }
    window.__gpuResources = {
        // Per device: created, destroyed and live (created − destroyed) by kind; seq = objects created so far
        snapshot: () => ({
            seq,
            devices: devices.map(({ created, destroyed }) => ({
                created: { ...created },
                destroyed: { ...destroyed },
                live: Object.fromEntries(Object.keys(created).map(k => [k, created[k] - destroyed[k]])),
            })),
        }),
        // Objects created after `since` (a snapshot's seq) and not destroyed yet
        liveSince: (since = 0) => [...live.values()].filter(e => e.seq > since),
    };
}

// The live-object snapshot of trackGpuResources, summed over devices: { seq, buffer, texture, querySet, devices }
export async function gpuResources(page) {
    const s = await page.evaluate(() => window.__gpuResources?.snapshot() ?? null);
    if (!s) throw new Error('trackGpuResources was not injected (page.addInitScript(trackGpuResources) before goto)');
    const total = { buffer: 0, texture: 0, querySet: 0 };
    for (const d of s.devices) for (const k of Object.keys(total)) total[k] += d.live[k];
    return { seq: s.seq, ...total, devices: s.devices.length };
}
