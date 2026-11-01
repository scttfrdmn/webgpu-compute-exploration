// Picks the Chromium flags for the browser suite (#77) — CI ONLY. Tries a few known ways of getting a
// WebGPU adapter in headless Chromium on a GPU-less Linux runner, prints the adapter each one yields, and
// accepts the first whose default adapter is a software device. Writes BROWSER_FLAGS / BROWSER_CHANNEL
// to $GITHUB_ENV for the test step; exits 1 if no candidate gives a software adapter.
import './guard.mjs';   // first: refuses unless CI=true on Linux, before Playwright is loaded

import { appendFileSync } from 'node:fs';
import { chromium } from '@playwright/test';
import { startServer } from './server.mjs';
import { probeAdapterInPage, isSoftwareAdapter, describeAdapter } from './adapter.mjs';

const COMMON = ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'];
const CANDIDATES = [
    // Dawn on Vulkan; the job forces Mesa lavapipe's ICD, so the only Vulkan device is llvmpipe
    { name: 'vulkan (lavapipe ICD)', args: [...COMMON, '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-vulkan-surface', '--ignore-gpu-blocklist'] },
    // Chromium's bundled SwiftShader Vulkan as the WebGPU adapter
    { name: 'swiftshader adapter', args: [...COMMON, '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] },
    { name: 'swiftshader vulkan', args: [...COMMON, '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
];
const CHANNELS = [undefined, 'chromium'];   // headless shell, then the full Chromium's new headless mode

const withTimeout = (promise, ms, what) => Promise.race([promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms))]);

const server = await startServer(0);
const url = `http://127.0.0.1:${server.address().port}/test/browser/blank.html`;
let chosen = null;

outer:
for (const channel of CHANNELS) {
    for (const candidate of CANDIDATES) {
        const label = `${candidate.name} / ${channel ?? 'headless shell'}`;
        let browser;
        try {
            browser = await chromium.launch({ channel, args: candidate.args, timeout: 30000 });
            const page = await browser.newPage();
            await page.goto(url);
            const info = await withTimeout(page.evaluate(probeAdapterInPage), 20000, 'adapter probe');
            const software = isSoftwareAdapter(info);
            console.log(`[probe] ${label}: ${browser.version()} · ${describeAdapter(info)} · software=${software}`);
            if (software) { chosen = { ...candidate, channel, info }; break outer; }
        } catch (error) {
            console.log(`[probe] ${label}: failed: ${error.message.split('\n')[0]}`);
        } finally {
            await browser?.close().catch(() => {});
        }
    }
}
server.close();

if (!chosen) {
    console.error('[probe] no flag set gave a software WebGPU adapter; refusing to run the browser suite');
    process.exit(1);
}
console.log(`[probe] using ${chosen.name} on ${chosen.channel ?? 'headless shell'}: ${chosen.args.join(' ')}`);
if (process.env.GITHUB_ENV) {
    appendFileSync(process.env.GITHUB_ENV,
        `BROWSER_FLAGS=${JSON.stringify(chosen.args)}\nBROWSER_CHANNEL=${chosen.channel ?? ''}\n`);
}
