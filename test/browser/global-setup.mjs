// Re-checks, with the exact launch options the tests use, that the default WebGPU adapter is a software
// device before any test runs (#77). Aborts the whole run otherwise.
import { refuse, launchSettings } from './guard.mjs';
import { chromium } from '@playwright/test';
import { startServer } from './server.mjs';
import { probeAdapterInPage, isSoftwareAdapter, describeAdapter } from './adapter.mjs';

export default async function globalSetup() {
    const { args, channel } = launchSettings();
    const server = await startServer(0);
    const browser = await chromium.launch({ channel, args, timeout: 30000 });
    try {
        const page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${server.address().port}/test/browser/blank.html`);
        const info = await page.evaluate(probeAdapterInPage);
        console.log(`[test/browser] Chromium ${browser.version()} (${channel ?? 'headless shell'}) · ${args.join(' ')}`);
        console.log(`[test/browser] adapter: ${describeAdapter(info)}`);
        if (!isSoftwareAdapter(info)) refuse(`the WebGPU adapter is not a software device (${describeAdapter(info)})`);
    } finally {
        await browser.close();
        server.close();
    }
}
