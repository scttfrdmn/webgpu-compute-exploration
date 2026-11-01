// Playwright config for the real-browser regression suite (#77) — CI ONLY, software WebGPU ONLY.
// Run by the `browser` job in .github/workflows/ci.yml after probe.mjs has chosen the Chromium flags:
//   npx playwright test -c test/browser/playwright.config.mjs
// Never run locally (see guard.mjs and CLAUDE.md). @playwright/test is installed in CI with --no-save.
import { launchSettings } from './guard.mjs';   // first: refuses unless CI=true on Linux
import { defineConfig } from '@playwright/test';

const { args, channel } = launchSettings();
const PORT = 8123;

export default defineConfig({
    testDir: '.',
    testMatch: /.*\.spec\.mjs$/,
    globalSetup: './global-setup.mjs',
    workers: 1,                 // one software GPU (the runner's CPU cores): no parallel pages
    fullyParallel: false,
    retries: 1,                 // a retried pass is reported as "flaky" in the log
    timeout: 90_000,
    expect: { timeout: 20_000 },
    reporter: [['list']],
    forbidOnly: true,
    use: {
        baseURL: `http://127.0.0.1:${PORT}/`,
        browserName: 'chromium',
        channel,
        launchOptions: { args },
        viewport: { width: 1280, height: 800 },
    },
    webServer: {
        command: `node server.mjs ${PORT}`,
        url: `http://127.0.0.1:${PORT}/test/browser/blank.html`,
        reuseExistingServer: false,
        timeout: 20_000,
    },
});
