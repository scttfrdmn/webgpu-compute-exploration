// SAFETY GUARD for the real-browser suite (test/browser/, #77) — CI ONLY, software GPU ONLY.
//
// These tests drive the site in headless Chromium with WebGPU enabled. On a developer machine that would
// run the examples on the real GPU, and a headless-Chrome WebGPU run has hung a maintainer's Mac before
// (#32). So, like test/gpu/harness.mjs, everything here refuses to run unless
//   1. CI=true (GitHub Actions sets it) and the platform is Linux — checked when this module is first
//      imported, before Playwright or any browser is loaded (every entry point imports it first), and
//   2. the Chromium flags come from test/browser/probe.mjs (BROWSER_FLAGS), which only accepts a flag set
//      whose default adapter is a software device (llvmpipe / lavapipe / SwiftShader, or a fallback
//      adapter); global-setup.mjs re-checks the adapter with the final flags before any test starts.
// Never run test/browser/ locally (no `npx playwright test`, no `npx playwright install`): push a branch
// and read the CI log instead. See CLAUDE.md.

export function refuse(why) {
    process.stderr.write(`\n[test/browser] REFUSING TO RUN: ${why}\n`
        + '[test/browser] Browser tests run only in GitHub Actions on Linux with a software WebGPU adapter.\n'
        + '[test/browser] They must never run on a developer machine. Use `npm test` for the CPU suite.\n');
    process.exit(2);
}

if (process.env.CI !== 'true') refuse(`CI is not "true" (CI=${process.env.CI ?? '<unset>'})`);
if (process.platform !== 'linux') refuse(`platform is ${process.platform}, not linux`);

// The Chromium launch options chosen by probe.mjs (written to $GITHUB_ENV by the CI job)
export function launchSettings() {
    const raw = process.env.BROWSER_FLAGS;
    if (!raw) refuse('BROWSER_FLAGS is not set: run test/browser/probe.mjs first (it picks flags for a software adapter)');
    let args;
    try { args = JSON.parse(raw); } catch { refuse(`BROWSER_FLAGS is not JSON: ${raw}`); }
    if (!Array.isArray(args) || !args.includes('--enable-unsafe-webgpu')) refuse(`unexpected BROWSER_FLAGS: ${raw}`);
    return { args, channel: process.env.BROWSER_CHANNEL || undefined };
}
