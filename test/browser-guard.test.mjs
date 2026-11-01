// The real-browser suite (test/browser/, #77) is CI-only: its guard must refuse outside CI or off Linux
// before Playwright is loaded, and only software adapters may pass the adapter check. CPU only: this
// spawns `node` on the guard module (which exits immediately) and never loads Playwright or a browser.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { isSoftwareAdapter, describeAdapter } from './browser/adapter.mjs';

const dir = join(dirname(fileURLToPath(import.meta.url)), 'browser');

function runGuard(env) {
    const clean = { ...process.env };
    delete clean.CI;
    return spawnSync(process.execPath, [join(dir, 'guard.mjs')], { env: { ...clean, ...env }, encoding: 'utf8', timeout: 10000 });
}

test('guard refuses when CI is not "true"', () => {
    for (const env of [{}, { CI: '1' }, { CI: 'false' }]) {
        const r = runGuard(env);
        assert.equal(r.status, 2, `CI=${env.CI}: exit ${r.status}`);
        assert.match(r.stderr, /REFUSING TO RUN: CI is not "true"/);
    }
});

test('guard refuses off Linux even with CI=true', { skip: process.platform === 'linux' }, () => {
    const r = runGuard({ CI: 'true' });
    assert.equal(r.status, 2);
    assert.match(r.stderr, /REFUSING TO RUN: platform is/);
});

test('every entry point imports the guard before Playwright', () => {
    const files = readdirSync(dir).filter(f => /\.(spec\.)?mjs$/.test(f) && !['guard.mjs', 'adapter.mjs', 'server.mjs'].includes(f));
    assert.ok(files.length >= 8, files.join(', '));
    for (const file of files) {
        const imports = [...readFileSync(join(dir, file), 'utf8').matchAll(/^import\s+(?:[^'"]*from\s+)?['"]([^'"]+)['"]/gm)].map(m => m[1]);
        const guard = imports.findIndex(s => s === './guard.mjs' || s === './fixtures.mjs');
        const playwright = imports.findIndex(s => s.startsWith('@playwright/'));
        assert.ok(guard === 0, `${file}: first import must be ./guard.mjs or ./fixtures.mjs (${imports.join(', ')})`);
        assert.ok(playwright === -1 || playwright > guard, `${file}: Playwright imported before the guard`);
    }
    const fixtures = readFileSync(join(dir, 'fixtures.mjs'), 'utf8');
    assert.match(fixtures, /^import '\.\/guard\.mjs';/m);
});

test('only software adapters pass the adapter check', () => {
    const base = { ok: true, vendor: '', architecture: '', device: '', description: '', isFallbackAdapter: false, features: [] };
    const software = [
        { ...base, vendor: 'mesa', description: 'llvmpipe (LLVM 17.0.6, 256 bits)' },
        { ...base, vendor: 'google', architecture: 'swiftshader', description: 'SwiftShader Device (Subzero)' },
        { ...base, isFallbackAdapter: true },
    ];
    const hardware = [
        { ...base, vendor: 'apple', architecture: 'metal-3' },
        { ...base, vendor: 'nvidia', architecture: 'ampere', description: 'NVIDIA GeForce RTX 3080' },
        { ...base, vendor: 'intel', architecture: 'gen-12lp' },
        { ...base, description: 'llvmpipe', ok: false, why: 'requestDevice() failed' },
        { ok: false, why: 'navigator.gpu is undefined' },
        null,
    ];
    for (const info of software) assert.equal(isSoftwareAdapter(info), true, describeAdapter(info));
    for (const info of hardware) assert.equal(isSoftwareAdapter(info), false, describeAdapter(info));
});
