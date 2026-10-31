#!/usr/bin/env node
// Validates every WGSL module the examples compile, offline with naga (no GPU, no browser).
//
// Each js/examples/*.js that calls createShaderModule must export
//   wgslModules() -> [{ name, code }, ...]
// returning the exact strings it passes to createShaderModule (every variant, or a set covering
// every code path). This script writes each module to a temp dir and runs `naga <file>.wgsl`.
//
// Usage: node tools/validate-wgsl.mjs        (naga from $NAGA, else `naga` on PATH)
// Exits non-zero if any module fails to validate, if an example creates shader modules without
// exporting wgslModules(), or if naga cannot be run. Warns when the number of exported modules is
// smaller than the number of createShaderModule( call sites in the file.

import { readdirSync, readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const examplesDir = join(root, 'js', 'examples');
const naga = process.env.NAGA || 'naga';

const version = spawnSync(naga, ['--version'], { encoding: 'utf8' });
if (version.error || version.status !== 0) {
    console.error(`Cannot run naga (${naga}): ${version.error?.message ?? version.stderr}`);
    console.error('Install it with `cargo install naga-cli` or set NAGA=/path/to/naga.');
    process.exit(2);
}
console.log(`naga ${version.stdout.trim()}`);

const outDir = mkdtempSync(join(tmpdir(), 'validate-wgsl-'));
let failures = 0;
let total = 0;

try {
    for (const file of readdirSync(examplesDir).filter(f => f.endsWith('.js')).sort()) {
        const path = join(examplesDir, file);
        const source = readFileSync(path, 'utf8');
        const callSites = (source.match(/createShaderModule\(/g) || []).length;
        let mod;
        try {
            mod = await import(pathToFileURL(path).href);
        } catch (e) {
            // Examples must not touch browser globals (GPUBufferUsage, document, ...) at import time.
            console.log(`FAIL ${file}: import failed: ${e.message}`);
            failures++;
            continue;
        }

        if (typeof mod.wgslModules !== 'function') {
            if (callSites > 0) {
                console.log(`FAIL ${file}: calls createShaderModule but does not export wgslModules()`);
                failures++;
            }
            continue;
        }

        const modules = mod.wgslModules();
        const stem = file.replace(/\.js$/, '');
        if (modules.length < callSites) {
            console.warn(`WARN ${file}: exports ${modules.length} module(s) but has ${callSites} createShaderModule( call site(s)`);
        }
        const seen = new Set();
        for (const { name, code } of modules) {
            total++;
            const label = `${stem}/${name}`;
            if (typeof code !== 'string' || !code.trim() || seen.has(name)) {
                console.log(`FAIL ${label}: ${seen.has(name) ? 'duplicate name' : 'code is not a non-empty string'}`);
                failures++;
                continue;
            }
            seen.add(name);
            const wgsl = join(outDir, `${stem}.${name}.wgsl`);
            writeFileSync(wgsl, code);
            const r = spawnSync(naga, [wgsl], { encoding: 'utf8' });
            if (r.status === 0) {
                console.log(`OK   ${label}`);
            } else {
                console.log(`FAIL ${label}`);
                console.log((r.stderr || r.stdout || String(r.error)).trim().replace(/^/gm, '     '));
                failures++;
            }
        }
    }
} finally {
    rmSync(outDir, { recursive: true, force: true });
}

console.log(`\n${total} module(s) checked, ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
