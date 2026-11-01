import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Subset of WGSL reserved words; using any of these as an identifier fails to compile.
const WGSL_RESERVED = new Set(['target', 'filter', 'common', 'active', 'pass', 'patch', 'shared', 'static', 'super', 'template', 'this', 'type', 'use', 'with']);

test('ray-marching shader uses no WGSL reserved words as identifiers', () => {
    const source = readFileSync(new URL('./ray-marching.js', import.meta.url), 'utf8');
    const shader = source.match(/const RAY_MARCH_SHADER = `([\s\S]*?)`;/)[1];
    const code = shader.replace(/\/\/.*$/gm, '');
    const used = [...new Set(code.match(/\b[A-Za-z_]\w*\b/g))].filter(id => WGSL_RESERVED.has(id));
    assert.deepEqual(used, [], `WGSL reserved word used as identifier: ${used.join(', ')}`);
});
