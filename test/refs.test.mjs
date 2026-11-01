// Deployable CPU references (js/refs/*.js), shared by the test suites and the site.
//
// Each module must load in a browser page as is: it imports cleanly with no globals installed
// (no window / document / navigator.gpu, no mocks), imports nothing from node:*, examples or tests,
// and leaves globalThis and Math.random untouched. Every exported function must be exercised by an
// existing test, checked statically: the function is imported from js/refs/ by a test file, or by a
// test helper that uses it or re-exports it under a name some test imports from that helper.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, relative, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const refsDir = join(root, 'js', 'refs');
const modules = readdirSync(refsDir).filter(f => f.endsWith('.js')).sort();

const read = p => readFileSync(p, 'utf8');
const stripComments = src => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:\\])\/\/.*$/gm, '$1');
// Static `import ... from 'x'`, `export ... from 'x'`, bare `import 'x'` and dynamic `import('x')`.
const specifiers = src => [...stripComments(src).matchAll(/(?:\bfrom\s*|\bimport\s*\(?\s*)(['"])([^'"]+)\1/g)].map(m => m[2]);

// `import { a, b as c } from 'spec'` / `export { a as b } from 'spec'` clauses: [{ kind, spec, names: [[orig, local]] }]
function clauses(src) {
    const out = [];
    for (const m of stripComments(src).matchAll(/\b(import|export)\s*\{([^}]*)\}\s*from\s*(['"])([^'"]+)\3/g)) {
        const names = m[2].split(',').map(s => s.trim()).filter(Boolean)
            .map(s => { const [orig, local = orig] = s.split(/\s+as\s+/); return [orig.trim(), local.trim()]; });
        out.push({ kind: m[1], spec: m[4], names, text: m[0] });
    }
    return out;
}
const usesWord = (src, word) => new RegExp(`(^|[^\\w$.])${word.replace(/\$/g, '\\$')}(?![\\w$])`).test(src);

function testFiles() {
    const list = [];
    for (const dir of ['test', 'test/gpu', 'test/helpers'])
        for (const f of readdirSync(join(root, dir)))
            if (f.endsWith('.mjs') && !(dir === 'test' && f === 'refs.test.mjs')) list.push(join(root, dir, f));
    return list;
}

test('js/refs modules exist and each starts with a header comment', () => {
    assert.ok(modules.length >= 10, `found ${modules.length} modules`);
    for (const m of modules) assert.match(read(join(refsDir, m)), /^\/\/ .*reference|^\/\/ .*PRNG/i, `${m}: header comment`);
});

test('js/refs modules import only sibling js/refs modules (no node:*, examples, tests)', () => {
    for (const m of modules) {
        for (const spec of specifiers(read(join(refsDir, m)))) {
            assert.ok(!spec.startsWith('node:'), `${m} imports ${spec}`);
            assert.match(spec, /^\.\/[\w-]+\.js$/, `${m} imports ${spec} (only ./sibling.js allowed)`);
            assert.ok(modules.includes(spec.slice(2)), `${m} imports missing ${spec}`);
        }
    }
});

test('js/refs modules import with no globals installed and have no top-level side effects', async (t) => {
    assert.equal(typeof globalThis.window, 'undefined');
    assert.equal(typeof globalThis.document, 'undefined');
    assert.equal(globalThis.navigator?.gpu, undefined);
    const before = new Set(Reflect.ownKeys(globalThis));
    const random = Math.random;
    for (const m of modules) {
        const ns = await import(pathToFileURL(join(refsDir, m)).href);
        assert.ok(Object.keys(ns).length > 0, `${m} exports something`);
        t.diagnostic(`${m}: ${Object.keys(ns).join(', ')}`);
    }
    const added = Reflect.ownKeys(globalThis).filter(k => !before.has(k));
    assert.deepEqual(added, [], 'importing js/refs added globals');
    assert.equal(Math.random, random, 'importing js/refs replaced Math.random');
});

test('every exported js/refs function is used by an existing test', async () => {
    const files = testFiles().map(p => ({ p, rel: relative(root, p), src: read(p) }));
    const isRefs = (file, spec) => resolve(dirname(file), spec).startsWith(refsDir);
    const covered = new Map();   // "module.js:name" -> where
    for (const { p, rel, src } of files) {
        const isHelper = rel.startsWith('test/helpers/');
        for (const c of clauses(src)) {
            if (!isRefs(p, c.spec)) continue;
            const mod = relative(refsDir, resolve(dirname(p), c.spec));
            for (const [orig, local] of c.names) {
                let ok;
                if (!isHelper) ok = c.kind === 'export' || usesWord(src.replace(c.text, ''), local);
                else if (c.kind === 'import') ok = usesWord(src.replace(c.text, ''), local);
                else {
                    // Re-exported by a helper: some test must import that name from the helper.
                    const helper = rel.split('/').pop();
                    ok = files.some(f => !f.rel.startsWith('test/helpers/') && f.src.includes(helper) && usesWord(f.src, local));
                }
                if (ok) covered.set(`${mod}:${orig}`, rel);
            }
        }
    }
    const missing = [];
    for (const m of modules) {
        const ns = await import(pathToFileURL(join(refsDir, m)).href);
        // A helper called by another, covered export of the same module (e.g. metric64 inside
        // errorsVsBrute) is exercised through it: its name occurs beyond its own declaration.
        const body = stripComments(read(join(refsDir, m)));
        const anyCovered = Object.keys(ns).some(n => covered.has(`${m}:${n}`));
        const calledInside = name => anyCovered
            && (body.match(new RegExp(`(^|[^\\w$.])${name}\\(`, 'gm')) ?? []).length > 1;
        for (const [name, value] of Object.entries(ns))
            if (typeof value === 'function' && !covered.has(`${m}:${name}`) && !calledInside(name)) missing.push(`${m}:${name}`);
    }
    assert.deepEqual(missing, [], 'exported functions with no test use');
});
