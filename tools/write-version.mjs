// Writes <dir>/version.json = { version, commit, builtAt } for the deployed site (run by
// .github/workflows/pages.yml; read by js/source-info.js). commit comes from $GITHUB_SHA.
//   node tools/write-version.mjs _site

import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export function versionRecord(packageJson, commit, now = new Date()) {
    if (!/^[0-9a-f]{40}$/.test(commit ?? '')) throw new Error(`GITHUB_SHA is not a commit SHA: ${commit}`);
    return { version: packageJson.version, commit, builtAt: now.toISOString() };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const dir = process.argv[2];
    if (!dir) throw new Error('usage: node tools/write-version.mjs <site dir>');
    const root = join(dirname(fileURLToPath(import.meta.url)), '..');
    const record = versionRecord(JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')), process.env.GITHUB_SHA);
    writeFileSync(join(dir, 'version.json'), JSON.stringify(record) + '\n');
    console.log(`version.json: ${JSON.stringify(record)}`);
}
