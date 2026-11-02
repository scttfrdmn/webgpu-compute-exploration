// Writes version.json at the repository root from the checked-out commit, so a page served locally
// (`python3 -m http.server`) records which commit it ran instead of "local" — needed for validation bundles
// (validate.html, #78), whose file names and source fields carry the commit. Refuses when tracked files have
// uncommitted changes (the commit would not describe what ran). Do not commit the generated file.
//   node tools/local-version.mjs           # write version.json
//   node tools/local-version.mjs --remove  # delete it again

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { versionRecord } from './write-version.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const target = join(root, 'version.json');
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();

if (process.argv.includes('--remove')) {
    if (existsSync(target)) rmSync(target);
    console.log('version.json removed');
} else {
    const dirty = git('status', '--porcelain', '--untracked-files=no');
    if (dirty) {
        console.error(`Tracked files have uncommitted changes; commit or stash them first:\n${dirty}`);
        process.exit(1);
    }
    const record = versionRecord(JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')), git('rev-parse', 'HEAD'));
    writeFileSync(target, JSON.stringify(record) + '\n');
    console.log(`version.json: ${JSON.stringify(record)} (remove with --remove; do not commit it)`);
}
