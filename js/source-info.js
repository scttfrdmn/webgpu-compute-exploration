// Which revision of the site is running, for result records (js/teach/result-record.js) and the device
// report. The Pages workflow (.github/workflows/pages.yml) writes version.json next to index.html:
//     { "version": "0.6.0", "commit": "<40-hex GITHUB_SHA>", "builtAt": "<ISO time>" }
// A local checkout served with `python3 -m http.server` has no version.json (the request 404s, which is
// expected): the version then comes from package.json and the commit is "local".

export const REPO_URL = 'https://github.com/scttfrdmn/webgpu-compute-exploration';
export const LOCAL_COMMIT = 'local';

const COMMIT = /^[0-9a-f]{7,40}$/;

// Pure: { version, commit, builtAt, url } from the parsed version.json (or null) and package.json (or null)
export function resolveSourceInfo(versionJson, packageJson) {
    if (versionJson && typeof versionJson.version === 'string' && versionJson.version
        && typeof versionJson.commit === 'string' && COMMIT.test(versionJson.commit)) {
        return {
            version: versionJson.version,
            commit: versionJson.commit,
            builtAt: typeof versionJson.builtAt === 'string' ? versionJson.builtAt : null,
            url: `${REPO_URL}/tree/${versionJson.commit}`,
        };
    }
    const version = typeof packageJson?.version === 'string' && packageJson.version ? packageJson.version : 'unknown';
    return { version, commit: LOCAL_COMMIT, builtAt: null, url: null };
}

async function fetchJSON(fetchFn, path) {
    if (typeof fetchFn !== 'function') return null;
    try {
        const response = await fetchFn(path, { cache: 'no-store' });
        return response.ok ? await response.json() : null;
    } catch {
        return null;
    }
}

export async function loadSourceInfo(fetchFn = globalThis.fetch) {
    const deployed = resolveSourceInfo(await fetchJSON(fetchFn, 'version.json'), null);
    if (deployed.commit !== LOCAL_COMMIT) return deployed;
    return resolveSourceInfo(null, await fetchJSON(fetchFn, 'package.json'));
}

let cached = null;
// Once per page load
export function sourceInfo() {
    cached ??= loadSourceInfo();
    return cached;
}
