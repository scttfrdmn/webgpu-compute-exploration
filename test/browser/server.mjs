// Minimal static file server for the browser suite: serves the repository root on 127.0.0.1 (a secure
// context, so navigator.gpu exists). `node test/browser/server.mjs [port]`, or import startServer().
// Serving files is harmless, so this module has no CI guard of its own; the suite's entry points do.

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('../../', import.meta.url)));
const TYPES = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.md': 'text/markdown; charset=utf-8',
    '.wasm': 'application/wasm', '.png': 'image/png', '.svg': 'image/svg+xml', '.bin': 'application/octet-stream',
    '.txt': 'text/plain; charset=utf-8', '.ico': 'image/x-icon',
};

export function startServer(port = 0, host = '127.0.0.1') {
    const server = createServer(async (req, res) => {
        try {
            const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
            let file = normalize(join(ROOT, path));
            if (file !== ROOT && !file.startsWith(ROOT + sep)) { res.writeHead(403).end(); return; }
            if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
            const body = await readFile(file);
            res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
            res.end(body);
        } catch {
            res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
        }
    });
    return new Promise((done) => server.listen(port, host, () => done(server)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const server = await startServer(Number(process.argv[2] ?? process.env.PORT ?? 8123));
    console.log(`[test/browser] serving ${ROOT} at http://127.0.0.1:${server.address().port}/`);
}
