#!/usr/bin/env python3
"""Local development server: python3 tools/serve.py [port] (default 8000).

Same as `python3 -m http.server`, but every response says Cache-Control: no-cache. The plain server sends
no cache headers, so browsers reuse ES modules fetched days earlier without asking; a page whose imports
changed then fails to link against the stale copy and never starts.
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = Path(__file__).resolve().parent.parent
    server = ThreadingHTTPServer(('', port), partial(NoCacheHandler, directory=str(root)))
    print(f'Serving {root} at http://localhost:{port}/ (no-cache)')
    server.serve_forever()
