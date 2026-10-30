# CLAUDE.md

Static WebGPU demo site: `index.html` + ES modules in `js/`, no bundler, no npm dependencies.
One example (#10) uses a Rust/WASM crate in `wasm/`.

## Run / build
- Serve: `python3 -m http.server 8000` (or `npm run dev`), open http://localhost:8000 in a WebGPU browser.
- WASM: `./wasm/build.sh` → `js/wasm-pkg/` (gitignored). Needs a `rustc` with `wasm32-unknown-unknown`;
  on this machine Homebrew's `rustc` shadows rustup's, so run with `PATH=~/.cargo/bin:$PATH`.
- Checks (same as CI): `for f in $(git ls-files '*.js'); do node --check "$f"; done` and the WASM build.
- There are no automated tests; examples must be verified in a browser.
- Do not drive the page with headless Chrome + `--enable-unsafe-webgpu` on macOS: loading it starts six GPU
  simulations at once and hung the machine (see #32). Ask the user to check in their own browser.

## Layout
- `js/main.js` — gets the GPU device, wires buttons, eagerly inits the interactive examples.
- `js/webgpu-utils.js` — `getGPUDevice`, `createBuffer`, `readBuffer` helpers.
- `js/examples/<name>.js` — one self-contained example per file; WGSL shaders are inline template strings.
  One-shot benchmarks export `run*(gpuDevice)` returning HTML; interactive ones export `init*(gpuDevice)`.

## Adding an example
1. Add `js/examples/<name>.js`, a section with canvas/controls in `index.html`, and wire it in `js/main.js`.
2. Target 60 FPS (30 minimum for heavy rendering); offer size/quality controls, presets, and a performance readout.
3. Comment the shader with the algorithm.
4. Add it to the README feature list and project structure, and to `CHANGELOG.md` under Unreleased.

## Project tracking
Work lives in GitHub issues, milestones (v0.1.0–v0.4.0) and project board #85 — not in local files.
Labels: `type: *`, `priority: *`, `area: *`. Stay on 0.x versions.
