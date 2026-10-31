# CLAUDE.md

Static WebGPU demo site: `index.html` + ES modules in `js/`, no bundler, no npm dependencies.
One example (#10) uses a Rust/WASM crate in `wasm/`.

## Run / build
- Serve: `python3 -m http.server 8000` (or `npm run dev`), open http://localhost:8000 in a WebGPU browser.
- WASM: `./wasm/build.sh` → `js/wasm-pkg/` (gitignored). Needs a `rustc` with `wasm32-unknown-unknown`;
  on this machine Homebrew's `rustc` shadows rustup's, so run with `PATH=~/.cargo/bin:$PATH`.
- Checks (same as CI): `for f in $(git ls-files '*.js'); do node --check "$f"; done`, the WASM build, and
  `npm run validate:wgsl` (set `NAGA=/path/to/naga` if it is not on PATH), and `npm test` (`node --test 'test/*.test.mjs'`).
- `test/` is CPU-only: it runs the examples' exported JS and CPU replicas of the shader algorithms. Rendering
  and GPU behaviour still have to be verified in a browser. Never run tests on the GPU or in a headless browser.
- `test/gpu/` (`*.gpu.test.mjs`) executes the shipped WGSL through Dawn's Node bindings (`webgpu` npm, version pinned
  in the `gpu-software` CI job) on Mesa lavapipe and compares with the CPU references. It is **CI-only**: never run
  it, `npm install webgpu`, or any other WebGPU runtime on a developer's machine from automation — iterate by pushing
  a branch and reading the CI log. `test/gpu/harness.mjs` aborts unless `CI=true`, Linux, and a software adapter;
  keep that guard. `npm test` must never import `test/gpu/`; shared references go in `test/helpers/`.
- WGSL is validated without a GPU by `tools/validate-wgsl.mjs`: it imports each example's `wgslModules()` and runs
  `naga` (naga-cli 30.0.1, pinned in `.github/workflows/ci.yml`) on every module. It catches the same errors
  Chrome's compiler reports. Examples must not touch browser globals at import time.
- 3D examples use column-major matrices (`Mat4` in molecular-dynamics.js / dla-3d.js) with WebGPU's [0, 1] depth range.
- Simulation parameters can be checked without a GPU by porting the shader math to a small Node script (done for SPH,
  DLA and the ray marcher).
- Uniform structs with `u32` fields must be written through a `Uint32Array` view of the buffer, and vec2/vec3/vec4
  fields follow WGSL alignment (8/16 bytes) — writing ints as floats turns small counts into ~1e9.
- Do not drive the page with headless Chrome + `--enable-unsafe-webgpu` on macOS: loading it starts six GPU
  simulations at once and hung the machine (see #32). Ask the user to check in their own browser.

## Layout
- `js/main.js` — gets the GPU device, wires the one-shot buttons, and creates each interactive example when its
  canvas nears the viewport (IntersectionObserver). Nothing simulates until Start; ray marching runs only while visible.
- `js/webgpu-utils.js` — `getGPUDevice`, `createBuffer`, `readBuffer` helpers.
- `data/mnist/` — MNIST subset + cached weights for examples 21/22 (CC BY-SA 3.0, not MIT); `tools/train-mnist.mjs`
  regenerates `weights.bin` on the CPU with the same math as the GPU kernels in `js/examples/mnist-model.js`.
- `js/examples/<name>.js` — one self-contained example per file; WGSL shaders are inline template strings.
  One-shot benchmarks export `run*(gpuDevice)` returning HTML; interactive ones export `init*(gpuDevice)`.
  Every `start()` must return early if already running, or repeated clicks stack render loops.
- `index.html` — all styling is inline in `<style>` (CSS variables at `:root`). Each example is a
  `<section class="example" id="…">` with an `<h3>`, listed in the sticky `nav.toc`; `.controls`, `.seed-buttons`,
  `.hint` and `.output` (hidden while empty) are the shared building blocks. Canvases are CSS-scaled, so pointer
  handlers must convert client coordinates with `canvas.width / rect.width`.
- `assets/social-preview.png` — 1280×640 Mandelbulb render used by the README and Open Graph tags.

## Adding an example
1. Add `js/examples/<name>.js`, a `<section class="example">` (plus a `nav.toc` link) in `index.html`, and register
   its canvas id in `initInteractiveExamplesOnVisible` in `js/main.js` (or wire a button for one-shot examples).
2. Target 60 FPS (30 minimum for heavy rendering); offer size/quality controls, presets, and a performance readout.
3. Comment the shader with the algorithm. Export `wgslModules()` returning `[{ name, code }]` with every module
   passed to `createShaderModule` (each variant, if code is built at runtime); CI fails examples without it.
4. Add `test/<name>.test.mjs` (`node:test` and `node:assert` only): run the example's exported pure JS
   directly, check each uniform writer's bytes against its WGSL struct (`test/helpers/wgsl-layout.mjs` and a
   `test/helpers/mock-gpu.mjs` device), and compare a labeled CPU replica of the shader algorithm to a
   straightforward reference with a documented tolerance, printing the measured numbers.
5. Add it to the README feature list and project structure, and to `CHANGELOG.md` under Unreleased.

## Project tracking
Work lives in GitHub issues, milestones (v0.1.0–v0.4.0) and project board #85 — not in local files.
Labels: `type: *`, `priority: *`, `area: *`. Stay on 0.x versions.
