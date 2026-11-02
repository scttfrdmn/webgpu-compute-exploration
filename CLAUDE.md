# CLAUDE.md

Static WebGPU demo site: `index.html` + ES modules in `js/`, no bundler, no npm dependencies.
One example (#10) uses a Rust/WASM crate in `wasm/`.

## Run / build
- Serve: `python3 tools/serve.py` (or `npm run dev`; it is `http.server` with `Cache-Control: no-cache`, so browsers
  never run stale cached modules), open http://localhost:8000 in a WebGPU browser.
- WASM: `./wasm/build.sh` → `js/wasm-pkg/` (gitignored). Needs a `rustc` with `wasm32-unknown-unknown`;
  on this machine Homebrew's `rustc` shadows rustup's, so run with `PATH=~/.cargo/bin:$PATH`.
- Checks (same as CI): `for f in $(git ls-files '*.js'); do node --check "$f"; done`, the WASM build, and
  `npm run validate:wgsl` (set `NAGA=/path/to/naga` if it is not on PATH), and `npm test` (`node --test 'test/*.test.mjs'`).
- `test/` is CPU-only: it runs the examples' exported JS and CPU replicas of the shader algorithms. Rendering
  and GPU behaviour still have to be verified in a browser. Never run tests on the GPU or drive a browser locally.
- `test/gpu/` (`*.gpu.test.mjs`) executes the shipped WGSL through Dawn's Node bindings (`webgpu` npm, version pinned
  in the `gpu-software` CI job) on Mesa lavapipe and compares with the CPU references. It is **CI-only**: never run
  it, `npm install webgpu`, or any other WebGPU runtime on a developer's machine from automation — iterate by pushing
  a branch and reading the CI log. `test/gpu/harness.mjs` aborts unless `CI=true`, Linux, and a software adapter;
  keep that guard. `npm test` must never import `test/gpu/`. Pure CPU references (float64 DFT, CPU sort, direct
  N-body sum, …) live in `js/refs/` so the in-browser Measure tab can use them too (no `node:` imports, no DOM,
  no test mocks); test-only helpers (mocks, layout parser, kernel replicas) stay in `test/helpers/`.
- WGSL is validated without a GPU by `tools/validate-wgsl.mjs`: it imports each example's `wgslModules()` and runs
  `naga` (naga-cli 30.0.1, pinned in `.github/workflows/ci.yml`) on every module. It catches the same errors
  Chrome's compiler reports. Examples must not touch browser globals at import time.
- 3D examples use column-major matrices (`Mat4` in molecular-dynamics.js / dla-3d.js) with WebGPU's [0, 1] depth range.
- Simulation parameters can be checked without a GPU by porting the shader math to a small Node script (done for SPH,
  DLA and the ray marcher).
- Uniform structs with `u32` fields must be written through a `Uint32Array` view of the buffer, and vec2/vec3/vec4
  fields follow WGSL alignment (8/16 bytes) — writing ints as floats turns small counts into ~1e9.
- `test/browser/` (Playwright, `*.spec.mjs`) drives the real page in headless Chromium on a software WebGPU adapter
  (SwiftShader; lavapipe forced as the only Vulkan ICD) in the CI `browser` job: tabs, deep links, lifecycle, Measure
  runs and Cancel, the no-timestamp fallback, device loss, the learning path, the device report. It is **CI-only**:
  `test/browser/guard.mjs` refuses to run unless `CI=true` on Linux with `BROWSER_FLAGS` from `probe.mjs`, and the
  adapter must be software. Never run `npx playwright`, `playwright install` or the suite locally; iterate by pushing a
  `ci/**` branch. `npm test` must not import `test/browser/`.
- Do not drive the page with headless Chrome + `--enable-unsafe-webgpu` on macOS: loading it starts six GPU
  simulations at once and hung the machine (see #32). Ask the user to check in their own browser.

## Layout
- `js/main.js` — gets the GPU device, wires the one-shot buttons, and creates each interactive example when its
  canvas nears the viewport (IntersectionObserver). Nothing simulates until Start; ray marching runs only while visible.
  A running example pauses when its canvas scrolls off screen and resumes when it returns, unless the user paused it
  or left its Explore tab (audio examples are never auto-resumed). `js/example-control.js` `pauseExample` does the
  pausing through the example's own buttons; new examples should have `-pause`/`-start` buttons or a primary toggle.
- `js/webgpu-utils.js` — `getGPUDevice`, `createBuffer`, `readBuffer` helpers.
- `js/device-report.js` — `buildDeviceReport` for the header's opt-in "Copy device report" button. It reads only a
  whitelist of plain number/boolean fields (`fps`, `running`, `steps`, `numParticles`, …) from example instances and
  never calls their methods; expose new measurements as such fields to have them reported.
- `data/mnist/` — MNIST subset + cached weights for examples 21/22 (CC BY-SA 3.0, not MIT); `tools/train-mnist.mjs`
  regenerates `weights.bin` on the CPU with the same math as the GPU kernels in `js/examples/mnist-model.js`.
- `js/examples/<name>.js` — one self-contained example per file; WGSL shaders are inline template strings.
  One-shot benchmarks export `run*(gpuDevice)` returning HTML; interactive ones export `init*(gpuDevice)`.
  Every `start()` must return early if already running, or repeated clicks stack render loops.
- `css/site.css` — all styling, shared by `index.html` and `learn.html` (CSS variables at `:root`).
- `js/teach/` — the teaching layer: `tabs.js` adds Explore / Understand / Measure tabs to every example (deep
  links `#<id>/understand`, `#<id>/measure`; leaving Explore stops the example via its own Pause/Stop button),
  `markdown.js` + `tex-mathml.js` render `content/understand/<id>.md` (escaped Markdown, `$…$` TeX subset → MathML),
  `measure.js` + `chart.js` run `js/teach/experiments/<id>.js` and chart the results, `registry.js` lists which
  examples have content, `app-state.js` shares the device and instances with `main.js`, `stats.js` summarizes
  repeats (median, IQR, CV), `result-record.js` builds and validates schema v1 records (Download / Compare).
- `js/teach/flagships.js` — data for the landing page's three reference investigations; each card cites `interpret()`
  claims by their stable `key`, and `test/flagships.test.mjs` fails if a card's wording drifts from the claims.
- `schema/experiment-result.v1.json` — the experiment result schema. `tools/write-version.mjs` writes `version.json`
  (version, commit, build time) into the Pages build; `js/source-info.js` reads it (commit "local" when served locally).
- `validate.html` + `js/teach/validation.js` (plan, environment, bundle; pure) + `js/teach/validate.js` (runner and
  page) — the physical-GPU validation run. Bump `VALIDATION_PLAN.version` whenever its entries or settings change.
  Bundles go in `data/validation/<env>/` (environments from `data/validation/matrix.json`); then run
  `node tools/validation-summary.mjs --write` — `test/validation.test.mjs` fails if the README block is stale.
- `learn.html` + `js/teach/path.js` (data) + `js/teach/learn.js` (renderer) — the learning path; it never loads
  WebGPU. New examples should be placed in a track step; `test/learn-path.test.mjs` requires every example to appear.
- `index.html` — page structure. Each example is a
  `<section class="example" id="…">` with an `<h3>`, listed in the sticky `nav.toc`; `.controls`, `.seed-buttons`,
  `.hint` and `.output` (hidden while empty) are the shared building blocks. Canvases are CSS-scaled, so pointer
  handlers must convert client coordinates with `canvas.width / rect.width`.
- `assets/social-preview.png` — 1280×640 Mandelbulb render used by the README and Open Graph tags.

## Adding an example
1. Add `js/examples/<name>.js`, a `<section class="example">` (plus a `nav.toc` link) in `index.html`, and register
   its canvas id in `initInteractiveExamplesOnVisible` in `js/main.js` (or wire a button for one-shot examples).
2. Target 60 FPS (30 minimum for heavy rendering); offer size/quality controls, presets, and a performance readout.
   Measure GPU time with `js/gpu-timer.js` (`GpuTimer`: pass descriptors get `timestampWrites`, `resolve(encoder)`
   before `finish()`, `read()` after submit, never awaited in the animation loop) and show it next to wall time
   ("GPU: x.xx ms · wall: y.yy ms"); it is a no-op when the adapter lacks `timestamp-query`.
3. Comment the shader with the algorithm. Export `wgslModules()` returning `[{ name, code }]` with every module
   passed to `createShaderModule` (each variant, if code is built at runtime); CI fails examples without it.
4. Add `test/<name>.test.mjs` (`node:test` and `node:assert` only): run the example's exported pure JS
   directly, check each uniform writer's bytes against its WGSL struct (`test/helpers/wgsl-layout.mjs` and a
   `test/helpers/mock-gpu.mjs` device), and compare a labeled CPU replica of the shader algorithm to a
   straightforward reference with a documented tolerance, printing the measured numbers.
5. Add it to the README feature list and project structure, and to `CHANGELOG.md` under Unreleased.
6. Teaching content (see `content/understand/nbody.md` and `js/teach/experiments/nbody.js` as the template):
   an Understand page with the outline in `test/teach-content.test.mjs` (`REQUIRED_SECTIONS`), and 1–3 Measure
   experiments (interface at the top of `js/teach/measure.js`) that use fixed parameter lists with hard caps, a
   `budgetMs` ≤ 20 s, `await ctx.step()` between steps, and a CPU reference from `js/refs/`. The budget is checked
   between steps, so each step's GPU work must be capped on its own. Time with `ctx.measure(runOnce, { warmup, repeats })`
   (medians + IQR via `withSpread`; phases via `js/phase-clock.js`), declare `columns[].unit`, and add
   `verify(rows, settings)` checks against references where a result can be right or wrong; every run produces a
   schema v1 record (`schema/experiment-result.v1.json`, built by `js/teach/result-record.js`). GPU work for
   experiments goes through a small compute-only export of the example (no canvas), covered by a CPU test and a
   `test/gpu/` test. List the example in `js/teach/registry.js`; `npm test` lints all registered content.

## Project tracking
Work lives in GitHub issues, milestones (v0.1.0 onward) and project board #85 — not in local files.
Labels: `type: *`, `priority: *`, `area: *`. Stay on 0.x versions.
