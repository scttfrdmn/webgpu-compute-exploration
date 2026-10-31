# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

### Added
- Every example exports `wgslModules()`, the exact WGSL it passes to `createShaderModule`, and
  `tools/validate-wgsl.mjs` (`npm run validate:wgsl`) validates all 63 modules with naga. A new
  `wgsl` CI job runs it on every push and pull request, and fails if an example creates shader
  modules without exporting them.
- CPU reference test suite in `test/` (`npm test`, i.e. `node --test 'test/*.test.mjs'`; Node built-ins only, no
  GPU or browser), one file per example. Tests run the examples' exported JavaScript directly, check
  every uniform writer's bytes against offsets computed from the example's WGSL struct, and compare
  CPU replicas of the shader algorithms against straightforward references (naive DFT, CPU sort,
  brute-force Voronoi, direct N-body sum, direct convolution, queue BFS, finite-difference gradients,
  energy and stability checks), printing the measured errors. A new `test` CI job runs it.

### Changed
- package.json description and keywords, the CITATION abstract, and the GitHub
  repository description and topics now cover all 25 examples.

## [0.4.0] - 2026-10-31

### Added
- Example 19, cloth simulation: a 32×32 to 128×128 particle fabric with
  structural, shear and bend links, solved with position-based dynamics on the
  GPU (Verlet integration, long-range tethers to the pins, Gauss-Seidel
  constraint iterations split into 12 race-free color sets, sphere and floor
  collisions with Coulomb friction). Wind, optional tearing with a threshold,
  shift-drag grabbing, five presets (two corners, top edge, curtain, drop on
  sphere, flag), wireframe and particle views, and an FPS readout.
- Example 20, Monte Carlo path tracer: a compute shader traces one path per
  pixel per frame (spheres, quads, rotated boxes and planes; diffuse, metal,
  glass and emissive materials; next-event estimation with MIS, Russian
  roulette, 1-8 bounces) into an accumulation buffer, tone-mapped with ACES or
  Reinhard. Cornell box, Spheres, Caustics and Materials scenes; orbit camera
  with depth of field and click-to-focus; resolution, exposure and firefly
  clamp controls; spp, FPS and rays-per-second readout; stops at a target spp.
- Example 21, MNIST: Train: a small CNN (conv 5×5×8, max-pool, conv 5×5×16,
  max-pool, dense 256→10; 5,994 weights) trained in the browser entirely in
  compute shaders on 10,000 MNIST images: GPU batch assembly with random ±2 px
  shifts, forward pass, softmax cross-entropy, full backprop, race-free
  per-sample gradient reduction and SGD with momentum, weight decay and
  learning-rate decay; live loss and accuracy charts, test accuracy on 2,000
  images after every epoch, first-layer filters, Export weights and Use in
  Recognize. A CPU trainer with the same algorithm (`tools/train-mnist.mjs`)
  produced the cached weights (98.00% test accuracy).
- Example 22, MNIST: Recognize: draw a digit (mouse, pen or touch), MNIST-style
  preprocessing (fit 20×20, center of mass) and GPU inference with the cached
  weights or those from example 21; confidence bars, the 28×28 network input,
  activation maps of both conv layers, and random test digits with their true
  labels. MNIST data and derived weights are CC BY-SA 3.0 (`data/mnist/`).
- Example 23, N-body gravity: 4,096 to 262,144 bodies integrated with
  kick-drift-kick leapfrog, with a brute-force O(N²) solver (workgroup-shared
  tiles, up to 32,768 bodies) and Barnes-Hut with the tree rebuilt on the GPU
  every step (bounding-box reduction, Morton codes, radix sort, Karras radix
  tree, bottom-up centers of mass with atomic visit counters, stack-based
  traversal with a θ slider). Spiral galaxy, galaxy collision, Plummer sphere
  and cold collapse presets; glowing additive sprites colored by speed or
  origin, optional trails, orbit camera; FPS, interactions per frame and an
  energy-drift readout from a once-per-second GPU reduction.
- Example 24, convolution reverb: uniformly partitioned overlap-save FFT
  convolution on the GPU (frequency-domain delay line, one multiply-accumulate
  dispatch over up to 563 partitions, stereo packed into one complex FFT) of
  a live signal streamed through an AudioWorklet, with a fixed latency of
  (1 + D) blocks and an underrun counter. Procedural Small room, Concert hall,
  Cathedral, Plate and Reverse impulse responses (frequency-dependent decay,
  early reflections, pre-delay) or an uploaded WAV; synth, microphone and file
  sources; decay, pre-delay, mix, output gain, Bypass and an A/B switch to the
  browser's ConvolverNode. Shows the IR waveform, spectrogram, Schroeder decay
  and RT60, an input/wet scope and level meters, plus a GPU-vs-CPU accuracy
  self-test and realtime throughput.

## [0.3.0] - 2026-10-30

### Added
- The WASM Mandelbrot example is now number 19.
- Example 14, FFT audio visualizer: Web Audio samples from a built-in synth
  demo, the microphone or an audio file are windowed (Hann) and transformed on
  the GPU by a radix-2 Cooley-Tukey FFT (bit reversal, then one dispatch per
  butterfly stage) of size 1024, 2048 or 4096; dB magnitudes, a log-frequency
  spectrogram ring buffer and beat detection also run on the GPU. Bars,
  Waveform and Spectrogram views, smoothing and dB range sliders, presets, a
  beat flash, Mute, and an FPS readout.
- Example 15, parallel sorting: bitonic sort (global passes, or workgroup
  memory for j ≤ 256) and 4-bit LSD radix sort benchmarked on 65,536 to
  4,194,304 u32 or f32 keys, verified against the CPU sort and timed against
  `TypedArray.prototype.sort`, plus an animated bar chart of bitonic sort with
  Play / Pause / Step / Shuffle, data patterns, a stages-per-second slider and
  the current k and j.
- Example 16, K-means clustering: Lloyd's algorithm on 10k-500k 2D points
  (Gaussian blobs plus uniform noise) with K = 2-16, one GPU iteration per step:
  nearest-centroid assignment, a two-level workgroup reduction for the cluster
  means (empty clusters keep their centroid), and a changed-points counter that
  stops the run at convergence. Random or k-means++ initialization, centroid
  trails, an optional Voronoi overlay, Start/Pause/Step/Reset/Regenerate,
  iteration speed, and iteration, changed points, inertia and FPS readouts.
- Example 17, Voronoi diagrams with the Jump Flooding Algorithm: up to 1024
  seeds on a 1024×768 grid, a seed pass plus log2(N) flood passes (optional
  JFA+1), Euclidean / Manhattan / Chebyshev metrics, cell / distance-field /
  combined views, click to add, drag to move and right-click to remove seeds,
  Random / Jittered grid / Sunflower / Clusters layouts, an Animate mode with
  drifting seeds, a GPU brute-force exact check, and an FPS readout.
- Example 18, maze generation and solving: recursive backtracker, Kruskal's and
  Prim's mazes (63×47 up to 1023×767 cells) generated on the CPU with a Loops
  slider that adds alternative routes, solved on the GPU by a breadth-first
  wavefront (one dispatch per distance level, 1-512 per frame), with the flood
  colored by distance, the shortest path traced back from the goal, click /
  shift-click to move start and goal, and generation time, BFS levels, cells
  visited, path length and FPS readouts.

## [0.2.0] - 2026-10-30

### Added
- Example 10, Gray-Scott reaction-diffusion: a 512×384 wrap-around grid with
  Coral, Mitosis, Maze, Holes, Chaos and Cells presets, feed/kill and
  steps-per-frame sliders, three color ramps, mouse painting, and an FPS readout.
  The WASM Mandelbrot example is now number 11.
- Example 11, wave equation: water ripples on a 512×384 grid from a damped
  leapfrog finite-difference solver, with Open water, Double slit and Ellipse
  presets, click-to-drop ripples, shift-drag walls, rain, an absorbing-edge
  toggle, wave speed / damping / steps-per-frame sliders, lit water shading, and
  an FPS readout.
- Example 12, Falling Sand: sand, water, stone, oil, fire, plant and acid on a
  320×240 grid updated with the Margolus neighborhood (2×2 blocks, alternating
  offsets, swap-only movement), mouse painting and erasing, brush size and
  steps-per-frame sliders, a Sandbox scene, and an FPS readout.
- Example 13, procedural noise: a compute shader evaluates 3D Perlin or simplex
  noise (PCG hash, no permutation table) as fBm with 1-8 octaves, lacunarity,
  persistence, ridged mode and domain warping for a 1024×768 field, colored as
  terrain, clouds, marble or grayscale. Terrain, Mountains, Clouds, Marble and
  Warped presets, drag-to-pan, wheel zoom, time animation and an FPS readout.

### Fixed
- Losing the GPU device (driver reset, GPU process crash) left a frozen page. The status
  badge now reports the loss, example controls are disabled, and a Reload button appears.
  Uncaught WebGPU validation errors are logged with a `[WebGPU]` prefix.

## [0.1.0] - 2026-10-30

### Added
- Molecular dynamics shows the periodic simulation box as faint glass faces with
  glowing edges; the default camera now frames the whole box.
- Page redesign: hero header with a WebGPU status badge, sticky example navigation,
  grouped sections, primary/secondary buttons, keyboard focus styles, canvases that
  scale to the window, and result panels hidden until used.
- Search and link-preview metadata (description, Open Graph, Twitter card, canonical
  URL) and a Mandelbulb social preview image (`assets/social-preview.png`).
- Ten WebGPU/WGSL examples: vector addition, matrix multiplication, Gaussian blur,
  Game of Life, molecular dynamics, SPH fluid, ray-marched 3D fractals, boids,
  3D diffusion-limited aggregation, and a Rust/WASM-coordinated Mandelbrot set.
- GitHub Actions CI (JavaScript syntax check, WASM build) and GitHub Pages deployment.
- `CHANGELOG.md`, `CLAUDE.md`, `CITATION.cff`.

### Changed
- Version set to 0.1.0; filled in `package.json` and `Cargo.toml` metadata.
- README: current browser support, MIT license section, contributor guidance.
- Future example ideas moved from `FUTURE_EXAMPLES.md` to GitHub issues.
- `wasm/build.sh` is executable, runs from any directory, and reports a missing
  `wasm32-unknown-unknown` target clearly.

### Fixed
- Page text claimed features that don't exist (SPH surface tension and click-to-spawn
  barriers, boids spatial hashing); README now matches the code.
- Game of Life leaked a staging buffer on every cell click.
- Image blur looked like a no-op: the smooth test image hid a 5×5 blur. Now a
  sharp-edged test pattern, σ = 4 over a 25×25 kernel, and Original/Blurred captions.
- `readBuffer` no longer leaks its staging buffer and accepts the typed-array
  type to return; `createBuffer` handles any typed array.
- Integer (`u32`) uniform fields in boids, molecular dynamics, SPH, DLA and ray
  marching were written as floats, so counts like `numBoids = 2000` reached the GPU
  as ~1.1 billion and loops ran for billions of iterations, hanging the GPU.
- Boids uniform buffer was misaligned from `boundsMin` onward.
- WGSL compile errors in molecular dynamics, SPH, ray marching and DLA
  (duplicate `main` entry points, reserved words `target`/`active`, assignment to
  a `let`, malformed ray-marching fragment shader).
- Molecular dynamics and DLA drew nothing: the 4x4 matrix multiply used row-major
  indexing on column-major matrices, so the view-projection was inverted. Projection
  now uses WebGPU's [0, 1] depth range; molecular dynamics uses a real icosahedron mesh.
- SPH used 3D kernels in 2D with inconsistent units (density ~1e-3 vs rest density 1000),
  divided gravity by density, and lost every other integration step in its buffer
  ping-pong. Now 2D kernels, normalized mass, 6 stable substeps per frame, and a working
  Spawn button. Parameters were checked against a CPU replica across the slider ranges.
- Ray marching was almost black: its ambient-occlusion term went negative with camera
  distance. Fixed AO, Menger sponge (floored modulo), Mandelbox scale, image orientation,
  and added a background gradient.
- Boids mouse interaction used CSS pixels instead of canvas pixels, had attract/repel
  reversed, and was too weak to notice.
- DLA walkers moved in straight lines (same random seed for every step within a frame).
  Each walker now carries its own PCG random state.
- Interactive examples are created when scrolled into view instead of all at page
  load; boids no longer auto-starts; ray marching renders only while visible;
  repeated Start clicks no longer stack extra render loops.

[Unreleased]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/releases/tag/v0.1.0
