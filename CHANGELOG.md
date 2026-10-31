# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

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

[Unreleased]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/webgpu-compute-exploration/releases/tag/v0.1.0
