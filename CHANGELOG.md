# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

### Added
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
- `readBuffer` no longer leaks its staging buffer and accepts the typed-array
  type to return; `createBuffer` handles any typed array.
- Integer (`u32`) uniform fields in boids, molecular dynamics, SPH, DLA and ray
  marching were written as floats, so counts like `numBoids = 2000` reached the GPU
  as ~1.1 billion and loops ran for billions of iterations, hanging the GPU.
- Boids uniform buffer was misaligned from `boundsMin` onward.
- WGSL compile errors in molecular dynamics, SPH, ray marching and DLA
  (duplicate `main` entry points, reserved words `target`/`active`, assignment to
  a `let`, malformed ray-marching fragment shader).
- Interactive examples are created when scrolled into view instead of all at page
  load; boids no longer auto-starts; ray marching renders only while visible;
  repeated Start clicks no longer stack extra render loops.

[Unreleased]: https://github.com/scttfrdmn/webgpu-compute-exploration/commits/main
