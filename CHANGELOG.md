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

[Unreleased]: https://github.com/scttfrdmn/webgpu-compute-exploration/commits/main
