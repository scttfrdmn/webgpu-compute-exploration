# WebGPU Compute Exploration

[![CI](https://github.com/scttfrdmn/webgpu-compute-exploration/actions/workflows/ci.yml/badge.svg)](https://github.com/scttfrdmn/webgpu-compute-exploration/actions/workflows/ci.yml)

**Live demo:** https://scttfrdmn.github.io/webgpu-compute-exploration/

A comprehensive exploration of WebGPU's computational capabilities, featuring pure JavaScript examples and WASM integration for high-performance computing in the browser.

## Features

### 1. Vector Addition
Demonstrates basic GPU compute with parallel vector addition on 1 million elements.
- **Shader**: WGSL compute shader with workgroup size optimization
- **Use Case**: Foundation for understanding GPU parallelism
- **Performance**: Processes millions of elements per millisecond

### 2. Matrix Multiplication
Implements efficient matrix multiplication using 2D workgroups.
- **Dimensions**: 512×512 matrices
- **Shader**: Optimized compute shader with proper indexing
- **Use Case**: Neural networks, 3D graphics, scientific computing
- **Performance**: GFLOPS measurements included

### 3. Image Processing (Gaussian Blur)
Real-time image filtering using compute shaders.
- **Algorithm**: Gaussian blur with configurable kernel
- **Shader**: 2D convolution with boundary handling
- **Use Case**: Video filters, anti-aliasing, depth of field
- **Visualization**: Side-by-side original and processed images

### 4. Conway's Game of Life
Interactive cellular automaton with GPU acceleration.
- **Grid Size**: 200×150 cells (30,000 cells)
- **Interaction**: Click to toggle cells, load classic patterns
- **Patterns**: Glider, Pulsar, Gosper Glider Gun, R-Pentomino, and more
- **Performance**: Real-time simulation at 60 FPS with zero CPU overhead
- **Features**: Play/pause, step-through, adjustable speed

### 5. Molecular Dynamics Simulation
Real-time 3D physics simulation of interacting particles.
- **Particles**: 100-5000 atoms with Lennard-Jones potential
- **Physics**: Velocity Verlet integration, periodic boundaries
- **Rendering**: Instanced 3D spheres with Phong shading
- **Interaction**: Mouse drag to rotate, wheel to zoom, right-drag to pan
- **Visualization**: Temperature-based coloring (blue=cold, red=hot)
- **Performance**: 60 FPS with up to 5000 particles, O(N²) force calculations on GPU

### 6. SPH Fluid Simulation
Smoothed Particle Hydrodynamics for realistic fluid dynamics.
- **Particles**: 2000 fluid particles
- **Physics**: SPH kernels (Poly6, Spiky, Viscosity), pressure and viscosity forces
- **Features**: Adjustable gravity and viscosity, interactive spawning
- **Algorithm**: 3-pass compute (density, forces, integration)
- **Visualization**: Velocity-based coloring with alpha blending
- **Performance**: Real-time at 60 FPS, O(N²) neighbor search

### 7. Ray Marching: 3D Fractals
Volumetric rendering of complex 3D fractals using distance fields.
- **Fractals**: Mandelbulb, Julia Set, Menger Sponge, Mandelbox
- **Technique**: Sphere tracing with distance estimators
- **Quality**: Adjustable steps (32-256) for performance/quality trade-off
- **Rendering**: Phong lighting, ambient occlusion, fog
- **Interaction**: Drag to rotate, wheel to zoom, auto-rotation
- **Performance**: Real-time fragment shader ray marching

### 8. Boids Flocking Simulation
Craig Reynolds' emergent flocking behavior with thousands of agents.
- **Agents**: 100-10,000 boids with adjustable speed
- **Behaviors**: Separation, alignment, cohesion + special modes
- **Modes**: Normal flocking, predator/prey, scatter, vortex
- **Interaction**: Click to attract, right-click to repel
- **Visualization**: Triangle agents colored by speed
- **Performance**: 60 FPS with 10K agents, fully GPU-parallelized

### 9. Diffusion Limited Aggregation (3D)
Fractal growth simulation creating dendritic crystal structures.
- **Grid**: 64³ voxel space (262,144 cells)
- **Particles**: 5,000 walkers performing random walks
- **Algorithm**: Particles stick when adjacent to aggregate
- **Seeds**: Single center, line, circle, or multiple points
- **Rendering**: Instanced cubes with gradient coloring
- **Interaction**: Drag to rotate, wheel to zoom, adjustable stickiness

### 10. WASM Integration (Mandelbrot Set)
Demonstrates Rust/WASM coordinating WebGPU compute shaders.
- **Architecture**: Rust handles configuration, WebGPU performs computation
- **Rendering**: 800×600 fractal with HSV color mapping
- **Use Case**: Complex applications requiring both CPU and GPU
- **Performance**: Full pipeline benchmarking

## Browser Support

WebGPU ships by default in (as of late 2026):
- **Chrome / Edge 113+** on Windows, macOS and ChromeOS; Android 121+. On Linux it is behind `chrome://flags/#enable-unsafe-webgpu` for most hardware.
- **Firefox 141+** on Windows and **147+** on Apple silicon Macs. Linux, Android and Intel Macs are Nightly-only.
- **Safari 26+** on macOS, iOS, iPadOS and visionOS.

Check support at: https://caniuse.com/webgpu

## Setup

### Quick Start (JavaScript Only)

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000 in a WebGPU-compatible browser.

### Full Setup (with WASM)

1. **Install Rust** via rustup (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. **Install wasm-pack**:
```bash
cargo install wasm-pack
```

3. **Build WASM module**:
```bash
./wasm/build.sh   # or: npm run build-wasm
```

   `build.sh` needs a `rustc` with the `wasm32-unknown-unknown` target. With rustup: `rustup target add wasm32-unknown-unknown`. If a Homebrew `rustc` shadows rustup's, put `~/.cargo/bin` first on `PATH`.

4. **Start development server**:
```bash
python3 -m http.server 8000
```

5. Open http://localhost:8000

## Project Structure

```
webgpu/
├── .github/workflows/                 # CI and GitHub Pages deployment
├── index.html                          # Main application page
├── js/
│   ├── main.js                        # Application entry point
│   ├── webgpu-utils.js                # WebGPU helper functions
│   ├── examples/
│   │   ├── vector-addition.js         # Example 1: Vector addition
│   │   ├── matrix-multiplication.js   # Example 2: Matrix multiply
│   │   ├── image-blur.js              # Example 3: Image processing
│   │   ├── game-of-life.js            # Example 4: Conway's Game of Life
│   │   ├── molecular-dynamics.js      # Example 5: Molecular dynamics
│   │   ├── sph-fluid.js               # Example 6: SPH fluid simulation
│   │   ├── ray-marching.js            # Example 7: Ray marching fractals
│   │   ├── boids.js                   # Example 8: Boids flocking
│   │   ├── dla-3d.js                  # Example 9: 3D DLA
│   │   └── wasm-mandelbrot.js         # Example 10: WASM integration
│   └── wasm-pkg/                      # Built WASM module (generated)
├── wasm/
│   ├── Cargo.toml                     # Rust dependencies
│   ├── build.sh                       # WASM build script
│   └── src/
│       └── lib.rs                     # Rust/WASM source code
├── CHANGELOG.md
├── CITATION.cff
└── README.md
```

## Technical Details

### WebGPU Compute Shaders

All examples use WGSL (WebGPU Shading Language), the modern shader language designed specifically for WebGPU:

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Your compute code here
}
```

### Key Concepts

**Workgroups**: GPUs process data in parallel workgroups. Each workgroup contains multiple threads (invocations).

**Storage Buffers**: Used for large data that the shader reads/writes.

**Uniform Buffers**: Used for small configuration data that's constant across invocations.

**Pipeline**: Defines how shaders and resources are bound together.

### WASM Integration Pattern

The WASM example demonstrates a powerful pattern:

1. **Rust/WASM**: Manages application state, configuration, and complex CPU logic
2. **WebGPU**: Performs compute-intensive parallel operations
3. **Canvas API**: Renders results via WASM

This architecture is ideal for:
- Game engines (logic in WASM, rendering in WebGPU)
- Scientific computing (algorithms in Rust, computation on GPU)
- ML inference (model management in WASM, matrix ops on GPU)

## Performance Tips

1. **Buffer Management**: Reuse buffers when possible to avoid allocation overhead
2. **Workgroup Size**: Optimize for your GPU (typically 64, 128, or 256)
3. **Memory Layout**: Use contiguous memory for better cache performance
4. **Async Operations**: WebGPU operations are asynchronous - use promises properly
5. **Data Transfer**: Minimize CPU↔GPU transfers, keep data on GPU when possible

## Learning Resources

- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [WGSL Specification](https://gpuweb.github.io/gpuweb/wgsl/)
- [WebGPU Samples](https://webgpu.github.io/webgpu-samples/)
- [GPU Gems (Advanced Techniques)](https://developer.nvidia.com/gpugems/gpugems/contributors)
- [Shadertoy](https://www.shadertoy.com/) — a large body of shader techniques
- Books: *Real-Time Rendering* (Akenine-Möller et al.), *Physically Based Rendering* (Pharr, Jakob, Humphreys)
- Papers: "Position Based Dynamics" (Müller et al.), "Particle-Based Fluid Simulation for Interactive Applications" (Müller et al.), "Fast Parallel GPU-Sorting Using a Hybrid Algorithm" (Sintorn & Assarsson)

## Extending This Project

Planned examples (reaction-diffusion, wave equation, falling sand, noise, FFT, sorting, k-means, Voronoi, maze solving, N-body, cloth, MNIST inference, convolution reverb, path tracing) are tracked as [issues](https://github.com/scttfrdmn/webgpu-compute-exploration/issues?q=is%3Aissue+label%3A%22type%3A+feature%22), grouped into milestones and on the [project board](https://github.com/users/scttfrdmn/projects/85).

## Support

If you find this project helpful, consider supporting my work:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/scttfrdmn)

## Troubleshooting

### WebGPU Not Detected
- Ensure you're using a compatible browser version
- Check chrome://gpu/ (Chrome/Edge) or about:support (Firefox)
- Update your graphics drivers

### WASM Module Fails to Load
- Run the build script: `./wasm/build.sh`
- Check that `js/wasm-pkg/` directory exists
- Ensure wasm-pack installed correctly

### Performance Issues
- Check GPU utilization in browser DevTools
- Reduce workgroup size if crashes occur
- Monitor buffer sizes and memory usage

## License

[MIT](LICENSE) © 2025-2026 Scott Friedman

## Contributing

This is an exploration project. If you add new examples or improvements:
1. Keep examples focused and well-documented
2. Include performance metrics; target 60 FPS for interactive simulations (30 FPS minimum for heavy rendering)
3. Offer quality/size controls so the example scales down to weaker and mobile GPUs
4. Provide interactive controls and preset configurations
5. Add shader code comments explaining the algorithm
6. Update this README and CHANGELOG with your additions
