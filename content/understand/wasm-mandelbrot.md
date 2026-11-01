## Overview

The Mandelbrot set is the set of complex numbers $c$ for which a very simple iteration stays bounded.
Drawing it means running that iteration for every pixel, which is the definition of an embarrassingly
parallel problem. This example splits the work between three parties: **Rust compiled to WebAssembly**
holds the view configuration and turns iteration counts into colours, a **WebGPU compute shader** runs
the iteration for all 480,000 pixels, and JavaScript moves the data between them.

For a GPU programmer, it is the smallest complete example of the integration pattern: CPU-side code in a
compiled language prepares a uniform buffer, the GPU does the arithmetic, and the result comes back for
the CPU to use. For a computational scientist, it is a clean case of **finite precision meeting chaos**:
near the boundary of the set, single and double precision disagree, and at deep zoom single precision
cannot even tell neighbouring pixels apart. The [Measure](#wasm-mandelbrot/measure) tab measures both.

## The model

For each pixel the shader maps its position to a point $c = c_x + i\,c_y$ of the complex plane and
iterates

$$
z_0 = 0, \qquad z_{n+1} = z_n^2 + c, \qquad
\text{that is} \quad x_{n+1} = x_n^2 - y_n^2 + c_x, \quad y_{n+1} = 2\,x_n y_n + c_y
$$

until $|z_n|^2 \ge 4$ or the iteration limit (256) is reached. Once $|z_n| > 2$ the orbit is guaranteed
to diverge, because then $|z_{n+1}| \ge |z_n|^2 - |c| > |z_n|$ for every $c$ in the set's neighbourhood
($|c| \le 2$). The pixel's value is the number of iterations before escape; points that never escape are
taken to be in the set and drawn black. The view is centred at $-0.5$ with a scale of
$4 / (\text{zoom} \cdot \min(\text{width}, \text{height}))$ per pixel.

## Mapping to the GPU

### Who does what

1. **WASM** creates a `MandelbrotConfig` (size, iteration limit, zoom, centre) and returns its six
   uniform values with `get_shader_params()`, in the order of the WGSL `Params` struct (all `f32`).
2. **JavaScript** writes them into a 24-byte uniform buffer and dispatches the compute shader with
   8 × 8 workgroups, one thread per pixel: $100 \times 75$ workgroups for 800 × 600.
3. Each thread iterates its pixel and writes the count, as an `f32`, to a storage buffer.
4. JavaScript copies the 1.9 MB of counts to a mappable buffer and reads them back.
5. **WASM** maps each count to a colour (a hue that cycles with the count) and draws the image into
   the canvas with `putImageData`.

> **GPU:** the loop `for (var i = 0u; i < max_iter; i++) { if (|z|² ≥ 4) { break; } … }` has a
> uniform bound but a per-thread exit, so neighbouring threads in a SIMD group leave it at different
> times. The group keeps running until its slowest pixel is done.

### Why WASM coordinates and WebGPU computes

The two are good at different things. The per-pixel iteration is uniform floating-point work over
hundreds of thousands of independent items, which a GPU does far faster than one CPU core. The rest
(configuration, validation, colour mapping, talking to the DOM) is sequential logic with branches and
small data, which suits a CPU, and Rust compiled to WASM runs it at near-native speed with memory safety.
WASM has no direct access to WebGPU here, so JavaScript is the glue: it owns the device and the buffers.

The split has a cost: the counts make a round trip from GPU memory to the CPU (1.9 MB) so that WASM can
colour them, and then go to the canvas. A pure-GPU version would colour in a fragment shader and never
read back. The [Measure](#wasm-mandelbrot/measure) tab times exactly this: the GPU kernel, the GPU with
its readback, the Rust function `compute_mandelbrot_cpu` and its JavaScript transcription.

## Numerical behaviour

> **Numerics:** the shader works in f32 (24-bit significand, about 7 digits); the reference, a
> transcription of the Rust `compute_mandelbrot_cpu`, in float64. Inside the set and far outside it the
> counts agree exactly. Near the boundary the orbit is chaotic, so a rounding difference in the last
> digit grows roughly geometrically, by a factor $|2 z_n|$ per iteration, until the two orbits escape at
> different times.

In CI, on a software GPU, 0.085% of the pixels of the 640 × 480 default view differ from float64, and
every differing pixel has an iteration count of at least 38: pixels that escape quickly always agree.

**Deep zoom.** The spacing between neighbouring f32 numbers near 0.74 is $2^{-24} \approx 6 \times
10^{-8}$. The pixel spacing is $4 / (\text{zoom} \cdot \text{height})$, so once the zoom passes about
$10^{5}$ (for a 192-pixel-high view) adjacent pixels round to the same $c$ and the image turns into blocks.
CI measures, at the Seahorse valley ($-0.7436 + 0.1318\,i$), 256 × 192 pixels, 1,024 iterations:

| zoom | 1 | 10³ | 10⁵ | 10⁶ | 10⁷ |
|---|---|---|---|---|---|
| pixels ≠ float64 | 0.13% | 9.2% | 40% | 57% | 20% |
| distinct f32 $c_x$ per row (of 256) | 256 | 256 | 256 | 90 | 9 |

(At $10^7$ fewer pixels differ only because the float64 view is nearly uniform there too.) Deep-zoom
renderers use double precision, or perturbation theory: one high-precision reference orbit plus
low-precision differences for every other pixel.

The counts are stored as `f32`, which represents every integer up to $2^{24}$ exactly, far above the
iteration limit.

## Why it maps well — or poorly — to a GPU

**Well:** every pixel is independent, needs no memory but its own output, and does pure arithmetic in
registers. The workload is the classic GPU demonstration for that reason.

**Poorly:** the work per pixel varies by a factor of up to 256 between a pixel that escapes at once and
one inside the set, so SIMD groups on the boundary of the set waste lanes. The single dispatch is also
short, so for small images the fixed cost of a submit and a 1.9 MB readback is larger than the
computation, and the CPU can win: the [Measure](#wasm-mandelbrot/measure) experiment shows where.

## Try this

> **Try this:** press **Run WASM + WebGPU** in [Explore](#wasm-mandelbrot) and read the timing
> breakdown: GPU computation (with readback) against WASM rendering. Then run
> [Resolution vs time](#wasm-mandelbrot/measure) and find the size at which the GPU starts to win.

- Run the same experiment at 64 and at 1,024 iterations: the CPU time grows with the iteration count,
  the GPU time much less.
- Run [Zoom vs agreement with float64](#wasm-mandelbrot/measure) and look at the distinct-values column:
  that is f32 running out of digits.
- If the WASM column is empty, build the package with `wasm/build.sh` and reload the page.

## References

- B. B. Mandelbrot, [Fractal aspects of the iteration of z → λz(1 − z) for complex λ and z](https://doi.org/10.1111/j.1749-6632.1980.tb29690.x),
  *Ann. N. Y. Acad. Sci.* 357, 249 (1980).
- A. Douady and J. H. Hubbard, *Étude dynamique des polynômes complexes*, Publications Mathématiques
  d'Orsay (1984–85): the escape radius 2 and the connectedness of the set.
- The Rust and WebAssembly working group, [The Rust and WebAssembly book](https://rustwasm.github.io/docs/book/):
  wasm-bindgen and wasm-pack, used to build `js/wasm-pkg/`.
- K. I. Martin, *Superfractalthing maths* (2013): perturbation theory for deep zooms.

## Read the code

- [`wasm/src/lib.rs`](wasm/src/lib.rs) is the Rust side: `MandelbrotConfig`, `get_shader_params`,
  `render_to_canvas` and the CPU reference `compute_mandelbrot_cpu`.
- [`js/examples/wasm-mandelbrot.js`](js/examples/wasm-mandelbrot.js) has the WGSL shader and the glue;
  `measureMandelbrot` and `loadWasmModule`, at its end, are what the [Measure](#wasm-mandelbrot/measure)
  tab uses, and [`js/teach/experiments/wasm-mandelbrot.js`](js/teach/experiments/wasm-mandelbrot.js) has
  the experiments.
- [`js/refs/mandelbrot.js`](js/refs/mandelbrot.js) is the JavaScript transcription of the Rust
  reference; [`test/wasm-mandelbrot.test.mjs`](test/wasm-mandelbrot.test.mjs) checks a CPU replica of
  the shader and the uniform order against it.
- [`test/gpu/wasm-mandelbrot-measure.gpu.test.mjs`](test/gpu/wasm-mandelbrot-measure.gpu.test.mjs) runs
  the shipped shader on a software GPU in CI. It covers the WebGPU kernel only: the WASM package is a
  build artifact that the GPU job does not build.
