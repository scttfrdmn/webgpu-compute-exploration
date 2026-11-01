## Overview

A Gaussian blur replaces every pixel by a weighted average of its neighbourhood, with weights that fall
off as a bell curve. It is the standard smoothing filter of image processing, the first step of edge
detectors and of scale-space analysis, and, mathematically, the solution of the heat equation after a
fixed time. The example blurs a $512 \times 512$ test image with $\sigma = 4$ pixels using a
$25 \times 25$ kernel: 625 weighted samples per pixel, 164 million for the image.

For a GPU programmer this is the canonical **stencil**: every output reads a window of inputs, and
neighbouring outputs read overlapping windows, so the question is how well the memory system serves that
reuse. For a computational scientist it is a **discrete convolution**, with choices that matter:
truncating the kernel, normalizing at the boundary, and the fact that a 2D Gaussian factors into two 1D
ones, which this kernel does *not* exploit.

## The model

The continuous 2D Gaussian with standard deviation $\sigma$ is

$$
G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
= g_\sigma(x)\, g_\sigma(y)
$$

with $g_\sigma$ the 1D Gaussian. The shader samples it at integer offsets $|k_x|, |k_y| \le h$ with
$h = 12 = 3\sigma$, and divides by the sum of the weights it actually used:

$$
\text{out}(x, y) = \frac{\sum_{(k_x, k_y) \in W} G_\sigma(k_x, k_y)\, \text{in}(x + k_x, y + k_y)}{\sum_{(k_x, k_y) \in W} G_\sigma(k_x, k_y)}
$$

where $W$ is the part of the $25 \times 25$ window that lies inside the image. Dividing by the weight
sum does two jobs: it makes the truncated kernel sum to 1 (the tails beyond $3\sigma$ hold about 0.5%
of the 2D mass), and at the edges it averages over the pixels that exist instead of treating the outside
as black, so borders are not darkened. The $1/2\pi\sigma^2$ factor cancels and could be dropped.

Because $G_\sigma(k_x, k_y) = g_\sigma(k_x)\,g_\sigma(k_y)$ and the clipped window $W$ is a rectangle,
the same result comes from a horizontal 1D pass followed by a vertical one, each renormalized. That
**separable** form costs $2(2h + 1) = 50$ taps per pixel instead of $(2h + 1)^2 = 625$. The CPU
reference (`separableBlurPixel`) is computed that way, which makes it an independent check.

## Mapping to the GPU

### Data and threads

| Binding | WGSL | Contents |
|---|---|---|
| 0 | `var<storage, read> input: array<vec4<f32>>` | RGBA pixels, 16 bytes each, row-major |
| 1 | `var<storage, read_write> output: array<vec4<f32>>` | the blurred image |
| 2 | `var<uniform> dimensions: vec2<u32>` | width, height |

The shader uses $8 \times 8$ workgroups and one invocation per output pixel, so the host dispatches
$64 \times 64$ workgroups for $512^2$. $\sigma$ and the kernel size are WGSL constants
(`const sigma = 4.0`, `const kernelSize = 25`), so the compiler knows the loop bounds. Each invocation
runs the double loop over the window, skips taps outside the image, evaluates the weight with an `exp`,
and accumulates a `vec4` sum and a scalar weight sum. One write per pixel ends it; there is no
synchronization at all.

### Memory reuse

Every input pixel is read by all 625 outputs whose window contains it. A workgroup of $8 \times 8$
outputs needs a $32 \times 32$ block of input (8 plus 12 on each side), 1,024 pixels, but issues
$64 \times 625 = 40{,}000$ loads: each pixel is loaded about 39 times. The kernel relies on the caches to
make those repeats cheap.

> **GPU:** the classic optimization stages that $32 \times 32$ block in `var<workgroup>` memory once, with
> a `workgroupBarrier()` before the sums. At 16 bytes per pixel the block is exactly 16 KB, the default
> `maxComputeWorkgroupStorageSize`: a larger $\sigma$ would not fit, which is another reason real
> implementations go separable, and often store pixels as 8-bit or `f16` values.

The weights cost too: each tap computes an `exp` (a transcendental, on a slower unit on most GPUs) of a
value that depends only on $(k_x, k_y)$. Precomputing the 25 one-dimensional weights once would remove
625 `exp` calls per pixel.

### CPU↔GPU traffic

`runImageBlur` builds the test image on the CPU (a checkerboard of 32-pixel squares, thin bright rings
and 1-pixel diagonal lines, chosen because sharp edges show the blur), uploads 4 MB, runs one pass,
reads 4 MB back and draws both images to 2D canvases. Its wall time includes all of that. The Measure
tab's `measureBlur` uploads once and times the dispatch alone.

## Numerical behaviour

> **Numerics:** the weights are positive and the inputs lie in $[0, 1]$, so the sum has no
> cancellation: 625 `f32` additions of positive terms, then one division. The error against a float64
> evaluation stays at the level of `f32` rounding and grows only slowly with the number of terms: in CI,
> $3.3 \times 10^{-7}$ at $\sigma = 1$ (49 terms), $1.4 \times 10^{-6}$ at $\sigma = 4$ and
> $2.8 \times 10^{-6}$ at $\sigma = 8$ (2,401 terms), for values in $[0, 1]$.

**Truncation.** Cutting the kernel at $3\sigma$ drops the tails; renormalizing makes the result an
average with a slightly narrower, truncated Gaussian. The difference from an untruncated blur is small
(the weights at $3\sigma$ are $e^{-4.5} \approx 1\%$ of the center weight) but visible in high-contrast
images as a faint square footprint.

**Sampling.** The weights are point samples of $G_\sigma$ at integer offsets, not integrals over a
pixel. Down to $\sigma \approx 1$ that is very accurate; well below it the samples are too coarse (at
$\sigma = 0.5$ the sampled kernel's variance is 0.21 instead of 0.25), and a different discretization,
such as Lindeberg's discrete Gaussian, is better. The Measure tab's smallest $\sigma$ is 1.

**Frequency response.** The Fourier transform of a Gaussian is a Gaussian, so the filter multiplies
spatial frequency $f$ (cycles per pixel) by $\exp(-2\pi^2\sigma^2 f^2)$. The diagonal lines repeat every
5.7 pixels and are damped by a factor $5 \times 10^{-5}$: they vanish into gray. The rings repeat every
24 pixels and keep about 58% of their contrast, and the 32-pixel squares keep their shape with soft
edges.

## Why it maps well — or poorly — to a GPU

**It maps well.** Every output is independent, the loop bounds are compile-time constants, and the only
branch, the edge test, is the same for every thread except near the border. Neighbouring threads read
neighbouring pixels, so the loads coalesce and the caches see heavy reuse.

**It does far more work than necessary.** The direct 2D loop costs $(6\sigma + 1)^2$ taps per pixel and
grows with $\sigma^2$: 49 taps at $\sigma = 1$, 2,401 at $\sigma = 8$. The separable version costs
$2(6\sigma + 1)$ and grows with $\sigma$; at $\sigma = 4$ it is 12.5× less work, at $\sigma = 8$ about 25×.
For large $\sigma$ an FFT convolution, or a few passes of a box filter, costs the same for any radius.
The direct kernel is the right first version: it is obviously correct, and it is the reference the
faster ones are tested against.

## Try this

> **Try this:** run [Explore](#gaussian-blur) and look at the diagonal lines in the top-left quadrant:
> they are 5.7 pixels apart and become a smooth gray, while the rings fade and the 32-pixel squares keep
> their structure.

- Run [blur radius vs cost](#gaussian-blur/measure) and check that the time grows with the number of taps
  (the reference line), not with $\sigma$. What would the separable version cost at each $\sigma$?
- In the same experiment, the error column should stay between $10^{-7}$ and a few $10^{-6}$: fifty
  times more positive terms cost less than a factor ten in accuracy. (In CI, on a CPU-emulated GPU, the
  time grew 32× from $\sigma = 1$ to $\sigma = 8$ for 49× the taps.)
- Run [image size vs throughput](#gaussian-blur/measure) and find the size where the Mpixel/s level off.
  Below it, the GPU is not full.

## References

- R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson (2018), chapter 3: spatial
  filtering and Gaussian kernels.
- T. Lindeberg, [Scale-space for discrete signals](https://doi.org/10.1109/34.49051), *IEEE Trans. Pattern
  Anal. Mach. Intell.* 12, 234 (1990): sampled versus discrete Gaussian kernels.
- P. Rákos, [Efficient Gaussian blur with linear sampling](https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/)
  (2010): separable GPU blurs.
- P. S. Heckbert, *Filtering by repeated integration*, SIGGRAPH (1986): blurs whose cost does not depend
  on the radius.

## Read the code

- [`js/examples/image-blur.js`](js/examples/image-blur.js) has the shader, the test image, `runImageBlur`,
  and the compute-only `blurShaderCode` and `measureBlur` used by the Measure tab.
- [`js/teach/experiments/gaussian-blur.js`](js/teach/experiments/gaussian-blur.js) has the two experiments;
  [`js/refs/basic.js`](js/refs/basic.js) has the separable float64 reference.
- [`test/image-blur.test.mjs`](test/image-blur.test.mjs) and [`test/gaussian-blur-measure.test.mjs`](test/gaussian-blur-measure.test.mjs)
  check the JavaScript and a CPU replica of the direct sum; [`test/gpu/gaussian-blur-measure.gpu.test.mjs`](test/gpu/gaussian-blur-measure.gpu.test.mjs)
  runs the shipped shader on a software GPU in CI.
