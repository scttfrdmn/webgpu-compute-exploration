## Overview

A path tracer computes a photograph of a virtual scene by following rays of light backwards, from the
camera into the scene, bouncing them off surfaces at random until they reach a light or leave. Each
traced path is one random sample of the light arriving at a pixel; the average of many samples converges
to the exact image. This example traces **one path per pixel per frame** on the GPU and keeps a running
sum, so the picture starts noisy and sharpens as samples arrive, until it stops by itself at the target
sample count.

For a computational scientist this is Monte Carlo integration in its purest form: an integral over the
space of all light paths, an unbiased estimator, and noise that falls as $1/\sqrt{N}$, which you can
check against an exact answer in the [Measure](#path-tracer/measure) tab. For a GPU programmer it is the
textbook **divergent** workload: neighbouring pixels take paths of different lengths through different
materials, so threads in a SIMD group stop agreeing on what to do next.

> **Note:** The scenes are small (at most 64 spheres, quads, boxes and planes), so every ray is tested
> against every primitive. Production renderers use a bounding volume hierarchy instead.

## The model

### The rendering equation

The radiance leaving a point $\mathbf{x}$ in direction $\omega_o$ is what the surface emits plus what it
reflects of all the light arriving from the hemisphere $\Omega$ above it (Kajiya 1986):

$$
L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o)\,
L_i(\mathbf{x}, \omega_i)\, \cos\theta_i \, d\omega_i
$$

The incoming radiance $L_i$ is itself the outgoing radiance of another point, so the equation is
recursive. Unrolling it gives an integral over paths of every length.

### The Monte Carlo estimator

An integral $I = \int f(x)\,dx$ is estimated by drawing samples $X_k$ from a density $p$ and averaging

$$
\langle I \rangle_N = \frac{1}{N} \sum_{k=1}^{N} \frac{f(X_k)}{p(X_k)}, \qquad
\mathrm{E}\left[\langle I \rangle_N\right] = I, \qquad
\sigma\left(\langle I \rangle_N\right) = \frac{\sigma_1}{\sqrt{N}}
$$

The estimate is **unbiased** for any $p$ that is non-zero where $f$ is, and its noise falls as
$1/\sqrt{N}$. A better $p$, one closer in shape to $f$, lowers the constant $\sigma_1$. For a diffuse
surface of albedo $\rho$, $f_r = \rho/\pi$, and the shader samples the bounce direction with the cosine
density $p(\omega) = \cos\theta/\pi$ (Malley's method), so the weight of a bounce is exactly
$f_r \cos\theta / p = \rho$.

### Next-event estimation and multiple importance sampling

A small light is hard to hit by bouncing at random. At every diffuse bounce the shader therefore also
sends a **shadow ray** to a point on one light chosen at random (next-event estimation, NEE): uniformly
over the area of a quad light, or over the cone a sphere light subtends. The same light can now be
reached in two ways, by the shadow ray (density $p_L$) and by a bounce that happens to hit it ($p_B$).
**Multiple importance sampling** (Veach and Guibas 1995) counts each with the power heuristic weight

$$
w_L = \frac{p_L^2}{p_L^2 + p_B^2}, \qquad w_B = \frac{p_B^2}{p_L^2 + p_B^2}, \qquad w_L + w_B = 1
$$

so nothing is counted twice, and each technique dominates where it is the better one: light sampling for
small lights, BSDF sampling for large ones and for glossy reflection.

### Russian roulette, metal and glass

A path could bounce forever. After the third bounce, Russian roulette continues it with probability
$q = \min(\max(\text{throughput}), 0.95)$ (at least 0.05) and divides the surviving throughput by $q$:
the expected contribution is unchanged, so the estimator stays unbiased, and dark paths end early. Metal
reflects around the mirror direction, perturbed by a random point in a ball scaled by the roughness.
Glass chooses between reflection and refraction with Schlick's approximation of the Fresnel factor, so
no weight is needed.

## Mapping to the GPU

### One thread per pixel, one dispatch per sample

`cs_trace` runs in 8 × 8 workgroups, one thread per pixel. Each thread seeds a **PCG** random number
generator from its pixel index and the sample number, traces one complete path (up to 8 bounces, at most
17 rays including shadow rays), and adds its radiance to its own `vec4<f32>` of the accumulation buffer
(the fourth component counts samples). Because each pixel only touches its own entry, there are no races
and no atomics on the image.

| Buffer | Contents |
|---|---|
| `accum` | `vec4<f32>` per pixel: radiance sum and sample count (up to 800 × 600) |
| `prims` | 96 bytes per primitive (six `vec4`): geometry, albedo, emission, shape and material; lights first |
| `params` | 160-byte uniform: size, sample count, bounces, camera basis, sky, seed, NEE flag |
| `rayCounter` | one `atomic<u32>`: rays traced in the last frame |

A full-screen fragment pass then divides by the count, applies exposure, ACES or Reinhard tone mapping
and the sRGB curve. Changing the camera or the scene clears the sum; exposure and tone mapping do not.

### Bounded work per frame

A long GPU job can hang a machine, so the work per submit is capped: one sample per pixel per dispatch,
one dispatch per frame, at most two frames queued, at most 8 bounces and 64 primitives, every loop with a
constant bound. The worst case is 480,000 pixels × 17 rays × 64 primitives, about $5 \times 10^{8}$
ray-primitive tests per frame; the default Cornell box needs about 3.7 rays per pixel.

> **GPU:** the ray statistics use a two-level reduction: each thread adds its ray count to a
> `var<workgroup>` atomic, then one thread per workgroup adds that sum to the global counter. That is 64
> times fewer global atomics, and only 4 bytes are read back.

Bad samples (NaN or infinite radiance from a degenerate hit) are dropped before they reach the sum,
because a single NaN would poison the pixel forever. An optional firefly clamp limits each sample's
brightness: it converges faster to a slightly wrong (biased) image.

## Numerical behaviour

> **Numerics:** the image is unbiased: its expectation is the exact solution of the rendering equation
> for this scene. The error is pure noise, and it shrinks as $1/\sqrt{N}$. Halving it costs four times as
> many samples.

**The furnace test.** A closed box whose six walls all emit $L_e$ and reflect with albedo $\rho$ has a
known answer: every path vertex sees $L_e$, so a path of up to $B$ bounces has expected radiance
$L_e \sum_{k=0}^{B} \rho^k$ in every pixel. In CI, on a software GPU, with $\rho = 0.5$ and 5 bounces
the image mean is 1.967 ± 0.002 against the exact 1.96875, with or without NEE. For the noise, the
[Measure](#path-tracer/measure) experiment at 100 × 75 pixels gives:

| samples per pixel | 1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| relative RMSE, NEE + MIS | 0.29 | 0.15 | 0.073 | 0.036 | 0.018 |
| relative RMSE, BSDF only | 0.33 | 0.17 | 0.083 | 0.041 | 0.021 |

The fitted log-log slope is −0.500 in both cases: exactly $1/\sqrt{N}$.

**A perfect sampler.** With 2 bounces and NEE off, the furnace image has *no noise at all*: every bounce
has weight exactly $\rho$, every wall emits the same $L_e$ and Russian roulette has not started yet, so
every path returns $1 + \rho + \rho^2 = 1.75$. When the sampling density matches the integrand exactly,
the variance is zero. Turning NEE on adds noise in this scene (relative RMSE 0.05 at 4 spp), because the
MIS split between six equally bright lights is no longer exact.

**Fireflies.** In the Cornell box the relative RMSE at 1 spp is 4.2 against a 2,048-spp reference:
caustics through the glass sphere are reached only by rare paths that carry a lot of energy, and NEE
cannot help (a shadow ray cannot pass through glass). The fitted slope is −0.47 rather than −0.5,
because the reference has noise of its own and the rare paths take time to average out.

**Precision.** Radiance sums are `f32`. At 4,096 samples a sum near 4,000 still has about 7 significant
digits, far below the Monte Carlo noise. Ray offsets of $10^{-3}$ along the normal avoid self-hits.

## Why it maps well — or poorly — to a GPU

**Well:** pixels are independent, so there is no communication at all between threads, and the scene is
tiny and read-only. Testing every ray against every primitive in the same order makes the scene loads
uniform across a SIMD group: every lane reads the same primitive at the same time, which the hardware can
broadcast from cache.

**Poorly:** everything after the first hit diverges. One lane hits glass and refracts, its neighbour hits
a diffuse wall and sends a shadow ray, a third escapes to the sky and is finished. The SIMD group runs as
long as its slowest path, with lanes masked off, and Russian roulette makes path lengths even more
uneven. Production GPU renderers fight this with wavefront path tracing, which sorts rays by what they
need to do next (Laine et al. 2013), and with hardware ray-tracing units for the traversal.

> **GPU:** in the [bounces experiment](#path-tracer/measure) the rays per pixel in the Cornell box rise
> from 2.2 at 1 bounce to 3.8 at 8 (in CI). On a GPU the time per sample follows the *longest* path in
> each SIMD group rather than the average, so compare the time column with the ray column on yours.

## Try this

> **Try this:** start the [Cornell box](#path-tracer) and watch the noise fade, then turn **Light
> sampling** off. Small lights and the walls they light get far noisier. Then run
> [Samples per pixel vs noise](#path-tracer/measure) on the Cornell box with and without NEE and compare
> the two curves against the $1/\sqrt{N}$ line.

- Run the same experiment on the **furnace box**: the measured mean should match the exact
  $L_e \sum \rho^k$ to within the noise, a direct check that the estimator is unbiased.
- Run [Bounces vs cost and energy](#path-tracer/measure) on the furnace box. Each bounce adds
  $\rho^k$, half of the previous one, and the rays per pixel level off once Russian roulette starts.
- In [Explore](#path-tracer), load **Caustics** and set a firefly clamp. The image converges faster
  and the caustic under the glass sphere loses its brightest parts: that is the bias.
- Lower the resolution to 50%: four times fewer threads per sample, so samples can arrive up to four
  times faster, and the noise after a given time falls by up to a factor of two.

## References

- J. T. Kajiya, [The rendering equation](https://doi.org/10.1145/15886.15902), *SIGGRAPH* (1986).
- E. Veach and L. J. Guibas, [Optimally combining sampling techniques for Monte Carlo rendering](https://doi.org/10.1145/218380.218498),
  *SIGGRAPH* (1995): multiple importance sampling and the power heuristic.
- E. Veach, [Robust Monte Carlo methods for light transport simulation](https://graphics.stanford.edu/papers/veach_thesis/),
  PhD thesis, Stanford (1997).
- M. Pharr, W. Jakob and G. Humphreys, [Physically Based Rendering](https://pbr-book.org/), 3rd ed. (2016):
  Monte Carlo integration, Russian roulette, light sampling.
- M. E. O'Neill, [PCG: a family of simple fast space-efficient statistically good algorithms for random number generation](https://www.pcg-random.org/paper.html) (2014).
- T. Duff, J. Burgess, P. Christensen, C. Hery, A. Kensler, M. Liani and R. Villemin, [Building an orthonormal basis, revisited](https://jcgt.org/published/0006/01/01/),
  *JCGT* 6(1) (2017).
- C. Schlick, [An inexpensive BRDF model for physically-based rendering](https://doi.org/10.1111/1467-8659.1330233),
  *Computer Graphics Forum* 13, 233 (1994).
- S. Laine, T. Karras and T. Aila, [Megakernels considered harmful: wavefront path tracing on GPUs](https://doi.org/10.1145/2492045.2492060),
  High-Performance Graphics (2013).

## Read the code

- [`js/examples/path-tracer.js`](js/examples/path-tracer.js) has the trace shader (`tracePath`,
  `sampleLight`, `powerHeuristic`, the intersection routines) and the display shader; the file header
  has the uniform and scene layouts and the workload bounds.
- `measurePathTracer` and `furnaceScene`, at the end of the same file, are the compute-only path the
  [Measure](#path-tracer/measure) tab uses; the experiments are in
  [`js/teach/experiments/path-tracer.js`](js/teach/experiments/path-tracer.js), and the exact furnace
  radiance and the image error in [`js/refs/path-tracer.js`](js/refs/path-tracer.js).
- [`test/path-tracer.test.mjs`](test/path-tracer.test.mjs) checks a CPU replica of the sampling code
  against analytic solid angles and the furnace box.
- [`test/gpu/path-tracer-measure.gpu.test.mjs`](test/gpu/path-tracer-measure.gpu.test.mjs) runs the
  shipped shader on a software GPU in CI; it is the source of the CI numbers on this page.
