## Overview

Ray marching renders a scene that is described not by triangles but by a **distance function**: a
function $d(\mathbf{p})$ that tells how far point $\mathbf{p}$ is from the nearest surface, or at least
gives a safe lower bound. A ray can then advance by $d$ at every step without ever passing through a
surface. This **sphere tracing** (Hart, 1996) makes it possible to draw objects that have no mesh at all,
such as 3D fractals, whose distance can be estimated from the iteration that defines them.

The example traces four fractals (the Mandelbulb, a quaternion Julia set, the Menger sponge and the
Mandelbox) in a fragment shader, one ray per pixel of a $1000 \times 700$ canvas, with lighting and ambient
occlusion from extra distance samples. For a GPU programmer it is the textbook case of **divergence**:
neighbouring pixels need very different numbers of steps. For a computational scientist it is
root-finding along a line with a Lipschitz bound, and the fractals are a classic from complex dynamics.

## The model

### Sphere tracing

A ray starts at the camera $\mathbf{o}$ with unit direction $\mathbf{u}$. If $d$ is a distance bound,
the ball of radius $d(\mathbf{p})$ around $\mathbf{p}$ contains no surface, so the iteration

$$
t_0 = 0, \qquad t_{k+1} = t_k + \alpha\, d(\mathbf{o} + t_k \mathbf{u})
$$

never steps past the first intersection when $\alpha \le 1$. The example uses $\alpha = 0.5$. It stops with
a **hit** when $d < 0.001$, an **escape** when $t > 50$, or when it runs **out of steps** (the Quality slider,
32 to 256, default 128). A function with $|d(\mathbf{p}) - d(\mathbf{q})| \le |\mathbf{p} - \mathbf{q}|$
(Lipschitz constant 1) is a valid bound wherever it is an exact distance or underestimates it; with constant
$L$, any $\alpha \le 1/L$ is safe.

### Distance estimators for fractals

For an escape-time fractal $z \mapsto f(z) + c$, the distance to the set's boundary is estimated from the
orbit $z_n$ and its derivative $z'_n$:

$$
d \approx \frac{1}{2} \frac{|z_n| \log |z_n|}{|z'_n|}
$$

(derived from the Douady-Hubbard potential of the set; Hart, Sandin and Kauffman, 1989, applied it to
quaternion Julia sets). The four scenes:

- **Mandelbulb** (power $p$, the Power slider, default 8): $z \mapsto z^p + c$ in spherical coordinates,
  up to 15 iterations, escape at $|z| > 2$, with $|z'|$ tracked as $p|z|^{p-1}|z'| + 1$.
- **Julia set**: the quaternion map $z \mapsto z^2 + c$ with $c = (0.18, 0.88, 0.24, 0.16)$, 12 iterations;
  the estimator is $\tfrac14 \sqrt{|z|^2/|z'|^2} \log |z|^2$, the same formula written with squares.
- **Menger sponge**: an exact construction, not an estimate: the distance to a cube, intersected with three
  carved cross-shaped tunnels at each of 4 levels (Quílez's formulation).
- **Mandelbox**: box and sphere folds with scale $-1.5$, 12 iterations, distance $|p|/|dr|$, drawn at a third
  of its size to fit the view.

The Menger and Mandelbox estimators are 1-Lipschitz (the CPU tests check this on random pairs). The two
escape-time estimators are not, because the iteration count changes abruptly across space; the damping
$\alpha = 0.5$ is what keeps them from tunnelling through thin parts.

### Shading

At a hit, the normal is the central-difference gradient of $d$ with offset 0.001 (six more evaluations),
lighting is Lambert plus a Phong highlight, and **ambient occlusion** compares $d$ at five points along the
normal with their distance from the surface: where $d$ is smaller than expected, something is nearby and the
point is darkened. Fog fades hits toward the background with distance.

## Mapping to the GPU

### A fragment shader, not a compute shader

The example draws a full-screen quad (two triangles from six hard-coded vertices) and does all the work in
`fs_main`, which runs once per pixel. The only input is a 64-byte uniform with the resolution, the time, the
power, `maxSteps` and `fractalType` (both `u32`, written through a `Uint32Array`), and the camera rotation
and zoom. The camera orbits the origin at distance 3 (the wheel zooms between 1.5 and 6); with auto-rotate on,
the time advances 0.016 per frame and the camera turns 0.1 radian per unit of time. The view spans
$\pm 0.71$ horizontally and $\pm 0.5$ vertically in camera units, about 71° by 53°.

Nothing is computed ahead and nothing is read back. Each frame costs up to
$1000 \times 700 \times \text{maxSteps}$ distance evaluations (the readout's figure), plus 11 per hit pixel for
shading, and each Mandelbulb evaluation is up to 15 iterations with `pow`, `acos`, `atan2`, `sin` and `cos`.
The example renders only while its canvas is on screen.

### Divergence

The GPU runs pixels in SIMD groups of 32 or 64 lanes (for fragments, in $2 \times 2$ quads within them), and a
group keeps running until its *slowest* lane finishes. A ray that hits the body of the fractal converges in a
few dozen steps; a ray that grazes the surface creeps along it in tiny steps; a ray that misses must walk to
$t = 50$. When these are neighbours, the fast lanes sit idle. The cost of a frame is therefore set by the
distribution of steps *per group*, not the mean per pixel.

> **GPU:** the Measure tab cannot read anything from a fragment shader but the color, so it adds a compute
> shader (`MARCH_STATS_SHADER`) that is the shipped shader source plus one entry point: the same camera ray
> and the same loop as `rayMarch`, recording per pixel how the loop ended and how many steps it took. The
> frame time comes from the shipped render pipeline itself, drawn into an offscreen texture.

## Numerical behaviour

> **Numerics:** sphere tracing converges linearly when the ray hits a surface head-on, since each step
> removes a fixed fraction of the remaining distance, but slowly at grazing angles, where the distance shrinks
> much faster than the remaining path. Rays that pass close to a surface without hitting it are the most
> expensive of all.

**Step budget.** In CI, the shipped shader on a software GPU, Mandelbulb at $500 \times 350$:

| max steps | 32 | 64 | 96 | 128 | 160 | 192+ |
|---|---|---|---|---|---|---|
| out of steps | 45% | 9.7% | 2.8% | 0.8% | 0.1% | 0% |
| hit | 8.4% | 24% | 29% | 30% | 30.5% | 30.6% |

with 37.8 distance evaluations per pixel on average once everything converges. A pixel that runs out of steps
draws as background, so at low Quality the fractal loses its thin parts first. Above 192 steps a larger
budget changes nothing. At 128 steps and $160 \times 112$, the Julia set still had 4.3% of pixels out of
steps, the Menger sponge 0.1% and the Mandelbox none.

**Threshold.** The hit test $d < 0.001$ is in world units, the same near and far. A cone-tracing renderer
would scale it with $t$ (the size of a pixel at that distance), which saves steps on distant surfaces.

**Precision.** Everything is `f32`. With $t \le 50$ and a threshold of $10^{-3}$, rounding (about $3 \times
10^{-6}$ at $t = 50$) is far below the threshold. In CI, the GPU and a float64 tracer agreed on the outcome
(hit, escape, out of steps) of **every** pixel of a $160 \times 112$ view for all four fractals. The
escape-time estimators themselves are approximations, which is a larger effect than rounding: the CPU tests
compare the traced silhouettes with a brute-force membership test and allow up to 8% of pixels, all on the
edge, to differ.

## Why it maps well — or poorly — to a GPU

**It maps well:** 700,000 independent pixels, no memory traffic beyond a uniform, and arithmetic-heavy inner
loops. This is the kind of work fragment shaders are built for, and why ray-marched fractals run in real time
at all.

**It maps poorly where rays diverge.** The silhouette, where neighbouring rays hit, miss and graze, costs the
most, and the loop count is data-dependent. Common fixes are to march a coarse image first and start each fine
ray from its tile's conservative distance, to cap steps adaptively, or to sort rays by expected cost (more
common in compute-based renderers than in fragment shaders).

## Try this

> **Try this:** in [Explore](#ray-marching), set Quality to 32 on the Mandelbulb. The thin parts disappear and
> the edges turn to background: those pixels ran out of steps.

- Run [step budget vs cost and convergence](#ray-marching/measure) for the Mandelbulb, then the Menger
  sponge. Which one converges with fewer steps, and why does the sponge's exact distance help?
- Set Power to 2 and then 12 and watch the frame rate. Higher powers make a more wrinkled surface, more
  grazing rays and more steps.
- Compare the experiment's mean steps per pixel with Quality: the frame cost tracks the slowest pixels of each
  SIMD group, not the mean.

## References

- J. C. Hart, [Sphere tracing: a geometric method for the antialiased ray tracing of implicit surfaces](https://doi.org/10.1007/s003710050084),
  *The Visual Computer* 12, 527 (1996).
- J. C. Hart, D. J. Sandin and L. H. Kauffman, [Ray tracing deterministic 3-D fractals](https://doi.org/10.1145/74334.74363),
  SIGGRAPH (1989): distance estimates for quaternion Julia sets.
- I. Quílez, [Distance functions](https://iquilezles.org/articles/distfunctions/) and
  [Menger fractal](https://iquilezles.org/articles/menger/): the box SDF and the sponge construction.
- D. White and P. Nylander, [The Mandelbulb](https://www.skytopia.com/project/fractal/mandelbulb.html) (2009),
  and T. Lowe, [The Mandelbox](https://sites.google.com/site/mandelbox/) (2010).
- H.-O. Peitgen and D. Saupe (eds.), *The Science of Fractal Images*, Springer (1988), chapter 4: the
  distance estimator method for escape-time fractals.

## Read the code

- [`js/examples/ray-marching.js`](js/examples/ray-marching.js) has the four distance estimators, `rayMarch`,
  the shading in `fs_main`, the renderer, and the compute-only `measureRayMarch` with its statistics shader.
- [`js/teach/experiments/ray-marching.js`](js/teach/experiments/ray-marching.js) has the experiment;
  [`js/refs/ray-marching.js`](js/refs/ray-marching.js) has the float64 estimators and tracer.
- [`test/ray-marching.test.mjs`](test/ray-marching.test.mjs) checks the estimators against brute-force membership
  marches; [`test/gpu/ray-marching-measure.gpu.test.mjs`](test/gpu/ray-marching-measure.gpu.test.mjs) runs the
  shipped shader on a software GPU in CI and is the source of the numbers above.
