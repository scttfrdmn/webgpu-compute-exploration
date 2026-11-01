## Overview

Procedural noise is a random-looking but smooth and repeatable function of position. Ken Perlin invented
gradient noise in 1983, after working on the film *Tron*; summed over several scales it makes terrain, clouds, marble and
fire, and it is still the basic building block of procedural textures and worlds. This example evaluates
3D Perlin or simplex noise for every pixel of a 1024 × 768 field, as **fractal Brownian motion** (fBm),
optionally with ridges and **domain warping**, and colors the result.

For a GPU programmer it is the opposite of the stencil examples: every pixel is independent, the shader
reads no memory except its uniforms, and the work per pixel is large. It is limited by arithmetic, which
is what GPUs have most of. For a computational scientist it is a lesson in what single-precision
coordinates do to a function that must be evaluated at large arguments.

## The model

### Gradient (Perlin) noise

Put a pseudo-random unit-ish gradient $\mathbf{g}$ at every integer lattice point. For a point
$\mathbf{p}$ in the cell with corner $\mathbf{i} = \lfloor \mathbf{p} \rfloor$ and offset
$\mathbf{f} = \mathbf{p} - \mathbf{i}$, each of the 8 corners $\mathbf{c}$ contributes the linear ramp
$\mathbf{g}_{\mathbf{c}} \cdot (\mathbf{f} - \mathbf{c})$, and the ramps are blended trilinearly with the
quintic fade

$$
s(t) = 6t^5 - 15t^4 + 10t^3 ,
$$

whose first and second derivatives vanish at 0 and 1, so the noise and its gradient are continuous
across cells (Perlin 2002). The noise is zero at every lattice point, and its values lie in about
$[-1, 1]$.

The gradients come from the 12 edge directions of a cube, $(\pm 1, \pm 1, 0)$ and permutations,
selected by the top 4 bits of a hash of the corner. Instead of Perlin's 256-entry permutation table, the
shader hashes the integer corner with **PCG3D** (Jarzynski and Olano 2020), a few integer multiplies,
adds and shifts, so there is no table to store or look up.

### Simplex noise

Perlin's later **simplex noise** (2001) replaces the cube by tetrahedra. Skewing the coordinates by
$F_3 = 1/3$ turns the cubic lattice into one whose cells split into six tetrahedra; ranking the
components of the offset finds the one containing $\mathbf{p}$, and only its 4 corners contribute, each
with a radial falloff instead of a blend:

$$
n(\mathbf{p}) = 72 \sum_{k=0}^{3} \max\left(0,\; 0.5 - \vert \mathbf{d}_k \vert^2\right)^4 \, \mathbf{g}_k \cdot \mathbf{d}_k ,
$$

where $\mathbf{d}_k$ is the offset from corner $k$, unskewed by $G_3 = 1/6$. The squared radius 0.5
reaches exactly to the opposite face of each tetrahedron, so the sum has no seams; Gustavson's 0.6 leaves
small discontinuities in 3D. The factor 72 (`SIMPLEX_SCALE`) maps the result to about $[-1, 1]$.

### Fractal sums and warping

fBm adds octaves of noise at growing frequencies and shrinking amplitudes:

$$
\mathrm{fbm}(\mathbf{p}) = \frac{\sum_{o=0}^{N-1} a^o\, n(\mathbf{p}_o)}{\sum_{o=0}^{N-1} a^o}, \qquad
\mathbf{p}_{o+1} = \ell\, \mathbf{p}_o + \mathbf{t}
$$

with persistence (gain) $a$, lacunarity $\ell$, and a fixed shift $\mathbf{t} = (19.1, 7.3, 3.7)$ so the
lattices of different octaves do not line up at the origin. **Ridged** mode folds each octave as
$2(1 - \vert n \vert)^2 - 1$, turning zero crossings into sharp crests. **Domain warping** (Quilez)
evaluates $\mathrm{fbm}(\mathbf{p} + w\,\mathbf{q})$ with $\mathbf{q} = (\mathrm{fbm}(\mathbf{p}),
\mathrm{fbm}(\mathbf{p} + \mathbf{c}))$: three fBm sums per pixel. Marble is
$\sin(5(x + y/2) + 8\,\mathrm{fbm})$, Perlin's 1985 recipe.

Pixel $(x, y)$ maps to $\mathbf{p} = (\text{world}_x f, \text{world}_y f, t + 0.5)$, where the view is one
world unit tall, $f$ is the frequency and $t$ the animation time. The $+0.5$ keeps the first octave off a
lattice plane, where Perlin noise would be zero everywhere.

## Mapping to the GPU

### Buffers and dispatch

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `field` | `f32` | $W \times H$ (786,432) | fBm value per pixel |
| `params` | `Params` | 1 (56 bytes) | size, view center and zoom, frequency, time, octaves, lacunarity, gain, warp, type, ridged, color mode |

`cs_main` runs in 8 × 8 workgroups, one thread per pixel: $128 \times 96$ workgroups for the 1024 × 768
canvas. A frame is one compute dispatch and one full-screen triangle whose fragment shader colors the
field (terrain bands with hillshading from neighbouring values, clouds, marble, or grayscale). Nothing is
read back. While paused, the example recomputes only when a control or the view changes.

### Work per pixel

The octave loop has a fixed bound, `MAX_OCTAVES` = 8, and breaks at the uniform octave count, so every
thread in a dispatch runs the same number of iterations. The Terrain preset (7 octaves, warp 0.4) makes
$1024 \times 768 \times 7 \times 3 \approx 1.65 \times 10^7$ noise evaluations per frame; each Perlin
evaluation is 8 hashes and gradients and 7 blends, roughly 150 instructions.

> **GPU:** this kernel reads 56 bytes of uniforms and writes 4 bytes per pixel, and does thousands of
> arithmetic operations in between. Its arithmetic intensity is hundreds of operations per byte, so it
> runs at the GPU's compute throughput, not its memory bandwidth. Integer multiplies (the hash) are full
> speed on current GPUs.

## Numerical behaviour

> **Numerics:** in `f32` the position inside a lattice cell, $\mathbf{p} - \lfloor \mathbf{p} \rfloor$,
> has fewer bits the larger $\mathbf{p}$ is. High octaves are evaluated at large coordinates (the
> frequency doubles each octave), so their error grows, but their weight shrinks as fast.

**Accuracy.** The [statistics experiment](#noise/measure) compares the GPU field with a float64 Perlin
(a weighted corner sum) and simplex noise (a brute-force sum over a 5 × 5 × 5 block of lattice points),
both independent formulations sharing only the hash and the gradient table. In CI, on a 256² field at
frequency 8, the largest difference on 4,096 sampled pixels is:

| octaves | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Perlin | 1.0e-7 | 3.9e-7 | 2.4e-6 | 4.0e-6 |
| simplex | 1.4e-6 | 2.6e-6 | 8.4e-6 | 1.5e-5 |

One octave of Perlin noise is accurate to about one `f32` ulp: its cell offset $\mathbf{p} - \lfloor \mathbf{p} \rfloor$
is computed exactly in floating point. Simplex starts higher because its offsets go through the skew and
unskew, $\mathbf{p} + s$ and $\mathbf{i} - t$, each rounded to the precision of the coordinate (about
$5 \times 10^{-7}$ at $\vert p \vert \approx 4$), and the noise multiplies that by its slope. By the eighth
octave the coordinates are around $3 \times 10^3$ (the shifts $\mathbf{t}$ add up), where an `f32` has a
resolution of $2.4 \times 10^{-4}$, but that octave has weight $2^{-7}/1.99$.

**Statistics.** One octave of Perlin noise has a standard deviation of 0.273 on this field and simplex
0.369, with means near 0. Normalizing fBm by $\sum a^o$ keeps its range inside $[-1, 1]$ but shrinks its
spread: if the octaves were independent with equal spread $\sigma$, the sum would have
$\sigma \sqrt{\sum a^{2o}} / \sum a^o$, which is $0.58\,\sigma$ for 8 octaves at $a = 0.5$. CI measures
0.158 for Perlin (prediction 0.158) and 0.210 for simplex (prediction 0.214).

**Zooming and panning.** The view center and scale are `f32` uniforms. Zooming in far from the origin
eventually runs out of bits in the world coordinate: the image turns blocky. The example limits zoom to
64×; real terrain engines evaluate noise relative to a local origin, or in double
precision on the CPU.

## Why it maps well — or poorly — to a GPU

**It maps as well as anything can.** Every pixel is independent, there is no memory traffic to speak of,
and the octave count is uniform, so threads in a SIMD group stay in lockstep through the loop. Doubling
the octaves or turning warping on multiplies the work, and the time follows.

**The one divergent part is simplex noise's tetrahedron search**, six-way nested `if`s on the order of
the offset's components; neighbouring pixels usually agree, so the cost is small. Perlin noise does more
work per evaluation (8 corners instead of 4) but has no data-dependent branches at all. Which is faster
depends on the GPU, and it is worth measuring.

> **GPU:** one CPU core needs on the order of a second for the 16.5 million noise evaluations of one
> Terrain frame. The GPU does it many times per second, because the work is pure arithmetic in
> parallel, with nothing to wait for.

## Try this

> **Try this:** in [Explore](#noise), choose Grayscale colors, set **Octaves** to 1 and then 8. The
> large shapes stay put; each octave adds detail at half the size.

- In [Explore](#noise), switch between Perlin and Simplex at 1 octave. Perlin's features tend to line
  up with the axes of its cubic lattice; simplex has no preferred directions to the eye.
- Load **Warped** and drag **Warp** from 0 to 4. The cost per pixel triples as soon as warp is above 0;
  compare the GPU time in the readout.
- Run [octaves and resolution vs time](#noise/measure) at 1024² and 2048². Is the time proportional to
  octaves × pixels on your GPU? Which noise is cheaper?
- Run [field statistics vs the float64 reference](#noise/measure) for each noise. How close is the
  independent-octave estimate of the spread, and why is it not exact?

## References

- K. Perlin, [An image synthesizer](https://doi.org/10.1145/325165.325247), *SIGGRAPH* (1985): gradient
  noise, turbulence and marble.
- K. Perlin, [Improving noise](https://doi.org/10.1145/566654.566636), *SIGGRAPH* (2002): the quintic fade
  and the 12 edge gradients.
- K. Perlin, *Noise hardware*, in *Real-Time Shading*, SIGGRAPH course notes (2001): simplex noise.
- S. Gustavson, *Simplex noise demystified*, Linköping University (2005): the skew factors and the
  algorithm the shader follows.
- M. Jarzynski and M. Olano, [Hash functions for GPU rendering](https://jcgt.org/published/0009/03/02/),
  *Journal of Computer Graphics Techniques* 9(3), 21 (2020): PCG3D.
- I. Quilez, [Domain warping](https://iquilezles.org/articles/warp/) (2002).
- D. S. Ebert, F. K. Musgrave, D. Peachey, K. Perlin and S. Worley, *Texturing and Modeling: A Procedural
  Approach*, 3rd ed., Morgan Kaufmann (2003): fBm, ridged multifractals and their parameters.

## Read the code

- [`js/examples/noise.js`](js/examples/noise.js): `perlin3`, `simplex3`, `fbm` and `warpedFbm` in the
  compute shader, each with its algorithm in a comment; the file header has the `Params` layout.
  `SIMPLEX_SCALE` and `SIMPLEX_FALLOFF` are JavaScript exports spliced into the WGSL.
- [`js/teach/experiments/noise.js`](js/teach/experiments/noise.js) has the two Measure experiments, built
  on `measureNoise` (compute-only, at the end of the example file).
- [`js/refs/noise.js`](js/refs/noise.js) has the independent float64 Perlin, simplex and fBm.
- [`test/noise.test.mjs`](test/noise.test.mjs) checks a CPU replica against them, the continuity of both
  noises and the gradient distribution;
  [`test/gpu/noise-measure.gpu.test.mjs`](test/gpu/noise-measure.gpu.test.mjs) runs the shipped shader on
  a software GPU in CI and is the source of the numbers on this page.
