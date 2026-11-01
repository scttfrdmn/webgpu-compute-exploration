## Overview

Two chemicals, $U$ and $V$, spread through a thin layer and react with each other. With the right feed
and kill rates, an almost uniform mixture organizes itself into spots, stripes, mazes or dividing
"cells". Alan Turing proposed in 1952 that this kind of instability could lay down the patterns of
animal coats. The Gray-Scott model is one of the simplest systems that does it.

The example solves the model on a 512 × 384 grid that wraps around at the edges (a torus), one GPU thread
per cell, 1 to 32 steps per frame. For a GPU programmer it is the cleanest possible **stencil**
computation: every cell reads its 3 × 3 neighbourhood and writes one value, with a ping-pong pair of
buffers. For a computational scientist it is an explicit finite-difference scheme for a stiff,
nonlinear PDE, run at the largest time step the stencil allows, in single precision.

> **Note:** Units are grid cells and steps: $\Delta x = 1$, $\Delta t = 1$. The concentrations stay in
> $[0, 1]$; the picture shows $V$.

## The model

### The Gray-Scott equations

$U$ is fed in from a reservoir at rate $F$, the autocatalytic reaction $U + 2V \to 3V$ converts it into
$V$, and $V$ is removed at rate $F + k$:

$$
\begin{aligned}
\frac{\partial U}{\partial t} &= D_u \nabla^2 U - U V^2 + F (1 - U) \\
\frac{\partial V}{\partial t} &= D_v \nabla^2 V + U V^2 - (F + k) V
\end{aligned}
$$

The shader uses $D_u = 1$ and $D_v = 0.5$: $V$ diffuses half as fast as $U$. That difference is the
whole trick. A local excess of $V$ grows by eating $U$; the $U$ it needs diffuses in quickly from
around it, while the $V$ it makes stays put. Short-range activation plus long-range inhibition is
Turing's mechanism, and it fixes a typical length scale for the spots and stripes.

The six presets are $(F, k)$ pairs from the regions Pearson (1993) mapped: Coral (0.0545, 0.062),
Mitosis (0.0367, 0.0649), Maze (0.029, 0.057), Holes (0.039, 0.058), Chaos (0.018, 0.051) and Cells
(0.062, 0.0609). Outside those regions the pattern either dies out ($V \to 0$ everywhere) or fills the
plane uniformly.

### The discrete Laplacian

$\nabla^2$ is a 3 × 3 convolution with weight $-1$ at the center, $0.2$ on the four edge neighbours and
$0.05$ on the four corners (the kernel of Karl Sims' well-known tutorial). The weights sum to zero, so a
constant field does not diffuse. A Taylor expansion of a smooth field gives

$$
\sum_{i,j} w_{ij}\, u(x + i, y + j) \approx \tfrac{1}{2} \Big(\sum_{i,j} w_{ij}\, i^2\Big) \nabla^2 u = 0.3\, \nabla^2 u
$$

so the stencil is $0.3\,\nabla^2$, not $\nabla^2$: the effective diffusion coefficients are $0.3$ and
$0.15$ cells² per step. Including the corners makes the stencil nearly isotropic, so patterns do not
line up with the grid axes the way they do with the 5-point Laplacian.

### Forward Euler

Each step is one explicit Euler update, clamped to $[0, 1]$:

$$
U_{n+1} = U_n + \Delta t \left(D_u L U_n - U_n V_n^2 + F (1 - U_n)\right), \qquad
V_{n+1} = V_n + \Delta t \left(D_v L V_n + U_n V_n^2 - (F + k) V_n\right)
$$

with $L$ the stencil above. The grid starts at $U = 1, V = 0$ with a few random 10 × 10 patches of
$U = 0.5$, $V = 0.25 \pm 0.01$; painting with the mouse writes the same mix.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `cellsIn` / `cellsOut` | `vec2<f32>` | $W \times H$ each | $(U, V)$ per cell, row-major |
| `params` | `Params` | 1 (48 bytes) | grid size, feed, kill, brush stroke, flags, palette |

The two cell buffers are a **ping-pong** pair: bind group 0 reads buffer 0 and writes buffer 1, bind
group 1 the reverse, and the example swaps after every step. A cell's update needs the old values of its
neighbours, so it cannot be done in place: a thread would read a neighbour that another thread has
already overwritten. At 512 × 384 each buffer is 1.5 MB.

### Workgroups and dispatches

`cs_main` runs in 8 × 8 workgroups, one thread per cell, so a step is $\lceil W/8 \rceil \times
\lceil H/8 \rceil = 64 \times 48$ workgroups. Threads past the edge of a grid that is not a multiple of 8
return immediately. The wrap-around is an integer modulo, `(x + w) % w`, on every neighbour index.

A frame encodes all its steps (12 by default) as consecutive dispatches in **one compute pass**, then
one full-screen triangle that colors $V$. WebGPU guarantees that each dispatch sees the storage writes of
the previous one, so no barriers or fences are needed between steps. Nothing is read back to the CPU.

> **GPU:** a 3 × 3 stencil reads each value nine times, from nine different threads. The kernel does not
> stage a tile in `var<workgroup>` memory; it relies on the GPU's caches, which hold the three rows a
> workgroup touches. For a stencil this small that is usually enough; the
> [grid-size experiment](#reaction-diffusion/measure) shows the throughput you actually get.

## Numerical behaviour

> **Numerics:** forward Euler on a diffusion term is stable only while $\Delta t\,(D\,|\lambda_{\max}| + r) \le 2$,
> where $\lambda$ ranges over the eigenvalues of the stencil and $r$ is the linear decay rate. Here
> $|\lambda_{\max}| = 1.6$ (the checkerboard mode), so with $D_u = \Delta t = 1$ the worst preset sits at
> 1.66: the time step is close to the largest one allowed.

**Stability.** The Fourier symbol of the stencil is
$\lambda(k_x, k_y) = -1 + 0.4 (\cos k_x + \cos k_y) + 0.2 \cos k_x \cos k_y$, which is never positive
and reaches $-1.6$ at $k_x = k_y = \pi$. A mode with symbol $\lambda$ is multiplied by
$1 + \Delta t\,(D_u \lambda - F)$ per step, so it stays bounded while that factor is at least $-1$. The
CPU test checks every preset: the largest value of $\Delta t\,(D\,|\lambda| + F + k)$ is 1.66, for Cells.
Doubling $\Delta t$ would make the checkerboard mode grow without bound.

**Accuracy.** The scheme is first order in time and second order in space, and with $\Delta t = 1$ it is
not a converged solution of the PDE: a smaller time step would change the fine details of the pattern.
What it does reproduce is the qualitative behaviour (which pattern forms for which $F, k$), and that is
how the model is normally used.

**Single precision.** The GPU stores and computes in `f32`. A CPU float64 solution of exactly the same
discrete equations, from the same start, drifts away from it at a rate that depends on the pattern. In
CI (the shipped shader on a software GPU, 128 × 128):

| steps | 1 | 100 | 1,000 | 2,000 | 3,000 | 4,000 |
|---|---|---|---|---|---|---|
| Coral: max $\vert f_{32} - f_{64} \vert$ | 7.2e-8 | 2.6e-6 | 3.1e-5 | 1.0e-4 | 2.8e-4 | 5.1e-4 |
| Chaos: max $\vert f_{32} - f_{64} \vert$ | 7.1e-8 | 3.2e-6 | 1.6e-4 | 1.7e-3 | 1.8e-2 | 0.76 |

In both, the first step agrees to the rounding of an `f32`. Coral's difference grows slowly. Chaos keeps
splitting and annihilating spots, so a tiny difference decides whether a spot splits a step earlier or
later, and the two solutions separate exponentially, about tenfold every 1,000 steps and then faster:
by step 4,000 some cells differ by 0.76, a spot in one solution where the other has none, although the
statistics of the two patterns (the share of cells with $V > 0.1$ is 43% in both) are the same. This is
sensitivity to initial conditions, not a bug in either solver.

## Why it maps well — or poorly — to a GPU

**It maps very well.** Every cell does the same small amount of work with no branches (the brush test is
uniform across a dispatch), neighbouring threads read neighbouring memory, and there is no communication
except through the next dispatch. It is the textbook data-parallel stencil.

**What limits it is memory, not arithmetic.** Each cell update does about 40 floating-point operations
(plus the integer arithmetic of the wrap-around) and moves 8 bytes in and 8 bytes out of DRAM, if the
other eight reads hit the cache. That ratio, about 2.5 operations per byte, is far below what a GPU needs
to be compute-bound, so the time per step is set by memory bandwidth at large grids, and by dispatch
overhead at small ones. In CI, on a CPU-emulated GPU, a step costs 0.6 to 1 ms at 128² and 6 to 10 ms
at 512², a few tens of millions of cell updates per second; a discrete GPU does billions.

> **GPU:** at 512 × 384 one step is only 196,608 threads, 3,072 workgroups. That is enough to fill a
> laptop GPU but not a large desktop one, which is one reason the example runs up to 32 steps per frame:
> the per-dispatch overhead is paid once per step, and the per-frame overhead once per 32 steps.

## Try this

> **Try this:** in [Explore](#reaction-diffusion), pick Mitosis, press Start and watch the spots grow and
> divide. Then drag **Kill** up by 0.002 and watch them die; drag it down and they fill the plane.

- Paint into a quiet area of Holes or Cells. The pattern invades it at a fixed speed: a travelling front
  is one of the things this model is known for.
- Set **Steps / frame** to 1 and then 32 in [Explore](#reaction-diffusion) and compare the GPU time in
  the readout: the cost is per step, the frame overhead is not.
- Run [grid size vs time per step](#reaction-diffusion/measure). At which grid does the GPU line stop
  being flat? Compare the cell-update rate with your GPU's memory bandwidth divided by 16 bytes.
- Run [single precision vs float64](#reaction-diffusion/measure) with Coral and then with Chaos. Same
  equations, same start: why does one stay within 1e-4 while the other reaches 1e-2?

## References

- A. M. Turing, [The chemical basis of morphogenesis](https://doi.org/10.1098/rstb.1952.0012),
  *Phil. Trans. R. Soc. B* 237, 37 (1952).
- P. Gray and S. K. Scott, [Autocatalytic reactions in the isothermal, continuous stirred tank reactor](https://doi.org/10.1016/0009-2509(84)87017-7),
  *Chem. Eng. Sci.* 39, 1087 (1984): the reaction scheme.
- J. E. Pearson, [Complex patterns in a simple system](https://doi.org/10.1126/science.261.5118.189),
  *Science* 261, 189 (1993): the map of patterns in the $(F, k)$ plane that the presets come from.
- K. Sims, [Reaction-diffusion tutorial](https://www.karlsims.com/rd.html): the 3 × 3 kernel and
  $D_u = 1$, $D_v = 0.5$, $\Delta t = 1$ used here.
- R. J. LeVeque, [Finite Difference Methods for Ordinary and Partial Differential Equations](https://doi.org/10.1137/1.9780898717839),
  SIAM (2007), chapters 9 and 10: stability of explicit schemes for diffusion.

## Read the code

- [`js/examples/reaction-diffusion.js`](js/examples/reaction-diffusion.js): `cs_main` has the update,
  the file header has the `Params` layout, and `frame()` the dispatch loop. The model constants `DU`,
  `DV`, `DT`, `W_EDGE`, `W_CORNER` are JavaScript exports spliced into the WGSL.
- [`js/teach/experiments/reaction-diffusion.js`](js/teach/experiments/reaction-diffusion.js) has the
  two Measure experiments, built on `measureSteps` (compute-only, at the end of the example file).
- [`js/refs/reaction-diffusion.js`](js/refs/reaction-diffusion.js) has the float64 step they compare
  against.
- [`test/reaction-diffusion.test.mjs`](test/reaction-diffusion.test.mjs) checks the stencil, the
  stability bound and a CPU replica of the shader;
  [`test/gpu/reaction-diffusion-measure.gpu.test.mjs`](test/gpu/reaction-diffusion-measure.gpu.test.mjs)
  runs the shipped shader on a software GPU in CI and is the source of the numbers on this page.
