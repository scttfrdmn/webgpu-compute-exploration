## Overview

Frost on a window, mineral dendrites in rock, the branches of an electrodeposit and soot particles all
grow the same way: particles wander at random until they touch the growing cluster, and stick. Witten and
Sander (1981) turned this into a model, **diffusion-limited aggregation** (DLA), and showed that it makes
**fractals**: branching structures whose mass grows more slowly with size than a solid's does.

This example grows a DLA cluster in a 64³ voxel grid with 5,000 random walkers, one GPU thread per
walker, and draws every occupied voxel as a cube. For a GPU programmer it is a lesson in **atomics on a
shared grid** and in what happens when threads finish at different times. For a computational scientist
it is a Monte Carlo growth model with a measurable fractal dimension, and a small enough system that the
finite size of the box shows.

> **Note:** a walker that sticks is retired for good, and nothing replaces it, so the aggregate stops
> growing at about 5,001 cells (the seed plus every walker). Reset starts over.

## The model

### Random walk and sticking

Each walker sits on a lattice cell. Every step it moves one cell along a random axis, $\pm x$, $\pm y$ or
$\pm z$. Then:

- if the move would leave the 64³ box, the walker is put back at a random point of the **spawn sphere**,
  radius $0.45 \times 64 = 28.8$ cells around the center;
- if the target cell is already part of the aggregate, the walker stays where it is;
- if any of its 26 neighbours (faces, edges and corners) is occupied, it sticks with probability $p$,
  the **stickiness**: its cell becomes occupied and the walker retires.

The seed is one cell at the center (or a line, a disc or four points). All 5,000 walkers start on the
spawn sphere at once and walk simultaneously, so this is a many-walker variant of the original model,
which released one walker at a time from far away.

### Fractal dimension

For a solid ball, the number of cells within distance $r$ of the center grows as $r^3$. For a DLA
cluster it grows as

$$
N(r) \propto r^{D}
$$

with a **fractal dimension** $D$ below 3: about 1.71 in two dimensions and about 2.5 in three (Meakin
1983). Branches screen the interior: a walker coming from outside is far more likely to touch a tip than
to reach a fjord between branches, so tips grow and holes stay empty. The Measure tab estimates $D$ by
the **mass-radius** method, a least-squares fit of $\log N(r)$ against $\log r$ for $2 \le r \le R_g$,
where the radius of gyration about the seed is

$$
R_g = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \vert \mathbf{x}_i - \mathbf{x}_0 \vert^2} .
$$

A lower stickiness lets walkers bounce off the outer branches and reach further in before they stick,
so the cluster is denser and more compact.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `particles` | `Particle` (16 bytes) | 5,000 | position `vec3<f32>`, `rngState` `u32` (0 = retired) |
| `grid` | `atomic<u32>` | $64^3 = 262{,}144$ | 1 = occupied |
| `params` | `Params` | 1 (16 bytes) | walker count, grid size, stickiness, time |

Each walker carries its own random state, a 32-bit PCG hash advanced for every decision, so there is no
shared random-number generator to contend for.

### Dispatches

`main` runs in 1D workgroups of 64, one thread per walker: 79 workgroups. A frame runs 1 to 50 walk steps
(10 by default) as consecutive dispatches in one compute pass, then one render pass. Retired walkers
return on their first instruction. Nothing is read back to the CPU.

### Atomics

The grid is the only shared state. A walker reads 27 cells per step (its target and 26 neighbours) with
`atomicLoad` and marks its own cell with `atomicStore`. These are relaxed atomics: they make each access
indivisible and visible to other threads, but they do not order a walker's read against another walker's
write in the same dispatch. Two things can therefore happen in parallel that a one-walker-at-a-time
simulation would order:

- Two walkers next to the same branch stick in the same step. That is harmless: each had an occupied
  neighbour when it looked, so the aggregate stays connected.
- Two walkers stick into the same cell, or a walker sticks on a cell that has just become occupied. Then
  one walker adds no new cell. In CI, at stickiness 1, 6 to 8 of about 4,970 stuck walkers did.

> **GPU:** the occupancy test is a read and the mark is a write, never a read-modify-write, so no
> `atomicCompareExchange` loop is needed. The cost is a small, measurable loss of mass, not a corrupted
> grid. Every aggregate CI grows on a software GPU is 100% 26-connected to its seed.

### Rendering

The render pass draws one instanced cube per **grid cell**, 262,144 instances of 36 vertices, and the
vertex shader collapses the unoccupied ones to a degenerate point. That is 9.4 million vertex invocations
per frame whether the aggregate has 1 cell or 5,000: simple, but the render pass, not the simulation, is
the expensive part of a frame.

## Numerical behaviour

> **Numerics:** a single aggregate of 5,000 cells is small for a fractal measurement. The mass-radius fit
> covers less than one decade of $r$, and the spawn sphere caps the size, so estimates of $D$ scatter
> by $\pm 0.15$ between runs and stickiness values.

**What CI measures.** On the software GPU in CI, at 64³ with 5,000 walkers (the Explore settings):

| stickiness | 1 | 0.5 | 0.2 | 0.1 | 0.05 | 0.03 |
|---|---|---|---|---|---|---|
| steps until half the walkers stuck | 700 | 1,000 | 1,400 | 1,600 | 2,400 | 3,500 |
| radius of gyration (cells) | 24.8 | 23.7 | 21.8 | 21.0 | 19.4 | 17.4 |
| mass-radius $D$ | 2.43 | 2.30 | 2.44 | 2.42 | 2.32 | 2.64 |

The trend in compactness and growth time is clear; the trend in $D$ is not. A CPU port of the shader on
the same configuration gives values in the same band (2.25 to 2.55 for stickiness 1 to 0.03), and the
expected value for 3D DLA is about 2.5.

**Finite size.** The walkers start at radius 28.8, and the cluster's radius of gyration at stickiness 1
is 24.8: the aggregate fills most of the region the walkers come from. Real DLA studies release walkers
far outside the cluster and grow millions of particles; here the box shapes the outer branches. The
box-counting dimension, another common estimate, is even more affected: on clusters like these it gives
values between 1.3 and 1.7, set by how sparse the outer branches are, not by the scaling.

**Randomness and determinism.** Each walker's stream is deterministic, but the order in which walkers
see each other's sticking depends on the GPU's scheduling, so two GPUs, or two runs, give different
clusters from the same seed. Statistics like $R_g$ and $D$ are reproducible; the exact shape is not.

## Why it maps well — or poorly — to a GPU

**Walking maps well.** Walkers are independent, their state fits in 16 bytes, and the walk is a few
hashes and compares. A GPU does billions of such steps per second.

**Sticking maps poorly.** Each step does 27 scattered reads of the shared grid, a cache-unfriendly
pattern, and threads in one SIMD group finish the neighbour loop at different times (it returns at the
first occupied neighbour). Worse, threads retire: as the walkers stick, more and more of each workgroup
does nothing, and at the end a dispatch launches 79 workgroups to move a handful of walkers.

> **GPU:** 5,000 walkers are far too few to fill a modern GPU, which needs tens of thousands of threads
> in flight. The [scaling experiment](#dla/measure) shows how the time per step stays flat as the walker
> count grows, until the GPU is finally busy.

## Try this

> **Try this:** in [Explore](#dla), set **Stickiness** to 100% and press Start; then Reset, set it to 5%
> and start again. The low-stickiness cluster is smaller, denser and slower to grow.

- Try the **Horizontal Line** and **Multiple Seeds** buttons in [Explore](#dla). With several seeds the
  branches compete for walkers, and the first to grow screens the others.
- Push **Walk steps / frame** from 10 to 50: the cluster grows 5× faster per frame. Does the frame
  rate drop 5×? If not, the render pass, not the walk, sets the cost of a frame.
- Run [stickiness vs structure](#dla/measure) at 64³ and compare your $R_g$ and $D$ with the CI table
  above. Your GPU grows a different cluster; are the statistics the same?
- Run [walkers vs time per step](#dla/measure). Up to how many walkers does a step cost the same on
  your GPU?

## References

- T. A. Witten and L. M. Sander, [Diffusion-limited aggregation, a kinetic critical phenomenon](https://doi.org/10.1103/PhysRevLett.47.1400),
  *Phys. Rev. Lett.* 47, 1400 (1981).
- P. Meakin, [Diffusion-controlled cluster formation in 2–6-dimensional space](https://doi.org/10.1103/PhysRevA.27.1495),
  *Phys. Rev. A* 27, 1495 (1983): $D \approx 2.5$ in three dimensions.
- T. Vicsek, *Fractal Growth Phenomena*, 2nd ed., World Scientific (1992): DLA, sticking probability,
  and how fractal dimensions are measured.
- M. Jarzynski and M. Olano, [Hash functions for GPU rendering](https://jcgt.org/published/0009/03/02/),
  *Journal of Computer Graphics Techniques* 9(3), 21 (2020): the per-walker PCG hash.

## Read the code

- [`js/examples/dla-3d.js`](js/examples/dla-3d.js): `PARTICLE_UPDATE_SHADER` has the walk and stick
  rule, `RENDER_VERTEX_SHADER` the instanced cubes, and `DLA3D.step()` the dispatch loop.
- [`js/teach/experiments/dla.js`](js/teach/experiments/dla.js) has the two Measure experiments, built on
  `measureDLA` (compute-only, at the end of the example file).
- [`js/refs/dla.js`](js/refs/dla.js) measures an aggregate on the CPU: radius of gyration, mass-radius
  dimension and connectivity to the seed.
- [`test/dla-3d.test.mjs`](test/dla-3d.test.mjs) has a CPU replica of the shader and checks the stick
  rule, connectivity and the dimension;
  [`test/gpu/dla-measure.gpu.test.mjs`](test/gpu/dla-measure.gpu.test.mjs) runs the shipped shader on a
  software GPU in CI.
