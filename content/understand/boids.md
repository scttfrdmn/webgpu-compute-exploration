## Overview

Boids (Reynolds, 1987) are agents that follow three local steering rules: **separation** (don't crowd your
neighbours), **alignment** (fly the way they fly) and **cohesion** (stay with them). No boid knows the shape
of the flock, yet flocks form, split, merge and wheel. It was one of the first demonstrations that simple
local rules produce collective behaviour, and the idea underlies models of bird flocks, fish schools and
crowds.

The example moves up to 10,000 boids in a $1000 \times 700$ pixel world that wraps around at the edges, one
compute dispatch per frame, with a mouse force and four behaviour modes. For a GPU programmer it is an
all-pairs neighbour kernel with uniform work and broadcast loads. For a computational scientist it is a
discrete-time dynamical system with an **order parameter**, the polarization, that measures how ordered the
flock is.

## The model

### Steering

Each boid $i$ has a position $\mathbf{x}_i$ and a velocity $\mathbf{v}_i$, in pixels and pixels per frame.
Following Reynolds, each rule computes a **desired velocity** of length $v_{\max}$, and the steering force is
the difference from the current velocity, capped at $F_{\max}$:

$$
\mathbf{s} = \operatorname{limit}\!\left(v_{\max}\,\hat{\mathbf{d}} - \mathbf{v}_i,\; F_{\max}\right),
\qquad \operatorname{limit}(\mathbf{a}, m) = \mathbf{a} \min\!\left(1, \frac{m}{|\mathbf{a}|}\right)
$$

where $\hat{\mathbf{d}}$ is a unit vector chosen by the rule, from the neighbours $j$ within that rule's
radius:

| Rule | Radius | Direction $\hat{\mathbf{d}}$ | Weight |
|---|---|---|---|
| separation | 25 px | $\sum_j \frac{\mathbf{x}_i - \mathbf{x}_j}{\vert \mathbf{x}_i - \mathbf{x}_j \vert^2}$, normalized | 1.5 |
| alignment | 50 px | the mean of $\mathbf{v}_j$, normalized | 1 |
| cohesion | 50 px | toward the mean of $\mathbf{x}_j$, normalized | 1 |

The separation term weights each neighbour by $1/r$ (the unit vector away from it, divided by the distance),
so close neighbours dominate. The defaults are $v_{\max} = 4$ px per frame (the Speed slider sets
$v_{\max}$ to its value divided by 10) and $F_{\max} = 0.1$.

### Update

The acceleration is the weighted sum of the three, plus the mouse force (toward the cursor while the left
button is held, away with the right, up to 1 px/frame² at the cursor and fading to zero at 150 px) and the
behaviour mode. Then, with a time step of one frame,

$$
\mathbf{v}_i \leftarrow \operatorname{limit}(\mathbf{v}_i + \mathbf{a}_i,\; v_{\max}), \qquad
\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathbf{v}_i
$$

and a boid that leaves one edge reappears at the opposite one. The modes replace or add a term: **predator**
makes boid 0 (drawn red) steer toward the center while the others flee it within 150 px; **scatter** weakens
the flocking forces to 10% and pushes everyone away from the center; **vortex** adds a strong swirl around the
center.

> **Note:** the world wraps, but neighbour distances do not: two boids on opposite edges are close on the
> torus but 1,000 px apart in the shader's distance, so they ignore each other. A flock crossing an edge is
> briefly split in two.

## Mapping to the GPU

### Buffers and threads

| Buffer | WGSL | Contents |
|---|---|---|
| `boidBuffers[0]`, `[1]` | `array<Boid>` | `Boid { position: vec2<f32>, velocity: vec2<f32> }`, 16 bytes |
| `paramsBuffer` | `Params` (80 bytes, in a 128-byte buffer) | $N$, radii, weights, $v_{\max}$, $F_{\max}$, bounds, mouse, mode |
| `uniformBuffer` | `Uniforms` | the canvas size, for rendering |

`Params` mixes `u32` and `f32` fields and three `vec2<f32>`, which must start on 8-byte boundaries, so the
writer leaves one padding float and writes the counts through a `Uint32Array` view. The compute shader runs
in workgroups of 64, one invocation per boid, and ping-pongs between the two boid buffers as the other
particle examples do: a step reads only the previous positions.

### Three loops over all boids

`separation`, `alignment` and `cohesion` are three separate functions, and each loops over **all** $N$
boids, computing the distance and testing it against its radius. A step therefore costs $3N^2$ distance
tests, 12 million at the default 2,000 and 300 million at 10,000. (The readout's "O(N²) = N²" counts one
loop.) Every thread of a workgroup reads the same `boidsIn[i]` at the same time, a broadcast load that the
hardware serves once, and the branch on the radius is short, so divergence is small.

> **GPU:** two cheap improvements are possible without changing the result: one loop that evaluates all three
> rules (the float64 reference does this), which cuts the loads by 3×, and tiling the boids through
> `var<workgroup>` memory as the N-body example's brute-force kernel does. The real fix is a uniform grid of
> 50-pixel cells, so each boid visits only the 9 cells around it: $\mathcal{O}(N)$ work, the "spatial hashing"
> the readout mentions.

### Per frame

`animate()` runs one compute dispatch and one instanced draw (a triangle per boid, rotated to its velocity,
colored by speed) per animation frame. Nothing is read back; the mouse position and mode go into the uniform
each frame.

## Numerical behaviour

> **Numerics:** boids conserve nothing, so there is no invariant to check. The rules are discontinuous
> (a neighbour 24.9 px away counts, one at 25.1 does not) and the update is chaotic: two runs that start a
> rounding error apart soon differ boid by boid. What can be compared is a single step, and statistics.

**One step.** From the same start, one GPU step agrees with a float64 step to $3 \times 10^{-5}$ px in CI, which
is half an `f32` unit in the last place for coordinates between 512 and 1,000: the difference is the rounding
of the stored coordinates, nothing more. After hundreds of steps, individual boids differ, because every difference is
amplified, but the flock's statistics do not.

**Order parameter.** The polarization

$$
\phi = \frac{1}{N} \left| \sum_i \frac{\mathbf{v}_i}{|\mathbf{v}_i|} \right|
$$

is 1 when all boids fly the same way and about $1/\sqrt{N}$ for random headings. Alignment acts only within 50
px, so order starts locally: small flocks with different headings form first and the global $\phi$ rises as they
merge. In CI, 500 boids went from $\phi = 0.08$ to 0.36 after 150 steps, 0.65 after 300 and 0.92 after 600, then
stayed between 0.87 and 0.92 as flocks met and split. The Vicsek model (1995) is the physicists' minimal version
of the same idea, with a noise term that drives a transition between ordered and disordered phases.

**The time step is one frame.** Velocities are in pixels per frame, so the dynamics depend on the frame rate: a
120 Hz display runs the flock twice as fast as a 60 Hz one. There is no $\Delta t$ to refine; the model is
defined in discrete time.

## Why it maps well — or poorly — to a GPU

**It maps well.** Every boid does exactly the same work, with no atomics and no communication beyond reading the
previous state, and the inner loops are broadcast loads plus a few flops. Thousands of boids fill a GPU, and the
wall time follows $N^2$ closely once they do.

**The algorithm wastes almost all of that work.** At the default density a boid has only a few neighbours within
50 px, yet it tests all $N$ boids three times. The quadratic cost is why the slider stops at 10,000; a grid
search would allow millions.

## Try this

> **Try this:** in [Explore](#boids), press Reset and Start, and watch the flocks form. Then hold the right mouse
> button in the middle of a flock and release it.

- Try **Add Predator** and watch boid 0: it steers to the center and the others clear a space around it.
  **Vortex** adds a swirl strong enough to override the flocking rules.
- Watch a flock cross an edge: it reappears on the other side but is briefly two flocks, since neighbour
  distances do not wrap.
- Run [emergence of order](#boids/measure) with 500, 2,000 and 4,000 boids and compare how quickly the
  polarization rises and how high it gets.
- Run [boids vs time per step](#boids/measure) and compare with the $N^2$ line; the distance tests per second
  are the throughput a grid search would spend on useful neighbours instead.

## References

- C. W. Reynolds, [Flocks, herds and schools: a distributed behavioral model](https://doi.org/10.1145/37402.37406),
  SIGGRAPH (1987).
- C. W. Reynolds, [Steering behaviors for autonomous characters](https://www.red3d.com/cwr/steer/gdc99/),
  Game Developers Conference (1999): desired velocity minus current velocity, capped.
- T. Vicsek, A. Czirók, E. Ben-Jacob, I. Cohen and O. Shochet, [Novel type of phase transition in a system of self-driven particles](https://doi.org/10.1103/PhysRevLett.75.1226),
  *Phys. Rev. Lett.* 75, 1226 (1995): the polarization as an order parameter.
- S. Green, [Particle simulation using CUDA](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf),
  NVIDIA (2010): the uniform-grid neighbour search on a GPU.

## Read the code

- [`js/examples/boids.js`](js/examples/boids.js) has the compute shader (the three rules, the modes, the update),
  the render shader, the `BoidsSimulation` class, and `measureBoids`, the compute-only version the Measure tab
  uses.
- [`js/teach/experiments/boids.js`](js/teach/experiments/boids.js) has the two experiments;
  [`js/refs/boids.js`](js/refs/boids.js) has the float64 step and the polarization.
- [`test/boids.test.mjs`](test/boids.test.mjs) checks a CPU replica of the shader on hand-computed cases and
  emergent alignment; [`test/gpu/boids-measure.gpu.test.mjs`](test/gpu/boids-measure.gpu.test.mjs) runs the
  shipped shader on a software GPU in CI and is the source of the numbers above.
