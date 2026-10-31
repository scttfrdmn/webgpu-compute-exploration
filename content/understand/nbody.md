## Overview

A star cluster or galaxy is a set of point masses that attract each other. Every body pulls on every
other body, so the obvious force calculation costs $N(N-1)$ pair interactions per time step: about
$2.7 \times 10^{8}$ at $N = 16{,}384$. This example does that sum directly (**brute force**), and also with
the **Barnes-Hut** tree algorithm, which treats a distant group of bodies as one point at its center of
mass and cuts the cost to $\mathcal{O}(N \log N)$.

The interesting part for a GPU programmer is that the tree is **rebuilt from scratch on the GPU every
step**, in 31 compute dispatches, with no data sent back to the CPU. For a computational scientist, the
interesting part is the price of that speed: a controllable force error (set by the opening angle
$\theta$), and what a second-order symplectic integrator does with it.

> **Note:** Units are $G = 1$, total mass 1, scale length about 1. Every body has mass $m = 1/N$.

## The model

### Softened gravity

The acceleration of body $i$ is the sum over all other bodies, with **Plummer softening** $\varepsilon$:

$$
\mathbf{a}_i = G \sum_{j \ne i} m_j \frac{\mathbf{x}_j - \mathbf{x}_i}{\left(|\mathbf{x}_j - \mathbf{x}_i|^2 + \varepsilon^2\right)^{3/2}},
\qquad
\phi_i = -G \sum_{j \ne i} \frac{m_j}{\sqrt{|\mathbf{x}_j - \mathbf{x}_i|^2 + \varepsilon^2}}
$$

Softening replaces each point mass by a small Plummer sphere of radius $\varepsilon$. It removes the
$1/r^2$ singularity, so two bodies passing close together get a bounded kick instead of an arbitrarily
large one. That matters because a fixed time step cannot resolve a near-collision. The default is
$\varepsilon = 0.05$, adjustable from 0.005 to 0.2.

### Leapfrog (kick-drift-kick)

Positions and velocities advance with a fixed step $\Delta t$:

$$
\begin{aligned}
\mathbf{v}_{n+1/2} &= \mathbf{v}_n + \tfrac{\Delta t}{2}\,\mathbf{a}(\mathbf{x}_n) \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \Delta t\,\mathbf{v}_{n+1/2} \\
\mathbf{v}_{n+1} &= \mathbf{v}_{n+1/2} + \tfrac{\Delta t}{2}\,\mathbf{a}(\mathbf{x}_{n+1})
\end{aligned}
$$

There is one force evaluation per step: the forces at $\mathbf{x}_{n+1}$ are kept for the first kick
of the next step.

### Energy

The total energy $E = K + W$ is the readout that tells you whether the simulation is trustworthy:

$$
K = \sum_i \tfrac{1}{2} m_i |\mathbf{v}_i|^2, \qquad W = \tfrac{1}{2} \sum_i m_i \phi_i
$$

The $\tfrac{1}{2}$ in $W$ is there because each pair appears twice, once from each side. The example
shows the relative drift $(E - E_0)/|E_0|$ against the first sample after a reset, and the virial ratio
$2K/|W|$, which is about 1 for a cluster in equilibrium such as the Plummer sphere.

### Barnes-Hut

Barnes and Hut (1986) put the bodies in a hierarchy of boxes. Seen from body $i$, a node of size $l$
(the longest side of its bounding box) whose center of mass is a distance $d$ away is used as one point
mass if

$$
d \ge \frac{l}{\theta} + \delta
$$

and otherwise opened, with its children tested in turn. Here $\delta$ is the distance from the node's
center of mass to its box center. Barnes (1994) added this term to guard against a node whose mass sits
in one corner. A node whose box contains the body is always opened. Smaller $\theta$ is more accurate;
$\theta \to 0$ opens everything and gives back the direct sum.

## Mapping to the GPU

### Buffers and layout

Every per-body quantity is a `vec4<f32>` so that one 16-byte load fetches it:

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `positions` | `vec4<f32>` | $N$ | $x, y, z$, mass |
| `velocities` | `vec4<f32>` | $N$ | $v_x, v_y, v_z$, origin tag (for coloring) |
| `accel` | `vec4<f32>` | $N$ | $a_x, a_y, a_z$, potential $\phi$ |
| `keysA/B`, `valsA/B` | `u32` | $N$ each | Morton codes and body indices (sort ping-pong) |
| `children` | `vec2<u32>` | $N-1$ | left and right child of each internal node |
| `parents`, `visits` | `u32` | $2N-1$, $N-1$ | parent links; atomic visit counters |
| `nodeMass` | `vec4<f32>` | $2N-1$ | center of mass, total mass |
| `nodeBox` | 2 × `vec4<f32>` | $2N-1$ | box min + size $l$, box max + offset $\delta$ |

Internal nodes are numbered $0 \dots N-2$ with the root at 0, and the leaf of the $k$-th body in sorted
order is node $N-1+k$. All of this is about 184 bytes per body, or 48 MB at the maximum of 262,144
bodies (trails, when on, add 128 bytes per body). The uniforms are one 32-byte `SimParams` (count, tiles, $\Delta t$, $\varepsilon^2$, $\theta$, …)
and eight 16-byte `RadixParams` slots bound with a dynamic offset, one per sort pass.

### Brute force: tiles in workgroup memory

`cs_brute_force` gives each thread one body, in workgroups of 256 (the scheme of GPU Gems 3,
chapter 31). The workgroup walks over all bodies in **tiles of 256**: every thread loads one body of
the tile into a 4 KB `var<workgroup>` array, a `workgroupBarrier()` makes the tile visible, and then each
thread sums its body against all 256 entries. Each position is read from global memory once per
workgroup instead of once per thread, so the loop is limited by arithmetic, not memory. The last tile
is padded with zero-mass bodies, and threads past $N$ still help load tiles and reach every barrier,
because a barrier must be reached by the whole workgroup.

> **GPU:** a workgroup barrier inside a loop is only valid if the loop count is the same for every
> thread. Here it is the uniform `numTiles`, capped at 128 tiles ($N \le 32{,}768$).

### Barnes-Hut: the tree build, step by step

The 31 dispatches go into one compute pass (two on frames that measure GPU time). WebGPU guarantees
that the storage writes of one dispatch are visible to the next, so the stages need no fences between
them.

1. **Bounding box** (`cs_bounds_partial`, `cs_bounds_final`): a min/max reduction, 1,024 bodies per
   workgroup and then one workgroup over the partial results.
2. **Morton codes** (`cs_morton`): each position is normalized to the bounding cube, quantized to 1,024
   cells per axis, and the bits of $x, y, z$ are interleaved into a 30-bit code. Sorting by the code
   orders the bodies along a Z-order curve, so every subtree is a compact cluster in space.
3. **Radix sort** of (code, index) pairs: least significant digit first, 4-bit digits, 8 passes of
   histogram, scan and stable scatter (the same algorithm as the parallel sort example).
4. **Karras tree** (`cs_init_nodes`, `cs_build_tree`): with $N$ sorted leaves, a binary radix tree has
   exactly $N-1$ internal nodes, and Karras (2012) showed that node $i$ can find its key range and split
   point from the sorted codes alone, by an exponential search followed by binary searches. So all
   nodes are built **at once, in one dispatch**. Equal codes are made unique by appending the index.
5. **Bottom-up summaries** (`cs_bottom_up`): one thread per leaf climbs toward the root, computing each
   node's mass, center of mass, box, $l$ and $\delta$ from its two children.
6. **Traversal** (`cs_bh_force`): one thread per body, in Morton order, walks the tree depth-first with
   a private 64-entry stack and applies the opening test above.

Including the two kicks, a Barnes-Hut step is **33 dispatches** and a brute-force step is 3. Each step
after the first in a frame gets its own command buffer, so no single submission carries more than one
step of force work.

### Synchronization: why the second visitor proceeds

In step 5, a thread arriving at a parent does not know whether the other subtree is finished. Each
internal node has a counter, and the thread does `atomicAdd(&visits[p], 1u)`. The **first** thread to
arrive (old value 0) stops. The **second** one knows that both children are complete, since the first
thread got there only after finishing its own side. It combines the children and continues upward. Every
node is therefore completed exactly once, by its second visitor, after both of its subtrees.

> **GPU:** WGSL atomics are *relaxed*, and there is no device-scope memory fence. In principle, the
> counter does not order the sibling's node data before the second visitor reads it. So the node data
> in this pass is written and read with `atomicStore` / `atomicLoad` (via `bitcast`), which keeps every
> access device-coherent. Internal masses are also cleared before each build and the traversal skips
> zero-mass nodes, so a missed update would cost one force contribution for one step, never a hang.

### Bounded loops and CPU↔GPU traffic

Every loop in the shaders has a constant cap: the Karras searches stop after 32 iterations, the climb
after 64 levels, and the traversal after `MAX_VISITS` = 32,768 node visits. Each level of the tree
splits at a later bit of the 30-bit code plus the 18-bit index, so the depth is at most 49, below the
64-entry stack. The CPU uploads bodies only on a reset or a change of preset or count. While the simulation runs, the **only readback is 16 bytes** ($K$, $W$, interaction count), once
per second, from a two-dispatch energy reduction.

## Numerical behaviour

> **Numerics:** the force error of Barnes-Hut grows like $\theta^2$. Leapfrog's energy error
> oscillates with an amplitude proportional to $\Delta t^2$, without growing, as long as the forces are
> exact. With the tree, the force error sets a floor that a smaller $\Delta t$ cannot remove.

**Opening angle.** A node is replaced by a monopole at its center of mass. Expanding about the center of
mass removes the dipole term, so the leading error is the quadrupole, smaller than the monopole by
about $(l/d)^2 \le \theta^2$. The software-GPU run in CI on a Plummer sphere of 4,096 bodies confirms
this:

| $\theta$ | 0.3 | 0.5 | 0.7 | 1.2 |
|---|---|---|---|---|
| median relative force error | 1.5e-3 | 4.2e-3 | 8.6e-3 | 2.6e-2 |
| interactions per body | 997 | 487 | 292 | 134 |

The error grows by about 17× from $\theta$ = 0.3 to 1.2, where $(1.2/0.3)^2 = 16$, while the work falls
7×. Brute force needs 4,095 interactions per body. The 99th percentile is 6–7× the median; the worst
cases are mostly bodies near the center, whose net force nearly cancels, so that a small absolute error
is a large relative one.

**Softening** changes the physics on purpose. Forces below the scale $\varepsilon$ are weakened, and the
largest pair force is at $r = \varepsilon/\sqrt{2}$. Close encounters, which the fixed time step could
not resolve, then do not inject energy. A smaller $\varepsilon$ is more faithful to point masses but
needs a smaller $\Delta t$.

**Leapfrog** is second order, time-reversible and **symplectic**: it conserves, almost exactly, a
nearby "shadow" energy that differs from $E$ by $\mathcal{O}(\Delta t^2)$. So $E$ oscillates in a band instead of
drifting. In CI, brute force on 2,048 bodies over $t = 2$ gave a maximum $|\Delta E/E_0|$ of
3.4e-5, 5.5e-6, 1.3e-6 and 3.0e-7 for $\Delta t$ = 0.04, 0.02, 0.01 and 0.005: about 4× per halving.
Barnes-Hut breaks the argument, because its force is not the gradient of one fixed potential (nodes
switch between opened and closed as bodies move, and the tree is rebuilt every step). In the same CI run
with $\theta = 0.5$, the energy error stayed between 6.4e-4 and 6.9e-4 for every one of those steps.

**Single precision.** Everything is `f32`. The brute-force sum agrees with a float64 direct sum to about
$10^{-6}$ relative (99th percentile $2.6 \times 10^{-6}$ for 3,000 bodies). The tree with $\theta \to 0$
reaches the same level, $8.8 \times 10^{-7}$ median, although it adds in a different order. The energy
reduction is good to about $10^{-7}$, below every drift above.

## Why it maps well — or poorly — to a GPU

**Brute force maps very well.** The work is uniform, the inner loop has no branches (the self-interaction
is masked with a `select`), and tiling makes it compute-bound: one global load feeds 256 interactions, each about 20
floating-point operations. It is the textbook way to fill a GPU, and it is still $\mathcal{O}(N^2)$.

**Barnes-Hut traversal maps poorly, and works anyway.** Threads in the same SIMD group (warp or
wavefront) walk different paths through the tree. They diverge on every opening test, and their loads
from `nodeMass` and `nodeBox` are scattered gathers instead of coalesced reads. The 64-entry stack per
thread is private memory that competes with registers. Two choices limit the damage. Threads are
assigned in **Morton order**, so neighbours in a SIMD group are neighbours in space and take similar
paths. The opening test also reads only three `vec4`s per node: the center of mass and mass, and the
box corners with $l$ and $\delta$ packed into their spare `w` components.

**The tree build is where the GPU design matters.** The classic Barnes-Hut build inserts bodies one at
a time from the root, which is sequential, or which needs locks on a GPU. Karras's radix tree replaces
it with a sort and one fully parallel dispatch, with $\mathcal{O}(N)$ work and no locks. The bottom-up
pass needs only one atomic per node. The cost is a fixed overhead of about 30 small dispatches, several
of which (the scans, the final reductions) run as a single workgroup.

> **GPU:** that fixed overhead is why brute force wins at small $N$. On the CPU-emulated GPU in CI, the
> two cross between 8,192 and 16,384 bodies. Where they cross on a real GPU depends on its compute
> throughput, and on how much a launch costs. Measure it on yours.

## Try this

> **Try this:** in [Explore](#nbody), pick the Plummer sphere at 16,384 bodies and switch **Mode**
> between Barnes-Hut and brute force. Compare the GPU time and the energy drift line in the readout.

- Drag **θ** from 0.3 to 1.0 in [Explore](#nbody) and watch the interactions per body fall and the
  energy drift grow. Then run [θ vs error](#nbody/measure) to put numbers on both, and check the
  measured median against the $\theta^2$ reference line.
- Load **Cold collapse** with softening 0.005, then 0.1. The collapse at $t \approx \pi$ is the
  hardest moment for a fixed step, and the energy readout shows it.
- Run [Barnes-Hut vs brute force](#nbody/measure) to find the crossover $N$ on your GPU. Is it above
  or below the CI value? The tree-build share column shows how much of Barnes-Hut is fixed overhead.
- Run [time step vs energy error](#nbody/measure) with brute force and then Barnes-Hut. The first
  should fall about 4× per halving of $\Delta t$; the second should flatten at the tree's force error.
- Set the time step to 0.02 in [Explore](#nbody) with the Galaxy collision, then to 0.005. The
  pictures look alike; the energy readout does not.

## References

- J. Barnes and P. Hut, [A hierarchical O(N log N) force-calculation algorithm](https://doi.org/10.1038/324446a0),
  *Nature* 324, 446 (1986).
- T. Karras, [Maximizing parallelism in the construction of BVHs, octrees, and k-d trees](https://doi.org/10.2312/EGGH/HPG12/033-037),
  High-Performance Graphics (2012).
- C. Apetrei, [Fast and simple agglomerative LBVH construction](https://doi.org/10.2312/cgvc.20141206),
  Computer Graphics and Visual Computing (2014): the bottom-up pass with visit counters.
- L. Nyland, M. Harris and J. Prins, [Fast N-body simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda),
  *GPU Gems 3*, chapter 31 (2007): the tiled brute-force kernel.
- L. Hernquist, [Performance characteristics of tree codes](https://ui.adsabs.harvard.edu/abs/1987ApJS...64..715H),
  *ApJS* 64, 715 (1987): force errors and cost as a function of $\theta$.
- H. C. Plummer, [On the problem of distribution in globular star clusters](https://doi.org/10.1093/mnras/71.5.460),
  *MNRAS* 71, 460 (1911), and S. J. Aarseth, M. Hénon and R. Wielen,
  [A comparison of numerical methods for the study of star cluster dynamics](https://ui.adsabs.harvard.edu/abs/1974A&A....37..183A),
  *A&A* 37, 183 (1974): the Plummer sphere and how to sample it.
- A. Toomre, [On the gravitational stability of a disk of stars](https://ui.adsabs.harvard.edu/abs/1964ApJ...139.1217T),
  *ApJ* 139, 1217 (1964): the $Q$ criterion that sets the disk's velocity dispersion; and A. Toomre and
  J. Toomre, [Galactic bridges and tails](https://ui.adsabs.harvard.edu/abs/1972ApJ...178..623T),
  *ApJ* 178, 623 (1972), for the collision preset.
- E. Hairer, C. Lubich and G. Wanner, [Geometric numerical integration illustrated by the Störmer–Verlet method](https://doi.org/10.1017/S0962492902000144),
  *Acta Numerica* 12, 399 (2003): why leapfrog conserves energy so well.
- W. Dehnen and J. I. Read, [N-body simulations of gravitational dynamics](https://doi.org/10.1140/epjp/i2011-11055-3),
  *Eur. Phys. J. Plus* 126, 55 (2011): softening, time steps and tree codes.

## Read the code

- [`js/examples/nbody.js`](js/examples/nbody.js) has all the shaders, each with its algorithm in a
  comment. The file header has the uniform layouts and the bounds on every loop. Start at
  `encodeForces` for the dispatch order.
- [`js/teach/experiments/nbody.js`](js/teach/experiments/nbody.js) has the three Measure experiments,
  built on `measureForces` and `measureEnergyDrift` (compute-only, at the end of the example file).
- [`js/refs/nbody.js`](js/refs/nbody.js) has the float64 direct sum that the experiments compare against.
- [`test/helpers/nbody-refs.mjs`](test/helpers/nbody-refs.mjs) has CPU replicas of every shader (Morton
  codes, radix sort, Karras build, bottom-up, traversal), and [`test/nbody.test.mjs`](test/nbody.test.mjs)
  checks them.
- [`test/gpu/nbody.gpu.test.mjs`](test/gpu/nbody.gpu.test.mjs) and
  [`test/gpu/nbody-measure.gpu.test.mjs`](test/gpu/nbody-measure.gpu.test.mjs) run the shipped WGSL on a
  software GPU in CI. The second one is the source of the CI numbers on this page.
