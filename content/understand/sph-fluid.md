## Overview

Smoothed Particle Hydrodynamics (SPH) simulates a fluid with particles instead of a grid. Each particle
carries a bit of mass; density, pressure and viscous forces at a particle are weighted sums over its
neighbours, with a smooth **kernel** of radius $h$ as the weight. The method was invented for astrophysics
(Lucy; Gingold and Monaghan, 1977) and brought to interactive graphics by Müller, Charypar and Gross
(2003), whose formulation this example follows.

The example is a 2D **dam break**: a square block of 2,000 particles falls under gravity, hits the floor of
a $1000 \times 700$ pixel box and splashes. Every particle's density and force is computed on the GPU in six
substeps per frame. For a GPU programmer it is three dependent passes per substep and an all-pairs
neighbour search. For a computational scientist it is a **weakly compressible** fluid, and the example's
stiffness lets it compress far more than water would, which the Measure tab quantifies.

## The model

### Kernels and density

All quantities live on particles of equal mass $m$. The density at particle $i$ is a kernel-weighted sum,
itself included:

$$
\rho_i = \sum_j m\, W_{\text{poly6}}(r_{ij}, h), \qquad
W_{\text{poly6}}(r, h) = \frac{4}{\pi h^8} \left(h^2 - r^2\right)^3 \quad (r < h)
$$

The prefactor normalizes the kernel in 2D, $\int W\, dA = 1$. The example uses $h = 16$ pixels and starts
the particles on a square lattice of spacing $h/2$, with $m$ chosen (by `latticeDensity()`) so that an
interior particle has exactly the rest density $\rho_0 = 1$.

### Pressure

Pressure follows from density through a stiff equation of state, clamped at zero:

$$
p_i = \max\big(k\,(\rho_i - \rho_0),\; 0\big), \qquad k = 10^6
$$

A fluid that is compressed pushes back; a fluid that is stretched does not pull. The clamp avoids the
clumping ("tensile instability") that negative pressure causes in SPH, at the price of having no surface
tension at all.

### Forces and integration

Following Müller et al., the pressure force uses the gradient of the "spiky" kernel (which does not vanish
at $r = 0$, so close particles keep repelling), with the pressures averaged so that the pair forces are
symmetric; viscosity uses the Laplacian of a third kernel:

$$
\mathbf{f}_i = -\sum_{j \ne i} m \frac{p_i + p_j}{2\rho_j} \nabla W_{\text{spiky}}(\mathbf{r}_{ij}, h)
+ \mu \sum_{j \ne i} m \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W_{\text{visc}}(r_{ij}, h)
+ \rho_i\, \mathbf{g}
$$

with $\nabla W_{\text{spiky}} = -\frac{30}{\pi h^5}(h - r)^2\,\hat{\mathbf{r}}$ and
$\nabla^2 W_{\text{visc}} = \frac{40}{\pi h^5}(h - r)$, again in their 2D normalizations. These are force
*densities*; the acceleration is $\mathbf{a}_i = \mathbf{f}_i / \rho_i$, so the pressure and viscous
accelerations between two particles are equal and opposite and the method conserves momentum. Each
substep of $\Delta t = 1/360$ s then does semi-implicit Euler (velocity first, then position with the new
velocity). A particle that crosses a wall is put back on it and its normal velocity is reversed and damped
to 30%. The Gravity and Viscosity sliders set $g$ (default 500 px/s², downward, which is $+y$ on screen)
and $\mu$ (default 300).

## Mapping to the GPU

### Buffers

| Buffer | WGSL | Contents |
|---|---|---|
| `particleBuffers[0]`, `[1]` | `array<Particle>`, 32 bytes each | position, velocity (`vec2<f32>`), density, pressure, padding |
| `forcesBuffer` | `array<vec2<f32>>` | the force density of each particle |
| `paramsBuffer` | `Params` | $N$, $h$, $\rho_0$, $k$, $m$, $\mu$, $g$, $\Delta t$, the box, the damping |

The buffers are sized for twice the initial count, so **Add Water** can append blocks of 100 particles
(up to 4,000) without reallocating.

### Three passes per substep

Each substep is three compute passes in workgroups of 64, one invocation per particle:

1. **Density** reads buffer 0 and writes buffer 1: every particle sums Poly6 over all $N$ particles and
   stores $\rho_i$ and $p_i$.
2. **Forces** reads buffer 1 (it needs every neighbour's density and pressure, which only exist after pass
   1 is complete) and writes `forcesBuffer`.
3. **Integrate** reads buffer 1 and the forces, and writes the moved particles back to buffer 0.

So the state is in buffer 0 between substeps, and all 18 passes of a frame go into one command buffer with
one submit. The pass boundaries are the synchronization: WebGPU makes each dispatch's writes visible to
the next, which is exactly the global barrier SPH needs between "everyone's density" and "everyone's force".
A `GpuTimer` times each pass, and the readout splits the GPU time per frame into density, forces,
integrate and render.

> **GPU:** both neighbour loops run over **all** $N$ particles and test the distance, although only a
> handful lie within $h$ (8 on the initial lattice of spacing $h/2$, a few more when compressed). That is $2N^2$
> distance tests per substep, 48 million per frame at 2,000 particles. A uniform grid of cell size $h$,
> rebuilt each substep with a counting sort, reduces the search to the 9 surrounding cells; the readout's
> note about spatial hashing refers to this.

### Rendering

One instanced draw renders each particle as a small screen-space quad colored by speed, reading the
positions straight from the storage buffer; nothing is copied back to the CPU.

## Numerical behaviour

> **Numerics:** with $p = k(\rho - \rho_0)$, the speed of sound is $c = \sqrt{dp/d\rho} = \sqrt{k}$:
> 1,000 px/s for the example. Weakly compressible SPH keeps density variations near 1% only if
> $c \ge 10\, v_{\max}$ (Monaghan, 1994), because they scale like the Mach number squared. The splash
> reaches about 1,000 px/s, so the example's fluid runs at Mach 1 and compresses a lot.

**Compression.** In CI, the shipped shaders on a software GPU with 500 particles reached a peak density of
$\rho/\rho_0 - 1$ = 5.4, 1.8, 0.79 and 0.34 at $k$ = 2.5e5, 1e6, 4e6 and 1.6e7: at the default $k = 10^6$
the impact compresses the fluid to almost 3× its rest density. Each 4× in $k$ (2× in $c$) cuts the peak by
2–3×, less than the 4× that $1/c^2$ predicts for a smooth flow, because the impact is not one.

**Stability.** An explicit scheme needs the sound to cross less than a kernel radius per step, the CFL
condition $c\,\Delta t / h \le C$, with $C$ around 0.4 in the SPH literature (the exact value depends on
the scheme and the viscosity). The $k$ values above give $c\,\Delta t/h$ = 0.09, 0.17, 0.35 and 0.69, all
stable, and $k = 6.4 \times 10^7$ gives 1.39, which blew up in CI: particles reached 27,000 px/s against at
most 2,400 in the stable runs. Keeping the splash within about 1% of the rest density would need
$c \approx 10\,v_{\max} \approx 10^4$ px/s, so $k = 10^8$, and at $c\,\Delta t / h = 0.4$ some 26 substeps per
frame instead of 6. That trade-off is why interactive fluids are compressible, and why incompressible
solvers (PCISPH, IISPH, position-based fluids) exist.

**Viscosity** damps the splash, and the wall damping removes energy at every bounce, so the fluid comes to
rest. The explicit viscous term has its own limit, $\Delta t \le 0.125\, h^2 \rho / \mu$, about 0.1 s
at the defaults, far from binding.

**Precision.** The density pass is a sum of positive terms, and in CI it agreed with a float64 sum over the
same positions to $2.5 \times 10^{-7}$ relative, at the start and mid-splash. Single precision is not a
limitation here; the time step and the stiffness are.

## Why it maps well — or poorly — to a GPU

**The passes map well.** Every particle does the same work, the passes need no atomics, and their order
gives the global synchronization for free. Particles stay on the GPU from frame to frame.

**The neighbour search does not.** $\mathcal{O}(N^2)$ distance tests for $\mathcal{O}(N)$ useful
interactions is why the example stops at 4,000 particles. A grid-based neighbour search is the standard fix
and is itself a nice GPU exercise: histogram the particles into cells (atomics), prefix-sum the counts,
scatter the particles into cell order (the same steps as the parallel sort example), then loop over 9 cells
per particle. The cost per substep becomes $\mathcal{O}(N)$, and sorting particles by cell also makes the
neighbour loads coherent in memory.

## Try this

> **Try this:** in [Explore](#sph-fluid), press Start and watch the block hit the floor. Then set
> Viscosity to 1 and Reset: the splash is wilder and takes longer to settle.

- Set Gravity to 0 and press Reset, then Start: the block stays where it is. Its interior is at rest
  density (zero pressure, up to rounding) and its edges are below it (pressure clamped to zero), so there
  is no force to speak of. A fluid whose pressure only pushes has no cohesion and no surface tension.
- Run [gas constant vs compression](#sph-fluid/measure). Find the stiffest $k$ that is still stable, and
  compare its $c\,\Delta t / h$ with the CFL limit.
- Run [particles vs time per substep](#sph-fluid/measure) and check which pass dominates and how close the
  wall time follows the $N^2$ line.

## References

- M. Müller, D. Charypar and M. Gross, [Particle-based fluid simulation for interactive applications](https://doi.org/10.2312/SCA03/154-159),
  Symposium on Computer Animation (2003): the kernels and force formulation used here.
- J. J. Monaghan, [Simulating free surface flows with SPH](https://doi.org/10.1006/jcph.1994.1034),
  *J. Comput. Phys.* 110, 399 (1994): weak compressibility and the speed-of-sound rule.
- J. J. Monaghan, [Smoothed particle hydrodynamics](https://doi.org/10.1088/0034-4885/68/8/R01),
  *Rep. Prog. Phys.* 68, 1703 (2005): a review of the method and its stability.
- M. Ihmsen, J. Orthmann, B. Solenthaler, A. Kolb and M. Teschner, [SPH fluids in computer graphics](https://doi.org/10.2312/egst.20141034),
  Eurographics State of the Art Reports (2014): neighbour search, incompressibility and boundaries.
- R. A. Gingold and J. J. Monaghan, [Smoothed particle hydrodynamics: theory and application to non-spherical stars](https://doi.org/10.1093/mnras/181.3.375),
  *MNRAS* 181, 375 (1977).

## Read the code

- [`js/examples/sph-fluid.js`](js/examples/sph-fluid.js) has the three compute shaders, the render shader,
  `step()` (the 18 passes of a frame) and the compute-only `measureSPHFrame` and `runSPH`.
- [`js/teach/experiments/sph-fluid.js`](js/teach/experiments/sph-fluid.js) has the two experiments;
  [`js/refs/sph.js`](js/refs/sph.js) has the float64 density sum.
- [`test/sph-fluid.test.mjs`](test/sph-fluid.test.mjs) checks a CPU replica of the three passes (kernel
  normalizations, the rest-density lattice, stability at the slider extremes);
  [`test/gpu/sph-fluid-measure.gpu.test.mjs`](test/gpu/sph-fluid-measure.gpu.test.mjs) runs the shipped
  shaders on a software GPU in CI and is the source of the numbers above.
