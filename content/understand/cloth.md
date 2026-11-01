## Overview

A piece of fabric is modelled as an $n \times n$ grid of particles ($n$ = 32 to 128, up to 16,384
particles) joined by about $6n^2$ distance constraints: **structural** links to the four neighbours,
**shear** links across both cell diagonals, and **bend** links that skip one particle. Gravity, wind and
air drag move the particles; the constraints pull them back to their rest lengths.

The method is **position-based dynamics** (PBD). Instead of computing spring forces and integrating
them, which needs tiny time steps when the springs are stiff, PBD moves the particles freely and then
*projects* their positions back onto the constraints, a few times per step. It is unconditionally
stable and cheap, and the price is that stiffness depends on how many projection passes you can afford.

For a GPU programmer, the interesting part is how a solver that is sequential by nature (each projection
uses the result of the previous one) becomes a sequence of race-free parallel dispatches, through
**graph coloring**. For a computational scientist, it is what a fixed iteration budget does to accuracy:
the cloth is never quite inextensible, and the [Measure](#cloth/measure) tab shows by how much.

> **Note:** Units are metres and seconds. Each frame advances the cloth in fixed substeps of
> $h = 1/240$ s (4 per frame at 60 FPS, at most 8), so the simulation does not depend on the frame rate.

## The model

### Verlet integration

Each particle stores its position $\mathbf{x}$, its previous position $\mathbf{x}^{*}$ and an inverse
mass $w$ ($w = 0$ pins it). The velocity is implicit in the difference of the two positions:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} + d\,(\mathbf{x} - \mathbf{x}^{*}) + \mathbf{a}\,h^2,
\qquad d = e^{-0.4\,h}
$$

where $\mathbf{a}$ is gravity plus air drag against the (gusty) wind, split into a pressure term along
the cloth normal and a weaker term along the surface. The factor $d$ removes a little velocity each
substep, so a swinging cloth comes to rest. Speeds are clamped at 25 m/s as a safety net.

### Projecting a distance constraint

A link between particles $a$ and $b$ with rest length $L$ is the constraint

$$
C(\mathbf{x}_a, \mathbf{x}_b) = |\mathbf{x}_b - \mathbf{x}_a| - L = 0 .
$$

PBD moves both ends along the link, in proportion to their inverse masses, so that $C$ becomes zero
(scaled by a stiffness $k$):

$$
\Delta\mathbf{x}_a = +\,k\,\frac{w_a}{w_a + w_b}\,C\,\hat{\mathbf{d}}, \qquad
\Delta\mathbf{x}_b = -\,k\,\frac{w_b}{w_a + w_b}\,C\,\hat{\mathbf{d}}, \qquad
\hat{\mathbf{d}} = \frac{\mathbf{x}_b - \mathbf{x}_a}{|\mathbf{x}_b - \mathbf{x}_a|}
$$

This is a Newton step on $C$ projected onto the constraint gradient, and it conserves linear momentum
(the two corrections are opposite and mass-weighted). One pass over all the links is one **iteration**;
each substep runs `iterations` of them (8 by default, up to 30).

Because the passes compound, the per-pass stiffness is chosen so that the total after $N$ iterations is
the slider's value $k$: $k_{\text{pass}} = 1 - (1 - k)^{1/N}$. Shear and bend links use $0.5\,k$ and
$0.2\,k$, so the fabric resists stretching more than shearing or folding.

### Tethers, collisions and tearing

Three more effects close the model:

- **Long-range attachments** (Kim et al. 2012). A few Gauss-Seidel iterations cannot carry the weight
  of 128 rows up to the pins, so the links next to the pins would stretch to almost twice their length.
  Each particle is therefore also tied to its two nearest pins by a one-sided constraint: it may not be
  farther from a pin than in the flat rest layout, $|\mathbf{x} - \mathbf{x}_{\text{pin}}| \le L_{\text{pin}}$.
- **Collisions** with the sphere and the floor project particles back to the surface (plus a 1.5 cm
  margin) and apply Coulomb friction with $\mu = 0.6$, the penetration depth standing in for the normal
  impulse (Macklin et al. 2014).
- **Tearing**: a structural or shear link stretched beyond the threshold (default $1.8\,L$) breaks for
  good; bend links and anti-diagonals that span a broken link break with it.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `positions`, `previous` | `vec4<f32>` | $n^2$ each | $x, y, z$, inverse mass |
| `constraints` | 32-byte struct | about $6n^2$ | $a$, $b$, rest length, kind, two "parent" links |
| `broken` | `u32` | one per constraint | torn flags |
| `links` | 48-byte struct | $n^2$ | the particle's own link indices, its two tether pins and their rest distances |
| `normals` | `vec4<f32>` | $n^2$ | per-vertex normals (lighting and the next substep's wind) |
| `tornCount` | `atomic<u32>` | 1 | number of broken links |

Constraints are **sorted by color** on the CPU when the grid is built, and a small uniform per color holds
its `[start, count)` range; the solver dispatch selects it with a dynamic uniform offset. The 128-byte
`SimParams` uniform holds gravity, wind, the sphere, the per-pass stiffness of each kind, $h$, damping,
friction and the tear threshold. The whole state stays on the GPU; the only readbacks are one copy of the
positions per mouse grab (the pick runs on the CPU) and a 4-byte torn counter while tearing is on.

### Race-free Gauss-Seidel: graph coloring

A solver thread projects one constraint and writes **both** of its particles. Two constraints that share
a particle must therefore never run in the same dispatch: their read-modify-writes would race, and one
correction would be lost. The fix is to split the constraints into **colors**, sets in which no two
constraints share a particle, and dispatch the colors one after another.

The grid makes an exact coloring easy. Horizontal links starting in even columns never touch each other,
nor do those starting in odd columns; the same holds for vertical links by row, for both diagonals by
column, and for the bend links by $\lfloor i/2 \rfloor$ (they span two cells). Six kinds × two parities
gives **12 colors**, about $n^2/2$ constraints each.

> **GPU:** WebGPU guarantees that the storage writes of one dispatch are visible to the next dispatch in
> the same pass. So the 12 color dispatches need no barriers or atomics, and the solve is a true
> Gauss-Seidel sweep: every color sees the corrections of the colors before it.

The alternative, a **Jacobi** solve that projects all constraints at once and averages the corrections
per particle, needs atomics (or a second gather pass) and under-relaxation, and converges more slowly.

### Dispatch sequence

One substep is, in a single compute pass with workgroups of 64:

1. `cs_integrate` — one thread per particle, the Verlet step.
2. `cs_tether` — one thread per particle; it writes only its own particle, so it is race-free in one
   dispatch. It switches off once anything has torn.
3. `cs_solve` × iterations × 12 colors — one thread per constraint of the color.
4. `cs_collide` — one thread per particle, sphere and floor.

That is $3 + 12 \times \text{iterations}$ dispatches per substep, 99 at the default 8 iterations, so a
60 FPS frame with 4 substeps issues about 400 dispatches, followed by one `cs_normals` dispatch and the
render pass. A 128 × 128 cloth at 8 iterations projects about 97,000 constraints × 8 × 240 substeps
= 186 million constraint projections per second.

## Numerical behaviour

> **Numerics:** a Gauss-Seidel iteration removes only part of the constraint error, and the remaining
> error is what you see as stretch. More iterations make the cloth stiffer; so does a smaller substep.
> PBD's effective stiffness therefore depends on the iteration count and $h$, not only on $k$.

**Iterations vs stretch.** The error left after $N$ iterations decays roughly geometrically for a
single chain of links, but a hanging cloth is a long chain (the weight of every row below hangs from
each link), and the slowest modes converge very slowly. In CI, the shipped shader on a software GPU,
32 × 32 cloth hanging from two corners, after 2 s:

| iterations | 1 | 2 | 4 | 8 | 16 | 30 |
|---|---|---|---|---|---|---|
| 99th percentile overstretch | 0.78% | 0.56% | 0.39% | 0.29% | 0.17% | 0.10% |
| dispatches per substep | 15 | 27 | 51 | 99 | 195 | 363 |

So the error falls slower than $1/N$ at first and about as $1/N$ later, while the cost grows linearly in
$N$. The largest single link (0.92% at 1 iteration, 0.22% at 30) is noisier because the cloth is still
swinging. Without the tethers these numbers would be far larger near the pins.

**Agreement with the CPU replica.** The CI test drives a float64 replica of the five kernels
(`test/helpers/cloth-refs.mjs`) with the same uniforms and chunks. After 256 substeps the stretch
statistics agree with the shipped shader to 0.1%, while individual positions differ by a few millimetres:
a swinging cloth amplifies f32 rounding differences, which is a property of the dynamics, not a bug.

**Stability.** Verlet with position projection does not blow up when $k$ or the iteration count is large,
which is the reason PBD is used in games; the CPU tests also check that the damped cloth never gains
energy. The cost is accuracy: PBD dissipates energy in the projection, and the stiffness is not a
material constant. XPBD (Macklin et al. 2016) adds a compliance term that makes the stiffness
independent of the iteration count and the time step.

**Order independence.** Within one color the projections do not interact, so their order cannot matter.
The CPU tests check it: shuffling the order of the threads inside every color gives bit-identical
positions after ten frames.

## Why it maps well — or poorly — to a GPU

**Well:** every kernel is one thread per particle or per constraint with a fixed, small amount of
arithmetic, and the coloring turns a sequential solver into parallel dispatches with no atomics and
no locks. The data never leaves the GPU, and the per-substep work is a predictable
$\mathcal{O}(n^2 \times \text{iterations})$.

**Poorly:** the work per dispatch is small. At 64 × 64 a color holds about 2,000 constraints, 32
workgroups, which does not fill a large GPU, and there are about 100 such dispatches per substep. So the
cost of a substep is set more by the number of dispatches than by the arithmetic, and it grows more
slowly than $n^2$ until the grid is large. The constraint loads are also gathers (each thread reads two
particles at arbitrary indices), although the coloring by row and column keeps neighbouring threads on
neighbouring particles.

> **GPU:** on the CPU-emulated GPU in CI the opposite holds, since every invocation costs CPU time, and
> the time grows with the particle count. Run [Grid size vs time per substep](#cloth/measure) to see
> where your GPU sits.

## Try this

> **Try this:** in [Explore](#cloth), choose **Hang from two corners** and drag **Iterations** down to 1.
> The cloth stretches and sags visibly. Then run [Solver iterations vs stretch](#cloth/measure) and
> compare the 99th percentile with the $1/N$ reference line.

- Turn on **Wireframe** in [Explore](#cloth) at 1 iteration and look at the links next to the pins:
  the tethers keep the cloth from sagging, but not each link from stretching.
- Turn on **Tearing**, shift-drag the cloth and pull hard. Then raise **Iterations** and try again: a
  stiffer solve spreads the pull over more links. Does it take a harder pull to tear?
- Run [Grid size vs time per substep](#cloth/measure) at 4 and at 16 iterations. If the time per
  substep hardly changes from 32 × 32 to 64 × 64, your GPU is limited by dispatches, not by arithmetic.
- Load **Flag** at 20 m/s of wind and lower the stiffness. The flutter comes from the crosswind
  component of the gusts, which has a component along any cloth normal.

## References

- M. Müller, B. Heidelberger, M. Hennix and J. Ratcliff, [Position based dynamics](https://doi.org/10.1016/j.jvcir.2007.01.005),
  *J. Visual Communication and Image Representation* 18, 109 (2007).
- L. Verlet, [Computer "experiments" on classical fluids](https://doi.org/10.1103/PhysRev.159.98),
  *Phys. Rev.* 159, 98 (1967): the position Verlet integrator.
- T.-Y. Kim, N. Chentanez and M. Müller-Fischer, *Long range attachments — a method to simulate
  inextensible clothing in computer games*, ACM SIGGRAPH/Eurographics Symposium on Computer Animation
  (2012): the tethers.
- M. Macklin, M. Müller, N. Chentanez and T.-Y. Kim, [Unified particle physics for real-time applications](https://doi.org/10.1145/2601097.2601152),
  *ACM Trans. Graph.* 33, 153 (2014): friction from the penetration depth, and parallel constraint solving on the GPU.
- M. Macklin, M. Müller and N. Chentanez, [XPBD: position-based simulation of compliant constrained dynamics](https://doi.org/10.1145/2994258.2994272),
  Motion in Games (2016): stiffness independent of the iteration count.
- J. Bender, M. Müller, M. A. Otaduy, M. Teschner and M. Macklin, [A survey on position-based simulation methods in computer graphics](https://doi.org/10.1111/cgf.12346),
  *Computer Graphics Forum* 33, 228 (2014): Gauss-Seidel vs Jacobi solvers, and why the stiffness depends on the iterations.

## Read the code

- [`js/examples/cloth.js`](js/examples/cloth.js) has the compute shader (`cs_integrate`, `cs_tether`,
  `cs_solve`, `cs_collide`, `cs_normals`) with the algorithm in comments; the file header has the
  uniform layouts. `buildCloth` builds the 12-colored constraint list, `encodeCompute` the dispatch sequence.
- `measureCloth`, at the end of the same file, is the compute-only path the [Measure](#cloth/measure)
  tab uses; [`js/teach/experiments/cloth.js`](js/teach/experiments/cloth.js) has the two experiments,
  and [`js/refs/cloth.js`](js/refs/cloth.js) the stretch statistics they report.
- [`test/helpers/cloth-refs.mjs`](test/helpers/cloth-refs.mjs) is the CPU replica of the kernels;
  [`test/cloth.test.mjs`](test/cloth.test.mjs) checks the coloring, the uniforms, order independence,
  stability and tearing with it.
- [`test/gpu/cloth-measure.gpu.test.mjs`](test/gpu/cloth-measure.gpu.test.mjs) runs the shipped shader
  on a software GPU in CI against the replica; it is the source of the CI numbers on this page.
