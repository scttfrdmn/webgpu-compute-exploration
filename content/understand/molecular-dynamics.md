## Overview

Molecular dynamics (MD) follows atoms through time by integrating Newton's equations with forces from an
interatomic potential. This example uses the **Lennard-Jones** potential, the standard model of a noble
gas such as argon: atoms repel strongly at short range and attract weakly beyond. Up to 5,000 atoms move in
a periodic cubic box of side 30, and every force is computed on the GPU, one thread per atom.

For a GPU programmer it is an all-pairs force kernel with periodic boundaries, like the brute-force half of
the N-body example but without its tiling. For a computational scientist it is a chance to look at three
classic MD ingredients and their honest simplifications: a truncated potential, a first-order symplectic
integrator, and a velocity-rescaling "thermostat" that is not a canonical one.

> **Note:** units are reduced Lennard-Jones units: $\varepsilon = \sigma = m = 1$, so lengths are in
> $\sigma$, energies in $\varepsilon$ and times in $\sigma\sqrt{m/\varepsilon}$ (about 2 ps for argon).

## The model

### The Lennard-Jones potential

Two atoms a distance $r$ apart have potential energy

$$
U(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]
$$

with its minimum $U = -\varepsilon$ at $r_{\min} = 2^{1/6}\sigma \approx 1.12\sigma$. The force on atom $i$
from atom $j$, with $\mathbf{r} = \mathbf{x}_i - \mathbf{x}_j$, is

$$
\mathbf{F}_{ij} = -\nabla_i U = \frac{24\varepsilon}{r^2} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right] \mathbf{r}
$$

which is repulsive (along $+\mathbf{r}$) inside $r_{\min}$ and attractive outside. The shader sets the force
to zero beyond the cutoff $r_c = 2.5\sigma$, where $|U| < 0.02\varepsilon$, and below $r = 0.1\sigma$, a guard
against division by zero that no physical trajectory reaches.

### Periodic boundaries

The box has side $L = 30$ and wraps in all three directions. Each pair uses the **minimum image**: every
component of $\mathbf{r}$ is folded into $[-L/2, L/2]$, so an atom interacts with the nearest periodic copy
of its partner. That is correct only because $r_c < L/2$. After each step, positions that left the box
are wrapped back by one box length.

### Integration

Each step of size $\Delta t = 0.016$ is **semi-implicit (symplectic) Euler**:

$$
\mathbf{v}_{n+1} = \mathbf{v}_n + \Delta t\, \mathbf{F}(\mathbf{x}_n), \qquad
\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t\, \mathbf{v}_{n+1}
$$

Updating the velocity first and moving with the *new* velocity is what makes the method symplectic. The
plain explicit Euler method, which moves with the old velocity, gains energy steadily on every
oscillation; this one does not.

### The thermostat

After the integration step, the shader rescales each atom's velocity toward a target speed
$v_* = \sqrt{0.01\,T}$, where $T$ is the Temperature slider:

$$
\mathbf{v} \leftarrow \mathbf{v} \left[ 1 - 0.01 \left( 1 - \frac{v_*}{|\mathbf{v}|} \right) \right]
$$

so every atom's speed relaxes toward $v_*$ by 1% per step. This keeps the simulation lively, but it is
**not** a thermostat in the statistical-mechanics sense: it pushes every atom toward the *same* speed
instead of a Maxwell-Boltzmann distribution, and it does not conserve total momentum. The kinetic
temperature it aims at is $T^* = v_*^2/3 = 0.01\,T/3$, which is 0.1 at the default $T = 30$.

### The initial state

`initializeAtoms` places the atoms on a simple cubic lattice that fills the box,
$\lceil N^{1/3} \rceil$ sites per side, and gives each velocity component a uniform random value in
$\pm 0.005\,T$. At the default 1,000 atoms the lattice spacing is 3, wider than the cutoff, so nothing
interacts at first: the density is $1000/30^3 \approx 0.037\,\sigma^{-3}$, a dilute gas, and atoms only
feel each other when they meet.

## Mapping to the GPU

### Buffers and layout

| Buffer | WGSL | Contents |
|---|---|---|
| `atomBuffers[0]`, `[1]` | `array<Atom>` | `Atom { position: vec3<f32>, velocity: vec3<f32> }`, 32 bytes |
| `paramsBuffer` | `Params` (16 bytes) | `numAtoms: u32`, `deltaTime`, `boxSize`, `temperature` |
| `uniformBuffer` | `Uniforms` | the camera's view-projection matrix, for rendering |

A `vec3<f32>` is aligned to 16 bytes, so each `Atom` holds two padding floats: 8 floats per atom on the CPU
side. The count is written through a `Uint32Array` view because it is a `u32`.

### One thread per atom, all pairs

The compute shader runs in workgroups of 64, $\lceil N/64 \rceil$ of them. Each invocation loads its atom,
loops over all $N$ atoms (skipping itself), applies the minimum image and the cutoff, and sums the
forces; then it integrates, wraps, applies the thermostat and writes the new atom to the *other* buffer.
The two bind groups ping-pong between the buffers, so a step reads only old positions, which is what the
integrator requires. One step costs $N(N-1)$ pair evaluations, about $10^6$ at the default.

> **GPU:** every invocation reads every atom from global memory, $32N$ bytes per atom per step. The N-body
> example's brute-force kernel instead loads tiles of 256 bodies into `var<workgroup>` memory and reuses
> each load 256 times. The same tiling would apply here unchanged, and a **cell list** (bins of side
> $r_c$) would cut the work from $N^2$ to about $N$ times the 50–60 neighbours an atom has in a liquid.

### Per frame

`animate()` runs one compute step and one render pass per animation frame. The render pass draws each atom
as an instanced icosahedron (60 vertices, colored by speed) with a depth buffer, then the box's faint faces
and glowing edges. Nothing is read back: positions stay on the GPU between the compute and the vertex
shader.

## Numerical behaviour

> **Numerics:** semi-implicit Euler is first order and symplectic. In a closed system (no thermostat), its
> energy error stays bounded and oscillates instead of drifting, with an amplitude proportional to
> $\Delta t$. Leapfrog, used by the N-body example, has the same structure with an error proportional to
> $\Delta t^2$.

The shader does not compute the energy, so the Measure tab reads the atoms back and computes it on the CPU
in float64. Two details matter. The truncated force is the gradient of the **shifted** potential
$U(r) - U(r_c)$ for $r < r_c$ (and 0 beyond), so that is the potential whose sum with the kinetic energy is
conserved; adding the unshifted $U$ would make the energy jump every time a pair crosses $r_c$. And the
force itself still jumps at $r_c$ (by $|F(r_c)| \approx 0.039$), which adds a small error each time a pair
crosses it, of the same order in $\Delta t$.

The energy experiment uses a denser box than Explore ($\rho = 0.8\,\sigma^{-3}$, a liquid density, so every
atom has neighbours) and the **NVE variant** of the shader: the same source with the thermostat block
removed. On a software GPU in CI, 512 atoms over $t = 4$ gave:

| $\Delta t$ | 0.016 | 0.008 | 0.004 | 0.002 |
|---|---|---|---|---|
| max $\vert \Delta E / E_0 \vert$ | 4.8e-3 | 2.6e-3 | 1.3e-3 | 6.6e-4 |

a factor of 2 per halving: first order. The simple cubic start is mechanically unstable, so the lattice
melts and the kinetic energy grows about 47 times, while the total energy stays within those bounds. With
the thermostat on, the energy changes by 7–10% over $t = 2$ whatever $\Delta t$ is: it follows the
thermostat.

**Single precision.** Positions within $\pm 15$ carry about $10^{-6}$ absolute rounding, far below the
integrator's error. One GPU step agrees with a float64 step to $1.2 \times 10^{-7}$ in position. The shader
computes $(\sigma/r)^6$ with `pow`, which GPUs evaluate as $2^{6 \log_2 x}$, slightly less accurate than
three multiplications but not visibly so here.

## Why it maps well — or poorly — to a GPU

**The force loop maps well.** Every atom does the same $N$ iterations, the loads of `atomsIn[j]` are the
same address for all threads of a workgroup at once (a broadcast), and there is no write conflict, since
each thread writes only its own atom. Thousands of atoms are enough to fill a GPU.

**The algorithm does not scale.** All-pairs is $\mathcal{O}(N^2)$ although each atom has only a few dozen
neighbours inside $r_c$; in the example's dilute box most of the loop computes distances to atoms that are
too far away. Production MD codes (GROMACS, LAMMPS, OpenMM) use cell lists and neighbour lists, built on the
GPU, and are $\mathcal{O}(N)$. They also use higher-order integrators (velocity Verlet) and thermostats that
sample the canonical ensemble (Nosé-Hoover, Langevin, or Bussi's stochastic rescaling).

> **GPU:** the integration and the force sum are in the same kernel, which is possible only because
> each atom needs nothing but old positions. A two-kernel design (forces, then integration) would be needed
> for velocity Verlet, which uses the forces at the new positions.

## Try this

> **Try this:** in [Explore](#molecular-dynamics), set Temperature to 1 and press Start. The thermostat
> slows every atom toward $v_* = 0.1$; watch for atoms that meet and stay together.

- Set Atoms to 5,000 and compare the motion with 1,000. The lattice spacing is then $30/18 = 1.67$,
  inside the cutoff and beyond $r_{\min}$, so every atom feels its lattice neighbours' attraction from the
  start; the perfect lattice is in balance, but the random velocities soon break it up.
- Run [time step vs energy error](#molecular-dynamics/measure) with the thermostat off. The maximum error
  should halve with $\Delta t$; compare with the N-body example's [leapfrog](#nbody/measure), where it
  falls 4×.
- Run it again with the thermostat on: the energy no longer has anything to do with $\Delta t$.
- Run [atoms vs time per step](#molecular-dynamics/measure) and compare with the $N^2$ line.

## References

- J. E. Lennard-Jones, [On the determination of molecular fields. II.](https://doi.org/10.1098/rspa.1924.0082),
  *Proc. R. Soc. Lond. A* 106, 463 (1924).
- L. Verlet, [Computer "experiments" on classical fluids. I.](https://doi.org/10.1103/PhysRev.159.98),
  *Phys. Rev.* 159, 98 (1967): Lennard-Jones argon, the Verlet integrator and neighbour lists.
- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford (2017): cutoffs,
  shifted potentials, periodic boundaries and thermostats.
- D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed., Academic Press (2002).
- G. Bussi, D. Donadio and M. Parrinello, [Canonical sampling through velocity rescaling](https://doi.org/10.1063/1.2408420),
  *J. Chem. Phys.* 126, 014101 (2007).
- E. Hairer, C. Lubich and G. Wanner, [Geometric numerical integration illustrated by the Störmer–Verlet method](https://doi.org/10.1017/S0962492902000144),
  *Acta Numerica* 12, 399 (2003): symplectic Euler and why its energy error stays bounded.

## Read the code

- [`js/examples/molecular-dynamics.js`](js/examples/molecular-dynamics.js) has the compute shader (forces,
  integration, thermostat), its NVE variant, the render and box shaders, and `measureMD`, the compute-only
  version the Measure tab uses.
- [`js/teach/experiments/molecular-dynamics.js`](js/teach/experiments/molecular-dynamics.js) has the two
  experiments; [`js/refs/molecular-dynamics.js`](js/refs/molecular-dynamics.js) has the float64 energy and
  integrator step.
- [`test/molecular-dynamics.test.mjs`](test/molecular-dynamics.test.mjs) checks a CPU replica of the
  shader (it found the force sign error fixed in #64); [`test/gpu/molecular-dynamics-measure.gpu.test.mjs`](test/gpu/molecular-dynamics-measure.gpu.test.mjs)
  runs the shipped kernels on a software GPU in CI and is the source of the numbers above.
