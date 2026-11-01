## Overview

Ripples on a pond, sound in a room and light passing through two slits all obey the same equation: the
**wave equation**. This example solves it on a 512 × 384 grid with the oldest scheme there is, the
leapfrog finite difference, and shades the height field as a water surface. You can drop ripples, draw
walls, and load a double slit or an elliptical mirror.

For a GPU programmer it is a stencil computation that needs **two time levels** but only two buffers,
because each thread overwrites its own oldest value in place. For a computational scientist it is the
standard example of a **stability limit**: the scheme is exact in its energy bookkeeping for any time
step, yet it explodes as soon as the Courant number passes $1/\sqrt{2}$.

> **Note:** Units are grid cells and steps ($\Delta x = \Delta t = 1$), so the Courant number
> $C = c\,\Delta t / \Delta x$ is the wave speed in cells per step. Outside the grid and inside walls,
> $u = 0$.

## The model

### The damped wave equation

The height $u(x, y, t)$ satisfies

$$
\frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t} = c^2 \nabla^2 u
$$

where $\gamma$ is a small damping rate (the Damping slider sets $\gamma\,\Delta t$, 0.004 by default).
With $\gamma = 0$ the total energy, kinetic plus potential, is conserved.

### Leapfrog in time, five points in space

Centered differences in both time and space give the explicit scheme

$$
u^{n+1}_{ij} = 2 u^n_{ij} - u^{n-1}_{ij} + C^2 \left(u^n_{i-1,j} + u^n_{i+1,j} + u^n_{i,j-1} + u^n_{i,j+1} - 4 u^n_{ij}\right)
$$

and the damped version the shader uses, with $g = \gamma\,\Delta t / 2$:

$$
u^{n+1} = \frac{2 u^n - (1 - g)\, u^{n-1} + C^2 L u^n}{1 + g}
$$

where $L$ is the 5-point Laplacian. It is second-order accurate in space and time and needs no linear
solve: every new value depends only on old ones.

### Sources, walls and the sponge

- A **click** adds a "Mexican hat" ripple $(1 - s)\,e^{-s}$, $s = r^2 / 2\sigma^2$ with $\sigma = 4$ cells,
  to both time levels, so the drop starts at rest. Its integral over the plane is zero, so drops never
  raise the mean water level.
- **Walls** are cells held at $u = 0$, which reflect waves with a sign flip, like the outer edge.
- The **plane wave** of the double-slit preset drives column $x = 30$ as $0.6 \sin(2\pi n / 20)$: a
  period of 20 steps, so a wavelength of $20\,C$ cells.
- The **absorbing sponge** adds damping that grows quadratically over the last 24 cells before each
  edge, up to $g = 0.08$, so outgoing waves fade instead of reflecting.
- **Rain** hashes the step number into up to two random drops per step (3% chance each).

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `heightCur` / `heightPrev` | `f32` | $W \times H$ each | $u^n$ and $u^{n-1}$ (roles swap every step) |
| `walls` | `u32` | $W \times H$ | 1 = wall |
| `params` | `Params` | 16 slots of 256 bytes | grid size, $C^2$, $g$, step index, flags, cursor |

A three-level scheme seems to need three buffers, but it does not. Each thread reads $u^n$ at its cell
and four neighbours from `heightCur`, reads $u^{n-1}$ **at its own cell only** from `heightPrev`, and
writes $u^{n+1}$ back into that same cell of `heightPrev`. No other thread reads that cell of
`heightPrev` in this dispatch, so the in-place update is race-free. Then the bind groups swap roles:
the buffer just written is the new $u^n$.

### Dispatches and uniforms

`cs_step` runs in 8 × 8 workgroups, one thread per cell. A frame runs an optional `cs_edit` pass (drops
and wall strokes), then 1 to 16 `cs_step` dispatches in one compute pass, then the shading pass. Each
step needs its own step index (for the source phase and the rain hash), so the CPU writes one `Params`
copy per step into a 4 KB uniform buffer, 256 bytes apart, and each dispatch selects its copy with a
**dynamic offset**. All of a frame's parameters reach the GPU in one `writeBuffer`, and nothing comes
back.

> **GPU:** `minUniformBufferOffsetAlignment` is 256 bytes on most devices, so a 72-byte struct takes a
> 256-byte slot. Dynamic offsets are the cheap way to give consecutive dispatches different uniforms
> without creating a bind group for each.

### Shading

The render pass reconstructs a normal from central differences of the bilinearly interpolated heights,
and combines a color ramp of the height with diffuse and specular light. It reads the current height
buffer directly; there is no copy to a texture.

## Numerical behaviour

> **Numerics:** the undamped scheme conserves a discrete energy exactly, for any $C$. Below the CFL
> limit that energy is positive definite and bounds the solution; above it, it is not, and the
> checkerboard mode grows by a factor $|g| > 1$ per step.

**The CFL condition.** Insert a Fourier mode $u^n = g^n e^{i(k_x x + k_y y)}$ into the scheme. With
$\lambda = -4\left(\sin^2(k_x/2) + \sin^2(k_y/2)\right) \in [-8, 0]$ the eigenvalue of $L$,

$$
g^2 - (2 + C^2 \lambda)\, g + 1 = 0 .
$$

The two roots multiply to 1. They are complex with $|g| = 1$ (the mode oscillates) while
$|2 + C^2\lambda| \le 2$, and real with one root outside the unit circle otherwise. The worst mode is the
checkerboard, $\lambda = -8$, so the scheme is stable exactly when

$$
C \le \frac{1}{\sqrt{2}} \approx 0.707 .
$$

Physically, information in the scheme moves at most one cell per step in each direction, and the
domain of dependence of the numerical solution must contain that of the true one (Courant, Friedrichs
and Lewy, 1928). The Explore slider stops at 0.70; the Measure tab goes past the limit.

**Energy.** The quantity

$$
E^{n+1/2} = \Vert u^{n+1} - u^n \Vert^2 - C^2 \left\langle u^{n+1}, L u^n \right\rangle
$$

is the same at every step of the undamped scheme (a short algebraic identity), and it is positive for
every nonzero $u$ only when $C \le 1/\sqrt{2}$. The CPU suite measures its drift in `f32`: about
2e-7 over 1,500 steps at $C$ = 0.5 and 0.7. Damping and the sponge make it decrease monotonically.

**Dispersion.** The scheme is not exact for waves that span few cells. In one dimension its dispersion
relation is $\sin(\omega/2) = C \sin(k/2)$, so short waves travel slower than $c$. At $C = 0.5$ a
10-cell wave moves about 1.2% too slowly; the double-slit test measures a wavelength of 9.87 cells for the
nominal 10. The error also depends on direction (the 5-point Laplacian is not isotropic), which is why a
ripple drawn on a coarse grid turns slightly square.

**Blow-up beyond the limit.** In CI the shipped shader, started from one smooth drop on a 128 × 96 grid,
runs 1,000 steps at every $C \le 0.705$ with the amplitude never above its starting value and the energy
constant to 4e-7. Past the limit it fails quickly, and the further past, the sooner:

| $C$ | 0.705 | 0.71 | 0.72 | 0.75 |
|---|---|---|---|---|
| largest $\vert g \vert$ (von Neumann) | 1 | 1.20 | 1.46 | 2.00 |
| steps until $\max\vert u \vert > 1000\,\max\vert u_0 \vert$ | never (1,000 run) | 110 | 50 | 30 |

The smooth drop has essentially no checkerboard component; `f32` rounding, around $10^{-7}$ of the
amplitude, seeds it, and it then grows by $\vert g \vert$ per step: $1.2^{110} \approx 5 \times 10^{8}$.

## Why it maps well — or poorly — to a GPU

**It maps very well.** The update is uniform, branch-free for open water (walls, the sponge and the
source column are cheap, mostly-coherent branches), and reads neighbouring memory. It needs no
communication within a step, and only the implicit ordering between dispatches across steps.

**It is memory-bound.** A step reads five heights (four of them neighbours, usually from cache), one old
height and one wall flag, and writes one height: about 16 bytes of DRAM traffic for a dozen
floating-point operations. Like any explicit stencil, it runs at the GPU's memory bandwidth on large
grids. Storing the heights as `f16` could nearly double the speed of a bandwidth-bound step, at a cost
in accuracy.

> **GPU:** at 512 × 384 one step is 196,608 cells. On a desktop GPU that takes a few microseconds, less
> than the overhead of submitting work, which is why the example runs up to 16 steps per frame. The
> [grid-size experiment](#wave-equation/measure) shows where your GPU starts being limited by the cells.

## Try this

> **Try this:** load the **Ellipse** in [Explore](#wave-equation) and press Start. The ripple from one
> focus reflects off the wall and converges on the other focus after about $2a / C$ steps.

- Set the wave speed to 0.70 in [Explore](#wave-equation), the closest to the limit the slider allows.
  Nothing goes wrong. Then run [Courant number vs stability](#wave-equation/measure) to see what happens
  just past it, and how quickly.
- Load **Double slit** and count the bright fringes on the right. Turn the absorbing edges off and watch
  the reflections spoil the pattern.
- Set damping to 0 and turn rain on: the energy only goes up. With the default damping it reaches a
  balance between drops and decay.
- Run [grid size vs time per step](#wave-equation/measure) and compare with the
  [reaction-diffusion](#reaction-diffusion/measure) one: which stencil moves more bytes per cell?

## References

- R. Courant, K. Friedrichs and H. Lewy, [Über die partiellen Differenzengleichungen der mathematischen Physik](https://doi.org/10.1007/BF01448839),
  *Math. Ann.* 100, 32 (1928): the CFL condition.
- R. J. LeVeque, [Finite Difference Methods for Ordinary and Partial Differential Equations](https://doi.org/10.1137/1.9780898717839),
  SIAM (2007), chapter 10: von Neumann stability analysis and the leapfrog scheme.
- L. N. Trefethen, [Group velocity in finite difference schemes](https://doi.org/10.1137/1024010),
  *SIAM Review* 24, 113 (1982): why short waves travel at the wrong speed.
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*,
  Artech House (2005), chapters 2 to 4: the same scheme for electromagnetic waves, its dispersion and
  absorbing boundaries.

## Read the code

- [`js/examples/wave-equation.js`](js/examples/wave-equation.js): `cs_step` has the update and the CFL
  comment, the file header has the `Params` layout and its dynamic-offset slots, and `frame()` the
  dispatch order.
- [`js/teach/experiments/wave-equation.js`](js/teach/experiments/wave-equation.js) has the two Measure
  experiments, built on `measureWave` (compute-only, at the end of the example file).
- [`js/refs/wave.js`](js/refs/wave.js) has the discrete energy and the von Neumann growth factor.
- [`test/wave-equation.test.mjs`](test/wave-equation.test.mjs) checks a CPU replica against the energy,
  the growth factor, the ellipse travel time and the two-slit fringes;
  [`test/gpu/wave-equation-measure.gpu.test.mjs`](test/gpu/wave-equation-measure.gpu.test.mjs) runs the
  shipped shader on a software GPU in CI and is the source of the CFL numbers on this page.
