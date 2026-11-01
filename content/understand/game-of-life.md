## Overview

Conway's Game of Life (1970) is a cellular automaton: a grid of cells, each alive or dead, updated in
lockstep by one local rule. A live cell with two or three live neighbours survives; a dead cell with
exactly three is born; everything else dies or stays dead. From that rule come gliders that travel,
oscillators, guns that emit gliders forever, and patterns that can compute anything a Turing machine can.

The example runs the rule on a $200 \times 150$ grid that wraps around at the edges (a torus), one
compute dispatch per generation, and draws the grid with a fragment shader. For a GPU programmer it is
the simplest **stencil** with **ping-pong buffers**: every cell reads its neighbours from one buffer and
writes to another. For a computational scientist it is the discrete cousin of a PDE solver: a local
update rule, a fixed grid, boundary conditions, and emergent behaviour that no single cell "knows" about.

## The model

Let $s_t(x, y) \in \{0, 1\}$ be the state of cell $(x, y)$ at generation $t$, and

$$
n_t(x, y) = \sum_{(dx, dy) \ne (0, 0)} s_t\big((x + dx) \mathrm{\ mod\ } W,\; (y + dy) \mathrm{\ mod\ } H\big), \qquad dx, dy \in \{-1, 0, 1\}
$$

the number of live cells among its eight neighbours (the **Moore neighbourhood**), with indices taken
modulo the grid size so that the edges wrap. The rule, called B3/S23 (born with 3, survives with 2 or 3), is

$$
s_{t+1}(x, y) = \begin{cases} 1 & \text{if } n_t = 3, \text{ or } s_t = 1 \text{ and } n_t = 2 \\ 0 & \text{otherwise} \end{cases}
$$

Every cell of generation $t + 1$ depends only on generation $t$. That is what makes the update parallel,
and also why it needs two copies of the grid: a cell that overwrote its own state would corrupt its
neighbours' counts in the same generation.

The seed patterns show the main kinds of behaviour: still lifes and oscillators (the **pulsar**, period 3;
the **pentadecathlon**, period 15), spaceships (the **glider** moves one cell diagonally every 4
generations, the **lightweight spaceship** two cells every 4), the **Gosper glider gun**, which emits a
glider every 30 generations, and **methuselahs** such as the R-pentomino, the acorn and diehard, small
patterns that evolve for hundreds or thousands of generations (diehard vanishes after 130).

## Mapping to the GPU

### Buffers and threads

| Buffer | WGSL | Contents |
|---|---|---|
| `gridBuffers[0]`, `gridBuffers[1]` | `array<u32>` | $W \cdot H$ cells, 0 or 1, row-major |
| `dimensionsBuffer` | `var<uniform> dimensions: vec2<u32>` | $W, H$ |

The compute shader runs in $8 \times 8$ workgroups, one invocation per cell; for $200 \times 150$ the host
dispatches $25 \times 19$ workgroups (30,400 invocations, of which 400 fall outside the grid and return
at once). Each invocation reads its own cell and its eight neighbours through `getCell`, which wraps the
coordinates with `(x + width) % width`, counts the live ones, applies the rule and writes one `u32`.

### Ping-pong and synchronization

There are two bind groups: one reads buffer 0 and writes buffer 1, the other the reverse. `step()` uses
`computeBindGroups[currentBuffer]` and then flips `currentBuffer`, so the newest generation is always in
`gridBuffers[currentBuffer]`, which `render()` reads. No barrier is needed inside a generation, because
no invocation reads what another writes. Between generations, WebGPU guarantees that a dispatch sees the
storage writes of every earlier dispatch, so even many generations in one compute pass (as the Measure
tab runs them) need no explicit synchronization.

> **GPU:** the interactive example submits one generation per animation frame (at most the Speed setting,
> 1 to 60 per second), so the GPU idles almost all the time; a $200 \times 150$ generation takes
> microseconds. The Measure tab runs 100 generations per compute pass to see what the stencil itself costs.

### Rendering and interaction

The render pass draws one full-screen quad. The fragment shader converts its interpolated texture
coordinate to a cell index, reads that cell from the storage buffer and outputs green or dark gray; each
cell covers $4 \times 4$ pixels of the $800 \times 600$ canvas. Row 0 of the grid is drawn at the bottom,
because the coordinate is 0 at clip-space $y = -1$. Clicking a cell copies the whole grid back to the CPU,
flips one cell and uploads the grid again, a round trip that is fine for a click but would be far too slow
per frame.

## Numerical behaviour

> **Numerics:** there is nothing to round. The state is exact integers, so the GPU and a CPU
> implementation must agree **cell for cell**, forever. Any difference at all is a bug, and the Measure
> tab checks for exactly that.

**Boundary conditions matter.** On a torus, a glider that leaves on the right comes back on the left and
can collide with whatever is in its way; on an infinite plane it would leave forever. The R-pentomino
settles after 1,103 generations on the infinite plane, but on the $200 \times 150$ torus its escaping
gliders come back into the debris, and in a CPU run it was still changing (613 live cells) after 3,000.
Every finite grid is an approximation of the infinite one, exactly as a periodic domain is for a fluid
simulation.

**Random soups thin out.** The Random button sets each cell alive with probability 0.3. Crowded regions
die immediately, sparse ones get births, and after a few hundred generations most of the grid is empty
apart from scattered still lifes and blinkers, the "ash". In CI, a $256 \times 256$ soup went from
a live fraction of 0.30 to 0.34 after one generation, 0.21 after 10, 0.095 after 100 and 0.046 after
1,000, with the GPU and CPU grids identical at every checkpoint.

**The state space is finite**, so every run on a fixed grid eventually repeats; how long that takes is
unpredictable, which is the point of the methuselahs.

## Why it maps well — or poorly — to a GPU

**It maps very well.** Every cell is independent within a generation, the work per cell is identical
(nine loads, a few comparisons, one store), and neighbouring invocations read overlapping neighbourhoods,
which the caches serve. It is memory-bound: about 9 loads per cell update for almost no arithmetic.

**It wastes memory.** Each 1-bit cell occupies a 32-bit `u32`, so the grid takes 32 times more memory and
bandwidth than it needs. High-performance Life codes pack 32 or 64 cells per word and update them with
bitwise operations (bit-sliced adders count the neighbours of a whole word at once), or skip regions that
did not change. The example keeps one cell per invocation because it is the clearest mapping.

**It is far too small to matter.** 30,000 cells are a fraction of the parallelism of any GPU. The
interesting question is how large the grid must be before the GPU is busy, which is the throughput
experiment.

## Try this

> **Try this:** in [Explore](#game-of-life), load the **Gosper Glider Gun** and press Start. Count 30
> generations per glider, and watch what happens when the first gliders wrap around the torus and come
> back toward the gun.

- Load **Diehard** and Step through it: it should vanish completely after 130 generations. Load the
  **R-pentomino** and compare: five cells that keep changing for more than a thousand generations.
- Run [random soup vs the CPU rule](#game-of-life/measure). The mismatch column must be 0 everywhere; the
  live fraction shows how quickly a soup turns to ash.
- Run [grid size vs cell updates per second](#game-of-life/measure) and find the size at which the
  throughput levels off. How many times larger than the example's 30,000 cells is it?
- Press Random, then Start with Speed at 60. The GPU work per frame is tiny: what limits the frame rate?

## References

- M. Gardner, [Mathematical Games: The fantastic combinations of John Conway's new solitaire game "life"](https://doi.org/10.1038/scientificamerican1070-120),
  *Scientific American* 223(4), 120 (1970).
- E. R. Berlekamp, J. H. Conway and R. K. Guy, *Winning Ways for Your Mathematical Plays*, vol. 2,
  Academic Press (1982), chapter 25: gliders, guns and the universality of Life.
- S. Wolfram, [Statistical mechanics of cellular automata](https://doi.org/10.1103/RevModPhys.55.601),
  *Rev. Mod. Phys.* 55, 601 (1983).
- [LifeWiki](https://conwaylife.com/wiki/), the reference for patterns and their periods, speeds and
  lifespans.

## Read the code

- [`js/examples/game-of-life.js`](js/examples/game-of-life.js) has the compute and render shaders, the
  seed patterns, the `GameOfLife` class, and `measureLife`, the compute-only version the Measure tab uses.
- [`js/teach/experiments/game-of-life.js`](js/teach/experiments/game-of-life.js) has the two experiments;
  [`js/refs/life.js`](js/refs/life.js) has the CPU rule they compare against.
- [`test/game-of-life.test.mjs`](test/game-of-life.test.mjs) checks the seed patterns' periods and speeds
  with a CPU replica; [`test/gpu/game-of-life-measure.gpu.test.mjs`](test/gpu/game-of-life-measure.gpu.test.mjs)
  runs the shipped shader on a software GPU in CI and compares it with the CPU rule.
