## Overview

"Falling sand" games simulate a box of grains, liquids, fire and plants one pixel at a time: sand piles
up, water levels out, oil floats on water and burns. Each material is a handful of local rules, and the
behaviour you see emerges from millions of cells applying them every step. This is a **cellular
automaton**, with a 320 × 240 grid of cells, each one 32-bit word.

The hard part on a GPU is not the rules but the **conflicts**. If every cell is a thread and two grains
both decide to fall into the same empty cell, one of them is lost. This example avoids that with the
**Margolus neighbourhood**: the grid is cut into 2 × 2 blocks, one thread owns each block, and cells
only move by swapping inside their block. Nothing is ever created or destroyed by movement, and no two
threads touch the same cell, so the grid can be updated in place without atomics or locks.

> **Note:** there is no continuous model behind the rules: this is a discrete system, and its
> "numerics" are about what it conserves, what is random, and what is biased.

## The model

### Cells

Each cell is one `u32`: bits 0–3 hold the material, bits 4–7 a random shade (the grain texture), bits
8–15 a state: the remaining life of fire, or the flow direction of a liquid. The materials, in order of
density, are fire (0) < empty (1) < oil (2) < water (3) < acid (4) < sand (5); stone and plant never move,
and cells outside the grid read as solid wall.

### The Margolus neighbourhood

Every step tiles the grid with 2 × 2 blocks; on odd steps the tiling is shifted by one cell in $x$ and
$y$. Without the shift a grain could never leave its block. With block corners at

$$
(x_0, y_0) = (2 b_x - s,\; 2 b_y - s), \qquad s = n \operatorname{mod} 2 ,
$$

every cell belongs to exactly one block per step. On odd steps the blocks along the edges hang off the
grid, which is why there are $\lfloor W/2 \rfloor + 1$ blocks per row; their outside cells are walls and
are never written.

### Rules, in order

One thread reads its block's four cells and applies, in this order:

1. **Brush** (first step of a frame only): paint the selected material, sprayed at 50% density for
   grains and liquids.
2. **Reactions** between every ordered pair of cells in the block, each cell at most once per step. Fire
   next to water goes out; fire spreads into oil (chance 24/256 per step) and plants (32/256) if the block
   also holds an empty cell, so a pool burns from its surface down; acid dissolves stone, sand and plants
   and is used up; a plant next to water turns it into plant with chance 1/256.
3. **Fire burns down**: its life counter drops by one per step, and it vanishes at zero.
4. **Movement, by swaps only**: in each column, a heavier cell above a lighter one falls (always into
   air for grains, with chance 192/256 for liquids, 128/256 through another fluid); otherwise it may slide
   diagonally (sand half the time); cells that did neither may move sideways. Liquids keep a flow
   direction and turn around when blocked, which levels a pool far faster than a random walk.

The two columns are processed in a random order each step, so there is no left/right bias.

### Randomness

Every chance comes from a PCG hash of (block $x$, block $y$, step counter), advanced for each decision.
There is no `Math.random` on the GPU and no random-number state in memory: the same step always gives
the same result, however the steps are batched into submits.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `cells` | `u32` | $W \times H$ (76,800) | material, shade, state |
| `params` | `Params` | 8 slots of 256 bytes | grid size, brush stroke and material, flags, step counter |

There is **one** cell buffer, updated in place. Within one dispatch every cell is read and written by
exactly one thread, so there is no race, and WebGPU makes each dispatch's writes visible to the next.
Ping-pong buffers, which a cell-per-thread stencil needs, would double the memory for nothing here.

### Dispatches

`cs_main` runs in 8 × 8 workgroups, one thread per block: $161 \times 121$ blocks, $21 \times 16 = 336$
workgroups per step. A frame runs 1 to 8 steps (3 by default) as consecutive dispatches in one compute
pass. Each step reads its own `Params` slot through a **dynamic offset**, so its step counter, and with it
the block offset and the random stream, changes without a CPU round trip. The render pass reads the same
buffer; nothing is read back.

> **GPU:** the block is a function-local `struct` of four `u32` and four `bool`, which the compiler keeps
> in registers. The rules are a few hundred instructions of integer comparisons and branches.

## Numerical behaviour

> **Numerics:** because movement only swaps cells, the number of cells of each material changes only
> through a reaction. That is an exact, integer invariant, and it is what the
> [conservation experiment](#falling-sand/measure) checks on your GPU.

**Conservation.** In CI, on a software GPU, the shipped sandbox with its plants turned to stone (so no
reaction can happen) keeps every material count unchanged over 2,000 steps. With a layer of fire on the
oil, 408 of the 708 oil cells burn in about 300 steps and the fire is out by step 1,000, while the counts
of sand, stone and water do not change by a single cell. With a patch of plants under water, 996 water
cells turn into plant in 2,000 steps, and the sum of the two stays exact.

**Determinism.** The randomness depends only on the block and the step counter, so a run is exactly
reproducible. CI runs 200 steps once with 8 steps per submit and once with 1, and gets the same grid bit
for bit.

**Speed of movement.** A grain moves at most one cell per step in each direction: there is no momentum
and no acceleration, so sand falls at a constant "terminal velocity" of one cell per step. Liquids fall
at about 3/4 of that, which lets them spread sideways as they fall.

**Bias and artefacts.** A block CA has a preferred grid. The fixed $2 \times 2$ tiling of a step means a
cell's partners alternate between two neighbourhoods, and rules that depend on the order within a block
(which column first, which cell reacts first) can produce drift or patterns. The random column order
removes the left/right bias of falling and sliding. The reaction loop always tries the pairs in the same
order, so in a block with two possible reactions the pair it reaches first gets the first chance; with
the low chances used here that is rarely visible.

## Why it maps well — or poorly — to a GPU

**It maps well.** Blocks are independent, the update needs no atomics, no barriers and no second buffer,
and all the arithmetic is on integers, so it is exact and identical on every GPU. The memory traffic is
small: each thread reads and writes four 32-bit words.

**Divergence is the cost.** Neighbouring threads usually hold different materials and take different
branches through the rules, and the 32 or 64 threads of a SIMD group run the union of their paths. An
empty block takes the shortest path, but it is still a thread: in the default sandbox about 84% of the
cells are empty, and they cost a dispatch slot every step. A production engine would track active regions
("dirty rectangles") and skip the rest; that needs a reduction or atomics per step.

> **GPU:** at 320 × 240 one step is 19,481 threads, too few to fill a large GPU and over in a few
> microseconds. The cost of a frame is then mostly the fixed cost of the dispatches and the submit, which
> the [scaling experiment](#falling-sand/measure) shows by running with 1 and with 8 steps per submit.

## Try this

> **Try this:** in [Explore](#falling-sand), pour **Water** onto the sand heap and then **Oil** on top of
> the water. Oil floats because water is denser; sand sinks through both.

- Paint **Fire** on the oil slick in the basin in [Explore](#falling-sand). The pool burns from its
  surface down: burning needs an empty cell in the block.
- Draw a **Plant** stem into water and wait: it grows one cell at a time, with chance 1/256 per step.
- Set **Steps / frame** to 8 and watch the frame rate; then to 1. The simulation is 8× slower per frame,
  but is the frame 8× cheaper?
- Run [what is conserved](#falling-sand/measure) with each scene. Which material counts change, and
  which reaction explains each change?
- Run [grid size and steps per submit](#falling-sand/measure). At which grid size does the cost per step
  start to grow with the number of cells on your GPU?

## References

- N. Margolus, [Physics-like models of computation](https://doi.org/10.1016/0167-2789(84)90252-5),
  *Physica D* 10, 81 (1984): block partitioning, the Margolus neighbourhood.
- T. Toffoli and N. Margolus, [Cellular Automata Machines: A New Environment for Modeling](https://mitpress.mit.edu/9780262526319/cellular-automata-machines/),
  MIT Press (1987): block rules, conservation and reversibility.
- M. Jarzynski and M. Olano, [Hash functions for GPU rendering](https://jcgt.org/published/0009/03/02/),
  *Journal of Computer Graphics Techniques* 9(3), 21 (2020): the PCG hash used for every random choice.
- M. E. O'Neill, [PCG: a family of simple fast space-efficient statistically good algorithms for random number generation](https://www.pcg-random.org/paper.html)
  (2014).

## Read the code

- [`js/examples/falling-sand.js`](js/examples/falling-sand.js): the header explains the Margolus scheme
  with a diagram and the cell encoding; `cs_main` has the rules in the order above, and `reaction()` the
  reaction table.
- [`js/teach/experiments/falling-sand.js`](js/teach/experiments/falling-sand.js) has the two Measure
  experiments and their scenes, built on `measureSand` (compute-only, at the end of the example file).
- [`js/refs/falling-sand.js`](js/refs/falling-sand.js) counts materials, the CPU check the experiments use.
- [`test/falling-sand.test.mjs`](test/falling-sand.test.mjs) has a CPU replica of the shader with the same
  random stream, and checks that blocks tile the grid and that non-reacting materials are conserved;
  [`test/gpu/falling-sand-measure.gpu.test.mjs`](test/gpu/falling-sand-measure.gpu.test.mjs) runs the
  shipped shader on a software GPU in CI.
