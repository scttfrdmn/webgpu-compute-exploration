## Overview

Conway's Game of Life (1970) is a grid of cells, each alive or dead, that all update at once by one
local rule. From that rule come shapes that travel, shapes that blink, guns that fire gliders forever,
and patterns that can compute anything a computer can. The example runs it on a $200 \times 150$
grid, one GPU pass per generation. It is the simplest case of a pattern that shows up all over science:
update each point of a grid from its neighbours, over and over, as in weather and fluid models.

## The idea

**The rule.** Let $n$ be the number of live cells among the eight neighbours of a cell (the four sides
and the four corners). Then in the next generation:

- a live cell with $n = 2$ or $n = 3$ stays alive;
- a dead cell with $n = 3$ is born;
- every other cell is dead.

This rule is written B3/S23: born with 3, survives with 2 or 3.

**Everything changes at once.** Generation $t + 1$ depends only on generation $t$. A cell that wrote its
new state into the grid while its neighbours were still counting would spoil their counts. So the
program keeps two grids, reads one and writes the other, and swaps them each generation. This is called
**ping-pong**.

**The edges wrap.** The grid is a torus: a cell on the right edge has neighbours on the left edge, and
the top row touches the bottom row. In the formula, the neighbour at $x + 1$ is really at
$(x + 1) \bmod W$, where $W = 200$ is the width.

**What the patterns do.** The **Glider** moves one cell on a diagonal every 4 generations, and the
**Lightweight Spaceship** two cells every 4. The **Pulsar** repeats every 3 generations and the
**Pentadecathlon** every 15. The **Gosper Glider Gun** fires a glider every 30. **Diehard**, **Acorn** and
the **R-Pentomino** are small shapes that keep changing for a long time; Diehard vanishes after 130
generations.

## How the GPU does it

- **One thread per cell.** The shader runs in $8 \times 8$ workgroups: $25 \times 19$ of them, 30,400
  threads for 30,000 cells. The 400 extra threads fall outside the grid and stop at once.
- **Each thread** reads its cell and its eight neighbours from one buffer, counts, applies the rule, and
  writes one number to the other buffer. Live is 1, dead is 0, each stored in a 32-bit integer.
- **No waiting inside a generation.** No thread reads what another writes, so none waits for another. Between generations, WebGPU makes sure each pass sees the writes of the one
  before.
- **Drawing.** A fragment shader reads the newest grid straight from the same buffer and colours each
  cell green (alive) or dark gray (dead).

> **GPU:** The example runs one generation per frame, at most 60 a second (the **Speed** slider). A
> generation of 30,000 cells takes the GPU only microseconds, so it is idle almost all the time. The
> Measure tab runs 100 generations per pass to see the real cost.

## How good is the answer

> **Numerics:** Cells hold whole numbers, so nothing is rounded. The GPU grid must match a CPU grid
> cell for cell, forever. Any difference is a bug, not an error to measure.

The **Random soup vs the CPU rule** experiment checks that. On a software GPU in CI, a $256 \times 256$
soup (30% alive at the start, like the **Random** button) matched the CPU exactly at every checkpoint up
to 1,000 generations. Its live share went from 0.30 to 0.34 after one generation, 0.21 after 10, 0.095
after 100 and 0.046 after 1,000. A random start burns down to scattered still shapes and blinkers.

The grid size matters, though, and that is a modelling choice rather than a bug. On an endless plane
the R-Pentomino settles after 1,103 generations. On the $200 \times 150$ torus its gliders wrap around
and crash back into the rest, and in a CPU run it was still changing after 3,000.

The other experiment, **Grid size vs cell updates per second**, runs grids from $128^2$ to $2048^2$.
The example's 30,000 cells are far too few to keep a GPU busy.

## Try this

- In [Explore](#game-of-life), load the **Gosper Glider Gun** and press **Start**. What happens when the
  first gliders wrap around and hit the gun?
- Load **Diehard** and use **Step** near the end. Is the grid empty at generation 130?
- Press **Random**, set **Speed** to 60, and press **Start**. The GPU work per frame is tiny, so what
  limits the frame rate?
- In [Measure](#game-of-life/measure), run **Grid size vs cell updates per second**. At which size does
  the rate stop growing?

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) has the rule as a
formula, the buffer layout and the ping-pong bind groups, why storing each one-bit cell in 32 bits wastes
memory, and how fast Life programs pack many cells into one word.
