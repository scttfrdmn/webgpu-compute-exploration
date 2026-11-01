## Overview

This example is about a question every GPU programmer meets: **which algorithms parallelize, and which
do not?** It answers it with mazes. Building a maze and solving one are both graph algorithms on the
same grid, but they behave very differently on a GPU.

- **Generation runs on the CPU.** The recursive backtracker, Kruskal's and Prim's algorithms each decide
  their next step from everything that happened before. There is nothing to hand to thousands of threads.
- **Solving runs on the GPU** as a **breadth-first search** (BFS): a wavefront that expands one distance
  level per compute dispatch, one thread per cell, with no queue and no locks. The shortest path is then
  traced back from the goal on the CPU.
- **Depth-first search and A\*** are not run on the GPU at all: each pops one cell at a time from a stack
  or a priority queue, so a step has no parallelism to offer.

Even the part that does run on the GPU has a sequential core: the number of BFS levels is the length of
the shortest path, and level $d + 1$ cannot start before level $d$ is done. The Measure tab puts numbers
on that: levels, dispatches, time, and the cost of asking the GPU whether it is finished.

## The model

### Mazes as graphs

The maze is a grid of $w \times h$ cells; an open passage between two neighbouring cells is an edge. A
**perfect** maze is a spanning tree of the grid graph: all $w h$ cells connected by exactly $w h - 1$
passages, so there is exactly one path between any two cells. **Loops** (braiding) then remove each
remaining interior wall with probability $b$, which creates shortcuts and several routes to the goal.

The three generators build the spanning tree in different ways, and each is sequential:

- **Recursive backtracker** (depth-first): walk to a random unvisited neighbour, carving as you go; at a dead
  end, back up along the stack. Every move depends on the whole history. It makes long, winding corridors.
- **Kruskal's**: shuffle all interior walls and remove a wall whenever its two cells are still in different
  trees, tested with a union-find structure (path halving, union by size). The answer for each wall
  depends on all earlier removals.
- **Prim's** (randomized): grow one tree, each step connecting a random frontier cell to a neighbour
  already in the tree.

### BFS levels

BFS from the start cell $s$ computes the distance $d(v)$ of every cell in **levels**:

$$
L_0 = \{ s \}, \qquad
L_{d+1} = \left\{ v \notin L_0 \cup \dots \cup L_d \; : \; \exists\, u \in L_d \text{ with } (u, v) \in E \right\}.
$$

Every cell of $L_{d+1}$ can be found independently of the others, which is the parallelism. The number of
levels to reach the goal $g$ is $D = d(g)$, the shortest-path length, and on a grid it can never be less
than the grid distance $|x_g - x_s| + |y_g - y_s|$. With the default corners of a 511 × 383 maze that
bound is $510 + 382 = 892$.

Once $d(g)$ is known, the path is found by walking back from $g$, each step to an open neighbour exactly
one closer: $D + 1$ cells, a shortest path by construction.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `walls` | `u32` | $w h$ | bits 0–3: open to the north, east, south, west; bit 4: on the path |
| `dist` | `atomic<u32>` | $w h$ | BFS distance, `0xFFFFFFFF` = not reached yet; start = 0 |
| `stats` | `atomic<u32>` | 2 | cells reached so far, largest distance assigned so far |
| `params` | `Params`, 32 B | 512 slots | maze size, start, goal, the level of this dispatch, … |

Both sides of a passage carry its bit, so a cell can test its own word without reading its neighbours'
walls. The same cell bitmasks also drive the renderer, which draws the classic $(2w + 1) \times (2h + 1)$
pixel lattice from them. The bit values are defined once, in `js/refs/maze.js`, and the example generates
its WGSL constants from them.

### One dispatch per level

`cs_bfs_step` runs one thread per cell in 8 × 8 workgroups, over the whole grid, for level $L$:

1. If the goal already has a distance $\le L$, return: the flood is over.
2. If this cell already has a distance, return.
3. If any open neighbour has distance exactly $L$, store $L + 1$, and count the cell in `stats`.

This is the **pull** form of BFS: each unreached cell checks whether a neighbour is on the frontier,
instead of frontier cells pushing to their neighbours. Each thread writes only its own cell, and the only
value written during the dispatch is $L + 1$, never $L$, so reading a neighbour that is being written at
the same moment cannot change the result. The outcome does not depend on thread order and is identical to
a queue-based BFS. Atomics make those concurrent loads and stores well-defined in WGSL.

> **GPU:** consecutive dispatches in one compute pass are ordered, so level $L + 1$ sees every write of
> level $L$ with no barrier in the shader. A frame encodes up to 512 levels, each with its own uniform
> slot (dynamic offset) for $L$.

### Asking whether it is done

The CPU cannot see the flood without a readback. Every few frames it copies 12 bytes (`stats` and
`dist[goal]`) to a mappable buffer: the solve is finished when the goal has a distance, and the goal is
unreachable when the last level added no cell (the largest distance stayed below the number of
dispatches). Then the distance field is read back once, the path is traced on the CPU (up to tens of thousands of
cells walked one after another), and the walls are uploaded again with the path bits set.

## Numerical behaviour

> **Numerics:** there is no rounding here: distances are integers, and the GPU's distance field is
> exactly the queue BFS's, cell for cell, up to the goal. What varies is how much work it takes.

**Correctness.** The CI suites compare every cell with a CPU queue BFS for all three generators, perfect
and braided, for several batch sizes and for an unreachable goal. Cells closer than the goal match
exactly; cells farther away were never expanded and are still unreached, because the flood stops at the
goal's level. Batches overshoot by up to one batch of dispatches, which return at step 1.

**Levels are the critical path.** The number of dispatches is set by the maze, not by the GPU. For the
corner-to-corner path with the experiments' seed:

| maze | recursive backtracker | Kruskal's | Prim's |
|---|---|---|---|
| 63 × 47 (2,961 cells) | 984 | 212 | 156 |
| 255 × 191 (48,705 cells) | 6,690 | 822 | 624 |
| 1023 × 767 (784,641 cells) | 73,964 | 4,692 | 2,120 |

The backtracker's long corridors put almost a tenth of the largest maze on the solution path. In CI,
removing 1% of the remaining walls of a 511 × 383 Kruskal maze cut the levels from 2,020 to 1,474, and 30%
brought them to 896, four above the grid-distance bound of 892.

**Work efficiency.** A queue BFS touches each cell once. The level-synchronous GPU BFS checks *every* cell
in every level, $D \cdot w h$ checks in all: 4,692 × 784,641 ≈ $3.7 \times 10^{9}$ for the largest Kruskal
maze, about 4,700 times the queue's work. Most threads find their cell already reached, or no neighbour
on the frontier, and return.

## Why it maps well — or poorly — to a GPU

**One level maps well.** Every cell is independent, the access pattern is a regular 4-neighbour stencil,
and the result is deterministic without locks.

**A maze is the worst case for level-synchronous BFS.** A perfect maze is a tree of corridors, so the
frontier is only a handful of cells while the grid has hundreds of thousands: nearly every thread of
every dispatch does nothing useful, and the number of levels is as large as the path is long. Each level
also costs a dispatch. On the CPU-emulated GPU in CI the largest Kruskal maze took 22 s; the CPU queue BFS
took 48 ms. A GPU earns its keep on graphs with **wide frontiers and few levels**: open grids, braided
mazes, and the small-diameter networks of real-world graphs.

**Better GPU BFS exists, within limits.** Frontier queues built with prefix sums (Merrill, Garland and
Grimshaw) make the work proportional to the edges, and direction-optimizing BFS (Beamer, Asanović and
Patterson) switches between push and pull as the frontier grows and shrinks. None of them can remove the
$D$ sequential levels.

**What stays on the CPU, and why.** The generators are sequential by construction. Parallel spanning-tree
algorithms exist (Borůvka-style rounds with a GPU union-find), but they no longer follow Kruskal's random
order and need several synchronized passes; for one maze of under a million cells the CPU takes tens of
milliseconds. DFS and A\* solving pop one cell at a time from a stack or a priority queue: A\* may visit far
fewer cells than BFS with a good heuristic, but it offers no parallelism per step.

**The readback is a design choice.** Checking after every level costs one round trip per level; batching
levels saves round trips but runs some levels past the goal. The
[batch experiment](#maze/measure) measures that trade-off.

## Try this

> **Try this:** in [Explore](#maze), generate a 1023 × 767 maze with the **Recursive backtracker** and
> press **Solve**. Then switch to **Kruskal's** and **Prim's** with the same size, and compare how far the
> colored flood spreads before it reaches the goal.

- Run [maze size vs BFS levels](#maze/measure) with each generator. Does the time grow with the levels, the
  cells, or their product?
- Run [loops vs BFS levels](#maze/measure): how quickly do the levels approach the grid-distance bound of 892?
- Run [levels per submit vs wall time](#maze/measure). Where is the sweet spot on your GPU, and how many
  dispatches run past the goal at 512?
- In [Explore](#maze), move the start next to the goal (click near the bottom right): the flood stops after
  a few levels. Then move **Loops** to 30% and solve again.

## References

- E. F. Moore, The shortest path through a maze, *Proceedings of the International Symposium on the Theory
  of Switching*, Harvard University Press (1959): breadth-first search.
- J. B. Kruskal, [On the shortest spanning subtree of a graph and the traveling salesman problem](https://doi.org/10.1090/S0002-9939-1956-0078686-7),
  *Proceedings of the AMS* 7, 48 (1956).
- R. C. Prim, [Shortest connection networks and some generalizations](https://doi.org/10.1002/j.1538-7305.1957.tb01515.x),
  *Bell System Technical Journal* 36, 1389 (1957).
- R. E. Tarjan, [Efficiency of a good but not linear set union algorithm](https://doi.org/10.1145/321879.321884),
  *Journal of the ACM* 22, 215 (1975): the union-find behind Kruskal's generator.
- P. E. Hart, N. J. Nilsson and B. Raphael, [A formal basis for the heuristic determination of minimum cost paths](https://doi.org/10.1109/TSSC.1968.300136),
  *IEEE Transactions on Systems Science and Cybernetics* 4, 100 (1968): A\*.
- P. Harish and P. J. Narayanan, [Accelerating large graph algorithms on the GPU using CUDA](https://doi.org/10.1007/978-3-540-77220-0_21),
  HiPC (2007): level-synchronous BFS, one thread per vertex.
- D. Merrill, M. Garland and A. Grimshaw, [Scalable GPU graph traversal](https://doi.org/10.1145/2145816.2145832),
  PPoPP (2012): work-efficient frontier queues.
- S. Beamer, K. Asanović and D. Patterson, [Direction-optimizing breadth-first search](https://doi.org/10.1109/SC.2012.50),
  SC12 (2012): push and pull.

## Read the code

- [`js/examples/maze.js`](js/examples/maze.js) has the three generators, `cs_bfs_step` with the argument for
  why it is deterministic, the lattice renderer, and the `Params` layout in the file header. `frame` and
  `pollTermination` are the dispatch and readback sequence, `extractPath` the trace-back.
- [`js/teach/experiments/maze.js`](js/teach/experiments/maze.js) has the three Measure experiments, built on
  `measureBFS` (compute-only, at the end of the example file).
- [`js/refs/maze.js`](js/refs/maze.js) has the queue BFS the experiments compare against, and the cell
  encoding shared with the example.
- [`test/maze.test.mjs`](test/maze.test.mjs) checks the generators and a CPU replica of the shader;
  [`test/gpu/maze.gpu.test.mjs`](test/gpu/maze.gpu.test.mjs) and
  [`test/gpu/maze-measure.gpu.test.mjs`](test/gpu/maze-measure.gpu.test.mjs) run the shipped WGSL on a
  software GPU in CI. The second one is the source of the CI numbers on this page.
