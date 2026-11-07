## Overview

"Falling sand" games fill a box with grains, liquids, fire and plants, one pixel at a time: sand piles
up, water levels out, oil floats on water and burns. Each material follows a few local rules, and
everything you see comes from 76,800 cells ($320 \times 240$) applying them every step: a **cellular
automaton**. The hard part on a GPU is the **conflicts**: if every cell were a thread, two grains could
both fall into the same empty cell at once, and one would be lost.

## The idea

**Blocks instead of cells.** Each step cuts the grid into $2 \times 2$ blocks, and one thread owns each
block. A thread reads its four cells, applies the rules, and writes the same four cells back. Cells move
only by **swapping** inside their block. On every other step the blocks are shifted by one cell in $x$
and $y$; without that shift, a grain could never leave its block. This scheme is called the **Margolus
neighbourhood** (Margolus, 1984).

**The rules, in order.** In each block:

1. **Reactions.** Fire next to water goes out. Fire spreads into oil (chance $24/256$ per step) and
   plants ($32/256$), but only if the block also holds an empty cell, so a pool burns from its surface
   down. Acid dissolves stone, sand and plants and is used up with them. A plant next to water turns it
   into plant, with chance $1/256$.
2. **Fire burns down.** Its life counter drops by one each step, and at zero it is gone.
3. **Movement.** The materials are ranked by weight: fire < empty < oil < water < acid < sand. Stone and
   plant never move. A heavier cell above a lighter one swaps with it: always for a grain falling through
   air, with chance $192/256$ for a liquid, and $128/256$ when sinking through a liquid. If it
   cannot fall, it may slide down on a diagonal. Liquids also flow sideways in a remembered direction
   and turn around when blocked, which levels a pool quickly.

**Chance without a die.** Every random choice comes from a hash of the block's position and the step
number, so the same step always gives the same result.

## How the GPU does it

- **One thread per block.** Workgroups are $8 \times 8$ threads. On a shifted step the blocks along the
  edges hang off the grid, so a step has up to $161 \times 121 = 19{,}481$ threads. Cells outside the grid
  act as walls.
- **One buffer, updated in place.** In one step every cell belongs to exactly one block, so no two threads
  ever touch the same cell. Unlike the Game of Life, no second copy of the grid is needed.
- **Several steps per frame.** A frame runs 1 to 8 steps (the **Steps / frame** slider, 3 by default) as
  separate dispatches in one compute pass. Drawing reads the same buffer; nothing is copied back.

> **GPU:** Neighbouring threads often hold different materials and take different paths through the
> rules. Threads that run together in a group must step through all of those paths.

## How good is the answer

> **Numerics:** Movement only swaps cells, so the count of each material can change only through a
> reaction. That is an exact rule in whole numbers, and the **What is conserved** experiment checks it.

On a software GPU in CI:

- With the sandbox's plants turned to stone (no reaction can happen), every count stayed the same over
  2,000 steps.
- With fire on the oil, 408 of the 708 oil cells burned in about 300 steps, and the fire was out by step
  1,000. The counts of sand, stone and water did not change by a single cell.
- With plants under water, 996 water cells turned into plant in 2,000 steps, and the sum of water and
  plant stayed exact.

Runs are also exactly repeatable: 200 steps gave the same grid, bit for bit, with 8 steps per submit or 1.

The model is not physics, though: there is no speeding up as things fall. Sand drops a steady one cell
per step, and liquids about three quarters of that.

## Try this

- In [Explore](#falling-sand), pour **Water** onto the sand heap, then **Oil** on top. Which ends up where?
- Paint **Fire** on the oil slick in the basin. Does the pool burn all at once or from the top?
- Set **Steps / frame** to 8, then to 1. The sand moves 8 times slower, but does the frame rate go up 8
  times?
- In [Measure](#falling-sand/measure), run **What is conserved** with each **Scene**, and then **Grid size
  and steps per submit vs time per step**.

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) shows how a cell is
packed into 32 bits, the block offsets as a formula, the bias that rule order can cause, and why empty
blocks still cost time.
