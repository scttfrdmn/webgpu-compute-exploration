## What you see

The picture is a box of 320 by 240 tiny squares, called cells. Each cell holds one thing: sand (tan),
water (blue), oil (dark brown), stone (gray), plant (green), acid (bright yellow-green), fire (red to
yellow), or nothing at all (dark). It starts with a sandbox: two stone ledges, a heap of sand, a blob of
water, and a stone basin full of water with a thin layer of oil on top.

Press **Start**, and the sand falls and piles up, and the water runs and levels out. Pick a material
under **Material** and hold the left mouse button to paint with it. The right button erases. Painting
works even while the box is paused.

## What is happening

The program moves time forward in small steps. In each step, it cuts the whole box into little blocks of
2 by 2 cells. Each block looks at its four cells and follows a few simple rules:

- **Falling.** If a heavier thing sits on top of a lighter one, they may swap. So sand falls through air,
  and also sinks through water and oil, just more slowly. Oil is lighter than water, so it floats.
- **Sliding.** If sand cannot fall straight down, it may slide down to the side. That is how a heap
  forms. Water and oil also flow sideways, so a pool ends up flat.
- **Reactions.** Fire next to water goes out. Fire spreads into oil and plants, but only where there is
  air next to it, so a pool of oil burns from the top down. Acid eats stone, sand and plants, and is used
  up. A plant next to water slowly turns the water into more plant.
- **Burning out.** Fire only lasts a short time, then it is gone.

Many rules use chance, like a roll of the dice. The program does not keep a die. It mixes up the block's
place and the step number into a number that looks random. So the same step always gives the same roll.

> **Numerics:** Things only move by trading places inside a block. Nothing is made or lost by moving.
> The amount of sand, stone or water can only change through a reaction, like fire or acid. The Measure
> tab counts every material to check this.

## Why a GPU helps

A GPU is like a huge team of workers who all work at the same time. Here each worker gets one 2 by 2
block. That is about 19,000 workers for one step.

Why blocks and not single cells? Picture two grains of sand, each told to fall into the same empty spot
at the same time. One of them would be lost. With blocks, every cell belongs to just one worker, so two
workers never fight over a cell. The next step shifts the blocks by one cell, so that grains can cross
from one block into the next.

> **GPU:** The team runs 3 steps for every picture it draws, or up to 8 if you move the **Steps / frame**
> slider. The picture is drawn straight from the same memory, so nothing has to go back to the normal
> processor.

## Try this

- Open [Explore](#falling-sand), press **Start**, and pick **Water**. Pour it onto the sand heap. Then
  pour **Oil** on top. Which one ends up on top?
- Pick **Fire** and paint it on the oil in the basin. Does it all burn at once?
- Pick **Acid** and pour it on a stone ledge. Watch it eat a hole.
- Paint a **Plant** stem into the water and wait. It grows very slowly.
- Change the **Brush** slider to paint with a bigger or smaller brush.
- In [Measure](#falling-sand/measure), run **What is conserved**. Which counts change, and which reaction
  changed them?
