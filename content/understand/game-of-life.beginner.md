## What you see

The picture is a grid of 200 by 150 small squares, called cells. Each cell is either alive (green) or
dead (dark gray). Press **Start**, and the grid changes again and again. Each new grid is called a
generation. Some shapes sit still, some blink, some crawl across the screen, and some grow and then die.

You can draw your own shapes: click a cell to flip it between alive and dead. The buttons under
**Patterns** load famous shapes in the middle of the grid, and **Random** makes about 3 in every 10 cells
alive.

## What is happening

Every cell looks at its eight neighbors: the cells above, below, left, right, and on the four corners.
It counts how many of them are alive. Then it follows three rules:

1. A live cell with 2 or 3 live neighbors stays alive.
2. A dead cell with exactly 3 live neighbors comes to life.
3. Every other cell is dead in the next generation. Too few neighbors is lonely, and too many is
   crowded.

That is the whole game. John Conway made it up in 1970. Nobody tells a glider to move, but the rules
make it move one cell on a slant every 4 generations. The **Gosper Glider Gun** shoots out a new glider
every 30 generations, forever. **Diehard** is just 7 cells. It changes for 130 generations and then
vanishes.

All the cells change at the same moment. A cell must count its neighbors as they were in the last
generation, not as they are being changed. So the program keeps two copies of the grid. It reads the old
one and writes the new one. Then the two swap jobs for the next generation.

The grid has no edges. A glider that leaves on the right comes back on the left, and one that leaves at
the top comes back at the bottom. It is like the screen of an old video game.

> **Numerics:** A cell is just alive or dead, so there is nothing to round. A GPU and a normal processor
> must give the same grid, cell for cell, every time. Any difference at all would be a bug.

## Why a GPU helps

A GPU is like a huge team of workers who all work at the same time. Here each worker gets one cell. It
counts the neighbors, checks the rules, and writes the answer in the new grid. No worker needs to wait
for any other, because they all read from the old grid.

> **GPU:** This grid has 30,000 cells, which is small for a GPU. One generation takes it far less than a
> thousandth of a second. Most of the time the GPU is waiting for the next frame. To see what it can
> really do, the Measure tab runs grids up to 2,048 by 2,048, with 100 generations at a time.

## Try this

- Open [Explore](#game-of-life), press **Glider**, then **Start**. Follow it until it wraps around the
  edge.
- Press **Diehard**, then press **Step** over and over. Does it really vanish? (It takes 130 steps, so
  you may want to press **Start** instead.)
- Load the **Gosper Glider Gun**. What happens when the first gliders wrap around and come back to the
  gun?
- Press **Random**, then **Start**. Watch the crowd thin out until only small still shapes and blinkers
  are left.
- Press **Clear** and draw your own shape with clicks. Can you make one that blinks?
- In [Measure](#game-of-life/measure), run **Random soup vs the CPU rule**. The column of cells that
  differ from the CPU should be 0 at every step.
