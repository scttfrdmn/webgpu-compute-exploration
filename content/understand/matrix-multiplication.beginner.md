## What you see

This example has no picture. You press **Run Matrix Multiplication**, and a report appears with a
check mark. It says how long the work took and whether the answer passed a test.

A matrix is a grid of numbers, like a page of a spreadsheet. Here there are two grids, A and B. Each one
has 512 rows and 512 columns, and each box holds a random number between 0 and 1. The GPU multiplies the
two grids and makes a third grid, C, of the same size.

## What is happening

Each box in C is made the same way. Take one row of A and one column of B. Each has 512 numbers. Multiply
the first number of the row by the first number of the column, the second by the second, and so on. Then
add up all 512 results. That sum goes in one box of C.

C has 512 times 512 boxes. That is 262,144 boxes, and each needs 512 multiplies and 512 adds. In all it
is about 268 million steps of math for one product.

The report shows two speeds. The "wall time" counts the whole trip: getting the GPU ready, sending the
numbers over, doing the math, and bringing the answer back. The GPU time, if your browser can measure it,
counts only the math. The math is often a small part of the trip.

To check the answer, the program works out the first box of C again on the normal processor and compares
the two. If they match to within 0.01, the test passes.

> **Numerics:** A computer keeps only about seven digits of each number, so each add rounds a little.
> With 512 adds those tiny errors build up, but only to about one part in a million. That is why the
> check allows a small gap and does not ask for an exact match.

## Why a GPU helps

A GPU is like a huge team of workers who all work at the same time. Every box of C is its own job, and
no job has to wait for any other. So the example gives each worker one box. That is 262,144 workers for
one product.

There is a catch. The math is quick, but fetching numbers from memory is slow. Each worker needs a whole
row and a whole column, and many workers need the same ones. Think of a class doing homework from one
book on a far shelf. If each student walks to the shelf for every number, they spend most of the time
walking.

> **GPU:** The faster version, called tiled, works like a team that copies one page of the book onto a
> table they share. A group of 256 workers loads a small block of A and of B into fast shared memory
> once, and then every worker in the group uses it. Each number is fetched from far memory 16 times less
> often.

On an Apple M4 Pro laptop in Chrome, one product of two 512 by 512 grids took about 110 ms in plain
JavaScript on one core. The simple GPU version took about 0.75 ms, and the tiled one about 0.33 ms.

## Try this

- Open [Explore](#matrix-multiplication) and press **Run Matrix Multiplication** a few times. Is the
  first run slower? The first run also has to prepare the GPU program.
- Compare the wall time with the GPU time in the report. Which part of the trip takes longest?
- In [Measure](#matrix-multiplication/measure), run **CPU vs naive GPU vs tiled GPU**. It tries many grid
  sizes. When the grid gets twice as wide, how much longer does it take? (Hint: each box has twice as many
  numbers, and there are four times as many boxes.)
- Look at the error chart. Are the three lines close to each other?
