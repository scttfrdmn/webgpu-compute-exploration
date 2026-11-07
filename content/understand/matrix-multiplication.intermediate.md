## Overview

The example multiplies two $512 \times 512$ matrices filled with random numbers between 0 and 1, once,
when you press **Run Matrix Multiplication**. Matrix products are the core of neural networks, of 3D
graphics and of much of science and engineering, and the benchmark that ranks supercomputers is built on
them. The report shows the time, the speed in GFLOPS (billions of math operations per second), and a
check of one entry against the CPU.

## The idea

**One entry is a dot product.** Entry $c_{ij}$ of $C = AB$ is row $i$ of $A$ times column $j$ of $B$:

$$c_{ij} = \sum_{k=0}^{N-1} a_{ik} b_{kj}$$

For $n \times n$ matrices there are $n^2$ entries, and each needs $n$ multiplies and $n$ adds. That is
$2n^3$ operations: about 268 million for $n = 512$. Double $n$ and the work goes up eight times.

**Work grows faster than data.** The three matrices hold $3n^2$ numbers, but the work is $2n^3$, so each
number is used $n$ times. That is the special thing about this problem. A program that keeps numbers
close to the math units while they are reused can run near the chip's top speed. A program that fetches
every number from main memory each time it needs it spends most of its time waiting.

**Rounding.** Each number is stored as a 32-bit float with about seven digits. Every add rounds, so an
entry that is a sum of $n$ terms collects $n$ small errors. In the worst case they all point the same
way and the error grows like $n$. In practice they partly cancel, and the typical error grows only like
$\sqrt{n}$ times the rounding unit $u = 2^{-24} \approx 6 \times 10^{-8}$.

## How the GPU does it

Every entry of $C$ is independent, so each GPU thread computes one entry.

- **The naive kernel** (the one the button runs) groups threads in $8 \times 8$ workgroups, $64 \times 64$
  of them for $n = 512$. Each thread loops over $k$, loads $a_{ik}$ and $b_{kj}$ from global memory, and
  adds their product to a running sum. That is two loads for every multiply-add.
- **The tiled kernel** (in the [Measure](#matrix-multiplication/measure) tab) uses $16 \times 16$
  workgroups. The $k$ loop is cut into phases. In each phase every thread loads one number of $A$ and one
  of $B$ into fast workgroup memory, so the group holds a $16 \times 16$ tile of each. A barrier waits
  until all 256 threads are done. Then each thread does 16 multiply-adds from the shared tiles. A second
  barrier stops anyone from loading the next tile while others still read this one. Each number fetched
  from global memory is now used by 16 threads.
- **The CPU baseline** is careful single-threaded JavaScript that works on blocks of $64 \times 64$ so
  they stay in the CPU's cache.

> **GPU:** The tiled kernel adds the terms in the same order as the naive one, $k = 0, 1, 2, \ldots$, and
> pads with exact zeros at the edges. So both give the same answer, bit for bit.

## How good is the answer

The Measure experiment checks each method against a float64 product of 16 sample rows.

- **Error.** On an Apple M4 Pro (Chrome and Safari), the largest relative error was at most 1.2 times
  $\sqrt{n}\,u$ at every size, about $1.2 \times 10^{-6}$ at $n = 512$. The worst-case bound $n u$
  would allow more than 20 times as much. The naive and tiled results were identical.
- **Time grows like $n^3$.** Fitting time against $n$ on a log-log plot gave slopes of 2.98 (naive) and
  2.97 (tiled) on the M4 Pro in both browsers, close to the predicted 3.
- **Speed.** In Chrome on the M4 Pro at $n = 512$, the JavaScript loop took 110 ms (2.4 GFLOPS), the naive
  kernel 0.75 ms and the tiled one 0.33 ms (about 800 GFLOPS). At $n = 2048$ tiling was 2.3 times faster
  than naive in Chrome, but only 1.16 times in Safari on the same chip. How much tiling helps depends on
  how well the caches already served the naive kernel.

> **Numerics:** The one-shot button checks only $c_{00}$, with a tolerance of 0.01. The Measure tab's
> check is much stricter: it allows $4\sqrt{n}\,u$.

## Try this

- In [Explore](#matrix-multiplication), press **Run Matrix Multiplication** a few times. The first run
  also compiles the shader. Compare the wall time with the GPU time.
- In [Measure](#matrix-multiplication/measure), run **CPU vs naive GPU vs tiled GPU**. On the time chart,
  do the lines follow the $n^3$ line? Set **Largest n** to 2,048 for more sizes.
- Find the smallest $n$ where each GPU line in the GFLOPS chart stops rising. Below it, the fixed cost of
  starting the GPU work is larger than the math.
- Read the **Measured vs predicted** table. Does tiling beat the naive kernel on your GPU, and by how much?

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) explains arithmetic
intensity and the roofline model, why the naive kernel's memory access pattern is poor, the rounding
bounds for dot products, and the full results measured in CI on a software GPU.
