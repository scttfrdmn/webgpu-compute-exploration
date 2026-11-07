## Overview

Sorting does not look parallel. Quicksort and merge sort, the usual CPU methods, decide what to do next
after every comparison, based on the data. A GPU wants the opposite: thousands of threads doing the
same thing at once. This example sorts up to 4,194,304 numbers (keys) on the GPU in three ways and
checks every result against the browser's own sort. The bar chart shows one of them, bitonic sort, one stage at a time.

## The idea

**Bitonic sort is a fixed plan.** A sorting network is a list of compare-and-swap steps chosen in
advance: compare the keys at positions $i$ and $i + j$, and swap them if they are out of order. The plan
is the same for every input. Bitonic sort (Batcher, 1968) builds sorted blocks of size $k = 2, 4, 8,
\ldots, n$. Half the blocks are sorted up and half down, since an up block next to a down block is easy
to merge. Each merge is a series of stages with distance $j = k/2, k/4, \ldots, 1$. For $n = 2^m$ keys
there are

$$S = \frac{m(m+1)}{2}$$

stages: 55 for the 1024 bars in the chart, 210 for $2^{20}$ keys. That is about 5 times more comparing
than a good one-at-a-time sort, but every comparison in a stage can happen at once.

**Padding.** The plan needs $n$ to be a power of two, so other sizes are filled up with the largest
key, `0xFFFFFFFF` (the gray bars in the chart).

**Radix sort never compares keys.** It sorts by one 4-bit digit at a time, starting with the lowest:
16 buckets, 8 passes for 32-bit keys. Each pass must be **stable**, so keys with the same digit keep
the order from the last pass. That way the earlier passes are not undone.

**Float keys.** For `f32` values, one pass first flips the bits of each value (all of them for a
negative value, only the sign bit otherwise) so that integer order matches number order, and one pass
undoes it after the sort.

## How the GPU does it

- **Bitonic (global):** one dispatch per stage, one thread per pair. Every key is in exactly one pair,
  so no two threads touch the same key.
- **Bitonic (workgroup memory):** all stages that stay inside a 512-key block run in fast shared memory,
  with barriers between them. At $2^{20}$ keys this cuts the dispatches from 210 to 78.
- **Radix:** each pass is three dispatches. First, each workgroup counts the digits in its tile of 1,024
  keys. Second, a **prefix sum** (scan) turns those counts into starting places: for each tile and
  digit, how many keys come before it. Third, each key is written to its place, plus its rank among the
  equal digits in its tile.

> **GPU:** A scan answers "how many come before me?" for every item at once, so thousands of threads
> can each find their spot without waiting in line.

## How good is the answer

Sorting has no rounding, so the answer is either right or wrong. Each run compares the GPU result with
the browser's sort bit for bit. In the Apple M4 Pro bundles (Chrome and Safari), every sort of both u32
and f32 keys matched exactly.

Speed is the real question. In Chrome on the M4 Pro, for 4,194,304 u32 keys:

| method | time (ms) |
|---|---|
| radix | 5.8 |
| bitonic, workgroup memory | 12.9 |
| bitonic, global | 28.8 |
| browser sort (CPU) | 71.5 |

That is the predicted order, and Safari on the same chip gave the same one. Global bitonic was faster
than the CPU from 131,072 keys on, the smallest size it could time well. But moving the keys there and
back took another 5.5 ms (2.8 ms up, 2.7 ms down). A GPU sort pays most when the data is already on the GPU and stays there.

> **GPU:** In Chrome, the fitted slopes of the GPU times against $N$ on a log-log plot were 0.6 to 0.9,
> flatter than the 1.0 to 1.2 the work predicts. That does not mean a better method: at these sizes a
> fixed cost per dispatch is still a large part of the time.

## Try this

- In [Explore](#parallel-sort), press **Step** with 1024 bars and read the stage number. Does the
  total change when you set **Data** to **Reversed** or **Nearly sorted**?
- Press **Run benchmark** for each **Algorithm** at 1,048,576 keys. Then pick **1,000,000** for **N**:
  how much does the padding cost bitonic sort? Does radix care?
- Switch **Keys** to **f32**. How much do the extra encode and decode passes cost?
- Run the experiment in [Measure](#parallel-sort/measure) and look at where each GPU line crosses the
  CPU line.

## Going further

The Advanced level of this page (choose **Advanced** in the level menu at the top) shows how each thread
finds its pair, the scan and the stable scatter in detail, why the CPU wins on a software GPU, and how
the slopes, crossovers and rankings are decided.
