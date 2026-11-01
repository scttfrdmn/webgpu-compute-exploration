## Overview

Sorting is the classic algorithm that does not look parallel. Quicksort and merge sort, the usual CPU
choices, branch on every comparison and move data in patterns that depend on the data. A GPU wants
the opposite: thousands of threads doing the same thing at the same time, with no data-dependent
control flow.

This example sorts up to 4,194,304 32-bit keys on the GPU in three ways and checks every result bit for
bit against the browser's built-in sort:

- **Bitonic sort** (Batcher, 1968), a *sorting network*: a fixed schedule of compare-exchange
  stages that does not depend on the data at all. $\mathcal{O}(n \log^2 n)$ work.
- **Bitonic sort with workgroup memory**: the same network, with most stages run inside 512-key blocks in
  fast on-chip memory instead of one dispatch each.
- **LSD radix sort**, 4 bits at a time: $\mathcal{O}(n)$ work in 8 passes, each built from a histogram,
  a **prefix sum** (scan) and a scatter.

The bar chart animates bitonic sort one stage at a time; the benchmark measures the real thing. The
lesson for GPU programmers is that a scan turns "where does this key go?" into a parallel question; the
lesson for everyone is that asymptotic work is only part of the cost.

## The model

### Bitonic sort

A sequence is **bitonic** if it first rises and then falls (or is a rotation of such a sequence).
Batcher's observation: comparing element $i$ with element $i + n/2$ of a bitonic sequence of length
$n$ and swapping where needed splits it into two bitonic halves, every element of the first no larger
than any element of the second. Repeating that on the halves sorts the sequence.

The full sort builds bitonic sequences of growing length $k = 2, 4, \dots, n$. For each $k$ it runs
compare-exchange stages with distances $j = k/2, k/4, \dots, 1$. In stage $(k, j)$ every index $i$ whose
bit $j$ is clear is compared with $i + j$; blocks whose bit $k$ is clear sort ascending, the others
descending. The number of stages, the **depth** of the network, is

$$
S(n) = \sum_{m=1}^{\log_2 n} m = \frac{\log_2 n \,(\log_2 n + 1)}{2},
\qquad
W(n) = \frac{n}{2} \, S(n) \text{ compare-exchanges.}
$$

At $n = 2^{20}$ that is 210 stages and $1.1 \times 10^{8}$ compare-exchanges, about 5 times the
$n \log_2 n \approx 2.1 \times 10^{7}$ comparisons of a good sequential sort. Lengths that are not a
power of two are padded with the largest key, `0xFFFFFFFF`, which sorts to the end.

### LSD radix sort

Radix sort never compares keys. It sorts by one $b$-bit digit at a time, from the least significant up,
with a **stable** counting sort, so that ties keep the order of the previous pass. After $32 / b$ passes
the keys are sorted. Here $b = 4$: 16 buckets, 8 passes.

Each pass splits the keys into tiles of 1,024 and counts the digits per tile, $c_{d,t}$. The position of
a key with digit $d$ in tile $t$ is the number of keys with a smaller digit anywhere, plus the keys with
digit $d$ in earlier tiles, plus its rank among the equal digits in its own tile:

$$
\text{pos} = \sum_{d' < d} \sum_{t'} c_{d',t'} + \sum_{t' < t} c_{d,t'} + \text{rank}_{d,t}(\text{key}).
$$

Storing the histogram **digit-major** (entry $d \cdot T + t$ for $T$ tiles) makes the first two sums
together one exclusive prefix sum over $16\,T$ numbers. That scan is the heart of the algorithm.

### Floating-point keys

To sort `f32` values with an integer sort, each bit pattern $u$ is mapped to

$$
u' = \begin{cases} u \oplus \text{0xFFFFFFFF} & \text{sign bit set (negative)} \\ u \oplus \text{0x80000000} & \text{otherwise,} \end{cases}
$$

which makes unsigned order equal float order: more negative floats get smaller keys, and every
non-negative float sorts above every negative one. One dispatch encodes before the sort and one decodes
after it.

## Mapping to the GPU

### Bitonic: one dispatch per stage

`cs_bitonic_global` runs one thread per compare-exchange pair. Thread $t$ of stage $(k, j)$ finds the
lower index by inserting a zero bit at position $\log_2 j$ of $t$, $i = \lfloor t / j \rfloor \cdot 2j + (t \operatorname{mod} j)$,
and compares `keys[i]` with `keys[i + j]`. Every index appears in exactly one pair, so no two threads touch
the same key and no synchronization is needed inside a stage. Each stage is its own dispatch, with its
$(k, j)$ in a 16-byte uniform slot selected by a dynamic offset.

### Bitonic with workgroup memory

The stages with $j \le 256$ only exchange keys inside aligned 512-key blocks. `cs_bitonic_local_sort`
loads a block into a 2 KB `var<workgroup>` array, runs all 45 stages with $k \le 512$ separated by
`workgroupBarrier()`, and writes the block back. For each larger $k$, the stages with $j \ge 512$ run
globally and `cs_bitonic_local_merge` finishes $j = 256 \dots 1$ in one dispatch. At $n = 2^{20}$ this
cuts the dispatches from 210 to 78, and most compare-exchanges never touch global memory.

### Radix: histogram, scan, scatter

Each of the 8 passes is 3 dispatches, 24 in all; keys ping-pong between two buffers.

1. **`cs_radix_histogram`**: one 256-thread workgroup per 1,024-key tile counts digits with
   `atomicAdd` on 16 counters in workgroup memory and writes them digit-major.
2. **`cs_radix_scan`**: a *single* workgroup scans all $16\,T$ counts (reduce-then-scan). Each thread
   sums a contiguous chunk serially, a Hillis-Steele scan over the 256 chunk sums takes 8 steps in
   workgroup memory, and each thread rewrites its chunk as running offsets. One workgroup can handle
   65,536 entries, which is what caps the sort at $16 \cdot 4096$ tiles $= 2^{22}$ keys.
3. **`cs_radix_scatter`**: the tile is processed in 4 rounds of 256 keys. Each thread one-hot encodes its
   digit as a 16-bit counter packed into 8 words, and a Hillis-Steele scan of those packed counters gives
   every thread the number of equal digits held by lower threads: its stable rank. The key is written to
   tile offset + rank.

> **GPU:** a scan answers "how many before me?" for every element at once, in $\log_2$ of the group
> size steps. It is the building block that lets a GPU compact, partition and sort without locks.

### Transfers and timing

The benchmark uploads the keys, sorts the same input three times, and reads the result back once. The
wall time is submit to `onSubmittedWorkDone`, and the GPU time (with `timestamp-query`) covers the sort
pass alone. Neither includes the upload or the readback, which move $4n$ bytes each way and can take as
long as the sort itself: sorting on the GPU pays when the keys are already there and stay there.

## Numerical behaviour

> **Numerics:** sorting is exact. The only questions are whether every key arrives, in order, and what
> "in order" means for floating-point values.

**Verification.** Every benchmark run compares the GPU result with `Uint32Array.prototype.sort` (or
`Float32Array.prototype.sort`) on a copy of the same input: the two must be identical bit for bit, which
means sorted *and* a permutation of the input. The CI suites check edge cases too: empty, one, two and
three keys, sorted and reversed input, all-equal keys, keys equal to the padding value, few unique
values, and lengths just below and above powers of two.

**Floats.** The key map orders $-0$ before $+0$, the infinities at the ends and denormals in their
place, exactly like `Float32Array.prototype.sort`; CI checks ±0, ±∞, ±denormals and ±`FLT_MAX`. A NaN
with the sign bit set would sort first instead of last, which is why the benchmark's inputs are finite.

**Stability.** Radix sort must be stable, or the later passes would scramble the earlier ones; the
scatter's in-order rank makes it so. Bitonic sort is not stable, which does not matter for plain keys
but would for key-value pairs with equal keys.

**Padding.** Bitonic sort of $10^6$ keys sorts $2^{20}$, 4.9% more; at $2^{20} + 1$ it would sort
twice as many. Radix sort needs no padding.

## Why it maps well — or poorly — to a GPU

**Bitonic sort maps well and does too much.** It is the perfect GPU kernel: uniform work, no
data-dependent control flow (the compare-exchange is a select), every stage fully parallel. But it does
$\mathcal{O}(n \log^2 n)$ work, and in the global variant each of the 210 stages reads and writes the whole
array: about 1.7 GB of memory traffic for 4 MB of keys at $n = 2^{20}$. Running every stage with
$j \le 256$ in workgroup memory cuts both the dispatches and the global traffic by more than half;
the [benchmark](#parallel-sort/measure) shows what that is worth on your GPU.

**Radix sort does little work and has harder parts.** 24 dispatches and $\mathcal{O}(n)$ work, but:
the scatter writes keys with the same digit to 16 scattered places per tile (poorly coalesced), the
histogram uses atomics, and the scan runs on one workgroup while the rest of the GPU waits. Production
GPU radix sorts (Merrill and Grimshaw) use 8-bit digits, multi-level or decoupled-lookback scans and
warp-level primitives to fix exactly these.

**The CPU is a strong baseline.** A single core sorts a few million 32-bit keys in well under a
second, with no transfers. On the CPU-emulated GPU in CI, wall times in ms were:

| $N$ | bitonic | bitonic, workgroup memory | radix | CPU sort |
|---|---|---|---|---|
| 16,384 | 7.9 | 9.4 | 16.2 | 0.68 |
| 65,536 | 17.4 | 27.9 | 51.1 | 3.1 |
| 262,144 | 82.9 | 109.4 | 202.8 | 12.6 |

All of them verified, and all of them upside down: there the "GPU" is a set of CPU cores, so the CPU's
own sort wins, a barrier costs more than a global memory pass, and workgroup memory is ordinary cache.
These numbers show what CI can check (correctness, dispatch counts), not how a GPU ranks the
algorithms. On a real GPU the workgroup-memory variant should beat the global one, and the crossover
with the CPU sort is the question the [benchmark](#parallel-sort/measure) answers for your hardware.

## Try this

> **Try this:** in [Explore](#parallel-sort), press **Step** a few times with 1,024 bars, then **Play**.
> The background shows the $k$-blocks and their directions; the strip below the bars shows which pairs,
> $j$ apart, are compared next.

- Run [the scaling experiment](#parallel-sort/measure) with u32 keys. At which $N$ does radix sort pass
  the CPU sort on your GPU, and does bitonic sort with workgroup memory ever catch radix sort?
- Repeat with f32 keys: the encode and decode dispatches are cheap, so the curves should barely move.
- Compare GPU time and wall time at the smallest $N$: the gap is launch and queue overhead, which
  is why every algorithm looks the same for small inputs.
- In [Explore](#parallel-sort), choose 1,000 bars: the gray bars are the padding to 1,024 and end up
  at the right.

## References

- K. E. Batcher, [Sorting networks and their applications](https://doi.org/10.1145/1468075.1468121),
  AFIPS Spring Joint Computer Conference (1968): bitonic sort.
- W. D. Hillis and G. L. Steele, [Data parallel algorithms](https://doi.org/10.1145/7902.7903),
  *Communications of the ACM* 29, 1170 (1986): the step-doubling scan used in the scan and scatter kernels.
- M. Harris, S. Sengupta and J. D. Owens, [Parallel prefix sum (scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda),
  *GPU Gems 3*, chapter 39 (2007).
- N. Satish, M. Harris and M. Garland, [Designing efficient sorting algorithms for manycore GPUs](https://doi.org/10.1109/IPDPS.2009.5161005),
  IPDPS (2009): radix sort with per-tile histograms and a global scan.
- D. Merrill and A. Grimshaw, [High performance and scalable radix sorting](https://doi.org/10.1142/S0129626411000187),
  *Parallel Processing Letters* 21, 245 (2011).
- D. E. Knuth, *The Art of Computer Programming*, vol. 3, *Sorting and Searching*, 2nd ed. (1998):
  sorting networks (section 5.3.4) and radix sorting (section 5.2.5).

## Read the code

- [`js/examples/parallel-sort.js`](js/examples/parallel-sort.js) has the bitonic, radix and key-conversion
  shaders, each with its algorithm in a comment, the uniform layouts in the file header, and
  `bitonicSchedule` for the stage list. `runBenchmark` is the dispatch sequence.
- [`js/teach/experiments/parallel-sort.js`](js/teach/experiments/parallel-sort.js) has the Measure
  experiment, built on `measureSort` (compute-only, at the end of the example file).
- [`js/refs/sort.js`](js/refs/sort.js) has the CPU sort, the float key map and the edge cases the
  suites use.
- [`test/parallel-sort.test.mjs`](test/parallel-sort.test.mjs) checks CPU replicas of the kernels;
  [`test/gpu/parallel-sort.gpu.test.mjs`](test/gpu/parallel-sort.gpu.test.mjs) and
  [`test/gpu/parallel-sort-measure.gpu.test.mjs`](test/gpu/parallel-sort-measure.gpu.test.mjs) run the
  shipped WGSL on a software GPU in CI.
