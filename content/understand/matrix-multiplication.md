## Overview

The product $C = AB$ of an $M \times N$ and an $N \times K$ matrix is the most studied kernel in
high-performance computing: it is the core of dense linear algebra, of neural networks, and of the
LINPACK benchmark that ranks supercomputers. The example multiplies two $512 \times 512$ matrices of
random numbers with the simplest possible GPU kernel: one thread per element of $C$, each computing one
dot product.

The Measure tab puts that kernel next to two others on the same matrices: a **tiled** GPU kernel that
stages blocks of $A$ and $B$ in workgroup memory, and a **cache-blocked CPU loop** in JavaScript. Together
they show the central idea of the field: the arithmetic is the same in all three, and what differs is how
often each number is fetched from memory. For a computational scientist, matrix multiplication is also
the model problem for floating-point summation error, a sum of $N$ products rounded one at a time.

## The model

Each element is a dot product of a row of $A$ and a column of $B$:

$$
c_{ij} = \sum_{k=0}^{N-1} a_{ik}\, b_{kj}, \qquad 0 \le i < M,\; 0 \le j < K
$$

That is $MK$ dot products of length $N$, or $2MNK$ floating-point operations (one multiply and one add
per term): $2.7 \times 10^{8}$ for the example's $512^3$. The matrices are stored **row-major** in flat
arrays, $a_{ik}$ at index $iN + k$ and $b_{kj}$ at $kK + j$.

The ratio of work to data is what makes matrix multiplication special. The inputs and output are
$MN + NK + MK$ numbers, $3n^2$ for square matrices, but the work is $2n^3$. Every number is used $n$
times, so a kernel that keeps the numbers it loads close to the arithmetic units can run at the
processor's peak, unlike vector addition, which uses every number once.

### Arithmetic intensity and the roofline

The **arithmetic intensity** $I$ of a kernel is the number of floating-point operations it performs per
byte it moves from a given level of memory. The roofline model (Williams, Waterman and Patterson, 2009)
bounds the attainable throughput by the smaller of the compute peak $P$ and what the memory bandwidth $\beta$
can feed:

$$
\text{GFLOPS} \le \min\left(P,\; I \cdot \beta\right)
$$

A kernel with $I < P/\beta$, the **ridge point**, is memory-bound: a faster processor does not help it,
only fewer bytes per flop do. For current GPUs $P/\beta$ is roughly 10 to 100 flop per byte.

- **Naive kernel.** Every multiply-add loads $a_{ik}$ and $b_{kj}$ from global memory, 8 bytes for 2
  flops: $I = 0.25$ flop/byte. At $\beta = 500$ GB/s it could reach at most 125 GFLOPS, a few per cent of
  the peak of such a GPU — unless the caches serve most of those loads.
- **Tiled kernel**, tile edge $T$. A workgroup loads a $T \times T$ block of $A$ and one of $B$
  ($2T^2$ numbers, $8T^2$ bytes) and does $2T^3$ flops with them: $I = T/4$ flop/byte, $T = 16$ gives 4.
  The global traffic falls by the factor $T$; the remaining loads come from workgroup memory, which sits
  next to the arithmetic units.

So the prediction is directional: on a GPU whose caches cannot hide the naive kernel's traffic, the tiled
kernel is faster, and the larger $n$, the more the difference is the steady-state one.

## Mapping to the GPU

### The naive kernel

The shader runs in $8 \times 8$ workgroups. The host dispatches $\lceil M/8 \rceil \times \lceil K/8 \rceil$
of them (64 × 64 for the example), and invocation `global_id` = (x, y) computes $c_{xy}$:

| WGSL | Meaning |
|---|---|
| `matrixA`, `matrixB` | read-only storage, $A$ and $B$ row-major |
| `result` | read-write storage, $C$ row-major |
| `dimensions: vec3<u32>` | uniform $M, N, K$ (written through a `Uint32Array`) |
| `row = global_id.x`, `col = global_id.y` | the element this thread computes |

Threads with `row >= M` or `col >= K` return at once, so any size works. The loop then runs over
$k$ with one load from each matrix and a multiply-add per iteration, accumulating in a register, and
writes $c_{ij}$ once at the end.

Inside a workgroup, invocations are numbered `x + 8 y`, so the threads of one SIMD group (32 or 64
lanes) differ mostly in `x`, which here is the **row**. In iteration $k$ they read $a_{\text{row},k}$
from eight different rows, $4N$ bytes apart (2 KB for $N = 512$): eight separate cache lines. They read
$b_{k,\text{col}}$ for only a few neighbouring columns, which the hardware broadcasts. Each value is
reused by other threads only if it is still in cache when they need it.

### The tiled kernel

The `tiled` module (the Measure tab runs it; the one-shot example keeps the naive one) uses $16 \times 16$
workgroups, and each thread still computes one element of $C$: workgroup $(x, y)$ owns the
$16 \times 16$ block of $C$ at rows $16y \ldots 16y + 15$ and columns $16x \ldots 16x + 15$, and
thread $(l_x, l_y)$ owns row $16y + l_y$, column $16x + l_x$. Here `x` runs along **columns**, so
neighbouring threads read neighbouring addresses of $B$ and write neighbouring addresses of $C$.

The loop over $k$ is cut into $\lceil N/16 \rceil$ phases. In phase $t$:

1. Every thread loads **one** element of $A$, $a_{\text{row},\,16t + l_x}$, and one of $B$,
   $b_{16t + l_y,\,\text{col}}$, into two `var<workgroup>` arrays `tileA` and `tileB` (2 KB in all).
   Loads that fall outside the matrices store 0, which adds nothing to any sum.
2. `workgroupBarrier()`: no thread continues until all 256 have stored their elements, so the whole
   tile is visible to everyone.
3. Each thread does 16 multiply-adds from workgroup memory, $\sum_{kk} \text{tileA}[l_y][kk]\,\text{tileB}[kk][l_x]$.
4. `workgroupBarrier()` again, so that no thread overwrites a tile for phase $t + 1$ while another is
   still reading it.

Each number loaded from global memory is used by 16 threads. The barriers are legal because every thread
reaches them: the phase count depends only on the uniform $N$, and the `if` statements guard the loads and
the final store, never a barrier. The kernel has no **register blocking** (several outputs per thread,
which reuses each value from workgroup memory again from registers); fast kernels add it, together with
vectorized loads, to reach a large fraction of the peak (Volkov and Demmel, 2008).

> **GPU:** the tiled kernel also changes the access pattern: its threads read rows of $A$ and $B$
> contiguously (coalesced), where the naive kernel reads $A$ across eight rows. Part of the measured
> speed-up is therefore the access pattern, not only the reuse. Both are what "memory-bound" means.

### The CPU baseline

`matmulBlockedF32` in `js/refs/basic.js` is the loop a careful programmer would write in plain
JavaScript: for blocks of 64 values of $k$ and 64 columns $j$, every row $i$ streams the block of $B$
with the inner loop over $j$ running through contiguous memory. The $64 \times 64$ block of $B$ (16 KB)
stays in the L1/L2 cache while all rows use it; without the blocking (one block of the full width) the
same loop was about 1.9× slower at $n = 512$ on a laptop. It stores $C$ in `Float32Array`s like the GPU.

It is **one core, no SIMD, no packing**. An optimized BLAS (OpenBLAS, MKL, Apple Accelerate) on the same
CPU uses all cores, vector units and register blocking, and is typically 10 to 100 times faster. So the
comparison answers "is the GPU worth it compared with the JavaScript I would otherwise write", not "is
the GPU faster than the CPU". The CPU is capped at $n = 512$ ($2.7 \times 10^8$ flops, about 0.1 s at the
2.4 GFLOPS one core of a 2023 laptop reaches).

### CPU↔GPU traffic

The one-shot run creates the buffers with the inputs mapped at creation, submits one pass, and reads $C$
back. It verifies only $c_{00}$ against a CPU dot product (with a tolerance of 0.01), and its "GFLOPS
(wall time)" includes compiling the shader and moving 3 MB each way.

The Measure tab's `measureMatmul` times each step with a phase clock: **compile** (buffers, shader module,
pipeline, bind group), **upload** (the `writeBuffer` calls), **dispatch** (submit until the GPU reports the
work done; it includes the transfer of the inputs, which happens at the first use) and **readback**
(copy, map, copy out). Throughput uses the kernel's GPU time from timestamp queries when the browser
offers them, otherwise the dispatch wall time; the results table says which.

## Numerical behaviour

> **Numerics:** each $c_{ij}$ is a sum of $N$ rounded products, added one at a time. The classical bound
> is $|\hat c_{ij} - c_{ij}| \le \gamma_N \sum_k |a_{ik} b_{kj}|$ with $\gamma_N = N u / (1 - N u)$ and
> $u = 2^{-24}$. For the example's non-negative inputs the sum equals $\sum_k |a_{ik} b_{kj}|$, so the
> relative error is at most about $N u$.

The bound is a worst case: every rounding error has the same sign and the largest possible size. In
practice the errors are closer to independent, and the typical error grows like $\sqrt{N}\,u$ instead
(Higham and Mary, 2019). For $N = 512$ that is $1.3 \times 10^{-6}$ against a bound of
$3.1 \times 10^{-5}$. The shipped shader, run on a software GPU in CI, measured a largest relative error
(over 16 sampled rows) of

| $n$ | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| max relative error | 3.4e-7 | 6.1e-7 | 8.0e-7 | 1.1e-6 | 1.6e-6 |
| $\sqrt{n}\,u$ | 4.8e-7 | 6.7e-7 | 9.5e-7 | 1.3e-6 | 1.9e-6 |

growing like $\sqrt{n}$ and 40 to 600 times below the worst case. The errors are not random in one respect: with all-positive terms the running sum
grows, each new term is rounded relative to a larger total, and the late terms lose the most digits.

**Same order, same answer.** The tiled kernel adds the same terms in the same order,
$k = 0, 1, \ldots, N - 1$ (phase by phase, and in order within a phase), and its padding adds exact zeros.
So the two GPU kernels round identically, bit for bit, unless the compiler contracts `sum + a * b` into a
fused multiply-add (one rounding instead of two) in one kernel and not the other. In CI they were
identical in every element at every size tested, square and ragged ($37 \times 23 \times 19$ up to
$1024^3$).

**The CPU rounds once per term.** In JavaScript the product of two `f32` values is exact in a double
(24 + 24 significant bits fit in 53), and `C[i] += a * b` rounds the double sum to `f32` when it is stored.
That is one rounding per term, as with a fused multiply-add, so the CPU baseline is a little more accurate
than a GPU that rounds the product and the sum separately, with the same $\sqrt{N}\,u$ growth.

A different summation order changes the error: pairwise summation, or splitting the $k$ loop over several
partial sums that are added at the end, lowers the worst case from $N u$ toward $u \log_2 N$.

## Why it maps well — or poorly — to a GPU

**The parallelism is ideal.** There are $MK$ independent dot products (262,144 for the example), no
atomics, and uniform work per thread, since every loop runs exactly $N$ times.

**The data reuse has to be organized.** The $n^3$ work on $n^2$ data is only an advantage if each loaded
number is used many times while it is close at hand. The naive kernel asks memory for $2N$ numbers per
output and hopes the caches have them; the tiled kernel loads each number once per workgroup and shares it
through workgroup memory. Vendor libraries (cuBLAS, rocBLAS, Metal Performance Shaders) add register
blocking and reach a large fraction of the hardware's peak, which is why the example's GFLOPS figure is a
lesson, not a benchmark of the hardware.

**Small products are overhead.** A product of $64 \times 64$ matrices is half a million flops, a few
microseconds of GPU arithmetic, but a submit and its completion take tens to hundreds of microseconds,
and compiling a pipeline takes milliseconds. Doubling $n$ costs eight times the work but only four times
the memory, so large products amortize the fixed costs and small ones do not. A CPU has no launch cost,
which is why it wins at small $n$ and the GPU wins from some $n$ on.

**A software GPU is a CPU.** In CI the "GPU" is Mesa's lavapipe, which compiles the shader for the CI
machine's CPU and runs the workgroups on all its cores with 256-bit SIMD. Workgroup memory there is
ordinary memory and a barrier is a synchronization between threads. It still beats the JavaScript loop,
which uses one core and no SIMD, so that comparison measures the baseline's limits, not GPU hardware.

### Measured in CI (software GPU, not representative)

The whole experiment on lavapipe, wall-clock timing (no timestamp queries there), medians of 5 repeats
after a warm-up:

| $n$ | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| CPU (blocked JS) GFLOPS | 0.26 | 0.45 | 0.41 | 0.40 | — |
| naive GPU GFLOPS | 0.62 | 0.94 | 1.12 | 1.15 | 1.15 |
| tiled GPU GFLOPS | 0.76 | 0.82 | 2.03 | 1.39 | 1.48 |
| tiled vs naive | 1.22× | 0.87× | 1.81× | 1.21× | 1.28× |
| max relative error / $\sqrt{n}\,u$ (all three) | 0.71–0.72 | 0.85–0.90 | 0.84 | 0.81–0.86 | 0.86 |

The fitted log-log slopes of time against $n$ were 2.91 (95% interval 2.73 to 3.09) for the naive
kernel and 2.80 (1.67 to 3.92) for the tiled one over $n = 128 \ldots 1024$, and 2.82 (2.17 to 3.47) for
the CPU over $64 \ldots 512$: all consistent with $n^3$, the tiled one with a wide interval because its
$n = 128$ time was noisy. The tiled speed-up at $n = 1024$ was 1.28× (conservative range 1.24 to 1.33).
The CPU loop ran at 0.4 GFLOPS there, a sixth of what the same code reaches on one core of a 2023
laptop, probably because the runner's few cores were shared with lavapipe work from tests running in
parallel. At $n = 64$ the tiled kernel's phases were
compile 0.18 ms (a pipeline-cache hit after the warm-up), upload 0.02 ms, dispatch 0.69 ms and readback
0.16 ms; at $n = 1024$ the dispatch took 1.45 s and the 4 MB readback 17 ms. A later run of the same
code on another runner measured 0.75 and 0.89 GFLOPS at $n = 1024$ (speed-up 1.18×, range 1.15 to 1.70),
slopes 2.73 and 2.71, and the GPU ahead of the CPU only from $n = 256$: shared machines move these numbers
by tens of per cent, which is why each is reported with its spread. On a real GPU every
number in this table changes, most of all the GPU columns.

## Try this

> **Try this:** run [Explore](#matrix-multiplication) a few times. The first run includes compiling the
> shader; compare its wall time with later runs and with the GPU time.

- Run [CPU vs naive GPU vs tiled GPU](#matrix-multiplication/measure). On the GFLOPS chart, find where
  each GPU line stops rising: below that, the launch and synchronization overhead dominates. On the time
  chart, compare the slopes with the $n^3$ line.
- Read the **Measured vs predicted** table. The fitted slopes only use the sizes whose median time is at
  least 1 ms, because shorter measurements are launch- or timer-bound. On a fast GPU that can leave
  fewer than three sizes, and the claim stays undecided: the GPU is too fast for these sizes to show its
  $n^3$ growth.
- Check the tiled-vs-naive speed-up at $n = 1024$ and its range. The intensity argument predicts more
  than 1; the factor depends on how well your GPU's caches already served the naive kernel. The implied
  load bandwidth of the naive kernel ($8n^3$ bytes over its time) tells you which: if it exceeds your
  GPU's memory bandwidth, caches were doing the reuse.
- Look at the phases charts at $n = 64$: is the dispatch time (submit to done) larger than the kernel
  time? Where does the GPU overtake one CPU core, and does that hold with upload and readback included?
- On the error chart all three lines should sit near $\sqrt{n}\,u$; the verification allows
  $4\sqrt{n}\,u$. If the naive and tiled errors differ, your compiler contracted one of them into fused
  multiply-adds.

## References

- J. J. Dongarra, P. Luszczek and A. Petitet, [The LINPACK benchmark: past, present and future](https://doi.org/10.1002/cpe.728),
  *Concurrency and Computation* 15, 803 (2003).
- V. Volkov and J. W. Demmel, [Benchmarking GPUs to tune dense linear algebra](https://doi.org/10.1109/SC.2008.5214359),
  SC08 (2008): register blocking and why naive kernels are memory-bound.
- S. Williams, A. Waterman and D. Patterson, [Roofline: an insightful visual performance model for multicore architectures](https://doi.org/10.1145/1498765.1498785),
  *Communications of the ACM* 52(4), 65 (2009).
- K. Goto and R. A. van de Geijn, [Anatomy of high-performance matrix multiplication](https://doi.org/10.1145/1356052.1356053),
  *ACM Transactions on Mathematical Software* 34(3), 12 (2008): blocking for the CPU cache hierarchy.
- N. J. Higham, [Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027),
  2nd ed., SIAM (2002), chapter 3: error bounds for inner products and $\gamma_n$.
- N. J. Higham and T. Mary, [A new approach to probabilistic rounding error analysis](https://doi.org/10.1137/18M1226312),
  *SIAM J. Sci. Comput.* 41, A2815 (2019): why errors grow like $\sqrt{n}\,u$.

## Read the code

- [`js/examples/matrix-multiplication.js`](js/examples/matrix-multiplication.js) has both shaders (the
  tiled one with its algorithm in comments), `runMatrixMultiplication` and the compute-only
  `measureMatmul`.
- [`js/teach/experiments/matrix-multiplication.js`](js/teach/experiments/matrix-multiplication.js) has
  the experiment, its verification bound and the interpretation; [`js/refs/basic.js`](js/refs/basic.js) the
  float64 product (`matmulRowsF64`) and the CPU baseline (`matmulBlockedF32`).
- [`test/matrix-multiplication.test.mjs`](test/matrix-multiplication.test.mjs) checks CPU replicas of both
  kernels (bit-identical to each other) against float64;
  [`test/matrix-multiplication-measure.test.mjs`](test/matrix-multiplication-measure.test.mjs) the CPU
  baseline and the experiment's checks; [`test/gpu/matrix-multiplication-measure.gpu.test.mjs`](test/gpu/matrix-multiplication-measure.gpu.test.mjs)
  runs both shaders and the whole experiment on a software GPU in CI.
