## Overview

The product $C = AB$ of an $M \times N$ and an $N \times K$ matrix is the most studied kernel in
high-performance computing: it is the core of dense linear algebra, of neural networks, and of the
LINPACK benchmark that ranks supercomputers. The example multiplies two $512 \times 512$ matrices of
random numbers with the simplest possible GPU kernel: one thread per element of $C$, each computing one
dot product.

That simplicity is the point. For a GPU programmer, the kernel shows 2D workgroups and index arithmetic,
and why the naive version is far from the hardware's peak: it leaves the reuse of data to the caches.
For a computational scientist, it is the model problem for floating-point summation error, a sum of $N$
products rounded one at a time.

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

## Mapping to the GPU

### Threads and indices

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

### Memory access

Inside a workgroup, invocations are numbered `x + 8 y`, so the threads of one SIMD group (32 or 64
lanes) differ mostly in `x`, which here is the **row**. In iteration $k$ they read $a_{\text{row},k}$
from eight different rows, $4N$ bytes apart (2 KB for $N = 512$): eight separate cache lines. They read
$b_{k,\text{col}}$ for only a few neighbouring columns, which the hardware broadcasts. Each of the 8
values of a row of $A$ is reused by the threads that share that row, but only through the cache.

> **GPU:** this kernel loads $2N$ numbers to do $2N$ flops, an intensity of 0.25 flop per byte from
> cache. Fast GPU matrix kernels **tile** the product instead: a workgroup loads a $T \times T$ block of
> $A$ and of $B$ into `var<workgroup>` memory once, synchronizes with `workgroupBarrier()`, and every
> thread then reuses each loaded value $T$ times. Register blocking (several outputs per thread) adds
> another factor. Together they turn the kernel from cache-bound into compute-bound.

### CPU↔GPU traffic

The one-shot run creates the buffers with the inputs mapped at creation, submits one pass, and reads $C$
back. It verifies only $c_{00}$ against a CPU dot product (with a tolerance of 0.01), and its "GFLOPS
(wall time)" includes compiling the shader and moving 3 MB each way. The Measure tab's `measureMatmul`
uploads first, times the kernel alone (GPU timestamps, and submit until done), then reads $C$ back and
checks 16 whole rows against a float64 product.

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

Two things change the error without changing the product. A driver may contract `sum + a * b` into a
fused multiply-add, which rounds once instead of twice; the bound still holds. And a different summation
order changes it: pairwise summation, or splitting the $k$ loop over several partial sums that are added
at the end, lowers the worst case from $N u$ toward $u \log_2 N$.

## Why it maps well — or poorly — to a GPU

**The parallelism is ideal.** There are $MK$ independent dot products (262,144 for the example), no
synchronization, no atomics, and uniform work per thread, since every loop runs exactly $N$ times.

**The data reuse is left on the table.** The $n^3$ work on $n^2$ data is only an advantage if each loaded
number is used many times while it is close at hand. This kernel asks memory for $2N$ numbers per output
and hopes the caches have them. Below a few hundred, the matrices fit in cache and the GPU mostly waits on
launch overhead; above that, the strided loads of $A$ limit it. A tiled kernel on the same GPU is
several times faster, and vendor libraries (cuBLAS, rocBLAS, Metal Performance Shaders) reach a large
fraction of the hardware's peak, which is why the example's GFLOPS figure is a lesson, not a benchmark of
the hardware.

**Growth is cubic.** Doubling $n$ costs eight times the work but only four times the memory, so large
products amortize the fixed costs and small ones do not.

## Try this

> **Try this:** run [Explore](#matrix-multiplication) a few times. The first run includes compiling the
> shader; compare its wall time with later runs and with the GPU time.

- Run [size vs GFLOPS](#matrix-multiplication/measure) and find where the GFLOPS stop rising. Below that,
  the dispatch overhead dominates; above it, memory access does. Compare the wall-time line with the
  $n^3$ reference.
- On the error chart, the measured error should sit far below the $N u$ line. Is it closer to
  $\sqrt{N}\,u$? (At $n = 1024$, $\sqrt{N}\,u = 1.9 \times 10^{-6}$.)
- Compare your GFLOPS with the peak `f32` throughput of your GPU. The gap is what tiling would recover.

## References

- J. J. Dongarra, P. Luszczek and A. Petitet, [The LINPACK benchmark: past, present and future](https://doi.org/10.1002/cpe.728),
  *Concurrency and Computation* 15, 803 (2003).
- V. Volkov and J. W. Demmel, [Benchmarking GPUs to tune dense linear algebra](https://doi.org/10.1109/SC.2008.5214359),
  SC08 (2008): register blocking and why naive kernels are memory-bound.
- N. J. Higham, [Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027),
  2nd ed., SIAM (2002), chapter 3: error bounds for inner products and $\gamma_n$.
- N. J. Higham and T. Mary, [A new approach to probabilistic rounding error analysis](https://doi.org/10.1137/18M1226312),
  *SIAM J. Sci. Comput.* 41, A2815 (2019): why errors grow like $\sqrt{n}\,u$.
- S. Williams, A. Waterman and D. Patterson, [Roofline: an insightful visual performance model for multicore architectures](https://doi.org/10.1145/1498765.1498785),
  *Communications of the ACM* 52(4), 65 (2009).

## Read the code

- [`js/examples/matrix-multiplication.js`](js/examples/matrix-multiplication.js) has the shader,
  `runMatrixMultiplication` and the compute-only `measureMatmul`.
- [`js/teach/experiments/matrix-multiplication.js`](js/teach/experiments/matrix-multiplication.js) has
  the experiment; [`js/refs/basic.js`](js/refs/basic.js) the float64 product (`matmulRowsF64`).
- [`test/matrix-multiplication.test.mjs`](test/matrix-multiplication.test.mjs) checks a CPU replica of
  the kernel against float64; [`test/gpu/matrix-multiplication-measure.gpu.test.mjs`](test/gpu/matrix-multiplication-measure.gpu.test.mjs)
  runs the shipped shader on a software GPU in CI.
