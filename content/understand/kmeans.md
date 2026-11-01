## Overview

$k$-means clustering splits $N$ points into $K$ groups so that each point is close to the center of its
group. It is one of the most used algorithms in data analysis, and its standard solver, **Lloyd's
algorithm**, alternates two simple steps until nothing changes: give every point to its nearest center,
then move every center to the mean of its points.

This example runs Lloyd's algorithm on up to 500,000 points in 2D, one iteration per step, entirely on the
GPU. The assignment step is embarrassingly parallel. The update step is a **reduction**: summing hundreds
of thousands of coordinates into $K$ means, which on a GPU means a tree of partial sums in workgroup
memory, because WGSL has no floating-point atomics. The only data the CPU reads back is an 80-byte
summary per iteration.

For a computational scientist, $k$-means is also a good example of an optimization method with a
guaranteed descent and no guarantee of the best answer: where it ends depends on where it starts.

## The model

### The objective

Given points $\mathbf{x}_1, \dots, \mathbf{x}_N$ and centers $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$,
$k$-means minimizes the within-cluster sum of squares, the **inertia**

$$
J(c, \boldsymbol{\mu}) = \sum_{i=1}^{N} \left\| \mathbf{x}_i - \boldsymbol{\mu}_{c(i)} \right\|^2 ,
$$

over the assignment $c(i) \in \{1, \dots, K\}$ and the centers. Finding the global minimum is NP-hard,
even in the plane.

### Lloyd's iteration

Each iteration lowers $J$ in two steps:

$$
c(i) \leftarrow \arg\min_{j} \left\| \mathbf{x}_i - \boldsymbol{\mu}_j \right\|^2 ,
\qquad
\boldsymbol{\mu}_j \leftarrow \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{x}_i ,
\quad C_j = \{ i : c(i) = j \}.
$$

The assignment cannot increase $J$ (each point takes its best center), and neither can the update (the
mean minimizes the sum of squared distances to a fixed set of points). There are finitely many
assignments, so the iteration stops after finitely many steps, when no label changes. That is a
**local** minimum: a fixed point of both steps, not necessarily the best clustering. Ties go to the
lower index, and an empty cluster keeps its center.

### Initialization

The result depends on the starting centers. **Random** (Forgy) initialization picks $K$ distinct data
points. **k-means++** (Arthur and Vassilvitskii, 2007) picks the first center at random and each next one
with probability proportional to $D(\mathbf{x})^2$, the squared distance to the nearest center chosen so
far. It spreads the centers out and is $\mathcal{O}(\log K)$-competitive with the optimum in expectation.
The example runs it on the CPU, on a sample of at most 20,000 points.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `points` | `vec2<f32>` | $N$ | the data, uploaded once |
| `labels` | `u32` | $N$ | current cluster of each point (`0xFFFFFFFF` before the first iteration) |
| `centroids` | `vec2<f32>` | 16 | the centers |
| `partials` | `vec4<f32>` | 512 × 16 | per workgroup and cluster: $\sum x$, $\sum y$, count, $\sum d^2$ |
| `stats` | struct | 80 bytes | changed labels (atomic), inertia, empty clusters, 16 counts |
| `trail` | `vec2<f32>` | 16 × 24 | recent center positions, for drawing |

The uniform `Params` is 16 bytes ($N$, $K$, the number of assign workgroups) and is the same for both
dispatches, so one slot suffices.

### One iteration: two dispatches

The CPU clears `stats`, then one compute pass runs:

1. **`cs_assign`**, $\lceil N / 1024 \rceil$ workgroups of 256 threads, 4 points per thread (coalesced:
   thread $t$ handles points $g \cdot 1024 + 256 i + t$). Each point finds its nearest center by squared
   distance, writes its label if it changed, and counts the change in a workgroup atomic. Then, for each
   cluster $c$, every thread forms $(\sum x, \sum y, n, \sum d^2)$ over its own points of cluster $c$, and the
   workgroup **tree-reduces** the 256 `vec4`s in workgroup memory, halving the active threads each step
   with a `workgroupBarrier()` in between. Thread 0 stores the result in `partials`.
2. **`cs_update`**, one workgroup: for each cluster, thread $t$ adds the partials of workgroups $t$ and
   $t + 256$, a second tree reduction combines the 256 results, and thread 0 divides by the count.

> **GPU:** WGSL atomics exist only for `i32` and `u32`. A float sum over many threads has to be a
> reduction: partial sums in workgroup memory, then a second pass over the partials. As a bonus, the
> result is deterministic: the same inputs give the same bits every time.

### CPU↔GPU traffic

After the pass, `stats` is copied into a mappable buffer and read: 80 bytes per iteration, with at most
one readback in flight so iterations never queue up behind it. The CPU uses it for the readout and to stop
when no label changed. Points are uploaded only when the data set changes.

## Numerical behaviour

> **Numerics:** the GPU computes distances and sums in single precision, yet in CI it reproduces a
> float64 run from the same starting centers label for label. Which local minimum it reaches depends on
> the initialization, not on the precision.

**Reductions are accurate.** Summing $n$ numbers one after another accumulates an error that can grow
like $n u$ (with $u = 2^{-24}$); a balanced tree of sums grows like $u \log_2 n$ (Higham, 1993). The
two-level reduction here is such a tree: in CI the `f32` inertia differs from a float64 sum at the same
centers by $1$ to $8 \times 10^{-8}$, and the centers from a float64 Lloyd run by less than $10^{-7}$, in a
domain of size about 2.

**Ties and near-ties.** Both implementations give ties to the lower index. A point almost equidistant from
two centers could still be assigned differently in `f32` and float64; in the CI runs (up to 100,000
points, 10 initializations) that never happened, and the labels agreed exactly at convergence.

**Local minima.** In CI, on 100,000 points drawn from 5 Gaussian blobs with $K = 5$:

| initialization | iterations to converge | final inertia |
|---|---|---|
| k-means++, 5 trials | 5, 5, 4, 34, 4 | 3,170 in four trials, 7,970 in one |
| random, 5 trials | 48, 79, 59, 48, 77 | 3,170 in three trials, about 8,000 in two |

The GPU and the float64 run took the same number of iterations and reached the same inertia in every
trial. In a worse minimum, typically one center covers two blobs while two others split one; no amount
of precision fixes that, but a better start does. That is why practical codes run several
initializations and keep the best.

**Convergence speed.** Lloyd's algorithm usually needs few iterations from a good start, and many
small moves near the end from a poor one: the number of changed labels per iteration drops from $N$ to
a handful, then to zero.

## Why it maps well — or poorly — to a GPU

**The assignment maps very well.** Every point is independent: $N K$ distance evaluations with no
communication, regular coalesced loads, and a tiny array of centers that every thread reads.

**The update is a reduction, which maps well with care.** The tree reduction does $\mathcal{O}(N)$ work in
$\mathcal{O}(\log N)$ steps, but its structure costs something: `cs_assign` runs one 8-step tree per
cluster, so its barrier count grows with $K$, and half the threads go idle at every step. `cs_update`
runs on a single workgroup, which is fine here because there are at most 512 partials per cluster.

**The arithmetic intensity is low.** With $K \le 16$, each point brings 8 bytes and needs a few dozen
floating-point operations. A large $N$ is memory-bound and a small $N$ is launch-bound, and on a
CPU-emulated GPU (the CI runner) the GPU path is slower than a plain float64 JavaScript step. On a real GPU,
[measure it](#kmeans/measure).

**Sequential across iterations.** Each iteration needs the previous centers, so iterations cannot
overlap; the per-iteration readback adds latency that only matters if you want results as fast as
possible rather than at an animation's pace.

## Try this

> **Try this:** in [Explore](#kmeans), set **True clusters** to 5 and **K** to 5, choose **Random**
> initialization, and press **Reset** a few times, then **Start**. Some runs end with two centers sharing a
> blob: a local minimum. Switch to **k-means++** and compare.

- Run [Converged inertia: GPU vs CPU](#kmeans/measure) with both initializations. Do your GPU's labels
  match the float64 run exactly, as in CI? Which trials end at a higher inertia?
- Run [Points vs time per iteration](#kmeans/measure): is the GPU time linear in $N$? Where does the GPU
  wall time overtake the CPU step?
- Run [Clusters vs time per iteration](#kmeans/measure). The distance work is linear in $K$; is the
  time? Part of it is the per-cluster reduction.
- In [Explore](#kmeans), set **Noise** to 50% with **K** = 8: the centers still converge, to a partition
  of noise that the Voronoi overlay makes visible.

## References

- S. P. Lloyd, [Least squares quantization in PCM](https://doi.org/10.1109/TIT.1982.1056489),
  *IEEE Transactions on Information Theory* 28, 129 (1982): the iteration (written in 1957).
- D. Arthur and S. Vassilvitskii, [k-means++: the advantages of careful seeding](https://dl.acm.org/doi/10.5555/1283383.1283494),
  ACM-SIAM Symposium on Discrete Algorithms (2007).
- M. Harris, [Optimizing parallel reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf),
  NVIDIA (2007): the tree reduction in shared (workgroup) memory.
- N. J. Higham, [The accuracy of floating point summation](https://doi.org/10.1137/0914050),
  *SIAM Journal on Scientific Computing* 14, 783 (1993): why pairwise sums are accurate.
- M. Mahajan, P. Nimbhorkar and K. Varadarajan, [The planar k-means problem is NP-hard](https://doi.org/10.1016/j.tcs.2010.05.034),
  *Theoretical Computer Science* 442, 13 (2012).

## Read the code

- [`js/examples/kmeans.js`](js/examples/kmeans.js) has `cs_assign` and `cs_update` with the algorithm in
  comments, the uniform and Stats layouts in the file header, and `encodeIteration` for the dispatch
  sequence. The data generator and the k-means++ seeding are plain JavaScript in the same file.
- [`js/teach/experiments/kmeans.js`](js/teach/experiments/kmeans.js) has the three Measure experiments,
  built on `measureKMeans` (compute-only, at the end of the example file).
- [`js/refs/kmeans.js`](js/refs/kmeans.js) has the float64 Lloyd step the experiments compare against.
- [`test/kmeans.test.mjs`](test/kmeans.test.mjs) checks a CPU replica of the shaders;
  [`test/gpu/kmeans.gpu.test.mjs`](test/gpu/kmeans.gpu.test.mjs) and
  [`test/gpu/kmeans-measure.gpu.test.mjs`](test/gpu/kmeans-measure.gpu.test.mjs) run the shipped WGSL on a
  software GPU in CI. The second one is the source of the CI numbers on this page.
