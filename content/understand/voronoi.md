## Overview

Scatter some points (the **seeds**) over a plane and color every location by its nearest seed: the result
is a **Voronoi diagram**. It appears wherever "closest" matters, from cell biology and crystal grains to
cell-tower coverage, path planning and procedural textures, and its companion, the **distance field**
(the distance to the nearest seed at every point), is a workhorse of graphics and image processing.

On a pixel grid the obvious method tests every seed at every pixel: $P \cdot S$ distance evaluations for
$P$ pixels and $S$ seeds. This example uses the **Jump Flooding Algorithm** (JFA) of Rong and Tan
(2006) instead: about $\log_2 N$ passes over the grid, each pixel looking at only 9 others per pass, so
the cost does not depend on the number of seeds at all. On a 1024 × 768 grid that is 10 passes.

The price is that JFA is an *approximation*: on a small fraction of pixels it picks a seed that is not
the nearest. The example can count those pixels exactly with a brute-force pass on the GPU, and the
Measure tab turns that count into an error rate for each metric, with and without the extra pass of
**JFA+1**.

## The model

### Voronoi cells and metrics

The cell of seed $\mathbf{s}_i$ is the set of points no farther from it than from any other seed:

$$
V_i = \left\{ \mathbf{x} : d(\mathbf{x}, \mathbf{s}_i) \le d(\mathbf{x}, \mathbf{s}_j) \text{ for all } j \right\}.
$$

The example offers three metrics, with $\Delta = \mathbf{x} - \mathbf{s}$:

$$
d_2 = \sqrt{\Delta_x^2 + \Delta_y^2}, \qquad d_1 = |\Delta_x| + |\Delta_y|, \qquad d_\infty = \max(|\Delta_x|, |\Delta_y|).
$$

Euclidean cells are convex polygons. Manhattan and Chebyshev cells have edges at 0°, 45° and 90° and can
have whole regions of ties; ties go to the lower seed index. For the Euclidean metric the shader compares
squared distances, which order seeds the same way without a square root.

### Jump flooding

Let $N$ be the smallest power of two with $N \ge \max(w, h)$. The **step schedule** is

$$
k = \frac{N}{2}, \frac{N}{4}, \dots, 2, 1 \qquad (\log_2 N \text{ passes; JFA+1 appends one more } k = 1).
$$

Every pixel stores the index of the best seed it has seen. Initially only the seeds' own pixels know a
seed. In the pass with step $k$, pixel $\mathbf{p}$ looks at the seeds stored at itself and at its 8
neighbours $\mathbf{p} + k\,(\delta_x, \delta_y)$ with $\delta_x, \delta_y \in \{-1, 0, 1\}$, and keeps the
one nearest to its own center:

$$
\text{id}'(\mathbf{p}) = \arg\min_{\mathbf{q} \in \{\mathbf{p} + k\boldsymbol{\delta}\}} d\big(\mathbf{p}, \mathbf{s}_{\text{id}(\mathbf{q})}\big).
$$

Any offset with $|\Delta_x|, |\Delta_y| < N$ is a signed sum of the steps $N/2, \dots, 1$ in each axis, so a
seed's index can reach every pixel of the grid in $\log_2 N$ passes, where a step-1 flood would need $N$.

### Why it can err

A seed reaches pixel $\mathbf{p}$ only along a chain of pixels that each *kept* that seed in an earlier
pass. If some pixel on the chain was closer to a different seed, it stored that one instead, and the
information is lost: $\mathbf{p}$ ends up with the second-best seed. This happens where cells are thin
or where a seed's cell is cut off from it by others on the coarse grid of an early pass, so the wrong
pixels sit along cell borders, and the seed they got is only slightly farther than the right one. One
more pass at $k = 1$ (JFA+1) lets each pixel check its immediate neighbours again and repairs most of
them.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `ids` (×2) | `u32` | $w h$ | seed index per pixel, ping-pong between passes (`0xFFFFFFFF` = none yet) |
| `seeds` | `Seed`, 16 B | 1,024 | position in pixels (`vec2<f32>`), packed RGBA color |
| `params` | `Params`, 32 B | 16 slots | grid size, step $k$, metric, seed count, view, … |
| `counters` | `atomic<u32>` | 2 | exact check: wrong pixels, differing pixels |

Each flood pass needs its own $k$, so `params` holds one 32-byte copy per pass in 256-byte slots, and each
dispatch is bound with a dynamic offset. All passes of a frame go into one compute pass.

### Dispatch sequence

1. **`cs_clear`** sets every pixel to "no seed" (8 × 8 workgroups).
2. **`cs_seed`**, one thread per seed, writes the seed's index into its pixel with `atomicMin`, so if two
   seeds share a pixel the lower index wins no matter how the threads are scheduled.
3. **`cs_flood`**, $\log_2 N$ (or $\log_2 N + 1$) times: one thread per pixel reads 9 indices from the
   input buffer, evaluates up to 9 distances, and writes the best index to the output buffer. Ping-pong
   buffers mean no thread ever reads a value written in the same pass.
4. A full-screen fragment pass colors the cells, the distance field or both, straight from the ids.

**Exact check.** `cs_exact` tests every seed at every pixel ($P \cdot S$ work, the method JFA avoids) and
compares with the JFA result, counting pixels whose JFA seed is strictly farther (`counters[0]`) and
pixels whose index merely differs (`counters[1]`, which also counts exact ties). Only those 8 bytes are
read back.

> **GPU:** a reduction to a count does not need the data on the CPU. Two atomic counters turn a
> 786,432-pixel comparison into an 8-byte readback.

## Numerical behaviour

> **Numerics:** JFA's errors are rare, small and at cell borders: in CI about one pixel in 7,000 at
> 1,024 seeds; where the float64 check measured it, the wrong seed was at most 1.4 pixels farther than
> the nearest one. JFA+1 removes most of the rest.

**Error rates.** The software-GPU run in CI counted wrong pixels on a 512 × 512 grid, summed over three
uniform random seed sets per row (786,432 pixels):

| seeds | 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|
| Euclidean, JFA / JFA+1 | 0 / 0 | 4 / 0 | 32 / 2 | 102 / 14 |
| Manhattan, JFA / JFA+1 | 0 / 0 | 0 / 0 | 2 / 0 | 40 / 18 |
| Chebyshev, JFA / JFA+1 | 10 / 1 | 14 / 6 | 80 / 26 | 122 / 12 |

More seeds mean smaller cells, more borders, and more chances for a chain to be cut. That matches
Rong and Tan's observation that errors are well under 1% and mostly fixed by JFA+1. Wherever the
float64 CPU check also ran (up to 64 seeds), it counted exactly the same pixels as the GPU check.

**Seeds that share a pixel.** The seed pass stores one index per pixel, so a second seed in the same
pixel vanishes, and its whole cell counts as errors. With uniformly random seeds this is the birthday
problem: 256 seeds on 262,144 pixels collide with probability about $S^2 / 2P \approx 12\%$. The first CI
run of the experiment hit exactly that, with 1,736 wrong pixels instead of 32; the experiment now places
every seed in its own pixel. The **Random seeds** button in [Explore](#voronoi) can collide the same way,
rarely; **Exact check** then reports a whole cell of wrong pixels.

**Precision.** Distances are compared in `f32`. On a 1024 × 768 grid a squared distance reaches about
$1.6 \times 10^{6}$, where `f32` resolves steps of about 0.1, so two seeds at almost the same distance can
be ordered differently than in float64. Such pixels are genuine ties at pixel resolution; the CPU check
allows a relative tolerance of $10^{-6}$ for them.

## Why it maps well — or poorly — to a GPU

**Each pass maps very well.** One thread per pixel, a fixed 9-point stencil, no branches beyond bounds
checks, and no communication within a pass. The work is $\mathcal{O}(P \log N)$ regardless of the
number of seeds: at 1,024 seeds that is about 90 distance evaluations per pixel against 1,024 for brute
force.

**Memory access is the weak spot.** The 9 reads are $k$ pixels apart. For small $k$ they hit the same
cache lines as the neighbouring threads; for $k$ = 512 every neighbour is in a different row far away,
so the early passes are scattered gathers. Each pass also writes the whole grid. The cost per pixel and
pass is therefore roughly constant, and the [grid size experiment](#voronoi/measure) measures it.

**The exact answer is sequential or quadratic.** Fortune's sweep-line algorithm computes the exact
diagram in $\mathcal{O}(S \log S)$ but is inherently sequential; brute force is parallel but
$\mathcal{O}(P S)$. JFA trades a tiny, measurable error for a method that is both parallel and
independent of $S$, and that is why it is the standard GPU method.

## Try this

> **Try this:** in [Explore](#voronoi), choose 1,024 seeds and the **Chebyshev** metric, press **Exact
> check**, then tick **JFA+1** and check again. The count of wrong pixels should drop several-fold.

- Run [JFA vs JFA+1](#voronoi/measure) for each metric. Which metric has the most errors on your GPU,
  and does JFA+1 fix all of them? Compare with the CI table above.
- Repeat with **Clusters**: crowded seeds make thin cells. Does the error rate rise?
- Run [grid size vs time](#voronoi/measure). Each doubling of the side adds one pass and four times the
  pixels; is the GPU ns per pixel-pass constant?
- In [Explore](#voronoi), drag a seed slowly across a neighbour in the **Distance field** view and watch
  the contour lines bend at the cell borders.

## References

- G. Rong and T.-S. Tan, [Jump flooding in GPU with applications to Voronoi diagram and distance transform](https://doi.org/10.1145/1111411.1111431),
  ACM Symposium on Interactive 3D Graphics and Games (2006).
- G. Rong and T.-S. Tan, [Variants of jump flooding algorithm for computing discrete Voronoi diagrams](https://doi.org/10.1109/ISVD.2007.41),
  International Symposium on Voronoi Diagrams (2007): JFA+1, 1+JFA and their error rates.
- F. Aurenhammer, [Voronoi diagrams: a survey of a fundamental geometric data structure](https://doi.org/10.1145/116873.116880),
  *ACM Computing Surveys* 23, 345 (1991).
- K. E. Hoff III, T. Culver, J. Keyser, M. Lin and D. Manocha, [Fast computation of generalized Voronoi diagrams using graphics hardware](https://doi.org/10.1145/311535.311567),
  SIGGRAPH (1999): the earlier rasterization approach.
- S. Fortune, [A sweepline algorithm for Voronoi diagrams](https://doi.org/10.1007/BF01840357),
  *Algorithmica* 2, 153 (1987): the exact sequential algorithm.

## Read the code

- [`js/examples/voronoi-jfa.js`](js/examples/voronoi-jfa.js) has the seed, flood, exact-check and render
  shaders with the algorithm in comments, the uniform and seed layouts in the file header, `jfaSteps` for
  the schedule, and `encodeCompute` for the dispatch sequence.
- [`js/teach/experiments/voronoi.js`](js/teach/experiments/voronoi.js) has the two Measure experiments,
  built on `measureJFA` (compute-only, at the end of the example file).
- [`js/refs/voronoi.js`](js/refs/voronoi.js) has the float64 brute force used for the CPU check.
- [`test/voronoi-jfa.test.mjs`](test/voronoi-jfa.test.mjs) checks a CPU replica of the shaders;
  [`test/gpu/voronoi-jfa.gpu.test.mjs`](test/gpu/voronoi-jfa.gpu.test.mjs) and
  [`test/gpu/voronoi-measure.gpu.test.mjs`](test/gpu/voronoi-measure.gpu.test.mjs) run the shipped WGSL on
  a software GPU in CI. The second one is the source of the CI numbers on this page.
