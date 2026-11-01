## Overview

Vector addition, $c_i = a_i + b_i$ for $i = 0 \dots N-1$, is the "hello world" of GPU computing: one
thread per element, no communication between threads, and one floating-point operation per element.
The example adds two vectors of one million `f32` values and reads the sum back.

It is worth studying precisely because there is almost nothing to compute. Every cost you see is the cost
of *moving data* and of *talking to the GPU*: creating buffers, uploading the inputs, launching the
kernel, waiting for it, and copying the result back. For a GPU programmer it is the cleanest way to see
the launch overhead and the memory bandwidth of a device. For a computational scientist it is the
extreme case of a **memory-bound** kernel, the bottom left corner of the roofline model.

## The model

Each element is independent:

$$
c_i = \mathrm{fl}(a_i + b_i), \qquad i = 0, 1, \dots, N - 1
$$

where $\mathrm{fl}$ rounds the exact sum to the nearest `f32`. The work is $N$ additions; the data moved
is $8N$ bytes read and $4N$ bytes written. The ratio of the two, the **arithmetic intensity**, is

$$
I = \frac{N \text{ flops}}{12 N \text{ bytes}} \approx 0.083 \text{ flop/byte}
$$

A GPU that does $10^{13}$ flop/s and moves $5 \times 10^{11}$ bytes/s needs an intensity of about 20 to
keep its arithmetic units busy. At 0.083, the adder idles more than 99% of the time, waiting for memory,
whatever the GPU. The only number that matters is the bandwidth, and the measure of success is how
close the kernel gets to it.

## Mapping to the GPU

### Buffers and the dispatch

| Binding | WGSL | Contents |
|---|---|---|
| 0 | `var<storage, read> input1: array<f32>` | $a$, $4N$ bytes |
| 1 | `var<storage, read> input2: array<f32>` | $b$, $4N$ bytes |
| 2 | `var<storage, read_write> output: array<f32>` | $c$, $4N$ bytes |

The shader runs in workgroups of 64 invocations (`@workgroup_size(64)`), and the host dispatches
$\lceil N/64 \rceil$ of them, 15,625 for one million elements. Invocation `global_id.x` handles element
`global_id.x`. The last workgroup may be partly outside the array, so the shader guards the access with
`index < arrayLength(&input1)`; the runtime-sized arrays know their own length.

Consecutive invocations touch consecutive 4-byte words, so the 32 or 64 threads of one SIMD group read
one contiguous block of memory: the loads are **coalesced**, which is the access pattern GPU memory
systems are built for.

> **GPU:** a dispatch has at most 65,535 workgroups per dimension (the default
> `maxComputeWorkgroupsPerDimension`). With 64-wide workgroups and only `global_id.x` used, one dispatch
> can cover at most $65{,}535 \times 64 = 4{,}194{,}240$ elements. Larger vectors need a 2D dispatch or a
> loop inside the shader (a "grid-stride" loop).

### What the one-shot run does, and what it times

`runVectorAddition` fills the inputs with $a_i = i$ and $b_i = 2i$, creates the three buffers (the
inputs are written while mapped at creation), compiles the shader, builds an explicit bind group layout
and pipeline, records one compute pass, submits it, and copies the output to a `MAP_READ` buffer to read
it on the CPU. It checks the first 100 sums. Its "computation time" is the wall-clock time of **all of
that**, including filling the arrays in JavaScript and compiling the shader. The GPU time next to it
comes from timestamp queries around the compute pass alone (when the browser offers `timestamp-query`).

The Measure tab separates the pieces. `measureVectorAdd` creates the buffers and the pipeline first,
then times three things: the kernel by GPU timestamps, the kernel as the CPU sees it (submit until
`onSubmittedWorkDone`), and a **round trip** that uploads both inputs with `writeBuffer`, runs the
kernel, and copies the result back.

## Numerical behaviour

> **Numerics:** WGSL requires `f32` addition to be correctly rounded. The GPU result must therefore
> equal the exact sum rounded once to `f32`, bit for bit, with a relative error of at most
> $u = 2^{-24} \approx 6 \times 10^{-8}$.

The exact sum of two `f32` values fits in a `double` (for operands of similar magnitude), so the CPU
reference is simply `Math.fround(a + b)`, and any difference is a bug, not noise. The experiment counts
those differences at every size; in CI, on a software GPU, there were none at any $N$ up to 4,194,240,
and the largest relative error was $5.96 \times 10^{-8}$, just under $u$. The example's own inputs are integers below $2^{24}$, which `f32`
represents exactly, so its sums are exact too.

One thing may still differ between devices: the WGSL specification allows implementations to flush
subnormal numbers (below about $1.2 \times 10^{-38}$) to zero. The inputs here never get that small, and
a single addition gives a compiler nothing to reorder or fuse, so every correct implementation must
produce the same bits.

## Why it maps well — or poorly — to a GPU

**It maps perfectly, and it is still slow.** The kernel is embarrassingly parallel, branch-free apart from
the bounds check, and perfectly coalesced, so it runs at close to the memory bandwidth of the device
once $N$ is large enough to fill it. But the bandwidth is all it can reach: a CPU streams two arrays at
tens of GB/s, a GPU at hundreds, and neither is limited by the adder.

**Small vectors are pure overhead.** A dispatch has a fixed cost (encoding, submission, the driver,
scheduling the workgroups) of tens of microseconds or more. Below some size, $10^4$ to $10^6$ elements
depending on the GPU, the kernel time barely changes with $N$, so the effective bandwidth grows in
proportion to $N$. On the CPU-emulated GPU in CI the kernel took 0.18 ms for 1,024 elements and 0.25 ms
for 16,384, 16 times more work for 40% more time.

**The round trip dominates everything.** Uploading $8N$ bytes and reading back $4N$ costs more than the
kernel, by a large factor on a real GPU, because the data crosses from JavaScript to the GPU process and,
on a discrete GPU, over PCIe. Even in CI, where the "GPU" memory is ordinary RAM, the round trip for
4,194,240 elements took 49 ms against 16 ms for the kernel. A computation worth doing on the GPU keeps its data there for many kernels: one
addition does not pay for its transfer. That is the lesson every other example on this page builds on.

## Try this

> **Try this:** run [Explore](#vector-addition) and compare the GPU time with the "computation time".
> The difference is setup and transfer, not arithmetic.

- Run [size vs bandwidth](#vector-addition/measure). Find the $N$ where the kernel GB/s stops growing:
  that is where your GPU is full. Compare its plateau with the published memory bandwidth of your GPU.
- On the same chart, compare the round-trip line with the kernel line. How many additions per element
  would the kernel need before it cost as much as the transfer?
- Check the last column: every sum should equal the correctly rounded `f32` sum. If one does not, the
  browser's GPU stack has a bug worth reporting.

## References

- S. Williams, A. Waterman and D. Patterson, [Roofline: an insightful visual performance model for multicore architectures](https://doi.org/10.1145/1498765.1498785),
  *Communications of the ACM* 52(4), 65 (2009): arithmetic intensity and memory-bound kernels.
- J. D. McCalpin, [Memory bandwidth and machine balance in current high performance computers](https://www.cs.virginia.edu/stream/ref.html),
  IEEE TCCA Newsletter (1995): the STREAM benchmarks, of which this kernel is the "add" case.
- [WebGPU Shading Language](https://www.w3.org/TR/WGSL/), W3C: floating-point accuracy of `f32` operations
  and the limits on dispatch sizes.
- N. J. Higham, [Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027),
  2nd ed., SIAM (2002), chapter 2: the model $\mathrm{fl}(x + y) = (x + y)(1 + \delta)$, $|\delta| \le u$.

## Read the code

- [`js/examples/vector-addition.js`](js/examples/vector-addition.js) has the shader, `runVectorAddition`
  (the one-shot example) and `measureVectorAdd` (the compute-only version the Measure tab uses).
- [`js/teach/experiments/vector-addition.js`](js/teach/experiments/vector-addition.js) has the
  experiment and its rounding check.
- [`test/vector-addition.test.mjs`](test/vector-addition.test.mjs) and
  [`test/vector-addition-measure.test.mjs`](test/vector-addition-measure.test.mjs) check the shipped
  JavaScript on the CPU; [`test/gpu/vector-addition-measure.gpu.test.mjs`](test/gpu/vector-addition-measure.gpu.test.mjs)
  runs the shader on a software GPU in CI.
