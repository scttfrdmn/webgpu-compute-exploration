## Overview

A concert hall turns a hand clap into a long, decaying tail: thousands of reflections arriving from
every direction. That response, recorded or simulated, is the hall's **impulse response** (IR) $h$, and
for a linear acoustic space the sound of any signal $x$ in the hall is the convolution $y = x * h$. A
convolution reverb computes that convolution for live audio. Here the IR is up to 6 seconds long, stereo,
and the convolution runs on the GPU, block by block, fast enough for real time.

Direct convolution would cost $L$ multiply-adds per output sample: at 48 kHz and a 6 s IR, 288,000 per
sample per channel, about 28 billion per second. The example instead uses **uniformly partitioned
overlap-save** FFT convolution, which does the same job for a few million operations per block. For a
computational scientist it is an exact algorithm, not an approximation, whose only error is f32
rounding, and the [Measure](#convolution-reverb/measure) tab checks it against a float64 direct
convolution. For a GPU programmer it is a **latency-bound streaming** problem: every block must make a
round trip through the GPU before the audio hardware needs it.

> **Note:** The Measure experiments run offline on synthetic noise: nothing is played and no microphone
> permission is needed.

## The model

### Convolution and the FFT

The output is $y[n] = \sum_{i} x[i]\, h[n - i]$. The convolution theorem turns it into a product of
spectra: if $X = \mathcal{F}(x)$ and $H = \mathcal{F}(h)$, then the **circular** convolution of two
length-$N$ sequences is $\mathcal{F}^{-1}(X \cdot H)$, and the FFT computes $\mathcal{F}$ in
$\mathcal{O}(N \log N)$ operations (Cooley and Tukey 1965). Circular means the result wraps around;
zero padding controls where.

### Partitioning the impulse response

A single FFT of the whole IR would force a block of the IR's length, 6 seconds of latency. Instead
$h$ is cut into $P = \lceil L / B \rceil$ partitions $h_j$ of $B$ samples (Gardner 1995; Wefers 2015).
Each is zero-padded to $N = 2B$ and transformed **once**, when the IR is loaded: $H_j = \mathcal{F}(h_j)$.
For each input block $k$ of $B$ new samples the GPU then:

1. transforms the window of the previous and current block, $X_k = \mathcal{F}([x_{k-1}, x_k])$,
   $N$ samples;
2. stores $X_k$ in a **frequency-domain delay line**, a ring of the last $P$ spectra;
3. multiply-accumulates over all partitions,

$$
Y_k = \sum_{j=0}^{P-1} X_{k-j} \cdot H_j ,
$$

4. inverse-transforms, $y = \mathcal{F}^{-1}(Y_k)$, and keeps the **last** $B$ samples.

The first $B$ samples of $y$ are corrupted by the circular wrap-around and are thrown away: that is
**overlap-save**. Because the FFT is linear, summing the partition products in the frequency domain and
transforming once gives exactly $\sum_j (x * h_j)$ delayed by $jB$, which is $x * h$.

### Two real channels in one complex FFT

Audio is real, and a real signal's spectrum is Hermitian, $X[N-k] = \overline{X[k]}$. So the left and
right channels are packed as one complex signal $z = x_L + i\,x_R$, transformed once, and separated:

$$
X_L[k] = \frac{Z[k] + \overline{Z[N-k]}}{2}, \qquad X_R[k] = \frac{Z[k] - \overline{Z[N-k]}}{2i}
$$

for bins $0 \dots N/2$ only. The filters are prepared the same way, and the multiply-accumulate builds
$Y = Y_L + i\,Y_R$ so that one inverse FFT returns the left output in the real part and the right in the
imaginary part.

### Cost

Per block and channel: one $N$-point FFT, $P\,(B + 1)$ complex multiply-adds and one inverse FFT, instead
of $B \cdot L$ multiply-adds. For $B = 1024$ and a 6 s IR at 48 kHz ($P = 282$) that is about 290,000
complex multiply-adds plus two 2,048-point FFTs per channel, against 295 million real multiply-adds for
direct convolution: roughly 200 times less arithmetic.

## Mapping to the GPU

### Buffers

| Buffer | Element | Size | Contents |
|---|---|---|---|
| `filters` | `vec4<f32>` | $P\,(B+1)$ | $(H_L, H_R)$ per bin, partition-major, prepared once per IR |
| `spectra` | `vec4<f32>` | $R\,(B+1)$ | the delay line: $(X_L, X_R)$ of the last $R$ input blocks |
| `src`, `dst` | `vec2<f32>` | $P \cdot N$ | ping-pong scratch for the FFT stages |
| `twiddles` | `vec2<f32>` | $N/2$ | $W_N^k = e^{-2\pi i k/N}$ |
| `wet` | `vec2<f32>` | $B$ | the output block (left, right) |

The ring has room for $R = \lceil 576{,}000 / B \rceil$ spectra (6 s at 96 kHz), so the buffers are
allocated once for the largest IR; about 9 MB each for the filters and the delay line.

### One block, 26 dispatches

For $B = 1024$ ($N = 2048$, 11 butterfly stages), in one compute pass with workgroups of 64:

1. `cs_pack_block` — the window into bit-reversed order (2,048 threads).
2. `cs_fft_forward` × 11 — one radix-2 butterfly stage per dispatch, 1,024 threads each. Each stage
   reads its own uniform slot at a dynamic offset, which tells it its stage number.
3. `cs_split` — separate $X_L$ and $X_R$ into the delay-line slot of this block (1,025 threads).
4. `cs_mac` — one thread per frequency bin, looping over all $P$ partitions: the whole sum
   $\sum_j X_{k-j} H_j$ in a single dispatch, with no atomics.
5. `cs_fft_inverse` × 11 and `cs_output` — inverse transform, scale by $1/N$, keep the last $B$ samples.

Then the $B$ output frames are copied to a mappable buffer and read back. Preparing an IR is the same
pipeline run once over all $P$ partitions as a batch.

> **GPU:** the FFT is not a single kernel but $\log_2 N$ dispatches, one per stage, because each stage
> needs all of the previous stage's results, and a dispatch boundary is WebGPU's only device-wide
> barrier. With $N = 2048$ the packed signal (16 KB) would just fit in one workgroup's memory, but one
> workgroup would use a small fraction of the GPU.

### Streaming

An AudioWorklet (on the audio thread) collects $B$ input frames and posts them to the main thread,
which uploads the window, encodes the block and maps the wet block back; the worklet plays block $k$ at a
fixed time, $(1 + D)\,B$ frames after it started to arrive: $B$ frames because a block must be complete
before it can be transformed, plus $D$ blocks of slack (**GPU slack** in Explore) for the round trip. A
block that is not back in time is played as silence and counted as an underrun; the GPU state never
skips a block, so the next one is correct again. The dry signal and the browser's own ConvolverNode,
for A/B comparison, are delayed by the same amount.

## Numerical behaviour

> **Numerics:** partitioned FFT convolution is exact in exact arithmetic. In f32 its error is a few
> rounding units of the output's peak, independent of the IR length: in CI, on a software GPU, the
> streamed output differs from a float64 direct convolution by 2.2 to 3.3 × 10⁻⁷ of the peak, for every
> IR length from 0.25 to 6 s and both block sizes.

**Why the error does not grow with the IR.** A radix-2 FFT of size $N$ has a relative rounding error of
about $\varepsilon \log_2 N$ (with $\varepsilon = 6 \times 10^{-8}$ for f32), and the error of a sum of
$P$ products grows at most like $P\,\varepsilon$, typically like $\sqrt{P}\,\varepsilon$ when the rounding
errors are random. But the partitions of a decaying IR are not equal: the late ones are 60 dB quieter
than the first, so they contribute little error, and the result is dominated by the first few
partitions and the two FFTs. That is why the [accuracy experiment](#convolution-reverb/measure) shows a
flat line.

**The self-test.** Whenever an IR is loaded, the example runs one block of seeded noise followed by $P$
blocks of silence through the GPU, reads back five or six blocks spread over the whole response, and compares
them with a direct CPU convolution (cheap here, because the input is a single block). CI measures 2.5 ×
10⁻⁷ of the peak; the page shows the same number for your GPU.

**Time aliasing and latency.** Overlap-save is exact only because each partition is padded to
$N = 2B$: the linear convolution of $B$ input and $B$ IR samples has $2B - 1$ samples, which fit. Uniform
partitions make the latency $B$ frames (21.3 ms at 1,024 and 48 kHz) plus the slack; non-uniform
partitions (small ones first) can lower it to a few samples at the cost of more bookkeeping.

## Why it maps well — or poorly — to a GPU

**Well:** the multiply-accumulate, which is most of the arithmetic for a long IR, is perfectly parallel
over frequency bins: thread $k$ loops over the partitions and reads bin $k$ of every spectrum, a
coalesced pattern, and nothing is shared between threads. The FFT stages are also uniform, branch-free
work.

**Poorly:** the work per block is small (about 1,000 threads per dispatch) and every block needs a
readback before the audio hardware can play it. The GPU is idle most of the time, and the latency of the
round trip, not the arithmetic, decides whether the deadline of $B / f_s$ is met. A busy GPU, for example
one that is also rendering a demanding page, can delay a block, which the worklet then plays as silence.

> **GPU:** in the [deadline experiment](#convolution-reverb/measure) compare the round trip with the
> deadline: if the ratio stays far below 100% your GPU has room for longer IRs, more channels, or a
> smaller block (lower latency). The GPU time column, when timestamps are available, shows how little of
> the round trip is computation.

## Try this

> **Try this:** in [Explore](#convolution-reverb), start the **Synth demo** in the **Cathedral**, then
> switch **A/B** between the GPU and the browser's ConvolverNode. They should sound the same; the
> self-test line in the readout shows how closely they agree numerically.

- Set **Block** to 2048 and **GPU slack** to 1 block in [Explore](#convolution-reverb), and watch the
  underrun counter. Then try 1024 with 3 blocks: lower latency, more margin per block.
- Run [IR length vs time per block](#convolution-reverb/measure) at both block sizes. How much of the
  21.3 ms (or 42.7 ms) deadline does your GPU use at 6 s?
- Run [Partitioned FFT convolution vs direct convolution](#convolution-reverb/measure): the error should
  sit near $3 \times 10^{-7}$ for every IR length.
- Load **Reverse**: the same machinery plays a tail that grows instead of decaying, because the
  convolution does not care what $h$ looks like.

## References

- W. G. Gardner, *Efficient convolution without input-output delay*, J. Audio Eng. Soc. 43, 127 (1995):
  partitioned convolution.
- F. Wefers, *Partitioned convolution algorithms for real-time auralization*, PhD thesis, RWTH Aachen,
  Logos Verlag (2015): uniformly partitioned overlap-save and the frequency-domain delay line.
- J. W. Cooley and J. W. Tukey, [An algorithm for the machine calculation of complex Fourier series](https://doi.org/10.1090/S0025-5718-1965-0178586-1),
  *Math. Comp.* 19, 297 (1965).
- T. G. Stockham, *High-speed convolution and correlation*, AFIPS Spring Joint Computer Conference
  (1966): fast convolution by sectioning.
- M. R. Schroeder, [New method of measuring reverberation time](https://doi.org/10.1121/1.1909343),
  *J. Acoust. Soc. Am.* 37, 409 (1965): the backward-integrated energy decay used for the RT60 readout.
- J. O. Smith III, [Mathematics of the discrete Fourier transform](https://ccrma.stanford.edu/~jos/mdft/),
  CCRMA (2007): the convolution theorem and the real-signal symmetries.

## Read the code

- [`js/examples/convolution-reverb.js`](js/examples/convolution-reverb.js) has the compute shader
  (`cs_pack_block`, `cs_fft_forward`, `cs_split`, `cs_mac`, `cs_output`, …) with the algorithm in
  comments, the `GPUConvolver` class (`uploadIR`, `submitBlock`, `selfTest`), the AudioWorklet bridge and
  the procedural IRs. `measureConvolver`, at the end, is the offline path of the
  [Measure](#convolution-reverb/measure) tab.
- [`js/teach/experiments/convolution-reverb.js`](js/teach/experiments/convolution-reverb.js) has the
  experiments, and [`js/refs/convolution.js`](js/refs/convolution.js) the float64 direct convolution.
- [`test/convolution-reverb.test.mjs`](test/convolution-reverb.test.mjs) runs the shipped `GPUConvolver`
  against CPU replicas of the kernels; [`test/gpu/convolution-reverb.gpu.test.mjs`](test/gpu/convolution-reverb.gpu.test.mjs)
  and [`test/gpu/convolution-reverb-measure.gpu.test.mjs`](test/gpu/convolution-reverb-measure.gpu.test.mjs)
  run the shipped shader on a software GPU in CI.
