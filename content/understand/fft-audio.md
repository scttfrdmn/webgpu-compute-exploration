## Overview

A spectrum analyzer answers one question many times a second: how much of each frequency is in the
last few tens of milliseconds of sound? The tool for that is the **discrete Fourier transform** (DFT),
and the algorithm that makes it cheap is the **fast Fourier transform** (FFT), which computes the same
$N$ numbers in $\mathcal{O}(N \log N)$ operations instead of $\mathcal{O}(N^2)$.

In this example the Web Audio API hands over $N$ = 1,024, 2,048 or 4,096 time-domain samples per frame
and the GPU does everything else: a Hann window, a radix-2 Cooley-Tukey FFT with **one compute dispatch
per stage**, magnitudes in dB, 256 log-spaced frequency bands, a scrolling spectrogram and a beat
detector. The CPU uploads $4N$ bytes per frame and never reads anything back.

For a GPU programmer this is a small, clean example of a multi-pass algorithm with ping-pong buffers
and per-dispatch uniforms. For a computational scientist it is a chance to look at what single precision
does to a transform, and at the window's role in what the spectrum shows.

> **Note:** the Measure experiments use synthetic signals (tones, white noise, a chirp) generated in the
> page. They never open the microphone or play audio.

## The model

### The DFT

For samples $x_0, \dots, x_{N-1}$ the DFT is

$$
X_k = \sum_{n=0}^{N-1} x_n \, W_N^{nk}, \qquad W_N = e^{-2\pi i / N}, \qquad k = 0, \dots, N-1 .
$$

Bin $k$ is the frequency $f_k = k f_s / N$, where $f_s$ is the sample rate. For real input, $X_{N-k}$ is the
complex conjugate of $X_k$, so bins $0 \dots N/2$ carry all the information. At $f_s$ = 48 kHz and
$N$ = 2,048 a bin is 23.4 Hz wide and a frame is 42.7 ms long: finer frequency resolution costs time
resolution, the trade-off behind the **FFT size** control.

### Radix-2 Cooley-Tukey

Splitting the sum into even and odd $n$ gives two DFTs of half the size:

$$
X_k = E_k + W_N^{k} O_k, \qquad X_{k + N/2} = E_k - W_N^{k} O_k, \qquad k < N/2 ,
$$

where $E$ and $O$ are the DFTs of $x_0, x_2, \dots$ and $x_1, x_3, \dots$. Repeating the split
$\log_2 N$ times leaves DFTs of size 1. The pair of equations above is the **butterfly**: two inputs,
one complex multiplication by a **twiddle factor** $W_N^k$, two outputs. Each of the $\log_2 N$ stages
does $N/2$ butterflies, so the whole FFT is $\tfrac{N}{2} \log_2 N$ butterflies: 24,576 at $N$ = 4,096,
where the direct sum needs $N^2 \approx 1.7 \times 10^{7}$ multiply-adds.

The iterative form works bottom-up. If the input is first stored in **bit-reversed order** (index
$n$ with its $\log_2 N$ bits reversed), stage $s$ combines neighbouring blocks of size $h = 2^s$ into
blocks of size $2h$, and the output comes out in natural order.

### Window and scaling

A frame cut out of a longer signal starts and ends abruptly, which smears every tone over many bins
(**spectral leakage**). The shader multiplies by the periodic Hann window first:

$$
w_n = \tfrac{1}{2} - \tfrac{1}{2} \cos\!\left(\frac{2 \pi n}{N}\right), \qquad
|X_k|_{\text{dB}} = 20 \log_{10}\!\left(\frac{4}{N} |X_k|\right).
$$

The factor $4/N$ is $2/N$ for a one-sided spectrum times 2 for the window's coherent gain of
$\tfrac{1}{2}$, so a full-scale sine centred on a bin reads 0 dB. Its two neighbours read −6.02 dB,
and the window's sidelobes fall off quickly: a bin-centred tone is below −100 dB five bins away.

## Mapping to the GPU

### Buffers and layout

| Buffer | Element | Count | Contents |
|---|---|---|---|
| `samples` | `f32` | $N$ | this frame's time-domain samples (the only upload) |
| `complexBuffers[0/1]` | `vec2<f32>` | 4,096 each | ping-pong buffers of the FFT stages |
| `twiddles` | `vec2<f32>` | $N/2$ | $W_N^k$, computed once per size on the CPU in double precision |
| `spectrum` | `vec2<f32>` | $N/2 + 1$ | smoothed magnitude and dB per bin |
| `analysis` | struct | 4 + 256 | beat detector state and the 256 band levels |
| `spectrogram` | `f32` | 512 × 256 | ring buffer of past band columns |

The uniform `Params` (56 bytes: $N$, $\log_2 N$, stage, sample rate, smoothing, dB range, …) is
written once per stage, each copy in its own 256-byte slot, and every butterfly dispatch is bound with a
**dynamic offset** to its own slot.

> **GPU:** all `writeBuffer` calls of a frame land before any of its dispatches run, so one shared
> uniform would give every stage the last value written. One slot per dispatch is the standard fix.

### Dispatch sequence

One compute pass per frame, $1 + \log_2 N + 3$ dispatches of 64-thread workgroups (15 at $N$ = 2,048):

1. `cs_window`: one thread per sample multiplies by $w_n$ and writes to the bit-reversed index, computed
   with the WGSL built-in `reverseBits(i) >> (32 - log2n)`.
2. `cs_butterfly`, $\log_2 N$ times: thread $t$ owns one butterfly of stage $s$. With $h = 2^s$ and
   $j = t \operatorname{mod} h$ it reads $a$ = `src[i0]` and $b$ = `src[i0 + h]`, where
   $i_0 = \lfloor t / h \rfloor \cdot 2h + j$, and writes $a \pm W b$ with
   $W = W_N^{\,j N / 2h}$ from the table. Stage $s$ reads buffer $s \operatorname{mod} 2$ and writes the other one.
3. `cs_magnitude`, `cs_bands`, `cs_beat`: magnitudes and dB with exponential smoothing; 256 log-spaced
   rows from 20 Hz to 20 kHz (the loudest bin in each row, or an interpolated value where a row is
   narrower than a bin); the low-band energy compared against its running average.

No barriers are needed inside a stage, because every thread writes two outputs that no other thread
of that stage touches. Between stages, WebGPU orders the storage writes of one dispatch before the next.

### CPU↔GPU traffic

Per frame the CPU writes $4N$ bytes of samples and $256 \log_2 N$ bytes of uniforms; the render pass
reads `analysis` and `spectrogram` directly. Nothing is read back while the visualizer runs. The
Measure tab is different on purpose: it reads the complex spectrum back to compare it with a CPU DFT.

## Numerical behaviour

> **Numerics:** the butterflies of a single-precision FFT add an error of a few units in the last
> place, growing like $\log_2 N$. The window, computed in the shader with a single-precision cosine,
> can contribute more than the butterflies, depending on the GPU.

**Butterfly error.** Each stage adds a rounding error of order $u = 2^{-24} \approx 6 \times 10^{-8}$
relative to the data, so the error of a radix-2 FFT is bounded by a small multiple of $u \log_2 N$
(Higham, chapter 24). The software-GPU run in CI, on white noise, measured the error relative to the
largest bin against a float64 DFT of the same windowed input:

| $N$ | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| max error, butterflies | 8.4e-8 | 1.2e-7 | 1.6e-7 | 1.3e-7 | 1.5e-7 |
| rms error, butterflies | 3.2e-8 | 3.8e-8 | 4.7e-8 | 4.8e-8 | 4.8e-8 |
| max error, end to end | 9.2e-8 | 1.3e-7 | 1.7e-7 | 1.6e-7 | 1.9e-7 |

That is two or three rounding units, growing slowly with $N$ (the bound is a worst case; random errors
partly cancel). The CPU suite's replica of the same arithmetic gives the same 1.5 to $2 \times 10^{-7}$.
Parseval's theorem, $\sum_k |X_k|^2 = N \sum_n |x_n|^2$, holds to better than $2 \times 10^{-7}$.

**Twiddles.** The table is computed in double precision on the CPU and rounded once to `f32`,
which is more accurate than evaluating $\sin$ and $\cos$ of large arguments in the shader.

**Window.** `cs_window` evaluates $\cos(2\pi n / N)$ in `f32`. WGSL only promises an absolute error
of $2^{-11} \approx 5 \times 10^{-4}$ for `cos` on $[-\pi, \pi]$. The CI driver's `cos` is far better (the
window is within $2 \times 10^{-7}$ of float64), so its end-to-end error in the table is only slightly
above the butterflies'. On another GPU the same butterflies can come with a larger window error; the
accuracy experiment measures the two separately.

**What the error means in dB.** An error of $10^{-7}$ relative to the largest bin is a floor about
140 dB below the peak: the very bottom of the example's widest display range, and far under the
noise of any real recording. Single precision is not what limits this analyzer; leakage and the
window are.

**Leakage and resolution.** A tone between two bins spreads over its neighbours (up to 1.42 dB
lower at the peak for Hann, the scalloping loss), and two tones closer than about two bins merge. The
synth demo's chord (220, 329.63 and 554.37 Hz) is voiced so that it is resolvable at $N$ = 2,048.

## Why it maps well — or poorly — to a GPU

**Each stage maps perfectly.** $N/2$ independent butterflies with no branches and no communication
inside the stage, and a regular memory pattern. Every stage is a textbook data-parallel kernel.

**The whole FFT is far too small to fill a GPU.** At $N$ = 4,096 a stage is 2,048 threads, 32
workgroups. A GPU with tens of thousands of lanes is mostly idle, and the 16 dispatches cost more in
launch overhead than in arithmetic. In CI, on a CPU-emulated GPU, the GPU's wall time was 0.5 to 0.9 ms
at every size, while a plain float64 JavaScript FFT took 0.03 to 0.17 ms. The
[time experiment](#fft-audio/measure) shows where your GPU lands.

**Global memory between stages.** Each stage reads and writes the whole $8N$-byte array. Fast GPU
FFT libraries keep several stages in workgroup memory and use higher radices to cut the number of
passes. Here that has a WebGPU-specific limit: a 4,096-point complex `f32` array is 32 KB, twice the
default `maxComputeWorkgroupStorageSize` of 16 KB, so it cannot sit in one workgroup on every device.

**Why do it on the GPU at all, then?** Because the spectrum is drawn on the GPU. Bands, spectrogram
and beat detection run where the data already is, and nothing travels back to the CPU. Batched FFTs
(many signals or a 2D image at once) are where the GPU's throughput really pays.

> **GPU:** a transform that runs once per frame on 4,096 points is latency-bound, not
> throughput-bound. Count dispatches, not FLOPs.

## Try this

> **Try this:** start the **Synth demo** in [Explore](#fft-audio), switch the view to Spectrogram,
> and change the **FFT size** from 1,024 to 4,096. The chord's three lines sharpen; the kick drum's
> vertical stripes blur.

- Run [FFT size vs accuracy](#fft-audio/measure) with white noise. Is the butterflies' error close
  to the CI value, and does it follow the $\log_2 N$ reference? Then compare the end-to-end column:
  that is your GPU's `cos`.
- Repeat with **Three tones**: the weakest tone is 66 dB below the first. The FFT's error floor is far
  lower, so the error columns barely change.
- Run [FFT size vs time](#fft-audio/measure) and compare GPU time, GPU wall time and the CPU FFT.
  Where does the naive DFT stop being usable?
- In [Explore](#fft-audio), set **Smoothing** to 0 and then 0.9 with the microphone or a file, and
  watch how the bars trade responsiveness for stability.

## References

- J. W. Cooley and J. W. Tukey, [An algorithm for the machine calculation of complex Fourier series](https://doi.org/10.1090/S0025-5718-1965-0178586-1),
  *Mathematics of Computation* 19, 297 (1965).
- F. J. Harris, [On the use of windows for harmonic analysis with the discrete Fourier transform](https://doi.org/10.1109/PROC.1978.10837),
  *Proceedings of the IEEE* 66, 51 (1978): leakage, the Hann window, coherent gain and scalloping loss.
- N. J. Higham, [Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027),
  2nd ed., SIAM (2002), chapter 24: the error bound of the FFT in floating point.
- M. Frigo and S. G. Johnson, [The design and implementation of FFTW3](https://doi.org/10.1109/JPROC.2004.840301),
  *Proceedings of the IEEE* 93, 216 (2005): how fast FFTs choose radices and memory layouts.
- W3C, [Web Audio API](https://www.w3.org/TR/webaudio/): `AnalyserNode`, whose smoothing the
  magnitude pass imitates.

## Read the code

- [`js/examples/fft-audio.js`](js/examples/fft-audio.js) has every shader with its algorithm in a
  comment, and the `Params` layout in the file header. `encodeFFT` and `encodeAnalysis` are the dispatch
  sequence.
- [`js/teach/experiments/fft-audio.js`](js/teach/experiments/fft-audio.js) has the two Measure
  experiments and the synthetic signals, built on `measureFFT` (compute-only, at the end of the example file).
- [`js/refs/fft.js`](js/refs/fft.js) has the float64 naive DFT and the float64 radix-2 FFT the
  experiments compare against.
- [`test/fft-audio.test.mjs`](test/fft-audio.test.mjs) checks a CPU replica of the shaders;
  [`test/gpu/fft-audio.gpu.test.mjs`](test/gpu/fft-audio.gpu.test.mjs) and
  [`test/gpu/fft-audio-measure.gpu.test.mjs`](test/gpu/fft-audio-measure.gpu.test.mjs) run the shipped
  WGSL on a software GPU in CI. The second one is the source of the CI numbers on this page.
