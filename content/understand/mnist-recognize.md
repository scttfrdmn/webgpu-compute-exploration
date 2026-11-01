## Overview

Draw a digit and a convolutional neural network classifies it on the GPU, showing you the 28 × 28 image
it actually sees and the 24 activation maps of its two convolution layers. The network and its kernels
are the ones that [MNIST: Train](#mnist-train) trains in the browser; by default it uses weights trained
on the CPU by the same algorithm and cached in the repository, which recognize 98.00% of the 2,000 test
images. The [Train page](#mnist-train/understand) explains how the weights are learned; this one is about
**inference**: what the forward pass computes, how a drawing becomes a network input, and how to judge
the network's accuracy.

For a GPU programmer, the lesson is latency: one classification is a few hundred thousand multiply-adds,
far too little to keep a GPU busy, so the time you wait is the round trip of a submit and a readback. For
a computational scientist it is a lesson in measurement: an accuracy measured on 2,000 images is an
estimate with an error bar, and the GPU's f32 arithmetic should agree with an exact float64 computation.

## The model

### The forward pass

The input $x$ is the 28 × 28 image with values in $[0, 1]$. The first layer correlates it with 8 learned
5 × 5 filters and applies a ReLU; the second does the same with 16 filters over all 8 channels:

$$
a^{(1)}_{c,y,x} = \max\left(0,\; b^{(1)}_c + \sum_{u=0}^{4}\sum_{v=0}^{4} W^{(1)}_{c,u,v}\, x_{y+u,\,x+v}\right),
\qquad
a^{(2)}_{o,y,x} = \max\left(0,\; b^{(2)}_o + \sum_{c=0}^{7}\sum_{u,v} W^{(2)}_{o,c,u,v}\, p^{(1)}_{c,\,y+u,\,x+v}\right)
$$

Each is followed by a 2 × 2 **max-pool**, $p_{c,y,x} = \max_{0 \le i,j \le 1} a_{c,\,2y+i,\,2x+j}$, which
halves the resolution: 24 → 12 after the first layer, 8 → 4 after the second. The 16 × 4 × 4 = 256 pooled
values feed a dense layer, and softmax turns its 10 logits into probabilities:

$$
z_k = b^{(3)}_k + \sum_{j=0}^{255} W^{(3)}_{k,j}\, p^{(2)}_j, \qquad
P(\text{digit} = k) = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}
$$

The prediction is the most probable class. About 330,000 multiply-adds per image, 5,994 weights.

### What the layers see

The first-layer filters are small edge and stroke detectors (the Train example draws them). The maps in
this example's lower panel show where each filter responds: conv1 maps are 24 × 24 and still look like
the digit; conv2 maps are 8 × 8 combinations of strokes. Max-pooling makes the response tolerant to
shifts of a pixel or two, which is why the network can read digits that are not exactly centered.

### Preprocessing a drawing

MNIST digits were made in a particular way: each digit's bounding box was scaled to fit 20 × 20 pixels,
keeping the aspect ratio and anti-aliasing the strokes, and placed in a 28 × 28 frame so that its
**center of mass** is at the center. A drawing goes through the same steps on the CPU: the bounding box
of the ink, an area-averaged resize to fit 20 × 20 (so strokes become gray at their edges), and a shift
that moves the center of mass to (14, 14). Test images are already in this format and skip it.

> **Note:** A network only knows the distribution it was trained on. A digit drawn very thin, very thick,
> rotated, or written in a style the training set lacks is *out of distribution*, and the network may be
> confidently wrong. The probabilities are not a calibrated measure of certainty for such inputs.

## Mapping to the GPU

### One inference, one submit

The network runs with batch 1 through the same `MNISTGPU` driver as training, in **mode 2**: the CPU
writes the 784 input floats straight into the work buffer, so there is no gather and no label.

1. `queue.writeBuffer` of the 3 KB input.
2. One compute pass: `cs_conv1`, `cs_pool1`, `cs_conv2`, `cs_pool2`, `cs_dense`, `cs_softmax`, and
   `cs_mapmax`, which finds each activation map's maximum so the display can scale it.
3. A render pass that draws the input and all 24 ReLU maps directly from the work buffer: the
   activations never leave the GPU.
4. A copy of the 10 probabilities (40 bytes) to a mappable buffer, read back for the confidence bars.

One inference is in flight at a time. While you draw, a new one starts at most every 120 ms; if one is
still running, only the latest drawing is queued.

> **GPU:** every kernel is one thread per output value (4,608 for conv1, 1,024 for conv2, 10 for the
> dense layer), in workgroups of 64. At batch 1 that is at most 72 workgroups per dispatch, a small
> fraction of a GPU. The forward pass costs microseconds; the readback round trip costs much more.

### Evaluation in batches

The [Measure](#mnist-recognize/measure) tab evaluates the test set in batches of up to 128 in
**mode 1**: the test images are uploaded once and gathered by index, the forward pass runs for every
image of the batch at once, and a single thread per batch adds the loss and the number of correct
predictions to a 32-byte statistics buffer. The probabilities of every image are copied out after each
batch so they can be compared with the CPU, one readback for the whole set.

## Numerical behaviour

> **Numerics:** the GPU computes in f32 and the reference in float64. In CI, the shipped kernels and the
> float64 forward pass of `js/refs/mnist.js` predict the same digit for all 2,000 test images, and their
> probabilities differ by at most $1.0 \times 10^{-6}$.

**How precise is 98.00%?** Each test image is a Bernoulli trial: right or wrong. With accuracy $p$ on $n$
images, the binomial standard error is

$$
\sigma_p = \sqrt{\frac{p\,(1 - p)}{n}}, \qquad n = 2000,\; p = 0.98: \quad \sigma_p = 0.31\%
$$

so the 95% interval is roughly 97.4% to 98.6%. Measured on the first $n$ test images in CI:

| test images | 125 | 250 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|---|
| accuracy | 100% | 99.6% | 98.6% | 98.4% | 98.0% |

The first images happen to be easy; all the values are compatible with 98% within about two standard
errors. Two models that differ by 0.3 percentage points cannot be told apart on 2,000 images. The full
MNIST test set has 10,000 images, which would shrink the error bar by a factor of √5.

**Softmax and saturation.** The softmax is computed with the largest logit subtracted first
(log-sum-exp), so it cannot overflow. A confident prediction has probabilities like 0.9999 and $10^{-6}$;
f32 resolves them, but the bars round to 100% and 0%.

## Why it maps well — or poorly — to a GPU

**Well:** a convolution is a dot product per output value with no dependencies between outputs, and the
batch dimension multiplies the parallelism: 128 images at once is 590,000 conv1 threads.

**Poorly:** a single classification is tiny. Most of the 7 dispatches launch a few workgroups, the dense
layer launches one, and the CPU then waits for 40 bytes. On a GPU that time is the latency of a submit
and a readback, typically a fraction of a millisecond to a few milliseconds, and almost none of it is
arithmetic. That is why inference services batch requests: the latency of one request barely changes,
and the throughput grows with the batch until the GPU is full.

> **GPU:** in CI the cost per image fell about 3× from batch 1 to batch 128, even though the "GPU" there
> is emulated on a CPU core pool, where every thread costs real time. Run
> [Inference latency vs batch size](#mnist-recognize/measure) to find the batch that fills your GPU.

## Try this

> **Try this:** in [Explore](#mnist-recognize), draw a 7 slowly and watch the probabilities change as
> the stroke grows, then a 1 and a 7 that look alike. The conv2 maps show which stroke combinations
> separate them.

- Press **Random test digit** a few times; then run [Test accuracy of the cached
  weights](#mnist-recognize/measure) and check that your GPU reproduces 1,960 / 2,000 and agrees with the
  float64 CPU on every image.
- Draw a digit small in a corner of the square: preprocessing scales and centers it, so the network sees
  the same 28 × 28 input as for a large one.
- Train for one epoch in [MNIST: Train](#mnist-train), press **Use in Recognize**, and draw the same
  digits again. Then press **Use cached weights** to switch back.
- Run [Inference latency vs batch size](#mnist-recognize/measure). Where does the time per submit start
  to grow with the batch?

## References

- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791),
  *Proc. IEEE* 86, 2278 (1998): the MNIST data, its preprocessing, and convolutional networks.
- V. Nair and G. E. Hinton, [Rectified linear units improve restricted Boltzmann machines](https://icml.cc/Conferences/2010/papers/432.pdf),
  ICML (2010): the ReLU.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, [On calibration of modern neural networks](https://proceedings.mlr.press/v70/guo17a.html),
  ICML (2017): why softmax confidence is not certainty.
- V. Dumoulin and F. Visin, [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) (2016):
  output sizes of valid convolutions and pooling.

## Read the code

- [`js/examples/mnist-recognize.js`](js/examples/mnist-recognize.js) has the preprocessing
  (`preprocessDrawing`), the one-submit inference and the activation-map shader.
- [`js/examples/mnist-model.js`](js/examples/mnist-model.js) has the network kernels shared with training
  and, at its end, `evaluateWeights` and `measureInference`, the compute-only paths of the
  [Measure](#mnist-recognize/measure) tab.
- [`js/refs/mnist.js`](js/refs/mnist.js) is the float64 forward pass written out from the architecture;
  [`js/teach/experiments/mnist-recognize.js`](js/teach/experiments/mnist-recognize.js) compares the GPU
  with it.
- [`test/mnist-recognize.test.mjs`](test/mnist-recognize.test.mjs) checks the preprocessing;
  [`test/gpu/mnist-recognize-measure.gpu.test.mjs`](test/gpu/mnist-recognize-measure.gpu.test.mjs) runs
  the shipped kernels on a software GPU in CI and is the source of the CI numbers on this page.
