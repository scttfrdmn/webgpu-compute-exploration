## Overview

This example trains a small convolutional neural network (CNN) to read handwritten digits, entirely in
compute shaders: assembling each mini-batch, the forward pass, the loss, backpropagation through every
layer, and the weight update. It learns from 10,000 MNIST training images and, after every epoch, is
tested on 2,000 images it has never seen. The same network, with weights trained on the CPU by the same
algorithm, classifies your drawings in [MNIST: Recognize](#mnist-recognize); that page explains the
forward pass and how well the network does. This one is about **learning**.

For a computational scientist, training is a stochastic optimization problem: minimizing an average loss
over the data with noisy gradient steps, and the learning rate decides whether it converges quickly,
slowly, or not at all. For a GPU programmer, it is a lesson in **race-free reductions**: thousands of
threads contribute to the gradient of the same weight, and this example adds them up without a single
atomic, in a fixed order, so a GPU run repeats the CPU trainer's run to seven digits.

> **Note:** Defaults are batch 64, learning rate $\eta_0 = 0.04$ decaying by 0.85 per epoch, momentum
> 0.9, weight decay $5 \times 10^{-4}$, ±2-pixel random shifts and 15 epochs. With seed 1 this is
> exactly the run that produced the cached weights, which reached 98.00% test accuracy.

## The model

### The network

Each 28 × 28 image (bytes scaled to $[0, 1]$) goes through two convolution layers and a dense layer:

| Layer | Output | Parameters |
|---|---|---|
| conv 5 × 5, 8 filters, ReLU | 8 × 24 × 24 | 200 + 8 |
| 2 × 2 max-pool | 8 × 12 × 12 | |
| conv 5 × 5, 16 filters (over 8 channels), ReLU | 16 × 8 × 8 | 3,200 + 16 |
| 2 × 2 max-pool | 16 × 4 × 4 = 256 | |
| dense 256 → 10, softmax | 10 probabilities | 2,560 + 10 |

That is 5,994 parameters $\mathbf{w}$. The [Recognize page](#mnist-recognize/understand) writes the
forward pass out layer by layer.

### Softmax cross-entropy

The dense layer produces logits $z_k$; softmax turns them into probabilities, and the loss of a sample
with label $y$ is the negative log-probability of the right class:

$$
p_k = \frac{e^{z_k - m}}{\sum_j e^{z_j - m}}, \qquad
\ell = -\log p_y = m + \log \sum_j e^{z_j - m} - z_y, \qquad
\frac{\partial \ell}{\partial z_k} = p_k - [k = y]
$$

with $m = \max_j z_j$ subtracted first (log-sum-exp), so no exponential can overflow. The gradient with
respect to the logits is simply "predicted minus true", the start of backpropagation.

### Backpropagation

The chain rule carries $\partial \ell / \partial z$ back through every layer. Through the dense layer,
$\partial \ell / \partial p^{(2)}_j = \sum_k W^{(3)}_{kj}\, \partial \ell / \partial z_k$. Through a max-pool,
the gradient goes only to the element that was the window's maximum; through a ReLU, only where its input
was positive. Through a convolution it is a "full" correlation with the flipped kernel. The weight
gradients are correlations of an activation with the gradient below it, for example

$$
\frac{\partial \ell}{\partial W^{(2)}_{o,c,u,v}} = \sum_{y,x} \delta^{(2)}_{o,y,x}\; p^{(1)}_{c,\,y+u,\,x+v},
\qquad \delta^{(2)} = \frac{\partial \ell}{\partial a^{(2)}}
$$

### SGD with momentum

For a mini-batch $B$ the update is (Polyak's heavy ball; Sutskever et al. 2013)

$$
\mathbf{g} = \operatorname{clip}\left(\frac{1}{|B|}\sum_{s \in B} \nabla \ell_s + \lambda \mathbf{w},\; \pm 5\right), \qquad
\mathbf{v} \leftarrow \mu \mathbf{v} + \mathbf{g}, \qquad
\mathbf{w} \leftarrow \mathbf{w} - \eta\, \mathbf{v}
$$

with $\mu = 0.9$ and $\eta = \eta_0 \cdot 0.85^{\text{epoch}}$. Momentum averages the noisy gradients
over about $1/(1 - \mu) = 10$ steps; the weight decay $\lambda$ keeps the weights small, and the clip
guards against a single divergent step. Weights start from a seeded He initialization,
$w \sim \mathcal{N}(0, 2/\text{fan-in})$, so that ReLU layers neither shrink nor blow up the signal.

## Mapping to the GPU

### Buffers

| Buffer | Size | Contents |
|---|---|---|
| `images`, `labels` | 7.8 MB + 10 KB | the training set as packed bytes, uploaded once |
| `order` | 40 KB | this epoch's shuffled sample order, written by the CPU once per epoch |
| `weights` | 48 KB | 5,994 weights followed by their 5,994 momentum values |
| `work` | 7.6 MB | every activation and its gradient for 128 samples |
| `grads` | 3.1 MB | per-sample gradients, 128 × 5,994 |
| `stats` | 32 bytes | summed loss and correct counts, the only regular readback |

### One mini-batch, 14 dispatches

All dispatches use workgroups of 64 and one thread per output element, so **every kernel is a gather**:
each thread computes one value from what the previous kernel wrote, and writes only that value.

1. `cs_gather` — one thread per (sample, pixel): fetch the image through the shuffled order and shift
   it by a hashed random $(dx, dy)$ (augmentation).
2. `cs_conv1`, `cs_pool1`, `cs_conv2`, `cs_pool2`, `cs_dense`, `cs_softmax` — the forward pass and loss.
3. `cs_stats` — one thread adds the batch's loss and correct count to `stats`.
4. `cs_dense_back`, `cs_pool2_back`, `cs_conv2_back`, `cs_pool1_back` — backpropagation to every layer.
5. `cs_grads` — one thread per (sample, parameter), 64 × 5,994 = 383,616 threads, each writing its own
   sample's gradient of its own parameter into `grads`.
6. `cs_update` — one thread per parameter sums its column of `grads` in sample order and applies the
   momentum update.

Up to 16 mini-batches are encoded into one compute pass, each with its own 48-byte uniform slot at a
dynamic offset. WebGPU makes each dispatch's writes visible to the next, so no barriers are needed.

> **GPU:** the obvious way to accumulate a gradient is for every sample's thread to `atomicAdd` into the
> weight's gradient. WGSL has no floating-point atomics, and even where hardware has them the order of
> additions would vary from run to run, so the result would not be reproducible. Writing per-sample
> gradients to separate slots and summing them in a fixed order costs 3 MB of memory and one extra
> dispatch, and makes training deterministic.

### CPU↔GPU traffic

The CPU uploads the data once, writes a 40 KB shuffle per epoch and a few hundred bytes of uniforms per
frame, and reads back 32 bytes of statistics (at most one readback in flight). The first-layer filters
are drawn straight from the weights buffer by a render pass. Only **Export** and **Use in Recognize**
read the 24 KB of weights back (**Export weights** saves them in the same file format as the cached weights).

## Numerical behaviour

> **Numerics:** the GPU kernels and the CPU trainer `tools/train-mnist.mjs` use the same initialization,
> shuffle, augmentation and summation order, the GPU in f32 and the CPU in f64 for local sums. After 40
> or 80 mini-batches in CI their weights differ by about $2 \times 10^{-7}$, and they agree on the test
> accuracy to the image (1,823 / 2,000 after 80 batches at $\eta = 0.04$).

**The learning rate.** A short run of 80 mini-batches (5,120 images, half an epoch) from the same
initialization, measured in CI and identical to the CPU trainer:

| $\eta$ | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 | 0.15 |
|---|---|---|---|---|---|---|
| test accuracy | 76.4% | 82.3% | 87.2% | 91.2% | 90.4% | 74.0% |
| test loss | 0.72 | 0.57 | 0.41 | 0.29 | 0.32 | 0.80 |

Too small a step learns slowly; too large a step overshoots, and with momentum ($\eta/(1-\mu)$ is the
effective step) the loss oscillates instead of falling. In between there is a broad optimum, and the
default 0.04 sits in it. Over 15 epochs the decay $0.85^{\text{epoch}}$ shrinks the step to a tenth of its start, which
lets the noisy SGD iterate settle into a minimum.

**Generalization.** Training accuracy is measured on augmented, shifted images while the weights change
during the epoch; test accuracy on the fixed 2,000 images after the epoch. A test accuracy far below the
training accuracy would mean overfitting; with 5,994 parameters, 10,000 images and random shifts the gap
stays small.

**Determinism.** Same seed, same run: the shuffle is a seeded Fisher-Yates on the CPU, the shifts are a
PCG hash of the epoch and the sample's position, and every sum has a fixed order. The only differences
between GPU runs on different hardware come from f32 rounding (fused multiply-adds, `exp` and `log`
implementations), which grow slowly through training.

## Why it maps well — or poorly — to a GPU

**Well:** every layer is many independent dot products of length 25 to 256, which is exactly what a GPU
does best, and the batch dimension multiplies the parallelism: `cs_grads` alone launches 383,616 threads
per batch of 64.

**Poorly:** the network is tiny. The dense layer of one sample is 10 threads, `cs_stats` is one thread,
and `cs_softmax` is one thread per sample, so many of the 14 dispatches run far below the GPU's width.
At small batch sizes a mini-batch costs about the same as a larger one, because the fixed cost of the
dispatches dominates; [Batch size vs time per mini-batch](#mnist-train/measure) shows where that stops on
your GPU. The kernels also read their inputs straight from storage buffers without tiling them in
workgroup memory, which is simple and fast enough here, but a large network would want the tiled
matrix-multiply approach of the [matrix multiplication](#matrix-multiplication) example.

## Try this

> **Try this:** in [Explore](#mnist-train), set the learning rate to 0.15 and press **Start**. The loss
> curve jumps around and test accuracy lags. Then run [Learning rate vs test
> accuracy](#mnist-train/measure) to see the whole curve, and compare your GPU's accuracies with CI's.

- Train with the defaults for two or three epochs in [Explore](#mnist-train), then press **Use in
  Recognize** and draw digits in [MNIST: Recognize](#mnist-recognize). How does it compare with the
  cached 15-epoch weights?
- Turn off **Random shifts** and train again. Training accuracy rises faster, but does test accuracy?
- Run [Batch size vs time per mini-batch](#mnist-train/measure). If the time per batch hardly changes
  from 8 to 64, your GPU is limited by the number of dispatches, not by arithmetic.
- Compare the learning-rate experiment with CI: the accuracies should match to a few tenths of a percent,
  because training is deterministic up to f32 rounding.

## References

- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791),
  *Proc. IEEE* 86, 2278 (1998): convolutional networks and the MNIST data.
- D. E. Rumelhart, G. E. Hinton and R. J. Williams, [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0),
  *Nature* 323, 533 (1986).
- K. He, X. Zhang, S. Ren and J. Sun, [Delving deep into rectifiers](https://arxiv.org/abs/1502.01852),
  ICCV (2015): the He initialization.
- B. T. Polyak, *Some methods of speeding up the convergence of iteration methods*, USSR Comput. Math.
  Math. Phys. 4, 1 (1964): momentum (the heavy-ball method).
- I. Sutskever, J. Martens, G. Dahl and G. Hinton, [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html),
  ICML (2013).
- L. Bottou, F. E. Curtis and J. Nocedal, [Optimization methods for large-scale machine learning](https://doi.org/10.1137/16M1080173),
  *SIAM Review* 60, 223 (2018): SGD, step sizes and their convergence.

## Read the code

- [`js/examples/mnist-model.js`](js/examples/mnist-model.js) is shared by both MNIST examples: the WGSL
  kernels (`buildComputeShader`), the `MNISTGPU` driver, initialization, shuffle and augmentation, and
  the weights file format. `measureTraining` at its end is the compute-only path of the
  [Measure](#mnist-train/measure) tab.
- [`js/examples/mnist-train.js`](js/examples/mnist-train.js) plans the mini-batches, reads the
  statistics and draws the curves and filters.
- [`tools/train-mnist.mjs`](tools/train-mnist.mjs) is the CPU trainer with the same math; it produced
  `data/mnist/weights.bin`. [`js/refs/mnist.js`](js/refs/mnist.js) is the float64 forward pass the
  experiments check the GPU against.
- [`js/teach/experiments/mnist-train.js`](js/teach/experiments/mnist-train.js) has the experiments;
  [`test/gpu/mnist-model.gpu.test.mjs`](test/gpu/mnist-model.gpu.test.mjs) and
  [`test/gpu/mnist-train-measure.gpu.test.mjs`](test/gpu/mnist-train-measure.gpu.test.mjs) compare the
  shipped kernels with the CPU trainer on a software GPU in CI.
