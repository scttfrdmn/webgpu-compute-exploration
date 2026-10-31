# MNIST subset and cached weights

Data for examples 21 (MNIST: Train) and 22 (MNIST: Recognize). License: CC BY-SA 3.0, see
[LICENSE.md](LICENSE.md).

## Files

| File | Contents | Raw size |
|---|---|---|
| `train-images.u8.gz` | 10,000 images × 784 uint8 pixels (28×28, row-major, 0 = background, 255 = ink) | 7,840,000 bytes |
| `train-labels.u8.gz` | 10,000 uint8 labels (0-9) | 10,000 bytes |
| `test-images.u8.gz` | 2,000 images × 784 uint8 pixels | 1,568,000 bytes |
| `test-labels.u8.gz` | 2,000 uint8 labels | 2,000 bytes |
| `weights.bin` | trained network weights (format below) | 24,108 bytes |

The data files are gzip-compressed raw bytes with no IDX header: image `i` is bytes
`784·i … 784·i + 783`. They are the first 10,000 images of the original MNIST training set and the
first 2,000 of its test set, in the original order. The browser fetches them and decompresses with
`DecompressionStream('gzip')` when the bytes start with the gzip magic `1f 8b` (a server that
decodes gzip itself delivers them uncompressed, which also works).

## Network

Defined in `js/examples/mnist-model.js`, used by both examples and by the trainer: input 28×28
(pixel / 255) → conv 5×5, 8 filters → ReLU → 2×2 max-pool → conv 5×5, 16 filters → ReLU →
2×2 max-pool → dense 256 → 10 → softmax. 5,994 parameters.

## Weights format (`MNW1`, little-endian)

| Offset | Type | Contents |
|---|---|---|
| 0 | char[4] | magic `MNW1` |
| 4 | u32 | version = 1 |
| 8 | u32 | tensor count = 6 |
| 12 | 6 × { u32 rank; u32 dims[4] } | shapes, unused dims 0 (20 bytes per tensor) |
| 132 | f32[5994] | the tensors, in this order |

| Tensor | Shape | Values | Meaning |
|---|---|---|---|
| W1 | [8, 1, 5, 5] | 200 | conv1 kernels [filter][channel][ky][kx] |
| b1 | [8] | 8 | conv1 biases |
| W2 | [16, 8, 5, 5] | 3,200 | conv2 kernels [filter][input channel][ky][kx] |
| b2 | [16] | 16 | conv2 biases |
| W3 | [10, 256] | 2,560 | dense weights [class][j], j = channel·16 + y·4 + x of the 16×4×4 pooled maps |
| b3 | [10] | 10 | dense biases |

Example 21's **Export weights** button writes files in the same format, which can replace
`weights.bin`.

## Regenerating `weights.bin`

```sh
node tools/train-mnist.mjs                 # writes data/mnist/weights.bin
node tools/train-mnist.mjs --epochs 5 --out /path/to/other.bin
```

Options: `--seed 1 --batch 64 --lr 0.04 --lrDecay 0.85 --momentum 0.9 --weightDecay 0.0005
--maxShift 2 --epochs 15 --out data/mnist/weights.bin` (the defaults). The trainer runs on the CPU
with the same algorithm as the GPU kernels of example 21: seeded He initialization, a seeded
shuffle every epoch, random translations of up to ±2 pixels, softmax cross-entropy, and SGD with
momentum, L2 weight decay and a learning rate of 0.04 · 0.85^epoch. It is deterministic for a
given seed. The committed file reached 98.00% (1,960 / 2,000) test accuracy after 15 epochs, in
about 2.5 minutes on one Apple silicon CPU core.
