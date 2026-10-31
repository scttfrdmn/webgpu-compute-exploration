# License of the MNIST data and derived weights

The files in this directory are a subset of the MNIST database of handwritten digits by
Yann LeCun, Corinna Cortes and Christopher J.C. Burges, made available under the
[Creative Commons Attribution-Share Alike 3.0 license (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).

- `train-images.u8.gz`, `train-labels.u8.gz`: the first 10,000 images and labels of the MNIST
  training set
- `test-images.u8.gz`, `test-labels.u8.gz`: the first 2,000 images and labels of the MNIST test set
- `weights.bin`: network weights trained on the training subset above, a work derived from the data

Changes: the subsets were cut from the original IDX files, the IDX headers removed, and the raw
bytes gzip-compressed; the pixel and label values are unchanged.

This license applies only to the data files and the derived weights in this directory (and to
weights you train from them, for example with example 21 or `tools/train-mnist.mjs`). The code of
this repository, including the code that loads and trains on this data, stays under the
[MIT license](../../LICENSE).
