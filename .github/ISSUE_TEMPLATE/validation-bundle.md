---
name: Validation bundle
about: Contribute a physical-GPU validation run from validate.html (the validation matrix)
title: "Validation bundle: "
---

Thanks for running the validation protocol. Each bundle adds one real device to the validation matrix
(see "Validation on physical GPUs" in the README). A maintainer checks the bundle and commits it under
`data/validation/<environment>/`; you do not need to open a pull request.

## How to run it

1. Open the **deployed** page, https://scttfrdmn.github.io/webgpu-compute-exploration/validate.html, in the
   browser you want to record. Do not use `validate.html?smoke=1` (a reduced plan for CI, which the matrix
   rejects) or a local copy without its commit.
2. Close other tabs that use the GPU, plug in a laptop, pick the **Matrix environment**, fill in the fields
   the page asks for, and press **Run validation** (about a minute; keep the tab in front).
3. Press **Download validation bundle (JSON)**. The file is named `<date>-<commit>.json`; a name ending in
   `-local.json` or `-smoke.json` means it was not run on the deployed site with the full plan.

## Environment

<!--
The fields of data/validation/matrix.json's "record" list. The bundle already contains them; repeat them here
so the issue can be read without opening the file, and say if you corrected a detected value.
-->

- Matrix environment (`environment.id`, e.g. apple-silicon-chrome, or "not listed"):
- Browser and version (`browser.version`; check it against the browser's About page):
- OS and version (`os.version`; Safari freezes the macOS version in the user agent, so enter it from About This Mac):
- Hardware (`hardware`: Mac model and chip from About This Mac, e.g. "MacBook Air 13-inch, M2"; other devices: model and GPU):
- timestamp-query on the adapter (`timestampQuery.adapter`, detected): yes / no
- timestamp-query on the device (`timestampQuery.device`, detected): yes / no
- Notes (power, browser flags such as Chrome's WebGPU developer features, anything unusual):

## The bundle

<!--
Drag the downloaded .json file into this box to attach it (bundles are about 100 KB, too large to paste).
If GitHub does not accept the .json file, compress it to a .zip and attach that. Do not edit the file.
-->

(attach the bundle here)

## Results on the page

<!-- Optional: anything the Results table showed that looked wrong — a FAIL, a stopped experiment, a device loss. -->

## Checklist

- [ ] Run on the deployed site (https://scttfrdmn.github.io/webgpu-compute-exploration/validate.html), not `?smoke=1` and not a local copy.
- [ ] The full plan ran to the end (no Cancel, no device loss), with nothing else using the GPU.
- [ ] The file name ends in `-<7-character commit>.json` and the file is unedited.
- [ ] I am fine with the bundle (browser, OS, GPU adapter and user agent; no personal data) being committed to the public repository.
