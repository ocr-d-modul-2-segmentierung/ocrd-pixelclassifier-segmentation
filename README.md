# page-segmentation module for OCR-d

## Introduction

This module implements a page segmentation algorithm based on a Fully
Convolutional Network (FCN). The FCN creates a classification for each pixel in
a binary image. This result is then segmented per class using XY cuts.

## Requirements

- For GPU-Support: [CUDA](https://developer.nvidia.com/cuda-downloads) and
  [CUDNN](https://developer.nvidia.com/cudnn)
- other requirements are installed via Makefile / pip, see `requirements.txt`
  in repository root.

## Installation

If you want to use GPU support, set the environment variable `TENSORFLOW_GPU`
to a nonempty value, otherwise leave it unset. Then:

```bash
make deps
```

to install dependencies and

```sh
make install
```

to install the package.

Both are python packages installed via pip, so you may want to activate
a virtalenv before installing.

## Usage

`ocrd-pc-segmentation` follows the [ocrd CLI](https://ocr-d.github.io/cli).

It expects a binary page image and produces region entries in the PageXML file.

## Configuration

The following parameters are recognized in the JSON parameter file:

- `overwrite_regions`: remove previously existing text regions
- `xheight`: height of character "x" in pixels used during training.
- `model`: pixel-classifier model path. The special values `__DEFAULT__` and `__LEGACY__` load the bundled default model or previous default model respectively.
- `gpu_allow_growth`: required for GPU use with some graphic cards
  (set to true, if you get CUDNN_INTERNAL_ERROR)
- `resize_height`: scale down pixelclassifier output to this height before postprocessing. Independent of training / used model.
  (performance / quality tradeoff, defaults to 300)

## Testing

There is a simple CLI test, that will run the tool on a single image from the assets repository.

`make test-cli`

## Training

To train models for the pixel classifier, see [its README](https://github.com/ocr-d-modul-2-segmentierung/page-segmentation/blob/master/README.md)
