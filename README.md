# page-segmentation module for OCRd

## Requirements

- For GPU-Support: [CUDA](https://developer.nvidia.com/cuda-downloads) and [CUDNN](https://developer.nvidia.com/cudnn)
- other requirements are installed via Makefile / pip, see `requirements.txt`
  in repository root and in `page-segmentation` submodule.
- *TODO: publish all dependencies for pip*

## Setup

```bash
make dep
# or
make dep-gpu
```

then

```sh
make install
```

## Running

`ocrd-pc-segmentation` follows the [ocrd CLI](https://ocr-d.github.io/cli).

## Configuration

The following parameters are recognized in the JSON parameter file:

- `overwrite_regions`: remove previously existing text regions
- `char_height`: height of character "n" in pixels
- `model`: pixel-classifier model path
- `gpu_allow_growth`: required for GPU use with some graphic cards
  (set to true, if you get CUDNN_INTERNAL_ERROR)
- `resize_height`: scale down pixelclassifier output to this height
  (performance / quality tradeoff, defaults to 300)
