# page-segmentation module for OCRd

## Requirements

- For GPU-Support: [CUDA](https://developer.nvidia.com/cuda-downloads) and [CUDNN](https://developer.nvidia.com/cudnn)
- other requirements are installed via Makefile / pip, see `requirements.txt`
  in repository root and in `page-segmentation` submodule.

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

The main script is `ocrd-pc-seg-process`. It takes two parameters: a `--model` for the
pixel classifier and an `--image`, which must be a binarized image. For example
with the included model:

```
ocrd-pc-seg-process --pc_model model/narren_dta02_eval_normaldh_maskfix_03 \
    --image abel_leibmedicus_1699_0007.bin.png
```

This creates a folder with the basename of the image (e.g. `abel_leibmedicus_1699_0007/`)
with line images in `segmentation/${basename}_${paragraph_nr}_paragraph` and
PageXML in `segmentation/clip_${filename}.xml` (where `$filename` is the input
file name,`$basename` is `$filename` without extensions, `$paragraph_nr` is
successive ocropus pagagraph number).
