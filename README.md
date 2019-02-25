# page-segmentation module for OCRd

## Requirements

- `virtualenv2` and `virtualenv3`

## Setup

After checking out the repository, run `setup.sh`. This will create the required
virtualenvs and install dependencies.

## Running

The main script is `seg_process`. It takes two parameters: a `--model` for the
pixel classifier and an `--image`, which must be a binarized image. For example
with the included model:

```
./seg_process --pc_model model/narren_dta02_eval_normaldh_maskfix_03 \
    --image abel_leibmedicus_1699_0007.bin.png
```

This creates a folder with the basename of the image (e.g. `abel_leibmedicus_1699_0007/`)
with line images in `segmentation/${basename}_${paragraph_nr}_paragraph` and
PageXML in `segmentation/clip_${filename}.xml` (where `$filename` is the input
file name,`$basename` is `$filename` without extensions, `$paragraph_nr` is
successive ocropus pagagraph number).
