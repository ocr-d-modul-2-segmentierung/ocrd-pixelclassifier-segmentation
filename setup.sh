#!/bin/sh

cd "$(dirname "$0")"

git submodule init
git submodule update

virtualenv3 page-segmentation_venv
. page-segmentation_venv/bin/activate
pip install ./page-segmentation
deactivate

virtualenv2 mptv_venv
. mptv_venv/bin/activate
pip install ./mptv
pip install -r mptv/requirements.txt
deactivate
