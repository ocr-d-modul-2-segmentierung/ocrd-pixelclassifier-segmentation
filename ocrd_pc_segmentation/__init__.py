import os

DEFAULT_SEGMENTATION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'lgt-model', 'model.h5'
)

LEGACY_SEGMENTATION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'dta2-model', 'model.h5'
)
