import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ocrd_pc_segmentation.ocrd_segmentation import PixelClassifierSegmentation


@click.command()
@ocrd_cli_options
def ocrd_pc_segmentation(*args, **kwargs):
    return ocrd_cli_wrap_processor(PixelClassifierSegmentation, *args, **kwargs)
