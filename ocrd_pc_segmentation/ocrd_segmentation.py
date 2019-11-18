from __future__ import absolute_import

import json
import os.path

import numpy as np
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    TextRegionType,
    ImageRegionType,
    NoiseRegionType,
    CoordsType,
    to_xml,
)
from ocrd_utils import (
    getLogger,
    concat_padded,
    MIMETYPE_PAGE,
)
from pkg_resources import resource_string

OCRD_TOOL = json.loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))

TOOL = 'ocrd-pixelclassifier-segmentation'
LOG = getLogger('processor.PixelClassifierSegmentation')
FALLBACK_IMAGE_GRP = 'OCR-D-SEG-BLOCK'


class PixelClassifierSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(PixelClassifierSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Performs segmentation on the input binary image

        Produces a PageXML file as output.
        """
        overwrite_regions = self.parameter['overwrite_regions']
        char_height = self.parameter['char_height']
        gpu_allow_growth = self.parameter['gpu_allow_growth']
        resize_height = self.parameter['resize_height']

        model = self.parameter['model']
        if model == '__DEFAULT__':
            from ocrd_pc_segmentation import DEFAULT_SEGMENTATION_MODEL_PATH
            model = DEFAULT_SEGMENTATION_MODEL_PATH


        try:
            page_grp, image_grp = self.output_file_grp.split(',')
        except ValueError:
            page_grp = self.output_file_grp
            image_grp = FALLBACK_IMAGE_GRP
            LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, image_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            metadata = pcgts.get_Metadata()  # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 # FIXME: externalRef is invalid by pagecontent.xsd, but ocrd does not reflect this
                                 # what we want here is `externalModel="ocrd-tool" externalId="parameters"`
                                 Labels=[LabelsType(  # externalRef="parameters",
                                     Label=[LabelType(type_=name,
                                                      value=self.parameter[name])
                                            for name in self.parameter.keys()])]))
            page = pcgts.get_Page()
            if page.get_TextRegion():
                if overwrite_regions:
                    LOG.info('removing existing TextRegions')
                    page.set_TextRegion([])
                else:
                    LOG.warning('keeping existing TextRegions')

            page.set_AdvertRegion([])
            page.set_ChartRegion([])
            page.set_ChemRegion([])
            page.set_GraphicRegion([])
            page.set_ImageRegion([])
            page.set_LineDrawingRegion([])
            page.set_MathsRegion([])
            page.set_MusicRegion([])
            page.set_NoiseRegion([])
            page.set_SeparatorRegion([])
            page.set_TableRegion([])
            page.set_UnknownRegion([])

            page_image, page_xywh, _ = self.workspace.image_from_page(page, page_id)

            self._process_page(page, np.asarray(page_image), page_xywh, char_height, model,
                               gpu_allow_growth, resize_height)

            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            file_id = input_file.ID.replace(self.input_file_grp, page_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(page_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=page_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(page_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))

    @staticmethod
    def _process_page(page, page_image, page_xywh, char_height, model, gpu_allow_growth,
                      resize_height):

        # TODO: does this still need to be cropped or do we not need page_xywh?
        #       Same for points below
        #       page_image[page_xywh["x"]:page_xywh["w"], page_xywh["y"]:page_xywh["h"]]

        from ocr4all_pixel_classifier.lib.pc_segmentation import find_segments
        from ocr4all_pixel_classifier.scripts.find_segments import predict_masks, \
            DEFAULT_IMAGE_MAP, DEFAULT_REVERSE_IMAGE_MAP
        from ocr4all_pixel_classifier.lib.pc_segmentation import Segment

        image_map = DEFAULT_IMAGE_MAP
        rev_image_map = DEFAULT_REVERSE_IMAGE_MAP

        masks = predict_masks(None,
                              page_image,
                              image_map,
                              char_height,
                              model,
                              post_processors=None,
                              gpu_allow_growth=gpu_allow_growth,
                              )

        orig_height, orig_width = page_image.shape[0:2]
        mask_image = masks.inverted_overlay

        segments_text, segments_image = find_segments(orig_height, mask_image, char_height,
                                                      resize_height, rev_image_map)

        def add_region(region: Segment, index: int, type: str):
            indexed_id = "region%04d" % index
            points = str([
                (region.x_start, region.y_start),
                (region.x_start, region.y_end),
                (region.x_end, region.y_start),
                (region.x_end, region.y_end),
            ])
            coords = CoordsType(points=points)
            if type == "text":
                page.add_TextRegion(TextRegionType(id=indexed_id, Coords=coords))
            elif type == "image":
                page.add_ImageRegion(ImageRegionType(id=indexed_id, Coords=coords))
            else:
                page.add_NoiseRegion(NoiseRegionType(id=indexed_id, Coords=coords))

        count = 0
        for r in segments_text:
            add_region(r, count, "text")
            count += 1
        for r in segments_image:
            add_region(r, count, "image")
            count += 1
