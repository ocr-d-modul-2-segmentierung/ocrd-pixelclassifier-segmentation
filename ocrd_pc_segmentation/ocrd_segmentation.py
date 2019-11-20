from __future__ import absolute_import

import json
import os.path

import numpy as np
from ocr4all_pixel_classifier.lib.pc_segmentation import Segment
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


def polygon_from_segment(segment: Segment):
    from ocrd_utils import polygon_from_bbox
    return polygon_from_bbox(segment.x_start, segment.y_start, segment.x_end, segment.y_end)


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
        xheight = self.parameter['xheight']
        gpu_allow_growth = self.parameter['gpu_allow_growth']
        resize_height = self.parameter['resize_height']

        model = self.parameter['model']
        if model == '__DEFAULT__':
            from ocrd_pc_segmentation import DEFAULT_SEGMENTATION_MODEL_PATH
            model = DEFAULT_SEGMENTATION_MODEL_PATH

        page_grp = self.output_file_grp

        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            metadata = pcgts.get_Metadata()  # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
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

            page_image, page_coords, _ = self.workspace.image_from_page(page, page_id)

            self._process_page(page, np.asarray(page_image), page_coords, xheight, model,
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
    def _process_page(page, page_image, page_coords, xheight, model, gpu_allow_growth,
                      resize_height):

        from ocr4all_pixel_classifier.lib.pc_segmentation import find_segments
        from ocr4all_pixel_classifier.scripts.find_segments import predict_masks, \
            DEFAULT_IMAGE_MAP, DEFAULT_REVERSE_IMAGE_MAP

        image_map = DEFAULT_IMAGE_MAP
        rev_image_map = DEFAULT_REVERSE_IMAGE_MAP

        masks = predict_masks(None,
                              page_image,
                              image_map,
                              xheight,
                              model,
                              post_processors=None,
                              gpu_allow_growth=gpu_allow_growth,
                              )

        orig_height, orig_width = page_image.shape[0:2]
        mask_image = masks.inverted_overlay

        segments_text, segments_image = find_segments(orig_height, mask_image, xheight,
                                                      resize_height, rev_image_map)

        def add_region(region: Segment, index: int, region_type: str):
            from ocrd_utils import coordinates_for_segment, points_from_polygon
            polygon = polygon_from_segment(region)
            polygon = coordinates_for_segment(polygon, page_image, page_coords)
            points = points_from_polygon(polygon)

            indexed_id = "region%04d" % index
            coords = CoordsType(points=points)
            if region_type == "text":
                page.add_TextRegion(TextRegionType(id=indexed_id, Coords=coords))
            elif region_type == "image":
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
