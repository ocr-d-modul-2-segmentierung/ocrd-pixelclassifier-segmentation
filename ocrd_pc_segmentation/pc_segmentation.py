#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import re
import os
import sys

import datetime
import xml.etree.ElementTree as ET

#np.set_printoptions(threshold=np.nan)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--inverted", type=str, help="Path to the inverted image which should be analyzed", required=True)
	parser.add_argument("--binary", type=str, help="Path to the binary image", required=True)
	parser.add_argument("--output_dir", type=str, help="Directory to write segmentation files", required=True)
	parser.add_argument("--char_height", type=int, help="Average height of character m or n, ...", required=True)
	args = parser.parse_args()

	# Color mapping to match results of the Pixel Classifier
	color_mapping = {
		"image"       : [0, 255, 0],
		"text"        : [0, 0, 255],
	}

	# Determines the height in px to which the image should be resized before analyzing
	# Speedup vs Accuracy ?
	resize_height = 300


	if not os.path.isfile(args.inverted):
		print("Error: Inverted image file not found: " + args.inverted, file=sys.stderr)
		raise SystemExit()
	if not os.path.isfile(args.binary):
		print("Error: Binary image file not found", file=sys.stderr)
		raise SystemExit()

	# Get image name
	image_basename, image_ext = get_image_basename(args.binary)
	if image_basename == "" or image_ext == "":
		print("Error: New image name could not be determined", file=sys.stderr)
		raise SystemExit()

	image_name = image_basename + image_ext

	# Load and get dimensions of the binary image
	image = cv2.imread(args.binary)
	orig_height = image.shape[0]
	orig_width  = image.shape[1]

	# Load and get dimensions of the inverted image
	image = cv2.imread(args.inverted)
	height = image.shape[0]
	width  = image.shape[1]
	# Scale image to specific height for more generic usage of threshold values
	scale_percent = resize_height / height
	height = int(image.shape[0] * scale_percent)
	width  = int(image.shape[1] * scale_percent)
	image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

	# Get resizing factor of the scaled image compared to the original one (binary)
	absolute_resize_factor = height / orig_height

	# Determines how many pixels in one line/column need to exist to indicate a match
	px_threshold_line   = int(args.char_height * absolute_resize_factor)
	px_threshold_column = int(args.char_height * absolute_resize_factor)
	# Determines the size of a gap in pixels to split the found matches into segments
	split_size_horizontal = int(args.char_height * 2 * absolute_resize_factor)
	split_size_vertical   = int(args.char_height * absolute_resize_factor)


	# Calculate x-y-cut and get its segments
	segments_text  = get_xy_cut(image, height, width, px_threshold_line, px_threshold_column, split_size_horizontal, split_size_vertical, color_mapping["text"])
	segments_image = get_xy_cut(image, height, width, px_threshold_line, px_threshold_column, split_size_horizontal, split_size_vertical, color_mapping["image"])

	# Mark the found segments in the image
	for segment in segments_text:
		cv2.rectangle(image, (segment["x_start"], segment["y_start"]), (segment["x_end"], segment["y_end"]), color_mapping["text"], 1)
	for segment in segments_image:
		cv2.rectangle(image, (segment["x_start"], segment["y_start"]), (segment["x_end"], segment["y_end"]), color_mapping["image"], 1)
	cv2.imwrite(os.path.join(args.output_dir, image_basename + "_cut" + image_ext), image)

	# Write PageXML
	create_page_xml(args.binary, resize_height, segments_text, segments_image, os.path.join(args.output_dir, "clip_" + image_name + ".xml"))

	# Create an image for each text segment for OCR
	create_segment_images(args.binary, image_basename, image_ext, resize_height, segments_text, args.output_dir)



def get_image_basename(image_name):
	image_name = os.path.basename(image_name)
	img_name_match = re.search('(.*)(\..*)(\..*)', image_name, re.IGNORECASE)
	if img_name_match:
		return  img_name_match.group(1), img_name_match.group(2) + img_name_match.group(3)
	else:
		img_name_match = re.search('(.*)(\..*)', image_name, re.IGNORECASE)
		if img_name_match:
			return img_name_match.group(1), img_name_match.group(2)
	return "", ""


def get_line_indication(image, height, segment, px_threshold, color_match):
	line_indication = np.zeros(height, np.uint8)

	for y in range(segment["y_start"], segment["y_end"]):
		px_amount = 0
		for x in range(segment["x_start"], segment["x_end"]):
			color = image[y, x]

			# Match on RGB channel only (colors seem to change on resize)
			if  (color_match[0] == 0 and color[0] != 0) or (color_match[1] == 0 and color[1] != 0) or (color_match[2] == 0 and color[2] != 0) or \
				(color_match[0] != 0 and color[0] == 0) or (color_match[1] != 0 and color[1] == 0) or (color_match[2] != 0 and color[2] == 0):
				continue

			px_amount += 1
			if px_amount >= px_threshold:
				line_indication[y] = 1
				break

	return line_indication

def get_column_indication(image, width, segment, px_threshold, color_match):
	column_indication = np.zeros(width, np.uint8)

	for x in range(segment["x_start"], segment["x_end"]):
		px_amount  = 0
		for y in range(segment["y_start"], segment["y_end"]):
			color = image[y, x]

			# Match on RGB channel only (colors seem to change on resize)
			if  (color_match[0] == 0 and color[0] != 0) or (color_match[1] == 0 and color[1] != 0) or (color_match[2] == 0 and color[2] != 0) or \
				(color_match[0] != 0 and color[0] == 0) or (color_match[1] != 0 and color[1] == 0) or (color_match[2] != 0 and color[2] == 0):
				continue

			px_amount += 1
			if px_amount >= px_threshold:
				column_indication[x] = 1
				break

	return column_indication


def get_gaps(indication):
	gaps = []
	gap_index = 0

	for i in range(0, len(indication)):
		match = indication[i]
		if match == 0:
			if gap_index < len(gaps):
				gaps[gap_index]["length"] += 1
			else:
				gaps.append({})
				gaps[gap_index]["start"] = i
				gaps[gap_index]["length"] = 1
		else:
			if gap_index < len(gaps):
				gap_index += 1

	return gaps


def get_sliced_coords(gaps, default_start, default_end):
	sliced_start = default_start
	sliced_end   = default_end

	for gap in gaps:
		if gap["start"] == 0:
			# Start is a gap
			sliced_start = gap["length"]
		elif (gap["start"] + gap["length"]) >= default_end:
			# End is a gap
			sliced_end = gap["start"]
	return sliced_start, sliced_end

def get_segments(gaps, start_px, end_px, px_threshold, split_size):
	segments = []
	segment_start = start_px

	for gap in gaps:
		if gap["start"] > start_px and (gap["start"] + gap["length"]) < end_px:
			if gap["length"] > split_size:
				if (segment_start + gap["start"]) > px_threshold:
					segments.append({"start" : segment_start, "end": gap["start"] })
					segment_start = gap["start"] + gap["length"]
	# Always append last segment that remains
	if segment_start < end_px:
		if (segment_start + end_px) > px_threshold:
			segments.append({"start" : segment_start, "end" : end_px})

	return segments


def get_y_cut(image, height, segment, px_threshold, split_size, color_match):
	# Get initial line indication of the image
	# Each line is scanned and marked if it matches the given color based on a given threshold of pixels
	# If the number of color pixels in the line is greater than the threshold mark it as a match
	line_indication = get_line_indication(image, height, segment, px_threshold, color_match)
	# Identify line gaps in the image (lines that are not marked as the given color)
	line_gaps = get_gaps(line_indication)
	# Find important part of the image without unnecessary left+right part of the page
	# This corresponds with a starting and ending gap
	sliced_start, sliced_end = get_sliced_coords(line_gaps, segment["y_start"], segment["y_end"])
	# Extract final segments that remain (y-cut)
	return get_segments(line_gaps, sliced_start, sliced_end, px_threshold, split_size)

def get_x_cut(image, width, segment, px_threshold, split_size, color_match):
	# Get initial column and column indication of the image
	# Each column is scanned and marked if it matches the given color based on a given threshold of pixels
	# If the number of color pixels in the column is greater than the threshold mark it as a match
	column_indication = get_column_indication(image, width, segment, px_threshold, color_match)
	# Identify column gaps in the image (column that are not marked as the given color)
	column_gaps = get_gaps(column_indication)
	# Find important part of the image without unnecessary top+bottom part of the page
	# This corresponds with a starting and ending gap
	sliced_start, sliced_end = get_sliced_coords(column_gaps, segment["x_start"], segment["x_end"])
	# Extract final segments that remain (x-cut)
	return get_segments(column_gaps, sliced_start, sliced_end, px_threshold, split_size)

def get_xy_cut(image, height, width, px_threshold_line, px_threshold_column, split_size_horizontal, split_size_vertical, color_match):
	segments_final = []
	segments_new   = [{"done" : False, "y_start" : 0, "y_end" : height, "x_start" : 0, "x_end" : width}]

	iter = 0
	while len(segments_final) != len(segments_new):
		iter += 1
		segments_final = segments_new
		segments_new = []

		for segment in segments_final:
			# Segment is in final cut and cannot be splitted further
			if segment["done"] == True:
				segments_new.append(segment)
				continue

			segments_cut = []
			y_segments = get_y_cut(image, height, segment, px_threshold_line, split_size_horizontal, color_match)
			for y_segment in y_segments:
				x_segments = get_x_cut(image, width, segment, px_threshold_column, split_size_vertical, color_match)
				for x_segment in x_segments:
					segments_cut.append({"done" : False, "y_start" : y_segment["start"], "y_end" : y_segment["end"], "x_start" : x_segment["start"], "x_end" : x_segment["end"]})

			if len(segments_cut) == 1:
				# Segment stayed the same (maybe sliced), make it final
				segments_cut[0]["done"] = True
				segments_new.append(segments_cut[0])
			else:
				# Segment is splitted again in smaller pieces
				# Loop will be rerun to verify new segments
				segments_new = segments_new + segments_cut

	return segments_new


def get_xml_point_string(segment, coord_factor):
	return " ".join(["{:.0f},{:.0f}"] * 4).format(*[ x * coord_factor for x in [
			segment["x_start"], segment["y_start"],
			segment["x_end"],   segment["y_start"],
			segment["x_end"],   segment["y_end"],
			segment["x_start"], segment["y_end"],
		]])

def create_page_xml(image_path, resize_height, segments_text, segments_image, outfile):
	image  = cv2.imread(image_path)
	height = image.shape[0]
	width  = image.shape[1]
	coord_factor = height / resize_height

	pcgts = ET.Element("PcGts", {
		"xmlns" : "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15",
		"xmlns:xsi" : "http://www.w3.org/2001/XMLSchema-instance",
		"xsi:schemaLocation" : "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd"
	})

	metadata = ET.SubElement(pcgts, "Metadata")
	creator  = ET.SubElement(metadata, "Creator")
	creator.text = "User123"
	created  = ET.SubElement(metadata, "Created")
	created.text = datetime.datetime.now().isoformat()
	last_change = ET.SubElement(metadata, "LastChange")
	last_change.text = datetime.datetime.now().isoformat()

	page = ET.SubElement(pcgts, "Page", {
		"imageFilename" : image_path,
		"imageWidth" : str(width),
		"imageHeight" : str(height),
	})

	count = 0
	for segment in segments_text:
		region = ET.SubElement(page, "TextRegion", { "id" : "r" + str(count), "type" : "paragraph" })
		coords = ET.SubElement(region, "Coords", {"points" : get_xml_point_string(segment, coord_factor)})
		count += 1
	for segment in segments_image:
		region = ET.SubElement(page, "GraphicRegion", { "id" : "r" + str(count) })
		coords = ET.SubElement(region, "Coords", { "points" : get_xml_point_string(segment, coord_factor) })
		count += 1

	tree = ET.ElementTree(pcgts)
	tree.write(outfile, xml_declaration=True, encoding='utf-8', method="xml")


def create_segment_images(image_file, image_basename, image_ext, resize_height, segments, output_dir):
	image  = cv2.imread(image_file)
	height = image.shape[0]
	width  = image.shape[1]
	coord_factor = height / resize_height

	count = 0
	for segment in segments:
		segment_image = image[
			int(segment["y_start"] * coord_factor):int(segment["y_end"] * coord_factor),
			int(segment["x_start"] * coord_factor):int(segment["x_end"] * coord_factor)
		]
		cv2.imwrite(os.path.join(output_dir, image_basename + "__" + '{:03}'.format(count) + "__paragraph" + image_ext), segment_image)
		count += 1



if __name__ == "__main__":
	main()
