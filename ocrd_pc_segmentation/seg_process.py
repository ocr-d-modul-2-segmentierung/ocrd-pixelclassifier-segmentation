#!/usr/bin/env python3

import argparse
import os
import sys
import re
from glob import glob
from shutil import copyfile
import xml.etree.ElementTree as ET

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", type=str, help="Path to the binary image which should be analyzed", required=True)
	parser.add_argument("--pc_model", type=str, help="Path to the Pixel Classifier model which should be used for segmentation", required=True)
	args = parser.parse_args()

	if not os.path.isfile(args.image):
		print("Error: Image file not found", file=sys.stderr)
		raise SystemExit()

	if not os.path.isfile(args.pc_model + ".index"):
		print("Error: Pixel Classifier model not found", file=sys.stderr)
		raise SystemExit()

	image_basename, image_ext = get_image_basename(args.image)
	if image_basename == "" or image_ext == "":
		print("Error: Image name could not be parsed", file=sys.stderr)
		raise SystemExit()

	image_name  = image_basename + image_ext
	image_dir   = os.path.dirname(args.image)
	process_dir = os.path.join(image_dir, image_basename)

	if not os.path.exists(process_dir):
		os.mkdir(process_dir)

	copyfile(args.image, os.path.join(process_dir, image_name))

	# Necessary scripts/projects:
	# ./
	# ./ocr_process.py
	# ./pc_segmentation.py
	# ./page-segmentation/ (Pixel Classifier Gitlab Repository)
	# ./mptv/ (OCRopus Git Repository)

	# FS structure for each image (example: buchner_theoria02_1683_0146.bin.png):
	# buchner_theoria02_1683_0146/
	# buchner_theoria02_1683_0146/buchner_theoria02_1683_0146.bin.png
	# buchner_theoria02_1683_0146/color			(Pixel Classifier mask image)
	# buchner_theoria02_1683_0146/inverted		(Pixel Classifier inverted image)
	# buchner_theoria02_1683_0146/overlay		(Pixel Classifier overlay image)
	# buchner_theoria02_1683_0146/segmentation	(Segmentation files)

	# Fetch character height
	exit = os.system("ocrd-compute-normalizations --input_dir " + process_dir
		+ " --output_dir " + process_dir)
	if exit != 0:
		print("Error: ocrd-compute-normalizations exited with an error", file=sys.stderr)
		raise SystemExit()

	char_height = get_char_height(process_dir)
	if char_height == -1:
		print("Error: char_height could not be fetched", file=sys.stderr)
		raise SystemExit()

	# Execute Pixel Classifier and get text/image prediction
	exit = os.system("ocrd-pixel-classifier --image " + args.image
		+ " --binary " + args.image + " --load " + args.pc_model + " --char_height " + str(char_height) + " --output " + process_dir)
	if exit != 0:
		print("Error: ocrd-pixel-classifier exited with an error", file=sys.stderr)
		raise SystemExit()

	# Create segmentation directory
	segmentation_dir = os.path.join(process_dir, "segmentation")
	if not os.path.exists(segmentation_dir):
		os.mkdir(segmentation_dir)

	# Determine text/image regions based on Pixel Classifier prediction
	exit = os.system("ocrd-pc-seg-single --inverted " + os.path.join(process_dir, "inverted", image_name) +
		" --char_height " + str(char_height) + " --binary " + args.image + " --output_dir " + segmentation_dir)
	if exit != 0:
		print("Error: ocrd-pc-seg-single exited with an error", file=sys.stderr)
		raise SystemExit()

	# Create line images based on text regions
	os.chdir(segmentation_dir)
	for segment_file in glob(image_basename + "__*__paragraph" + image_ext):
		os.system("ocropus-gpageseg-with-coords -n " + segment_file)

	# Get line coordinates for all regions
	region_line_coords = get_region_line_coordinates(glob(os.path.join(".", "*", "")))

	# Write line coordinates back to PageXML
	add_region_lines_to_pagexml(region_line_coords, 'clip_' + image_name + '.xml')


def get_image_basename(image):
	image = os.path.basename(image)
	img_name_match = re.search('(.*)(\..*)(\..*)', image, re.IGNORECASE)
	if img_name_match:
		return  img_name_match.group(1), img_name_match.group(2) + img_name_match.group(3)
	else:
		img_name_match = re.search('(.*)(\..*)', image, re.IGNORECASE)
		if img_name_match:
			return img_name_match.group(1), img_name_match.group(2)
	return "", ""

def get_char_height(norm_file_directory):
	for file in os.listdir(norm_file_directory):
		if file.endswith(".norm"):
			with open(os.path.join(norm_file_directory, file), 'r') as norm_file:
				file_content = re.search('"char_height": ([0-9]+)', norm_file.read(), re.IGNORECASE)
				if file_content:
					return int(file_content.group(1))
	return -1

def calculate_line_coordinates(region_line_coords, region_coords):
	# 0 : x_start, 1 : y_start, 2: x_end, 5: y_end
	region_coords = re.split(' |,', region_coords)
	# 0 : y_start, 1 : x_start, 2: y_end, 3: x_end
	region_line_coords = region_line_coords.split(',')

        # Adapt line coordinates from their region to the absolute image
	image_line_x_start = int(region_line_coords[1]) + int(region_coords[0])
	image_line_y_start = int(region_line_coords[0]) + int(region_coords[1])
	image_line_x_end   = int(region_line_coords[3]) + int(region_coords[0])
	image_line_y_end   = int(region_line_coords[2]) + int(region_coords[1])

	return (str(image_line_x_start) + ',' + str(image_line_y_start) + " " +
			str(image_line_x_end)   + ',' + str(image_line_y_start) + " " +
			str(image_line_x_end)   + ',' + str(image_line_y_end) + " " +
			str(image_line_x_start) + ',' + str(image_line_y_end))

def get_region_line_coordinates(region_dirs):
	region_line_coords = {}
	for region_dir in region_dirs:
		# Find region identifier
		region_num_match = re.search('__([0-9]{3})__', region_dir)
		if region_num_match:
			region_num = region_num_match.group(1)
			region_line_coords[region_num] = {}

			# Find all line coordinate files for the current region
			for line_file in os.listdir(region_dir):
				if line_file.endswith(".coords"):
					# Find line identifier
					line_num_match = re.search('__([0-9]{3})\.', line_file)
					if line_num_match:
						line_num = line_num_match.group(1)

						# Extract and store coordinates
						with open(os.path.join(region_dir, line_file), 'r') as coords_file:
							region_line_coords[region_num][line_num] = coords_file.read()
	return region_line_coords

def add_region_lines_to_pagexml(region_line_coords, pagexml):
	schema = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15}'
	tree = ET.parse(pagexml)
	ET.register_namespace('', schema[1:-1])
	root = tree.getroot()

	for region_num in region_line_coords:
		region_id = 'r' + str(int(region_num))
		region = root.find(".//" + schema + "TextRegion[@id='" + region_id + "']")
		region_coords = root.find(".//" + schema + "TextRegion[@id='" + region_id + "']/" + schema + "Coords")

		# Add TextLine with Coords elements inside TextRegion element
		for line_num in region_line_coords[region_num]:
			line_id = region_id + '_' + str(line_num)
			coords = calculate_line_coordinates(region_line_coords[region_num][line_num], region_coords.get('points'))
			textline = ET.SubElement(region, "TextLine", { "id" : line_id })
			line_coords = ET.SubElement(textline, "Coords", { "points" : coords })

	tree.write(pagexml)


if __name__ == "__main__":
	main()
